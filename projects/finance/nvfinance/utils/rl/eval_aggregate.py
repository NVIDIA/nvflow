#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Finance reporting over canonical native Gym rollout repeats.

Produces the evaluation leaderboard -- accuracy, balanced accuracy by
``question_type``, pass@k with cross-repeat variance, and the tool/turn/token
breakdown -- reusing Gym's pass@k/subset primitives. Finance-specific reporting (question-type
order, tool/turn/token rows) is the glue Gym doesn't provide.

CLI:
    python -m nvfinance.utils.rl.eval_aggregate \
        <rollouts.jsonl> <materialized-inputs.jsonl> <output_dir> \
        [--expected-repeats N] \
        [--max-turns 50] [--ctx-window 120000]

Writes ``summary.txt`` (human-readable) and ``metrics.json``.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

# Fixed display order, matching the historical report.
QTYPE_ORDER = [
    "Adjustments",
    "Beat or Miss",
    "Complex Retrieval",
    "Financial Modeling  Projections",
    "Market Analysis",
    "Numerical Reasoning",
    "Simple retrieval - Qualitative",
    "Simple retrieval - Quantitative",
    "Trends",
]

# Display order only. Tools not listed here are appended alphabetically, and a
# listed tool the benchmark does not offer is not reported at all.
TOOL_ORDER = [
    "web_search",
    "edgar_search",
    "sec_filing_search",
    "parse_html_page",
    "retrieve_information",
    "price_history",
    "calculator",
    "submit_final_result",
]
TOOL_LABELS = {"web_search": "Web search calls / question"}

# function_call names outside the tools offered to the model (reasoning leaked
# into the name, garbled/hallucinated names) are bucketed here so they can't
# distort the tool-call rows.
INVALID_TOOL = "invalid_tool_call"


# Gym pass@k / subset primitives. Prefer the real functions from
# nemo_gym.reward_profile; the analyze job runs in the gym image's base
# interpreter (no nemo_gym -- that lives in the CLI venv), so fall back to a
# verbatim, stdlib-only port. Keep _pass_majority_metrics / _subset_metrics in
# sync with nemo_gym/reward_profile.py.


def _load_gym_primitives():
    try:
        from nemo_gym.reward_profile import (
            compute_pass_majority_metrics,
            compute_subset_metrics,
        )

        return compute_pass_majority_metrics, compute_subset_metrics
    except ImportError:
        return _pass_majority_metrics, _subset_metrics


def _pass_majority_metrics(tasks, score_fn=None, answer_key=None):
    """Verbatim port of ``nemo_gym.reward_profile.compute_pass_majority_metrics``.

    Only the ``answer_key`` (majority@k) branch is dropped -- eval never sets it.
    """
    if not tasks:
        return {}, [], [], 0
    if score_fn is None:
        score_fn = lambda r: {"accuracy": r["reward"]}  # noqa: E731

    max_k = max(len(rollouts) for rollouts in tasks)
    metrics: dict = {}

    all_score_dicts: list[list[dict]] = []
    for rollouts in tasks:
        task_scores = []
        for r in rollouts:
            raw = score_fn(r)
            task_scores.append({k: (int(v) if isinstance(v, bool) else v) for k, v in raw.items()})
        all_score_dicts.append(task_scores)

    score_names = sorted({name for ts in all_score_dicts for s in ts for name in s})

    for k in range(1, max_k + 1):
        for name in score_names:
            pass_values = []
            for ts in all_score_dicts:
                vals = [s.get(name) for s in ts if name in s]
                if not vals or k > len(vals):
                    continue
                is_binary = all(v in (0, 1, 0.0, 1.0) for v in vals)
                if is_binary:
                    n_total = len(vals)
                    n_incorrect = sum(1 for v in vals if not v)
                    if n_incorrect < k:
                        pass_values.append(1.0)
                    else:
                        pass_values.append(1.0 - math.comb(n_incorrect, k) / math.comb(n_total, k))
                else:
                    pass_values.append(max(vals[:k]))
            if pass_values:
                metrics[f"pass@{k}/{name}"] = 100.0 * sum(pass_values) / len(pass_values)

            avg_values = []
            for ts in all_score_dicts:
                vals = [s.get(name) for s in ts[:k] if name in s]
                if vals:
                    avg_values.append(sum(vals) / len(vals))
            if avg_values:
                metrics[f"pass@1[avg-of-{k}]/{name}"] = 100.0 * sum(avg_values) / len(avg_values)

    per_sample: dict = {name: [] for name in score_names}
    for run_idx in range(max_k):
        for name in score_names:
            run_scores = [
                ts[run_idx].get(name)
                for ts in all_score_dicts
                if run_idx < len(ts) and name in ts[run_idx]
            ]
            if run_scores:
                per_sample[name].append(100.0 * sum(run_scores) / len(run_scores))
    per_sample = {k: v for k, v in per_sample.items() if v}
    metrics["per_sample_aggregate"] = per_sample

    if max_k > 1:
        for k in range(2, max_k + 1):
            for name in score_names:
                run_averages = per_sample.get(name, [])[:k]
                if len(run_averages) >= 2:
                    mean_val = sum(run_averages) / len(run_averages)
                    variance = sum((x - mean_val) ** 2 for x in run_averages) / (
                        len(run_averages) - 1
                    )
                    std_dev = math.sqrt(variance)
                    metrics[f"pass@1[avg-of-{k}]/{name}/std_dev_across_runs"] = std_dev
                    metrics[f"pass@1[avg-of-{k}]/{name}/std_err_across_runs"] = std_dev / math.sqrt(
                        len(run_averages)
                    )

    return metrics, all_score_dicts, score_names, max_k


def _subset_metrics(tasks, subset_key, score_fn=None, answer_key=None):
    """Verbatim port of ``nemo_gym.reward_profile.compute_subset_metrics``."""
    subsets: dict = {}
    for task_rollouts in tasks:
        value = task_rollouts[0].get(subset_key) if task_rollouts else None
        if value:
            subsets.setdefault(value, []).append(task_rollouts)

    metrics: dict = {}
    for subset_name, subset_tasks in subsets.items():
        subset_metrics, _, _, _ = _pass_majority_metrics(subset_tasks, score_fn=score_fn)
        for key, value in subset_metrics.items():
            if key == "per_sample_aggregate":
                continue
            metrics[f"{subset_name}/{key}"] = value
    return metrics


# ---------------------------------------------------------------------------
# I/O + per-rollout parsing
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _parse_row(row: dict, materialized: dict) -> dict:
    """Flatten the finance-agent fields the report needs from one rollout."""
    response = row.get("response") or {}
    output_items = response.get("output") or []
    usage = response.get("usage") or {}

    offered = {
        t.get("name")
        for t in (response.get("tools") or [])
        if isinstance(t, dict) and t.get("name")
    }

    tool_calls: Counter = Counter()
    num_turns = 0
    for item in output_items:
        if isinstance(item, dict) and item.get("type") == "function_call":
            name = item.get("name")
            tool_calls[name if name in offered else INVALID_TOOL] += 1
            num_turns += 1

    return {
        "uuid": materialized.get("uuid", ""),
        "question_type": materialized.get("question_type", "unknown"),
        "reward": float(row.get("reward", 0.0)),
        "num_turns": num_turns,
        "tool_calls": tool_calls,
        "offered_tools": offered,
        "output_tokens": usage.get("output_tokens", 0),
        "input_tokens": usage.get("input_tokens", 0),
    }


# ---------------------------------------------------------------------------
# Per-seed finance metrics + cross-seed mean/std (finance glue Gym lacks)
# ---------------------------------------------------------------------------


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mu = _mean(vals)
    return (sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def _repeat_metrics(
    parsed: list[dict], tool_names: list[str], max_turns: int, ctx_window: int
) -> dict:
    n = len(parsed)
    if n == 0:
        return {}
    rewards = [p["reward"] for p in parsed]

    type_rewards: dict[str, list[float]] = defaultdict(list)
    for p in parsed:
        type_rewards[p["question_type"]].append(p["reward"])
    per_type = {qt: {"accuracy": _mean(rs), "count": len(rs)} for qt, rs in type_rewards.items()}
    balanced = _mean([v["accuracy"] for v in per_type.values()]) if per_type else 0.0

    turns = [p["num_turns"] for p in parsed]
    tool_avgs = {name: sum(p["tool_calls"].get(name, 0) for p in parsed) / n for name in tool_names}

    return {
        "num_questions": n,
        "overall_accuracy": _mean(rewards),
        "balanced_accuracy": balanced,
        "avg_turns_per_question": _mean(turns),
        "max_turns_count": sum(1 for t in turns if t >= max_turns),
        "tool_calls_per_question": tool_avgs,
        "avg_output_tokens_per_question": _mean([p["output_tokens"] for p in parsed]),
        "ctx_window_agent_count": sum(1 for p in parsed if p["input_tokens"] > ctx_window),
        "per_type_accuracy": per_type,
    }


def _aggregate_repeat_metrics(
    all_repeats: list[dict], tool_names: list[str], type_counts: dict[str, int]
) -> dict:
    def stat(vals: list[float]) -> dict:
        return {"mean": _mean(vals), "std": _std(vals)}

    agg = {
        "num_repeats": len(all_repeats),
        "num_questions": all_repeats[0]["num_questions"],
        "overall_accuracy": stat([m["overall_accuracy"] for m in all_repeats]),
        "balanced_accuracy": stat([m["balanced_accuracy"] for m in all_repeats]),
        "avg_turns_per_question": stat([m["avg_turns_per_question"] for m in all_repeats]),
        "max_turns_count": stat([m["max_turns_count"] for m in all_repeats]),
        "avg_output_tokens_per_question": stat(
            [m["avg_output_tokens_per_question"] for m in all_repeats]
        ),
        "ctx_window_agent_count": stat([m["ctx_window_agent_count"] for m in all_repeats]),
        "tool_calls_per_question": {
            name: stat([m["tool_calls_per_question"][name] for m in all_repeats])
            for name in tool_names
        },
        "per_type_accuracy": {},
    }
    for qt in sorted(type_counts):
        vals = [
            m["per_type_accuracy"][qt]["accuracy"]
            for m in all_repeats
            if qt in m["per_type_accuracy"]
        ]
        agg["per_type_accuracy"][qt] = {**stat(vals), "count": type_counts[qt]}
    return agg


# ---------------------------------------------------------------------------
# Gym-native pass@k (across seeds == across Gym rollout-index)
# ---------------------------------------------------------------------------


def _gym_pass_metrics(parsed_by_repeat: list[list[dict]]) -> dict:
    """Group rollouts by question (uuid) across repeats and hand them to Gym's
    pass@k / subset primitives.  Returns a flat dict merged into metrics.json.
    """
    compute_pass_majority_metrics, compute_subset_metrics = _load_gym_primitives()

    by_uuid: dict[str, list[dict]] = defaultdict(list)
    for repeat_rows in parsed_by_repeat:
        for p in repeat_rows:
            if p["uuid"]:
                by_uuid[p["uuid"]].append(p)
    tasks = list(by_uuid.values())
    if not tasks:
        return {}

    score_fn = lambda r: {"accuracy": r["reward"]}  # noqa: E731
    overall, _, _, _ = compute_pass_majority_metrics(tasks, score_fn=score_fn)
    overall.pop("per_sample_aggregate", None)
    subset = compute_subset_metrics(tasks, subset_key="question_type", score_fn=score_fn)
    return {"overall": overall, "by_question_type": subset}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _ordered(items: list[str], order: list[str]) -> list[str]:
    head = [x for x in order if x in items]
    tail = sorted(x for x in items if x not in order)
    return head + tail


def _format_summary(agg: dict, gym_metrics: dict, tool_names: list[str], max_turns: int) -> str:
    def pct(s: dict) -> str:
        return f"{100 * s['mean']:6.2f} ± {100 * s['std']:.2f}"

    def num(s: dict, dec: int = 1) -> str:
        return f"{s['mean']:6.{dec}f} ± {s['std']:.{dec}f}"

    rows: list[tuple[str, str]] = [
        ("Balanced Accuracy", pct(agg["balanced_accuracy"])),
        ("Accuracy", pct(agg["overall_accuracy"])),
    ]
    # Turn and tool rows only mean something where the model is given tools.
    if tool_names:
        rows.append(("Avg. Num turns / question", num(agg["avg_turns_per_question"])))
        rows.append(
            (
                f"Reach max turns question count (max={max_turns})",
                f"{round(agg['max_turns_count']['mean'] * agg['num_repeats'])}",
            )
        )
        for name in _ordered(tool_names, TOOL_ORDER):
            rows.append((TOOL_LABELS.get(name, name), num(agg["tool_calls_per_question"][name], 2)))
    rows.append(
        (
            "Avg. Out tokens across all turns / question",
            num(agg["avg_output_tokens_per_question"], 0),
        )
    )
    for qt in _ordered(list(agg["per_type_accuracy"]), QTYPE_ORDER):
        info = agg["per_type_accuracy"][qt]
        rows.append((f"{qt} (n={info['count']})", pct(info)))

    # Gym-native pass@k (overall), reported alongside the accuracy leaderboard.
    ov = gym_metrics.get("overall", {})
    pass_rows = sorted((k, v) for k, v in ov.items() if k.startswith("pass@") and "std" not in k)

    width = max(len(lbl) for lbl, _ in rows)
    lines = [
        f"GYM-NATIVE EVAL AGGREGATION ({agg['num_repeats']} repeats x "
        f"{agg['num_questions']} questions)",
        "=" * 68,
        "Metrics (mean ± std across repeats):",
    ]
    lines += [f"  {lbl:<{width}}  {val}" for lbl, val in rows]
    if pass_rows:
        lines += ["", "Gym pass@k (combinatorial, across repeats):"]
        lines += [f"  {k:<{width}}  {v:.2f}" for k, v in pass_rows]
    lines += ["=" * 68]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def aggregate(
    rollouts_file: str,
    materialized_inputs_file: str,
    output_dir: str,
    *,
    expected_repeats: int | None = None,
    max_turns: int = 50,
    ctx_window: int = 120_000,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rollouts = _load_jsonl(Path(rollouts_file))
    materialized = _load_jsonl(Path(materialized_inputs_file))
    key = lambda row: (row["_ng_task_index"], row["_ng_rollout_index"])  # noqa: E731
    inputs_by_key = {key(row): row for row in materialized}
    results_by_key = {key(row): row for row in rollouts}
    if len(inputs_by_key) != len(materialized) or len(results_by_key) != len(rollouts):
        raise RuntimeError("duplicate task/repeat identity in canonical Gym artifacts")
    if inputs_by_key.keys() != results_by_key.keys() or not rollouts:
        raise RuntimeError("rollouts and materialized inputs must have identical non-empty keys")

    parsed_by_index: dict[int, list[dict]] = defaultdict(list)
    for identity in sorted(results_by_key):
        repeat_index = identity[1]
        if isinstance(repeat_index, bool) or not isinstance(repeat_index, int):
            raise RuntimeError(f"invalid _ng_rollout_index: {repeat_index!r}")
        parsed_by_index[repeat_index].append(
            _parse_row(results_by_key[identity], inputs_by_key[identity])
        )
    repeat_indices = sorted(parsed_by_index)
    if repeat_indices != list(range(len(repeat_indices))):
        raise RuntimeError(f"repeat indices must be contiguous from zero: {repeat_indices}")
    if expected_repeats is not None and len(repeat_indices) != expected_repeats:
        raise RuntimeError(
            f"Expected {expected_repeats} repeats, found {len(repeat_indices)}: {repeat_indices}"
        )
    parsed_by_repeat = [parsed_by_index[index] for index in repeat_indices]

    # Report the tools this benchmark actually offers the model, plus any name the
    # model invented (bucketed as INVALID_TOOL). QA benchmarks offer none, so they
    # get no tool rows at all.
    tool_names = sorted(
        {name for repeat in parsed_by_repeat for p in repeat for name in p["offered_tools"]}
        | {name for repeat in parsed_by_repeat for p in repeat for name in p["tool_calls"]}
    )
    # Canonical per-type counts from the first repeat (all repeats cover the same
    # question set) so the report shows n= per question type.
    type_counts = Counter(p["question_type"] for p in parsed_by_repeat[0])

    repeat_metrics = [
        _repeat_metrics(repeat, tool_names, max_turns, ctx_window) for repeat in parsed_by_repeat
    ]
    agg = _aggregate_repeat_metrics(repeat_metrics, tool_names, type_counts)
    gym_metrics = _gym_pass_metrics(parsed_by_repeat)

    metrics = {"finance": agg, "gym": gym_metrics}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    summary = _format_summary(agg, gym_metrics, tool_names, max_turns)
    print(summary)
    (out / "summary.txt").write_text(summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finance reporting over canonical native Gym repeats."
    )
    parser.add_argument("rollouts_file")
    parser.add_argument("materialized_inputs_file")
    parser.add_argument("output_dir", help="Directory to write aggregated outputs.")
    parser.add_argument("--expected-repeats", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--ctx-window", type=int, default=120_000)
    args = parser.parse_args()

    aggregate(
        args.rollouts_file,
        args.materialized_inputs_file,
        args.output_dir,
        expected_repeats=args.expected_repeats,
        max_turns=args.max_turns,
        ctx_window=args.ctx_window,
    )


if __name__ == "__main__":
    main()
