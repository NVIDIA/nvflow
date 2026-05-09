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
"""Analyze rollouts -- reward distribution and judge verdicts.

Standalone script that runs inside the Slurm container with python3.

Usage:
    python analyze_rollouts.py <rollouts.jsonl> <output_dir> [title] [judge_schema]

Produces:
    <output_dir>/summary.txt        -- human-readable analysis
    <output_dir>/step_metrics.json  -- machine-readable token & step metrics
    <output_dir>/best.jsonl         -- samples with reward == max_reward
    <output_dir>/worst.jsonl        -- samples with reward == min_reward
    <output_dir>/intermediate.jsonl -- samples with min_reward < reward < max_reward
    <output_dir>/judge_failed.jsonl -- samples where judge produced no result

Judge output schema is auto-detected from the data unless explicitly
provided via the ``judge_schema`` argument.  To add a new schema, add an
entry to :data:`_FIELD_TO_SCHEMA` and a handler to :data:`_SCHEMA_HANDLERS`.

Cross-seed difficulty analysis (pass@k, per-question pass rates) is
handled separately by aggregate_seeds.py.
"""

import json
import statistics
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from nvflow.utils import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Judge schema handlers
# ---------------------------------------------------------------------------
# Each handler: (rollouts) -> (verdict_counts, judge_failed)

JudgeHandler = Callable[[list[dict]], tuple[Counter, list[dict]]]


def _judge_evaluations(rollouts: list[dict]) -> tuple[Counter, list[dict]]:
    """``judge_evaluations`` list with ``verdict_label`` (equivalence_llm_judge)."""
    verdict_counts: Counter = Counter()
    judge_failed: list[dict] = []
    for r in rollouts:
        evals = r.get("judge_evaluations", [])
        if not evals:
            judge_failed.append(r)
        for ev in evals:
            verdict_counts[ev.get("verdict_label", "UNKNOWN")] += 1
    return verdict_counts, judge_failed


def _judge_rating(rollouts: list[dict]) -> tuple[Counter, list[dict]]:
    """``judge_rating`` numeric field (finance_sec_search)."""
    verdict_counts: Counter = Counter()
    judge_failed: list[dict] = []
    for r in rollouts:
        rating = r.get("judge_rating")
        if rating is None:
            judge_failed.append(r)
        else:
            verdict_counts[f"rating={rating}"] += 1
    return verdict_counts, judge_failed


def _judge_exact_match(rollouts: list[dict]) -> tuple[Counter, list[dict]]:
    """``extracted_answer`` vs ``expected_answer`` (mcqa -- no external judge)."""
    verdict_counts: Counter = Counter()
    for r in rollouts:
        matched = r.get("extracted_answer") == r.get("expected_answer")
        verdict_counts["match" if matched else "mismatch"] += 1
    return verdict_counts, []


_SCHEMA_HANDLERS: dict[str, JudgeHandler] = {
    "evaluations": _judge_evaluations,
    "rating": _judge_rating,
    "exact_match": _judge_exact_match,
}

_FIELD_TO_SCHEMA = [
    ("judge_evaluations", "evaluations"),
    ("judge_rating", "rating"),
    ("extracted_answer", "exact_match"),
]

_NO_JUDGE_SCHEMAS = {"exact_match"}


def _detect_judge_schema(rollouts: list[dict]) -> str | None:
    """Auto-detect judge output schema by scanning the first few rollouts.

    Returns a key from :data:`_SCHEMA_HANDLERS`, or ``None`` if no
    recognised judge fields are found (reward-only data).
    """
    sample = rollouts[: min(10, len(rollouts))]
    for field, schema in _FIELD_TO_SCHEMA:
        if any(r.get(field) is not None for r in sample):
            return schema
    return None


# ---------------------------------------------------------------------------
# Token & step metrics helpers
# ---------------------------------------------------------------------------

_ERROR_SUBSTRINGS = ("error", "failed", "timed out")


def _dist(values: list[int | float]) -> dict[str, int | float]:
    """Return min/max/mean/median for a list of numeric values."""
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 1),
        "median": round(statistics.median(values), 1),
    }


def _extract_token_metrics(
    rollouts: list[dict], *, max_reward: float, min_reward: float
) -> dict | None:
    """Extract token usage metrics from ``response.usage`` (all environments).

    Returns ``None`` if no rollouts have usage data.
    """
    input_toks: list[int] = []
    output_toks: list[int] = []
    total_toks: list[int] = []
    missing = 0

    for r in rollouts:
        usage = (r.get("response") or {}).get("usage")
        if not usage:
            missing += 1
            continue
        input_toks.append(usage.get("input_tokens", 0))
        output_toks.append(usage.get("output_tokens", 0))
        total_toks.append(usage.get("total_tokens", 0))

    if not input_toks:
        return None

    metrics: dict = {
        "input_tokens_per_rollout": _dist(input_toks),
        "output_tokens_per_rollout": _dist(output_toks),
        "total_tokens_per_rollout": _dist(total_toks),
        "missing_usage": missing,
    }

    _reward_groups: list[tuple[str, Callable[[float], bool]]] = [
        ("best", lambda rw: rw == max_reward),
        ("worst", lambda rw: rw == min_reward),
        ("intermediate", lambda rw: min_reward < rw < max_reward),
    ]

    by_reward: dict[str, dict] = {}
    for label, match in _reward_groups:
        sub_in: list[int] = []
        sub_out: list[int] = []
        sub_tot: list[int] = []
        for r in rollouts:
            usage = (r.get("response") or {}).get("usage")
            if not usage or not match(r.get("reward", 0.0)):
                continue
            sub_in.append(usage.get("input_tokens", 0))
            sub_out.append(usage.get("output_tokens", 0))
            sub_tot.append(usage.get("total_tokens", 0))
        if sub_in:
            by_reward[label] = {
                "count": len(sub_in),
                "input_tokens": _dist(sub_in),
                "output_tokens": _dist(sub_out),
                "total_tokens": _dist(sub_tot),
            }
    metrics["by_reward"] = by_reward
    return metrics


def _is_tool_error(output_text: str) -> bool:
    """Best-effort check whether a function_call_output indicates an error."""
    lower = output_text.lower()
    return any(kw in lower for kw in _ERROR_SUBSTRINGS)


def _extract_step_metrics(
    rollouts: list[dict], *, max_reward: float, min_reward: float
) -> dict | None:
    """Extract multi-step agent metrics from ``response.output``.

    Returns ``None`` when no rollouts contain tool calls (single-step env).
    """
    per_rollout: list[dict] = []

    for r in rollouts:
        output = (r.get("response") or {}).get("output") or []
        items = [item for item in output if isinstance(item, dict)]
        if not items:
            continue

        tool_calls = [i for i in items if i.get("type") == "function_call"]
        tool_outputs = [i for i in items if i.get("type") == "function_call_output"]
        tool_errors = sum(1 for i in tool_outputs if _is_tool_error(i.get("output", "")))

        last_type = items[-1].get("type", "unknown")
        if last_type == "message":
            outcome = "Completed"
        elif last_type in ("function_call_output", "function_call"):
            outcome = "Truncated (mid-tool)"
        elif last_type == "reasoning":
            outcome = "Truncated (mid-reasoning)"
        else:
            outcome = "Completed"

        per_rollout.append(
            {
                "steps": len(items),
                "tool_calls": len(tool_calls),
                "tool_names": [tc.get("name", "unknown") for tc in tool_calls],
                "tool_errors": tool_errors,
                "tool_outputs": len(tool_outputs),
                "outcome": outcome,
                "last_type": last_type,
                "reward": r.get("reward", 0.0),
                "incomplete_details": (r.get("response") or {}).get("incomplete_details"),
            }
        )

    if not per_rollout:
        return None

    tc_counts = [p["tool_calls"] for p in per_rollout]
    if max(tc_counts) == 0:
        return None

    step_counts = [p["steps"] for p in per_rollout]
    total_tool_outputs = sum(p["tool_outputs"] for p in per_rollout)
    total_tool_errors = sum(p["tool_errors"] for p in per_rollout)

    tool_names: Counter = Counter()
    step_types: Counter = Counter()
    error_messages: Counter = Counter()
    for r in rollouts:
        for item in (r.get("response") or {}).get("output") or []:
            if isinstance(item, dict):
                t = item.get("type", "unknown")
                step_types[t] += 1
                if t == "function_call":
                    tool_names[item.get("name", "unknown")] += 1
                elif t == "function_call_output":
                    out_text = item.get("output", "")
                    if _is_tool_error(out_text):
                        snippet = out_text[:120].replace("\n", " ").strip()
                        error_messages[snippet] += 1

    outcome_counts = Counter(p["outcome"] for p in per_rollout)

    max_tc_observed = max(tc_counts)
    truncation_reasons: Counter = Counter()
    for p in per_rollout:
        if p["outcome"].startswith("Truncated"):
            inc = p["incomplete_details"]
            if isinstance(inc, dict) and inc.get("reason"):
                truncation_reasons[inc["reason"]] += 1
            elif p["tool_calls"] == max_tc_observed and max_tc_observed > 1:
                truncation_reasons["tool call ceiling (inferred)"] += 1
            elif p["last_type"] == "reasoning":
                truncation_reasons["max_output_tokens (inferred)"] += 1
            else:
                truncation_reasons["unknown"] += 1

    metrics: dict = {
        "total_rollouts": len(per_rollout),
        "steps_per_rollout": _dist(step_counts),
        "tool_calls_per_rollout": _dist(tc_counts),
        "tool_name_distribution": dict(tool_names.most_common()),
        "tool_error_rate": {
            "errors": total_tool_errors,
            "total": total_tool_outputs,
            "rate": round(total_tool_errors / total_tool_outputs, 4) if total_tool_outputs else 0.0,
        },
        "step_type_distribution": dict(step_types.most_common()),
        "outcome_distribution": dict(outcome_counts.most_common()),
        "truncation_details": dict(truncation_reasons.most_common()),
        "top_tool_errors": [
            {"message": msg, "count": cnt} for msg, cnt in error_messages.most_common(10)
        ],
    }

    by_reward: dict[str, dict] = {}
    for label, reward_val in [("best", max_reward), ("worst", min_reward)]:
        subset = [p for p in per_rollout if p["reward"] == reward_val]
        if not subset:
            continue
        sub_tc = [p["tool_calls"] for p in subset]
        sub_steps = [p["steps"] for p in subset]
        sub_errors = sum(p["tool_errors"] for p in subset)
        sub_outputs = sum(p["tool_outputs"] for p in subset)
        sub_outcomes = Counter(p["outcome"] for p in subset)
        sub_tool_names: Counter = Counter()
        for p in subset:
            sub_tool_names.update(p["tool_names"])
        by_reward[label] = {
            "count": len(subset),
            "tool_calls_per_rollout": _dist(sub_tc),
            "steps_per_rollout": _dist(sub_steps),
            "tool_error_rate": round(sub_errors / sub_outputs, 4) if sub_outputs else 0.0,
            "outcome_distribution": dict(sub_outcomes.most_common()),
            "tool_name_distribution": dict(sub_tool_names.most_common()),
        }

    intermediate_count = sum(1 for p in per_rollout if min_reward < p["reward"] < max_reward)
    if intermediate_count:
        by_reward["intermediate_excluded"] = intermediate_count

    metrics["by_reward"] = by_reward
    return metrics


# ---------------------------------------------------------------------------
# Summary formatting helpers
# ---------------------------------------------------------------------------


def _fmt_dist(d: dict, fmt: str = ",d") -> str:
    """Format a dist dict as ``min=X  max=Y  mean=Z  median=W``."""
    if fmt == ",d":
        return (
            f"min={d['min']:,d}  max={d['max']:,d}  "
            f"mean={int(d['mean']):,d}  median={int(d['median']):,d}"
        )
    return f"min={d['min']:.1f}  max={d['max']:.1f}  mean={d['mean']:.1f}  median={d['median']:.1f}"


def _format_token_section(tm: dict) -> list[str]:
    """Format the unconditional token usage summary section."""
    lines = [
        "",
        "Token Usage:",
        f"  Input tokens:    {_fmt_dist(tm['input_tokens_per_rollout'])}",
        f"  Output tokens:   {_fmt_dist(tm['output_tokens_per_rollout'])}",
        f"  Total tokens:    {_fmt_dist(tm['total_tokens_per_rollout'])}",
    ]
    if tm.get("missing_usage"):
        lines.append(f"  (missing usage data for {tm['missing_usage']} rollouts)")

    by_rw = tm.get("by_reward", {})
    if by_rw:
        header_parts = []
        col_data: list[tuple[str, dict]] = []
        for label in ("best", "worst", "intermediate"):
            if label in by_rw:
                header_parts.append(f"{label.capitalize():>20s}")
                col_data.append((label.capitalize(), by_rw[label]))

        if col_data:
            lines.append("")
            lines.append("  By reward outcome:")
            lines.append(f"  {'':25s}{''.join(header_parts)}")
            counts_row = "".join(f"{d['count']:>20,d}" for _, d in col_data)
            lines.append(f"  {'Samples:':25s}{counts_row}")
            for metric_key, metric_label in [
                ("input_tokens", "Input tokens (mean)"),
                ("output_tokens", "Output tokens (mean)"),
                ("total_tokens", "Total tokens (mean)"),
            ]:
                vals = "".join(f"{int(d[metric_key]['mean']):>20,d}" for _, d in col_data)
                lines.append(f"  {metric_label + ':':25s}{vals}")

    lines.append("")
    return lines


def _format_step_section(sm: dict) -> list[str]:
    """Format the multi-step agent metrics summary section."""
    total = sm["total_rollouts"]
    err = sm["tool_error_rate"]
    lines = [
        "Agent Step Metrics (multi-step):",
        f"  Steps/rollout:       {_fmt_dist(sm['steps_per_rollout'])}",
        f"  Tool calls/rollout:  {_fmt_dist(sm['tool_calls_per_rollout'])}",
        f"  Tool error rate:     {err['rate']:.1%} ({err['errors']}/{err['total']})",
        "",
    ]

    lines.append("  Outcome:")
    for outcome in ("Completed", "Truncated (mid-tool)", "Truncated (mid-reasoning)"):
        count = sm["outcome_distribution"].get(outcome, 0)
        lines.append(f"    {outcome + ':':30s}{count:5d}  ({count / total * 100:5.1f}%)")
    lines.append("")

    trunc = sm.get("truncation_details", {})
    if trunc:
        lines.append("  Truncation reasons:")
        for reason, count in sorted(trunc.items(), key=lambda x: -x[1]):
            lines.append(f"    {reason + ':':38s}{count:5d}")
        lines.append("")

    top_errors = sm.get("top_tool_errors", [])
    if top_errors:
        lines.append("  Top tool errors:")
        for entry in top_errors:
            lines.append(f"    [{entry['count']:4d}x] {entry['message']}")
        lines.append("")

    total_tc = sum(sm["tool_name_distribution"].values())
    lines.append("  Tool call distribution:")
    for name, count in sm["tool_name_distribution"].items():
        lines.append(f"    {name + ':':34s}{count:5d}  ({count / total_tc * 100:5.1f}%)")
    lines.append("")

    total_steps = sum(sm["step_type_distribution"].values())
    lines.append("  Step type distribution:")
    for stype, count in sm["step_type_distribution"].items():
        lines.append(f"    {stype + ':':30s}{count:5d}  ({count / total_steps * 100:5.1f}%)")
    lines.append("")

    by_rw = sm.get("by_reward", {})
    best = by_rw.get("best")
    worst = by_rw.get("worst")
    if best or worst:
        cols = [("Best", best), ("Worst", worst)]
        lines.append("  By reward outcome:")
        lines.append(f"  {'':30s}{''.join(f'{lbl:>20s}' for lbl, _ in cols)}")

        def _col_val(grp: dict | None, key: str, sub: str = "mean") -> str:
            if not grp:
                return f"{'N/A':>20s}"
            v = grp.get(key)
            if isinstance(v, dict):
                return f"{v[sub]:>20.1f}"
            if isinstance(v, int | float):
                return f"{v:>20.1%}"
            return f"{'N/A':>20s}"

        for row_label, key, sub in [
            ("Tool calls (mean):", "tool_calls_per_rollout", "mean"),
            ("Tool calls (median):", "tool_calls_per_rollout", "median"),
            ("Steps (mean):", "steps_per_rollout", "mean"),
            ("Tool error rate:", "tool_error_rate", "mean"),
        ]:
            vals = "".join(_col_val(grp, key, sub) for _, grp in cols)
            lines.append(f"  {row_label:30s}{vals}")

        for row_label, outcome_key in [("Completed:", "Completed"), ("Truncated:", None)]:
            parts: list[str] = []
            for _, grp in cols:
                if not grp:
                    parts.append(f"{'N/A':>20s}")
                    continue
                od = grp.get("outcome_distribution", {})
                n = grp["count"]
                if outcome_key:
                    c = od.get(outcome_key, 0)
                else:
                    c = sum(v for k, v in od.items() if k.startswith("Truncated"))
                parts.append(f"{c:>10d} ({c / n * 100:4.1f}%)")
            lines.append(f"  {row_label:30s}{''.join(parts)}")

    intermediate_exc = by_rw.get("intermediate_excluded")
    if intermediate_exc:
        lines.append(
            f"  ({intermediate_exc} intermediate-reward rollouts excluded from stratification)"
        )

    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def analyze(
    rollout_file: str,
    output_dir: str,
    title: str = "ROLLOUT ANALYSIS",
    judge_schema: str = "auto",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(rollout_file) as f:
        rollouts = [json.loads(line) for line in f if line.strip()]

    if not rollouts:
        logger.warning("No rollouts found.")
        (out / "summary.txt").write_text("No rollouts collected.\n")
        return

    total = len(rollouts)
    rewards = [r.get("reward", 0.0) for r in rollouts]
    max_reward = max(rewards)
    min_reward = min(rewards)
    best = [r for r in rollouts if r.get("reward", 0.0) == max_reward]
    worst = [r for r in rollouts if r.get("reward", 0.0) == min_reward]
    intermediate = [r for r in rollouts if min_reward < r.get("reward", 0.0) < max_reward]

    avg_reward = sum(rewards) / total

    # -- Judge analysis ---------------------------------------------------
    schema = judge_schema if judge_schema != "auto" else _detect_judge_schema(rollouts)
    verdict_counts: Counter = Counter()
    judge_failed: list[dict] = []

    if schema is not None:
        handler = _SCHEMA_HANDLERS.get(schema)
        if handler is None:
            logger.warning("Unknown judge schema '%s', ignoring judge analysis", schema)
            schema = None
        else:
            verdict_counts, judge_failed = handler(rollouts)

    # -- Summary ----------------------------------------------------------
    lines = [
        title,
        "=" * 60,
        f"Total samples:     {total}",
        f"Best (r={max_reward}):    {len(best):5d} ({len(best) / total * 100:5.1f}%)",
        f"Worst (r={min_reward}):   {len(worst):5d} ({len(worst) / total * 100:5.1f}%)",
        f"Intermediate:      {len(intermediate):5d} ({len(intermediate) / total * 100:5.1f}%)",
    ]
    if schema is not None and schema not in _NO_JUDGE_SCHEMAS:
        lines.append(
            f"Judge failed:      {len(judge_failed):5d} ({len(judge_failed) / total * 100:5.1f}%)"
        )
    lines += [
        "",
        f"pass@1:            {avg_reward:.4f} ({avg_reward * 100:.1f}%)",
        "",
        "Reward distribution:",
        f"  Mean:  {avg_reward:.4f}",
        f"  Min:   {min_reward:.4f}",
        f"  Max:   {max_reward:.4f}",
        "",
        f"Judge type: {schema or 'none (reward only)'}",
    ]
    if verdict_counts:
        lines.append("Judge verdicts:")
        for verdict, count in verdict_counts.most_common():
            lines.append(f"  {verdict}: {count}")
    lines.append("")

    lines.append("RL signal assessment:")
    reward_std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
    if max_reward == min_reward:
        lines.append("  WARNING: All rewards identical -- no RL signal.")
    elif reward_std < 0.01 * (max_reward - min_reward):
        lines.append("  WARNING: Very low variance -- weak RL signal.")
    else:
        lines.append(
            f"  OK: Mixed rewards (mean={avg_reward:.4f}, std={reward_std:.4f}) -- good signal for RL."
        )
    lines.append("")

    # -- Token metrics (unconditional) ------------------------------------
    token_metrics = _extract_token_metrics(rollouts, max_reward=max_reward, min_reward=min_reward)
    all_metrics: dict = {}
    if token_metrics:
        lines.extend(_format_token_section(token_metrics))
        all_metrics["token_metrics"] = token_metrics

    # -- Step metrics (multi-step only) ------------------------------------
    step_metrics = _extract_step_metrics(rollouts, max_reward=max_reward, min_reward=min_reward)
    if step_metrics:
        lines.extend(_format_step_section(step_metrics))
        all_metrics["step_metrics"] = step_metrics

    lines.append("Interactive viewer (browse individual rollouts with Gradio):")
    lines.append(f"  ng_viewer +jsonl_fpath={rollout_file}")
    lines.append("=" * 60)

    summary = "\n".join(lines)
    logger.info("\n%s", summary)
    (out / "summary.txt").write_text(summary + "\n")

    if all_metrics:
        (out / "step_metrics.json").write_text(json.dumps(all_metrics, indent=2) + "\n")
        logger.info("  Saved metrics -> step_metrics.json")

    def _save(name: str, items: list[dict]) -> None:
        if items:
            with open(out / f"{name}.jsonl", "w") as f:
                for item in items:
                    f.write(json.dumps(item) + "\n")
            logger.info("  Saved %d samples -> %s.jsonl", len(items), name)

    _save("best", best)
    _save("worst", worst)
    _save("intermediate", intermediate)
    _save("judge_failed", judge_failed)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        logger.error(
            "Usage: python analyze_rollouts.py <rollouts.jsonl> <output_dir> [title] [judge_schema]"
        )
        sys.exit(1)
    _title = sys.argv[3] if len(sys.argv) > 3 else "ROLLOUT ANALYSIS"
    _schema = sys.argv[4] if len(sys.argv) > 4 else "auto"
    analyze(sys.argv[1], sys.argv[2], _title, judge_schema=_schema)
