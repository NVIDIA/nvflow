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
"""Collate finance-agent eval stats across checkpoints into one shareable CSV.

Point it at a directory whose immediate subdirectories are the runs to compare
(e.g. a checkpoint sweep root): each subdirectory becomes a CSV column, each
metric a row. Reads `question_type` from the rollout rows (no benchmark file
needed); tool/turn/token averages are per-rollout means over `response.output`
and `response.usage`. Accuracy is avg pass@1 (mean binary reward), averaged
across each checkpoint's native Gym repeats.

Pass --split-file <eval.jsonl> to split each run's rows into two subsets by
question membership (matched on question text): rows in the file vs the rest.
Two CSVs are written (`<out>.in-split.csv` / `<out>.out-split.csv`) so you can
compare how a metric diverges between the subsets across a checkpoint sweep.

Usage:
    python projects/finance/scripts/eval_metrics.py <root_dir> [-o out.csv] [--with-std] [--max-turns 50]
                                   [--split-file <eval.jsonl>]

Example:
    # one column per model in an eval campaign:
    python projects/finance/scripts/eval_metrics.py outputs/finance/baselines -o baselines.csv

    # split each run by whether a question is in the given benchmark file:
    python projects/finance/scripts/eval_metrics.py outputs/finance/baselines \
        --split-file outputs/finance/baselines/step-1-eval-data/fabv1_noweb/eval.jsonl
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# Smart punctuation -> ASCII so datasets that differ only by curly quotes/dashes
# (e.g. "Uber’s" vs "Uber's") still match on question text.
_PUNCT_FOLD = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
}

# Fixed display order mirroring the team spreadsheet; unknown types are appended.
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

# Any function_call whose name isn't one of the tools actually offered to the
# model (read from response.tools) is bucketed here -- covers both parser
# artifacts (reasoning leaked into the name) and hallucinated tool names.
INVALID_TOOL = "invalid_tool_call"


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals):
    if len(vals) < 2:
        return 0.0
    mu = _mean(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))


def _natural_key(name):
    """Sort key so step10 < step25 < step41 (trailing integer, then name)."""
    nums = re.findall(r"\d+", name)
    return (int(nums[-1]) if nums else math.inf, name)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _norm_q(s):
    """Normalize a question string for cross-dataset matching (uuids differ)."""
    s = unicodedata.normalize("NFKC", s or "")
    s = "".join(_PUNCT_FOLD.get(c, c) for c in s)
    return re.sub(r"\s+", " ", s.strip().lower())


def load_split_keys(path):
    """Normalized question texts from a benchmark eval.jsonl (the split membership set)."""
    keys = set()
    for row in load_jsonl(path):
        q = row.get("question") or row.get("problem")
        if q:
            keys.add(_norm_q(q))
    return keys


def parse_entry(entry):
    """Flatten one rollout row into the stats we report."""
    resp = entry.get("response") or {}
    output_items = resp.get("output") or []
    usage = resp.get("usage") or {}

    offered = {
        t.get("name") for t in (resp.get("tools") or []) if isinstance(t, dict) and t.get("name")
    }

    tool_calls = Counter()
    for item in output_items:
        if isinstance(item, dict) and item.get("type") == "function_call":
            name = item.get("name")
            tool_calls[name if name in offered else INVALID_TOOL] += 1

    # Turn == one tool-call round (matches compute_metrics.py's num_turns).
    num_turns = sum(
        1 for it in output_items if isinstance(it, dict) and it.get("type") == "function_call"
    )
    submitted = tool_calls.get("submit_final_result", 0) > 0

    return {
        "reward": float(entry.get("reward", 0.0) or 0.0),
        "question_type": entry.get("question_type", "unknown"),
        "question": entry.get("question") or entry.get("problem") or "",
        "num_turns": num_turns,
        "output_tokens": usage.get("output_tokens", 0) or 0,
        "input_tokens": usage.get("input_tokens", 0) or 0,
        "tool_calls": tool_calls,
        "submitted": submitted,
        "max_output_tokens": resp.get("max_output_tokens"),
        "model": resp.get("model"),
    }


def repeat_metrics(parsed, tool_names, max_turns):
    """Metrics for one native Gym rollout index."""
    n = len(parsed)
    rewards = [p["reward"] for p in parsed]

    type_rewards = defaultdict(list)
    for p in parsed:
        type_rewards[p["question_type"]].append(p["reward"])
    type_acc = {qt: _mean(rs) for qt, rs in type_rewards.items()}
    balanced = _mean(list(type_acc.values()))

    tool_avgs = {name: sum(p["tool_calls"].get(name, 0) for p in parsed) / n for name in tool_names}

    return {
        "num_questions": n,
        "pass@1": _mean(rewards),
        "balanced_accuracy": balanced,
        "avg_turns": _mean([p["num_turns"] for p in parsed]),
        "no_submit": sum(1 for p in parsed if not p["submitted"]),
        "reach_max_turns": sum(1 for p in parsed if p["num_turns"] >= max_turns),
        "avg_output_tokens": _mean([p["output_tokens"] for p in parsed]),
        "avg_input_tokens": _mean([p["input_tokens"] for p in parsed]),
        "tool_calls": tool_avgs,
        "type_acc": type_acc,
        "type_count": {qt: len(rs) for qt, rs in type_rewards.items()},
    }


def find_columns(root):
    """Map column label to canonical rollout and materialized-input files.

    Each immediate subdir is one run. If a run has a single rollout dir the
    column is just the run name; if it has several (multiple benchmarks) each
    gets its own '<run>/<benchmark>' column so they never get mixed.
    """
    columns = {}
    for sub in sorted(
        (p for p in root.iterdir() if p.is_dir()), key=lambda p: _natural_key(p.name)
    ):
        rollout_files = [
            path
            for path in sub.rglob("rollouts.jsonl")
            if path.parent.parent.name == "job-1-analyze-rollouts"
        ]
        if not rollout_files:
            continue
        multi = len(rollout_files) > 1
        for rollouts_file in sorted(rollout_files):
            outputs_dir = rollouts_file.parent
            materialized = outputs_dir / "rollouts_materialized_inputs.jsonl"
            if not materialized.is_file():
                continue
            cell_dir = outputs_dir.parents[2]
            relative_cell = cell_dir.relative_to(sub).as_posix()
            label = f"{sub.name}/{relative_cell}" if multi else sub.name
            columns[label] = (rollouts_file, materialized)
    return columns


def load_columns(root):
    """Read canonical runs once: label -> list of per-repeat parsed rows."""
    columns = find_columns(root)
    if not columns:
        sys.exit(f"No canonical Gym rollout files found under {root}")
    result = {}
    for label, (rollouts_file, materialized_file) in columns.items():
        key = lambda row: (row["_ng_task_index"], row["_ng_rollout_index"])  # noqa: E731
        inputs = {key(row): row for row in load_jsonl(materialized_file)}
        by_repeat = defaultdict(list)
        for rollout in load_jsonl(rollouts_file):
            identity = key(rollout)
            by_repeat[identity[1]].append(parse_entry({**inputs[identity], **rollout}))
        result[label] = [by_repeat[index] for index in sorted(by_repeat)]
    return result


def discover(raw):
    """Tool names + ordered question types across all runs (keeps CSV rows aligned)."""
    tool_names, qtypes = set(), set()
    for parsed_repeats in raw.values():
        for parsed in parsed_repeats:
            for p in parsed:
                tool_names.update(p["tool_calls"].keys())
                qtypes.add(p["question_type"])
    tool_names = sorted(tool_names)
    ordered_qtypes = [q for q in QTYPE_ORDER if q in qtypes] + sorted(
        q for q in qtypes if q not in QTYPE_ORDER
    )
    return tool_names, ordered_qtypes


def aggregate(raw, tool_names, ordered_qtypes, max_turns, keep=None):
    """Per-column aggregate (mean +/- std across repeats)."""
    results = {}
    for label, parsed_repeats in raw.items():
        if keep is not None:
            parsed_repeats = [[p for p in parsed if keep(p)] for parsed in parsed_repeats]
        per_repeat = [
            repeat_metrics(parsed, tool_names, max_turns) for parsed in parsed_repeats if parsed
        ]
        if not per_repeat:
            continue
        agg = {"num_repeats": len(per_repeat)}
        agg["num_questions"] = per_repeat[0]["num_questions"]
        for key in (
            "pass@1",
            "balanced_accuracy",
            "avg_turns",
            "no_submit",
            "reach_max_turns",
            "avg_output_tokens",
            "avg_input_tokens",
        ):
            vals = [m[key] for m in per_repeat]
            agg[key] = (_mean(vals), _std(vals))
        agg["tool_calls"] = {
            name: (
                _mean([m["tool_calls"][name] for m in per_repeat]),
                _std([m["tool_calls"][name] for m in per_repeat]),
            )
            for name in tool_names
        }
        agg["type_acc"] = {}
        agg["type_count"] = {}
        for qt in ordered_qtypes:
            vals = [m["type_acc"][qt] for m in per_repeat if qt in m["type_acc"]]
            agg["type_acc"][qt] = (_mean(vals), _std(vals)) if vals else (float("nan"), 0.0)
            agg["type_count"][qt] = max(
                (m["type_count"].get(qt, 0) for m in per_repeat),
                default=0,
            )
        # Model + max tokens: read from any row.
        model = next(
            (p["model"] for parsed in parsed_repeats for p in parsed if p["model"]),
            "",
        )
        maxtok = next(
            (
                p["max_output_tokens"]
                for parsed in parsed_repeats
                for p in parsed
                if p["max_output_tokens"]
            ),
            "",
        )
        agg["model"] = os.path.basename(model) if model else ""
        agg["max_output_tokens"] = maxtok
        results[label] = agg

    return results


def build_rows(results, tool_names, ordered_qtypes, cols, with_std, max_turns):
    def pct(stat):
        m, s = stat
        if math.isnan(m):
            return ""
        return f"{100 * m:.2f} ± {100 * s:.2f}" if with_std else f"{100 * m:.2f}"

    def num(stat, dec=2):
        m, s = stat
        return f"{m:.{dec}f} ± {s:.{dec}f}" if with_std else f"{m:.{dec}f}"

    rows = []
    rows.append(["Model"] + [results[c]["model"] for c in cols])
    rows.append(["Tools"] + [" + ".join(tool_names) for _ in cols])
    rows.append(["Repeats"] + [results[c]["num_repeats"] for c in cols])
    rows.append(["Number of questions"] + [results[c]["num_questions"] for c in cols])
    rows.append(["MAX output tokens / inference"] + [results[c]["max_output_tokens"] for c in cols])
    rows.append([])
    rows.append(["Accuracy (avg pass@1, %)"] + [pct(results[c]["pass@1"]) for c in cols])
    rows.append(
        ["Balanced accuracy (macro over types, %)"]
        + [pct(results[c]["balanced_accuracy"]) for c in cols]
    )
    rows.append([])
    rows.append(
        ["Avg tool-call rounds / question"] + [num(results[c]["avg_turns"], 1) for c in cols]
    )
    rows.append(
        ["Rollouts without submit (per repeat)"] + [num(results[c]["no_submit"], 1) for c in cols]
    )
    rows.append(
        [f"Reached {max_turns}+ rounds (per repeat)"]
        + [num(results[c]["reach_max_turns"], 1) for c in cols]
    )
    rows.append(
        ["Avg output tokens / question"] + [num(results[c]["avg_output_tokens"], 0) for c in cols]
    )
    rows.append(
        ["Avg input tokens / question"] + [num(results[c]["avg_input_tokens"], 0) for c in cols]
    )
    rows.append([])
    for name in tool_names:
        rows.append([f"{name} / question"] + [num(results[c]["tool_calls"][name], 2) for c in cols])
    rows.append([])
    for qt in ordered_qtypes:
        n = next(
            (results[c]["type_count"][qt] for c in cols if results[c]["type_count"].get(qt)), 0
        )
        label = f"{qt} (n={n}, %)"
        rows.append([label] + [pct(results[c]["type_acc"][qt]) for c in cols])
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Collate finance-agent eval stats across checkpoint folders into one CSV."
    )
    parser.add_argument("root", help="Directory whose immediate subdirs are the runs to compare.")
    parser.add_argument(
        "-o", "--output", default=None, help="Output CSV path (default: <root>/eval_metrics.csv)."
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="Tool-call rounds counted as hitting the step cap (default: 50).",
    )
    parser.add_argument(
        "--with-std",
        action="store_true",
        help="Append '± std' (across repeats) to each numeric cell.",
    )
    parser.add_argument(
        "--split-file",
        default=None,
        help="Benchmark eval.jsonl; split rows into in-file vs the rest "
        "(matched on question text) and write two CSVs.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    raw = load_columns(root)
    tool_names, ordered_qtypes = discover(raw)
    out_path = Path(args.output) if args.output else root / "eval_metrics.csv"

    def write_csv(results, path):
        if not results:
            print(f"Skipped {path} (no rows matched)")
            return
        cols = sorted(results.keys(), key=_natural_key)
        rows = build_rows(results, tool_names, ordered_qtypes, cols, args.with_std, args.max_turns)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Metric"] + cols)
            w.writerows(rows)
        print(f"Wrote {path}  (columns: {len(cols)})")

    if args.split_file:
        keys = load_split_keys(args.split_file)
        all_qs = {_norm_q(p["question"]) for ps in raw.values() for parsed in ps for p in parsed}
        matched = keys & all_qs
        print(
            f"Split file: {len(keys)} questions; matched {len(matched)} in data, "
            f"{len(all_qs) - len(matched)} others."
        )
        stem, ext = os.path.splitext(str(out_path))
        write_csv(
            aggregate(
                raw,
                tool_names,
                ordered_qtypes,
                args.max_turns,
                keep=lambda p: _norm_q(p["question"]) in keys,
            ),
            f"{stem}.in-split{ext}",
        )
        write_csv(
            aggregate(
                raw,
                tool_names,
                ordered_qtypes,
                args.max_turns,
                keep=lambda p: _norm_q(p["question"]) not in keys,
            ),
            f"{stem}.out-split{ext}",
        )
    else:
        write_csv(aggregate(raw, tool_names, ordered_qtypes, args.max_turns), out_path)


if __name__ == "__main__":
    main()
