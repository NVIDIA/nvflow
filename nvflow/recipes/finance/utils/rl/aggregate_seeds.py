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
"""Aggregate rollouts across seeds and compute pass@k metrics.

Reads merged rollout files (``output-rs*.jsonl``) from a
``collect_rollouts`` output directory, groups rows by ``uuid`` across
seeds, and computes the unbiased pass@k estimator for every k from 1 to num_seeds.

Standalone script that runs inside the Slurm container with python3.

Usage:
    python aggregate_seeds.py <rollout_dir> <output_dir>

Produces:
    <output_dir>/summary.txt       -- human-readable report
    <output_dir>/metrics.json      -- machine-readable metrics
    <output_dir>/difficulty.jsonl   -- per-question pass rates and pass@k
"""

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from nvflow.utils import setup_logger

logger = setup_logger(__name__)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k.

    Probability that at least one of k randomly chosen samples (without
    replacement) from n total samples is correct, given c correct samples.

    Formula: 1 - C(n-c, k) / C(n, k)
    Same estimator used by nemo-skills and Gym (pass_k_utils.py).
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / i for i in range(n - c + 1, n + 1))


def aggregate(rollout_dir: str, output_dir: str) -> None:
    rollout_path = Path(rollout_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rollout_files = sorted(rollout_path.glob("output-rs*.jsonl"))
    rollout_files = [
        f for f in rollout_files if "_chunk_" not in f.name and not f.name.endswith("-async")
    ]

    if not rollout_files:
        logger.warning("No rollout files found.")
        (out / "summary.txt").write_text("No rollout files found.\n")
        return

    num_seeds = len(rollout_files)
    logger.info("Found %d seed file(s): %s", num_seeds, [f.name for f in rollout_files])

    by_uuid: dict[str, list[dict]] = defaultdict(list)
    total_rows = 0

    for rf in rollout_files:
        with open(rf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                uid = row.get("uuid", "")
                if uid:
                    by_uuid[uid].append(row)
                total_rows += 1

    num_questions = len(by_uuid)
    logger.info("Total rows: %d, unique questions (uuid): %d", total_rows, num_questions)

    if not by_uuid:
        logger.warning("No uuid-keyed rows found.")
        (out / "summary.txt").write_text("No uuid-keyed rows found.\n")
        return

    k_values = list(range(1, num_seeds + 1))
    records: list[dict] = []

    for uid, rows in by_uuid.items():
        n = len(rows)
        c = sum(1 for r in rows if r.get("reward", 0.0) == 1.0)
        question_type = rows[0].get("question_type", "unknown")

        rec: dict = {
            "uuid": uid,
            "n": n,
            "c": c,
            "pass_rate": c / n if n > 0 else 0.0,
            "question_type": question_type,
            "question": rows[0].get("question", ""),
            "expected_answer": rows[0].get("expected_answer", ""),
        }
        for k in k_values:
            if k <= n:
                rec[f"pass@{k}"] = pass_at_k(n, c, k)
        records.append(rec)

    # Aggregate pass@k (macro average across questions).
    metrics: dict = {
        "num_seeds": num_seeds,
        "num_questions": num_questions,
        "total_rows": total_rows,
    }

    for k in k_values:
        key = f"pass@{k}"
        values = [r[key] for r in records if key in r]
        if values:
            metrics[key] = sum(values) / len(values)

    # Breakdown by question_type.
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_type[r["question_type"]].append(r)

    type_metrics: dict[str, dict] = {}
    for qt in sorted(by_type.keys()):
        qt_records = by_type[qt]
        tm: dict = {"count": len(qt_records)}
        for k in k_values:
            key = f"pass@{k}"
            values = [r[key] for r in qt_records if key in r]
            if values:
                tm[key] = sum(values) / len(values)
        type_metrics[qt] = tm

    metrics["by_question_type"] = type_metrics

    # Difficulty distribution.
    pass_rates = [r["pass_rate"] for r in records]
    avg_pass_rate = sum(pass_rates) / num_questions

    buckets: Counter = Counter()
    for p in pass_rates:
        buckets[p] += 1

    metrics["difficulty"] = {
        "avg_pass_rate": avg_pass_rate,
        "distribution": {f"{k:.4f}": v for k, v in sorted(buckets.items())},
    }

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics -> %s", out / "metrics.json")

    records_sorted = sorted(records, key=lambda r: r["pass_rate"])
    with open(out / "difficulty.jsonl", "w") as f:
        for r in records_sorted:
            f.write(json.dumps(r) + "\n")
    logger.info("Difficulty -> %s", out / "difficulty.jsonl")

    # Human-readable summary.
    sorted_rates = sorted(buckets.keys())
    mixed = sum(1 for p in pass_rates if 0.0 < p < 1.0)

    lines = [
        "CROSS-SEED AGGREGATION",
        "=" * 60,
        f"Seeds:             {num_seeds}",
        f"Questions (uuid):  {num_questions}",
        f"Total rows:        {total_rows}",
        f"Avg pass rate:     {avg_pass_rate:.1%}",
        f"Mixed (0<p<1):     {mixed} ({mixed / num_questions:.1%})",
        "",
    ]

    # pass@k by question type.
    header = f"  {'Type':<20} {'Count':>6}"
    for k in k_values:
        header += f"  {'pass@' + str(k):>8}"
    lines.append("pass@k (macro average across questions):")
    lines.append(header)
    lines.append("  " + "-" * (28 + 10 * len(k_values)))
    for qt in sorted(type_metrics.keys()):
        tm = type_metrics[qt]
        row = f"  {qt:<20} {tm['count']:>6}"
        for k in k_values:
            key = f"pass@{k}"
            if key in tm:
                row += f"  {tm[key] * 100:>7.1f}%"
            else:
                row += f"  {'N/A':>8}"
        lines.append(row)
    overall = f"  {'ALL':<20} {num_questions:>6}"
    for k in k_values:
        key = f"pass@{k}"
        if key in metrics:
            overall += f"  {metrics[key] * 100:>7.1f}%"
    lines.append("  " + "-" * (28 + 10 * len(k_values)))
    lines.append(overall)
    lines.append("")

    # Pass rate histogram.
    lines.append("Difficulty distribution:")
    for rate in sorted_rates:
        count = buckets[rate]
        pct = count / num_questions * 100
        bar = "#" * int(pct / 100 * 40)
        correct = round(rate * num_seeds)
        label = f"{correct}/{num_seeds}"
        lines.append(f"  {label:>5s} ({rate:5.1%}): {count:5d} ({pct:5.1f}%) {bar}")
    lines.append("")

    # Per-type difficulty breakdown.
    rate_labels = [f"{round(r * num_seeds)}/{num_seeds}" for r in sorted_rates]
    header = f"  {'Type':<15} {'Total':>5}"
    for lbl in rate_labels:
        header += f" {lbl:>5}"
    lines.append("By question type:")
    lines.append(header)
    lines.append("  " + "-" * (22 + 6 * len(rate_labels)))
    for qt in sorted(by_type.keys()):
        qt_records = by_type[qt]
        qt_buckets: Counter = Counter()
        for r in qt_records:
            qt_buckets[r["pass_rate"]] += 1
        row = f"  {qt:<15} {len(qt_records):>5}"
        for rate in sorted_rates:
            row += f" {qt_buckets.get(rate, 0):>5}"
        lines.append(row)
    lines.append("")
    lines.append("=" * 60)

    summary = "\n".join(lines)
    logger.info("\n%s", summary)
    (out / "summary.txt").write_text(summary + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        logger.error("Usage: python aggregate_seeds.py <rollout_dir> <output_dir>")
        sys.exit(1)
    aggregate(sys.argv[1], sys.argv[2])
