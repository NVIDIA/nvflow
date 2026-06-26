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
"""Aggregate rollouts across seeds and compute difficulty metrics.

Reads merged rollout files (``output-rs*.jsonl``) from a
``collect_rollouts`` output directory, groups rows by ``uuid`` across
seeds, and computes:

- **avg_reward**: mean reward across seeds per question.
- **reward_std**: sample standard deviation of rewards per question.
  Used by ``filter_training_data`` to identify learnable questions
  (those with non-zero reward variance provide GRPO gradient signal).
- **global_max**: highest reward observed across all data, used as the
  ``c`` threshold for pass@k (scale-independent).
- **pass@k**: unbiased combinatorial estimator (binary — only
  reward == global_max counts as correct).  Standard metric for
  reporting.

Standalone script that runs inside the Slurm container with python3.

Usage:
    python aggregate_seeds.py <rollout_dir> <output_dir>

Produces:
    <output_dir>/summary.txt       -- human-readable report
    <output_dir>/metrics.json      -- machine-readable metrics
    <output_dir>/difficulty.jsonl   -- per-question reward stats and pass@k
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


def _check_seed_survival(
    rollout_files: list[Path],
    expected_num_seeds: int | None,
) -> None:
    """Warn on partially-dead seeds; only fail if the live count is < 1.

    Dead-seed policy (NV-5): a rollout/verify seed that dies or yields no usable
    data mid-run should be *skipped with a logged warning* and aggregation
    proceeds over the survivors, as long as at least one seed survives.  Only a
    total wipe-out (zero usable seed files) is a hard, unambiguous failure.

    ``expected_num_seeds`` is the seed count the launcher *intended* to collect
    (``num_random_seeds``); when provided we can distinguish a partial death
    (fewer files than expected) from a clean full set.  When ``None`` we can
    only act on what is on disk.
    """
    live = len(rollout_files)
    if expected_num_seeds is not None and live < expected_num_seeds:
        print(
            "[nvflow] WARNING: dead-seed(s) detected -- found "
            f"{live} of {expected_num_seeds} expected seed file(s). "
            "Proceeding with the survivors; pass@k will be computed over "
            f"k=1..{live} only.",
            file=sys.stderr,
        )


def aggregate(
    rollout_dir: str,
    output_dir: str,
    output_filename: str = "difficulty.jsonl",
    expected_num_seeds: int | None = None,
) -> None:
    rollout_path = Path(rollout_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rollout_files = sorted(rollout_path.glob("output-rs*.jsonl"))
    rollout_files = [
        f for f in rollout_files if "_chunk_" not in f.name and not f.name.endswith("-async")
    ]
    # Skip any seed file that produced no usable bytes (a dead/empty seed):
    # keep it out of num_seeds so pass@k k-range and DP counts stay honest.
    nonempty_files = [f for f in rollout_files if f.stat().st_size > 0]
    if len(nonempty_files) != len(rollout_files):
        dead = [f.name for f in rollout_files if f.stat().st_size == 0]
        print(
            f"[nvflow] WARNING: dropping {len(dead)} empty seed file(s): {dead}",
            file=sys.stderr,
        )
    rollout_files = nonempty_files

    # Dead-seed policy (NV-5): fail loudly only when ALL seeds are gone;
    # otherwise warn and aggregate over the survivors.
    if not rollout_files:
        logger.error("No usable rollout files found -- all seeds are dead/empty.")
        (out / "summary.txt").write_text("No usable rollout files found (all seeds dead).\n")
        print(
            "[nvflow] ERROR: every rollout seed is dead/empty; refusing to emit "
            "an empty difficulty set. Check the rollout/judge logs.",
            file=sys.stderr,
        )
        sys.exit(1)

    _check_seed_survival(rollout_files, expected_num_seeds)

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
        logger.error("No uuid-keyed rows found across any seed -- no usable rollout data.")
        (out / "summary.txt").write_text("No uuid-keyed rows found.\n")
        print(
            "[nvflow] ERROR: no uuid-keyed rollout rows across any surviving seed; "
            "nothing to aggregate. Check the rollout output schema / judge.",
            file=sys.stderr,
        )
        sys.exit(1)

    k_values = list(range(1, num_seeds + 1))
    records: list[dict] = []

    global_max = max(r.get("reward", 0.0) for rows in by_uuid.values() for r in rows)

    if global_max <= 0:
        print(
            "[nvflow] WARNING: aggregate_seeds observed no positive rewards "
            f"(global_max={global_max}). Reporting pass@k=0 by convention; "
            "the underlying reward / verifier / judge is likely misconfigured.",
            file=sys.stderr,
        )

    for uid, rows in by_uuid.items():
        n = len(rows)
        rewards = [r.get("reward", 0.0) for r in rows]
        c = 0 if global_max <= 0 else sum(1 for rw in rewards if rw == global_max)
        avg_reward = sum(rewards) / n if n > 0 else 0.0
        reward_std = (
            (sum((rw - avg_reward) ** 2 for rw in rewards) / (n - 1)) ** 0.5 if n > 1 else 0.0
        )
        reward_min = min(rewards)
        reward_max = max(rewards)
        question_type = rows[0].get("question_type", "unknown")

        rec: dict = {
            "uuid": uid,
            "n": n,
            "c": c,
            "avg_reward": avg_reward,
            "reward_std": reward_std,
            "reward_min": reward_min,
            "reward_max": reward_max,
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
        "global_max_reward": global_max,
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

    # Difficulty distribution (reward_std buckets).
    reward_stds = [r["reward_std"] for r in records]
    avg_reward_val = sum(r["avg_reward"] for r in records) / num_questions
    avg_reward_std = sum(reward_stds) / num_questions

    buckets: Counter = Counter()
    for s in reward_stds:
        buckets[round(s, 4)] += 1

    metrics["difficulty"] = {
        "avg_reward": avg_reward_val,
        "avg_reward_std": avg_reward_std,
        "distribution": {f"{k:.4f}": v for k, v in sorted(buckets.items())},
    }

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics -> %s", out / "metrics.json")

    records_sorted = sorted(records, key=lambda r: r["reward_std"], reverse=True)
    difficulty_path = out / output_filename
    with open(difficulty_path, "w") as f:
        for r in records_sorted:
            f.write(json.dumps(r) + "\n")
    logger.info("Difficulty -> %s", difficulty_path)

    # Human-readable summary.
    sorted_std_keys = sorted(buckets.keys())
    has_signal = sum(1 for s in reward_stds if s > 0)

    lines = [
        "CROSS-SEED AGGREGATION",
        "=" * 60,
        f"Seeds:             {num_seeds}",
        f"Questions (uuid):  {num_questions}",
        f"Total rows:        {total_rows}",
        f"Global max reward: {global_max}",
        f"Avg reward:        {avg_reward_val:.4f}",
        f"Avg reward std:    {avg_reward_std:.4f}",
        f"Has signal (std>0):{has_signal} ({has_signal / num_questions:.1%})",
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

    # Reward std histogram.
    lines.append("Reward std distribution (per-question sample std across seeds):")
    for std_val in sorted_std_keys:
        count = buckets[std_val]
        pct = count / num_questions * 100
        bar = "#" * int(pct / 100 * 40)
        lines.append(f"  {std_val:6.4f}: {count:5d} ({pct:5.1f}%) {bar}")
    lines.append("")

    # Per-type difficulty breakdown.
    std_labels = [f"{s:.4f}" for s in sorted_std_keys]
    header = f"  {'Type':<15} {'Total':>5}"
    for lbl in std_labels:
        header += f" {lbl:>7}"
    lines.append("By question type:")
    lines.append(header)
    lines.append("  " + "-" * (22 + 8 * len(std_labels)))
    for qt in sorted(by_type.keys()):
        qt_records = by_type[qt]
        qt_buckets: Counter = Counter()
        for r in qt_records:
            qt_buckets[round(r["reward_std"], 4)] += 1
        row = f"  {qt:<15} {len(qt_records):>5}"
        for std_val in sorted_std_keys:
            row += f" {qt_buckets.get(std_val, 0):>7}"
        lines.append(row)
    lines.append("")
    lines.append("=" * 60)

    summary = "\n".join(lines)
    logger.info("\n%s", summary)
    (out / "summary.txt").write_text(summary + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate rollouts across seeds and compute difficulty metrics"
    )
    parser.add_argument("rollout_dir", help="Directory containing output-rs*.jsonl rollout files.")
    parser.add_argument("output_dir", help="Directory to write aggregated outputs.")
    parser.add_argument(
        "--output_filename",
        default="difficulty.jsonl",
        help="Filename for the per-question reward stats JSONL (default: difficulty.jsonl).",
    )
    parser.add_argument(
        "--expected_num_seeds",
        type=int,
        default=None,
        help=(
            "Seed count the launcher intended to collect (num_random_seeds). "
            "When set, a smaller live count is reported as a dead-seed warning "
            "(aggregation still proceeds over the survivors)."
        ),
    )
    args = parser.parse_args()
    aggregate(
        args.rollout_dir,
        args.output_dir,
        output_filename=args.output_filename,
        expected_num_seeds=args.expected_num_seeds,
    )
