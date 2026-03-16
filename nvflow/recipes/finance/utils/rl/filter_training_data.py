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
"""Filter training data using reward-profile difficulty analysis.

Joins ``train.jsonl`` (from prepare_data) with ``difficulty.jsonl``
(from collect_rollouts or compute_rewards aggregate) on ``uuid`` and
keeps only questions whose pass rate falls within a configurable
"sweet spot" range.  Questions that are too hard (pass_rate == 0) or
too easy (pass_rate == 1) provide little GRPO learning signal and are
removed by default.

Questions not found in ``difficulty.jsonl`` (unprofiled) are kept by
default -- only explicitly identified too-hard / too-easy questions are
removed.

Standalone script that runs inside the Slurm container with python3.

Usage:
    python filter_training_data.py <train.jsonl> <difficulty.jsonl> <output_dir> \\
        [--min-pass-rate 0.0] [--max-pass-rate 1.0] [--validation-data <val.jsonl>]

Produces:
    <output_dir>/train.jsonl                -- filtered training data (same schema)
    <output_dir>/validation.jsonl           -- validation data (copied unchanged)
    <output_dir>/filter/filter_report.json  -- filtering statistics
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from nvflow.utils import setup_logger

logger = setup_logger(__name__)


def filter_training_data(
    train_path: str,
    difficulty_path: str,
    output_dir: str,
    *,
    min_pass_rate: float = 0.0,
    max_pass_rate: float = 1.0,
    validation_path: str | None = None,
) -> dict[str, Any]:
    """Filter training data by pass-rate thresholds.

    Args:
        train_path: Path to prepare_data train.jsonl.
        difficulty_path: Path to aggregate/difficulty.jsonl.
        output_dir: Directory for filtered output files.
        min_pass_rate: Exclusive lower bound (questions with pass_rate <= min are removed).
        max_pass_rate: Exclusive upper bound (questions with pass_rate >= max are removed).
        validation_path: Optional path to validation.jsonl (copied unchanged).

    Returns:
        Report dict with filtering statistics.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    filter_dir = out / "filter"
    filter_dir.mkdir(parents=True, exist_ok=True)

    diff_file = Path(difficulty_path)
    if not diff_file.exists():
        logger.warning("difficulty file not found: %s", difficulty_path)
        logger.info("Passthrough mode: copying train.jsonl unchanged.")
        shutil.copy2(train_path, out / "train.jsonl")
        if validation_path and Path(validation_path).exists():
            shutil.copy2(validation_path, out / "validation.jsonl")
        report: dict[str, Any] = {"mode": "passthrough", "reason": "difficulty.jsonl not found"}
        _write_report(filter_dir, report)
        return report

    pass_rates: dict[str, float] = {}
    with open(diff_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            uid = rec.get("uuid", "")
            if uid:
                pass_rates[uid] = rec.get("pass_rate", 0.0)

    logger.info("Loaded %d questions from difficulty.jsonl", len(pass_rates))
    logger.info("Filter: keep %s < pass_rate < %s", min_pass_rate, max_pass_rate)

    total = 0
    kept = 0
    kept_no_profile = 0
    removed_too_hard = 0
    removed_too_easy = 0
    by_type_total: Counter = Counter()
    by_type_kept: Counter = Counter()

    with open(train_path) as fin, open(out / "train.jsonl", "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            uid = row.get("uuid", "")
            qtype = row.get("question_type", "unknown")
            by_type_total[qtype] += 1

            if uid not in pass_rates:
                kept += 1
                kept_no_profile += 1
                by_type_kept[qtype] += 1
                fout.write(json.dumps(row) + "\n")
                continue

            pr = pass_rates[uid]
            if pr <= min_pass_rate:
                removed_too_hard += 1
                continue
            if pr >= max_pass_rate:
                removed_too_easy += 1
                continue

            kept += 1
            by_type_kept[qtype] += 1
            fout.write(json.dumps(row) + "\n")

    if validation_path and Path(validation_path).exists():
        shutil.copy2(validation_path, out / "validation.jsonl")
        logger.info("Validation data copied unchanged -> %s", out / "validation.jsonl")

    report = {
        "mode": "filtered",
        "min_pass_rate": min_pass_rate,
        "max_pass_rate": max_pass_rate,
        "total_questions": total,
        "kept": kept,
        "kept_no_profile": kept_no_profile,
        "removed_too_hard": removed_too_hard,
        "removed_too_easy": removed_too_easy,
        "kept_pct": kept / total if total > 0 else 0.0,
        "by_question_type": {
            qt: {"total": by_type_total[qt], "kept": by_type_kept[qt]}
            for qt in sorted(by_type_total.keys())
        },
    }

    _write_report(filter_dir, report)
    _print_summary(report)
    return report


def _write_report(report_dir: Path, report: dict) -> None:
    with open(report_dir / "filter_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report -> %s", report_dir / "filter_report.json")


def _print_summary(report: dict) -> None:
    total = report["total_questions"]
    lines = [
        "",
        "=" * 60,
        "TRAINING DATA FILTER REPORT",
        "=" * 60,
        f"Total questions:        {total}",
        f"Kept (total):           {report['kept']} ({report['kept_pct']:.1%})",
        f"  Kept (no profile):    {report['kept_no_profile']}",
        f"Removed (too hard):     {report['removed_too_hard']}",
        f"Removed (too easy):     {report['removed_too_easy']}",
        "",
    ]

    by_type = report.get("by_question_type", {})
    if by_type:
        lines.append(f"  {'Type':<20} {'Total':>6} {'Kept':>6} {'Kept%':>7}")
        lines.append("  " + "-" * 42)
        for qt, counts in by_type.items():
            t, k = counts["total"], counts["kept"]
            pct = k / t * 100 if t > 0 else 0.0
            lines.append(f"  {qt:<20} {t:>6} {k:>6} {pct:>6.1f}%")
        lines.append("")

    lines.append("=" * 60)
    logger.info("\n%s", "\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter training data by reward-profile difficulty."
    )
    parser.add_argument("train_path", help="Path to train.jsonl")
    parser.add_argument("difficulty_path", help="Path to difficulty.jsonl")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.0,
        help="Exclusive lower bound (default: 0.0, removes 0%% pass rate)",
    )
    parser.add_argument(
        "--max-pass-rate",
        type=float,
        default=1.0,
        help="Exclusive upper bound (default: 1.0, removes 100%% pass rate)",
    )
    parser.add_argument(
        "--validation-data",
        default=None,
        help="Path to validation.jsonl (copied unchanged)",
    )

    args = parser.parse_args()
    filter_training_data(
        args.train_path,
        args.difficulty_path,
        args.output_dir,
        min_pass_rate=args.min_pass_rate,
        max_pass_rate=args.max_pass_rate,
        validation_path=args.validation_data,
    )
