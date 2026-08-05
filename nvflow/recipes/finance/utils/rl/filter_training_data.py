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
"""Filter training data using reward-variance difficulty analysis.

Joins ``train.jsonl`` (from prepare_data) with ``difficulty.jsonl``
(from collect_rollouts or compute_rewards aggregate) on ``uuid`` and
keeps only questions with measured RL signal -- i.e. those that were
profiled AND have reward variance >= ``min_reward_std``.

Drops two classes of samples:

* Profiled but zero-signal (``reward_std < min_reward_std``): all seeds
  got the same reward, so the prompt produces no GRPO gradient.
* Unprofiled (not present in difficulty.jsonl): no measured signal at
  all -- typically because ``collect_rollouts.rollout.max_num_samples``
  is set to a value smaller than ``len(train.jsonl)`` (smoke / sweep
  configs), or because rollouts crashed for that sample. In production
  ``max_num_samples`` is unset, every sample is rolled out, and the
  unprofiled set is empty -- so this only affects smoke / sweep runs.

Standalone script that runs inside the Slurm container with python3.

Usage:
    python filter_training_data.py <train.jsonl> <difficulty.jsonl> <output_dir> \\
        [--min-reward-std 1e-6] [--validation-data <val.jsonl>]

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
    min_reward_std: float = 1e-6,
    validation_path: str | None = None,
    policy_model: str | None = None,
    judge_model: str | None = None,
    train_filename: str = "train.jsonl",
    val_filename: str = "validation.jsonl",
    report_filename: str = "filter_report.json",
) -> dict[str, Any]:
    """Filter training data by reward variance threshold.

    Retains ONLY samples with measured RL signal: profiled in
    ``difficulty.jsonl`` AND ``reward_std >= min_reward_std``. Both
    zero-signal profiled samples and unprofiled samples are dropped.

    Args:
        train_path: Path to prepare_data train.jsonl.
        difficulty_path: Path to aggregate/difficulty.jsonl.
        output_dir: Directory for filtered output files.
        min_reward_std: Minimum reward_std to keep (questions below are removed).
        validation_path: Optional path to validation.jsonl (copied unchanged).
        policy_model: Optional policy model name for provenance in difficulty_profile.
        judge_model: Optional judge model name for provenance in difficulty_profile.
        train_filename: Filename for the filtered training output (default "train.jsonl").
        val_filename: Filename for the validation output (default "validation.jsonl").
        report_filename: Filename for the JSON filter report written to
            ``{output_dir}/filter/`` (default "filter_report.json").

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
        logger.info("Passthrough mode: copying train unchanged.")
        shutil.copy2(train_path, out / train_filename)
        if validation_path and Path(validation_path).exists():
            shutil.copy2(validation_path, out / val_filename)
        report: dict[str, Any] = {"mode": "passthrough", "reason": "difficulty.jsonl not found"}
        _write_report(filter_dir, report, report_filename=report_filename)
        return report

    # Load difficulty records to merge into output and use for filtering.
    profile_fields = ("avg_reward", "reward_std", "reward_min", "reward_max", "n", "c", "pass@1")

    difficulty: dict[str, dict[str, Any]] = {}
    with open(diff_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            uid = rec.get("uuid", "")
            if uid:
                difficulty[uid] = {k: rec.get(k) for k in profile_fields if k in rec}

    logger.info("Loaded %d questions from difficulty.jsonl", len(difficulty))
    logger.info("Filter: keep reward_std >= %s; drop unprofiled", min_reward_std)

    total = 0
    kept = 0
    removed_no_signal = 0
    removed_no_profile = 0
    by_type_total: Counter = Counter()
    by_type_kept: Counter = Counter()

    with open(train_path) as fin, open(out / train_filename, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            uid = row.get("uuid", "")
            qtype = row.get("question_type", "unknown")
            by_type_total[qtype] += 1

            diff_rec = difficulty.get(uid)

            if diff_rec is None:
                removed_no_profile += 1
                continue

            rs = diff_rec.get("reward_std", 0.0)
            if rs < min_reward_std:
                removed_no_signal += 1
                continue

            profile = dict(diff_rec)
            if policy_model:
                profile["policy_model"] = policy_model
            if judge_model:
                profile["judge_model"] = judge_model
            row["difficulty_profile"] = profile
            kept += 1
            by_type_kept[qtype] += 1
            fout.write(json.dumps(row) + "\n")

    if validation_path and Path(validation_path).exists():
        shutil.copy2(validation_path, out / val_filename)
        logger.info("Validation data copied unchanged -> %s", out / val_filename)

    report = {
        "mode": "filtered",
        "min_reward_std": min_reward_std,
        "total_questions": total,
        "kept": kept,
        "removed_no_signal": removed_no_signal,
        "removed_no_profile": removed_no_profile,
        "kept_pct": kept / total if total > 0 else 0.0,
        "by_question_type": {
            qt: {"total": by_type_total[qt], "kept": by_type_kept[qt]}
            for qt in sorted(by_type_total.keys())
        },
    }

    _write_report(filter_dir, report, report_filename=report_filename)
    _print_summary(report)
    return report


def _write_report(
    report_dir: Path, report: dict, *, report_filename: str = "filter_report.json"
) -> None:
    report_path = report_dir / report_filename
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report -> %s", report_path)


def _print_summary(report: dict) -> None:
    total = report["total_questions"]
    lines = [
        "",
        "=" * 60,
        "TRAINING DATA FILTER REPORT",
        "=" * 60,
        f"Total questions:        {total}",
        f"Kept (RL signal):       {report['kept']} ({report['kept_pct']:.1%})",
        f"Removed (no signal):    {report['removed_no_signal']}",
        f"Removed (no profile):   {report['removed_no_profile']}",
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
        "--min-reward-std",
        type=float,
        default=1e-6,
        help="Minimum reward_std to keep (default: 1e-6, removes zero-variance questions)",
    )
    parser.add_argument(
        "--validation-data",
        default=None,
        help="Path to validation.jsonl (copied unchanged)",
    )
    parser.add_argument(
        "--policy-model",
        default=None,
        help="Policy model name for provenance in difficulty_profile (optional)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model name for provenance in difficulty_profile (optional)",
    )
    parser.add_argument(
        "--train-filename",
        default="train.jsonl",
        help="Filename for the filtered training output (default: train.jsonl). "
        "Match the consumer's expected filename (training.py train_filename).",
    )
    parser.add_argument(
        "--val-filename",
        default="validation.jsonl",
        help="Filename for the validation output (default: validation.jsonl). "
        "Match the consumer's expected filename (training.py val_filename).",
    )
    parser.add_argument(
        "--report-filename",
        default="filter_report.json",
        help="Filename for the filter-report JSON inside {output_dir}/filter/ "
        "(default: filter_report.json).",
    )

    args = parser.parse_args()
    filter_training_data(
        args.train_path,
        args.difficulty_path,
        args.output_dir,
        min_reward_std=args.min_reward_std,
        validation_path=args.validation_data,
        policy_model=args.policy_model,
        judge_model=args.judge_model,
        train_filename=args.train_filename,
        val_filename=args.val_filename,
        report_filename=args.report_filename,
    )
