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
    python analyze_rollouts.py <rollouts.jsonl> <output_dir> [title]

Produces:
    <output_dir>/summary.txt       -- human-readable analysis
    <output_dir>/correct.jsonl     -- samples with reward == 1.0
    <output_dir>/incorrect.jsonl   -- samples with reward == 0.0
    <output_dir>/partial.jsonl     -- samples with 0 < reward < 1
    <output_dir>/judge_failed.jsonl -- samples with no judge evaluations

Cross-seed difficulty analysis (pass@k, per-question pass rates) is
handled separately by aggregate_seeds.py.
"""

import json
import sys
from collections import Counter
from pathlib import Path

from nvflow.utils import setup_logger

logger = setup_logger(__name__)


def analyze(
    rollout_file: str,
    output_dir: str,
    title: str = "ROLLOUT ANALYSIS",
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
    correct = [r for r in rollouts if r.get("reward", 0.0) == 1.0]
    incorrect = [r for r in rollouts if r.get("reward", 0.0) == 0.0]
    partial = [r for r in rollouts if 0.0 < r.get("reward", 0.0) < 1.0]

    avg_reward = sum(rewards) / total
    min_reward = min(rewards)
    max_reward = max(rewards)

    verdict_counts: Counter = Counter()
    judge_failed = []
    for r in rollouts:
        evals = r.get("judge_evaluations", [])
        if not evals:
            judge_failed.append(r)
        for ev in evals:
            verdict_counts[ev.get("verdict_label", "UNKNOWN")] += 1

    lines = [
        title,
        "=" * 60,
        f"Total samples:     {total}",
        f"Correct (1.0):     {len(correct):5d} ({len(correct) / total * 100:5.1f}%)",
        f"Incorrect (0.0):   {len(incorrect):5d} ({len(incorrect) / total * 100:5.1f}%)",
        f"Partial (0<r<1):   {len(partial):5d} ({len(partial) / total * 100:5.1f}%)",
        f"Judge failed:      {len(judge_failed):5d} ({len(judge_failed) / total * 100:5.1f}%)",
        "",
        f"pass@1:            {avg_reward:.4f} ({avg_reward * 100:.1f}%)",
        "",
        "Reward distribution:",
        f"  Mean:  {avg_reward:.4f}",
        f"  Min:   {min_reward:.4f}",
        f"  Max:   {max_reward:.4f}",
        "",
    ]

    if verdict_counts:
        lines.append("Judge verdicts (per evaluation):")
        for verdict, count in verdict_counts.most_common():
            lines.append(f"  {verdict}: {count}")
        lines.append("")

    lines.append("RL signal assessment:")
    if len(correct) == total:
        lines.append("  WARNING: All rewards are 1.0 -- no negative signal for RL.")
    elif len(incorrect) == total:
        lines.append("  WARNING: All rewards are 0.0 -- no positive signal for RL.")
    elif avg_reward > 0.95:
        lines.append("  WARNING: Very high baseline (>95%) -- limited room for RL improvement.")
    elif avg_reward < 0.05:
        lines.append("  WARNING: Very low baseline (<5%) -- model may struggle to learn.")
    else:
        lines.append(f"  OK: Mixed rewards ({avg_reward:.1%} accuracy) -- good signal for RL.")
    lines.append("")

    lines.append("Interactive viewer (browse individual rollouts with Gradio):")
    lines.append(f"  ng_viewer +jsonl_fpath={rollout_file}")
    lines.append("=" * 60)

    summary = "\n".join(lines)
    logger.info("\n%s", summary)
    (out / "summary.txt").write_text(summary + "\n")

    def _save(name, items):
        if items:
            with open(out / f"{name}.jsonl", "w") as f:
                for item in items:
                    f.write(json.dumps(item) + "\n")
            logger.info("  Saved %d samples -> %s.jsonl", len(items), name)

    _save("correct", correct)
    _save("incorrect", incorrect)
    _save("partial", partial)
    _save("judge_failed", judge_failed)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        logger.error("Usage: python analyze_rollouts.py <rollouts.jsonl> <output_dir> [title]")
        sys.exit(1)
    title = sys.argv[3] if len(sys.argv) > 3 else "ROLLOUT ANALYSIS"
    analyze(sys.argv[1], sys.argv[2], title)
