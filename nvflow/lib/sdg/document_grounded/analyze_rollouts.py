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
"""Sync-up + lightweight stats for rollout-based SDG.

Invoked as the ``analyze_module`` of :func:`nvflow.lib.rl.rollout.rollout`
(runs inside each seed's merge job, after ``enrich``):

    python -m nvflow.lib.sdg.document_grounded.analyze_rollouts <merged_file> <analysis_dir>

Two responsibilities:

1. **Path sync-up (plan L2)**: ``rollout()`` writes merged output to
   ``<output_dir>/rollout/output-rs<seed>.jsonl``, but existing SDG downstream
   stages glob ``output-rs*.jsonl`` at the stage *root* (``<output_dir>/``).
   This copies the enriched merged file up one level so downstream YAML paths
   and ``preprocess.py`` globs keep working unchanged.
2. **Stats**: write a small ``sdg_stats.json`` (row count, reward mean) for
   quick inspection.  Non-fatal; analysis is best-effort.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from nvflow.utils import setup_logger

logger = setup_logger(__name__)


def _sync_up(merged_file: Path) -> Path | None:
    """Copy ``<dir>/rollout/output-rsN.jsonl`` up to ``<dir>/output-rsN.jsonl``.

    Returns the destination path, or None if it would copy onto itself.
    """
    target = merged_file.parent.parent / merged_file.name
    try:
        if merged_file.resolve() == target.resolve():
            return None
    except OSError:
        pass
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(merged_file, target)
    logger.info("Synced merged rollout up: %s -> %s", merged_file, target)
    return target


def _stats(merged_file: Path, analysis_dir: Path) -> dict:
    num_rows = 0
    rewards: list[float] = []
    with open(merged_file) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            num_rows += 1
            reward = row.get("reward")
            if isinstance(reward, int | float) and not isinstance(reward, bool):
                rewards.append(float(reward))
    stats = {
        "num_rows": num_rows,
        "num_with_reward": len(rewards),
        "mean_reward": (sum(rewards) / len(rewards)) if rewards else None,
    }
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "sdg_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def analyze(merged_file: str, analysis_dir: str) -> None:
    merged = Path(merged_file)
    if not merged.exists():
        logger.warning("Merged file does not exist, skipping analyze: %s", merged_file)
        return
    _sync_up(merged)
    stats = _stats(merged, Path(analysis_dir))
    logger.info("SDG rollout stats for %s: %s", merged.name, stats)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("merged_file", help="Merged (enriched) rollout JSONL")
    parser.add_argument("analysis_dir", help="Directory for analysis artifacts")
    args = parser.parse_args(argv)
    analyze(args.merged_file, args.analysis_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
