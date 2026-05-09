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
"""Consolidate multiple WandB training runs into a single dashboard.

Reads metrics from individual WandB runs (filtered by group) and writes
them to a single consolidated WandB run with continuous step numbers.

Each GRPO/SFT training job creates a separate WandB dashboard. This script
merges them post-hoc into one continuous view.

Two modes:
  fresh (default): Creates a new consolidated WandB run.
  --append RUN_ID:  Resumes an existing consolidated run and adds new data.
                    Only logs steps beyond what was previously consolidated.

Usage:
    # First time -- new consolidated dashboard from all runs in a group
    uv run python scripts/wandb_consolidate.py \\
        --project finance-grpo \\
        --group grpo-training-finance_sec_search \\
        --name gspo-qwen3-30b-consolidated

    # After more training -- append new data to existing dashboard
    uv run python scripts/wandb_consolidate.py \\
        --project finance-grpo \\
        --group grpo-training-finance_sec_search \\
        --append <run_id>

    # Dry run -- show what would be logged
    uv run python scripts/wandb_consolidate.py \\
        --project finance-grpo \\
        --group grpo-training-finance_sec_search \\
        --dry-run

Requirements:
    pip install wandb
"""

from __future__ import annotations

import argparse
import os
import sys

CONSOLIDATED_TAG = "wandb_consolidate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate multiple WandB runs into a single dashboard."
    )
    parser.add_argument(
        "--project",
        required=True,
        help="WandB project name (e.g., finance-grpo).",
    )
    parser.add_argument(
        "--group",
        required=True,
        help="WandB group name to filter source runs (e.g., grpo-training-finance_sec_search).",
    )
    parser.add_argument(
        "--name",
        default="consolidated",
        help="WandB run name for the consolidated dashboard (default: consolidated).",
    )
    parser.add_argument(
        "--entity",
        default=None,
        help=(
            "WandB entity (team/user). If not set, uses the default from "
            "'wandb login'. Required if your default entity differs from "
            "the project owner."
        ),
    )
    parser.add_argument(
        "--append",
        metavar="RUN_ID",
        default=None,
        help="Resume an existing consolidated WandB run by ID and append new data.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without logging to WandB.",
    )
    parser.add_argument(
        "--skip-prefixes",
        nargs="*",
        default=["ray/"],
        help="Skip metrics whose key starts with these prefixes (default: ray/).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for WandB local run data. Defaults to "
            "{training-logs}/wandb_consolidated/ if not set. "
            "Prevents WandB from writing to the source tree."
        ),
    )
    return parser.parse_args()


def fetch_source_runs(
    api, project: str, entity: str | None, group: str, exclude_id: str | None = None
):
    """Fetch source runs in the given group, excluding the consolidated run."""
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, filters={"group": group})
    if not runs:
        print(f"No runs found in project={project}, group={group}")
        sys.exit(1)

    run_list = []
    for r in runs:
        if exclude_id and r.id == exclude_id:
            continue
        if r.tags and CONSOLIDATED_TAG in r.tags:
            continue
        run_list.append(r)

    run_list.sort(key=lambda r: r.summary.get("_step", 0))
    return run_list


def get_max_consolidated_step(api, project: str, entity: str | None, run_id: str) -> int:
    """Get the highest step already logged in the consolidated run."""
    path = f"{entity}/{project}" if entity else project
    try:
        run = api.run(f"{path}/{run_id}")
        max_step = run.summary.get("_step", -1)
        return int(max_step)
    except Exception:
        return -1


def collect_metrics(runs, skip_prefixes: list[str], min_step: int = -1):
    """Collect scalar metrics from source runs, skipping already-consolidated steps."""
    all_rows: list[tuple[int, dict[str, float]]] = []

    for run in runs:
        print(f"  Reading run: {run.name} ({run.id}), state={run.state}")
        row_count = 0
        skipped = 0
        for row in run.scan_history():
            step = row.get("_step")
            if step is None:
                continue
            if int(step) <= min_step:
                skipped += 1
                continue
            metrics = {}
            for k, v in row.items():
                if k.startswith("_"):
                    continue
                if any(k.startswith(p) for p in skip_prefixes):
                    continue
                if isinstance(v, int | float):
                    metrics[k] = v
            if metrics:
                all_rows.append((int(step), metrics))
                row_count += 1
        msg = f"    {row_count} steps collected"
        if skipped:
            msg += f" ({skipped} already consolidated, skipped)"
        print(msg)

    all_rows.sort(key=lambda x: x[0])
    return all_rows


def print_summary(rows: list[tuple[int, dict]], runs, min_step: int):
    """Print summary of data to consolidate."""
    if not rows:
        print("No new data to consolidate.")
        return
    steps = [r[0] for r in rows]
    all_keys: set[str] = set()
    for _, metrics in rows:
        all_keys.update(metrics.keys())
    print("\nConsolidation summary:")
    print(f"  Source runs: {len(runs)}")
    if min_step >= 0:
        print(f"  Already consolidated up to step: {min_step}")
    print(f"  New steps: {len(steps)} (min={min(steps)}, max={max(steps)})")
    print(f"  Unique metrics: {len(all_keys)}")
    print(f"  Total data points: {sum(len(m) for _, m in rows)}")


def consolidate(args: argparse.Namespace):
    """Main consolidation logic."""
    import wandb

    api = wandb.Api()

    min_step = -1
    exclude_id = None

    if args.append:
        min_step = get_max_consolidated_step(api, args.project, args.entity, args.append)
        exclude_id = args.append
        print(f"Append mode: consolidated run {args.append}, max step = {min_step}")

    print(f"Fetching source runs from project={args.project}, group={args.group}")
    runs = fetch_source_runs(api, args.project, args.entity, args.group, exclude_id=exclude_id)
    print(f"Found {len(runs)} source runs\n")

    print("Collecting metrics...")
    rows = collect_metrics(runs, args.skip_prefixes or [], min_step=min_step)
    print_summary(rows, runs, min_step)

    if not rows:
        print("Nothing to do.")
        return

    if args.dry_run:
        print("\n--dry-run: skipping WandB upload.")
        return

    init_kwargs: dict = {"project": args.project}
    if args.entity:
        init_kwargs["entity"] = args.entity

    if args.append:
        init_kwargs["id"] = args.append
        init_kwargs["resume"] = "must"
        print(f"\nAppending to existing WandB run: {args.append}")
    else:
        init_kwargs["name"] = args.name
        init_kwargs["group"] = args.group
        init_kwargs["tags"] = [CONSOLIDATED_TAG]
        print(f"\nCreating new WandB run: {args.name}")

    wandb_dir = args.output_dir
    if wandb_dir is None:
        wandb_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "wandb_consolidated"
        )
    os.makedirs(wandb_dir, exist_ok=True)
    init_kwargs["dir"] = wandb_dir

    run = wandb.init(**init_kwargs)
    print(f"Run ID: {run.id}")
    print(f"URL: {run.url}")

    logged = 0
    for step, metrics in rows:
        run.log(metrics, step=step)
        logged += len(metrics)

    run.finish()
    print(f"\nDone. Logged {logged} data points across {len(rows)} steps.")
    print(f"Run ID: {run.id} (use with --append for future updates)")


def main():
    args = parse_args()
    consolidate(args)


if __name__ == "__main__":
    main()
