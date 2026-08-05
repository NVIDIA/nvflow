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
"""Consolidate multiple WandB training runs into one continuous dashboard.

Each GRPO/SFT job (including each job of a resume chain) creates a separate WandB
run. This merges them post-hoc into a single run with continuous step numbers.

Source selection:
  --run-ids ID...   explicit runs (precise; best when many experiments share a group)
  --group NAME      all runs in a group, optionally narrowed by --since TIMESTAMP

Failure handling (automatic): runs are merged in chronological order and each is
truncated at its successor's resume step, so a failed/rolled-back job's stale tail
is dropped and the resumed job's data wins. Use --max-step to cap a trailing failed
run (the last run has no successor to bound it).

Modes:
  fresh (default)   create a new consolidated run
  --append RUN_ID   extend an existing consolidated run with new steps only

Usage:
    # fresh, from explicit run-ids (--entity required if the project is owned by
    # another team, e.g. nvidia; or set $WANDB_ENTITY)
    uv run python scripts/wandb_consolidate.py \\
        --project finance-grpo --entity nvidia \\
        --run-ids abc123 def456 ghi789 --name nano-consolidated

    # or by group + time window
    uv run python scripts/wandb_consolidate.py \\
        --project finance-grpo --group grpo-training-finance_sec_search \\
        --since 2026-06-28T19:00:00Z --name nano-consolidated

    # append later as the chain grows
    uv run python scripts/wandb_consolidate.py \\
        --project finance-grpo --run-ids abc123 def456 ghi789 jkl012 \\
        --append <consolidated_run_id>

    # preview only
    ... --dry-run

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
        default=None,
        help="WandB group to filter source runs. Required unless --run-ids is given.",
    )
    parser.add_argument(
        "--run-ids",
        nargs="*",
        default=None,
        help="Explicit source run IDs to consolidate (precise selection; ignores "
        "--group/--since). Preferred when multiple experiments share one group.",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Only include --group runs created at/after this ISO8601 UTC timestamp "
        "(e.g. 2026-06-28T19:00:00Z).",
    )
    parser.add_argument(
        "--name",
        default="consolidated",
        help="WandB run name for the consolidated dashboard (default: consolidated).",
    )
    parser.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY"),
        help=(
            "WandB entity (team/user). Defaults to $WANDB_ENTITY, else your "
            "'wandb login' default. Required for --run-ids when the project is "
            "owned by another entity (e.g. --entity nvidia)."
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
        "--skip-histograms",
        action="store_true",
        help="Do not carry over wandb.Histogram metrics (train/*/histogram, "
        "validation/*/histogram). By default histograms ARE carried; only "
        "artifact-backed tables/plots (e.g. full_result, *_plot_sample) are skipped.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Cap consolidation at this step (inclusive). Use to exclude a trailing "
        "failed job's rolled-back steps beyond its last good checkpoint (the last "
        "run has no successor to bound it automatically).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for WandB local run data (default: "
            "<repo>/outputs/wandb_consolidated/). Keeps WandB scratch files out "
            "of the source tree."
        ),
    )
    return parser.parse_args()


def fetch_source_runs(
    api,
    project: str,
    entity: str | None,
    group: str | None,
    exclude_id: str | None = None,
    run_ids: list[str] | None = None,
    since: str | None = None,
):
    """Fetch source runs to consolidate.

    Selection precedence:
      - run_ids: fetch exactly those runs (group/since ignored). Use when several
        experiments share one group.
      - group:   fetch the group, optionally filtered by `since` (created_at >= since).
    Excludes the --append target and any prior consolidated runs (CONSOLIDATED_TAG).

    Runs are returned in CHRONOLOGICAL (created_at) order, not by max step, so that
    on overlapping steps from a failed-then-resumed job the later run's data wins.
    """
    path = f"{entity}/{project}" if entity else project

    if run_ids:
        candidates = []
        for rid in run_ids:
            try:
                candidates.append(api.run(f"{path}/{rid}"))
            except Exception as e:  # noqa: BLE001
                print(
                    f"  WARN: could not fetch run id {rid}: {e}\n"
                    f"        (looked under entity='{entity}'; pass --entity or set "
                    f"$WANDB_ENTITY if the project is owned by another entity, e.g. nvidia)"
                )
    elif group:
        candidates = list(api.runs(path, filters={"group": group}))
    else:
        print("Provide either --group or --run-ids.")
        sys.exit(1)

    run_list = []
    for r in candidates:
        if exclude_id and r.id == exclude_id:
            continue
        if r.tags and CONSOLIDATED_TAG in r.tags:
            continue
        if since and not run_ids and (getattr(r, "created_at", "") or "") < since:
            continue
        run_list.append(r)

    if not run_list:
        print(
            f"No source runs matched (project={project}, group={group}, "
            f"run_ids={run_ids}, since={since})."
        )
        sys.exit(1)

    # Chronological: later (resume) runs supersede earlier failed ones on overlapping steps.
    run_list.sort(key=lambda r: getattr(r, "created_at", "") or "")
    return run_list


def _to_wandb_histogram(wandb, raw: dict):
    """Reconstruct a wandb.Histogram from a scan_history histogram dict.

    scan_history returns histograms as {"values": [counts...],
    "packedBins": {"min": m, "size": s, "count": n}, "_type": "histogram"}.
    wandb.Histogram(np_histogram=(counts, bin_edges)) needs len(bin_edges) ==
    len(counts) + 1, so we rebuild edges from the packed (min, size, count).
    Returns None if the dict can't be reconstructed.
    """
    try:
        values = list(raw["values"])
        pb = raw["packedBins"]
        mn, size, count = float(pb["min"]), float(pb["size"]), int(pb["count"])
        edges = [mn + i * size for i in range(count + 1)]
        if len(edges) != len(values) + 1:
            return None
        return wandb.Histogram(np_histogram=(values, edges))
    except Exception:  # noqa: BLE001
        return None


def get_max_consolidated_step(api, project: str, entity: str | None, run_id: str) -> int:
    """Get the highest step already logged in the consolidated run."""
    path = f"{entity}/{project}" if entity else project
    try:
        run = api.run(f"{path}/{run_id}")
        max_step = run.summary.get("_step", -1)
        return int(max_step)
    except Exception:
        return -1


def collect_metrics(
    runs,
    skip_prefixes: list[str],
    min_step: int = -1,
    max_step: int | None = None,
    include_histograms: bool = True,
):
    """Collect scalar (and optionally histogram) metrics from source runs.

    Runs are assumed in chronological order.

    Scalars are kept as-is. WandB histograms come back from scan_history as dicts
    ({"_type": "histogram", "values": [...], "packedBins": {min,size,count}}); we
    keep that raw dict and reconstruct a wandb.Histogram at log time. Artifact-backed
    non-scalars (table-file sample dumps, *_plot_sample images) are always skipped.

    Resume-lineage truncation: each run is valid only up to where the *next* run
    resumed (its successor's min step). Steps a job logged beyond its last
    carried-forward checkpoint were rolled back, so they are dropped. This handles
    failed jobs (empty valid range -> auto-dropped) and partial-success jobs
    (keep the checkpointed prefix, drop the rolled-back tail) using only step data.
    """
    # Pass 1: read each run's rows + its min step (runs are already chronological).
    per_run: list[tuple[int | None, list[tuple[int, dict[str, object]]], object]] = []
    for run in runs:
        print(f"  Reading run: {run.name} ({run.id}), state={run.state}")
        rows: list[tuple[int, dict[str, object]]] = []
        for row in run.scan_history():
            step = row.get("_step")
            if step is None:
                continue
            step = int(step)
            metrics: dict[str, object] = {}
            for k, v in row.items():
                if k.startswith("_") or any(k.startswith(p) for p in skip_prefixes):
                    continue
                if isinstance(v, int | float):
                    metrics[k] = v
                elif (
                    include_histograms
                    and isinstance(v, dict)
                    and v.get("_type") == "histogram"
                    and v.get("values") is not None
                    and isinstance(v.get("packedBins"), dict)
                ):
                    # Keep raw dict; reconstructed into wandb.Histogram at log time.
                    metrics[k] = v
            if metrics:
                rows.append((step, metrics))
        # Resume boundary = first *training* step (>0). step 0 is the per-job
        # val_at_start artifact (re-logged every job), NOT the resume point, so it
        # must be excluded or every run's boundary collapses to 0.
        resume_step = min((s for s, _ in rows if s > 0), default=None)
        per_run.append((resume_step, rows, run))

    # Pass 2: bound each run at its successor's resume point + drop already-consolidated.
    all_rows: list[tuple[int, dict[str, object]]] = []
    for i, (_, rows, run) in enumerate(per_run):
        upper = None  # exclusive upper bound = next run's min step
        for j in range(i + 1, len(per_run)):
            if per_run[j][0] is not None:
                upper = per_run[j][0]
                break
        kept = truncated = skipped = capped = 0
        for step, metrics in rows:
            if step <= min_step:
                skipped += 1
                continue
            if max_step is not None and step > max_step:
                capped += 1
                continue
            if upper is not None and step >= upper:
                truncated += 1
                continue
            all_rows.append((step, metrics))
            kept += 1
        msg = f"    {run.name}: kept {kept} steps"
        if truncated:
            msg += f", dropped {truncated} rolled-back (>= successor resume @ {upper})"
        if capped:
            msg += f", dropped {capped} above --max-step {max_step}"
        if skipped:
            msg += f", skipped {skipped} already-consolidated"
        print(msg)

    # Stable sort by step preserves chronological order on ties (later run wins).
    all_rows.sort(key=lambda x: x[0])
    return all_rows


def print_summary(rows: list[tuple[int, dict]], runs, min_step: int):
    """Print summary of data to consolidate."""
    if not rows:
        print("No new data to consolidate.")
        return
    steps = [r[0] for r in rows]
    all_keys: set[str] = set()
    hist_keys: set[str] = set()
    for _, metrics in rows:
        all_keys.update(metrics.keys())
        for k, v in metrics.items():
            if isinstance(v, dict) and v.get("_type") == "histogram":
                hist_keys.add(k)
    print("\nConsolidation summary:")
    print(f"  Source runs: {len(runs)}")
    if min_step >= 0:
        print(f"  Already consolidated up to step: {min_step}")
    print(f"  New steps: {len(steps)} (min={min(steps)}, max={max(steps)})")
    print(
        f"  Unique metrics: {len(all_keys)} (scalars: {len(all_keys) - len(hist_keys)}, "
        f"histograms: {len(hist_keys)})"
    )
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

    print(
        f"Fetching source runs from project={args.project} "
        f"(group={args.group}, run_ids={args.run_ids}, since={args.since})"
    )
    runs = fetch_source_runs(
        api,
        args.project,
        args.entity,
        args.group,
        exclude_id=exclude_id,
        run_ids=args.run_ids,
        since=args.since,
    )
    print(f"Found {len(runs)} source runs\n")

    print("Collecting metrics...")
    rows = collect_metrics(
        runs,
        args.skip_prefixes or [],
        min_step=min_step,
        max_step=args.max_step,
        include_histograms=not args.skip_histograms,
    )
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
        init_kwargs["group"] = args.group or args.name
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
    hist_logged = 0
    for step, metrics in rows:
        log_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, dict) and v.get("_type") == "histogram":
                hv = _to_wandb_histogram(wandb, v)
                if hv is not None:
                    log_metrics[k] = hv
                    hist_logged += 1
            else:
                log_metrics[k] = v
        run.log(log_metrics, step=step)
        logged += len(log_metrics)

    run.finish()
    print(
        f"\nDone. Logged {logged} data points across {len(rows)} steps "
        f"({hist_logged} histogram points)."
    )
    print(f"Run ID: {run.id} (use with --append for future updates)")


def main():
    args = parse_args()
    consolidate(args)


if __name__ == "__main__":
    main()
