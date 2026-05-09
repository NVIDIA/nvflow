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
"""Convert nvflow filtered train.jsonl + rollout outputs to the profiled format.

Reads:
  - train.jsonl (filtered, with difficulty_profile from aggregate_seeds)
  - output-rs*.jsonl (merged rollout outputs with response.usage token counts)

Produces per-task records with:
  - Original input fields (responses_create_params, expected_answer, template_metadata, agent_ref)
  - pass_rate, pass_rate_passed, pass_rate_total
  - Token count metrics: input_tokens/{mean,std,max,min}, output_tokens/..., total_tokens/...

Usage:
  python3 scripts/convert_to_profiled.py \
      --train-path <filter_7seeds/train.jsonl> \
      --rollout-dir <rollout/> \
      --output-path <profiled_output.jsonl>
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Convert to profiled format with token counts")
    parser.add_argument("--train-path", required=True, help="Filtered train.jsonl")
    parser.add_argument(
        "--rollout-dir", required=True, help="Directory with output-rs*.jsonl files"
    )
    parser.add_argument("--output-path", required=True, help="Output profiled jsonl")
    parser.add_argument("--dry-run", action="store_true", help="Process first 1000 records only")
    args = parser.parse_args()

    rollout_dir = Path(args.rollout_dir)
    rollout_files = sorted(rollout_dir.glob("output-rs*.jsonl"))
    if not rollout_files:
        print(f"ERROR: No output-rs*.jsonl files in {rollout_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(rollout_files)} rollout files: {[f.name for f in rollout_files]}")

    # Step 1: Load train.jsonl (filtered) — these are the tasks we want
    print("Loading train.jsonl...")
    train_by_uuid = {}
    with open(args.train_path) as f:
        for line in f:
            rec = json.loads(line)
            uid = rec.get("uuid", "")
            if uid:
                train_by_uuid[uid] = rec
    print(f"  {len(train_by_uuid)} tasks loaded")

    # Step 2: Stream through rollout files, collect token counts per uuid
    print("Streaming rollout files for token counts...")
    token_stats = defaultdict(lambda: {"input": [], "output": [], "total": [], "rewards": []})
    total_rows = 0

    for rf in rollout_files:
        print(f"  Reading {rf.name}...")
        with open(rf) as f:
            for line in f:
                row = json.loads(line)
                uid = row.get("uuid", "")
                if not uid or uid not in train_by_uuid:
                    continue

                reward = row.get("reward", 0.0)
                token_stats[uid]["rewards"].append(reward)

                usage = row.get("response", {}).get("usage")
                if usage:
                    token_stats[uid]["input"].append(usage.get("input_tokens", 0))
                    token_stats[uid]["output"].append(usage.get("output_tokens", 0))
                    token_stats[uid]["total"].append(usage.get("total_tokens", 0))

                total_rows += 1
                if args.dry_run and total_rows >= 10000:
                    break
        if args.dry_run and total_rows >= 10000:
            break

    print(f"  {total_rows} rollout rows matched, {len(token_stats)} unique tasks")

    # Step 3: Build profiled output
    print("Writing profiled output...")
    written = 0
    skipped_no_tokens = 0

    with open(args.output_path, "w") as out:
        for uid, train_rec in train_by_uuid.items():
            stats = token_stats.get(uid)
            if not stats or not stats["input"]:
                skipped_no_tokens += 1
                continue

            rewards = stats["rewards"]
            pass_rate_passed = sum(r for r in rewards)
            pass_rate_total = len(rewards)
            pass_rate = pass_rate_passed / pass_rate_total if pass_rate_total > 0 else 0.0

            profiled = {
                "responses_create_params": train_rec["responses_create_params"],
                "expected_answer": train_rec.get("expected_answer", ""),
            }
            if "template_metadata" in train_rec:
                profiled["template_metadata"] = train_rec["template_metadata"]
            if "agent_ref" in train_rec:
                profiled["agent_ref"] = train_rec["agent_ref"]
            if "uuid" in train_rec:
                profiled["uuid"] = train_rec["uuid"]
            if "_hash" in train_rec:
                profiled["_hash"] = train_rec["_hash"]
            if "_source" in train_rec:
                profiled["_source"] = train_rec["_source"]

            profiled["pass_rate"] = pass_rate
            profiled["pass_rate_passed"] = pass_rate_passed
            profiled["pass_rate_total"] = pass_rate_total

            inp = np.array(stats["input"])
            outp = np.array(stats["output"])
            tot = np.array(stats["total"])

            profiled["input_tokens/mean"] = float(np.mean(inp))
            profiled["input_tokens/std"] = float(np.std(inp))
            profiled["input_tokens/max"] = int(np.max(inp))
            profiled["input_tokens/min"] = int(np.min(inp))
            profiled["output_tokens/mean"] = float(np.mean(outp))
            profiled["output_tokens/std"] = float(np.std(outp))
            profiled["output_tokens/max"] = int(np.max(outp))
            profiled["output_tokens/min"] = int(np.min(outp))
            profiled["total_tokens/mean"] = float(np.mean(tot))
            profiled["total_tokens/std"] = float(np.std(tot))
            profiled["total_tokens/max"] = int(np.max(tot))
            profiled["total_tokens/min"] = int(np.min(tot))

            out.write(json.dumps(profiled, ensure_ascii=False) + "\n")
            written += 1

    print(f"Done. Written: {written}, skipped (no tokens): {skipped_no_tokens}")
    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
