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
"""Split Gym's input rows by reward (a ``nemo_gym.collect`` report extension).

Gym's analyze job runs it after merging the rollouts. A rollout succeeds when
its reward is 1.0, and a row passes when at least ``pass_rate`` of its rollouts
succeed (the extension's ``options``; default 1.0, every rollout). A rollout
without a finite reward, or whose response didn't complete, has no reply, or is
masked, counts as 0 and makes the row unscored. Rows are
written without the fields that rendering and Gym added, plus ``_reward``, the
mean reward, ``_success_rate``, and ``_replies``, the model's replies in rollout
order. It fails unless each row has one rollout per materialized row, so a
partial collection is never read as verdicts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from nvflow.integrations.nemo_gym.responses import assistant_text

ADDED_FIELDS = {"responses_create_params", "verifier", "prompt_config_used", "agent_ref"}
TASK_INDEX = "_ng_task_index"
ROLLOUT_INDEX = "_ng_rollout_index"


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _task(row: dict[str, Any], path: Path) -> str:
    if TASK_INDEX not in row:
        raise SystemExit(f"split_by_reward: a row of {path} has no {TASK_INDEX}")
    return json.dumps(row[TASK_INDEX], sort_keys=True)


def _reward(rollout: dict[str, Any]) -> float | None:
    value = rollout.get("reward")
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    response = rollout.get("response")
    if rollout.get("mask_sample") or (
        isinstance(response, dict)
        and (
            response.get("status", "completed") != "completed"
            or not assistant_text(response).strip()
        )
    ):
        return None
    return float(value)


def _rollout_order(rollout: dict[str, Any]) -> tuple[int, str]:
    index = rollout.get(ROLLOUT_INDEX)
    if isinstance(index, int) and not isinstance(index, bool):
        return index, ""
    return sys.maxsize, json.dumps(index, sort_keys=True)


def split_by_reward(
    rollouts: Path,
    materialized_inputs: Path,
    output_dir: Path,
    *,
    pass_rate: float = 1.0,
) -> dict[str, Any]:
    """Write ``passed.jsonl``, ``failed.jsonl``, and ``reward_split.json``."""
    outcomes: dict[str, list[tuple[tuple[int, str], float | None, str]]] = defaultdict(list)
    for rollout in _rows(rollouts):
        response = rollout.get("response")
        reply = assistant_text(response) if isinstance(response, dict) else ""
        outcomes[_task(rollout, rollouts)].append(
            (_rollout_order(rollout), _reward(rollout), reply)
        )
    expected: Counter[str] = Counter()
    records: dict[str, dict[str, Any]] = {}
    for row in _rows(materialized_inputs):
        task = _task(row, materialized_inputs)
        expected[task] += 1
        records.setdefault(
            task,
            {
                key: value
                for key, value in row.items()
                if key not in ADDED_FIELDS and not key.startswith("_ng_")
            },
        )
    mismatched = sum(len(outcomes.get(task, ())) != count for task, count in expected.items())
    unmatched = len(outcomes.keys() - expected.keys())
    if mismatched or unmatched:
        raise SystemExit(
            f"split_by_reward: {mismatched} row(s) lack one rollout per materialized row and "
            f"{unmatched} rollout task(s) match no row; rerun the collection"
        )
    passed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    unscored = 0
    for task, record in records.items():
        results = sorted(outcomes.pop(task), key=lambda outcome: outcome[0])
        scores = [score for _order, score, _reply in results]
        rewards = [0.0 if score is None else score for score in scores]
        unscored += None in scores
        record["_reward"] = sum(rewards) / len(rewards)
        record["_success_rate"] = sum(reward >= 1.0 for reward in rewards) / len(rewards)
        record["_replies"] = [reply for _order, _score, reply in results]
        (passed if record["_success_rate"] >= pass_rate else failed).append(record)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "passed.jsonl", passed)
    _write(output_dir / "failed.jsonl", failed)
    report = {
        "rows": len(records),
        "passed": len(passed),
        "failed": len(failed),
        "unscored": unscored,
        "pass_rate": pass_rate,
    }
    _write(output_dir / "reward_split.json", [report])
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("rollouts_file", type=Path)
    parser.add_argument("materialized_inputs_file", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-repeats", type=int, help="unused: counted from the rows")
    parser.add_argument("--options", default="{}", help='JSON options: {"pass_rate": 0.7}')
    args = parser.parse_args(argv)
    options = json.loads(args.options)
    if not isinstance(options, dict) or set(options) - {"pass_rate"}:
        parser.error("--options takes only pass_rate")
    pass_rate = options.get("pass_rate", 1.0)
    if (
        isinstance(pass_rate, bool)
        or not isinstance(pass_rate, int | float)
        or not 0 < pass_rate <= 1
    ):
        parser.error("pass_rate must be in (0, 1]")
    report = split_by_reward(
        args.rollouts_file,
        args.materialized_inputs_file,
        args.output_dir,
        pass_rate=float(pass_rate),
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
