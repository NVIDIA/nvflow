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
"""The standalone metrics tool reads canonical Gym repeats."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).parents[1] / "projects/finance/scripts/eval_metrics.py"
    spec = importlib.util.spec_from_file_location("finance_eval_metrics", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_load_columns_groups_canonical_rows_by_repeat(tmp_path: Path) -> None:
    module = _load_module()
    run = tmp_path / "model-a/collect-rollouts/secque/jobs/job-1-analyze-rollouts/outputs"
    run.mkdir(parents=True)
    inputs = []
    rollouts = []
    for task in range(2):
        for repeat in range(2):
            identity = {
                "_ng_task_index": task,
                "_ng_rollout_index": repeat,
            }
            inputs.append({**identity, "question": f"question-{task}"})
            rollouts.append(
                {
                    **identity,
                    "reward": float(task == repeat),
                    "response": {"output": [], "usage": {}},
                }
            )
    _write(run / "rollouts_materialized_inputs.jsonl", inputs)
    _write(run / "rollouts.jsonl", rollouts)

    columns = module.load_columns(tmp_path)

    assert list(columns) == ["model-a"]
    assert [len(repeat) for repeat in columns["model-a"]] == [2, 2]
    assert columns["model-a"][1][0]["question"] == "question-0"
