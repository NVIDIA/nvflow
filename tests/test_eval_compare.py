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
"""Native Gym comparison as an optional first-class eval stage."""

from __future__ import annotations

import shlex

import pytest

from nvflow.integrations.nemo_gym.compare import build_compare_command
from nvflow.stages.nemo_gym import CompareRolloutsStage


def _config() -> dict[str, object]:
    return {
        "contract_version": 1,
        "baseline_rollouts": "/runs/baseline/rollouts.jsonl",
        "baseline_completion": "/runs/baseline/completion.json",
        "candidate_rollouts": "/runs/candidate/rollouts.jsonl",
        "candidate_completion": "/runs/candidate/completion.json",
        "output_dir": "/runs/comparison",
        "container": "nemo-gym",
        "gym_path": "/opt/nemo-gym",
        "report_format": "both",
        "agent": "simple_agent",
    }


def test_compare_command_delegates_to_native_gym() -> None:
    command = build_compare_command(_config())
    arguments = shlex.split(command.splitlines()[-1])

    assert command.count("native_sharding validate") == 2
    assert command.index("native_sharding validate") < command.index("gym eval compare")
    assert "/runs/baseline/completion.json" in command
    assert "/runs/candidate/rollouts_materialized_inputs.jsonl" in command
    assert arguments == [
        "gym",
        "eval",
        "compare",
        "--baseline",
        "/runs/baseline/rollouts.jsonl",
        "--candidates",
        "/runs/candidate/rollouts.jsonl",
        "--output-dir",
        "/runs/comparison",
        "--report-format",
        "both",
        "--agent",
        "simple_agent",
    ]


def test_compare_validation_rejects_ambiguous_inputs() -> None:
    stage = CompareRolloutsStage()
    stage.validate_config(_config())

    same = _config()
    same["candidate_rollouts"] = same["baseline_rollouts"]
    with pytest.raises(ValueError, match="different baseline"):
        stage.validate_config(same)

    invalid_format = _config()
    invalid_format["report_format"] = "csv"
    with pytest.raises(ValueError, match="report_format"):
        stage.validate_config(invalid_format)
