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
"""Tests for canonical native NeMo Gym artifact paths."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import PurePosixPath

import pytest

from nvflow.integrations.nemo_gym.native_contract import (
    NATIVE_GYM_SOURCE_ROOT,
    NativeGymArtifactPaths,
    NativeGymOutputLayout,
    agent_metrics_path_for,
    aggregate_metrics_path_for,
    artifact_paths_for,
    failures_path_for,
    materialized_inputs_path_for,
    quality_summary_path_for,
    repeat_level_metrics_path_for,
    reward_profiling_path_for,
    rollout_attempt_limit,
    rollout_verdicts_path_for,
)


def test_native_runtime_contract_contains_only_supported_image_paths() -> None:
    assert NATIVE_GYM_SOURCE_ROOT == "/opt/nemo-gym"


def test_rollout_attempt_limit_preserves_gym_default_and_covers_job_chain() -> None:
    assert [rollout_attempt_limit(count) for count in range(5)] == [3, 3, 3, 4, 5]


def test_artifact_helpers_match_native_gym_naming_and_run_scope() -> None:
    output = PurePosixPath("/runs/nightly.v1/rollouts.jsonl")

    assert materialized_inputs_path_for(output) == (
        PurePosixPath("/runs/nightly.v1/rollouts_materialized_inputs.jsonl")
    )
    assert failures_path_for(output) == PurePosixPath("/runs/nightly.v1/rollouts_failures.jsonl")
    assert aggregate_metrics_path_for(output) == PurePosixPath(
        "/runs/nightly.v1/rollouts_aggregate_metrics.json"
    )
    assert reward_profiling_path_for(output) == PurePosixPath(
        "/runs/nightly.v1/rollouts_reward_profiling.jsonl"
    )
    assert agent_metrics_path_for(output) == PurePosixPath(
        "/runs/nightly.v1/rollouts_agent_metrics.json"
    )
    assert repeat_level_metrics_path_for(output) == PurePosixPath(
        "/runs/nightly.v1/rollouts_repeat_level_metrics.json"
    )
    assert quality_summary_path_for(output) == PurePosixPath(
        "/runs/nightly.v1/quality_summary.json"
    )
    assert rollout_verdicts_path_for(output) == PurePosixPath(
        "/runs/nightly.v1/rollout_verdicts.jsonl"
    )


def test_artifact_set_is_complete_and_immutable() -> None:
    artifacts = artifact_paths_for("rollouts.jsonl", "completion.json")

    assert artifacts == NativeGymArtifactPaths(
        rollouts=PurePosixPath("rollouts.jsonl"),
        materialized_inputs=PurePosixPath("rollouts_materialized_inputs.jsonl"),
        failures=PurePosixPath("rollouts_failures.jsonl"),
        aggregate_metrics=PurePosixPath("rollouts_aggregate_metrics.json"),
        reward_profiling=PurePosixPath("rollouts_reward_profiling.jsonl"),
        agent_metrics=PurePosixPath("rollouts_agent_metrics.json"),
        repeat_level_metrics=PurePosixPath("rollouts_repeat_level_metrics.json"),
        quality_summary=PurePosixPath("quality_summary.json"),
        rollout_verdicts=PurePosixPath("rollout_verdicts.jsonl"),
        completion=PurePosixPath("completion.json"),
    )
    assert artifacts.as_json_dict() == {
        "rollouts": "rollouts.jsonl",
        "materialized_inputs": "rollouts_materialized_inputs.jsonl",
        "failures": "rollouts_failures.jsonl",
        "aggregate_metrics": "rollouts_aggregate_metrics.json",
        "reward_profiling": "rollouts_reward_profiling.jsonl",
        "agent_metrics": "rollouts_agent_metrics.json",
        "repeat_level_metrics": "rollouts_repeat_level_metrics.json",
        "quality_summary": "quality_summary.json",
        "rollout_verdicts": "rollout_verdicts.jsonl",
        "completion": "completion.json",
    }

    with pytest.raises(FrozenInstanceError):
        artifacts.completion = PurePosixPath("other.json")  # type: ignore[misc]


def test_output_layout_groups_artifacts_by_logical_job() -> None:
    layout = NativeGymOutputLayout.collection("/runs/cell")

    assert layout.chunk_artifacts(2).rollouts == PurePosixPath(
        "/runs/cell/jobs/job-0-collect-rollouts/chunks/chunk-2/outputs/rollouts.jsonl"
    )
    assert layout.attempt_logs_dir(2, 1) == PurePosixPath(
        "/runs/cell/jobs/job-0-collect-rollouts/chunks/chunk-2/attempts/attempt-1/logs"
    )
    assert layout.analyzed_artifacts.rollouts == PurePosixPath(
        "/runs/cell/jobs/job-1-analyze-rollouts/outputs/rollouts.jsonl"
    )
    assert layout.dataset_completion == PurePosixPath(
        "/runs/cell/jobs/job-2-build-training-dataset/completion.json"
    )
    assert layout.profiled_dataset == PurePosixPath("/runs/cell/profiled.jsonl")
    assert layout.training_dataset == PurePosixPath("/runs/cell/train.jsonl")
