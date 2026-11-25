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
"""Dependency experiment naming for environment fan-out stages."""

from nvflow.core.dependencies import resolve_dependency_names


def _expname(stage: str, _config: dict) -> str:
    return f"workflow-{stage}"


def test_combined_environment_dependency_matches_training_experiment() -> None:
    stage_configs = {
        "training": {
            "environment_execution": "combined",
            "environment_execution_filter": "raw_train_data",
            "environments": {
                "equivalence": {"enabled": True, "raw_train_data": "/data/equivalence"},
                "disabled": {"enabled": False},
                "unready": {"enabled": True, "raw_train_data": None},
                "finance": {"enabled": True, "raw_train_data": "/data/finance"},
            },
        }
    }

    assert resolve_dependency_names(
        ["training"],
        stage_configs,
        _expname,
    ) == ["workflow-training-equivalence+finance"]


def test_environment_subset_preserves_requested_combined_order() -> None:
    stage_configs = {
        "training": {
            "environment_execution": "combined",
            "environments": {
                "equivalence": {"enabled": True},
                "finance": {"enabled": True},
            },
        }
    }

    assert resolve_dependency_names(
        ["training"],
        stage_configs,
        _expname,
        environment=["finance", "equivalence"],
    ) == ["workflow-training-finance+equivalence"]
