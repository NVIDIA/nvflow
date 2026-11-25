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
"""Curated NeMo Gym stages selected through workflow ``uses`` bindings."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .build_training_dataset import BuildTrainingDatasetStage
    from .collate_dataset import CollateDatasetStage
    from .collect_rollouts import CollectRolloutsStage
    from .compare_rollouts import CompareRolloutsStage
    from .prepare_benchmark import PrepareBenchmarkStage
    from .rejudge_rollouts import RejudgeRolloutsStage

_EXPORTS = {
    "CollateDatasetStage": ".collate_dataset",
    "CollectRolloutsStage": ".collect_rollouts",
    "CompareRolloutsStage": ".compare_rollouts",
    "PrepareBenchmarkStage": ".prepare_benchmark",
    "RejudgeRolloutsStage": ".rejudge_rollouts",
    "BuildTrainingDatasetStage": ".build_training_dataset",
}


def __getattr__(name: str) -> Any:
    """Lazily expose stage classes without importing every provider operation."""
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "CollateDatasetStage",
    "CollectRolloutsStage",
    "CompareRolloutsStage",
    "PrepareBenchmarkStage",
    "RejudgeRolloutsStage",
    "BuildTrainingDatasetStage",
]
