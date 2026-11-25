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
"""Shared configuration normalization for concrete NeMo Gym stages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nvflow.integrations.nemo_gym.native_contract import NativeGymOutputLayout
from nvflow.integrations.nemo_gym.reward_profiles import parse_dataset_selection_spec


def dataset_selection_paths(
    layout: NativeGymOutputLayout,
    selection: Mapping[str, Any],
) -> tuple[str, str, str, str, str]:
    """Return public datasets, completion, report, and diagnostics paths."""
    parse_dataset_selection_spec(selection)
    return (
        layout.profiled_dataset.as_posix(),
        layout.training_dataset.as_posix(),
        layout.dataset_completion.as_posix(),
        layout.dataset_selection_report.as_posix(),
        layout.unprofiled_samples.as_posix(),
    )


__all__ = [
    "dataset_selection_paths",
]
