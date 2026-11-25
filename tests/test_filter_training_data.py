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
"""Fail-closed contracts for reward-based training-data filtering."""

import pytest

from nvflow.integrations.nemo_gym.native_contract import NativeGymOutputLayout
from nvflow.integrations.nemo_gym.reward_profiles import (
    DatasetSelectionSpec,
    build_training_dataset,
)


def test_unprofiled_input_never_publishes_passthrough_training_data(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    train.write_text('{"uuid":"one"}\n', encoding="utf-8")
    output = tmp_path / "output"
    layout = NativeGymOutputLayout.collection(output)

    with pytest.raises(FileNotFoundError):
        build_training_dataset(
            train,
            tmp_path / "missing-profile.jsonl",
            layout.profiled_dataset,
            layout.training_dataset,
            layout.dataset_selection_report,
            layout.dataset_completion,
            layout.unprofiled_samples,
            DatasetSelectionSpec(mode="scalar_variance", min_std=1e-6),
        )

    assert not (output / "train.jsonl").exists()
