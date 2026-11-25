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
"""Shared training stage types: catalog entries and the project presets contract."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from nvflow.stages.nemo_rl.grpo import GRPOStage
from nvflow.stages.nemo_rl.sft import SFTStage

PROJECT_ROOT = Path(__file__).parents[1]
TRAINING_STAGE_TYPES = {
    "data.train_validation_split": "TrainValidationSplitStage",
    "nemo_gym.train_validation_split": "TrainValidationSplitStage",
    "nemo_rl.grpo": "GRPOStage",
    "nemo_rl.sft": "SFTStage",
}
SFT_CONFIG = {"training_data": "/data/train.jsonl", "output_dir": "/out", "model_name": "model"}
GRPO_CONFIG = {
    "output_dir": "/out",
    "model_name": "model",
    "data_source_dir": "/data",
    "environments": {"example": {}},
}


def test_training_stage_types_are_listed_and_resolve_lazily() -> None:
    script = f"""
import sys
from nvflow.stages.catalog import get_stage_type, list_stage_types, resolve_stage_type

expected = {TRAINING_STAGE_TYPES!r}
assert set(expected) <= {{spec.name for spec in list_stage_types()}}
imported = [get_stage_type(name).module for name in expected]
assert not [module for module in imported if module in sys.modules], imported
for name, class_name in expected.items():
    assert resolve_stage_type(name).__name__ == class_name
"""
    subprocess.run([sys.executable, "-c", script], cwd=PROJECT_ROOT, check=True)


@pytest.mark.parametrize(("stage", "config"), [(SFTStage, SFT_CONFIG), (GRPOStage, GRPO_CONFIG)])
def test_presets_come_only_from_the_project_presets_file(
    tmp_path: Path,
    stage: type[SFTStage] | type[GRPOStage],
    config: dict[str, Any],
) -> None:
    presets_file = tmp_path / "presets.yaml"
    presets_file.write_text("presets:\n  base:\n    policy: {}\n", encoding="utf-8")

    for incomplete in ({"preset": "base"}, {"presets_file": str(presets_file)}):
        with pytest.raises(ValueError, match="'preset' and 'presets_file'"):
            stage().validate_config({**config, **incomplete})
    with pytest.raises(ValueError, match="'presets_file' does not exist"):
        stage().validate_config(
            {**config, "preset": "base", "presets_file": str(tmp_path / "missing.yaml")}
        )
    with pytest.raises(ValueError, match="Unknown preset 'other'"):
        stage().validate_config({**config, "preset": "other", "presets_file": str(presets_file)})
    stage().validate_config({**config, "preset": "base", "presets_file": str(presets_file)})


def test_platform_ships_no_training_presets() -> None:
    assert not list((PROJECT_ROOT / "nvflow").rglob("*preset*.yaml"))
