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
"""GRPO training must run from NeMo-RL's project dir, so that its ``uv`` calls do
not pick up this project's ``[tool.uv] required-version``, while keeping the
packaged code importable via ``PYTHONPATH``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, cast

import pytest

from nvflow.execution import PlanningContext, TargetBinding

PACKAGED_CODE_DIR = "/nemo_run/code"
NEMO_RL_DIR = "/opt/nemo-rl"
TRAIN_ENTRY_POINT = "run_grpo_nemo_gym.py"


@pytest.fixture
def train_cmd(monkeypatch):
    """The resolved GRPO training command string.

    ``nemo_skills`` is faked so the guard also runs in the lightweight CI lane,
    which omits it.
    """
    for name in ("nemo_skills", "nemo_skills.pipeline", "nemo_skills.pipeline.nemo_rl"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    grpo_mod = types.ModuleType("nemo_skills.pipeline.nemo_rl.grpo")
    grpo_mod.get_timeout_str = lambda *_args, **_kwargs: "00:30:00"
    monkeypatch.setitem(sys.modules, "nemo_skills.pipeline.nemo_rl.grpo", grpo_mod)

    from nvflow.stages.nemo_rl.grpo import GRPOStage, PreparedGRPOConfig

    prepared = PreparedGRPOConfig(
        nemo_rl_config={},
        run_name="grpo-test",
        output_dir="/workspace/outputs/test",
        expname="grpo-test",
        hf_model_name="Qwen/Qwen3-4B",
        num_nodes=1,
        num_gpus=8,
        backend="megatron",
        gym_binding_commands=("resolve-gym-binding",),
    )
    return GRPOStage()._build_train_cmd(
        prepared,
        {
            "model_name": "Qwen/Qwen3-4B",
            "input_contract": "train_validation_split",
            "training_data": "/workspace/split/finance/train.jsonl",
            "environments": {"finance": {}},
        },
        config_snippet="true",
        config_path=f"{prepared.output_dir}/grpo_config.yaml",
    )


def test_training_runs_from_the_nemo_rl_project_dir(train_cmd):
    reason = f"training must cd to {NEMO_RL_DIR} before invoking {TRAIN_ENTRY_POINT}"
    change_dir = f"cd {NEMO_RL_DIR} "
    assert change_dir in train_cmd, reason
    assert train_cmd.index(change_dir) < train_cmd.index(TRAIN_ENTRY_POINT), reason


def test_training_never_runs_from_the_packaged_code_dir(train_cmd):
    assert f"cd {PACKAGED_CODE_DIR}" not in train_cmd


def test_packaged_code_stays_importable(train_cmd):
    assert f'NVFLOW_CODE_ROOT="${{NVFLOW_CODE_ROOT:-{PACKAGED_CODE_DIR}}}"' in train_cmd
    assert "PYTHONPATH=$PYTHONPATH:$NVFLOW_CODE_ROOT" in train_cmd


def test_gym_binding_is_resolved_before_training(train_cmd):
    from nvflow.execution.runtime import NRL_GYM_BIN, NRL_GYM_ROOT

    # Training runs in the nemo-rl image; /opt/nemo-gym exists only in the nemo-gym image.
    assert f"(export PATH={NRL_GYM_BIN}:$PATH && cd {NRL_GYM_ROOT} && " in train_cmd
    assert "/opt/nemo-gym" not in train_cmd and "/opt/nemo_gym_venv" not in train_cmd
    assert train_cmd.index("resolve-gym-binding") < train_cmd.index(TRAIN_ENTRY_POINT)


def test_training_command_is_valid_bash(train_cmd):
    import subprocess

    parsed = subprocess.run(["bash", "-n", "-c", train_cmd], capture_output=True, text=True)
    assert parsed.returncode == 0, parsed.stderr


def test_gym_binding_resolver_imports_without_nvflow_dependencies() -> None:
    import subprocess

    # The resolver runs inside Gym's venv, which has none of NVFlow's dependencies.
    probe = (
        "import sys, nvflow.integrations.nemo_gym.runtime_binding; "
        "sys.exit(any(name.startswith(('nvflow.core', 'nvflow.integrations.nemo_skills')) "
        "for name in sys.modules))"
    )
    assert subprocess.run([sys.executable, "-c", probe], check=False).returncode == 0


def test_split_marker_is_validated_before_training(train_cmd):
    validation = "nvflow.stages.data.split_contract --output-dir /workspace/split/finance"
    assert validation in train_cmd
    assert train_cmd.index(validation) < train_cmd.index(TRAIN_ENTRY_POINT)


def test_stage_only_training_requires_split_marker() -> None:
    from nvflow.stages.nemo_rl.grpo import GRPOStage

    class MissingFilesystem:
        @staticmethod
        def batch_exists(paths):
            return dict.fromkeys(paths, False)

    config = {
        "input_contract": "train_validation_split",
        "data_source_dir": "/workspace/split",
        "output_dir": "/workspace/output",
        "environments": {"finance": {"raw_train_data": "/unused"}},
    }

    with pytest.raises(ValueError, match="split.done"):
        GRPOStage().plan(
            PlanningContext(
                stage="training",
                config=config,
                launcher_config=config,
                target=TargetBinding.resolve("test"),
                expname="training",
                run_after=(),
                live=True,
                filesystem=cast(Any, MissingFilesystem()),
            )
        )


@pytest.mark.parametrize("deployment", ["gym", "nvflow"])
def test_managed_training_judge_requires_explicit_container_config(
    deployment: str,
    tmp_path: Path,
) -> None:
    from nvflow.stages.nemo_rl.grpo import GRPOStage

    presets_file = tmp_path / "grpo_presets.yaml"
    presets_file.write_text("presets:\n  grpo-base: {}\n", encoding="utf-8")
    config = {
        "output_dir": "/workspace/outputs/train",
        "model_name": "model",
        "data_source_dir": "/workspace/data",
        "presets_file": str(presets_file),
        "preset": "grpo-base",
        "environments": {
            "finance": {
                "models": {
                    "judge": {
                        "deployment": deployment,
                    }
                }
            }
        },
    }

    with pytest.raises(ValueError, match="'vllm_container'"):
        GRPOStage().validate_config(config)

    config["vllm_container"] = "vllm"
    GRPOStage().validate_config(config)
