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
"""Qualified worker-interpreter resolution.

NVFlow containers must ship with all required pre-built environments. Missing
environments fail closed instead of resolving dependencies at runtime.

All resolution happens **at bash runtime on the cluster node**, not at
Python submission time, because the venv paths exist inside the container.
"""

from __future__ import annotations

import base64

from nvflow.core.code_root import CODE_ROOT_TOKEN, render_code_root

__all__ = [
    "CHECKPOINT_SAVE_BY_ENV",
    "CHECKPOINT_SAVE_BY_OVERRIDE",
    "CODE_ROOT_TOKEN",
    "NRL_GYM_BIN",
    "NRL_GYM_ROOT",
    "NRL_PYTHON_PREAMBLE",
    "ray_venv_python_preamble",
    "render_code_root",
    "write_payload_command",
]

# Base64 carries arbitrary YAML through every quoting layer but hides code-root
# paths from the engines' command rendering, so they travel as this marker and the
# worker substitutes its own NVFLOW_CODE_ROOT after decoding.
_PAYLOAD_CODE_ROOT = "@NVFLOW_CODE_ROOT@"


def write_payload_command(content: str, path: str) -> str:
    """Return a shell command that writes *content* to *path* on the worker."""
    portable = content.replace(CODE_ROOT_TOKEN, _PAYLOAD_CODE_ROOT).replace(
        "/nemo_run/code", _PAYLOAD_CODE_ROOT
    )
    encoded = base64.b64encode(portable.encode()).decode()
    return (
        f"echo {encoded} | base64 -d | "
        f'sed "s|{_PAYLOAD_CODE_ROOT}|${{NVFLOW_CODE_ROOT:-/nemo_run/code}}|g" > {path}'
    )


# The nemo-rl image carries Gym inside NeMo-RL, next to NeMo-RL's own Gym actor
# venv; /opt/nemo-gym exists only in the nemo-gym image.
NRL_GYM_ROOT = "/opt/nemo-rl/3rdparty/Gym-workspace/Gym"
NRL_GYM_BIN = "/opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/bin"

# Direct Slurm exports the NeMo-RL save deadline derived from the partition
# time limit. The override expands to nothing wherever the variable is unset.
CHECKPOINT_SAVE_BY_ENV = "NVFLOW_CHECKPOINT_SAVE_BY"
CHECKPOINT_SAVE_BY_OVERRIDE = (
    "${NVFLOW_CHECKPOINT_SAVE_BY:+++checkpointing.checkpoint_must_save_by="
    "$NVFLOW_CHECKPOINT_SAVE_BY}"
)


# Bash preamble for SFT and GRPO training stages.
# Sets $NRL_PYTHON to the pre-built venv interpreter and fails closed when a
# supplied image does not satisfy the qualified runtime profile.
NRL_PYTHON_PREAMBLE = (
    "if [ -x /opt/nemo_rl_venv/bin/python ]; then "
    "  NRL_PYTHON=/opt/nemo_rl_venv/bin/python; "
    "else "
    "  echo 'ERROR: qualified nemo-rl image is missing /opt/nemo_rl_venv/bin/python' >&2; "
    "  exit 1; "
    "fi"
)


def ray_venv_python_preamble(venv_path: str, uv_extra: str) -> str:
    """Bash preamble resolving a Ray worker venv Python.

    Used by checkpoint conversion scripts which need specific Ray venvs
    (Megatron or Automodel) from the selected NeMo-RL image.

    Sets ``$CONVERT_PYTHON`` for use in the generated bash script.
    """
    return (
        f"if [ -x {venv_path} ]; then "
        f"  CONVERT_PYTHON={venv_path}; "
        f"else "
        f'  echo "ERROR: qualified nemo-rl image is missing {venv_path} ({uv_extra})" >&2; '
        f"  exit 1; "
        f"fi"
    )
