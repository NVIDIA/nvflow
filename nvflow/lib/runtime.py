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
"""Dual-mode Python interpreter resolution for Slurm jobs.

NVFlow containers ship with pre-built venvs (airgap mode). When those
venvs are absent (dev mode with host-mounted NeMo-RL source), the code
falls back to ``uv run`` which resolves dependencies on the fly.

All resolution happens **at bash runtime on the cluster node**, not at
Python submission time, because the venv paths exist inside the container.
"""

from __future__ import annotations

__all__ = [
    "NRL_PYTHON_PREAMBLE",
    "ray_venv_python_preamble",
]

# Bash preamble for SFT and GRPO training stages.
# Sets $NRL_PYTHON to the pre-built venv interpreter if available,
# otherwise falls back to uv with the NeMo-RL project context.
NRL_PYTHON_PREAMBLE = (
    "if [ -x /opt/nemo_rl_venv/bin/python ]; then "
    "  NRL_PYTHON=/opt/nemo_rl_venv/bin/python; "
    "else "
    "  export UV_PROJECT=/opt/NeMo-RL; "
    '  NRL_PYTHON="uv run --active python"; '
    "fi"
)


def ray_venv_python_preamble(venv_path: str, uv_extra: str) -> str:
    """Bash preamble resolving a Ray worker venv Python.

    Used by checkpoint conversion scripts which need specific Ray venvs
    (megatron, dtensor v1/v2) that are only present in the airgap container.

    Sets ``$CONVERT_PYTHON`` for use in the generated bash script.
    """
    return (
        f"if [ -x {venv_path} ]; then "
        f"  CONVERT_PYTHON={venv_path}; "
        f"else "
        f"  export UV_PROJECT=/opt/NeMo-RL; "
        f'  CONVERT_PYTHON="uv run --extra {uv_extra} python"; '
        f"fi"
    )
