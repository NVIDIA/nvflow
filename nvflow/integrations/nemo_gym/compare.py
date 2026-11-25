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
"""Native Gym comparison command construction."""

from __future__ import annotations

import shlex
from typing import Any

from nvflow.integrations.nemo_skills.filesystem import CONTAINER_CODE_DIR

from .native_commands import render_gym_command
from .native_contract import NativeGymArtifactPaths, artifact_paths_for


def _validate_bundle(artifacts: NativeGymArtifactPaths) -> str:
    return (
        f'PYTHONPATH="{CONTAINER_CODE_DIR}${{PYTHONPATH:+:$PYTHONPATH}}" '
        "python3 -m nvflow.integrations.nemo_gym.native_sharding validate "
        f"--materialized-inputs {shlex.quote(artifacts.materialized_inputs.as_posix())} "
        f"--successes {shlex.quote(artifacts.rollouts.as_posix())} "
        f"--completion {shlex.quote(artifacts.completion.as_posix())}"
    )


def build_compare_command(config: dict[str, Any]) -> str:
    """Render the pinned Gym one-baseline/one-candidate comparison command."""
    output_dir = str(config["output_dir"])
    args: list[str] = [
        "--baseline",
        str(config["baseline_rollouts"]),
        "--candidates",
        str(config["candidate_rollouts"]),
        "--output-dir",
        output_dir,
        "--report-format",
        str(config.get("report_format", "both")),
    ]
    for field, flag in (
        ("agent", "--agent"),
        ("baseline_agent", "--baseline-agent"),
        ("candidate_agent", "--candidate-agents"),
    ):
        value = config.get(field)
        if value:
            args.extend((flag, str(value)))
    compare = render_gym_command("gym", "eval", "compare", args=args)
    baseline = artifact_paths_for(
        str(config["baseline_rollouts"]),
        str(config["baseline_completion"]),
    )
    candidate = artifact_paths_for(
        str(config["candidate_rollouts"]),
        str(config["candidate_completion"]),
    )
    return (
        "set -euo pipefail\n"
        f"{_validate_bundle(baseline)}\n"
        f"{_validate_bundle(candidate)}\n"
        f"mkdir -p {shlex.quote(output_dir)}\n"
        f"cd {shlex.quote(str(config.get('gym_path', '/opt/nemo-gym')))}\n"
        f"{compare}"
    )


__all__ = ["build_compare_command"]
