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
"""Post-processing builders shared by rollout and reward collection."""

import json
import shlex

from nvflow.integrations.nemo_skills.filesystem import CONTAINER_CODE_DIR

from .reward_profiles import dataset_selection_sha256

DATASET_SELECTION_WORKER = "nvflow.integrations.nemo_gym.workers.build_training_dataset"
DATASET_VALIDATION_WORKER = "nvflow.integrations.nemo_gym.workers.validate_dataset_selection"


def build_blade_script(
    *,
    command: str | None,
    output_dir: str,
    blade_output_dir: str,
    rollouts: str,
    materialized_inputs: str,
    reward_profile: str,
) -> str:
    """Render optional BLADE analysis inside the CPU analyze job."""
    if command is None:
        return ""

    return (
        f"mkdir -p {shlex.quote(blade_output_dir)}\n"
        f"export NVFLOW_OUTPUT_DIR={shlex.quote(output_dir)}\n"
        f"export NVFLOW_BLADE_OUTPUT_DIR={shlex.quote(blade_output_dir)}\n"
        f"export NVFLOW_ROLLOUTS={shlex.quote(rollouts)}\n"
        f"export NVFLOW_MATERIALIZED_INPUTS={shlex.quote(materialized_inputs)}\n"
        f"export NVFLOW_REWARD_PROFILE={shlex.quote(reward_profile)}\n"
        f"{command}\n"
    )


def build_dataset_selection_cmd(
    *,
    source: str,
    reward_profile: str,
    materialized_inputs: str,
    rollouts: str,
    rollout_completion: str,
    profiled_output: str,
    training_output: str,
    report_output: str,
    completion_output: str,
    diagnostics_output: str,
    selection: dict[str, object],
    responses_create_params_overrides: dict[str, object] | None = None,
) -> str:
    """Render the shared normalize-and-select worker command."""
    overrides_argument = ""
    if responses_create_params_overrides:
        serialized = json.dumps(
            responses_create_params_overrides,
            separators=(",", ":"),
            sort_keys=True,
        )
        overrides_argument = (
            f"    --responses-create-params-overrides {shlex.quote(serialized)} \\\n"
        )
    selection_sha256 = dataset_selection_sha256(
        selection,
        responses_create_params_overrides,
    )
    return (
        "set -e\n"
        f"if [ -f {shlex.quote(completion_output)} ]; then\n"
        f'  if PYTHONPATH="{CONTAINER_CODE_DIR}${{PYTHONPATH:+:$PYTHONPATH}}" '
        f"python3 -m {DATASET_VALIDATION_WORKER} "
        f"--profiled {shlex.quote(profiled_output)} "
        f"--train {shlex.quote(training_output)} "
        f"--report {shlex.quote(report_output)} "
        f"--completion {shlex.quote(completion_output)} "
        f"--diagnostics {shlex.quote(diagnostics_output)} "
        f"--selection-sha256 {selection_sha256}; then\n"
        "    exit 0\n"
        "  fi\n"
        f"  rm -f {shlex.quote(completion_output)}\n"
        "fi\n"
        'echo "Build Profiled and Training Datasets"\n'
        f'PYTHONPATH="{CONTAINER_CODE_DIR}${{PYTHONPATH:+:$PYTHONPATH}}" '
        f"python3 -m {DATASET_SELECTION_WORKER} \\\n"
        f"    --source {shlex.quote(source)} \\\n"
        f"    --profile {shlex.quote(reward_profile)} \\\n"
        f"    --materialized-inputs {shlex.quote(materialized_inputs)} \\\n"
        f"    --rollouts {shlex.quote(rollouts)} \\\n"
        f"    --rollout-completion {shlex.quote(rollout_completion)} \\\n"
        f"    --profiled-output {shlex.quote(profiled_output)} \\\n"
        f"    --training-output {shlex.quote(training_output)} \\\n"
        f"    --report-output {shlex.quote(report_output)} \\\n"
        f"    --completion-output {shlex.quote(completion_output)} \\\n"
        f"    --diagnostics-output {shlex.quote(diagnostics_output)} \\\n"
        f"{overrides_argument}"
        f"    --spec {shlex.quote(json.dumps(selection, separators=(',', ':'), sort_keys=True))}\n"
        f"printf '%s\\n' {shlex.quote(f'Done. Training data in {training_output}')}\n"
    )


def build_dataset_selection_cleanup(*paths: str) -> str:
    """Remove stale selected-data artifacts when selection is disabled."""
    return "rm -f " + " ".join(shlex.quote(path) for path in paths) + "\n"


__all__ = [
    "DATASET_SELECTION_WORKER",
    "DATASET_VALIDATION_WORKER",
    "build_blade_script",
    "build_dataset_selection_cmd",
    "build_dataset_selection_cleanup",
]
