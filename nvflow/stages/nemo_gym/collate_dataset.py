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
"""Concrete shared stage for native ``gym dataset collate`` operations."""

from __future__ import annotations

import json
import shlex
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from nvflow.execution import ArtifactDeclaration, PlannedExecutionStage
from nvflow.execution.commands import build_python_cmd
from nvflow.integrations.nemo_gym.contracts import GymDatasetRef
from nvflow.integrations.nemo_gym.environments import selected_environments
from nvflow.integrations.nemo_gym.runtime_binding import render_runtime_binding_command

if TYPE_CHECKING:
    from nvflow.execution import PlanningContext, StagePlan, TaskPlan

_SHUFFLE_MODULE = "nvflow.integrations.nemo_gym.workers.shuffle_jsonl"


def _collate_operations(config: dict[str, Any]) -> list[dict[str, Any]]:
    environments = selected_environments(config, require_input=False)
    output_root = str(config["output_dir"]).rstrip("/")
    mode = str(config.get("mode", "train_preparation"))
    dataset_split = "example" if mode == "example_validation" else "train"
    operations = []
    for environment in environments:
        if not environment.source_data:
            raise ValueError(f"environment {environment.name!r} requires source_data")
        dataset_paths = {dataset_split: environment.source_data}
        output_dir = f"{output_root}/{environment.output_name}"
        datasets_list: list[GymDatasetRef] = []
        for split, path in dataset_paths.items():
            if not isinstance(split, str) or not split or not isinstance(path, str) or not path:
                raise ValueError(
                    f"environment {environment.name!r} dataset names and paths "
                    "must be non-empty strings"
                )
            normalized_split = "validation" if split in {"val", "validation"} else split
            datasets_list.append(
                GymDatasetRef(
                    path=path,
                    name=normalized_split,
                    split=normalized_split,
                )
            )
        if len({dataset.name for dataset in datasets_list}) != len(datasets_list):
            raise ValueError(
                f"environment {environment.name!r} has duplicate normalized dataset names"
            )
        datasets = tuple(datasets_list)
        overlay_path = f"{output_dir}/agent_config_overlay.yaml"
        binding_path = f"{output_dir}/gym_binding.json"
        paths = [*environment.config_paths, overlay_path]
        write_overlay = render_runtime_binding_command(
            config_paths=environment.config_paths,
            agent_name=environment.agent_name,
            output_config=overlay_path,
            output_binding=binding_path,
            datasets=tuple(
                {
                    "name": dataset.name,
                    "type": dataset.split,
                    "license": dataset.license,
                    "jsonl_fpath": dataset.path,
                }
                for dataset in datasets
            ),
        )
        command = (
            f"mkdir -p {shlex.quote(output_dir)} && "
            f"cd {shlex.quote(str(config['gym_path']))} && "
            f"{write_overlay} && "
            f"gym dataset collate "
            f"{shlex.quote(f'+config_paths={json.dumps(paths, separators=(",", ":"))}')} "
            f"+output_dirpath={shlex.quote(output_dir)} "
            f"+mode={shlex.quote(mode)} "
            "+error_on_almost_servers=false"
        )
        if config.get("should_download", False):
            command += " +should_download=true"
        if config.get("shuffle", False):
            if dataset_split != "train":
                raise ValueError("shuffle is supported only in train_preparation mode")
            command += " && " + build_python_cmd(
                _SHUFFLE_MODULE,
                input_file=f"{output_dir}/train.jsonl",
                random_seed=config.get("random_seed", 42),
            )
        outputs = (
            ArtifactDeclaration(
                f"{dataset_split}_dataset",
                f"{output_dir}/{dataset_split}.jsonl",
            ),
            ArtifactDeclaration("gym_binding", binding_path),
        )
        operations.append(
            {
                "name": environment.output_name,
                "source": environment.source_data,
                "output_dir": output_dir,
                "command": command,
                "outputs": outputs,
            }
        )
    return operations


class CollateDatasetStage(PlannedExecutionStage):
    """Collate project data into Gym-routed datasets."""

    operations = staticmethod(_collate_operations)

    def validate_config(self, config: dict[str, Any]) -> None:
        removed = sorted({"input_dir", "input_filename", "extra_args"} & config.keys())
        if removed:
            raise ValueError(
                "collate fields were removed; configure environments.<name>.source_data: "
                + ", ".join(removed)
            )
        for field in ("output_dir", "gym_path", "container"):
            if not isinstance(config.get(field), str) or not config[field]:
                raise ValueError(f"'{field}' must be a non-empty string in collate config")
        mode = config.get("mode", "train_preparation")
        if mode not in {"train_preparation", "example_validation"}:
            raise ValueError(f"mode must be train_preparation or example_validation, got {mode!r}")
        for field in ("should_download", "shuffle"):
            if not isinstance(config.get(field, False), bool):
                raise ValueError(f"'{field}' must be a boolean")
        random_seed = config.get("random_seed", 42)
        if isinstance(random_seed, bool) or not isinstance(random_seed, int):
            raise ValueError("'random_seed' must be an integer")
        self.operations(config)

    def plan(self, context: PlanningContext) -> StagePlan:
        from nvflow.execution import StagePlan, plan_run_cmd, rekey_tasks

        tasks: list[TaskPlan] = []
        for operation in self.operations(dict(context.config)):
            child_context = replace(
                context,
                expname=f"{context.expname}-{operation['name']}",
            )
            child = plan_run_cmd(
                child_context,
                operation["command"],
                cluster=context.target.cluster,
                container=context.config["container"],
                num_gpus=context.config.get("num_gpus", 0),
                log_dir=f"{operation['output_dir']}/logs",
                output_dir=operation["output_dir"],
                expname=child_context.expname,
                run_after=list(context.run_after) or None,
                installation_command=context.config.get("installation_command"),
            )
            contribution_tasks = list(
                rekey_tasks(
                    child.tasks,
                    stage=context.stage,
                    prefix=f"{context.stage}.{operation['name']}",
                )
            )
            contribution_tasks[-1] = replace(
                contribution_tasks[-1],
                outputs=operation["outputs"],
            )
            tasks.extend(contribution_tasks)
        if not tasks:
            return StagePlan.noop(context.stage, "environment filter selected no collate work")
        return StagePlan(stage=context.stage, tasks=tuple(tasks))


__all__ = ["CollateDatasetStage"]
