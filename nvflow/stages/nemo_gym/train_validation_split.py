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
"""Split each selected NeMo Gym environment's training data.

Each environment reads ``{input_dir}/{env_name}/{input_filename}`` and
writes to ``{output_dir}/{env_name}/`` through the generic
``data.train_validation_split`` splitter.  The input filename defaults to
``final_result.jsonl`` but can be overridden via ``input_filename``
in the stage config.
"""

from __future__ import annotations

import shlex
from dataclasses import replace
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from nvflow.execution import ArtifactDeclaration, PlannedExecutionStage, declare_task_outputs
from nvflow.integrations.nemo_gym.native_contract import NativeGymOutputLayout
from nvflow.stages.data.train_validation_split import split_command, validate_split_options

if TYPE_CHECKING:
    from nvflow.execution import PlanningContext, StagePlan, TaskPlan


class TrainValidationSplitStage(PlannedExecutionStage):
    """Split each selected environment into train and validation sets.

    Accepts the generic split options. With ``input_contract:
    dataset_selection``, it first validates the Gym dataset-selection
    evidence next to each input file.
    """

    @staticmethod
    def _dataset_layout(
        config: dict[str, Any],
        input_file: str,
    ) -> NativeGymOutputLayout | None:
        if config.get("input_contract") != "dataset_selection":
            return None
        return NativeGymOutputLayout.collection(str(PurePosixPath(input_file).parent))

    @staticmethod
    def _scope_dependencies(
        dependencies: list[str] | tuple[str, ...] | None,
        environment: str,
        environment_names: tuple[str, ...],
    ) -> list[str] | None:
        scoped = []
        for dependency in dependencies or ():
            matches = sorted(
                (name for name in environment_names if dependency.endswith(f"-{name}")),
                key=len,
                reverse=True,
            )
            if not matches or matches[0] == environment:
                scoped.append(dependency)
        return scoped or None

    @staticmethod
    def _required_input_paths(
        input_file: str,
        dataset_layout: NativeGymOutputLayout | None,
    ) -> list[str]:
        required = [input_file]
        if dataset_layout is not None:
            required.extend(
                (
                    dataset_layout.profiled_dataset.as_posix(),
                    dataset_layout.dataset_selection_report.as_posix(),
                    dataset_layout.dataset_completion.as_posix(),
                )
            )
        return required

    @classmethod
    def _require_declared_inputs(
        cls,
        *,
        input_file: str,
        dataset_layout: NativeGymOutputLayout | None,
        filesystem: Any,
        context: str,
    ) -> None:
        required = cls._required_input_paths(input_file, dataset_layout)
        present = filesystem.batch_exists(required)
        missing = [path for path in required if not present.get(path, False)]
        if missing:
            raise ValueError(
                f"{context}: required input artifact(s) are missing: " + ", ".join(missing)
            )

    @staticmethod
    def _command(
        *,
        input_file: str,
        output_dir: str,
        config: dict[str, Any],
    ) -> str:
        command = split_command(input_file=input_file, output_dir=output_dir, config=config)
        layout = TrainValidationSplitStage._dataset_layout(config, input_file)
        if layout is not None:
            validate = (
                "python -m "
                "nvflow.integrations.nemo_gym.workers.validate_dataset_selection "
                f"--profiled {shlex.quote(layout.profiled_dataset.as_posix())} "
                f"--train {shlex.quote(input_file)} "
                f"--report {shlex.quote(layout.dataset_selection_report.as_posix())} "
                f"--completion {shlex.quote(layout.dataset_completion.as_posix())} "
                f"--diagnostics {shlex.quote(layout.unprofiled_samples.as_posix())}"
            )
            command = f"{validate} && {command}"
        return command

    def plan(self, context: PlanningContext) -> StagePlan:
        from nvflow.execution import StagePlan, plan_run_cmd, rekey_tasks
        from nvflow.integrations.nemo_gym.environments import resolve_environments

        config = dict(context.config)
        input_filename = config.get("input_filename", "final_result.jsonl")
        environments = resolve_environments(config)
        environment_names = tuple(environments)
        operations = [
            (
                env_name,
                f"{config['input_dir']}/{env_name}/{input_filename}",
                f"{config['output_dir']}/{env_name}",
            )
            for env_name, env_config in environments.items()
            if env_config.get("raw_train_data")
        ]

        tasks: list[TaskPlan] = []
        for env_name, input_file, output_dir in operations:
            scoped_run_after = self._scope_dependencies(
                context.run_after,
                env_name,
                environment_names,
            )
            if context.live and not scoped_run_after:
                assert context.filesystem is not None
                self._require_declared_inputs(
                    input_file=input_file,
                    dataset_layout=self._dataset_layout(config, input_file),
                    filesystem=context.filesystem,
                    context="live train/validation split plan",
                )
            expname = f"{context.expname}-{env_name}"
            child_context = replace(
                context,
                expname=expname,
                run_after=tuple(scoped_run_after or ()),
            )
            with declare_task_outputs(
                ArtifactDeclaration("train", f"{output_dir}/train.jsonl"),
                ArtifactDeclaration("validation", f"{output_dir}/val.jsonl"),
                ArtifactDeclaration("completion", f"{output_dir}/split.done"),
            ):
                planned = plan_run_cmd(
                    child_context,
                    self._command(
                        input_file=input_file,
                        output_dir=output_dir,
                        config=config,
                    ),
                    cluster=context.target.cluster,
                    log_dir=f"{output_dir}/logs",
                    output_dir=output_dir,
                    expname=expname,
                    run_after=scoped_run_after,
                    sbatch_kwargs=(
                        {"dependency_type": "afterany"}
                        if config.get("input_contract") == "dataset_selection"
                        else None
                    ),
                )
            tasks.extend(
                rekey_tasks(
                    planned.tasks,
                    stage=context.stage,
                    prefix=f"{context.stage}.{env_name}",
                )
            )
        if not tasks:
            return StagePlan.noop(
                context.stage,
                "environment filter selected no train/validation split work",
            )
        return StagePlan(stage=context.stage, tasks=tuple(tasks))

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate that required configuration fields are present."""
        for field in ("environments", "input_dir", "output_dir"):
            if not config.get(field):
                raise ValueError(f"'{field}' is required in train_validation_split config")
        validate_split_options(config)
        if config.get("splits"):
            raise ValueError(
                "'splits' is not supported; each environment gets train and validation files"
            )
        input_contract = config.get("input_contract")
        if input_contract not in {None, "dataset_selection"}:
            raise ValueError("'input_contract' must be 'dataset_selection' when set")
