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
"""Concrete shared stage for native Gym benchmark preparation."""

from __future__ import annotations

import shlex
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from nvflow.execution import ArtifactDeclaration, PlannedExecutionStage
from nvflow.integrations.nemo_gym.contracts import (
    GymBenchmarkSpec,
    require_contract_version,
)

if TYPE_CHECKING:
    from nvflow.execution import PlanningContext, StagePlan


def _resolve_entry(entry: str | dict[str, Any]) -> GymBenchmarkSpec:
    if isinstance(entry, str):
        entry = {"benchmark": entry}
    if (
        not isinstance(entry, dict)
        or not isinstance(entry.get("benchmark"), str)
        or not entry["benchmark"]
    ):
        raise ValueError(f"each benchmark entry requires a selector, got {entry!r}")
    unknown = sorted(
        set(entry) - {"benchmark", "prepared_file", "prompt_config", "output_file", "overrides"}
    )
    if unknown:
        raise ValueError(f"unknown benchmark field(s): {', '.join(unknown)}")
    benchmark = entry["benchmark"]
    name = benchmark.split("/", 1)[0]
    configured_overrides = entry.get("overrides")
    overrides = () if configured_overrides is None else configured_overrides
    if isinstance(overrides, str) or not isinstance(overrides, list | tuple):
        raise ValueError("benchmark overrides must be a sequence")
    prepared_file = entry.get("prepared_file", f"benchmarks/{name}/data/{name}_benchmark.jsonl")
    output_file = entry.get("output_file", f"{name}/gym_eval.jsonl")
    prompt_config = entry.get(
        "prompt_config",
        f"benchmarks/{name}/prompts/default.yaml",
    )
    return GymBenchmarkSpec(
        benchmark=benchmark,
        prepared_file=prepared_file,
        prompt_config=prompt_config,
        output_file=output_file,
        overrides=tuple(overrides),
    )


def _benchmark_command(spec: GymBenchmarkSpec, output_dir: str) -> str:
    destination = f"{output_dir}/{spec.output_file}"
    destination_dir = str(PurePosixPath(destination).parent)
    if spec.prompt_config:
        place = (
            "gym dataset render "
            f"+input_jsonl_fpath={shlex.quote(spec.prepared_file)} "
            f"+prompt_config={shlex.quote(spec.prompt_config)} "
            f"+output_jsonl_fpath={shlex.quote(destination)}"
        )
    else:
        place = f"cp {shlex.quote(spec.prepared_file)} {shlex.quote(destination)}"
    # Overrides are trusted project configuration and may intentionally expand
    # worker environment variables (for example, gated-dataset credentials).
    extra = "".join(f" {override}" for override in spec.overrides)
    prepare = " && ".join(
        [
            f"gym eval prepare --benchmark {shlex.quote(spec.benchmark)}{extra} "
            "+use_cached_prepared_benchmarks=true",
            f"mkdir -p {shlex.quote(destination_dir)}",
            place,
        ]
    )
    skip = f"echo {shlex.quote(f'Using prepared data: {destination}')}"
    return f"if [ -s {shlex.quote(destination)} ]; then {skip}; else {prepare}; fi"


def _operation(config: dict[str, Any]) -> tuple[str, tuple[ArtifactDeclaration, ...]]:
    entries = config.get("benchmarks")
    if isinstance(entries, str) or not isinstance(entries, list | tuple) or not entries:
        raise ValueError("'benchmarks' must be a non-empty sequence")
    specs = tuple(_resolve_entry(entry) for entry in entries)
    output_files = [spec.output_file for spec in specs]
    if len(output_files) != len(set(output_files)):
        raise ValueError("benchmark entries must have unique output_file values")
    output_dir = str(config["output_dir"])
    gym_path = str(config["gym_path"])
    command = (
        "unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE && "
        f"cd {shlex.quote(gym_path)} && "
        + " && ".join(_benchmark_command(spec, output_dir) for spec in specs)
    )
    artifacts = tuple(
        ArtifactDeclaration(
            name=spec.benchmark,
            path=f"{output_dir}/{spec.output_file}",
            kind="file",
        )
        for spec in specs
    )
    return command, artifacts


class PrepareBenchmarkStage(PlannedExecutionStage):
    """Prepare and place one or more Gym benchmark datasets."""

    def validate_config(self, config: dict[str, Any]) -> None:
        require_contract_version(config.get("contract_version"))
        for field in ("output_dir", "gym_path", "container"):
            if not isinstance(config.get(field), str) or not config[field]:
                raise ValueError(
                    f"'{field}' must be a non-empty string in benchmark preparation config"
                )
        _operation(config)

    def plan(self, context: PlanningContext) -> StagePlan:
        from nvflow.execution import declare_task_outputs, plan_run_cmd

        config = dict(context.config)
        command, artifacts = _operation(config)
        with declare_task_outputs(*artifacts):
            return plan_run_cmd(
                context,
                command,
                cluster=context.target.cluster,
                container=config["container"],
                num_gpus=config.get("num_gpus", 0),
                log_dir=f"{config['output_dir']}/logs",
                expname=context.expname,
                run_after=list(context.run_after) or None,
                installation_command=config.get("installation_command"),
            )


__all__ = ["PrepareBenchmarkStage"]
