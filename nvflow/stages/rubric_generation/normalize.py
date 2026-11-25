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

"""Shared stage and worker CLI for rubric-input normalization."""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING, Any

from .contracts import METHODS, normalize_file
from .input_adapters import INPUT_FORMATS, normalize_input_format

if __name__ != "__main__":
    from nvflow.core import console
    from nvflow.execution import PlannedExecutionStage
    from nvflow.execution.commands import build_python_cmd

    if TYPE_CHECKING:
        from nvflow.execution import PlanningContext, StagePlan


if __name__ != "__main__":

    def build_normalize_command(config: dict[str, Any]) -> str:
        """Build the portable worker command for one normalization stage."""

        flags: dict[str, Any] = {
            "input_file": config["input_file"],
            "output_file": config["output_file"],
            "audit_file": config["audit_file"],
            "method": config["method"],
            "input_format": config.get("input_format", "canonical"),
        }
        if config.get("field_map"):
            flags["field_map_json"] = json.dumps(config["field_map"], sort_keys=True)
        if config.get("max_records") is not None:
            flags["max_records"] = config["max_records"]
        command = build_python_cmd("nvflow.stages.rubric_generation.normalize", **flags)
        if config.get("allow_invalid_records", False):
            command += " --allow-invalid-records"
        return command

    class NormalizeRubricInputStage(PlannedExecutionStage):
        """Map an arbitrary text recipe's fields into the rubric input contract."""

        capture_strategy = "run_cmd"

        def validate_config(self, config: dict[str, Any]) -> None:
            for field in ("input_file", "output_file", "audit_file", "method"):
                if not config.get(field):
                    raise ValueError(f"'{field}' is required for rubric input normalization")
            if config["method"] not in METHODS:
                raise ValueError(f"method must be one of {sorted(METHODS)}")
            normalize_input_format(config.get("input_format", "canonical"))
            if config.get("field_map") is not None and not isinstance(config["field_map"], dict):
                raise ValueError("field_map must be an object")
            if config.get("max_records") is not None and (
                not isinstance(config["max_records"], int)
                or isinstance(config["max_records"], bool)
                or config["max_records"] < 1
            ):
                raise ValueError("max_records must be a positive integer")

        def execute(
            self,
            config: dict[str, Any],
            cluster: str,
            expname: str,
            run_after: list[str] | None = None,
        ) -> None:
            from nvflow.execution import declare_task_outputs
            from nvflow.integrations.nemo_skills.commands import run_cmd, wrap_arguments

            console.status(f"Normalizing rubric inputs ({config['method']})")
            kwargs: dict[str, Any] = {
                "cluster": cluster,
                "num_gpus": 0,
                "log_dir": config.get("log_dir") or f"{config['output_file']}.logs",
                "expname": expname,
                "run_after": run_after,
            }
            if config.get("container"):
                kwargs["container"] = config["container"]
            if config.get("installation_command"):
                kwargs["installation_command"] = config["installation_command"]
            with declare_task_outputs(
                output_file=config["output_file"],
            ):
                run_cmd(
                    ctx=wrap_arguments(""),
                    command=build_normalize_command(config),
                    **kwargs,
                )

        def plan(self, context: PlanningContext) -> StagePlan:
            from nvflow.execution import declare_task_outputs, plan_run_cmd

            config = dict(context.config)
            parameters: dict[str, Any] = {
                "cluster": context.target.cluster,
                "num_gpus": 0,
                "log_dir": config.get("log_dir") or f"{config['output_file']}.logs",
                "expname": context.expname,
                "run_after": list(context.run_after) or None,
            }
            if config.get("container"):
                parameters["container"] = config["container"]
            if config.get("installation_command"):
                parameters["installation_command"] = config["installation_command"]
            with declare_task_outputs(output_file=config["output_file"]):
                return plan_run_cmd(
                    context,
                    build_normalize_command(config),
                    **parameters,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", "--input_file", required=True)
    parser.add_argument("--output-file", "--output_file", required=True)
    parser.add_argument("--audit-file", "--audit_file", required=True)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument(
        "--input-format", "--input_format", choices=sorted(INPUT_FORMATS), default="canonical"
    )
    parser.add_argument("--field-map-json", "--field_map_json")
    parser.add_argument("--max-records", "--max_records", type=int)
    parser.add_argument("--allow-invalid-records", action="store_true")
    args = parser.parse_args()
    field_map = json.loads(args.field_map_json) if args.field_map_json else None
    stats = normalize_file(
        input_file=args.input_file,
        output_file=args.output_file,
        audit_file=args.audit_file,
        method=args.method,
        input_format=args.input_format,
        field_map=field_map,
        allow_invalid_records=args.allow_invalid_records,
        max_records=args.max_records,
    )
    print(json.dumps(stats, sort_keys=True))


if __name__ == "__main__":
    main()
