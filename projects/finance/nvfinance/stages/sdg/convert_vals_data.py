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
"""Convert the prepared VALS Finance Agent benchmark into TSDG seed files."""

from pathlib import Path
from typing import Any

from nvflow.core import console
from nvflow.execution import ArtifactDeclaration, CapturedExecutionStage, declare_task_outputs
from nvflow.execution.commands import build_python_cmd


class ConvertValsDataStage(CapturedExecutionStage):
    """Create ``seed_questions.jsonl`` and ``company_info.tsv`` from VALS."""

    capture_strategy = "run_cmd"

    def build_tasks(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        from nvflow.integrations.nemo_skills.commands import run_cmd, wrap_arguments

        input_file = str(config["input_file"])
        output_file = str(config["output_file"])
        company_info_file = str(config["company_info_file"])

        console.status("Converting VALS questions to template-SDG seeds")
        console.detail("Input file", input_file)
        console.detail("Seed questions", output_file)
        console.detail("Company information", company_info_file)
        console.blank()

        command = build_python_cmd(
            "nvfinance.utils.sdg.convert_vals_data",
            input_file=input_file,
            output_file=output_file,
            company_info_file=company_info_file,
        )
        output_dir = Path(output_file).parent
        with declare_task_outputs(
            ArtifactDeclaration(name="seed-data", path=output_file, kind="file"),
            ArtifactDeclaration(name="company-info", path=company_info_file, kind="file"),
        ):
            run_cmd(
                ctx=wrap_arguments(command),
                cluster=cluster,
                log_dir=str(output_dir / "logs"),
                expname=expname,
                run_after=run_after,
                **config.get("stage_kwargs", {}),
            )

        console.success(f"VALS seed conversion scheduled: {output_file}")

    def validate_config(self, config: dict[str, Any]) -> None:
        for field in ("input_file", "output_file", "company_info_file"):
            if not isinstance(config.get(field), str) or not config[field]:
                raise ValueError(f"'{field}' must be a non-empty string")
