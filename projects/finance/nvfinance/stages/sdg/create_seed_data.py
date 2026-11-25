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
"""Create seed data for template-based SDG pipeline."""

from pathlib import Path
from typing import Any

from nvflow.core import console
from nvflow.execution import ArtifactDeclaration, CapturedExecutionStage, declare_task_outputs


class CreateSeedDataStage(CapturedExecutionStage):
    """Create seed data files from HuggingFace and SEC EDGAR."""

    capture_strategy = "run_cmd"

    def validate_config(self, config: dict[str, Any]) -> None:
        """Require a real SEC contact identity before any EDGAR request is made."""
        missing = [
            name
            for name, key in (
                ("SEC_IDENTITY_EMAIL", "sec_identity_email"),
                ("SEC_IDENTITY_COMPANY", "sec_identity_company"),
            )
            if not config.get(key)
        ]
        if not missing:
            return
        assignments = " ".join(f"{name}=<value>" for name in missing)
        raise ValueError(
            "create_seed_data queries the SEC EDGAR API, which requires a contact "
            "identity in the request User-Agent. "
            f"Set {' and '.join(missing)} to real values:\n"
            f"  export {assignments}"
        )

    def build_tasks(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute seed data creation."""
        from nvflow.integrations.nemo_skills.commands import run_cmd, wrap_arguments

        output_file = config["output_file"]
        company_info_file = config["company_info_file"]
        sec_identity_email = config["sec_identity_email"]
        sec_identity_company = config["sec_identity_company"]
        filter_company_list = config.get("filter_company_list")
        num_seed_questions = config.get("num_seed_questions")

        sec_identity = f"{sec_identity_company} {sec_identity_email}"

        console.status("Creating seed data for template-based SDG")
        console.detail("Output file", output_file)
        console.detail("Company info file", company_info_file)
        console.detail("SEC Identity", sec_identity)
        console.detail("Filter company list", filter_company_list)
        console.detail("Num seed questions", num_seed_questions)
        console.blank()

        cmd = (
            f"python -m nvfinance.utils.sdg.create_seed_data "
            f"--output_file {output_file} "
            f"--company_info_file {company_info_file} "
            f'--sec_identity "{sec_identity}"'
        )

        if filter_company_list:
            cmd += f" --filter_company_list {filter_company_list}"

        if num_seed_questions:
            cmd += f" --num_seed_questions {num_seed_questions}"

        # Set up log directory
        output_dir = Path(output_file).parent
        log_dir = str(output_dir / "logs")

        with declare_task_outputs(
            ArtifactDeclaration(name="seed-data", path=str(output_file), kind="file"),
            ArtifactDeclaration(
                name="company-info",
                path=str(company_info_file),
                kind="file",
            ),
        ):
            run_cmd(
                ctx=wrap_arguments(cmd),
                cluster=cluster,
                log_dir=log_dir,
                expname=expname,
                run_after=run_after,
                **config.get("stage_kwargs", {}),
            )

        console.success(f"Seed data creation scheduled: {output_file}")
