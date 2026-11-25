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
"""Reconcile multiple HopChain LLM judge outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nvflow.core import console
from nvflow.execution import ArtifactDeclaration, CapturedExecutionStage, declare_task_outputs


class ReconcileLLMJudgesStage(CapturedExecutionStage):
    """Reconcile multiple LLM judge outputs into a final accepted set."""

    capture_strategy = "run_cmd"

    def build_tasks(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Submit judge reconciliation."""

        from nvflow.integrations.nemo_skills.commands import run_cmd, wrap_arguments

        project_root = config["project_root"]
        script_path = (
            f"{project_root}/projects/hopchain/nvhopchain/utils/hopchain_reconcile_llm_judges.py"
        )
        output_dir = Path(config["output_dir"])

        command = (
            f'PYTHONPATH="{project_root}:$PYTHONPATH" '
            f'python3 "{script_path}" '
            f'--input "{config["input_file"]}" '
            f'--output "{config["output_file"]}" '
            f'--output-dir "{config["output_dir"]}" '
            f'--summary "{config["summary_file"]}"'
        )
        for judge_output_file in config["judge_output_files"]:
            command += f' --judge-input "{judge_output_file}"'

        console.status("Reconciling HopChain LLM judges")
        console.detail("Input file", config["input_file"])
        console.detail("Output file", config["output_file"])
        console.detail("Judge count", str(len(config["judge_output_files"])))

        with declare_task_outputs(
            ArtifactDeclaration(name="reconciled-candidates", path=config["output_file"]),
            ArtifactDeclaration(name="summary", path=config["summary_file"]),
        ):
            run_cmd(
                ctx=wrap_arguments(""),
                cluster=cluster,
                config_dir=config.get("cluster_config_dir"),
                command=command,
                container="nemo-skills",
                expname=expname,
                partition=config.get("partition"),
                time_min=str(config.get("time_min", 30)),
                num_nodes=1,
                num_tasks=1,
                run_after=run_after,
                log_dir=str(output_dir / "logs"),
                check_mounted_paths=True,
            )
        console.success("HopChain LLM judge reconciliation job submitted")

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate stage configuration."""
        required_fields = [
            "input_file",
            "output_file",
            "output_dir",
            "summary_file",
            "judge_output_files",
            "project_root",
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
