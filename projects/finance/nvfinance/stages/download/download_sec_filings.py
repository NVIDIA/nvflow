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
"""Download SEC filings and extract sections using in-tree utilities."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from nvflow.core import console
from nvflow.execution import PlannedExecutionStage

if TYPE_CHECKING:
    from nvflow.execution import PlanningContext, StagePlan


class DownloadSecFilingsStage(PlannedExecutionStage):
    """Download SEC filings (10-K, 10-Q, 8-K) from EDGAR and extract sections."""

    @staticmethod
    def _build_submission(
        worker_config: dict[str, Any],
        *,
        launcher_config: dict[str, Any],
    ) -> tuple[str, str]:
        """Build a worker-safe command using launcher-readable resources."""
        if "config" in launcher_config:
            config_path = Path(launcher_config["config"])
            console.detail("Loading config from", str(config_path))
            with open(config_path) as file:
                filings_config = yaml.safe_load(file)
            tickers = filings_config.get("tickers")
            start_year = filings_config.get("start_year")
            end_year = filings_config.get("end_year")
            forms = filings_config.get("forms", ["10-K", "10-Q", "8-K"])
        else:
            tickers = launcher_config["tickers"]
            start_year = launcher_config["start_year"]
            end_year = launcher_config["end_year"]
            forms = launcher_config.get("forms", ["10-K", "10-Q", "8-K"])

        output_dir = worker_config["output_dir"]
        sec_identity_email = worker_config["sec_identity_email"]
        sec_identity_company = worker_config["sec_identity_company"]
        download_only = worker_config.get("download_only", False)
        extract_only = worker_config.get("extract_only", False)
        if download_only and extract_only:
            raise ValueError("download_only and extract_only cannot both be set")

        tickers_str = " ".join(tickers) if isinstance(tickers, list) else tickers
        if isinstance(forms, str):
            forms = forms.split()
        # Quoted per form so "DEF 14A" reaches argparse as a single argument.
        forms_str = " ".join(shlex.quote(str(form)) for form in forms)
        phase_flag = ""
        if download_only:
            phase_flag = " --download_only"
        elif extract_only:
            phase_flag = " --extract_only"
        command = (
            "python -m nvfinance.utils.download.download_sec_filings "
            f'--tickers "{tickers_str}" '
            f"--forms {forms_str} "
            f"--start_year {start_year} "
            f"--end_year {end_year} "
            f"--output_dir {output_dir} "
            f'--sec_email "{sec_identity_email}" '
            f'--sec_company "{sec_identity_company}"'
            f"{phase_flag}"
        )
        return command, str(Path(output_dir) / "download-logs")

    def plan(self, context: PlanningContext) -> StagePlan:
        """Describe the same run_cmd invocation used by execution."""
        from nvflow.execution import (
            ArtifactDeclaration,
            CommandPlan,
            DependencyPlan,
            ResourceRequest,
            StagePlan,
            TaskPlan,
        )

        config = dict(context.config)
        command, log_dir = self._build_submission(
            config,
            launcher_config=dict(context.launcher_config),
        )
        stage_kwargs = dict(config.get("stage_kwargs", {}))
        resources = ResourceRequest(
            nodes=int(stage_kwargs.get("num_nodes", 1)),
            gpus_per_node=stage_kwargs.get("num_gpus"),
            tasks_per_node=int(stage_kwargs.get("num_tasks", 1)),
            partition=stage_kwargs.get("partition"),
            account=stage_kwargs.get("account"),
            exclusive=stage_kwargs.get("exclusive"),
        )
        task = TaskPlan(
            task_id=f"{context.stage}.main",
            stage=context.stage,
            name=context.expname,
            operation="nemo_skills.run_cmd",
            commands=(
                CommandPlan(
                    command=command,
                    container=str(stage_kwargs.get("container", "nemo-skills")),
                    runtime_role=str(stage_kwargs.get("container", "nemo-skills")),
                ),
            ),
            resources=(resources,),
            dependencies=tuple(DependencyPlan(name, external=True) for name in context.run_after),
            outputs=(
                ArtifactDeclaration(
                    name="filings",
                    path=str(config["output_dir"]),
                    kind="directory",
                ),
            ),
            parameters={
                "arguments": command,
                "cluster": context.target.cluster,
                "expname": context.expname,
                "log_dir": log_dir,
                "run_after": list(context.run_after),
                "stage_kwargs": stage_kwargs,
            },
        )
        return StagePlan(stage=context.stage, tasks=(task,))
