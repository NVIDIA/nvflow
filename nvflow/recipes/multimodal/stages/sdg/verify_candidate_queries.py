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
"""Verify generated HopChain candidate queries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nvflow.core import BaseStage, StageRegistry, console
from nvflow.recipes.multimodal.utils.runtime_env import (
    resolve_partition,
)


@StageRegistry.register(
    recipe="multimodal", workflow="hopchain_sdg", stage="verify_candidate_queries"
)
class VerifyCandidateQueriesStage(BaseStage):
    """Run machine-side structural verification on candidate queries."""

    workflow = "hopchain_sdg"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute candidate verification."""
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        project_root = config["project_root"]
        script_path = (
            f"{project_root}/nvflow/recipes/multimodal/utils/hopchain_verify_candidate_queries.py"
        )
        output_file = config["output_file"]
        summary_file = config["summary_file"]
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        command = (
            f'PYTHONPATH="{project_root}:$PYTHONPATH" '
            f'python3 "{script_path}" '
            f'--input "{config["input_file"]}" --output "{output_file}" --output-dir "{output_dir}" '
            f'--summary "{summary_file}" '
            f"--min-hop-count {config.get('min_hop_count', 4)}"
        )

        console.status("Verifying HopChain candidate queries")
        console.detail("Input file", config["input_file"])
        console.detail("Output file", output_file)

        run_cmd(
            ctx=wrap_arguments(""),
            cluster=cluster,
            config_dir=config.get("cluster_config_dir"),
            command=command,
            container="nemo-skills",
            expname=expname,
            partition=resolve_partition(config, cluster, cpu=True),
            time_min=str(config.get("time_min", 30)),
            num_nodes=1,
            num_tasks=1,
            run_after=run_after,
            log_dir=str(output_dir / "logs"),
        )
        console.success("Candidate verification job submitted")

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate stage configuration."""
        required_fields = [
            "input_file",
            "output_file",
            "output_dir",
            "summary_file",
            "project_root",
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
