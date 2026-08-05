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
"""Run LiteLLM judges over verified HopChain candidate queries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nvflow.core import BaseStage, StageRegistry, console
from nvflow.recipes.multimodal.utils.inference import resolve_project_path
from nvflow.recipes.multimodal.utils.runtime_env import (
    resolve_partition,
)


@StageRegistry.register(
    recipe="multimodal", workflow="hopchain_sdg", stage="judge_candidate_queries_openai"
)
class JudgeCandidateQueriesStage(BaseStage):
    """Evaluate verified candidate queries with an external LiteLLM judge."""

    workflow = "hopchain_sdg"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Submit a CPU judge job."""

        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        project_root = config["project_root"]
        script_path = f"{project_root}/nvflow/recipes/multimodal/utils/hopchain_run_llm_judge.py"
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = resolve_project_path(config["prompt_file"], project_root)
        command = (
            f'PYTHONPATH="{project_root}:$PYTHONPATH" '
            f'python3 "{script_path}" '
            f'--input "{config["input_file"]}" '
            f'--output "{config["output_file"]}" '
            f'--summary "{config["summary_file"]}" '
            f'--prompt "{prompt_file}" '
            f'--judge-name "{config["judge_name"]}" '
            f'--provider "{config["provider"]}" '
            f'--model "{config["model"]}" '
            f'--api-key-name "{config["api_key_name"]}" '
            f"--max-image-dimension {config.get('max_image_dimension', 1536)} "
            f"--temperature {config.get('temperature', 0.0)} "
            f"--top-p {config.get('top_p', 1.0)} "
            f"--timeout-seconds {config.get('timeout_seconds', 180.0)} "
            f"--max-retries {config.get('max_retries', 3)} "
            f"--max-workers {config.get('max_workers', 8)}"
        )
        if config.get("api_base"):
            command += f' --api-base "{config["api_base"]}"'
        if config.get("reasoning_effort"):
            command += f' --reasoning-effort "{config["reasoning_effort"]}"'

        console.status("Running HopChain LLM judge")
        console.detail("Input file", config["input_file"])
        console.detail("Output file", config["output_file"])
        console.detail("Prompt file", prompt_file)
        console.detail("Judge", config["judge_name"])
        console.detail("Provider", config["provider"])
        console.detail("Model", config["model"])
        if config.get("reasoning_effort"):
            console.detail("Reasoning effort", config["reasoning_effort"])

        run_cmd(
            ctx=wrap_arguments(""),
            cluster=cluster,
            config_dir=config.get("cluster_config_dir"),
            command=command,
            container="nemo-skills",
            expname=expname,
            partition=resolve_partition(config, cluster, cpu=True),
            time_min=str(config.get("time_min", 120)),
            num_nodes=1,
            num_tasks=1,
            run_after=run_after,
            log_dir=str(output_dir / "logs"),
        )
        console.success("HopChain LLM judge job submitted")

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate stage configuration."""
        required_fields = [
            "input_file",
            "output_file",
            "output_dir",
            "summary_file",
            "project_root",
            "prompt_file",
            "judge_name",
            "provider",
            "model",
            "api_key_name",
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
