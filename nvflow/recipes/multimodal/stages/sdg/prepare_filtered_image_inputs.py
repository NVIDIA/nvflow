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
"""Prepare filtered image inputs for HopChain SDG."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nvflow.core import BaseStage, StageRegistry, console
from nvflow.recipes.multimodal.utils.runtime_env import (
    resolve_partition,
)


@StageRegistry.register(
    recipe="multimodal", workflow="hopchain_sdg", stage="prepare_filtered_image_inputs"
)
class PrepareFilteredImageInputsStage(BaseStage):
    """Normalize image-filter outputs into SDG input records."""

    workflow = "hopchain_sdg"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute filtered-image handoff normalization."""
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        input_file = config["input_file"]
        output_file = config["output_file"]
        summary_file = config["summary_file"]
        output_dir = Path(output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        project_root = config["project_root"]
        script_path = (
            f"{project_root}/nvflow/recipes/multimodal/utils/prepare_filtered_image_inputs.py"
        )
        command = (
            f'PYTHONPATH="{project_root}:$PYTHONPATH" '
            f'python3 "{script_path}" '
            f'--input "{input_file}" --output "{output_file}" --summary "{summary_file}"'
        )
        if config.get("sample_count_per_domain") is not None:
            command += (
                f" --sample-count-per-domain {config['sample_count_per_domain']}"
                f" --sample-seed {config.get('sample_seed', 42)}"
            )
        elif config.get("sample_count") is not None:
            command += f" --sample-count {config['sample_count']} --sample-seed {config.get('sample_seed', 42)}"

        console.status("Preparing filtered image inputs")
        console.detail("Input file", input_file)
        console.detail("Output file", output_file)
        console.detail("Summary file", summary_file)
        if config.get("sample_count_per_domain") is not None:
            console.detail("Sample count per domain", str(config["sample_count_per_domain"]))
            console.detail("Sample seed", str(config.get("sample_seed", 42)))
        elif config.get("sample_count") is not None:
            console.detail("Sample count", str(config["sample_count"]))
            console.detail("Sample seed", str(config.get("sample_seed", 42)))

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

        console.success("Filtered image input preparation job submitted")

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate stage configuration."""
        required_fields = [
            "input_file",
            "output_file",
            "summary_file",
            "project_root",
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
