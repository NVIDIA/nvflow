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
"""Render final HopChain records into HTML review pages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nvflow.core import BaseStage, StageRegistry, console
from nvflow.recipes.multimodal.utils.runtime_env import (
    resolve_partition,
)


@StageRegistry.register(
    recipe="multimodal", workflow="hopchain_sdg", stage="visualize_final_hopchain_data"
)
@StageRegistry.register(
    recipe="multimodal", workflow="hopchain_sdg", stage="visualize_candidate_hopchain_data"
)
@StageRegistry.register(
    recipe="multimodal", workflow="hopchain_sdg", stage="visualize_reconciled_hopchain_data"
)
class VisualizeFinalHopchainDataStage(BaseStage):
    """Render paginated HTML previews for final HopChain data."""

    workflow = "hopchain_sdg"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute HTML visualization rendering."""

        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        project_root = config["project_root"]
        script_path = (
            f"{project_root}/nvflow/recipes/multimodal/utils/render_hopchain_final_dataset_html.py"
        )
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_file = config["summary_file"]

        command = (
            f'PYTHONPATH="{project_root}:$PYTHONPATH" '
            f'python3 "{script_path}" '
            f'--queries-input "{config["queries_input_file"]}" '
            f'--combinations-input "{config["combinations_input_file"]}" '
            f'--output-dir "{output_dir}" '
            f'--summary "{summary_file}" '
            f"--rows-per-file {config.get('rows_per_file', 100)} "
            f"--image-max-dimension {config.get('image_max_dimension', 1024)} "
            f'--title "{config.get("title", "HopChain Synthetic Data Review")}"'
        )
        if config.get("sample_count") is not None:
            command += f" --sample-count {config['sample_count']}"
            command += f" --sample-seed {config.get('sample_seed', 42)}"

        console.status("Rendering HopChain synthetic data HTML review")
        console.detail("Queries input", config["queries_input_file"])
        console.detail("Combinations input", config["combinations_input_file"])
        console.detail("Output dir", str(output_dir))

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
        console.success("Final HopChain HTML review job submitted")

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate stage configuration."""
        required_fields = [
            "queries_input_file",
            "combinations_input_file",
            "output_dir",
            "summary_file",
            "project_root",
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
