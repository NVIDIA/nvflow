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
"""Sample instance combinations for HopChain query generation."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from nvflow.core import BaseStage, StageRegistry, console
from nvflow.recipes.multimodal.utils.runtime_env import (
    resolve_partition,
)


@StageRegistry.register(
    recipe="multimodal", workflow="hopchain_sdg", stage="sample_instance_combinations"
)
class SampleInstanceCombinationsStage(BaseStage):
    """Sample deterministic instance combinations for SDG query generation."""

    workflow = "hopchain_sdg"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute combination sampling."""
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        project_root = config["project_root"]
        script_path = f"{project_root}/nvflow/recipes/multimodal/utils/hopchain_sample_instance_combinations.py"
        output_file = config["output_file"]
        summary_file = config["summary_file"]
        output_dir = Path(output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        command = (
            f'PYTHONPATH="{project_root}:$PYTHONPATH" '
            f'python3 "{script_path}" '
            f'--input "{config["input_file"]}" --output "{output_file}" --summary "{summary_file}" '
            f"--min-instances {config.get('min_instances', 3)} "
            f"--max-instances {config.get('max_instances', 6)} "
            f'--selection-strategy "{config.get("selection_strategy", "balanced_by_category_then_size")}" '
            f"--max-instances-considered-per-image {config.get('max_instances_considered_per_image', 12)} "
            f"--max-instances-per-category {config.get('max_instances_per_category', 2)} "
            f"--max-combinations-per-image {config.get('max_combinations_per_image', 1)} "
            f'--combination-size-strategy "{config.get("combination_size_strategy", "largest_first")}" '
            f"--sampling-seed {config.get('sampling_seed', 42)}"
        )
        if config.get("combination_size_weights"):
            command += (
                " --combination-size-weights "
                f"{shlex.quote(json.dumps(config['combination_size_weights'], separators=(',', ':')))}"
            )
        if config.get("debug_copy_selected_images", False):
            command += " --debug-copy-selected-images"
            debug_selected_images_dir = config.get("debug_selected_images_dir")
            if debug_selected_images_dir:
                command += f' --debug-selected-images-dir "{debug_selected_images_dir}"'
        if config.get("area_confidence_area_weight") is not None:
            command += f" --area-confidence-area-weight {config['area_confidence_area_weight']}"
        if config.get("area_confidence_confidence_weight") is not None:
            command += f" --area-confidence-confidence-weight {config['area_confidence_confidence_weight']}"
        if config.get("min_confidence_threshold") is not None:
            command += f" --min-confidence-threshold {config['min_confidence_threshold']}"
        if config.get("null_confidence_fallback") is not None:
            command += f" --null-confidence-fallback {config['null_confidence_fallback']}"
        if config.get("iou_dedup_threshold") is not None:
            threshold = config["iou_dedup_threshold"]
            if not 0.0 < threshold <= 1.0:
                raise ValueError("iou_dedup_threshold must be in the range (0, 1]")
            command += f" --iou-dedup-threshold {threshold}"
        if config.get("iou_dedup_candidate_pool_size") is not None:
            pool_size = config["iou_dedup_candidate_pool_size"]
            final_size = config.get("max_instances_considered_per_image", 12)
            if pool_size < final_size:
                raise ValueError(
                    "iou_dedup_candidate_pool_size must be >= max_instances_considered_per_image"
                )
            command += f" --iou-dedup-candidate-pool-size {pool_size}"

        console.status("Sampling instance combinations")
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
        console.success("Instance combination sampling job submitted")

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
