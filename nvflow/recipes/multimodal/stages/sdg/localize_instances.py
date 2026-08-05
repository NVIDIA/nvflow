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
"""HopChain native SAM 3.1 instance localization stage."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from nvflow.core import BaseStage, StageRegistry, console
from nvflow.recipes.multimodal.utils.inference import resolve_project_path
from nvflow.recipes.multimodal.utils.runtime_env import resolve_partition


@StageRegistry.register(recipe="multimodal", workflow="hopchain_sdg", stage="localize_instances")
class LocalizeInstancesStage(BaseStage):
    """Localize identified object categories with native SAM 3.1."""

    workflow = "hopchain_sdg"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Submit one SAM 3.1 job per configured chunk, followed by a merge."""
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        project_root = config["project_root"]
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        input_file = resolve_project_path(config["input_file"], project_root)
        prompt_file = resolve_project_path(config["prompt_file"], project_root)
        output_file = output_dir / "final_output.jsonl"
        summary_file = output_dir / "summary.json"
        crop_dir = output_dir / "crops"
        log_dir = output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        script_path = (
            f"{project_root}/nvflow/recipes/multimodal/utils/hopchain_localize_instances_sam3.py"
        )

        def build_command(
            *,
            chunk_output_file: Path,
            chunk_summary_file: Path,
            shard_index: int | None = None,
            num_shards: int | None = None,
        ) -> str:
            command = (
                f'PYTHONPATH="{project_root}:$PYTHONPATH" python3 "{script_path}" '
                f'--input "{input_file}" --output "{chunk_output_file}" '
                f'--summary "{chunk_summary_file}" --crop-dir "{crop_dir}" '
                f'--prompt "{prompt_file}" --model "{config["model"]}" '
                f"--threshold {config.get('threshold', 0.5)} "
                f"--dataloader-num-workers {config.get('dataloader_num_workers', 4)} "
                f"--dataloader-prefetch-factor {config.get('dataloader_prefetch_factor', 2)} "
                f"--max-localization-phrases-per-category "
                f"{config.get('max_localization_phrases_per_category', 3)} "
                f"--prompt-alias-iou-dedup-threshold "
                f"{config.get('prompt_alias_iou_dedup_threshold', 0.9)}"
            )
            if config.get("filter_list"):
                command += f" --filter-list {shlex.quote(json.dumps(config['filter_list']))}"
            if shard_index is not None and num_shards is not None:
                command += f" --num-shards {num_shards} --shard-index {shard_index}"
            if config.get("debug_save_annotated_images", False):
                command += " --debug-save-annotated-images"
                if config.get("debug_annotated_images_dir"):
                    command += (
                        f' --debug-annotated-images-dir "{config["debug_annotated_images_dir"]}"'
                    )
            if config.get("max_image_dimension") is not None:
                command += f" --max-image-dimension {config['max_image_dimension']}"
            return command

        console.status("Running native SAM 3.1 localization")
        console.detail("Input file", input_file)
        console.detail("Prompt file", prompt_file)
        console.detail("Output dir", str(output_dir))
        console.detail("Model", config["model"])

        container = config["container"]
        num_chunks = int(config.get("num_chunks", 1) or 1)
        if num_chunks == 1:
            run_cmd(
                ctx=wrap_arguments(""),
                cluster=cluster,
                config_dir=config.get("cluster_config_dir"),
                command=build_command(
                    chunk_output_file=output_file,
                    chunk_summary_file=summary_file,
                ),
                container=container,
                expname=expname,
                partition=resolve_partition(config, cluster),
                time_min=str(config.get("time_min", 60)),
                num_nodes=1,
                num_tasks=1,
                num_gpus=int(config.get("num_gpus", 1)),
                run_after=run_after,
                log_dir=str(log_dir),
                sbatch_kwargs={"retries": 3},
            )
            console.success("SAM 3.1 localization job submitted")
            return

        chunk_dir = output_dir / "temp" / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_expnames = []
        for chunk_idx in range(num_chunks):
            chunk_expname = f"{expname}-chunk{chunk_idx}"
            chunk_expnames.append(chunk_expname)
            run_cmd(
                ctx=wrap_arguments(""),
                cluster=cluster,
                config_dir=config.get("cluster_config_dir"),
                command=build_command(
                    chunk_output_file=chunk_dir / f"final_output.part-{chunk_idx:05d}.jsonl",
                    chunk_summary_file=chunk_dir / f"summary.part-{chunk_idx:05d}.json",
                    shard_index=chunk_idx,
                    num_shards=num_chunks,
                ),
                container=container,
                expname=chunk_expname,
                partition=resolve_partition(config, cluster),
                time_min=str(config.get("time_min", 60)),
                num_nodes=1,
                num_tasks=1,
                num_gpus=int(config.get("num_gpus", 1)),
                run_after=run_after,
                log_dir=str(log_dir),
                sbatch_kwargs={"retries": 3},
            )

        merge_command = (
            f'PYTHONPATH="{project_root}:$PYTHONPATH" python3 '
            f'"{project_root}/nvflow/recipes/multimodal/utils/'
            'hopchain_merge_localization_shards.py" '
            f'--inputs "{chunk_dir}/final_output.part-*.jsonl" '
            f'--summaries "{chunk_dir}/summary.part-*.json" '
            f'--output "{output_file}" --summary "{summary_file}"'
        )
        run_cmd(
            ctx=wrap_arguments(""),
            cluster=cluster,
            config_dir=config.get("cluster_config_dir"),
            command=merge_command,
            container="nemo-skills",
            expname=expname,
            partition=resolve_partition(
                config,
                cluster,
                cpu=True,
                override_key="merge_partition",
            ),
            time_min=str(config.get("merge_time_min", 30)),
            num_nodes=1,
            num_tasks=1,
            run_after=chunk_expnames,
            log_dir=str(log_dir),
        )
        console.success("SAM 3.1 localization jobs submitted")

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate native SAM 3.1 stage configuration."""
        required_fields = [
            "input_file",
            "output_dir",
            "project_root",
            "cluster_config_dir",
            "prompt_file",
            "model",
            "container",
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        if int(config.get("num_chunks", 1) or 1) < 1:
            raise ValueError("num_chunks must be greater than zero")
