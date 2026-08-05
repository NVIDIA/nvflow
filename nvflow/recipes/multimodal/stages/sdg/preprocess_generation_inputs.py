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
"""Preprocess HopChain inputs before chunked nemo-skills generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nvflow.core import BaseStage, StageRegistry, console
from nvflow.recipes.multimodal.utils.inference import ImageEncodingConfig, resolve_project_path
from nvflow.recipes.multimodal.utils.multimodal_model_configs import resolve_model_config
from nvflow.recipes.multimodal.utils.runtime_env import (
    resolve_partition,
)


def _run_preprocess_command(
    *,
    config: dict[str, Any],
    cluster: str,
    expname: str,
    run_after: list[str] | None,
    command: str,
    input_file: str,
    output_file: str,
    label: str,
) -> None:
    """Submit one CPU preprocessing command before chunked generation."""
    from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    console.status(label)
    console.detail("Input file", input_file)
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
        log_dir=str(output_path.parent / "logs"),
    )

    console.success("Preprocessing job submitted")


@StageRegistry.register(
    recipe="multimodal",
    workflow="hopchain_sdg",
    stage="preprocess_identify_categories",
)
class PreprocessIdentifyCategoriesStage(BaseStage):
    """Build OpenAI-format category-identification prompts once before chunking."""

    workflow = "hopchain_sdg"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        project_root = config["project_root"]
        model_config = resolve_model_config(config)
        image_encoding = ImageEncodingConfig.from_model_config(
            model_config,
            max_image_dimension=config.get("max_image_dimension"),
        )

        input_file = resolve_project_path(config["input_file"], project_root)
        output_file = config["output_file"]
        prompt_file = resolve_project_path(config["prompt_file"], project_root)
        command = (
            f'PYTHONPATH="{project_root}:$PYTHONPATH" python3 '
            "-m nvflow.recipes.multimodal.utils.hopchain_category_identification_preprocess "
            f'--input "{input_file}" --output "{output_file}" --prompt "{prompt_file}"'
        )
        command += image_encoding.get_preprocess_args()

        _run_preprocess_command(
            config=config,
            cluster=cluster,
            expname=expname,
            run_after=run_after,
            command=command,
            input_file=input_file,
            output_file=output_file,
            label="Preprocessing category-identification prompts",
        )

    def validate_config(self, config: dict[str, Any]) -> None:
        required_fields = [
            "input_file",
            "output_file",
            "prompt_file",
            "project_root",
            "cluster_config_dir",
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        resolve_model_config(config)


@StageRegistry.register(
    recipe="multimodal",
    workflow="hopchain_sdg",
    stage="preprocess_generate_multihop_queries",
)
class PreprocessGenerateMultihopQueriesStage(BaseStage):
    """Build OpenAI-format query-generation prompts once before chunking."""

    workflow = "hopchain_sdg"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        project_root = config["project_root"]
        model_config = resolve_model_config(config)
        image_encoding = ImageEncodingConfig.from_model_config(model_config)

        input_file = resolve_project_path(config["input_file"], project_root)
        output_file = config["output_file"]
        prompt_file = resolve_project_path(config["prompt_file"], project_root)
        command = (
            f'PYTHONPATH="{project_root}:$PYTHONPATH" python3 '
            "-m nvflow.recipes.multimodal.utils.hopchain_query_generation_preprocess "
            f'--input "{input_file}" --output "{output_file}" --prompt "{prompt_file}" '
            f"--num-queries {config.get('num_queries', 1)} "
            f'--target-hop-count-info "{config.get("target_hop_count_info", "4-5 hops")}"'
        )
        command += image_encoding.get_preprocess_args()

        _run_preprocess_command(
            config=config,
            cluster=cluster,
            expname=expname,
            run_after=run_after,
            command=command,
            input_file=input_file,
            output_file=output_file,
            label="Preprocessing HopChain query-generation prompts",
        )

    def validate_config(self, config: dict[str, Any]) -> None:
        required_fields = [
            "input_file",
            "output_file",
            "prompt_file",
            "project_root",
            "cluster_config_dir",
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        resolve_model_config(config)


@StageRegistry.register(
    recipe="multimodal",
    workflow="hopchain_sdg",
    stage="preprocess_filter_easy_candidates",
)
class PreprocessFilterEasyCandidatesStage(BaseStage):
    """Build Omni difficulty-filter prompts once before chunking."""

    workflow = "hopchain_sdg"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        project_root = config["project_root"]
        input_file = resolve_project_path(config["input_file"], project_root)
        output_file = config["output_file"]
        prompt_file = resolve_project_path(config["prompt_file"], project_root)
        k = config.get("k", 5)

        command = (
            f'PYTHONPATH="{project_root}:$PYTHONPATH" python3 '
            "-m nvflow.recipes.multimodal.utils.hopchain_difficulty_filter_preprocess "
            f'--input "{input_file}" '
            f'--output "{output_file}" '
            f'--prompt "{prompt_file}" '
            f"--k {k}"
        )

        _run_preprocess_command(
            config=config,
            cluster=cluster,
            expname=expname,
            run_after=run_after,
            command=command,
            input_file=input_file,
            output_file=output_file,
            label="Preprocessing HopChain difficulty-filter prompts",
        )

    def validate_config(self, config: dict[str, Any]) -> None:
        required_fields = [
            "input_file",
            "output_file",
            "prompt_file",
            "project_root",
            "cluster_config_dir",
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
