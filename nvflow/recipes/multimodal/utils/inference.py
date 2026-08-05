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
"""Inference helpers for multimodal nemo-skills stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from nvflow.recipes.multimodal.utils.image_filter_models import VisionModelConfig
from nvflow.recipes.multimodal.utils.runtime_env import resolve_partition


class ImageEncodingConfig(BaseModel):
    """Image transport settings derived from a model configuration."""

    use_base64_images: bool = Field(default=False)
    max_image_dimension: int | None = Field(default=None)

    @classmethod
    def from_model_config(
        cls,
        model_config: VisionModelConfig,
        max_image_dimension: int | None = None,
    ) -> ImageEncodingConfig:
        """Extract image encoding settings from a typed model config."""
        return cls(
            use_base64_images=model_config.use_base64_images,
            max_image_dimension=max_image_dimension
            if max_image_dimension is not None
            else model_config.max_image_dimension,
        )

    def get_preprocess_args(self) -> str:
        """Build CLI arguments for preprocess commands."""
        if not self.use_base64_images:
            return ""
        args = " --use-base64"
        if self.max_image_dimension:
            args += f" --max-image-dimension {self.max_image_dimension}"
        return args

    @property
    def description(self) -> str:
        """Human-readable encoding mode."""
        return (
            "base64 data URIs" if self.use_base64_images else "local paths encoded by NeMo-Skills"
        )


def build_inference_ctx_args(
    *,
    temperature: float,
    top_p: float,
    tokens_to_generate: int,
    max_concurrent_requests: int,
) -> str:
    """Build nemo-skills inference args for OpenAI-formatted multimodal prompts."""
    return (
        f"++inference.temperature={temperature} "
        f"++inference.top_p={top_p} "
        f"++inference.tokens_to_generate={tokens_to_generate} "
        f"++max_concurrent_requests={max_concurrent_requests} "
        "++prompt_format=openai "
    )


def build_generate_kwargs(
    *,
    ctx_args: str,
    cluster: str,
    model_config: VisionModelConfig,
    output_dir: str,
    input_file: str,
    expname: str,
    preprocess_cmd: str | None,
    postprocess_cmd: str,
    stage_config: dict[str, Any],
    run_after: list[str] | None = None,
) -> dict[str, Any]:
    """Build kwargs for `nemo_skills.pipeline.cli.generate()`."""
    from nemo_skills.pipeline.cli import wrap_arguments

    generate_kwargs: dict[str, Any] = {
        "ctx": wrap_arguments(ctx_args),
        "cluster": cluster,
        "model": model_config.model,
        "server_type": model_config.server_type,
        "server_gpus": model_config.server_gpus,
        "server_nodes": model_config.server_nodes,
        "server_args": model_config.server_args,
        "output_dir": output_dir,
        "input_file": input_file,
        "partition": resolve_partition(stage_config, cluster),
        "time_min": stage_config.get("time_min", model_config.time_min),
        "expname": expname,
        "preprocess_cmd": preprocess_cmd,
        "postprocess_cmd": postprocess_cmd,
        "sbatch_kwargs": {"retries": 3},
    }

    cluster_config_dir = stage_config.get("cluster_config_dir")
    if cluster_config_dir:
        generate_kwargs["config_dir"] = cluster_config_dir

    if run_after:
        generate_kwargs["run_after"] = run_after
    if stage_config.get("num_chunks"):
        generate_kwargs["num_chunks"] = stage_config["num_chunks"]

    return generate_kwargs


def resolve_project_path(path_value: str, project_root: str) -> str:
    """Resolve a repo-relative path against the configured project root."""
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(Path(project_root) / path)
