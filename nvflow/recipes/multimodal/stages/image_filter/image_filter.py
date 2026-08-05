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
"""HopChain image filtration stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nvflow.core import BaseStage, StageRegistry, console
from nvflow.recipes.multimodal.utils.image_filter_models import ImageDirectoryCatalog
from nvflow.recipes.multimodal.utils.inference import (
    ImageEncodingConfig,
    build_generate_kwargs,
    build_inference_ctx_args,
    resolve_project_path,
)
from nvflow.recipes.multimodal.utils.multimodal_model_configs import resolve_model_config

LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[5]


@StageRegistry.register(recipe="multimodal", workflow="hopchain_image_filter", stage="image_filter")
class ImageFilterStage(BaseStage):
    """Filter candidate images with a multimodal model before query synthesis."""

    workflow = "hopchain_image_filter"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute image filtering via nemo-skills generate."""
        from nemo_skills.pipeline.cli import generate

        project_root = config["project_root"]
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        model_config = resolve_model_config(config)
        image_encoding = ImageEncodingConfig.from_model_config(model_config)

        temp_dir = output_dir / "temp"
        input_catalog_file = self._resolve_input_catalog(config, output_dir=temp_dir)
        prompt_file = resolve_project_path(config["prompt_file"], project_root)
        preprocessed_input = temp_dir / "input_openai_format.jsonl"
        image_catalog = output_dir / "image_catalog.jsonl"
        raw_output = output_dir / "output.jsonl"
        final_output = output_dir / "final_output.jsonl"

        preprocess_cmd = (
            f'PYTHONPATH="{project_root}:$PYTHONPATH" python3 '
            "-m nvflow.recipes.multimodal.utils.image_filter_preprocess "
            f'--input-catalog "{input_catalog_file}" '
            f'--output "{preprocessed_input}" '
            f'--catalog-output "{image_catalog}" '
            f'--prompt "{prompt_file}"'
        )
        preprocess_cmd += image_encoding.get_preprocess_args()

        postprocess_cmd = (
            f'PYTHONPATH="{project_root}:$PYTHONPATH" python3 '
            "-m nvflow.recipes.multimodal.utils.image_filter_postprocess "
            f'--input "{raw_output}" '
            f'--output "{final_output}" '
            f"--min-complexity-score {config.get('min_complexity_score', 4)}"
        )
        for rating in config.get("allowed_quality_ratings", ["High", "Medium"]):
            postprocess_cmd += f' --allowed-quality-rating "{rating}"'

        tokens_to_generate = config.get("tokens_to_generate", model_config.tokens_to_generate)
        max_concurrent_requests = config.get(
            "max_concurrent_requests", model_config.max_concurrent_requests
        )
        temperature = config.get("temperature", 0.0)
        top_p = config.get("top_p", 1.0)

        ctx_args = build_inference_ctx_args(
            temperature=temperature,
            top_p=top_p,
            tokens_to_generate=tokens_to_generate,
            max_concurrent_requests=max_concurrent_requests,
        )

        console.status("Running HopChain image filtering")
        console.detail("Input catalog", input_catalog_file)
        console.detail("Prompt file", config["prompt_file"])
        console.detail("Output dir", str(output_dir))
        console.detail("Model", model_config.model)
        console.detail("Backend", model_config.server_type)
        console.detail(
            "GPUs",
            f"{model_config.server_nodes} node(s) x {model_config.server_gpus} GPUs",
        )
        console.detail("Image encoding", image_encoding.description)
        console.detail("Max tokens", str(tokens_to_generate))
        console.detail("Temperature", str(temperature))
        console.detail("Top P", str(top_p))
        console.detail("Num chunks", str(config.get("num_chunks", 1)))
        console.detail(
            "Keep quality ratings",
            ", ".join(config.get("allowed_quality_ratings", ["High", "Medium"])),
        )
        console.detail("Min complexity score", str(config.get("min_complexity_score", 4)))
        console.blank()
        console.status("Submitting job to cluster")

        generate_kwargs = build_generate_kwargs(
            ctx_args=ctx_args,
            cluster=cluster,
            model_config=model_config,
            output_dir=str(output_dir),
            input_file=str(preprocessed_input),
            expname=expname,
            preprocess_cmd=preprocess_cmd,
            postprocess_cmd=postprocess_cmd,
            stage_config=config,
            run_after=run_after,
        )
        generate(**generate_kwargs)

        console.blank()
        console.success("Image filter job submitted")
        console.detail("Resolved image catalog", str(image_catalog))
        console.detail("Raw model output", str(raw_output))
        console.detail("Final output", str(final_output))

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate required fields for the image filter stage."""
        required_fields = [
            "output_dir",
            "project_root",
            "prompt_file",
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

        has_catalog_file = bool(config.get("input_catalog_file"))
        has_inline_directories = bool(config.get("image_directories"))
        if has_catalog_file == has_inline_directories:
            raise ValueError("Configure exactly one of input_catalog_file or image_directories")

        resolve_model_config(config)

        if not 1 <= config.get("min_complexity_score", 4) <= 10:
            raise ValueError("min_complexity_score must be between 1 and 10")

        if "num_chunks" in config and config["num_chunks"] <= 0:
            raise ValueError("num_chunks must be greater than 0 when provided")

        allowed_quality_ratings = config.get("allowed_quality_ratings", ["High", "Medium"])
        if not allowed_quality_ratings:
            raise ValueError("allowed_quality_ratings must contain at least one rating")

        if has_catalog_file:
            input_catalog_path = self._resolve_local_path(config["input_catalog_file"])
            if not input_catalog_path.exists():
                if Path(config["input_catalog_file"]).is_absolute():
                    console.warning(
                        "Cannot validate absolute catalog path locally: "
                        f"{config['input_catalog_file']}"
                    )
                else:
                    raise ValueError(f"Input catalog file does not exist: {input_catalog_path}")
        else:
            catalog = ImageDirectoryCatalog.model_validate(config["image_directories"])
            for spec in catalog.root:
                directory = self._resolve_local_path(spec.directory)
                if not directory.is_dir():
                    raise ValueError(f"Image directory does not exist: {directory}")

        prompt_file_path = self._resolve_local_path(config["prompt_file"])
        if not prompt_file_path.exists():
            if Path(config["prompt_file"]).is_absolute():
                console.warning(
                    f"Cannot validate absolute prompt path locally: {config['prompt_file']}"
                )
            else:
                raise ValueError(f"Prompt file does not exist: {prompt_file_path}")

    def _resolve_local_path(self, path_value: str) -> Path:
        """Resolve a config path for local validation."""
        path = Path(path_value)
        if path.is_absolute():
            return path
        return LOCAL_PROJECT_ROOT / path

    def _resolve_input_catalog(self, config: dict[str, Any], *, output_dir: Path) -> str:
        """Return the configured catalog path, materializing inline YAML when needed."""
        if config.get("input_catalog_file"):
            return resolve_project_path(config["input_catalog_file"], config["project_root"])

        catalog = ImageDirectoryCatalog.model_validate(config["image_directories"])
        resolved_entries: list[dict[str, Any]] = []
        for spec in catalog.root:
            values = spec.model_dump()
            values["directory"] = resolve_project_path(spec.directory, config["project_root"])
            resolved_entries.append(values)

        output_dir.mkdir(parents=True, exist_ok=True)
        catalog_path = output_dir / "image_dir_catalog.json"
        catalog_path.write_text(json.dumps(resolved_entries, indent=2))
        return str(catalog_path)
