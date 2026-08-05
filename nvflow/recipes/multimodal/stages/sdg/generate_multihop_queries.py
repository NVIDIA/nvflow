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
"""HopChain multi-hop query generation stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nvflow.core import BaseStage, StageRegistry, console
from nvflow.recipes.multimodal.utils.inference import (
    build_generate_kwargs,
    build_inference_ctx_args,
    resolve_project_path,
)
from nvflow.recipes.multimodal.utils.multimodal_model_configs import resolve_model_config


@StageRegistry.register(
    recipe="multimodal", workflow="hopchain_sdg", stage="generate_multihop_queries"
)
class GenerateMultihopQueriesStage(BaseStage):
    """Generate paper-style HopChain multi-hop candidate queries."""

    workflow = "hopchain_sdg"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Submit query generation via nemo-skills generate."""
        from nemo_skills.pipeline.cli import generate

        project_root = config["project_root"]
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        model_config = resolve_model_config(config)

        input_file = resolve_project_path(config["input_file"], project_root)
        prompt_file = resolve_project_path(config["prompt_file"], project_root)
        preprocessed_input = config.get("preprocessed_input_file")
        if preprocessed_input is None:
            preprocessed_input = str(output_dir / "temp" / "input_openai_format.jsonl")
        raw_output = output_dir / "output.jsonl"
        final_output = output_dir / "final_output.jsonl"
        summary_file = output_dir / "summary.json"

        postprocess_cmd = (
            f'PYTHONPATH="{project_root}:$PYTHONPATH" python3 '
            "-m nvflow.recipes.multimodal.utils.hopchain_query_generation_postprocess "
            f'--input "{raw_output}" --output "{final_output}" --summary "{summary_file}"'
        )

        ctx_args = build_inference_ctx_args(
            temperature=config.get("temperature", 0.0),
            top_p=config.get("top_p", 1.0),
            tokens_to_generate=config.get("tokens_to_generate", model_config.tokens_to_generate),
            max_concurrent_requests=config.get(
                "max_concurrent_requests", model_config.max_concurrent_requests
            ),
        )

        console.status("Running HopChain query generation")
        console.detail("Input file", input_file)
        console.detail("Preprocessed input file", preprocessed_input)
        console.detail("Prompt file", prompt_file)
        console.detail("Output dir", str(output_dir))

        generate_kwargs = build_generate_kwargs(
            ctx_args=ctx_args,
            cluster=cluster,
            model_config=model_config,
            output_dir=str(output_dir),
            input_file=str(preprocessed_input),
            expname=expname,
            preprocess_cmd=None,
            postprocess_cmd=postprocess_cmd,
            stage_config=config,
            run_after=run_after,
        )
        generate(**generate_kwargs)
        console.success("HopChain query generation job submitted")

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate stage configuration."""
        required_fields = [
            "input_file",
            "output_dir",
            "preprocessed_input_file",
            "project_root",
            "prompt_file",
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        resolve_model_config(config)
