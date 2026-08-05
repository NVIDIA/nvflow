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
"""Generate and filter HopChain SFT reasoning traces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nvflow.core import BaseStage, StageRegistry, console
from nvflow.recipes.multimodal.utils.inference import (
    ImageEncodingConfig,
    build_generate_kwargs,
    build_inference_ctx_args,
    resolve_project_path,
)
from nvflow.recipes.multimodal.utils.multimodal_model_configs import resolve_model_config
from nvflow.recipes.multimodal.utils.runtime_env import (
    resolve_partition,
)


def _submit_preprocess_command(
    *,
    config: dict[str, Any],
    cluster: str,
    expname: str,
    run_after: list[str] | None,
    command: str,
    output_file: str,
    label: str,
) -> None:
    """Submit one CPU preprocessing command as its own NVFlow stage."""
    from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    console.status(label)
    console.detail("Output file", output_file)
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
        log_dir=str(output_path.parent / "logs"),
        reuse_code=config.get("reuse_code", True),
    )


@StageRegistry.register(
    recipe="multimodal",
    workflow="hopchain_sdg",
    stage="preprocess_generate_sft_reasoning_traces",
)
class PreprocessGenerateSFTReasoningTracesStage(BaseStage):
    """Build image+question teacher prompts for SFT trace generation."""

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
            f'"{project_root}/nvflow/recipes/multimodal/utils/hopchain_sft_trace_generation_preprocess.py" '
            f'--input "{input_file}" '
            f'--output "{output_file}" '
            f'--prompt "{prompt_file}" '
            f"--k {config.get('k', 3)}"
        )
        command += image_encoding.get_preprocess_args()

        _submit_preprocess_command(
            config=config,
            cluster=cluster,
            expname=expname,
            run_after=run_after,
            command=command,
            output_file=output_file,
            label="Preprocessing HopChain SFT trace generation prompts",
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
    recipe="multimodal", workflow="hopchain_sdg", stage="generate_sft_reasoning_traces"
)
class GenerateSFTReasoningTracesStage(BaseStage):
    """Generate k natural teacher reasoning traces per kept HopChain query."""

    workflow = "hopchain_sdg"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Submit Qwen trace generation on GPU."""
        from nemo_skills.pipeline.cli import generate

        project_root = config["project_root"]
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        model_config = resolve_model_config(config)
        preprocessed_input = config["preprocessed_input_file"]
        raw_output = output_dir / "output.jsonl"
        final_output = config.get("output_file") or str(output_dir / "final_result.jsonl")
        incorrect_output = config.get("incorrect_output_file") or str(
            output_dir / "incorrect_answers.jsonl"
        )
        summary_file = config["summary_file"]
        postprocess_cmd = (
            f'PYTHONPATH="{project_root}:$PYTHONPATH" python3 '
            f'"{project_root}/nvflow/recipes/multimodal/utils/hopchain_sft_trace_generation_postprocess.py" '
            f'--input "{raw_output}" '
            f'--output "{final_output}" '
            f'--incorrect-output "{incorrect_output}" '
            f'--summary "{summary_file}" '
            f'--model "{model_config.model}"'
        )

        ctx_args = build_inference_ctx_args(
            temperature=config.get("temperature", 0.6),
            top_p=config.get("top_p", 0.95),
            tokens_to_generate=config.get("tokens_to_generate", model_config.tokens_to_generate),
            max_concurrent_requests=config.get(
                "max_concurrent_requests", model_config.max_concurrent_requests
            ),
        )

        console.status("Running HopChain SFT trace generation")
        console.detail("Preprocessed input file", preprocessed_input)
        console.detail("Output dir", str(output_dir))
        console.detail("Model", model_config.model)

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
        generate_kwargs["reuse_code"] = config.get("reuse_code", True)
        generate(**generate_kwargs)
        console.success("HopChain SFT trace generation jobs submitted")

    def validate_config(self, config: dict[str, Any]) -> None:
        required_fields = [
            "preprocessed_input_file",
            "output_dir",
            "output_file",
            "incorrect_output_file",
            "summary_file",
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
    stage="preprocess_filter_sft_reasoning_traces",
)
class PreprocessFilterSFTReasoningTracesStage(BaseStage):
    """Build synthetic-hop judge prompts for correct SFT trace candidates."""

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
        command = (
            f'PYTHONPATH="{project_root}:$PYTHONPATH" python3 '
            f'"{project_root}/nvflow/recipes/multimodal/utils/hopchain_sft_trace_judge_preprocess.py" '
            f'--input "{input_file}" '
            f'--output "{output_file}" '
            f'--prompt "{prompt_file}"'
        )

        _submit_preprocess_command(
            config=config,
            cluster=cluster,
            expname=expname,
            run_after=run_after,
            command=command,
            output_file=output_file,
            label="Preprocessing HopChain SFT trace judge prompts",
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


@StageRegistry.register(
    recipe="multimodal", workflow="hopchain_sdg", stage="filter_sft_reasoning_traces"
)
class FilterSFTReasoningTracesStage(BaseStage):
    """Run the synthetic-hop trace judge on GPU and select one SFT trace per query."""

    workflow = "hopchain_sdg"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Submit Qwen trace-judge generation on GPU."""
        from nemo_skills.pipeline.cli import generate

        project_root = config["project_root"]
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        model_config = resolve_model_config(config)
        input_file = resolve_project_path(config["input_file"], project_root)
        preprocessed_input = config["preprocessed_input_file"]
        raw_output = output_dir / "output.jsonl"
        summary_file = config["summary_file"]
        postprocess_cmd = (
            f'PYTHONPATH="{project_root}:$PYTHONPATH" python3 '
            f'"{project_root}/nvflow/recipes/multimodal/utils/hopchain_sft_trace_filter_postprocess.py" '
            f'--candidates "{input_file}" '
            f'--judge-output "{raw_output}" '
            f'--output-dir "{output_dir}" '
            f'--summary "{summary_file}"'
        )

        ctx_args = build_inference_ctx_args(
            temperature=config.get("temperature", 0.0),
            top_p=config.get("top_p", 1.0),
            tokens_to_generate=config.get("tokens_to_generate", model_config.tokens_to_generate),
            max_concurrent_requests=config.get(
                "max_concurrent_requests", model_config.max_concurrent_requests
            ),
        )

        console.status("Running HopChain SFT trace judge on GPU")
        console.detail("Candidate file", input_file)
        console.detail("Preprocessed input file", preprocessed_input)
        console.detail("Output dir", str(output_dir))
        console.detail("Model", model_config.model)

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
        generate_kwargs["reuse_code"] = config.get("reuse_code", True)
        generate(**generate_kwargs)
        console.success("HopChain SFT trace filtering jobs submitted")

    def validate_config(self, config: dict[str, Any]) -> None:
        required_fields = [
            "input_file",
            "preprocessed_input_file",
            "output_dir",
            "summary_file",
            "project_root",
            "cluster_config_dir",
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        resolve_model_config(config)
