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

from nvflow.core import console
from nvflow.execution import ArtifactDeclaration, CapturedExecutionStage, declare_task_outputs
from nvhopchain.utils.inference import (
    ImageEncodingConfig,
    build_generate_kwargs,
    build_inference_ctx_args,
    resolve_project_path,
)
from nvhopchain.utils.multimodal_model_configs import resolve_model_config


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
    from nvflow.integrations.nemo_skills.commands import run_cmd, wrap_arguments

    output_path = Path(output_file)

    console.status(label)
    console.detail("Output file", output_file)
    with declare_task_outputs(output_file=output_file):
        run_cmd(
            ctx=wrap_arguments(""),
            cluster=cluster,
            config_dir=config.get("cluster_config_dir"),
            command=command,
            container="nemo-skills",
            expname=expname,
            partition=config.get("partition"),
            time_min=str(config.get("time_min", 120)),
            num_nodes=1,
            num_tasks=1,
            run_after=run_after,
            log_dir=str(output_path.parent / "logs"),
            check_mounted_paths=True,
            reuse_code=config.get("reuse_code", True),
        )


class PreprocessGenerateSFTReasoningTracesStage(CapturedExecutionStage):
    """Build image+question teacher prompts for SFT trace generation."""

    capture_strategy = "run_cmd"

    def build_tasks(
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
            f'"{project_root}/projects/hopchain/nvhopchain/utils/hopchain_sft_trace_generation_preprocess.py" '
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


class GenerateSFTReasoningTracesStage(CapturedExecutionStage):
    """Generate k natural teacher reasoning traces per kept HopChain query."""

    capture_strategy = "run_cmd"

    def build_tasks(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Submit Qwen trace generation on GPU."""
        from nvflow.integrations.nemo_skills.commands import generate

        project_root = config["project_root"]
        output_dir = Path(config["output_dir"])

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
            f'"{project_root}/projects/hopchain/nvhopchain/utils/hopchain_sft_trace_generation_postprocess.py" '
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
        with declare_task_outputs(
            ArtifactDeclaration(name="reasoning-traces", path=str(final_output)),
            ArtifactDeclaration(name="incorrect-traces", path=str(incorrect_output)),
            ArtifactDeclaration(name="summary", path=str(summary_file)),
            output_dir=output_dir,
        ):
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


class PreprocessFilterSFTReasoningTracesStage(CapturedExecutionStage):
    """Build synthetic-hop judge prompts for correct SFT trace candidates."""

    capture_strategy = "run_cmd"

    def build_tasks(
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
            f'"{project_root}/projects/hopchain/nvhopchain/utils/hopchain_sft_trace_judge_preprocess.py" '
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


class FilterSFTReasoningTracesStage(CapturedExecutionStage):
    """Run the synthetic-hop trace judge on GPU and select one SFT trace per query."""

    capture_strategy = "run_cmd"

    def build_tasks(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Submit Qwen trace-judge generation on GPU."""
        from nvflow.integrations.nemo_skills.commands import generate

        project_root = config["project_root"]
        output_dir = Path(config["output_dir"])

        model_config = resolve_model_config(config)
        input_file = resolve_project_path(config["input_file"], project_root)
        preprocessed_input = config["preprocessed_input_file"]
        raw_output = output_dir / "output.jsonl"
        summary_file = config["summary_file"]
        postprocess_cmd = (
            f'PYTHONPATH="{project_root}:$PYTHONPATH" python3 '
            f'"{project_root}/projects/hopchain/nvhopchain/utils/hopchain_sft_trace_filter_postprocess.py" '
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
        with declare_task_outputs(
            ArtifactDeclaration(
                name="filter-results",
                path=str(output_dir / "final_output.jsonl"),
            ),
            ArtifactDeclaration(
                name="kept-traces",
                path=str(output_dir / "kept_output.jsonl"),
            ),
            ArtifactDeclaration(
                name="sft-dataset",
                path=str(output_dir / "sft_output.jsonl"),
            ),
            ArtifactDeclaration(name="summary", path=str(summary_file)),
            output_dir=output_dir,
        ):
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
