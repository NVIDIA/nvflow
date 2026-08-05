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
"""Transform raw SDG dataset to standard training format (shared: SFT + GRPO).

For GRPO workflows with ``environments``, runs per-environment: each
environment's ``raw_train_data`` is transformed and written to
``{output_dir}/{env_name}/``.

For SFT workflows (no ``environments``), runs once with ``input_files``
and ``output_file`` from config (SFT single-dataset mode).
"""

from typing import Any

from nvflow.core import BaseStage, StageRegistry, console
from nvflow.lib.cli_cmd import build_python_cmd


@StageRegistry.register(recipe="finance", workflow="sft", stage="data_transformation")
@StageRegistry.register(recipe="finance", workflow="grpo", stage="data_transformation")
class DataTransformationStage(BaseStage):
    """Transform raw SDG dataset to standard training format (shared: SFT + GRPO).

    Normalises field names to a model-agnostic schema:
      problem, context, reasoning_content, generation, uuid, question_type

    Generates uuid: Deterministic hash based on problem + generation content.
    Computes length statistics for key fields.
    """

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute data transformation (per-environment when environments present)."""
        environments = config.get("environments")
        if environments:
            self._execute_per_env(config, cluster, expname, run_after)
        else:
            self._execute_sft(config, cluster, expname, run_after)

    def _execute_per_env(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        from nvflow.lib.rl.helpers import resolve_environments

        environments = resolve_environments(config)
        base_output_dir = config["output_dir"]
        num_chunks = config.get("num_chunks", 1)
        # Filenames are YAML-driven (default preserves existing behaviour).
        input_filename = config.get("input_filename", "final_result.jsonl")
        output_filename = config.get("output_filename", "final_result.jsonl")

        for env_name, env_cfg in environments.items():
            raw_data = env_cfg.get("raw_train_data")
            if not raw_data:
                console.warning(f"Skipping environment '{env_name}': no raw_train_data configured")
                continue

            env_output_dir = f"{base_output_dir}/{env_name}"
            env_output_file = f"{env_output_dir}/{output_filename}"
            source_format = env_cfg.get("source_format", "separated")
            reasoning_mode = env_cfg.get("reasoning_mode", "none")

            console.status(f"Transforming dataset for environment: {env_name}")
            console.detail("Input", f"{raw_data}/{input_filename}")
            console.detail("Output", f"{env_output_dir}/chunks/ ({num_chunks} chunks)")
            console.detail("Source format", source_format)
            console.detail("Reasoning mode", reasoning_mode)
            console.blank()

            self._submit_transform_job(
                input_files=[f"{raw_data}/{input_filename}"],
                output_file=env_output_file,
                output_dir=env_output_dir,
                source_format=source_format,
                reasoning_mode=reasoning_mode,
                num_chunks=num_chunks,
                config=config,
                cluster=cluster,
                expname=f"{expname}-{env_name}",
                run_after=run_after,
            )

    def _execute_sft(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """SFT single-dataset mode."""
        input_files = config["input_files"]
        if isinstance(input_files, str):
            input_files = [input_files]

        output_file = config["output_file"]
        source_format = config.get("source_format", "separated")
        reasoning_mode = config.get("reasoning_mode", "none")
        num_chunks = config.get("num_chunks", 1)
        output_dir = config.get("output_dir", "/tmp")

        console.status("Transforming dataset to training format")
        console.detail("Input files", f"{len(input_files)} file(s)")
        for idx, f in enumerate(input_files, 1):
            console.detail(f"  File {idx}", f)
        console.detail("Output", f"{output_dir}/chunks/ ({num_chunks} chunks)")
        console.detail("Source format", source_format)
        console.detail("Reasoning mode", reasoning_mode)
        console.blank()

        self._submit_transform_job(
            input_files=input_files,
            output_file=output_file,
            output_dir=output_dir,
            source_format=source_format,
            reasoning_mode=reasoning_mode,
            num_chunks=num_chunks,
            config=config,
            cluster=cluster,
            expname=expname,
            run_after=run_after,
        )

    def _submit_transform_job(
        self,
        *,
        input_files: list[str],
        output_file: str,
        output_dir: str,
        source_format: str,
        reasoning_mode: str,
        num_chunks: int,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None,
    ) -> None:
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        # ``dataset_transformer`` takes one or more positional input file
        # paths followed by ``--output_file`` and other flag options.
        # ``build_python_cmd`` shlex-quotes both positional and flag
        # values -- this matters for the SFT path too, since SFT shares
        # this stage and any future config that contains a path with
        # spaces or shell metacharacters would otherwise produce a
        # broken Slurm command.
        flag_kwargs: dict[str, str | int | float] = {
            "output_file": output_file,
            "source_format": source_format,
            "reasoning_mode": reasoning_mode,
        }
        if num_chunks > 1:
            flag_kwargs["num_chunks"] = num_chunks

        filter_outliers = config.get("filter_outliers", False)
        if filter_outliers:
            filter_config = config.get("filter_config", {})
            flag_kwargs["context_min_percentile"] = filter_config.get("context_min_percentile", 1.0)
            flag_kwargs["context_max_percentile"] = filter_config.get(
                "context_max_percentile", 99.0
            )
            flag_kwargs["reasoning_min_percentile"] = filter_config.get(
                "reasoning_min_percentile", 1.0
            )
            flag_kwargs["reasoning_max_percentile"] = filter_config.get(
                "reasoning_max_percentile", 99.0
            )

        cmd = build_python_cmd(
            "nvflow.recipes.finance.utils.shared.dataset_transformer",
            *input_files,
            **flag_kwargs,
        )

        # Boolean flags (no value) are appended directly; ``build_python_cmd``
        # cannot express value-less flags through kwargs.
        if filter_outliers:
            cmd += " --filter_outliers"
        if config.get("deduplicate_by_uuid", False):
            cmd += " --deduplicate_by_uuid"

        run_cmd(
            ctx=wrap_arguments(cmd),
            cluster=cluster,
            log_dir=f"{output_dir}/logs",
            expname=expname,
            run_after=run_after,
        )

        console.success(
            f"Data transformation job submitted → {output_dir}/chunks/ ({num_chunks} chunks)"
        )

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate that required configuration fields are present."""
        if config.get("environments"):
            if "output_dir" not in config:
                raise ValueError("'output_dir' is required in data_transformation config")
            for env_name, env_cfg in config["environments"].items():
                sf = env_cfg.get("source_format", "separated")
                rm = env_cfg.get("reasoning_mode", "none")
                if sf == "inline" and rm == "thinking":
                    raise ValueError(
                        f"Invalid combination in environment '{env_name}': "
                        f"source_format='inline' + reasoning_mode='thinking'. "
                        "Use reasoning_mode='natural' or 'none' for inline source format."
                    )
        else:
            for field in ("input_files", "output_file"):
                if field not in config:
                    raise ValueError(f"'{field}' is required in data_transformation config")
            if (
                config.get("source_format") == "inline"
                and config.get("reasoning_mode") == "thinking"
            ):
                raise ValueError(
                    "Invalid combination: source_format='inline' + reasoning_mode='thinking'. "
                    "Use reasoning_mode='natural' or 'none' for inline source format."
                )
