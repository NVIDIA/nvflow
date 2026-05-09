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
"""Apply prompt template to data_transformation output for GRPO training.

Formats the ``problem`` field using a YAML prompt template (merging
instruction + context + question) and optionally extracts the concise
answer after a configurable prefix (e.g. "Answer:") from ``generation``.

Runs per-environment: each environment specifies its own
``prompt_template`` and ``answer_prefix`` in the ``environments`` dict.
Input is read from ``{input_dir}/{env_name}/chunks`` and output is
written to ``{output_dir}/{env_name}/``.
"""

from typing import Any

from nvflow.core import BaseStage, StageRegistry, console


@StageRegistry.register(recipe="finance", workflow="grpo", stage="apply_prompt_template")
class ApplyPromptTemplateStage(BaseStage):
    """Apply prompt template and extract expected answer (per-environment).

    Runs ``prompt_template_applier.py`` inside a Slurm container (CPU-only).
    Reads chunked JSONL from data_transformation, writes processed chunks
    to the output directory.
    """

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Submit per-environment prompt template application Slurm jobs."""
        from nvflow.lib.rl.helpers import resolve_environments

        environments = resolve_environments(config)
        base_input_dir = config["input_dir"]
        base_output_dir = config["output_dir"]

        # Dynamic current_date knobs (all optional -- feature activates only
        # when sec_metadata_parquet is set in config).
        sec_metadata_parquet = config.get("sec_metadata_parquet")
        raw_sdg_filename = config.get("raw_sdg_filename", "final_result.jsonl")
        jitter_min_days = config.get("jitter_min_days", 1)
        jitter_max_days = config.get("jitter_max_days", 60)
        fallback_current_date = config.get("fallback_current_date", "2025-04-07")
        parquet_accession_column = config.get("parquet_accession_column", "accession_number")
        parquet_filing_date_column = config.get("parquet_filing_date_column", "filing_date")

        for env_name, env_cfg in environments.items():
            raw_train_data = env_cfg.get("raw_train_data")
            if not raw_train_data:
                console.warning(f"Skipping environment '{env_name}': no raw_train_data configured")
                continue
            prompt_template = env_cfg.get("prompt_template")
            if not prompt_template:
                console.warning(f"Skipping environment '{env_name}': no prompt_template configured")
                continue

            answer_prefix = env_cfg.get("answer_prefix")
            env_input_dir = f"{base_input_dir}/{env_name}/chunks"
            env_output_dir = f"{base_output_dir}/{env_name}"

            console.status(f"Applying prompt template for environment: {env_name}")
            console.detail("Input", env_input_dir)
            console.detail("Output", env_output_dir)
            console.detail("Template", prompt_template)
            console.detail("Answer prefix", answer_prefix or "(none -- keep full generation)")
            if sec_metadata_parquet:
                console.detail("Dynamic current_date", "enabled")
                console.detail("  Parquet", sec_metadata_parquet)
                console.detail("  Raw SDG dir", raw_train_data)
                console.detail("  Raw SDG file", raw_sdg_filename)
                console.detail("  Jitter range (days)", f"[{jitter_min_days}, {jitter_max_days}]")
                console.detail("  Fallback date", fallback_current_date)
            else:
                console.detail("Dynamic current_date", "disabled (sec_metadata_parquet not set)")
            console.blank()

            self._submit_job(
                input_dir=env_input_dir,
                output_dir=env_output_dir,
                prompt_template=prompt_template,
                answer_prefix=answer_prefix,
                sec_metadata_parquet=sec_metadata_parquet,
                raw_sdg_source_dir=raw_train_data,
                raw_sdg_filename=raw_sdg_filename,
                jitter_min_days=jitter_min_days,
                jitter_max_days=jitter_max_days,
                fallback_current_date=fallback_current_date,
                parquet_accession_column=parquet_accession_column,
                parquet_filing_date_column=parquet_filing_date_column,
                cluster=cluster,
                expname=f"{expname}-{env_name}",
                run_after=run_after,
            )

    def _submit_job(
        self,
        *,
        input_dir: str,
        output_dir: str,
        prompt_template: str,
        answer_prefix: str | None,
        sec_metadata_parquet: str | None,
        raw_sdg_source_dir: str,
        raw_sdg_filename: str,
        jitter_min_days: int,
        jitter_max_days: int,
        fallback_current_date: str,
        parquet_accession_column: str,
        parquet_filing_date_column: str,
        cluster: str,
        expname: str,
        run_after: list[str] | None,
    ) -> None:
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        cmd = (
            f"python -m nvflow.recipes.finance.utils.rl.prompt_template_applier "
            f"    '{input_dir}' '{output_dir}' "
            f"    --prompt_template '{prompt_template}'"
        )
        if answer_prefix:
            cmd += f" --answer_prefix '{answer_prefix}'"
        if sec_metadata_parquet:
            cmd += (
                f" --sec_metadata_parquet '{sec_metadata_parquet}'"
                f" --raw_sdg_source_dir '{raw_sdg_source_dir}'"
                f" --raw_sdg_filename '{raw_sdg_filename}'"
                f" --jitter_min_days {jitter_min_days}"
                f" --jitter_max_days {jitter_max_days}"
                f" --fallback_current_date '{fallback_current_date}'"
                f" --parquet_accession_column '{parquet_accession_column}'"
                f" --parquet_filing_date_column '{parquet_filing_date_column}'"
            )

        run_cmd(
            ctx=wrap_arguments(cmd),
            cluster=cluster,
            num_gpus=0,
            log_dir=f"{output_dir}/logs",
            expname=expname,
            run_after=run_after,
        )

        console.success(f"Prompt template job submitted → {output_dir}/")

    def validate_config(self, config: dict[str, Any]) -> None:
        """Check that all required fields are present."""
        for field in ("input_dir", "output_dir"):
            if not config.get(field):
                raise ValueError(f"'{field}' is required in apply_prompt_template config")
        if not config.get("environments"):
            raise ValueError("'environments' is required in apply_prompt_template config")
