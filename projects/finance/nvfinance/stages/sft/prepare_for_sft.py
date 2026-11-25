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
"""Prepare data for supervised fine-tuning."""

import re
from typing import Any

from nvflow.core import console
from nvflow.execution import CapturedExecutionStage, declare_task_outputs


class PrepareForSFTStage(CapturedExecutionStage):
    """Prepare and format training data for SFT.

    Calls nemo_skills.training.prepare_data to apply formatting, filters,
    and templates to raw training data for supervised fine-tuning.

    Optionally adds token counts to output for statistics and filtering.
    """

    capture_strategy = "run_cmd"

    def build_tasks(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute data preparation for SFT."""
        from nvflow.integrations.nemo_skills.commands import run_cmd, wrap_arguments

        input_dir = config["input_dir"]
        output_dir = config["output_dir"]
        prepare_data_kwargs = config.get("prepare_data_kwargs", {})
        prepare_data_ctx_args = prepare_data_kwargs.get("ctx_args", "")

        # Token counting option (filtering happens in train_validation_split)
        add_token_counts = config.get("add_token_counts", True)

        output_file = f"{output_dir}/final_result.jsonl"
        temp_dir = f"{output_dir}/temp_chunks"

        # Extract tokenizer path from ctx_args for token counting
        tokenizer_path = self._extract_tokenizer_path(prepare_data_ctx_args)

        console.status("Preparing data for SFT training")
        console.detail("Input directory", input_dir)
        console.detail("Output file", output_file)
        if add_token_counts and tokenizer_path:
            console.detail("Add token counts", "Yes")
        console.blank()

        # Build command to process each file separately to avoid OOM
        # Then concatenate all results into final output
        cmd = (
            'CONFIG_DIR=$(python -c "'
            "from importlib.resources import files; "
            "print(files('nvfinance.stages.sft').joinpath('config'))"
            '") && '
            f"mkdir -p {temp_dir} && "
            f"for file in {input_dir}/*.jsonl; do "
            f'  if [ -f "$file" ]; then '
            f'    filename=$(basename "$file"); '
            f'    echo "Processing $filename..."; '
            f"    python -m nemo_skills.training.prepare_data "
            f'      --config-path "$CONFIG_DIR" '
            f"      --config-name data_prep.yaml "
            f'      ++input_files="$file" '
            f'      ++output_path="{temp_dir}/$filename" '
            f"      {prepare_data_ctx_args} || exit 1; "
            f"  fi; "
            f"done && "
            f"mkdir -p {output_dir} && "
            f"cat {temp_dir}/*.jsonl > {output_file} && "
            f"rm -rf {temp_dir}"
        )

        # Add token counting step if enabled and tokenizer available
        if add_token_counts and tokenizer_path:
            cmd += (
                f" && python -m nvfinance.utils.sft.add_token_counts "
                f"    '{output_file}' "
                f"    --output_file '{output_file}' "
                f"    --tokenizer_path '{tokenizer_path}'"
            )

        # Submit data preparation job
        with declare_task_outputs(output_file=output_file):
            run_cmd(
                ctx=wrap_arguments(cmd),
                cluster=cluster,
                log_dir=f"{output_dir}/logs",
                expname=expname,
                run_after=run_after,
            )

        console.success(f"Data preparation job submitted → {output_file}")

    def _extract_tokenizer_path(self, ctx_args: str) -> str | None:
        """Extract tokenizer path from ctx_args string."""
        # Match ++tokenizer=/path/to/tokenizer or ++tokenizer='/path/to/tokenizer'
        match = re.search(r"\+\+tokenizer=(['\"]?)([^'\"\s]+)\1", ctx_args)
        return match.group(2) if match else None

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate that required configuration fields are present."""
        required = ["input_dir", "output_dir"]
        for field in required:
            if field not in config:
                raise ValueError(f"'{field}' is required in config")

        # Warn if add_token_counts is enabled but no tokenizer in ctx_args
        add_token_counts = config.get("add_token_counts", True)
        if add_token_counts:
            ctx_args = config.get("prepare_data_kwargs", {}).get("ctx_args", "")
            if "++tokenizer=" not in ctx_args:
                console.warning(
                    "add_token_counts is enabled but no tokenizer in ctx_args. "
                    "Token counts will not be added."
                )
