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
"""Convert Qwen3 training data to OpenAI messages format."""

from typing import Any

from nvflow.core import console
from nvflow.execution import CapturedExecutionStage, declare_task_outputs


class ConvertToMessagesStage(CapturedExecutionStage):
    """Convert Qwen3 chat-templated training data to OpenAI messages format.

    Parses Qwen3 chat template special tokens from prepare_for_sft output (train/val splits)
    and converts to structured messages array format compatible with OpenAI fine-tuning.

    Input format (from train_validation_split):
        {"input": "<|im_start|>...", "output": "...", "uuid": "..."}

    Output format (OpenAI messages):
        {
          "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "reasoning_content": "...", "content": "..."}
          ],
          "metadata": {"uuid": "...", "sdg_model": "..."}
        }
    """

    capture_strategy = "run_cmd"

    def build_tasks(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute conversion to messages format."""
        from nvflow.integrations.nemo_skills.commands import run_cmd, wrap_arguments

        input_dir = config["input_dir"]
        output_dir = config["output_dir"]
        extract_reasoning = config.get("extract_reasoning", True)
        sdg_model = config.get("sdg_model")

        # Build list of input files (train.jsonl and val.jsonl)
        input_files = []
        for filename in ["train.jsonl", "val.jsonl"]:
            input_files.append(f"{input_dir}/{filename}")

        console.status("Converting to OpenAI messages format (Qwen3)")
        console.detail("Input directory", input_dir)
        console.detail("Output directory", output_dir)
        console.detail("Extract reasoning", str(extract_reasoning))
        if sdg_model:
            console.detail("SDG model", sdg_model)
        console.blank()

        # Build conversion command
        input_files_str = " ".join(f"'{f}'" for f in input_files)
        cmd = (
            f"python -m nvfinance.utils.sft.messages_converter "
            f"    {input_files_str} "
            f"    --output_dir '{output_dir}'"
        )

        if not extract_reasoning:
            cmd += " --no_extract_reasoning"

        if sdg_model:
            cmd += f" --sdg_model '{sdg_model}'"

        # Submit conversion job
        with declare_task_outputs(output_dir=output_dir):
            run_cmd(
                ctx=wrap_arguments(cmd),
                cluster=cluster,
                log_dir=f"{output_dir}/logs",
                expname=expname,
                run_after=run_after,
            )

        console.success("Conversion job submitted")
        console.detail("→ Output dir", output_dir)
        console.detail("→ Train file", f"{output_dir}/train.jsonl")
        console.detail("→ Val file", f"{output_dir}/val.jsonl")

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate that required configuration fields are present."""
        required = ["input_dir", "output_dir"]
        for field in required:
            if field not in config:
                raise ValueError(f"'{field}' is required in config")
