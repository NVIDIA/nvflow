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
"""Group training data by sequence length."""

from typing import Any

from nvflow.core import console
from nvflow.execution import CapturedExecutionStage, declare_task_outputs


class SequenceLengthGroupingStage(CapturedExecutionStage):
    """Group training examples by total sequence length (input + output).

    Groups examples based on their total token length to reduce padding overhead
    during training. If total_token_length is pre-computed in the data (from
    prepare_for_sft), tokenizer_path is optional and bucketing is fast.
    """

    capture_strategy = "run_cmd"

    def build_tasks(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute grouping by sequence length."""
        from nvflow.integrations.nemo_skills.commands import run_cmd, wrap_arguments

        input_file = config["input_file"]
        output_dir = config["output_dir"]
        tokenizer_path = config.get("tokenizer_path")  # Optional if data has total_token_length
        bucket_sizes = config.get("bucket_sizes", [16000, 32000, 64000])

        console.status("Grouping data by sequence length")
        console.detail("Input file", input_file)
        console.detail("Output directory", output_dir)
        if tokenizer_path:
            console.detail("Tokenizer", tokenizer_path)
        else:
            console.detail("Tokenizer", "Not needed (using pre-computed lengths)")
        console.detail("Bucket sizes", str(bucket_sizes))
        console.blank()

        # Build the batching command
        bucket_sizes_str = " ".join(str(b) for b in bucket_sizes)
        cmd = (
            f"python -m nvfinance.utils.sft.sequence_batcher "
            f"    '{input_file}' "
            f"    --output_dir '{output_dir}' "
            f"    --to_bucket "
            f"    --bucket_sizes {bucket_sizes_str}"
        )

        if tokenizer_path:
            cmd += f" --tokenizer_path '{tokenizer_path}'"

        # Submit grouping job
        with declare_task_outputs(output_dir=output_dir):
            run_cmd(
                ctx=wrap_arguments(cmd),
                cluster=cluster,
                log_dir=f"{output_dir}/logs",
                expname=expname,
                run_after=run_after,
            )

        console.success(f"Sequence grouping job submitted → {output_dir}")

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate that required configuration fields are present."""
        required = ["input_file", "output_dir"]
        for field in required:
            if field not in config:
                raise ValueError(f"'{field}' is required in config")
