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
"""SEC Data Preprocessing Stage for Document-Grounded SDG.

This stage processes raw SEC filings (10-K and 10-Q HTML files) into structured JSONL data:
1. Chunk HTML files into Markdown, Clean HTML, and Original HTML
2. Generate CSV file lists from chunked files
3. Generate JSONL training data from CSVs
"""

from typing import Any

from nvflow.core import BaseStage, StageRegistry, console


@StageRegistry.register(
    recipe="finance",
    workflow="document_grounded_sdg",
    stage="dg_sdg_preprocess",
)
class DGSDGPreprocessStage(BaseStage):
    """Preprocess SEC filings for document-grounded SDG.

    This stage converts raw SEC HTML filings into structured JSONL data:
    1. Chunks HTML files by token count with overlap
    2. Generates CSV file lists for tracking chunks
    3. Creates JSONL training data with proper sampling distribution
    """

    workflow = "document_grounded_sdg"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute the SEC data preprocessing pipeline."""
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        input_dir = config["input_dir"]
        output_dir = config["output_dir"]
        distribution_dir = config["distribution_dir"]

        # Chunking settings
        max_tokens = config.get("max_tokens", 2000)
        overlap_tokens = config.get("overlap_tokens", 100)

        # Sampling settings
        total_samples = config.get("total_samples", 150000)
        max_skip_count = config.get("max_skip_count", 20000)
        seed = config.get("seed", 42)

        console.status("SEC Data Preprocessing")
        console.detail("Input dir", input_dir)
        console.detail("Output dir", output_dir)
        console.detail("Distribution dir", distribution_dir)
        console.detail("Max tokens", str(max_tokens))
        console.detail("Overlap tokens", str(overlap_tokens))
        console.detail("Total samples", str(total_samples))
        console.detail("Max skip count", str(max_skip_count))
        console.detail("Seed", str(seed))
        console.blank()

        preprocess_module = "nvflow.recipes.finance.utils.sdg.dg_sdg_data_preprocess"

        full_cmd = (
            f"python3 -m {preprocess_module} "
            f"--input_dir {input_dir} "
            f"--output_dir {output_dir} "
            f"--distribution_dir {distribution_dir} "
            f"--max_tokens {max_tokens} "
            f"--overlap_tokens {overlap_tokens} "
            f"--total_samples {total_samples} "
            f"--max_skip_count {max_skip_count} "
            f"--seed {seed}"
        )

        console.status("Running SEC data preprocessing")

        run_cmd(
            ctx=wrap_arguments(full_cmd),
            cluster=cluster,
            expname=expname,
            run_after=run_after,
        )

        console.success("SEC data preprocessing job submitted")
        console.detail("Output directory", output_dir)

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate stage configuration."""
        required = ["input_dir", "output_dir", "distribution_dir"]
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
