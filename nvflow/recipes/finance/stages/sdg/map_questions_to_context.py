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
"""Generate Q&A pairs for financial reasoning."""

from pathlib import Path
from typing import Any

from nvflow.core import BaseStage, StageRegistry, console

# Run with uv run nflow run map_questions_to_context --config=nvflow/recipes/finance/workflows/sdg/template-based-sdg.yaml


@StageRegistry.register(
    recipe="finance", workflow="template_based_sdg", stage="map_questions_to_context"
)
class MapQuestionsToContextStage(BaseStage):
    """Map questions to filing contexts."""

    workflow = "template_based_sdg"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Map questions to filing contexts."""
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        input_file = config["input_file"]
        output_file = config["output_file"]
        filings_dir = config["filings_dir"]
        filings_metadata = config["filings_metadata"]
        token_limit = config["token_limit"]

        console.status("Mapping questions to filing contexts")
        console.detail("Input file", input_file)
        console.detail("Output file", output_file)
        console.detail("Filings dir", filings_dir)
        console.detail("Filings metadata", filings_metadata)
        console.detail("Token limit", token_limit)
        console.blank()

        output_dir = Path(output_file).parent

        run_cmd(
            ctx=wrap_arguments(
                f"python3 -m nvflow.recipes.finance.utils.shared.question_context_utils --input_file {input_file} --filings_metadata {filings_metadata} --filings_dir {filings_dir} --output_file {output_file} --token_limit {token_limit}"
            ),
            cluster=cluster,
            **config.get("stage_kwargs", {}),
            expname=expname,
            log_dir=f"{output_dir}/map-questions-to-context-logs",
            run_after=run_after,
        )

        console.success("Mapped questions to filing contexts")
