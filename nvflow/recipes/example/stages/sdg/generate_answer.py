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
"""Generate answers using LLM - Example stage for demonstration."""

from pathlib import Path
from typing import Any

from nvflow.core import BaseStage, StageRegistry, console


@StageRegistry.register(recipe="example", workflow="sdg_simple", stage="generate_answer")
class GenerateAnswerStage(BaseStage):
    """Generate answers using LLM.

    This is a simple example stage demonstrating:
    - How to structure an SDG stage
    - Input/output directory handling
    - Configuration validation
    - Console output for user feedback
    - Integration with nemo-skills pipeline
    """

    workflow = "sdg_simple"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute answer generation.

        Args:
            config: Stage configuration with input_file, output_dir, etc.
            cluster: Cluster name (references cluster_configs/<cluster>.yaml)
            expname: Experiment name for this stage execution
            run_after: List of experiment names to wait for (dependency management)
        """
        from nemo_skills.pipeline.cli import generate, wrap_arguments

        input_file = Path(config["input_file"])
        output_dir = config["output_dir"]
        prompt_config = config.get("prompt_config", "default_prompt.yaml")
        inline_args = config.get("inline_args", "")

        console.status("Generating answers")
        console.detail("Input file", str(input_file))
        console.detail("Output directory", output_dir)
        console.detail("Prompt config", str(prompt_config))
        console.detail("Cluster", cluster)
        console.detail("Experiment name", expname)
        console.blank()

        ctx = wrap_arguments(f"++prompt_config={prompt_config} {inline_args}")

        console.detail("Stage kwargs", str(config.get("stage_kwargs", {})))

        generate(
            ctx=ctx,
            cluster=cluster,
            input_file=str(input_file),
            output_dir=output_dir,
            expname=expname,
            run_after=run_after,
            **config.get("stage_kwargs", {}),
        )

        console.blank()
        console.success("Answer generation job submitted successfully")

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate that required configuration fields are present.

        Args:
            config: Stage configuration to validate

        Raises:
            ValueError: If required fields are missing or invalid
        """
        required = ["input_file", "output_dir"]
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

        input_file = Path(config["input_file"])

        # Check if file exists locally OR if it's a cluster path (absolute)
        if not input_file.exists():
            if not input_file.is_absolute():
                raise ValueError(f"Input file does not exist: {input_file}")
            # Absolute path - assume it will exist on cluster
            console.warning(f"Cannot validate cluster path locally: {input_file}")

        if input_file.suffix and input_file.suffix != ".jsonl":
            raise ValueError(f"Input file must be a .jsonl file, got: {input_file.suffix}")
