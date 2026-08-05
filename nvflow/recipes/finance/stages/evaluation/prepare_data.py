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
"""Stage to prepare finance benchmark datasets.

Runs prepare.py scripts for custom finance benchmarks.
Uses custom utility script since nemo-skills prepare_data only works with built-in datasets.

Reference: https://nvidia-nemo.github.io/Skills/evaluation/
"""

from pathlib import Path
from typing import Any

from nvflow.core import BaseStage, StageRegistry, console


@StageRegistry.register(recipe="finance", workflow="eval", stage="prepare_data")
class PrepareFinanceBenchmarksStage(BaseStage):
    """Prepare custom finance benchmark datasets.

    Runs prepare.py scripts for each benchmark via a custom utility script.
    Note: nemo-skills' built-in prepare_data doesn't support custom datasets.

    Example:
        benchmarks: ["secque", "financebench"]
        output_dir: "/workspace/outputs/finance/eval-datasets"
    """

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute benchmark data preparation.

        Args:
            config: Stage configuration with benchmarks and data directory
            cluster: Cluster name for job submission
            expname: Experiment name for this stage execution
            run_after: List of experiment names to wait for
        """
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        # Get configuration - use dataset_names to avoid conflict with eval benchmarks config
        benchmarks = config.get("dataset_names", config.get("benchmarks", ["secque"]))
        stage_kwargs = config.get("stage_kwargs", {})

        # Resolve output_dir - must be an absolute path for cluster execution
        output_dir = config.get("output_dir")
        if not output_dir or not Path(output_dir).is_absolute():
            raise ValueError(
                f"output_dir must be an absolute path, got: '{output_dir}'. "
                "Example: output_dir: /workspace/outputs/finance/eval-datasets"
            )

        # Convert benchmarks to list if string
        if isinstance(benchmarks, str):
            benchmarks = [benchmarks]

        console.status("Submitting dataset preparation job")
        console.detail("Benchmarks", ", ".join(benchmarks))
        console.detail("Output directory (cluster)", output_dir)
        console.detail("Cluster", cluster)
        console.blank()

        # Build a single command that processes all benchmarks sequentially
        cmd_parts = []
        for benchmark in benchmarks:
            benchmark_cmd = (
                f"echo '=== Starting preparation for {benchmark} ===' && "
                f"python -m nvflow.recipes.finance.utils.evaluation.prepare_benchmark_data "
                f"--benchmarks {benchmark} "
                f"--output_dir '{output_dir}' && "
                f"echo '=== Completed preparation for {benchmark} ==='"
            )
            cmd_parts.append(benchmark_cmd)

        cmd = " && ".join(cmd_parts)

        # Submit single job
        run_cmd(
            ctx=wrap_arguments(cmd),
            cluster=cluster,
            log_dir=f"{output_dir}/logs",
            expname=expname,
            run_after=run_after,
            **stage_kwargs,
        )

        console.success(f"Dataset preparation job submitted → {output_dir}")
