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
"""Split dataset into train and validation sets (shared: SFT + GRPO).

For GRPO workflows with ``environments``, runs per-environment: reads
from ``{input_dir}/{env_name}/{input_filename}`` and writes to
``{output_dir}/{env_name}/``.  The input filename defaults to
``final_result.jsonl`` but can be overridden via ``input_filename``
in the stage config.

For SFT workflows (no ``environments``), runs once with ``input_file``
from config (SFT single-dataset mode).
"""

from typing import Any

from nvflow.core import BaseStage, StageRegistry, console


@StageRegistry.register(recipe="finance", workflow="sft", stage="train_validation_split")
@StageRegistry.register(recipe="finance", workflow="grpo", stage="train_validation_split")
class TrainValidationSplitStage(BaseStage):
    """Split dataset into train and validation sets (shared: SFT + GRPO).

    Performs stratified split to maintain distribution of question types
    (or other metadata field) across both train and validation sets.
    Output files are shuffled for better training.

    When ``keep_all_fields`` is True (GRPO), all input fields are preserved.
    When False (SFT default), only SFT-specific fields are kept.
    """

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute train/validation split (per-environment when environments present)."""
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
        base_input_dir = config["input_dir"]
        base_output_dir = config["output_dir"]

        for env_name, env_cfg in environments.items():
            if not env_cfg.get("raw_train_data"):
                console.warning(f"Skipping environment '{env_name}': no raw_train_data configured")
                continue
            input_filename = config.get("input_filename", "final_result.jsonl")
            env_input_file = f"{base_input_dir}/{env_name}/{input_filename}"
            env_output_dir = f"{base_output_dir}/{env_name}"

            console.status(f"Splitting dataset for environment: {env_name}")

            self._submit_split_job(
                input_file=env_input_file,
                output_dir=env_output_dir,
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
        console.status("Splitting dataset into train and validation sets")

        self._submit_split_job(
            input_file=config["input_file"],
            output_dir=config["output_dir"],
            config=config,
            cluster=cluster,
            expname=expname,
            run_after=run_after,
        )

    def _submit_split_job(
        self,
        *,
        input_file: str,
        output_dir: str,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None,
    ) -> None:
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        val_ratio = config.get("val_ratio", 0.1)
        stratify_by = config.get("stratify_by", "question_type")
        random_seed = config.get("random_seed", 42)
        max_token_length = config.get("max_token_length")
        sort_by = config.get("sort_by")
        sort_order = config.get("sort_order", "desc")

        console.detail("Input file", input_file)
        console.detail("Output directory", output_dir)
        console.detail("Val ratio", f"{val_ratio:.1%}")
        console.detail("Stratify by", stratify_by)
        console.detail("Random seed", str(random_seed))
        if max_token_length:
            console.detail("Max token length", f"{max_token_length:,}")
        if sort_by:
            console.detail("Sort by", f"{sort_by} ({sort_order})")
        console.blank()

        cmd = (
            f"python -m nvflow.recipes.finance.utils.shared.dataset_splitter "
            f"    '{input_file}' "
            f"    --output_dir '{output_dir}' "
            f"    --val_ratio {val_ratio} "
            f"    --stratify_by '{stratify_by}' "
            f"    --random_seed {random_seed}"
        )

        if max_token_length:
            cmd += f" --max_token_length {max_token_length}"
        if config.get("keep_all_fields", False):
            cmd += " --keep_all_fields"
        if sort_by:
            cmd += f" --sort_by '{sort_by}' --sort_order {sort_order}"

        run_cmd(
            ctx=wrap_arguments(cmd),
            cluster=cluster,
            log_dir=f"{output_dir}/logs",
            expname=expname,
            run_after=run_after,
        )

        console.success("Split job submitted")
        console.detail("-> Output dir", output_dir)
        console.detail("-> Train file", f"{output_dir}/train.jsonl")
        console.detail("-> Val file", f"{output_dir}/val.jsonl")

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate that required configuration fields are present."""
        if config.get("environments"):
            for field in ("input_dir", "output_dir"):
                if field not in config:
                    raise ValueError(f"'{field}' is required in train_validation_split config")
        else:
            for field in ("input_file", "output_dir"):
                if field not in config:
                    raise ValueError(f"'{field}' is required in train_validation_split config")

        if "val_ratio" in config:
            val_ratio = config["val_ratio"]
            if not (0 <= val_ratio <= 1):
                raise ValueError(
                    f"'val_ratio' must be between 0 and 1 (inclusive), got {val_ratio}"
                )
