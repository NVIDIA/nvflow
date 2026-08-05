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
"""Convert Q&A data to NeMo-Gym Responses API format for GRPO training.

Runs per-environment: reads from ``{input_dir}/{env_name}/`` and writes
to ``{output_dir}/{env_name}/final_result.jsonl``.
"""

from typing import Any

from nvflow.core import BaseStage, StageRegistry, console
from nvflow.lib.cli_cmd import build_python_cmd


@StageRegistry.register(recipe="finance", workflow="grpo", stage="convert_to_responses_api")
class ConvertToResponsesAPIStage(BaseStage):
    """Lossless conversion to NeMo-Gym ``responses_create_params`` format.

    Runs ``responses_api_converter.py`` inside a Slurm container (CPU-only).
    Expects apply_prompt_template output (JSONL with ``prompt`` field).
    """

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Submit per-environment data conversion Slurm jobs."""
        from nvflow.lib.rl.helpers import resolve_environments

        environments = resolve_environments(config)
        base_input_dir = config["input_dir"]
        base_output_dir = config["output_dir"]
        container = config["container"]

        for env_name, env_cfg in environments.items():
            if not env_cfg.get("raw_train_data"):
                console.warning(f"Skipping environment '{env_name}': no raw_train_data configured")
                continue
            env_input_path = f"{base_input_dir}/{env_name}"
            env_output_dir = f"{base_output_dir}/{env_name}"
            env_output_file = f"{env_output_dir}/final_result.jsonl"

            console.status(f"Converting data for environment: {env_name}")
            console.detail("Input", env_input_path)
            console.detail("Output", env_output_file)
            console.blank()

            self._submit_job(
                input_path=env_input_path,
                output_file=env_output_file,
                output_dir=env_output_dir,
                container=container,
                cluster=cluster,
                expname=f"{expname}-{env_name}",
                run_after=run_after,
            )

    def _submit_job(
        self,
        *,
        input_path: str,
        output_file: str,
        output_dir: str,
        container: str,
        cluster: str,
        expname: str,
        run_after: list[str] | None,
    ) -> None:
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        # ``responses_api_converter`` takes ``input_path`` and
        # ``output_file`` positionally, no flag options.
        cmd = build_python_cmd(
            "nvflow.recipes.finance.utils.rl.responses_api_converter",
            input_path,
            output_file,
        )

        run_cmd(
            ctx=wrap_arguments(cmd),
            cluster=cluster,
            container=container,
            num_gpus=0,
            log_dir=f"{output_dir}/logs",
            expname=expname,
            run_after=run_after,
        )

        console.success(f"Conversion job submitted → {output_file}")

    def validate_config(self, config: dict[str, Any]) -> None:
        """Check that all required fields are present."""
        for field in ("input_dir", "output_dir", "container"):
            if not config.get(field):
                raise ValueError(f"'{field}' is required in convert_to_responses_api config")
        if not config.get("environments"):
            raise ValueError("'environments' is required in convert_to_responses_api config")
