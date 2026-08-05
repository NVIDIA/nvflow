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
"""Document grounded SDG data post processing stage."""

from typing import Any

from nvflow.core import BaseStage, console

from ._helpers import build_trim_cmd


class DGSDGPostProcessStage(BaseStage):
    """Post process document grounded SDG data by cleaning fields and creating subsets."""

    workflow = "document_grounded_sdg"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute document grounded SDG data post processing."""
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        input_file = config["input_file"]
        output_dir = config["output_dir"]
        seed = config.get("seed", 42)
        postprocess_script = config["postprocess_script"]

        console.status("Post processing document grounded SDG data")
        console.detail("Input file", input_file)
        console.detail("Output dir", output_dir)
        console.detail("Random seed", str(seed))
        console.detail("Postprocess script", postprocess_script)
        console.blank()

        postprocess_cmd = (
            f"python {postprocess_script} "
            f"--input_file {input_file} "
            f"--output_dir {output_dir} "
            f"--seed {seed}"
        )
        trim_cmd = build_trim_cmd(
            stage_name="dgsdg_post_process",
            paths=[f"{output_dir}/final_result.jsonl"],
            domain_keep_fields=config.get("domain_keep_fields"),
            # ``responses_create_params`` is in ALWAYS_DROP; re-add it here so the
            # final Responses-API record retains the original request.
            extra_keep_fields=["responses_create_params"],
        )
        cmd = f"{postprocess_cmd} && {trim_cmd}"

        run_cmd(
            ctx=wrap_arguments(cmd),
            cluster=cluster,
            expname=expname,
            run_after=run_after,
        )

        console.success("Document grounded SDG data post processing job submitted")
        console.detail("Output files will be in", output_dir)

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate stage configuration."""
        required = ["input_file", "output_dir", "postprocess_script"]
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
