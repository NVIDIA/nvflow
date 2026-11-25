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

"""Merge heterogeneous SDG question sources for open-context agent rollouts."""

import json
from typing import Any

from nvflow.core import console
from nvflow.execution import CapturedExecutionStage, declare_task_outputs
from nvflow.execution.commands import build_python_cmd


class PrepareAgentQuestionsStage(CapturedExecutionStage):
    """Materialize one normalized agent-rollout dataset from configured SDG sources."""

    capture_strategy = "run_cmd"

    def build_tasks(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        from nvflow.integrations.nemo_skills.commands import run_cmd, wrap_arguments

        output_file = config["output_file"]
        cmd = build_python_cmd(
            "nvfinance.utils.sdg.prepare_agent_questions",
            sources_json=json.dumps(config["sources"], separators=(",", ":")),
            output_file=output_file,
            verification_threshold=config.get("verification_threshold", 1.0),
        )
        console.status("Preparing TSDG and RSDG questions for agentic rollouts")
        console.detail("Sources", str(len(config["sources"])))
        console.detail("Output", output_file)
        with declare_task_outputs(output_file=output_file):
            run_cmd(
                ctx=wrap_arguments(cmd),
                cluster=cluster,
                expname=expname,
                log_dir=f"{config['output_dir']}/logs",
                run_after=run_after,
            )

    def validate_config(self, config: dict[str, Any]) -> None:
        if not config.get("sources"):
            raise ValueError("'sources' must be a non-empty list")
        for field in ("output_dir", "output_file"):
            if not config.get(field):
                raise ValueError(f"'{field}' is required")
