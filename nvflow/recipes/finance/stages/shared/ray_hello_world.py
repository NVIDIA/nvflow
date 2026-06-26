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
"""Ray backend hello-world smoke test stage for finance recipe."""

import os
import shlex
from typing import Any

from nvflow.core import BaseStage, StageRegistry, console


@StageRegistry.register(recipe="finance", workflow="ray_hello_world", stage="hello_world")
class RayHelloWorldStage(BaseStage):
    """Submit a simple hello-world job through nemo-skills run_cmd.

    The selected cluster config is expected to set:
    backend.name: ray
    so execution goes through the Skills pipeline Ray backend implementation.
    """

    workflow = "ray_hello_world"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Submit a hello-world command job."""
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        output_dir = config["output_dir"]
        message = config.get("message", "hello world from nvflow finance ray backend")
        stage_kwargs = config.get("stage_kwargs", {})
        reserved = {"ctx", "cluster", "expname", "run_after", "num_gpus", "log_dir"}
        conflicts = reserved & stage_kwargs.keys()
        if conflicts:
            raise ValueError(
                f"stage_kwargs may not override reserved run_cmd arguments: {sorted(conflicts)}"
            )

        message_q = shlex.quote(message)
        output_dir_q = shlex.quote(output_dir)
        output_file_q = shlex.quote(f"{output_dir}/hello_world.txt")

        cmd = f"mkdir -p {output_dir_q} && echo {message_q} | tee {output_file_q}"

        console.status("Submitting finance ray hello-world job")
        console.detail("Cluster", cluster)
        console.detail("Output dir", output_dir)
        console.detail("Message", message)
        console.blank()

        run_cmd(
            ctx=wrap_arguments(cmd),
            cluster=cluster,
            expname=expname,
            run_after=run_after,
            num_gpus=0,
            log_dir=f"{output_dir}/logs",
            **stage_kwargs,
        )

        # Verify the job actually persisted output to a shared, orchestrator-visible path.
        # On a pre-provisioned Ray cluster, my_cluster.yaml `mounts:` are NOT applied (the
        # cluster was brought up with start_ray_on_slurm.sb's MOUNTS), so an output_dir under
        # an unmounted alias (e.g. /workspace) is written to ephemeral container storage — the
        # Ray job still SUCCEEDS, which would make this smoke a false pass. run_cmd blocks until
        # the Ray job completes, so the file is present here iff it landed on a shared mount.
        output_file = f"{output_dir}/hello_world.txt"
        if not os.path.exists(output_file):
            raise RuntimeError(
                f"Ray hello-world completed but its output is missing at {output_file}. "
                "On a pre-provisioned Ray cluster the bringup MOUNTS (scripts/start_ray_on_slurm.sb) "
                "must include this path — my_cluster.yaml `mounts:` are not applied to a precreated "
                "cluster. Point base_output_dir at a mounted shared path (or add the mount) and retry."
            )

        console.success(f"Hello-world job verified -> {output_file}")

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate stage config."""
        if not config.get("output_dir"):
            raise ValueError("'output_dir' is required in hello_world config")
