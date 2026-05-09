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
"""Pre-fetch SEC filing metadata cache (finance recipe).

Optional CPU-only stage that populates the SEC filing metadata cache
before rollout collection.  This avoids SEC.gov API calls during
GPU-intensive rollout jobs and eliminates race conditions when multiple
seeds share the same cache directory.

Runs per-environment: only environments whose config includes a
``prefetch`` block are processed; others are silently skipped.
"""

from typing import Any

from nvflow.core import BaseStage, StageRegistry, console


@StageRegistry.register(recipe="finance", workflow="grpo", stage="prefetch_cache")
class PrefetchCacheStage(BaseStage):
    """Pre-fetch environment-specific caches before rollout collection.

    Iterates over environments and submits a CPU-only Slurm job for each
    one that has a ``prefetch`` block in its environment config.
    """

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        from nvflow.lib.rl.helpers import resolve_environments

        environments = resolve_environments(config)
        gym_path = config["gym_path"]
        container = config["container"]
        installation_command = config.get("installation_command")

        submitted = 0
        for env_name, env_cfg in environments.items():
            prefetch = env_cfg.get("prefetch")
            if not prefetch:
                continue

            script = prefetch["script"]
            cache_dir = prefetch["cache_dir"]
            ticker_config = prefetch["ticker_config"]
            force = prefetch.get("force", False)

            cmd = f"cd {gym_path} && python {script} --cache_dir {cache_dir} --ticker_config {ticker_config}"
            if force:
                cmd += " --force"

            console.status(f"Prefetching cache for environment: {env_name}")
            console.detail("Script", script)
            console.detail("Cache dir", cache_dir)
            console.detail("Ticker config", ticker_config)
            console.detail("Force", str(force))
            console.blank()

            run_cmd(
                ctx=wrap_arguments(cmd),
                cluster=cluster,
                container=container,
                num_gpus=config.get("num_gpus", 0),
                log_dir=f"{cache_dir}/logs",
                expname=f"{expname}-{env_name}",
                run_after=run_after,
                installation_command=installation_command,
            )
            submitted += 1

        if submitted:
            console.success(f"Submitted {submitted} prefetch job(s)")
        else:
            console.success("No environments require prefetch -- skipping")

    def validate_config(self, config: dict[str, Any]) -> None:
        for field in ("gym_path", "container"):
            if not config.get(field):
                raise ValueError(f"'{field}' is required in prefetch_cache config")

        if not config.get("environments"):
            raise ValueError("'environments' dict is required in prefetch_cache config")

        environments = config["environments"]
        for env_name, env_cfg in environments.items():
            prefetch = env_cfg.get("prefetch")
            if not prefetch:
                continue
            for key in ("script", "cache_dir", "ticker_config"):
                if not prefetch.get(key):
                    raise ValueError(
                        f"'{key}' is required in environments.{env_name}.prefetch config"
                    )
