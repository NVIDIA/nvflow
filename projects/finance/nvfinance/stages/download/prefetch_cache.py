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
"""Warm the SEC filing metadata cache that the filing-search tool reads."""

import shlex
from typing import Any

from nvflow.core import console
from nvflow.execution import ArtifactDeclaration, CapturedExecutionStage, declare_task_outputs
from nvflow.execution.commands import build_python_script_cmd


class PrefetchCacheStage(CapturedExecutionStage):
    """Fetch filing metadata and the ticker registry for a ticker config.

    CPU-only. Warms what ``sec_filing_search`` looks up by ticker, so training
    and eval do not call sec.gov on the critical path. Filing bodies are not
    prefetched; they are cached on first read. Idempotent: companies already
    cached are skipped unless ``force`` is set.

    Independent of the downloaded corpus and the search index, so it carries no
    dependency on the ``download`` stage and can run on its own.
    """

    capture_strategy = "run_cmd"

    def build_tasks(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        from nvflow.integrations.nemo_skills.commands import run_cmd, wrap_arguments

        gym_path = config["gym_path"]
        cache_dir = config["cache_dir"]
        ticker_config = config["ticker_config"]
        supplementary_tickers = config.get("supplementary_tickers")

        console.status("Prefetching SEC filing metadata")
        console.detail("Cache dir", cache_dir)
        console.detail("Ticker config", ticker_config)
        if supplementary_tickers:
            console.detail("Supplementary tickers", supplementary_tickers)

        flags: dict[str, Any] = {"cache_dir": cache_dir, "ticker_config": ticker_config}
        if supplementary_tickers:
            flags["supplementary_tickers"] = supplementary_tickers
        script_cmd = build_python_script_cmd(config["script"], **flags)
        if config.get("force"):
            # Value-less flag: build_python_script_cmd only renders --key value pairs.
            script_cmd += " --force"
            console.detail("Force", "re-fetching every company")
        # The script imports its modules and resolves relative overlay paths
        # against the Gym root, so it has to run from there.
        cmd = f"cd {shlex.quote(gym_path)} && {script_cmd}"

        with declare_task_outputs(
            ArtifactDeclaration(
                name="sec-metadata-cache",
                path=str(cache_dir),
                kind="directory",
            )
        ):
            run_cmd(
                ctx=wrap_arguments(cmd),
                cluster=cluster,
                container=config.get("container", "nemo-gym"),
                num_gpus=config.get("num_gpus", 0),
                **config.get("stage_kwargs", {}),
                log_dir=f"{cache_dir}/logs",
                expname=expname,
                run_after=run_after,
                installation_command=config.get("installation_command"),
            )

        console.success("Submitted SEC metadata prefetch")

    def validate_config(self, config: dict[str, Any]) -> None:
        for field in ("gym_path", "script", "cache_dir", "ticker_config"):
            if not config.get(field):
                raise ValueError(f"'{field}' is required in prefetch_cache config")
