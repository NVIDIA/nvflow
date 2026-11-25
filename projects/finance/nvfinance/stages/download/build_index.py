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
"""Build the local SEC full-text search index from a downloaded corpus."""

from pathlib import Path
from typing import Any

from nvflow.core import console
from nvflow.execution import ArtifactDeclaration, CapturedExecutionStage, declare_task_outputs
from nvflow.execution.commands import build_python_cmd

_BUILDER_MODULE = "nvfinance.utils.index.build_index"


class BuildSecIndexStage(CapturedExecutionStage):
    """Index every downloaded primary document and exhibit for full-text search.

    Produces both files NeMo-Gym's ``edgar_search`` tool opens: the SQLite index
    and its ``.metadata`` companion.  CPU-only: the cost is HTML-to-text
    conversion over the whole corpus.
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

        artifact_root = config["artifact_root"]
        index_path = config["index_path"]

        console.status("Building local SEC search index")
        console.detail("Corpus", artifact_root)
        console.detail("Index", index_path)
        console.detail("Metadata companion", f"{index_path}.metadata")

        cmd = build_python_cmd(
            _BUILDER_MODULE,
            "build",
            artifact_root=artifact_root,
            index_path=index_path,
        )
        # Value-less flags: build_python_cmd only renders --key value pairs.
        if config.get("incremental"):
            cmd += " --incremental"
            console.detail("Mode", "incremental")
        if config.get("allow_missing_paths"):
            cmd += " --allow_missing_paths"
            console.detail("Missing corpus files", "allowed")

        with declare_task_outputs(
            ArtifactDeclaration(name="sec-fts-index", path=str(index_path)),
            ArtifactDeclaration(name="sec-fts-index-metadata", path=f"{index_path}.metadata"),
        ):
            run_cmd(
                ctx=wrap_arguments(cmd),
                cluster=cluster,
                container=config.get("container", "nemo-skills"),
                num_gpus=config.get("num_gpus", 0),
                **config.get("stage_kwargs", {}),
                expname=expname,
                log_dir=str(Path(index_path).parent / "index-logs"),
                run_after=run_after,
            )

        console.success("Submitted local SEC search index build")

    def validate_config(self, config: dict[str, Any]) -> None:
        for field in ("artifact_root", "index_path"):
            if not config.get(field):
                raise ValueError(f"'{field}' is required in build_index config")
