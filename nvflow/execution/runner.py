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
"""Workflow runner for executing stage sequences with dependency management."""

import contextlib
import os
import shlex
import sys
from pathlib import Path

from nvflow.core.console import detail, header, success
from nvflow.core.workflow_definition import (
    WorkflowDefinition,
    load_workflow,
)
from nvflow.execution import (
    LocalTargetFilesystem,
    ResolvedTarget,
    SubmissionReceipt,
    TargetBinding,
    WorkflowPlanner,
    preflight_target_config,
    resolve_target,
)
from nvflow.execution.engines import EngineContext, engine_for
from nvflow.execution.filesystem import TargetFilesystem
from nvflow.execution.job_labels import workflow_label

__all__ = ["WorkflowRunner"]


class WorkflowRunner:
    """Executes workflow stages with dependency management.

    The WorkflowRunner loads a workflow configuration file and executes
    the specified stages in order, respecting dependencies.

    Supports config inheritance via _base_ key:
        # models/qwen3_14b.yaml
        _base_: ../sft.yaml
        stages:
          sft:
            num_nodes: 32

    Example:
        >>> runner = WorkflowRunner("projects/finance/workflows/sft.yaml")
        >>> runner.run()  # Run all stages
        >>>
        >>> # Or run specific stage (short name from config)
        >>> runner.run(stages=["sft"])
    """

    def __init__(
        self,
        config_path: str,
        project_dir: str | None = None,
        *,
        cluster: str | None = None,
        project_search_paths: tuple[str, ...] = (),
        filesystem: TargetFilesystem | None = None,
        resolved_target: ResolvedTarget | None = None,
        target_config_path: Path | None = None,
    ):
        """Initialize the workflow runner.

        Args:
            config_path: Path to workflow configuration YAML file
            project_dir: Optional explicit root for an external project
            cluster: Optional operator-selected cluster override
            filesystem: Optional target filesystem implementation for live planning
        """
        self.definition: WorkflowDefinition = load_workflow(
            config_path,
            project_dir=project_dir,
            project_search_paths=project_search_paths,
        )
        self.config_path = self.definition.config_path
        self.config = self.definition.config
        self.project = self.definition.project
        self.workflow_name = self.definition.workflow_name
        self.workflow_type = self.definition.workflow_type
        self.target = TargetBinding.resolve(self.definition.cluster, cluster)
        self.cluster = self.target.cluster
        if resolved_target is not None and resolved_target.binding != self.target:
            raise ValueError("Resolved target does not match selected workflow target")
        self.resolved_target = resolved_target
        self.target_config_path = target_config_path
        self.filesystem = filesystem
        self.project_context = self.definition.project_context

    def run(
        self,
        stages: list[str] | None = None,
        environment: list[str] | None = None,
    ) -> SubmissionReceipt:
        """Run workflow stages.

        Args:
            stages: List of stage names to run. If None, runs all stages
                   defined in config's pipeline_stages.
            environment: Optional list of environment names to run for.
                   If None, runs all environments defined in config.

        Example:
            >>> runner.run()  # Run all stages, all environments
            >>> runner.run(stages=["download"])  # Run one stage
            >>> runner.run(environment=["equivalence_llm_judge"])  # Single env
            >>> runner.run(environment=["mcqa", "equivalence_llm_judge"])
        """
        all_stages = self.config["pipeline_stages"]
        stages_to_run = stages if stages else all_stages

        if environment:
            environments = self.config.get("environments", {})
            for env_name in environment:
                if env_name not in environments:
                    available = ", ".join(environments.keys())
                    raise ValueError(f"Unknown environment '{env_name}'. Available: {available}")

        resolved_target = self._resolve_target()
        filesystem = self.filesystem or self._target_filesystem(resolved_target)
        planner = WorkflowPlanner(
            self.definition,
            self.target,
            filesystem=filesystem,
        )
        with self._planning_target_scope(resolved_target):
            graph = planner.execution_graph(
                resolved_target,
                stages_to_run,
                environment,
                # Supplied Ray workload paths need not exist on the launcher;
                # runtime scripts keep their own resume guards.
                live=resolved_target.engine == "slurm",
            )
        if graph.provenance is not None and not graph.provenance.submittable:
            raise ValueError(
                "Project source is not submittable: "
                + (graph.provenance.reason or "clean committed source required")
            )

        self._preflight_pipeline_health(all_stages, stages_to_run)

        header(f"NVFlow - Running Workflow: {self.workflow_name}")
        detail("Workflow Type", self.workflow_type)
        detail("Cluster", self.cluster)
        detail("Stages to run", f"{len(stages_to_run)}/{len(all_stages)}")
        if environment:
            detail("Environment", ", ".join(environment))

        platform_sources = {
            stage.implementation["platform"]["source"]
            for stage in graph.stages
            if stage.implementation is not None
        }
        if len(platform_sources) != 1:
            raise ValueError("Execution graph must bind exactly one NVFlow platform source")
        platform_source = next(iter(platform_sources))
        workflow = workflow_label(self.project_context, self.config_path)
        engine = engine_for(resolved_target)
        receipt = engine.submit(
            graph,
            EngineContext(
                target=resolved_target,
                project_context=self.project_context,
                platform_source=platform_source,
                cache_dir=_execution_cache_dir(),
                workflow=workflow,
            ),
        )

        header("✅ Workflow Submitted")
        success(f"Submitted {len(stages_to_run)} stage(s): {', '.join(stages_to_run)}")
        handles = {
            handle for experiment in receipt.experiments for handle in experiment.scheduler_handles
        }
        if receipt.controller_handle is not None:
            handles.add(receipt.controller_handle)
        if handles:
            detail("Scheduler handles", ", ".join(sorted(handles)))
        selection = (
            f"--config {shlex.quote(workflow)}"
            if Path(workflow).is_absolute()
            else f"--workflow {workflow}"
        )
        detail("Track jobs", f"nvflow status --cluster {self.cluster} {selection}")
        return receipt

    def _resolve_target(self) -> ResolvedTarget:
        if self.resolved_target is not None:
            return self.resolved_target
        target_config_path = self.target_config_path
        if target_config_path is None:
            target_config_path = preflight_target_config(self.cluster)
        self.resolved_target = resolve_target(self.target, target_config_path)
        return self.resolved_target

    def _target_filesystem(self, target: ResolvedTarget) -> TargetFilesystem:
        if target.engine == "ray":
            return LocalTargetFilesystem()
        from nvflow.integrations.nemo_skills.filesystem import lazy_target_filesystem

        return lazy_target_filesystem(self.target.cluster)

    def _planning_target_scope(self, target: ResolvedTarget):
        if target.engine == "ray":
            return contextlib.nullcontext()
        from nvflow.integrations.nemo_skills.target import bind_target_config

        return bind_target_config(target.source_path)

    def _preflight_pipeline_health(
        self,
        all_stages: list[str],
        stages_to_run: list[str],
    ) -> None:
        """Warn when deferred stages have invalid implementation bindings.

        Subset runs only validate the requested stages, so siblings that
        cannot resolve sit unnoticed until the user runs the full pipeline.
        This surfaces them upfront without blocking the selected subset.
        """
        deferred = [s for s in all_stages if s not in stages_to_run]
        missing = []
        for stage_name in deferred:
            try:
                self.definition.resolve_stage_class(stage_name)
            except (KeyError, ValueError):
                missing.append(stage_name)
        if missing:
            print(
                "[nvflow] WARNING: pipeline declares stages that are not "
                "currently resolvable (check each workflow-local 'uses' value). "
                "The workflow "
                "WILL NOT complete end-to-end without fixing these:",
                file=sys.stderr,
            )
            for s in missing:
                print(f"  - {self.project}.{self.workflow_name}.{s}", file=sys.stderr)

    def validate_config(
        self,
        stages: list[str] | None = None,
        environment: list[str] | None = None,
    ) -> None:
        """Validate all or selected workflow stages without submitting work."""
        self.definition.validate(stages=stages, environment=environment)


def _execution_cache_dir() -> Path:
    configured = os.environ.get("NVFLOW_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return (root / "nvflow").resolve()
