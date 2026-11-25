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
"""Backend-neutral execution engine interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from nvflow.execution.contracts import ExecutionGraph
from nvflow.execution.receipts import SubmissionReceipt
from nvflow.execution.target import ResolvedTarget

if TYPE_CHECKING:
    from nvflow.project.context import ProjectContext


@dataclass(frozen=True, slots=True)
class EngineContext:
    """Launcher-owned context supplied identically to either engine."""

    target: ResolvedTarget
    project_context: ProjectContext
    platform_source: str
    cache_dir: Path
    # The selector naming the workflow, which every submitted job is labeled with.
    workflow: str | None = None


@dataclass(frozen=True, slots=True)
class ActiveJob:
    """One queued or running job, described the same way by either engine.

    An engine leaves a field None when its scheduler does not report it.
    """

    job_id: str
    name: str
    state: str
    elapsed: str
    location: str
    workflow: str | None = None
    stage: str | None = None
    experiment: str | None = None
    # Why a pending job is waiting, or where a running job runs.
    reason: str | None = None
    nodes: int | None = None
    gpus_per_node: int | None = None
    cpus: int | None = None
    start: str | None = None
    account: str | None = None
    priority: int | None = None


@dataclass(frozen=True, slots=True)
class JobListing:
    """Queued and running jobs, and why any location's jobs could not be read."""

    jobs: tuple[ActiveJob, ...] = ()
    unreachable: Mapping[str, str] = field(default_factory=dict)


class ExecutionEngine(ABC):
    """Submit one exact graph without inspecting project stage classes."""

    @abstractmethod
    def submit(
        self,
        graph: ExecutionGraph,
        context: EngineContext,
    ) -> SubmissionReceipt:
        """Submit *graph* and return a provider-neutral receipt."""

    @abstractmethod
    def active_jobs(self, target: ResolvedTarget) -> JobListing:
        """Return the launcher user's queued and running jobs on *target*."""

    def jobs_per_stage(self, graph: ExecutionGraph) -> Mapping[str, int] | None:
        """Return how many scheduler jobs submitting *graph* starts per stage, if known."""
        return None


__all__ = ["ActiveJob", "EngineContext", "ExecutionEngine", "JobListing"]
