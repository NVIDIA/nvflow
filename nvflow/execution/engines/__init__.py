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
"""One-time execution engine dispatch."""

from __future__ import annotations

from nvflow.execution.engines.base import ActiveJob, EngineContext, ExecutionEngine, JobListing
from nvflow.execution.target import ResolvedTarget


def engine_for(target: ResolvedTarget) -> ExecutionEngine:
    """Load exactly one engine after target resolution."""
    if target.engine == "slurm":
        from nvflow.execution.engines.slurm import DirectSlurmEngine

        return DirectSlurmEngine()
    if target.engine == "ray":
        from nvflow.execution.engines.ray import SuppliedRayJobsEngine

        return SuppliedRayJobsEngine()
    raise ValueError(f"Unsupported execution engine: {target.engine}")


__all__ = ["ActiveJob", "EngineContext", "ExecutionEngine", "JobListing", "engine_for"]
