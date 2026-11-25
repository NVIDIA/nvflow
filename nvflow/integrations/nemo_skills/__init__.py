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
"""NeMo-Skills planning and Slurm integration."""

from nvflow.integrations.nemo_skills.filesystem import (
    launcher_is_remote,
    target_filesystem,
)
from nvflow.integrations.nemo_skills.planning import (
    PlannedJobs,
    portable_cluster_config,
    stage_plan_from_jobs,
)
from nvflow.integrations.nemo_skills.slurm import dependency_sbatch_kwargs
from nvflow.integrations.nemo_skills.target import load_target_config

__all__ = [
    "PlannedJobs",
    "dependency_sbatch_kwargs",
    "launcher_is_remote",
    "load_target_config",
    "portable_cluster_config",
    "stage_plan_from_jobs",
    "target_filesystem",
]
