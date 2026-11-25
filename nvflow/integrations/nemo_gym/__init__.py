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
"""Native NeMo Gym data-plane integration.

Submodules:
    compare        -- native Gym comparison command construction
    environments   -- environment selection and NeMo-Gym config paths
    generation     -- document-grounded generation submission and planning
    job_scripts    -- shared nemo-skills job-script wrappers
    launcher_fs    -- compatibility imports for relocated filesystem helpers
    native_commands -- shell-safe native Gym CLI rendering
    native_contract -- canonical artifact paths
    native_rollout -- rollout/repeat/chunk orchestration
    native_rewards -- independent reward recomputation
    native_sharding -- deterministic sharding and completion evidence
    postprocess    -- shared analysis and dataset-publication command rendering
    reward_profiles -- fail-closed native profile normalization and selection
    shell_snippets -- reusable inline shell functions
    vllm_judge     -- vLLM and judge configuration helpers
    workers        -- standalone container worker entrypoints

Worker scripts run inside Slurm containers. This module remains import-free so
``python -m nvflow.integrations.nemo_gym.workers.<worker>`` does not import the
NeMo-Skills launcher dependency chain.
"""
