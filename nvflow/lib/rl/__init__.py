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
"""RL infrastructure library -- rollout collection and reward computation.

Submodules:
    rollout        -- Slurm pipeline for collecting rollouts
    verify         -- Slurm pipeline for re-judging rollouts
    helpers        -- shared utilities (vLLM config, judge config, shell templates)
    create_overlay -- standalone worker: creates symlinked model overlay dirs
    resume_filter  -- standalone worker: fine-grained resume filtering
    verify_worker  -- standalone worker: re-judges rollouts via NeMo-Gym ServerClient

Worker scripts (create_overlay, resume_filter, verify_worker) run
inside the Slurm container with ``PYTHONPATH=/workspace``.  This
__init__.py is intentionally kept import-free so that
``python -m nvflow.lib.rl.<worker>`` does not trigger the nemo_skills
dependency chain.
"""
