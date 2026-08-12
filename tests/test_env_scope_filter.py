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
"""Tests for `-e` environment filtering against per-stage `environments` scope.

The CLI ``-e`` flag is a *filter*, never an *expansion*: a stage that declares
an ``environments`` block must only run for the requested env(s) within that
scope, and is skipped (empty result) when none of the requested envs are in
scope. A stage with no ``environments`` block is unscoped and runs unchanged.
"""

from nvflow.core.workflow_runner import WorkflowRunner

_filter = WorkflowRunner._filter_stage_environment


class TestFilterStageEnvironment:
    def test_unscoped_stage_runs_unchanged(self):
        """No `environments` block -> the requested env(s) pass through."""
        assert _filter({}, ["equivalence_llm_judge"]) == ["equivalence_llm_judge"]
        assert _filter({"foo": "bar"}, ["a", "b"]) == ["a", "b"]

    def test_scoped_stage_intersects(self):
        """A scoped stage runs only for requested envs within its scope."""
        cfg = {"environments": {"finance_sec_search": {}, "equivalence_llm_judge": {}}}
        assert _filter(cfg, ["equivalence_llm_judge"]) == ["equivalence_llm_judge"]
        assert _filter(cfg, ["equivalence_llm_judge", "other"]) == ["equivalence_llm_judge"]

    def test_scoped_stage_out_of_scope_is_skipped(self):
        """Requested env outside the stage's scope -> empty (caller skips)."""
        cfg = {"environments": {"finance_sec_search": {}}}
        assert _filter(cfg, ["equivalence_llm_judge"]) == []

    def test_matches_resolve_env_names_for_scoped_stage(self):
        """Mirror of `_resolve_env_names` for the scoped case (deps == exec)."""
        cfg = {"environments": {"finance_sec_search": {}}}
        env = ["finance_sec_search", "equivalence_llm_judge"]
        assert _filter(cfg, env) == WorkflowRunner._resolve_env_names(cfg, env)
