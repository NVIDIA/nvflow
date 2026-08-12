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
"""Tests for embedded-eval judge resolution (BYO judge in airgap).

A ``judge:`` set under a workflow recipe's ``stages.eval`` block must take
precedence over the shared default in ``eval/base.yaml``, so a customer can
point evaluation at their own OpenAI-compatible judge without editing the
shared base config.  A present-but-empty override falls back to the base
default (``or`` semantics) rather than silently disabling the judge.
"""

from pathlib import Path

from nvflow.core.workflow_runner import WorkflowRunner
from nvflow.recipes.finance.stages.evaluation.evaluate import _resolve_judge

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestResolveJudge:
    BASE_JUDGE = {
        "model": "gpt-5-chat-latest",
        "server_type": "openai",
        "server_address": "https://api.openai.com/v1",
    }
    BYO_JUDGE = {
        "model": "openai/gpt-oss-120b",
        "server_type": "openai",
        "server_address": "http://judge-host:5001/v1",
    }

    def test_workflow_override_wins(self):
        """stages.eval.judge takes precedence over eval/base.yaml."""
        resolved = _resolve_judge(
            {"judge": self.BYO_JUDGE},
            {"judge": self.BASE_JUDGE},
        )
        assert resolved == self.BYO_JUDGE
        assert resolved["server_address"] == "http://judge-host:5001/v1"

    def test_falls_back_to_base_when_absent(self):
        """No override -> base default is used unchanged."""
        resolved = _resolve_judge({}, {"judge": self.BASE_JUDGE})
        assert resolved == self.BASE_JUDGE

    def test_empty_override_falls_back_to_base(self):
        """A present-but-empty judge falls through to base (safer for airgap)."""
        assert _resolve_judge({"judge": None}, {"judge": self.BASE_JUDGE}) == self.BASE_JUDGE
        assert _resolve_judge({"judge": {}}, {"judge": self.BASE_JUDGE}) == self.BASE_JUDGE

    def test_no_judge_anywhere_returns_none(self):
        """No judge configured anywhere -> None (caller handles)."""
        assert _resolve_judge({}, {}) is None


def test_ray_demo_throttles_shared_hosted_judge():
    """The customer-facing Ray recipe must not use NeMo-Skills' 512-request default."""
    recipe = REPO_ROOT / "nvflow/recipes/finance/workflows/eval/demo_ray.yaml"
    extra_args = WorkflowRunner(str(recipe)).config["judge"]["extra_args"]

    assert "++max_concurrent_requests=1" in extra_args
    assert "++server.max_retries=10" in extra_args
