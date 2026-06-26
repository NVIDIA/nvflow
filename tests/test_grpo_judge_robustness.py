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
"""Regression tests for the GRPO external-judge mid-run robustness guards.

Covers the differentiated mid-run watchdog string builder
(:meth:`GRPOStage._external_judge_midrun_watchdog`):

  - ``external_vllm`` (self-hosted): watchdog ENABLED -> emits a background
    poller that touches the ``ENDED`` sentinel after N consecutive failures.
  - ``openai`` (managed): watchdog DISABLED by default -> emits nothing, so a
    transient 429/503 never tears the run down; opt-in via config.
  - The watchdog is a pure string builder, so it is asserted without a cluster.
"""

from nvflow.recipes.finance.stages.rl.training import GRPOStage


class TestExternalJudgeMidRunWatchdog:
    """Mid-run watchdog must differ by judge type (NV-4)."""

    LOG_DIR = "/out/training-logs"
    URL = "http://judge-host:8000/v1"

    def _build(self, judge_mode, config=None):
        return GRPOStage._external_judge_midrun_watchdog(self.URL, judge_mode, self.LOG_DIR, config)

    def test_external_vllm_arms_watchdog_with_ended_sentinel(self):
        """Self-hosted external_vllm gets a background poller -> ENDED touch."""
        script = self._build("external_vllm")
        assert script  # non-empty
        assert "_nvflow_judge_watchdog" in script
        assert f"touch {self.LOG_DIR}/ENDED" in script
        # Runs in the background so it does not block the training command.
        assert "&" in script
        # Probes the server-root /health (not /v1/health).
        assert "http://judge-host:8000/health" in script
        assert "http://judge-host:8000/v1/models" in script

    def test_external_vllm_uses_consecutive_failure_threshold(self):
        """Counter resets on success (F=0) and trips at the threshold."""
        script = self._build("external_vllm")
        assert "F=0" in script  # reset on a healthy poll
        assert f"-ge {GRPOStage._MIDRUN_FAIL_THRESHOLD}" in script

    def test_openai_does_not_watchdog_by_default(self):
        """Managed openai judge: NO continuous poll/teardown on transient errors."""
        assert self._build("openai") == ""

    def test_openai_opt_in_enables_conservative_watchdog(self):
        """Operator may opt openai in; threshold defaults to a longer window."""
        script = self._build(
            "openai", config={"judge_midrun_watchdog": {"enable_for_openai": True}}
        )
        assert script
        assert f"touch {self.LOG_DIR}/ENDED" in script
        # openai opt-in window is >= the vLLM threshold (more conservative).
        assert "-ge 10" in script

    def test_config_can_disable_watchdog_entirely(self):
        """An explicit kill-switch suppresses the watchdog even for external_vllm."""
        script = self._build("external_vllm", config={"judge_midrun_watchdog": {"enabled": False}})
        assert script == ""

    def test_config_overrides_poll_and_threshold(self):
        """poll_secs / fail_threshold are configurable."""
        script = self._build(
            "external_vllm",
            config={"judge_midrun_watchdog": {"poll_secs": 30, "fail_threshold": 8}},
        )
        assert "sleep 30" in script
        assert "-ge 8" in script

    def test_local_and_policy_modes_emit_nothing(self):
        """local_vllm has its own monitor; policy_as_judge has no endpoint."""
        assert self._build("local_vllm") == ""
        assert self._build("policy_as_judge") == ""
