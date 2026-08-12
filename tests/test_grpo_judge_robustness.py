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

import sys
import types
from types import SimpleNamespace

import pytest

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


@pytest.mark.parametrize(
    ("executor", "judge_mode", "expect_health_gate", "expect_watchdog"),
    [
        ("slurm", "external_vllm", False, False),
        ("slurm", "openai", False, False),
        ("ray", "external_vllm", True, True),
        ("ray", "openai", False, False),
    ],
)
def test_submission_service_guards_are_ray_only(
    monkeypatch, executor, judge_mode, expect_health_gate, expect_watchdog
):
    """Ordinary Slurm gets the exact train command; only self-hosted Ray is gated."""
    captured = {}

    class _Exp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    # NVFlow keeps NeMo-Skills optional for local validation.  Stub the lazy
    # submission imports so this regression runs in the same minimal CI
    # environment instead of relying on an incidental developer installation.
    ns_mod = types.ModuleType("nemo_skills")
    pipeline_mod = types.ModuleType("nemo_skills.pipeline")
    nemo_rl_mod = types.ModuleType("nemo_skills.pipeline.nemo_rl")
    utils_mod = types.ModuleType("nemo_skills.pipeline.utils")
    grpo_mod = types.ModuleType("nemo_skills.pipeline.nemo_rl.grpo")
    exp_mod = types.ModuleType("nemo_skills.pipeline.utils.exp")
    grpo_mod.parse_kwargs = lambda value: {}
    exp_mod.get_exp = lambda *args, **kwargs: _Exp()
    exp_mod.run_exp = lambda *args, **kwargs: None
    exp_mod.add_task = lambda exp, **kwargs: captured.setdefault("cmd", kwargs["cmd"])
    ns_mod.pipeline = pipeline_mod
    pipeline_mod.nemo_rl = nemo_rl_mod
    pipeline_mod.utils = utils_mod
    nemo_rl_mod.grpo = grpo_mod
    utils_mod.exp = exp_mod
    for name, module in {
        "nemo_skills": ns_mod,
        "nemo_skills.pipeline": pipeline_mod,
        "nemo_skills.pipeline.nemo_rl": nemo_rl_mod,
        "nemo_skills.pipeline.nemo_rl.grpo": grpo_mod,
        "nemo_skills.pipeline.utils": utils_mod,
        "nemo_skills.pipeline.utils.exp": exp_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    stage = GRPOStage.__new__(GRPOStage)
    stage._config_shell_snippet = lambda prepared, **kwargs: ("", "/tmp/config.yaml")
    stage._build_train_cmd = lambda *args, **kwargs: "TRAIN"
    stage._resolve_external_judge_url = lambda *args, **kwargs: "http://judge:8000/v1"
    prepared = SimpleNamespace(
        judge_job_info=None,
        judge_mode=judge_mode,
        output_dir="/out",
        expname="grpo",
        num_gpus=8,
        num_nodes=1,
        run_after=None,
    )
    cluster_config = {
        "executor": "slurm" if executor == "slurm" else "none",
        "containers": {"nemo-rl": "image"},
    }
    if executor == "ray":
        cluster_config["backend"] = {
            "name": "ray",
            "dashboard_url": "http://ray-head:8265",
        }

    GRPOStage._submit_grpo_job(stage, prepared, cluster_config, {})

    command = captured["cmd"]
    assert ("Probing external judge" in command) is expect_health_gate
    assert ("_nvflow_judge_watchdog" in command) is expect_watchdog
    if not expect_health_gate and not expect_watchdog:
        assert command == "TRAIN"
