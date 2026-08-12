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
"""Regression: embedded checkpoint-eval must propagate ``eval.server_address``.

A recipe can point evaluation at an external, already-running OpenAI-compatible
endpoint (BYO-serve) by setting ``eval.server_address`` with ``gpus: 0`` — the
Ray SFT/GRPO eval recipes do exactly this.  When that happens nemo-skills must
NOT try to self-host the model: it needs ``server_address`` and
``server_gpus=None``.

The standalone baseline path (``_build_config_from_model``) has always done
this, but the embedded SFT/GRPO checkpoint-eval path (``EmbeddedEvalStage`` /
``_CheckpointEvaluator``) previously hard-coded ``server_gpus=config["gpus"]``
and dropped ``server_address`` entirely, so nemo-skills failed with
``Model 0 is not self-hosted (server_gpus=0/None) but server_address is
missing``.  These tests drive ``EmbeddedEvalStage.execute`` end-to-end to the
``nemo_skills.pipeline.cli.eval`` boundary and assert the endpoint is forwarded
for BYO-serve, while remaining a no-op for self-hosted recipes.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest

from nvflow.recipes.finance.stages.evaluation import evaluate as ev
from nvflow.recipes.finance.stages.evaluation.evaluate import EmbeddedEvalStage

BYO_ADDRESS = "http://eval-endpoint:5000/v1"

# The two embedded-eval branches that use skip_conversion=True (so the test
# never touches the on-cluster checkpoint-conversion / nemo_skills machinery).
STEP_CONFIGS = {
    "final": {"eval_steps": ["final"]},
    "hf-checkpoint": {"eval_steps": [100], "format": "hf"},
}


def _install_fake_nemo_skills(monkeypatch, nemo_eval):
    """Register a stub ``nemo_skills.pipeline.cli`` so the lazy import resolves.

    ``nemo_skills`` is not importable in this source-only test environment; the
    evaluator imports ``eval``/``wrap_arguments`` from it at call time, so we
    inject a stub module tree exposing exactly those two symbols.
    """
    cli = types.ModuleType("nemo_skills.pipeline.cli")
    cli.eval = nemo_eval
    cli.wrap_arguments = lambda s: s
    pipeline = types.ModuleType("nemo_skills.pipeline")
    pipeline.cli = cli
    root = types.ModuleType("nemo_skills")
    root.pipeline = pipeline
    monkeypatch.setitem(sys.modules, "nemo_skills", root)
    monkeypatch.setitem(sys.modules, "nemo_skills.pipeline", pipeline)
    monkeypatch.setitem(sys.modules, "nemo_skills.pipeline.cli", cli)


def _drive_eval(monkeypatch, step_config, rollout_config):
    """Run EmbeddedEvalStage.execute and return the kwargs passed to nemo_eval."""
    nemo_eval = MagicMock(return_value=None)
    _install_fake_nemo_skills(monkeypatch, nemo_eval)

    # Deterministic base config (avoids reading eval/base.yaml / path-roots).
    monkeypatch.setattr(
        ev,
        "_load_eval_base_config",
        lambda cluster=None: {
            "datasets_dir": "/workspace/outputs/finance/eval-datasets",
            "benchmarks": {"secque": {"seeds": 1}},
            "judge": {
                "model": "judge",
                "server_type": "openai",
                "server_address": "http://judge:5001/v1",
            },
            "conversion": {},
        },
    )
    # Skip the benchmark-prep submission (no cluster / PrepareFinanceBenchmarksStage).
    monkeypatch.setattr(EmbeddedEvalStage, "_prepare_benchmark_data", lambda self, **kw: "")

    config = {
        "checkpoint_path": "/workspace/outputs/run",
        "eval_output_dir": "/workspace/outputs/run/eval",
        **step_config,
        **rollout_config,
    }
    EmbeddedEvalStage().execute(config=config, cluster="myslurm-arm", expname="sft-eval")

    assert nemo_eval.call_count == 1
    return nemo_eval.call_args.kwargs


@pytest.mark.parametrize("step_key", list(STEP_CONFIGS))
def test_byo_serve_forwards_server_address(monkeypatch, step_key):
    """BYO-serve (server_address set, gpus:0) -> forward address, server_gpus=None."""
    kwargs = _drive_eval(
        monkeypatch,
        STEP_CONFIGS[step_key],
        {"server_address": BYO_ADDRESS, "gpus": 0},
    )
    assert kwargs["server_address"] == BYO_ADDRESS
    assert kwargs["server_gpus"] is None


@pytest.mark.parametrize("step_key", list(STEP_CONFIGS))
def test_self_hosted_preserves_server_gpus(monkeypatch, step_key):
    """No server_address -> address stays None and server_gpus keeps the recipe value."""
    kwargs = _drive_eval(
        monkeypatch,
        STEP_CONFIGS[step_key],
        {"gpus": 4},
    )
    assert kwargs["server_address"] is None
    assert kwargs["server_gpus"] == 4
