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
"""Tests for repeated native finance GroundingVerifier feature gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from nvflow.core.workflow_runner import WorkflowRunner
from nvflow.recipes.finance.stages.rl.evaluate_grounding import EvaluateGroundingStage
from nvflow.recipes.finance.utils.rl.grounding_feature_gate import (
    GroundingFeatureGateError,
    validate_grounding_feature,
)


def _write_run(
    root: Path,
    seed: int,
    status: str = "allow",
    models: dict[str, str] | None = None,
) -> None:
    rollout = root / "rollouts/finance_sec_search/rollout" / f"output-rs{seed}.jsonl"
    sidecar = root / "sidecars/finance_sec_search" / f"grounding-verifier-rs{seed}.jsonl"
    rollout.parent.mkdir(parents=True, exist_ok=True)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps({"response": {"id": f"response-{seed}"}}) + "\n"
    rollout.write_text(raw)
    Path(f"{rollout}.done").touch()
    sidecar.write_text(
        json.dumps(
            {
                "seed": seed,
                "line_number": 0,
                "evaluation_uuid": f"evaluation-{seed}",
                "raw_line_fingerprint": hashlib.sha256(raw.encode()).hexdigest()[:16],
                "verdict": {"status": status},
                "models": models,
            }
        )
        + "\n"
    )
    Path(f"{sidecar}.done").touch()


def _validate(root: Path, **overrides):
    options = {
        "feature": "test_feature",
        "environment": "finance_sec_search",
        "rollouts_dir": root / "rollouts",
        "sidecars_dir": root / "sidecars",
        "starting_seed": 0,
        "required_runs": 3,
        "expected_rows_per_run": 1,
        "max_unavailable_rate": 0.0,
    }
    return validate_grounding_feature(**{**options, **overrides})


def test_repeated_native_runs_pass_with_complete_sidecars(tmp_path):
    for seed in range(3):
        _write_run(tmp_path, seed, status="block" if seed == 2 else "allow")
    summary = _validate(tmp_path)
    assert summary["required_runs"] == 3
    assert summary["verdicts"] == {"allow": 2, "block": 1, "unavailable": 0}


def test_airgap_gate_requires_local_models_and_offline_flags(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    models = {
        "routing_model": str(model_root / "routing"),
        "nli_model": str(model_root / "nli"),
    }
    for model_path in models.values():
        path = Path(model_path)
        path.mkdir(parents=True)
        (path / "config.json").touch()
        (path / "model.safetensors").touch()
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(name, "1")
    for seed in range(3):
        _write_run(tmp_path, seed, models=models)

    assert _validate(tmp_path, require_offline=True, model_root=model_root)["passed"]

    monkeypatch.delenv("TRANSFORMERS_OFFLINE")
    with pytest.raises(GroundingFeatureGateError, match="TRANSFORMERS_OFFLINE"):
        _validate(tmp_path, require_offline=True, model_root=model_root)


@pytest.mark.parametrize("failure", ["missing_run", "fingerprint", "unavailable"])
def test_repeated_native_runs_fail_closed(tmp_path, failure):
    for seed in range(3):
        _write_run(tmp_path, seed, status="unavailable" if failure == "unavailable" else "allow")
    if failure == "missing_run":
        Path(f"{tmp_path}/rollouts/finance_sec_search/rollout/output-rs2.jsonl.done").unlink()
    elif failure == "fingerprint":
        sidecar = tmp_path / "sidecars/finance_sec_search/grounding-verifier-rs2.jsonl"
        row = json.loads(sidecar.read_text())
        row["raw_line_fingerprint"] = "wrong"
        sidecar.write_text(json.dumps(row) + "\n")
    with pytest.raises(GroundingFeatureGateError):
        _validate(tmp_path)


def test_airgap_profile_runs_three_finance_seeds():
    config = WorkflowRunner(
        "nvflow/recipes/finance/workflows/grpo/feature_gates/grounding_airgap.yaml"
    ).config
    assert config["pipeline_stages"] == ["collect_rollouts", "evaluate_grounding"]
    assert config["stages"]["collect_rollouts"]["rollout"]["num_random_seeds"] == 3
    assert config["stages"]["collect_rollouts"]["rollout"]["max_num_samples"] == 1
    assert config["stages"]["evaluate_grounding"]["feature_gate"] == {
        "name": "grounding_airgap",
        "required_runs": 3,
        "expected_rows_per_run": 1,
        "max_unavailable_rate": 0.0,
        "require_offline": True,
        "model_root": "/hf_models",
    }


def test_stage_submits_gate_after_every_seed(monkeypatch, stub_nemo_pipeline_cli):
    import nvflow.lib.rl.helpers as helpers

    submitted = stub_nemo_pipeline_cli
    monkeypatch.setattr(helpers, "resolve_environments", lambda config: {"finance_sec_search": {}})
    EvaluateGroundingStage().execute(
        {
            "rollouts_dir": "/rollouts",
            "output_dir": "/sidecars",
            "starting_seed": 0,
            "num_random_seeds": 3,
            "feature_gate": {
                "name": "grounding_airgap",
                "required_runs": 3,
                "expected_rows_per_run": 1,
            },
        },
        cluster="test-cluster",
        expname="grpo-evaluate-grounding",
        run_after=["collect-rollouts"],
    )
    assert len(submitted) == 4
    assert submitted[-1]["expname"] == "grpo-evaluate-grounding-finance_sec_search"
    assert submitted[-1]["run_after"] == [
        f"grpo-evaluate-grounding-finance_sec_search-seed{seed}" for seed in range(3)
    ]
    assert "grounding_feature_gate" in submitted[-1]["ctx"]


def test_stage_rejects_noncontiguous_feature_gate_seeds():
    with pytest.raises(ValueError, match="contiguous seeds"):
        EvaluateGroundingStage().validate_config(
            {
                "rollouts_dir": "/rollouts",
                "output_dir": "/sidecars",
                "starting_seed": 0,
                "seeds": [0, 2, 4],
                "feature_gate": {
                    "name": "test_feature",
                    "required_runs": 3,
                    "expected_rows_per_run": 1,
                },
            }
        )
