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
from collections.abc import Sequence
from pathlib import Path

import pytest
from nvfinance.stages.rl.evaluate_grounding import EvaluateGroundingStage
from nvfinance.utils.rl.grounding_feature_gate import (
    GroundingFeatureGateError,
    validate_grounding_feature,
)

from nvflow import load_workflow
from nvflow.integrations.nemo_gym.native_contract import NativeGymOutputLayout
from nvflow.integrations.nemo_gym.native_sharding import publish_rollout_completion

ROLLOUT_ROOT = "rollouts/finance_sec_search"
SIDECAR = "sidecars/finance_sec_search/grounding-verifier.jsonl"


def _write_runs(
    root: Path,
    statuses: Sequence[str] = ("allow", "allow", "allow"),
    models: dict[str, str] | None = None,
) -> None:
    """Write one collected rollout per run, each with a matching sidecar row."""
    artifacts = NativeGymOutputLayout.collection(root / ROLLOUT_ROOT).analyzed_artifacts
    rollouts = Path(artifacts.rollouts)
    inputs = Path(artifacts.materialized_inputs)
    sidecar = root / SIDECAR
    rollouts.parent.mkdir(parents=True)
    sidecar.parent.mkdir(parents=True)

    raw_rows = [
        json.dumps({"_ng_task_index": 0, "_ng_rollout_index": run, "response": {"id": f"r{run}"}})
        + "\n"
        for run in range(len(statuses))
    ]
    rollouts.write_text("".join(raw_rows))
    inputs.write_text(
        "".join(
            json.dumps({"_ng_task_index": 0, "_ng_rollout_index": run}) + "\n"
            for run in range(len(statuses))
        )
    )
    publish_rollout_completion(inputs, rollouts, artifacts.completion)
    sidecar.write_text(
        "".join(
            json.dumps(
                {
                    "line_number": run,
                    "_ng_rollout_index": run,
                    "evaluation_uuid": f"evaluation-{run}",
                    "raw_line_fingerprint": hashlib.sha256(raw.encode()).hexdigest()[:16],
                    "verdict": {"status": status},
                    "models": models,
                }
            )
            + "\n"
            for run, (raw, status) in enumerate(zip(raw_rows, statuses, strict=True))
        )
    )
    Path(f"{sidecar}.done").touch()


def _validate(root: Path, **overrides):
    options = {
        "feature": "test_feature",
        "environment": "finance_sec_search",
        "rollout_root": root / ROLLOUT_ROOT,
        "sidecar": root / SIDECAR,
        "required_runs": 3,
        "expected_rows_per_run": 1,
        "max_unavailable_rate": 0.0,
    }
    return validate_grounding_feature(**{**options, **overrides})


def test_repeated_native_runs_pass_with_complete_sidecars(tmp_path):
    _write_runs(tmp_path, statuses=("allow", "allow", "block"))
    summary = _validate(tmp_path)
    assert summary["required_runs"] == 3
    assert summary["rollout_indices"] == [0, 1, 2]
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
    _write_runs(tmp_path, models=models)

    assert _validate(tmp_path, require_offline=True, model_root=model_root)["passed"]

    monkeypatch.delenv("TRANSFORMERS_OFFLINE")
    with pytest.raises(GroundingFeatureGateError, match="TRANSFORMERS_OFFLINE"):
        _validate(tmp_path, require_offline=True, model_root=model_root)


@pytest.mark.parametrize("failure", ["missing_run", "incomplete", "fingerprint", "unavailable"])
def test_repeated_native_runs_fail_closed(tmp_path, failure):
    statuses = ["allow"] * (2 if failure == "missing_run" else 3)
    if failure == "unavailable":
        statuses[2] = "unavailable"
    _write_runs(tmp_path, statuses=statuses)
    if failure == "incomplete":
        artifacts = NativeGymOutputLayout.collection(tmp_path / ROLLOUT_ROOT).analyzed_artifacts
        Path(artifacts.completion).unlink()
    elif failure == "fingerprint":
        sidecar = tmp_path / SIDECAR
        rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
        rows[2]["raw_line_fingerprint"] = "wrong"
        sidecar.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(GroundingFeatureGateError):
        _validate(tmp_path)


def test_airgap_profile_runs_three_finance_repeats():
    config = load_workflow(
        Path(__file__).parents[1] / "workflows/grpo/feature_gates/grounding_airgap.yaml"
    ).config
    assert config["pipeline_stages"] == ["collect_rollouts", "evaluate_grounding"]
    assert config["stages"]["collect_rollouts"]["rollout"]["num_repeats"] == 3
    assert config["stages"]["collect_rollouts"]["rollout"]["max_num_samples"] == 1
    assert config["stages"]["evaluate_grounding"]["feature_gate"] == {
        "name": "grounding_airgap",
        "required_runs": 3,
        "expected_rows_per_run": 1,
        "max_unavailable_rate": 0.0,
        "require_offline": True,
        "model_root": "/hf_models",
    }


@pytest.mark.parametrize(
    "feature_gate",
    [
        {"name": "Bad Name", "required_runs": 3, "expected_rows_per_run": 1},
        {"name": "test_feature", "required_runs": 0, "expected_rows_per_run": 1},
        {"name": "test_feature", "required_runs": 3, "expected_rows_per_run": 0},
        {
            "name": "test_feature",
            "required_runs": 3,
            "expected_rows_per_run": 1,
            "max_unavailable_rate": 1.5,
        },
    ],
)
def test_stage_rejects_invalid_feature_gate(feature_gate):
    with pytest.raises(ValueError, match="feature_gate"):
        EvaluateGroundingStage().validate_config(
            {"rollouts_dir": "/rollouts", "output_dir": "/sidecars", "feature_gate": feature_gate}
        )
