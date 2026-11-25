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
"""CPU stage that verifies grounding in collected finance rollouts."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from nvfinance.grounding_verifier.embedder import DEFAULT_EMBEDDING_MODEL
from nvfinance.grounding_verifier.nli import DEFAULT_NLI_MODEL
from nvflow.core import console
from nvflow.execution import CapturedExecutionStage, declare_task_outputs
from nvflow.execution.commands import build_python_cmd

_SIDECAR_MODULE = "nvfinance.utils.rl.grounding_verifier"
_FEATURE_GATE_MODULE = "nvfinance.utils.rl.grounding_feature_gate"
_FEATURE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def build_evaluate_command(
    rollout_root: str,
    output_file: str,
    environment: str,
    routing_model: str = DEFAULT_EMBEDDING_MODEL,
    nli_model: str = DEFAULT_NLI_MODEL,
    routing_revision: str | None = None,
    nli_revision: str | None = None,
    evidence_excerpt_length: int = 500,
) -> str:
    """Build one side-effect-free GroundingVerifier CLI command."""
    flags: dict[str, str | int] = {
        "rollout_root": rollout_root,
        "output_file": output_file,
        "environment": environment,
        "routing_model": routing_model,
        "nli_model": nli_model,
        "evidence_excerpt_length": evidence_excerpt_length,
    }
    if routing_revision:
        flags["routing_model_revision"] = routing_revision
    if nli_revision:
        flags["nli_model_revision"] = nli_revision
    return build_python_cmd(_SIDECAR_MODULE, **flags)


def build_feature_gate_command(
    *,
    feature: str,
    environment: str,
    rollout_root: str,
    sidecar_file: str,
    output_file: str,
    required_runs: int,
    expected_rows_per_run: int,
    max_unavailable_rate: float,
    require_offline: bool,
    model_root: str,
) -> str:
    """Build the post-evaluation repeated-run acceptance command."""
    return build_python_cmd(
        _FEATURE_GATE_MODULE,
        feature=feature,
        environment=environment,
        rollout_root=rollout_root,
        sidecar_file=sidecar_file,
        output_file=output_file,
        required_runs=required_runs,
        expected_rows_per_run=expected_rows_per_run,
        max_unavailable_rate=max_unavailable_rate,
        require_offline=int(require_offline),
        model_root=model_root,
    )


class EvaluateGroundingStage(CapturedExecutionStage):
    """Evaluate each environment's collected rollouts, then run the optional feature gate."""

    capture_strategy = "run_cmd"

    def build_tasks(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Submit one GroundingVerifier evaluation job per environment."""
        from nvflow.integrations.nemo_gym.environments import resolve_environments
        from nvflow.integrations.nemo_skills.commands import run_cmd, wrap_arguments

        environments = resolve_environments(config)
        container = config.get("container", "nemo-skills")
        evaluator_options = {
            "routing_model": config.get("routing_model", DEFAULT_EMBEDDING_MODEL),
            "nli_model": config.get("nli_model", DEFAULT_NLI_MODEL),
            "routing_revision": config.get("routing_model_revision"),
            "nli_revision": config.get("nli_model_revision"),
            "evidence_excerpt_length": config.get("evidence_excerpt_length", 500),
        }
        feature_gate = config.get("feature_gate")

        for env_name, env_cfg in environments.items():
            # collect_rollouts writes each environment under its output_name.
            output_name = env_name if env_cfg.get("output_name") is None else env_cfg["output_name"]
            rollout_root = f"{config['rollouts_dir']}/{output_name}"
            env_output = f"{config['output_dir']}/{output_name}"
            sidecar = f"{env_output}/grounding-verifier.jsonl"
            evaluation_expname = f"{expname}-{env_name}"

            console.status(f"Submitting GroundingVerifier for {env_name}")
            console.detail("Rollouts", rollout_root)
            console.detail("Output", sidecar)

            command = build_evaluate_command(rollout_root, sidecar, env_name, **evaluator_options)
            with declare_task_outputs(output_file=sidecar):
                run_cmd(
                    ctx=wrap_arguments(command),
                    cluster=cluster,
                    container=container,
                    num_gpus=0,
                    log_dir=f"{env_output}/logs",
                    expname=evaluation_expname,
                    run_after=run_after,
                )

            if feature_gate:
                feature = feature_gate["name"]
                summary = f"{env_output}/feature-gate-{feature}.json"
                console.status(f"Submitting feature gate {feature} for {env_name}")
                gate_command = build_feature_gate_command(
                    feature=feature,
                    environment=env_name,
                    rollout_root=rollout_root,
                    sidecar_file=sidecar,
                    output_file=summary,
                    required_runs=feature_gate["required_runs"],
                    expected_rows_per_run=feature_gate["expected_rows_per_run"],
                    max_unavailable_rate=feature_gate.get("max_unavailable_rate", 0.0),
                    require_offline=feature_gate.get("require_offline", False),
                    model_root=feature_gate.get("model_root", "/hf_models"),
                )
                with declare_task_outputs(output_file=summary):
                    run_cmd(
                        ctx=wrap_arguments(gate_command),
                        cluster=cluster,
                        container=container,
                        num_gpus=0,
                        log_dir=f"{env_output}/logs",
                        expname=f"{evaluation_expname}-{feature}",
                        run_after=[evaluation_expname],
                    )

        console.success(f"GroundingVerifier jobs submitted for {len(environments)} environment(s)")

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate required paths and no overlap (canonicalized)."""
        missing = [name for name in ("rollouts_dir", "output_dir") if name not in config]
        if missing:
            raise ValueError(f"{missing[0]} is required in evaluate_grounding config")

        rollouts_dir = Path(os.path.expanduser(config["rollouts_dir"])).resolve(strict=False)
        output_dir = Path(os.path.expanduser(config["output_dir"])).resolve(strict=False)

        if rollouts_dir == output_dir:
            raise ValueError("rollouts_dir and output_dir must not be the same path")

        if output_dir.is_relative_to(rollouts_dir):
            raise ValueError("output_dir must not be inside rollouts_dir")
        if rollouts_dir.is_relative_to(output_dir):
            raise ValueError("rollouts_dir must not be inside output_dir")

        feature_gate = config.get("feature_gate")
        if not feature_gate:
            return
        feature = feature_gate.get("name", "")
        if not _FEATURE_NAME_RE.fullmatch(feature):
            raise ValueError("feature_gate.name must contain lowercase letters, digits, _ or -")
        if feature_gate.get("required_runs", 0) < 1:
            raise ValueError("feature_gate.required_runs must be positive")
        if feature_gate.get("expected_rows_per_run", 0) < 1:
            raise ValueError("feature_gate.expected_rows_per_run must be positive")
        max_unavailable = feature_gate.get("max_unavailable_rate", 0.0)
        if not 0 <= max_unavailable <= 1:
            raise ValueError("feature_gate.max_unavailable_rate must be between 0 and 1")
