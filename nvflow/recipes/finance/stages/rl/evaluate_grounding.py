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

from nvflow.core import BaseStage, StageRegistry, console
from nvflow.grounding_verifier.embedder import DEFAULT_EMBEDDING_MODEL
from nvflow.grounding_verifier.nli import DEFAULT_NLI_MODEL

_SIDECAR_MODULE = "nvflow.recipes.finance.utils.rl.grounding_verifier"
_FEATURE_GATE_MODULE = "nvflow.recipes.finance.utils.rl.grounding_feature_gate"
_FEATURE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def build_evaluate_command(
    input_file: str,
    output_file: str,
    seed: int,
    environment: str,
    routing_model: str = DEFAULT_EMBEDDING_MODEL,
    nli_model: str = DEFAULT_NLI_MODEL,
    routing_revision: str | None = None,
    nli_revision: str | None = None,
    evidence_excerpt_length: int = 500,
) -> str:
    """Build one side-effect-free GroundingVerifier CLI command."""
    from nvflow.lib.cli_cmd import build_python_cmd

    flags: dict[str, str | int] = {
        "input_file": input_file,
        "output_file": output_file,
        "seed": seed,
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
    rollouts_dir: str,
    sidecars_dir: str,
    output_file: str,
    starting_seed: int,
    required_runs: int,
    expected_rows_per_run: int,
    max_unavailable_rate: float,
    require_offline: bool,
    model_root: str,
) -> str:
    """Build the post-evaluation repeated-run acceptance command."""
    from nvflow.lib.cli_cmd import build_python_cmd

    return build_python_cmd(
        _FEATURE_GATE_MODULE,
        feature=feature,
        environment=environment,
        rollouts_dir=rollouts_dir,
        sidecars_dir=sidecars_dir,
        output_file=output_file,
        starting_seed=starting_seed,
        required_runs=required_runs,
        expected_rows_per_run=expected_rows_per_run,
        max_unavailable_rate=max_unavailable_rate,
        require_offline=int(require_offline),
        model_root=model_root,
    )


@StageRegistry.register(
    recipe="finance",
    workflow="grpo",
    stage="evaluate_grounding",
)
class EvaluateGroundingStage(BaseStage):
    """Submit one sidecar evaluation job per environment and seed."""

    workflow = "grpo"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Submit GroundingVerifier evaluation jobs (one per seed per env)."""
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        from nvflow.lib.rl.helpers import resolve_environments

        rollouts_dir, output_dir = config["rollouts_dir"], config["output_dir"]
        environments = resolve_environments(config)
        container = config.get("container", "nemo-skills")
        installation_command = config.get("installation_command", "true")
        starting_seed = config["starting_seed"]
        seeds = config.get(
            "seeds",
            list(range(starting_seed, starting_seed + config["num_random_seeds"])),
        )
        evaluator_options = {
            "routing_model": config.get("routing_model", DEFAULT_EMBEDDING_MODEL),
            "nli_model": config.get("nli_model", DEFAULT_NLI_MODEL),
            "routing_revision": config.get("routing_model_revision"),
            "nli_revision": config.get("nli_model_revision"),
            "evidence_excerpt_length": config.get("evidence_excerpt_length", 500),
        }
        feature_gate = config.get("feature_gate")

        for env_name, _env_cfg in environments.items():
            env_rollouts = f"{rollouts_dir}/{env_name}/rollout"
            env_output = f"{output_dir}/{env_name}"
            evaluation_jobs: list[str] = []

            for seed in seeds:
                input_file = f"{env_rollouts}/output-rs{seed}.jsonl"
                input_done = f"{input_file}.done"
                output_file = f"{env_output}/grounding-verifier-rs{seed}.jsonl"

                console.status(f"Submitting GroundingVerifier for {env_name} seed {seed}")
                console.detail("Input", input_file)
                console.detail("Input marker", input_done)
                console.detail("Output", output_file)

                cmd = build_evaluate_command(
                    input_file=input_file,
                    output_file=output_file,
                    seed=seed,
                    environment=env_name,
                    **evaluator_options,
                )
                evaluation_expname = f"{expname}-{env_name}-seed{seed}"
                evaluation_jobs.append(evaluation_expname)

                run_cmd(
                    ctx=wrap_arguments(cmd),
                    cluster=cluster,
                    log_dir=f"{env_output}/logs",
                    expname=evaluation_expname,
                    run_after=run_after,
                    container=container,
                    installation_command=installation_command,
                    num_gpus=config.get("num_gpus", 0),
                )

            if feature_gate:
                feature = feature_gate["name"]
                output_file = f"{env_output}/feature-gate-{feature}.json"
                console.status(f"Submitting feature gate {feature} for {env_name}")
                gate_cmd = build_feature_gate_command(
                    feature=feature,
                    environment=env_name,
                    rollouts_dir=rollouts_dir,
                    sidecars_dir=output_dir,
                    output_file=output_file,
                    starting_seed=starting_seed,
                    required_runs=feature_gate["required_runs"],
                    expected_rows_per_run=feature_gate["expected_rows_per_run"],
                    max_unavailable_rate=feature_gate.get("max_unavailable_rate", 0.0),
                    require_offline=feature_gate.get("require_offline", False),
                    model_root=feature_gate.get("model_root", "/hf_models"),
                )
                run_cmd(
                    ctx=wrap_arguments(gate_cmd),
                    cluster=cluster,
                    log_dir=f"{env_output}/logs",
                    expname=f"{expname}-{env_name}",
                    run_after=evaluation_jobs,
                    container=container,
                    installation_command=installation_command,
                    num_gpus=0,
                )

        console.success(
            f"GroundingVerifier jobs submitted for {len(seeds)} seed(s) "
            f"across {len(environments)} env(s)"
        )

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
        starting_seed = config.get("starting_seed", 0)
        seeds = config.get(
            "seeds",
            list(range(starting_seed, starting_seed + config.get("num_random_seeds", 0))),
        )
        if feature_gate.get("required_runs") != len(seeds):
            raise ValueError("feature_gate.required_runs must match the configured seed count")
        expected_seeds = list(range(starting_seed, starting_seed + len(seeds)))
        if list(seeds) != expected_seeds:
            raise ValueError("feature_gate requires contiguous seeds starting at starting_seed")
        if feature_gate.get("expected_rows_per_run", 0) < 1:
            raise ValueError("feature_gate.expected_rows_per_run must be positive")
        max_unavailable = feature_gate.get("max_unavailable_rate", 0.0)
        if not 0 <= max_unavailable <= 1:
            raise ValueError("feature_gate.max_unavailable_rate must be between 0 and 1")
