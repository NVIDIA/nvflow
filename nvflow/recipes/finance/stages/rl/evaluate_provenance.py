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
"""ProvenanceGuard evaluation stage (finance recipe).

Thin stage wrapper around the finance ProvenanceGuard sidecar CLI
(:mod:`nvflow.recipes.finance.utils.rl.provenanceguard`).  Registers as
``finance/grpo/evaluate_provenance`` and submits one CPU Slurm job per
(environment, seed) pair.

Opt-in stage: commented out in ``pipeline_stages`` in
``nvflow/recipes/finance/workflows/grpo/base.yaml``.  To enable,
uncomment the ``# - evaluate_provenance`` line.

Depends only on ``collect_rollouts``; no downstream dependencies.

Input path pattern::

    ${directories.step-5-collect-rollouts}/{env}/rollout/output-rs<seed>.jsonl

Input completion marker (required)::

    ${directories.step-5-collect-rollouts}/{env}/rollout/output-rs<seed>.jsonl.done

Output path pattern::

    ${directories.provenanceguard-eval}/{env}/provenanceguard-rs<seed>.jsonl

Output completion marker (created atomically after successful replace)::

    ${directories.provenanceguard-eval}/{env}/provenanceguard-rs<seed>.jsonl.done

Does not force ``HF_HUB_OFFLINE`` in production.  Does not create remote
directories on the local host.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from nvflow.core import BaseStage, StageRegistry, console

_SIDECAR_MODULE = "nvflow.recipes.finance.utils.rl.provenanceguard"


def build_evaluate_command(
    input_file: str,
    output_file: str,
    seed: int,
    environment: str,
    routing_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    routing_revision: str | None = None,
    nli_revision: str | None = None,
    evidence_excerpt_length: int = 500,
) -> str:
    """Build the rendered CLI command string for one ProvenanceGuard job.

    Pure function with no side effects — used by :meth:`execute` and
    tested independently.
    """
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


@StageRegistry.register(
    recipe="finance",
    workflow="grpo",
    stage="evaluate_provenance",
)
class EvaluateProvenanceStage(BaseStage):
    """CPU stage that runs ProvenanceGuard evaluation on rollout outputs.

    Reads merged rollout files from collect_rollouts output and writes
    sidecar provenance verdicts.  Depends only on collect_rollouts.
    """

    workflow = "grpo"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Submit ProvenanceGuard evaluation jobs (one per seed per env)."""
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        from nvflow.lib.rl.helpers import resolve_environments

        rollouts_dir = config["rollouts_dir"]
        output_dir = config["output_dir"]

        environments = resolve_environments(config)
        container = config.get("container", "nemo-skills")
        installation_command = config.get("installation_command", "true")

        routing_model = config.get("routing_model", "sentence-transformers/all-MiniLM-L6-v2")
        nli_model = config.get("nli_model", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
        routing_revision = config.get("routing_model_revision")
        nli_revision = config.get("nli_model_revision")

        evidence_excerpt_length = config.get("evidence_excerpt_length", 500)

        starting_seed = config["starting_seed"]
        num_seeds = config["num_random_seeds"]
        seeds = config.get(
            "seeds",
            list(range(starting_seed, starting_seed + num_seeds)),
        )

        for env_name, _env_cfg in environments.items():
            env_rollouts = f"{rollouts_dir}/{env_name}/rollout"
            env_output = f"{output_dir}/{env_name}"

            for seed in seeds:
                input_file = f"{env_rollouts}/output-rs{seed}.jsonl"
                input_done = f"{input_file}.done"
                output_file = f"{env_output}/provenanceguard-rs{seed}.jsonl"

                console.status(f"Submitting ProvenanceGuard for {env_name} seed {seed}")
                console.detail("Input", input_file)
                console.detail("Input marker", input_done)
                console.detail("Output", output_file)

                cmd = build_evaluate_command(
                    input_file=input_file,
                    output_file=output_file,
                    seed=seed,
                    environment=env_name,
                    routing_model=routing_model,
                    nli_model=nli_model,
                    routing_revision=routing_revision,
                    nli_revision=nli_revision,
                    evidence_excerpt_length=evidence_excerpt_length,
                )

                run_cmd(
                    ctx=wrap_arguments(cmd),
                    cluster=cluster,
                    log_dir=f"{env_output}/logs",
                    expname=f"{expname}-{env_name}-seed{seed}",
                    run_after=run_after,
                    container=container,
                    installation_command=installation_command,
                    num_gpus=config.get("num_gpus", 0),
                )

        console.success(
            f"ProvenanceGuard jobs submitted for {len(seeds)} seed(s) "
            f"across {len(environments)} env(s)"
        )

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate required paths and no overlap (canonicalized)."""
        for field_name in ("rollouts_dir", "output_dir"):
            if field_name not in config:
                raise ValueError(f"{field_name} is required in evaluate_provenance config")

        rollouts_dir = Path(os.path.expanduser(config["rollouts_dir"])).resolve(strict=False)
        output_dir = Path(os.path.expanduser(config["output_dir"])).resolve(strict=False)

        if rollouts_dir == output_dir:
            raise ValueError("rollouts_dir and output_dir must not be the same path")

        try:
            output_dir.relative_to(rollouts_dir)
        except ValueError:
            pass
        else:
            raise ValueError("output_dir must not be inside rollouts_dir")

        try:
            rollouts_dir.relative_to(output_dir)
        except ValueError:
            pass
        else:
            raise ValueError("rollouts_dir must not be inside output_dir")


__all__ = ["EvaluateProvenanceStage", "build_evaluate_command"]
