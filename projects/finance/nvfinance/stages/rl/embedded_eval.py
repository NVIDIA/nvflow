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
"""Evaluate training checkpoints on NeMo-Gym benchmarks.

Runs as the last stage of an SFT or GRPO pipeline. Each configured step is
resolved to a HuggingFace checkpoint -- converting from Megatron or DCP/FSDP
first when needed -- and then handed to the same rollout matrix the standalone
baseline campaigns use, so a checkpoint and a baseline are scored identically.

The eval settings (benchmark catalog, judge, serving profiles) come from a
campaign in ``workflows/eval``; a training config names that campaign, a serving
preset, the benchmarks to run, and ``eval_data_root``: the benchmark data the
evaluation campaign's ``eval_data`` stage prepared.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from nvfinance import PROJECT_ROOT
from nvfinance.stages.rl.eval_model import (
    GymCheckpointEvaluatorStage,
    GymModelEvaluatorStage,
)
from nvflow.execution import PlannedExecutionStage

if TYPE_CHECKING:
    from nvflow.execution import PlanningContext, StagePlan, TaskPlan
    from nvflow.project import ProjectContext

_EVAL_DIR = PROJECT_ROOT / "workflows" / "eval"

# Grades QA locally, so a training pipeline needs no hosted-API credentials.
# `eval.campaign` overrides it.
_DEFAULT_CAMPAIGN = "embedded.yaml"

_DEFAULT_BENCHMARKS = ["secque", "financebench"]

_KNOWN_KEYS = {
    "base_model",
    "base_output_dir",
    "baseline_model",
    "benchmarks",
    "campaign",
    "checkpoint_path",
    "conversion",
    "dependencies",
    "environments",
    "eval_data_root",
    "eval_output_dir",
    "eval_steps",
    "format",
    "preset",
    "run_name",
    "repeats",
}

# Keys the nemo-skills eval stage accepted, and where their replacement lives.
# Silently ignoring them would score a migrated config differently with no sign.
_RETIRED_KEYS = {
    "judge": "set `campaign` to a campaign whose benchmarks use the judge you want",
    "server_type": "serving is defined by the campaign's `model_presets`, chosen with `preset`",
    "gpus": "set `server.num_gpus` on the campaign preset named by `preset`",
    "server_args": "set the equivalent field on the campaign preset named by `preset`",
    "inference_args": "set `responses_create_params` on the campaign preset named by `preset`",
}


def _load_eval_campaign(
    data_root: str,
    campaign: str = _DEFAULT_CAMPAIGN,
    *,
    eval_data_root: str,
    project_context: ProjectContext | None = None,
) -> dict[str, Any]:
    """Load a campaign locally, then render its mappings for the worker."""
    from omegaconf import OmegaConf

    from nvflow.core.workflow_definition import load_config_with_inheritance

    campaign_path = _EVAL_DIR / campaign
    if not campaign_path.exists():
        available = sorted(p.name for p in _EVAL_DIR.glob("*.yaml"))
        raise ValueError(f"Unknown eval.campaign '{campaign}'. Available: {', '.join(available)}")

    config = load_config_with_inheritance(campaign_path)
    config.data_root = data_root
    config.eval_data_root = eval_data_root
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError(f"Eval campaign {campaign_path} must contain a mapping")
    resolved = cast(dict[str, Any], resolved)
    if project_context is not None:
        resolved = cast(
            dict[str, Any],
            project_context.render_execution_paths(resolved),
        )
    # The campaign's own baseline models are not part of a training pipeline.
    resolved.pop("models", None)
    return resolved


def _resolve_preset(campaign: dict[str, Any], preset: str | None) -> dict[str, Any]:
    """Return the named serving profile from the campaign's ``model_presets``."""
    presets = campaign.get("model_presets") or {}
    if not preset:
        raise ValueError(
            f"eval.preset is required. Available presets: {', '.join(sorted(presets))}"
        )
    if preset not in presets:
        raise ValueError(f"Unknown eval.preset '{preset}'. Available: {', '.join(sorted(presets))}")
    return dict(presets[preset])


def _for_training_environment(config: dict[str, Any]) -> dict[str, Any]:
    """Apply the ``eval`` mapping of the environment that trains alone."""
    from nvflow.integrations.nemo_gym.environments import deep_merge, training_environments

    environments = training_environments(config)
    if len(environments) != 1:
        return config
    (environment,) = environments.values()
    return deep_merge(config, environment.get("eval") or {})


def _eval_data_root(config: dict[str, Any]) -> str:
    if eval_data_root := config.get("eval_data_root"):
        return str(eval_data_root)
    raise ValueError(
        "eval.eval_data_root is required when eval_steps is set -- point it at the "
        "step-1-eval-data folder of the evaluation workflow that prepares the benchmarks"
    )


def _check_keys(values: dict[str, Any], prefix: str, known: set[str]) -> None:
    for key in values:
        if key in known or key.startswith("_"):
            continue
        if replacement := _RETIRED_KEYS.get(key):
            raise ValueError(f"{prefix}.{key} is no longer supported -- {replacement}")
        raise ValueError(f"Unknown {prefix}.{key}. Supported: {', '.join(sorted(known))}")


def _scope_environments(
    campaign: dict[str, Any],
    benchmarks: list[str] | None,
    repeats: int | None = None,
) -> dict[str, Any]:
    """Narrow the campaign benchmark catalog to *benchmarks*.

    *repeats* overrides the catalog's per-benchmark repeat count, which is what a
    smoke run wants; leaving it unset keeps checkpoints comparable to baselines.
    """
    catalog = campaign.get("benchmarks") or {}
    names = list(benchmarks or _DEFAULT_BENCHMARKS)
    unknown = [n for n in names if n not in catalog]
    if unknown:
        raise ValueError(
            f"Unknown benchmark(s) in eval.benchmarks: {', '.join(unknown)}. "
            f"Available: {', '.join(catalog)}"
        )
    scoped = {name: dict(catalog[name]) for name in names}
    if repeats is not None:
        for env in scoped.values():
            env["repeats"] = repeats
    return scoped


class GymEmbeddedEvalStage(PlannedExecutionStage):
    """Evaluate the checkpoints of the enclosing training run on Gym benchmarks."""

    def validate_config(self, config: dict[str, Any]) -> None:
        # Reject unrecognised keys rather than ignoring them: a stale key changes
        # how a checkpoint is scored, and an ignored one gives no sign of it.
        _check_keys(config, "eval", _KNOWN_KEYS)
        for name, environment in (config.get("environments") or {}).items():
            _check_keys(
                (environment or {}).get("eval") or {},
                f"environments.{name}.eval",
                _KNOWN_KEYS - {"dependencies", "environments"},
            )

        config = _for_training_environment(config)
        if not config.get("eval_steps"):
            return
        if not config.get("checkpoint_path"):
            raise ValueError("eval.checkpoint_path is required when eval_steps is set")
        _eval_data_root(config)
        if config.get("format", "megatron") == "megatron" and not config.get("base_model"):
            raise ValueError(
                "eval.base_model is required to convert a Megatron checkpoint -- it "
                "supplies the tokenizer and architecture the export needs"
            )

    def _entry_defaults(
        self,
        campaign: dict[str, Any],
        environments: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the campaign half of an eval entry.

        Standalone campaigns get every non-structural key merged into each stage
        by the workflow runner.  An embedded eval is not expanded by the runner,
        so it does the same merge here: everything except what this stage
        consumes to build the entry in the first place.

        *environments* is already narrowed to ``eval.benchmarks``, and no
        ``_environment`` is set: a training workflow's ``-e`` names a training
        environment, not a benchmark, so it must not reach the benchmark filter.
        """
        consumed = {"benchmarks", "model_presets", "stages", "models", "checkpoints"}
        return {
            **{k: v for k, v in campaign.items() if k not in consumed},
            "environments": environments,
        }

    def plan(self, context: PlanningContext) -> StagePlan:
        """Compose checkpoint and baseline plans without submission."""
        from nvflow.execution import PlanningContext, StagePlan, plan_stage, rekey_tasks

        config = _for_training_environment(dict(context.config))
        eval_steps = config.get("eval_steps") or []
        if isinstance(eval_steps, int | str):
            eval_steps = [eval_steps]
        if not eval_steps:
            return StagePlan.noop(context.stage, "no eval_steps configured")

        base_output_dir = config.get("base_output_dir", "")
        eval_output_dir = config.get("eval_output_dir") or f"{base_output_dir}/eval"
        campaign = _load_eval_campaign(
            eval_output_dir,
            config.get("campaign", _DEFAULT_CAMPAIGN),
            eval_data_root=_eval_data_root(config),
            project_context=context.project_context or self.bound_project_context,
        )
        model_preset = _resolve_preset(campaign, config.get("preset"))
        environments = _scope_environments(
            campaign,
            config.get("benchmarks"),
            config.get("repeats"),
        )
        tasks: list[TaskPlan] = []

        def append_tasks(child: StagePlan, label: str) -> None:
            tasks.extend(
                rekey_tasks(
                    child.tasks,
                    stage=context.stage,
                    prefix=f"{context.stage}.{label}",
                )
            )

        defaults = self._entry_defaults(campaign, environments)
        for step in eval_steps:
            entry = {
                **defaults,
                "model": model_preset,
                "checkpoint_path": config.get("checkpoint_path"),
                "format": config.get("format", "megatron"),
                "base_model": config.get("base_model"),
                "conversion": config.get("conversion"),
                "_step": step,
                "_source_name": "step",
            }
            child = plan_stage(
                GymCheckpointEvaluatorStage(),
                PlanningContext(
                    stage=f"checkpoint-{step}",
                    config=entry,
                    launcher_config=entry,
                    target=context.target,
                    expname=f"{context.expname}-step{step}",
                    run_after=context.run_after,
                    project_context=context.project_context,
                    live=context.live,
                    filesystem=context.filesystem,
                ),
            )
            append_tasks(child, f"step-{step}")

        if baseline_model := config.get("baseline_model"):
            entry = {
                **defaults,
                "model": {**model_preset, "model": baseline_model},
                "_source_name": "baseline",
            }
            child = plan_stage(
                GymModelEvaluatorStage(),
                PlanningContext(
                    stage="baseline",
                    config=entry,
                    launcher_config=entry,
                    target=context.target,
                    expname=f"{context.expname}-baseline",
                    run_after=context.run_after,
                    project_context=context.project_context,
                    live=context.live,
                    filesystem=context.filesystem,
                ),
            )
            append_tasks(child, "baseline")

        if not tasks:
            return StagePlan.noop(
                context.stage,
                (
                    "filesystem inspection selected no evaluation work"
                    if context.live
                    else "evaluation configuration selected no work"
                ),
            )
        return StagePlan(stage=context.stage, tasks=tuple(tasks))
