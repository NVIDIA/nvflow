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
"""Eval matrix: per-subject auto-expanded evaluator stages.

Registers a stage for each entry under an eval config's top-level ``models:``
section, and one per step for each entry under ``checkpoints:``. Each stage
cross-products its subject with the benchmark catalog and calls ``rollout()``
once per benchmark, yielding an M subjects x N benchmarks Slurm job matrix. The
CLI ``-e <benchmark>`` filter is forwarded to ``resolve_environments``.

A ``checkpoints:`` entry is scored exactly like a model, except its weights are
converted to HuggingFace first.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from nvflow.core import console
from nvflow.execution import PlannedExecutionStage

if TYPE_CHECKING:
    from nvflow.execution import PlanningContext, StagePlan
    from nvflow.execution.filesystem import TargetFilesystem

_UTILS = "nvfinance.utils.rl"
# Finance metrics renderer over canonical Gym repeats.
_EVAL_AGGREGATE_MODULE = f"{_UTILS}.eval_aggregate"


def _merge_mapping(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_mapping(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def build_gym_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """Build the model-level config the per-benchmark matrix expands.

    Injects model identity into the shared policy vLLM block and applies the
    per-model overrides; :func:`run_gym_eval_matrix` layers each benchmark's own
    profile on top. *config* is a ``models:`` entry (or an equivalent synthesised
    one) merged with the workflow defaults.
    """
    model = config.get("model", {}) or {}
    rollout = copy.deepcopy(config.get("rollout", {}) or {})

    rollout_models = copy.deepcopy(rollout.get("models", {}) or {})
    policy = dict(rollout_models.get("policy", {}) or {})
    policy["gym_model"] = _merge_mapping(
        policy.get("gym_model") or {},
        model.get("gym_model") or {},
    )
    for field in ("model", "base_url", "num_gpus", "server_nodes"):
        if model.get(field) is not None:
            policy[field] = model[field]
    policy_deployment = policy.get("deployment", "nvflow")
    policy["deployment"] = policy_deployment
    if policy_deployment == "gym":
        policy["num_gpus"] = int(policy.get("num_gpus", 1))
        policy["gym_model"].setdefault("vllm_serve_kwargs", {}).setdefault(
            "tensor_parallel_size",
            policy["num_gpus"],
        )
        policy.pop("base_url", None)
    rollout_models["policy"] = policy
    rollout["models"] = rollout_models

    if model.get("responses_create_params"):
        rollout["responses_create_params"] = {
            **(rollout.get("responses_create_params", {}) or {}),
            **model["responses_create_params"],
        }
    if model.get("num_repeats") is not None:
        rollout["num_repeats"] = model["num_repeats"]

    base_output_dir = config.get("base_output_dir") or config["data_root"]
    model_name = config.get("_source_name", "model")
    model_output_dir = config.get("model_output_dir")
    if model_output_dir is not None and (
        not isinstance(model_output_dir, str) or not model_output_dir
    ):
        raise ValueError("'model_output_dir' must be a non-empty string when set")
    model_output_dir = model_output_dir or f"{base_output_dir}/{model_name}"

    # Per-(model, benchmark) policy overrides (e.g. Qwen3 needs YaRN to reach
    # the agentic benchmark's 131K window). Merged into the env policy so they
    # win over the shared catalog profile.
    environments = config["environments"]
    bench_overrides = model.get("benchmark_models") or {}
    if bench_overrides:
        environments = copy.deepcopy(environments)
        for bench_name, endpoint_override in bench_overrides.items():
            if bench_name in environments:
                env = environments[bench_name]
                env_models = env.setdefault("models", {})
                env_policy = env_models.setdefault("policy", {})
                env_models["policy"] = _merge_mapping(
                    env_policy,
                    endpoint_override or {},
                )

    base_config: dict[str, Any] = {
        "contract_version": 1,
        "output_dir": f"{model_output_dir}/collect-rollouts",
        "container": config["container"],
        # Policy and judge vLLM servers. Omitting it would silently fall back to
        # the shared rollout default, which is GRPO's older training-matched pin.
        "vllm_container": config["vllm_container"],
        "installation_command": config.get("installation_command"),
        "environments": environments,
        "rollout": rollout,
    }
    if config.get("blade"):
        base_config["blade"] = copy.deepcopy(config["blade"])
    # Forward the -e benchmark filter so resolve_environments scopes the
    # catalog to the requested benchmark(s).
    if config.get("_environment"):
        base_config["_environment"] = config["_environment"]
    return base_config


class GymModelEvaluatorStage(PlannedExecutionStage):
    """Evaluate one model across the benchmark catalog.

    Config is the expanded ``models:`` entry merged with workflow defaults;
    model overrides live under a ``model:`` block so they don't clobber the
    forwarded ``rollout`` defaults.
    """

    @staticmethod
    def _validate_environment_filter(config: dict[str, Any]) -> None:
        """Reject an ``-e`` name that is not in the workflow's benchmark catalog."""
        selected = config.get("_environment")
        catalog = config.get("benchmarks") or {}
        if not selected or not catalog:
            return
        names = [selected] if isinstance(selected, str) else list(selected)
        unknown = [n for n in names if n not in catalog]
        if unknown:
            raise ValueError(
                f"Unknown benchmark(s): {', '.join(unknown)}. Available: {', '.join(catalog)}"
            )

    def validate_config(self, config: dict[str, Any]) -> None:
        for field in ("container", "environments"):
            if not config.get(field):
                raise ValueError(
                    f"'{field}' is required in eval model stage config "
                    f"(forwarded from the eval base config)"
                )
        if not config.get("base_output_dir") and not config.get("data_root"):
            raise ValueError("'base_output_dir' or 'data_root' is required")
        model = config.get("model", {}) or {}
        if not model.get("model") and not model.get("base_url"):
            raise ValueError(
                "model.model (local server) or model.base_url "
                "(external server) is required in the models: entry"
            )

    def plan(self, context: PlanningContext) -> StagePlan:
        """Build the model-by-benchmark rollout matrix without submission."""
        from nvflow.integrations.nemo_skills import stage_plan_from_jobs

        config = dict(context.config)
        self._validate_environment_filter(config)
        base_config = build_gym_model_config(config)
        planned = run_gym_eval_matrix(
            base_config,
            context.target.cluster,
            context.expname,
            list(context.run_after) or None,
            label=f"model '{config.get('_source_name', 'model')}'",
            inspect_live_state=context.live,
            filesystem=context.filesystem,
        )
        contribution = stage_plan_from_jobs(
            context.stage,
            planned,
            output_dir=base_config["output_dir"],
        )
        return contribution


def run_gym_eval_matrix(
    base_config: dict[str, Any],
    cluster: str,
    expname: str,
    run_after: list[str] | None = None,
    label: str = "model",
    inspect_live_state: bool = False,
    filesystem: TargetFilesystem | None = None,
) -> list[dict[str, Any]]:
    """Plan one rollout job per benchmark in the scoped environments.

    Eval drives serve + gym-eval-run through ``rollout()`` directly rather than
    ``CollectRolloutsStage`` (the GRPO training-data path).  Shared by the
    standalone model matrix and by embedded checkpoint eval; *label* only names
    the subject in log output.
    """
    from nvflow.integrations.nemo_gym.environments import (
        collect_target_config,
        resolve_scoped_environments,
        selected_environments,
    )
    from nvflow.integrations.nemo_gym.native_rollout import rollout

    scoped_environments = resolve_scoped_environments(base_config)
    if not scoped_environments:
        console.info(f"{label} is not scoped to the selected benchmark(s); skipping")
        return []

    normalization_config = {
        **base_config,
        "environments": scoped_environments,
    }
    normalization_config.pop("_environment", None)
    environments = selected_environments(normalization_config)
    if not environments:
        console.info(f"{label} has no enabled benchmark(s); skipping")
        return []

    console.status(f"Gym eval matrix: {label} x {len(environments)} benchmark(s)")
    console.blank()

    jobs: list[dict[str, Any]] = []
    for environment in environments:
        env_name = environment.name
        env_cfg = scoped_environments[env_name]
        env_config = collect_target_config(normalization_config, environment)
        env_rollout = env_config["rollout"]

        # vLLM checks prompt + max_tokens against the served window and rejects
        # the request when they don't fit. Every rollout would come back empty
        # and score zero, which reads as a bad model rather than a bad config.
        max_model_len = (
            env_rollout["models"]["policy"]
            .get("gym_model", {})
            .get("vllm_serve_kwargs", {})
            .get("max_model_len")
        )
        max_output_tokens = env_rollout["responses_create_params"].get("max_output_tokens")
        if max_model_len and max_output_tokens and max_output_tokens >= max_model_len:
            raise ValueError(
                f"benchmark '{env_name}': max_output_tokens ({max_output_tokens}) leaves "
                f"no room for the prompt in a {max_model_len}-token window. Lower "
                "responses_create_params.max_output_tokens or raise "
                "models.policy.gym_model.vllm_serve_kwargs.max_model_len."
            )

        # Gym owns repeat expansion; NVFlow's chunk count remains an independent
        # outer Slurm-sharding control.
        if env_cfg.get("repeats") is not None:
            repeats = int(env_cfg["repeats"])
            env_rollout["num_repeats"] = repeats
        if env_cfg.get("num_samples_in_parallel") is not None:
            env_rollout["num_samples_in_parallel"] = env_cfg["num_samples_in_parallel"]
        env_config["rollout"] = env_rollout

        # A benchmark may name its own output subtree, so two runs of the same
        # benchmark that differ only in configuration keep separate results.
        output_name = environment.output_name

        # Eval tail: render finance metrics from canonical Gym artifacts.
        planned = rollout(
            env_config,
            cluster,
            f"{expname}-{output_name}",
            run_after,
            report_module=_EVAL_AGGREGATE_MODULE,
            inspect_live_state=inspect_live_state,
            filesystem=filesystem,
        )
        for job in planned:
            job["_nvflow_experiment"] = f"{expname}-{output_name}"
        jobs.extend(planned)
    return jobs


class GymCheckpointEvaluatorStage(GymModelEvaluatorStage):
    """Score one step of a training run, converting it to HuggingFace first.

    Config is an expanded ``checkpoints:`` entry: the same thing a ``models:``
    entry is, except the weights are training output rather than a servable
    directory, so the path is produced by conversion instead of being given.
    """

    @staticmethod
    def _checkpoint_config(config: dict[str, Any]) -> dict[str, Any]:
        """Normalize one checkpoint entry for conversion planning."""
        return {
            **config,
            "checkpoint_path": config.get("checkpoint_path") or config.get("path"),
        }

    def validate_config(self, config: dict[str, Any]) -> None:
        if not (config.get("checkpoint_path") or config.get("path")):
            raise ValueError("'path' is required in a checkpoints: entry")
        if not config.get("eval_steps"):
            raise ValueError("'eval_steps' is required in a checkpoints: entry")
        checkpoint_format = config.get("format", "megatron")
        if checkpoint_format == "megatron" and not config.get("base_model"):
            raise ValueError(
                "'base_model' is required to convert a Megatron checkpoint -- it "
                "supplies the tokenizer and architecture the export needs"
            )

    def plan(self, context: PlanningContext) -> StagePlan:
        """Plan conversion and the same benchmark matrix used by model eval."""
        from nvflow.execution import StagePlan
        from nvflow.integrations.nemo_rl.checkpoints import plan_checkpoint_conversion
        from nvflow.integrations.nemo_skills import stage_plan_from_jobs

        config = dict(context.config)
        step = config["_step"]
        hf_model_path, conversion = plan_checkpoint_conversion(
            self._checkpoint_config(config),
            step,
            context.target.cluster,
            context.expname,
            list(context.run_after) or None,
            gpus_per_node=context.target_gpus_per_node("nemo-rl"),
            require_gpu_topology=context.engine == "ray",
        )
        run_after = [*context.run_after, conversion.name]

        eval_config = {
            **config,
            "model": {
                **(config.get("model") or {}),
                "model": hf_model_path,
            },
            "_source_name": f"{config.get('_source_name', 'checkpoint')}-{step}",
        }
        self._validate_environment_filter(eval_config)
        base_config = build_gym_model_config(eval_config)
        planned = run_gym_eval_matrix(
            base_config,
            context.target.cluster,
            context.expname,
            run_after or None,
            label=f"checkpoint '{step}'",
            inspect_live_state=context.live,
            filesystem=context.filesystem,
        )
        rollout_plan = stage_plan_from_jobs(
            context.stage,
            planned,
            output_dir=base_config["output_dir"],
        )
        if not rollout_plan.tasks:
            return StagePlan.noop(
                context.stage,
                (
                    "filesystem inspection selected no evaluation work"
                    if context.live
                    else "environment filter selected no evaluation work"
                ),
            )
        conversion = replace(conversion, task_id=f"{context.stage}.convert", stage=context.stage)
        return StagePlan(stage=context.stage, tasks=(conversion, *rollout_plan.tasks))
