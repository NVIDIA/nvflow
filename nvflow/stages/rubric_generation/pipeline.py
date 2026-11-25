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

"""Single-stage black box for portable rubric generation."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from nvflow.execution import PlannedExecutionStage

from .contracts import METHODS
from .generate import GenerateRubricsStage
from .input_adapters import normalize_input_format
from .normalize import NormalizeRubricInputStage
from .prepare_method import PrepareRubricMethodStage

if TYPE_CHECKING:
    from nvflow.core import BaseStage
    from nvflow.execution import PlanningContext, StagePlan, TaskPlan


def _path(root: str, suffix: str) -> str:
    return f"{root.rstrip('/')}/{suffix}"


def _stage_kwargs(config: dict[str, Any], override_key: str) -> dict[str, Any]:
    values = dict(config.get("stage_kwargs") or {})
    values.update(config.get(override_key) or {})
    return values


def rubric_output_file(config: dict[str, Any]) -> str:
    """Return the stable public rubric artifact path."""

    return str(config.get("output_file") or _path(str(config["output_dir"]), "rubrics.jsonl"))


def child_stage_configs(config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Build the private stage configurations used by the black box."""

    output_dir = str(config["output_dir"])
    normalized_file = _path(output_dir, "internal/normalize/normalized.jsonl")
    method_file = _path(output_dir, "internal/method/method-input.jsonl")

    normalize_config: dict[str, Any] = {
        "input_file": config["input_file"],
        "output_file": normalized_file,
        "audit_file": _path(output_dir, "normalization-audit.json"),
        "method": config["method"],
        "input_format": config.get("input_format", "canonical"),
        "field_map": config.get("field_map"),
        "max_records": config.get("max_records"),
        "allow_invalid_records": config.get("allow_invalid_records", False),
    }
    for key in ("container", "installation_command"):
        if config.get(key) is not None:
            normalize_config[key] = config[key]

    method_config: dict[str, Any] = {
        "input_file": normalized_file,
        "output_file": method_file,
        "method": config["method"],
        "strategy": config.get("strategy"),
        "rankings_file": _path(output_dir, "rankings.jsonl"),
        "audit_file": _path(output_dir, "ranking-audit.json"),
        "generation_dir": _path(output_dir, "internal/ranking-generation"),
        "stage_kwargs": _stage_kwargs(config, "ranking_stage_kwargs"),
    }
    if config.get("ranking_prompt_config") is not None:
        method_config["prompt_config"] = config["ranking_prompt_config"]
    if config.get("ranking_inline_args") is not None:
        method_config["inline_args"] = config["ranking_inline_args"]
    for key in ("container", "installation_command"):
        if config.get(key) is not None:
            method_config[key] = config[key]

    generation_config: dict[str, Any] = {
        "input_file": method_file,
        "output_file": rubric_output_file(config),
        "method": config["method"],
        "generation_dir": _path(output_dir, "internal/rubric-generation"),
        "stage_kwargs": _stage_kwargs(config, "generation_stage_kwargs"),
    }
    if config.get("prompt_config") is not None:
        generation_config["prompt_config"] = config["prompt_config"]
    if config.get("generation_inline_args") is not None:
        generation_config["inline_args"] = config["generation_inline_args"]

    return normalize_config, method_config, generation_config


class RubricGenerationStage(PlannedExecutionStage):
    """Normalize, prepare, generate, and validate rubrics behind one stage API."""

    def validate_config(self, config: dict[str, Any]) -> None:
        for field in ("input_file", "output_dir", "method"):
            if not config.get(field):
                raise ValueError(f"'{field}' is required for rubric generation")
        if config["method"] not in METHODS:
            raise ValueError(f"method must be one of {sorted(METHODS)}")
        normalize_input_format(config.get("input_format", "canonical"))
        for field in ("stage_kwargs", "ranking_stage_kwargs", "generation_stage_kwargs"):
            if config.get(field) is not None and not isinstance(config[field], dict):
                raise ValueError(f"{field} must be an object")

        child_stages: tuple[BaseStage, ...] = (
            NormalizeRubricInputStage(),
            PrepareRubricMethodStage(),
            GenerateRubricsStage(),
        )
        for child, child_config in zip(
            child_stages,
            child_stage_configs(config),
            strict=True,
        ):
            child.validate_config(child_config)

    def plan(self, context: PlanningContext) -> StagePlan:
        """Compose the private child stages into one exact public stage plan."""

        from nvflow.execution import StagePlan, plan_stage, rekey_tasks

        configs = child_stage_configs(dict(context.config))
        children: tuple[tuple[str, BaseStage], ...] = (
            ("normalize", NormalizeRubricInputStage()),
            ("prepare", PrepareRubricMethodStage()),
            ("generate", GenerateRubricsStage()),
        )
        tasks: list[TaskPlan] = []
        previous_expname: str | None = None

        for (label, child), child_config in zip(children, configs, strict=True):
            if context.project_context is not None:
                child.bind_project_context(context.project_context)
            child.validate_config(child_config)
            child_expname = f"{context.expname}-{label}"
            run_after = context.run_after if previous_expname is None else (previous_expname,)
            child_plan = plan_stage(
                child,
                replace(
                    context,
                    stage=label,
                    config=child_config,
                    launcher_config=child_config,
                    expname=child_expname,
                    run_after=run_after,
                ),
            )
            if child_plan is not None:
                tasks.extend(
                    rekey_tasks(
                        child_plan.tasks,
                        stage=context.stage,
                        prefix=f"{context.stage}.{label}",
                    )
                )
            previous_expname = child_expname

        if not tasks:
            return StagePlan.noop(context.stage, "rubric generation selected no work")
        return StagePlan(stage=context.stage, tasks=tuple(tasks))


__all__ = ["RubricGenerationStage", "child_stage_configs", "rubric_output_file"]
