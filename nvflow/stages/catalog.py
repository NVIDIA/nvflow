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
"""Curated, lazy catalog of platform-owned shared stage types."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nvflow.core.base_stage import BaseStage


@dataclass(frozen=True, slots=True)
class SharedStageType:
    """One stable declarative stage implementation exposed to projects."""

    name: str
    module: str
    class_name: str
    description: str
    documentation: str
    contract_versions: tuple[int, ...] = (1,)


_NEMO_GYM_DOCUMENTATION = (
    "https://github.com/NVIDIA/nvflow/blob/main/docs/platform/integrations/nemo-gym-data-plane.md"
)
_DOCUMENT_GROUNDED_DOCUMENTATION = (
    "https://github.com/NVIDIA/nvflow/blob/main/docs/guides/document-grounded-projects.md"
)
_TRAINING_DOCUMENTATION = (
    "https://github.com/NVIDIA/nvflow/blob/main/docs/guides/training-stages.md"
)
_RUBRIC_DOCUMENTATION = (
    "https://github.com/NVIDIA/nvflow/blob/main/docs/guides/rubric-generation.md"
)
_DATA_DOCUMENTATION = "https://github.com/NVIDIA/nvflow/blob/main/docs/guides/data-stages.md"


_STAGE_TYPES = {
    spec.name: spec
    for spec in (
        SharedStageType(
            "nemo_gym.prepare_benchmark",
            "nvflow.stages.nemo_gym.prepare_benchmark",
            "PrepareBenchmarkStage",
            "Prepare and render native NeMo Gym benchmarks.",
            _NEMO_GYM_DOCUMENTATION,
        ),
        SharedStageType(
            "nemo_gym.collate",
            "nvflow.stages.nemo_gym.collate_dataset",
            "CollateDatasetStage",
            "Collate project datasets with NeMo Gym agent routing.",
            _NEMO_GYM_DOCUMENTATION,
        ),
        SharedStageType(
            "nemo_gym.collect",
            "nvflow.stages.nemo_gym.collect_rollouts",
            "CollectRolloutsStage",
            "Collect canonical rollout, profile, and health artifacts.",
            _NEMO_GYM_DOCUMENTATION,
        ),
        SharedStageType(
            "nemo_gym.build_training_dataset",
            "nvflow.stages.nemo_gym.build_training_dataset",
            "BuildTrainingDatasetStage",
            "Publish profiled and selected training dataset records.",
            _NEMO_GYM_DOCUMENTATION,
        ),
        SharedStageType(
            "nemo_gym.rejudge",
            "nvflow.stages.nemo_gym.rejudge_rollouts",
            "RejudgeRolloutsStage",
            "Rejudge a completed native Gym rollout bundle.",
            _NEMO_GYM_DOCUMENTATION,
        ),
        SharedStageType(
            "nemo_gym.compare",
            "nvflow.stages.nemo_gym.compare_rollouts",
            "CompareRolloutsStage",
            "Compare two completed native Gym rollout bundles.",
            _NEMO_GYM_DOCUMENTATION,
        ),
        SharedStageType(
            "nemo_gym.train_validation_split",
            "nvflow.stages.nemo_gym.train_validation_split",
            "TrainValidationSplitStage",
            "Split each selected Gym environment's training data.",
            _TRAINING_DOCUMENTATION,
        ),
        SharedStageType(
            "nemo_rl.sft",
            "nvflow.stages.nemo_rl.sft",
            "SFTStage",
            "Supervised fine-tuning with NeMo-RL from project presets.",
            _TRAINING_DOCUMENTATION,
        ),
        SharedStageType(
            "nemo_rl.grpo",
            "nvflow.stages.nemo_rl.grpo",
            "GRPOStage",
            "GRPO training with NeMo-RL on NeMo Gym environments.",
            _TRAINING_DOCUMENTATION,
        ),
        SharedStageType(
            "data.train_validation_split",
            "nvflow.stages.data.train_validation_split",
            "TrainValidationSplitStage",
            "Split one JSONL dataset into train and validation files.",
            _TRAINING_DOCUMENTATION,
        ),
        SharedStageType(
            "data.dedup",
            "nvflow.stages.data.dedup",
            "DedupStage",
            "Remove exact and near-duplicate JSONL records.",
            _DATA_DOCUMENTATION,
        ),
        SharedStageType(
            "data.decontaminate",
            "nvflow.stages.data.decontaminate",
            "DecontaminateStage",
            "Drop JSONL records that overlap evaluation sets.",
            _DATA_DOCUMENTATION,
        ),
        SharedStageType(
            "data.filter",
            "nvflow.stages.data.filter_rows",
            "FilterRowsStage",
            "Drop JSONL records whose id appears in another file.",
            _DATA_DOCUMENTATION,
        ),
        SharedStageType(
            "nemo_gym.render",
            "nvflow.stages.nemo_gym.render_prompts",
            "RenderPromptsStage",
            "Render prompt templates and their tools into NeMo Gym rows.",
            _NEMO_GYM_DOCUMENTATION,
        ),
        SharedStageType(
            "sdg.document_grounded.preprocess",
            "nvflow.stages.sdg.document_grounded.dg_sdg_preprocess",
            "DGSDGPreprocessStage",
            "Prepare source documents for document-grounded generation.",
            _DOCUMENT_GROUNDED_DOCUMENTATION,
        ),
        SharedStageType(
            "sdg.document_grounded.generate_verified_questions",
            "nvflow.stages.sdg.document_grounded.generate_verified_questions",
            "GenerateVerifiedQuestionsStage",
            "Generate and verify document-grounded questions.",
            _DOCUMENT_GROUNDED_DOCUMENTATION,
        ),
        SharedStageType(
            "sdg.document_grounded.generate_answers",
            "nvflow.stages.sdg.document_grounded.generate_answers",
            "GenerateAnswersStage",
            "Generate document-grounded answers.",
            _DOCUMENT_GROUNDED_DOCUMENTATION,
        ),
        SharedStageType(
            "sdg.document_grounded.gym_genselect_answers",
            "nvflow.stages.sdg.document_grounded.gym_genselect_answers",
            "GymGenselectAnswersStage",
            "Select document-grounded answers through Gym.",
            _DOCUMENT_GROUNDED_DOCUMENTATION,
        ),
        SharedStageType(
            "sdg.document_grounded.evaluate_answers",
            "nvflow.stages.sdg.document_grounded.evaluate_answers",
            "EvaluateAnswersStage",
            "Evaluate document-grounded answers.",
            _DOCUMENT_GROUNDED_DOCUMENTATION,
        ),
        SharedStageType(
            "sdg.document_grounded.aggregate_answers",
            "nvflow.stages.sdg.document_grounded.aggregate_answers",
            "AggregateAnswersStage",
            "Aggregate document-grounded answer artifacts.",
            _DOCUMENT_GROUNDED_DOCUMENTATION,
        ),
        SharedStageType(
            "sdg.document_grounded.postprocess",
            "nvflow.stages.sdg.document_grounded.dgsdg_post_process",
            "DGSDGPostProcessStage",
            "Postprocess document-grounded datasets.",
            _DOCUMENT_GROUNDED_DOCUMENTATION,
        ),
        SharedStageType(
            "rubric.build",
            "nvflow.stages.rubric_generation.pipeline",
            "RubricGenerationStage",
            "Generate validated rubrics through one portable black-box stage.",
            _RUBRIC_DOCUMENTATION,
        ),
        SharedStageType(
            "rubric.normalize",
            "nvflow.stages.rubric_generation.normalize",
            "NormalizeRubricInputStage",
            "Normalize recipe records into the portable rubric input contract.",
            _RUBRIC_DOCUMENTATION,
        ),
        SharedStageType(
            "rubric.prepare_method",
            "nvflow.stages.rubric_generation.prepare_method",
            "PrepareRubricMethodStage",
            "Prepare direct or ranked contrastive inputs for rubric generation.",
            _RUBRIC_DOCUMENTATION,
        ),
        SharedStageType(
            "rubric.generate",
            "nvflow.stages.rubric_generation.generate",
            "GenerateRubricsStage",
            "Generate validated portable rubric artifacts.",
            _RUBRIC_DOCUMENTATION,
        ),
    )
}


def list_stage_types() -> tuple[SharedStageType, ...]:
    """Return shared stage types in stable name order without importing them."""
    return tuple(_STAGE_TYPES[name] for name in sorted(_STAGE_TYPES))


def get_stage_type(name: str) -> SharedStageType:
    """Return metadata for one curated stage type."""
    try:
        return _STAGE_TYPES[name]
    except KeyError:
        available = ", ".join(sorted(_STAGE_TYPES))
        raise ValueError(f"unknown shared stage type {name!r}; available: {available}") from None


def resolve_stage_type(name: str) -> type[BaseStage]:
    """Lazily import and return one curated shared stage class."""
    from nvflow.core import BaseStage

    spec = get_stage_type(name)
    stage_class = getattr(importlib.import_module(spec.module), spec.class_name)
    if not isinstance(stage_class, type) or not issubclass(stage_class, BaseStage):
        raise TypeError(
            f"shared stage type {name!r} resolved to a non-stage object: "
            f"{spec.module}.{spec.class_name}"
        )
    return stage_class


__all__ = [
    "SharedStageType",
    "get_stage_type",
    "list_stage_types",
    "resolve_stage_type",
]
