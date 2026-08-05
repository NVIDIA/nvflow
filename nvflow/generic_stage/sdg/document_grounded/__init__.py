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
"""Shared DG-SDG stages and per-recipe registration helper."""

from nvflow.core import StageRegistry

from .aggregate_answers import AggregateAnswersStage
from .dg_sdg_preprocess import DGSDGPreprocessStage
from .dgsdg_post_process import DGSDGPostProcessStage
from .evaluate_answers import EvaluateAnswersStage
from .generate_answers import GenerateAnswersStage
from .generate_verified_questions import GenerateVerifiedQuestionsStage
from .gym_genselect_answers import GymGenselectAnswersStage

WORKFLOW = "document_grounded_sdg"
SHARED_STAGES: list[tuple[type, str]] = [
    (AggregateAnswersStage, "aggregate_answers"),
    (EvaluateAnswersStage, "evaluate_answers"),
    (GymGenselectAnswersStage, "gym_genselect_answers"),
    (GenerateVerifiedQuestionsStage, "generate_verified_questions"),
    (GenerateAnswersStage, "generate_answers"),
    (DGSDGPreprocessStage, "dg_sdg_preprocess"),
    (DGSDGPostProcessStage, "dgsdg_post_process"),
]


def register_for_recipe(recipe: str) -> None:
    """Register all shared DG-SDG stages for a concrete recipe name."""
    for stage_class, stage_name in SHARED_STAGES:
        if StageRegistry.has(recipe=recipe, workflow=WORKFLOW, stage=stage_name):
            raise ValueError(
                f"register_for_recipe({recipe!r}) would re-register "
                f"{recipe}.{WORKFLOW}.{stage_name}. "
                "This usually means old per-recipe shim modules are still imported "
                "or the helper was called twice."
            )
        StageRegistry.register(recipe=recipe, workflow=WORKFLOW, stage=stage_name)(stage_class)
