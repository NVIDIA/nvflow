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
"""Characterization tests for maintained projects."""

from pathlib import Path

from nvflow.core import WorkflowDefinition
from nvflow.execution import WorkflowRunner

PROJECT_ROOT = Path(__file__).parents[2]


def test_maintained_workflows_resolve_project_owned_stage_types() -> None:
    cases = (
        ("projects/starter/workflows/generate.yaml", "generate_answer", "GenerateAnswerStage"),
        ("projects/finance/workflows/download/demo.yaml", "download", "DownloadSecFilingsStage"),
        ("projects/finance/workflows/sft/qwen3_4b.yaml", "training", "SFTStage"),
        (
            "projects/hopchain/workflows/image_filter/full.yaml",
            "image_filter",
            "ImageFilterStage",
        ),
        ("projects/finance/workflows/download/demo.yaml", "build_index", "BuildSecIndexStage"),
        (
            "projects/finance/workflows/download/demo.yaml",
            "prefetch_cache",
            "PrefetchCacheStage",
        ),
    )
    for relative_path, stage_name, class_name in cases:
        definition = WorkflowDefinition(PROJECT_ROOT / relative_path)
        assert definition.resolve_stage_class(stage_name).__name__ == class_name


def test_starter_experiment_name_uses_maintained_project_config() -> None:
    config = PROJECT_ROOT / "projects" / "starter" / "workflows" / "generate.yaml"
    runner = WorkflowRunner(str(config))

    assert runner.project_context.is_external is False
    assert runner.project == "starter"
    assert (
        runner.definition.get_expname("generate_answer", {}) == "answer_generation-generate_answer"
    )
    assert (
        runner.definition.get_expname("generate_answer", {"run_name": "trial"})
        == "answer_generation-generate_answer-trial"
    )
    assert runner.project_context.root == (PROJECT_ROOT / "projects" / "starter").resolve()
