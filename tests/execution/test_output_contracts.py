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
"""Task-scoped output declaration contracts."""

from __future__ import annotations

from typing import Any

import pytest
from nvfinance.stages.download.prefetch_cache import PrefetchCacheStage
from nvfinance.stages.rl.apply_prompt_template import (
    ApplyPromptTemplateStage,
)
from nvfinance.stages.rl.convert_to_responses_api import (
    ConvertToResponsesAPIStage,
)
from nvfinance.stages.rl.validate_questions import ValidateQuestionsStage
from nvfinance.stages.sdg.create_seed_data import CreateSeedDataStage
from nvfinance.stages.sdg.genselect_answers import GenselectAnswersStage
from nvfinance.stages.sdg.map_questions_to_context import (
    MapQuestionsToContextStage,
)
from nvfinance.stages.sft.convert_to_messages import ConvertToMessagesStage
from nvfinance.stages.sft.prepare_for_sft import PrepareForSFTStage
from nvfinance.stages.sft.sequence_length_grouping import (
    SequenceLengthGroupingStage,
)
from nvfinance.stages.shared.data_transformation import (
    DataTransformationStage,
)

from nvflow.execution import (
    ArtifactDeclaration,
    PlanningContext,
    StagePlan,
    TargetBinding,
    declare_task_outputs,
    plan_generate,
    plan_run_cmd,
)
from nvflow.execution.operations import task_from_invocation
from nvflow.stages.data.train_validation_split import TrainValidationSplitStage
from nvflow.stages.nemo_gym import CollateDatasetStage, PrepareBenchmarkStage
from nvflow.stages.sdg.document_grounded.aggregate_answers import AggregateAnswersStage


@pytest.fixture
def fake_nemo_cli() -> None:
    """Compatibility fixture retained for output-contract call sites."""


def _context(stage: str, config: dict[str, Any]) -> PlanningContext:
    return PlanningContext(
        stage=stage,
        config=config,
        launcher_config=config,
        target=TargetBinding.resolve("test-cluster"),
        expname=f"workflow-{stage}",
        run_after=(),
    )


def _capture(stage: Any, name: str, config: dict[str, Any]) -> StagePlan:
    return stage.execution(_context(name, config)).to_plan()


def test_public_helper_declares_files_and_directories_without_stage_fallback() -> None:
    context = _context("custom", {"output_dir": "/wrong/stage-wide-base"})

    file_plan = plan_run_cmd(
        context,
        "python -m custom.worker",
        output_file="/right/result.jsonl",
    )
    both_plan = plan_run_cmd(
        context,
        "python -m custom.worker",
        output_dir="/right/results",
        output_file="/right/results/result.jsonl",
    )
    generate_file_plan = plan_generate(
        context,
        "++prompt_config=/prompts/custom.yaml",
        input_file="/input/data.jsonl",
        output_dir="/right/generated",
        model="/models/test",
        server_gpus=1,
    )
    generated_rubric = ArtifactDeclaration(
        name="output",
        path="/right/rubrics.jsonl",
        kind="file",
    )
    with declare_task_outputs(generated_rubric):
        generate_override_plan = plan_generate(
            context,
            "++prompt_config=/prompts/custom.yaml",
            input_file="/input/data.jsonl",
            output_dir="/right/generated-rubrics",
            model="/models/test",
            server_gpus=1,
        )
    captured_without_output = task_from_invocation(
        context,
        index=0,
        operation="nemo_skills.run_cmd",
        positional=(),
        keyword={"command": "python -m custom.worker"},
    )

    assert file_plan.tasks[0].outputs == (
        ArtifactDeclaration(name="output", path="/right/result.jsonl", kind="file"),
    )
    assert both_plan.tasks[0].outputs == (
        ArtifactDeclaration(name="output_dir", path="/right/results", kind="directory"),
        ArtifactDeclaration(
            name="output_file",
            path="/right/results/result.jsonl",
            kind="file",
        ),
    )
    assert generate_file_plan.tasks[0].outputs == (
        ArtifactDeclaration(
            name="generation",
            path="/right/generated/output.jsonl",
            kind="file",
        ),
    )
    assert generate_override_plan.tasks[-1].outputs == (generated_rubric,)
    assert captured_without_output.outputs == ()


def test_empty_output_context_preserves_conventional_declarations() -> None:
    context = _context("custom", {})

    with declare_task_outputs():
        task = task_from_invocation(
            context,
            index=0,
            operation="nemo_skills.run_cmd",
            positional=(),
            keyword={
                "command": "python -m custom.worker",
                "output_file": "/output/result.jsonl",
            },
        )

    assert task.outputs == (
        ArtifactDeclaration(name="output", path="/output/result.jsonl", kind="file"),
    )


def test_fanout_stages_declare_per_environment_outputs(
    fake_nemo_cli: None,
) -> None:
    environments = {
        "alpha": {
            "raw_train_data": "/raw/alpha",
            "prompt_template": "/prompts/alpha.yaml",
        },
        "beta": {
            "raw_train_data": "/raw/beta",
            "prompt_template": "/prompts/beta.yaml",
        },
    }

    prompt_plan = _capture(
        ApplyPromptTemplateStage(),
        "apply_prompt_template",
        {
            "input_dir": "/input",
            "output_dir": "/prompted",
            "container": "nemo-skills",
            "environments": environments,
        },
    )
    response_plan = _capture(
        ConvertToResponsesAPIStage(),
        "convert_to_responses_api",
        {
            "input_dir": "/prompted",
            "output_dir": "/responses",
            "container": "nemo-skills",
            "environments": environments,
        },
    )
    transform_plan = _capture(
        DataTransformationStage(),
        "data_transformation",
        {
            "output_dir": "/transformed",
            "environments": environments,
            "num_chunks": 3,
        },
    )

    assert [task.outputs[0].path for task in prompt_plan.tasks] == [
        "/prompted/alpha",
        "/prompted/beta",
    ]
    assert [task.outputs[0].path for task in response_plan.tasks] == [
        "/responses/alpha/final_result.jsonl",
        "/responses/beta/final_result.jsonl",
    ]
    assert [task.outputs[0].path for task in transform_plan.tasks] == [
        "/transformed/alpha/chunks",
        "/transformed/beta/chunks",
    ]
    assert all(
        output.path not in {"/prompted", "/responses", "/transformed"}
        for output in (
            *prompt_plan.outputs,
            *response_plan.outputs,
            *transform_plan.outputs,
        )
    )


def test_remaining_capture_stages_declare_actual_outputs(fake_nemo_cli: None) -> None:
    gym_plan = _capture(
        PrepareBenchmarkStage(),
        "eval_data",
        {
            "contract_version": 1,
            "benchmarks": ["secque"],
            "output_dir": "/gym-data",
            "gym_path": "/opt/gym",
            "container": "nemo-gym",
        },
    )
    prepare_plan = _capture(
        CollateDatasetStage(),
        "prepare_data",
        {
            "contract_version": 1,
            "output_dir": "/prepared",
            "gym_path": "/opt/gym",
            "container": "nemo-gym",
            "environments": {
                "alpha": {
                    "enabled": True,
                    "source_data": "/responses/alpha/final_result.jsonl",
                    "input_data": "/prepared/alpha/train.jsonl",
                    "agent_name": "alpha-agent",
                    "config_paths": ["resources_servers/alpha/config.yaml"],
                }
            },
        },
    )
    messages_plan = _capture(
        ConvertToMessagesStage(),
        "convert_to_messages",
        {"input_dir": "/input", "output_dir": "/messages"},
    )
    sft_plan = _capture(
        PrepareForSFTStage(),
        "prepare_for_sft",
        {"input_dir": "/input", "output_dir": "/sft", "add_token_counts": False},
    )
    grouping_plan = _capture(
        SequenceLengthGroupingStage(),
        "sequence_length_grouping",
        {"input_file": "/sft/final_result.jsonl", "output_dir": "/grouped"},
    )
    split_plan = TrainValidationSplitStage().plan(
        _context(
            "train_validation_split",
            {"input_file": "/grouped/data.jsonl", "output_dir": "/split"},
        )
    )

    assert gym_plan.outputs[0].path == "/gym-data/secque/gym_eval.jsonl"
    assert {output.path for output in prepare_plan.outputs} == {
        "/prepared/alpha/train.jsonl",
        "/prepared/alpha/gym_binding.json",
    }
    assert messages_plan.outputs[0].path == "/messages"
    assert sft_plan.outputs[0].path == "/sft/final_result.jsonl"
    assert grouping_plan.outputs[0].path == "/grouped"
    assert {output.path for output in split_plan.outputs} == {
        "/split/train.jsonl",
        "/split/val.jsonl",
        "/split/split.done",
    }


def test_prefetch_declares_the_cache_directory(
    fake_nemo_cli: None,
) -> None:
    plan = _capture(
        PrefetchCacheStage(),
        "prefetch_cache",
        {
            "gym_path": "/opt/gym",
            "container": "nemo-gym",
            "script": "scripts/prefetch.py",
            "cache_dir": "/cache/demo",
            "ticker_config": "/configs/demo.yaml",
        },
    )

    assert len(plan.tasks) == 1
    assert plan.tasks[0].outputs == (
        ArtifactDeclaration(
            name="sec-metadata-cache",
            path="/cache/demo",
            kind="directory",
        ),
    )


def test_file_stages_declare_primary_and_secondary_artifacts(
    fake_nemo_cli: None,
) -> None:
    map_plan = _capture(
        MapQuestionsToContextStage(),
        "map_questions_to_context",
        {
            "input_file": "/input/questions.jsonl",
            "output_file": "/mapped/final_result.jsonl",
            "filings_dir": "/filings",
            "filings_metadata": "/filings/metadata.parquet",
            "token_limit": 30_000,
        },
    )
    seed_plan = _capture(
        CreateSeedDataStage(),
        "create_seed_data",
        {
            "output_file": "/seed/questions.jsonl",
            "company_info_file": "/seed/companies.tsv",
            "sec_identity_email": "test@example.com",
            "sec_identity_company": "Example",
        },
    )
    aggregate_plan = _capture(
        AggregateAnswersStage(),
        "aggregate_answers",
        {
            "input_dir": "/evaluated",
            "output_file": "/aggregate/answers.jsonl",
        },
    )
    genselect_plan = _capture(
        GenselectAnswersStage(),
        "genselect_answers",
        {
            "input_dir": "/rollouts",
            "output_file": "/selected/final_result.jsonl",
            "prompt_config": "/prompts/select.yaml",
            "stage_kwargs": {
                "model": "/models/genselect",
                "server_gpus": 1,
            },
        },
    )

    assert map_plan.outputs == (
        ArtifactDeclaration(
            name="output",
            path="/mapped/final_result.jsonl",
            kind="file",
        ),
    )
    assert seed_plan.outputs == (
        ArtifactDeclaration(name="seed-data", path="/seed/questions.jsonl", kind="file"),
        ArtifactDeclaration(name="company-info", path="/seed/companies.tsv", kind="file"),
    )
    assert aggregate_plan.outputs == (
        ArtifactDeclaration(name="output", path="/aggregate/answers.jsonl", kind="file"),
    )
    assert genselect_plan.tasks[0].outputs == (
        ArtifactDeclaration(
            name="output",
            path="/selected/final_result_prepped.jsonl",
            kind="file",
        ),
    )
    assert genselect_plan.tasks[1].outputs == (
        ArtifactDeclaration(
            name="output_dir",
            path="/selected/final_result",
            kind="directory",
        ),
        ArtifactDeclaration(
            name="output_file",
            path="/selected/final_result.jsonl",
            kind="file",
        ),
    )


def test_validate_questions_declares_intermediate_and_final_outputs(
    fake_nemo_cli: None,
) -> None:
    plan = _capture(
        ValidateQuestionsStage(),
        "validate_questions",
        {
            "source_data": "/source",
            "output_dir": "/validated",
            "prompt_config": "/prompts/validate.yaml",
            "environments": {"alpha": {}},
            "stage_kwargs": {
                "model": "/models/judge",
                "server_gpus": 1,
            },
        },
    )

    assert plan.tasks[0].outputs == (
        ArtifactDeclaration(
            name="output",
            path="/validated/alpha/phase1_regex/prefiltered.jsonl",
            kind="file",
        ),
    )
    assert plan.tasks[1].outputs == (
        ArtifactDeclaration(
            name="output_dir",
            path="/validated/alpha/phase2_llm",
            kind="directory",
        ),
        ArtifactDeclaration(
            name="output_file",
            path="/validated/alpha/final_result.jsonl",
            kind="file",
        ),
    )
