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

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

import nvflow
from nvflow.core.workflow_definition import load_workflow
from nvflow.execution import PlanningContext, TargetBinding, internalize_dependencies
from nvflow.project import resolver as resolver_module
from nvflow.stages.catalog import get_stage_type, resolve_stage_type
from nvflow.stages.rubric_generation.contracts import normalize_file, normalize_record
from nvflow.stages.rubric_generation.criteria_dedup import dedup_criteria
from nvflow.stages.rubric_generation.generate import (
    GenerateRubricsStage,
    build_rubric_postprocess_command,
    default_rubric_prompt,
    rubric_arguments,
)
from nvflow.stages.rubric_generation.normalize import (
    NormalizeRubricInputStage,
    build_normalize_command,
)
from nvflow.stages.rubric_generation.parsing import normalize_rubric, parse_json_object
from nvflow.stages.rubric_generation.pipeline import (
    RubricGenerationStage,
    child_stage_configs,
)
from nvflow.stages.rubric_generation.postprocess_rankings import postprocess_rankings
from nvflow.stages.rubric_generation.postprocess_rubrics import postprocess_rubrics
from nvflow.stages.rubric_generation.prepare_direct import prepare_direct
from nvflow.stages.rubric_generation.prepare_method import PrepareRubricMethodStage


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _rubric_payload() -> dict:
    return {
        "rubric": [
            {
                "id": "R1",
                "type": "fact",
                "criterion": "States that the answer is 42.",
                "severity": 3,
                "must_pass": True,
                "evidence_summary": "The context states 42.",
            },
            {
                "id": "R2",
                "type": "conclusion",
                "criterion": "Provides the requested conclusion.",
                "severity": 2,
                "must_pass": False,
                "evidence_summary": "The question requests a conclusion.",
            },
        ]
    }


def _planning_context(stage: str, config: dict) -> PlanningContext:
    return PlanningContext(
        stage=stage,
        config=config,
        launcher_config=config,
        target=TargetBinding.resolve("test-cluster"),
        expname=f"rubric-smoke-{stage}",
        run_after=(),
    )


def test_normalize_simple_with_recipe_field_map() -> None:
    record = {
        "uuid": "row-1",
        "problem": "What is the answer?",
        "documents": [{"text": "The answer is 42."}],
        "gold": "42",
        "metadata": {"domain": "example"},
    }
    normalized = normalize_record(
        record,
        method="simple",
        field_map={
            "id": "uuid",
            "question": "problem",
            "context": "documents",
            "reference_answer": "gold",
        },
    )
    assert normalized["id"] == "row-1"
    assert json.loads(normalized["context"]) == [{"text": "The answer is 42."}]
    assert normalized["reference_answer"] == "42"
    assert len(normalized["source_sha256"]) == 64
    assert len(normalized["context_sha256"]) == 64


def test_normalize_openai_response_simple() -> None:
    normalized = normalize_record(
        {
            "rubric_context": "The source says the launch date is October 1.",
            "responses_create_params": {
                "input": [
                    {"role": "system", "content": "Answer using the source."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "When is the launch date?",
                            }
                        ],
                    },
                ]
            },
            "response": {
                "id": "resp-123",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "The launch date is October 1.",
                            }
                        ],
                    }
                ],
            },
        },
        method="simple",
        input_format="openai_response",
    )
    assert normalized["id"] == "resp-123"
    assert normalized["question"] == "When is the launch date?"
    assert normalized["context"] == "The source says the launch date is October 1."
    assert normalized["reference_answer"] == "The launch date is October 1."
    assert normalized["input_format"] == "openai_responses"


def test_normalize_openai_responses_extracts_final_submission() -> None:
    normalized = normalize_record(
        {
            "context": "Evidence",
            "input": "What is supported?",
            "response": {
                "output": [
                    {
                        "type": "function_call",
                        "name": "submit_final_result",
                        "arguments": json.dumps({"final_result": "Supported answer"}),
                    }
                ]
            },
        },
        method="simple",
        input_format="openai_responses",
    )
    assert normalized["question"] == "What is supported?"
    assert normalized["reference_answer"] == "Supported answer"


def test_normalize_openai_responses_openrubric_candidates() -> None:
    normalized = normalize_record(
        {
            "responses_create_params": {
                "input": [{"role": "user", "content": "Which answer is supported?"}]
            },
            "metadata": {
                "rubric": {
                    "context": "The evidence supports answer B.",
                    "reference_answer": "Answer B",
                    "preferred_index": 1,
                }
            },
            "candidate_responses": [
                {"output": [{"type": "message", "role": "assistant", "content": "Answer A"}]},
                {"output": [{"type": "message", "role": "assistant", "content": "Answer B"}]},
            ],
        },
        method="openrubric_contrastive",
        input_format="openai_responses",
    )
    assert normalized["candidate_answers"] == ["Answer A", "Answer B"]
    assert normalized["preferred_index"] == 1
    assert normalized["preferred_answer"] == "Answer B"


def test_normalize_openai_responses_requires_explicit_context() -> None:
    with pytest.raises(ValueError, match="context must be a non-empty string"):
        normalize_record(
            {
                "input": "Question?",
                "response": {"output_text": "Answer"},
            },
            method="simple",
            input_format="openai_responses",
        )


def test_normalize_openrubric_candidates_derives_preferred_answer() -> None:
    normalized = normalize_record(
        {
            "question": "Q?",
            "context": "Evidence",
            "reference_answer": "Good",
            "candidate_answers": ["Bad", "Good"],
            "preferred_index": 1,
        },
        method="openrubric_contrastive",
    )
    assert normalized["preferred_answer"] == "Good"
    assert normalized["preferred_index"] == 1
    assert normalized["candidate_count"] == 2
    assert normalized["candidate_indices"] == [0, 1]


def test_a_single_candidate_asks_for_a_written_negative() -> None:
    """One rollout is a question with nothing to rank, not a broken record."""

    normalized = normalize_record(
        {
            "question": "Q?",
            "candidate_answers": ["Good"],
            "preferred_index": 0,
        },
        method="openrubric_contrastive",
    )
    assert normalized["preferred_answer"] == "Good"
    assert normalized["candidate_count"] == 1
    assert normalized["synthetic_negative_required"] is True


def test_a_single_candidate_uses_a_supplied_negative() -> None:
    normalized = normalize_record(
        {
            "question": "Q?",
            "candidate_answers": ["Good"],
            "preferred_index": 0,
            "rejected_answer": "Bad",
        },
        method="openrubric_contrastive",
    )
    assert normalized["candidate_answers"] == ["Good", "Bad"]
    assert "synthetic_negative_required" not in normalized


def test_normalize_real_openrubric_ranking_aliases() -> None:
    """The historical ranking input uses source_id and selected_index aliases."""
    normalized = normalize_record(
        {
            "source_id": "source-7",
            "problem": "Which answer is supported?",
            "context": "The filing reports $42 million.",
            "reference_answer": "The answer is $42 million.",
            "candidate_answers": ["The answer is $21 million.", "The answer is $42 million."],
            "selected_index": 1,
            "chosen_answer": "The answer is $42 million.",
        },
        method="openrubric_contrastive",
    )
    assert normalized["id"] == "source-7"
    assert normalized["preferred_index"] == 1
    assert normalized["preferred_answer"] == "The answer is $42 million."


def test_normalize_openrubric_reindexes_after_blank_candidates() -> None:
    normalized = normalize_record(
        {
            "question": "Q?",
            "context": "Evidence",
            "reference_answer": "Good",
            "candidate_answers": ["", "Wrong", "Good"],
            "preferred_index": 2,
        },
        method="openrubric_contrastive",
    )
    assert normalized["candidate_answers"] == ["Wrong", "Good"]
    assert normalized["preferred_index"] == 1
    assert normalized["preferred_answer"] == "Good"


def test_normalize_openrubric_accepts_supplied_pair() -> None:
    normalized = normalize_record(
        {
            "question": "Q?",
            "context": "Evidence",
            "preferred_answer": "Supported",
            "rejected_answer": "Contradicted",
        },
        method="openrubric_contrastive",
    )
    assert normalized["reference_answer"] == "Supported"


def test_normalize_openrubric_reference_only_requests_synthetic_negative() -> None:
    normalized = normalize_record(
        {"question": "Q?", "context": "Evidence", "reference_answer": "A"},
        method="openrubric_contrastive",
    )
    assert normalized["candidate_answers"] == ["A"]
    assert normalized["preferred_index"] == 0
    assert normalized["preferred_answer"] == "A"
    assert normalized["synthetic_negative_required"] is True


def test_ranked_candidates_need_no_context() -> None:
    """GenSelect output is already ranked, so the ranking is the grounding."""

    normalized = normalize_record(
        {
            "question": "Q?",
            "candidate_answers": ["Better", "Worse"],
            "preferred_index": 0,
        },
        method="openrubric_contrastive",
    )
    assert normalized["context"] == ""
    assert normalized["preferred_answer"] == "Better"
    assert "synthetic_negative_required" not in normalized


def test_a_supplied_pair_needs_no_context() -> None:
    normalized = normalize_record(
        {"question": "Q?", "preferred_answer": "Right", "rejected_answer": "Wrong"},
        method="openrubric_contrastive",
    )
    assert normalized["context"] == ""
    assert normalized["candidate_answers"] == ["Right", "Wrong"]


def test_a_synthetic_negative_runs_without_context() -> None:
    """Weaker without a document, but the gold answer is enough to work from."""

    normalized = normalize_record(
        {"question": "Q?", "reference_answer": "A"},
        method="openrubric_contrastive",
    )
    assert normalized["context"] == ""
    assert normalized["synthetic_negative_required"] is True


def test_simple_still_needs_context() -> None:
    with pytest.raises(ValueError, match="context"):
        normalize_record({"question": "Q?", "reference_answer": "A"}, method="simple")


def test_normalize_file_writes_audit_and_rejects(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "normalized.jsonl"
    audit = tmp_path / "audit.json"
    _write_jsonl(
        source,
        [
            {"question": "Good?", "context": "Evidence", "reference_answer": "Yes"},
            {"question": "Missing context", "reference_answer": "No"},
        ],
    )
    stats = normalize_file(
        input_file=source,
        output_file=output,
        audit_file=audit,
        method="simple",
        allow_invalid_records=True,
    )
    assert stats["valid"] == 1
    assert stats["rejected"] == 1
    assert json.loads(audit.read_text(encoding="utf-8"))["method"] == "simple"
    assert len(_read_jsonl(tmp_path / "normalized_rejected.jsonl")) == 1


def test_normalize_file_can_bound_a_shared_two_arm_smoke_sample(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_jsonl(
        source,
        [
            {"question": f"Q{index}?", "context": "Evidence", "reference_answer": "A"}
            for index in range(5)
        ],
    )
    stats = normalize_file(
        input_file=source,
        output_file=tmp_path / "normalized.jsonl",
        audit_file=tmp_path / "audit.json",
        method="simple",
        max_records=2,
    )
    assert stats["total"] == 2
    assert stats["valid"] == 2
    assert [row["question"] for row in _read_jsonl(tmp_path / "normalized.jsonl")] == [
        "Q0?",
        "Q1?",
    ]


def test_parse_and_validate_rubric() -> None:
    fenced = "```json\n" + json.dumps(_rubric_payload()) + "\n```"
    rubric = normalize_rubric(parse_json_object(fenced))
    assert [item["id"] for item in rubric] == ["R1", "R2"]
    assert rubric[0]["must_pass"] is True


def test_rubric_carries_gold_and_lookalike() -> None:
    payload = _rubric_payload()
    payload["rubric"][0]["gold"] = "  42  "
    payload["rubric"][0]["lookalike"] = "24"
    rubric = normalize_rubric(payload)
    assert (rubric[0]["gold"], rubric[0]["lookalike"]) == ("42", "24")
    # Absent on R2, so it loses only its offline discrimination check.
    assert (rubric[1]["gold"], rubric[1]["lookalike"]) == ("", "")


def test_rubric_rejects_duplicate_criterion() -> None:
    payload = _rubric_payload()
    payload["rubric"][1]["criterion"] = payload["rubric"][0]["criterion"].upper()
    with pytest.raises(ValueError, match="duplicates"):
        normalize_rubric(payload)


def test_prepare_direct_for_simple_and_contrastive(tmp_path: Path) -> None:
    simple = tmp_path / "simple.jsonl"
    simple_out = tmp_path / "simple-out.jsonl"
    _write_jsonl(
        simple,
        [{"id": "1", "question": "Q?", "context": "C", "reference_answer": "A"}],
    )
    assert (
        prepare_direct(input_file=simple, output_file=simple_out, method="simple")["prepared"] == 1
    )

    contrastive = tmp_path / "contrastive.jsonl"
    contrastive_out = tmp_path / "contrastive-out.jsonl"
    _write_jsonl(
        contrastive,
        [
            {
                "id": "2",
                "question": "Q?",
                "context": "C",
                "reference_answer": "A",
                "preferred_answer": "A",
                "rejected_answer": "B",
            }
        ],
    )
    assert (
        prepare_direct(
            input_file=contrastive,
            output_file=contrastive_out,
            method="openrubric_contrastive",
        )["prepared"]
        == 1
    )


def test_postprocess_rankings_builds_contrastive_pair(tmp_path: Path) -> None:
    generated = tmp_path / "ranking-generated.jsonl"
    output = tmp_path / "contrastive.jsonl"
    rankings = tmp_path / "rankings.jsonl"
    audit = tmp_path / "ranking-audit.json"
    _write_jsonl(
        generated,
        [
            {
                "id": "1",
                "question": "Q?",
                "context": "C",
                "reference_answer": "Good",
                "candidate_answers": ["Good", "Wrong"],
                "preferred_index": 0,
                "ranking_generation": json.dumps(
                    {
                        "ranking": [0, 1],
                        "rejected_index": 1,
                        "material_contrast": True,
                        "reason": "Candidate 1 contradicts the context.",
                    }
                ),
            }
        ],
    )
    stats = postprocess_rankings(
        input_file=generated,
        output_file=output,
        rankings_file=rankings,
        audit_file=audit,
    )
    assert stats["accepted"] == 1
    record = _read_jsonl(output)[0]
    assert record["preferred_answer"] == "Good"
    assert record["rejected_answer"] == "Wrong"


def test_postprocess_rankings_excludes_equivalent_candidates(tmp_path: Path) -> None:
    generated = tmp_path / "ranking-generated.jsonl"
    _write_jsonl(
        generated,
        [
            {
                "id": "1",
                "question": "Q?",
                "context": "C",
                "reference_answer": "A",
                "candidate_answers": ["A", "Equivalent A"],
                "preferred_index": 0,
                "ranking_generation": json.dumps(
                    {
                        "ranking": [0, 1],
                        "rejected_index": 1,
                        "material_contrast": False,
                        "reason": "Equivalent",
                    }
                ),
            }
        ],
    )
    with pytest.raises(ValueError, match="no materially contrastive"):
        postprocess_rankings(
            input_file=generated,
            output_file=tmp_path / "contrastive.jsonl",
            rankings_file=tmp_path / "rankings.jsonl",
            audit_file=tmp_path / "audit.json",
        )
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert audit["no_material_contrast"] == 1


def test_equivalent_candidates_fall_back_to_a_synthetic_negative(tmp_path: Path) -> None:
    """Two equally correct answers used to lose the question entirely."""

    generated = tmp_path / "ranking-generated.jsonl"
    _write_jsonl(
        generated,
        [
            {
                "id": "1",
                "question": "Q?",
                "context": "C",
                "reference_answer": "A",
                "candidate_answers": ["A", "Equivalent A"],
                "preferred_index": 0,
                "ranking_generation": json.dumps(
                    {
                        "ranking": [0, 1],
                        "rejected_index": 1,
                        "material_contrast": False,
                        "synthetic_rejected_answer": "A with every figure wrong",
                        "reason": "Equivalent, so a negative was written",
                    }
                ),
            }
        ],
    )
    stats = postprocess_rankings(
        input_file=generated,
        output_file=tmp_path / "contrastive.jsonl",
        rankings_file=tmp_path / "rankings.jsonl",
        audit_file=tmp_path / "audit.json",
    )
    assert stats["accepted"] == 1
    assert stats["synthetic_fallback"] == 1
    assert stats["no_material_contrast"] == 0
    prepared = _read_jsonl(tmp_path / "contrastive.jsonl")[0]
    assert prepared["rejected_answer"] == "A with every figure wrong"
    assert prepared["negative_source"] == "synthetic_fallback"


def test_a_fallback_negative_equal_to_the_preferred_answer_is_refused(tmp_path: Path) -> None:
    generated = tmp_path / "ranking-generated.jsonl"
    _write_jsonl(
        generated,
        [
            {
                "id": "1",
                "question": "Q?",
                "context": "C",
                "reference_answer": "A",
                "candidate_answers": ["A", "Equivalent A"],
                "preferred_index": 0,
                "ranking_generation": json.dumps(
                    {
                        "ranking": [0, 1],
                        "rejected_index": 1,
                        "material_contrast": False,
                        "synthetic_rejected_answer": "A",
                        "reason": "Equivalent",
                    }
                ),
            }
        ],
    )
    with pytest.raises(ValueError, match="no materially contrastive"):
        postprocess_rankings(
            input_file=generated,
            output_file=tmp_path / "contrastive.jsonl",
            rankings_file=tmp_path / "rankings.jsonl",
            audit_file=tmp_path / "audit.json",
        )
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert audit["no_material_contrast"] == 1


def test_postprocess_rankings_accepts_controlled_synthetic_negative(tmp_path: Path) -> None:
    generated = tmp_path / "ranking-generated.jsonl"
    output = tmp_path / "contrastive.jsonl"
    _write_jsonl(
        generated,
        [
            {
                "id": "1",
                "question": "What revenue was reported?",
                "context": "The filing reports revenue of $42 million.",
                "reference_answer": "Revenue was $42 million.",
                "candidate_answers": ["Revenue was $42 million."],
                "preferred_index": 0,
                "synthetic_negative_required": True,
                "ranking_generation": json.dumps(
                    {
                        "ranking": [0],
                        "rejected_index": -1,
                        "synthetic_rejected_answer": "Revenue was $24 million.",
                        "material_contrast": True,
                        "reason": "The synthetic answer contradicts the disclosed revenue.",
                    }
                ),
            }
        ],
    )
    stats = postprocess_rankings(
        input_file=generated,
        output_file=output,
        rankings_file=tmp_path / "rankings.jsonl",
        audit_file=tmp_path / "audit.json",
    )
    assert stats["accepted"] == 1
    assert stats["synthetic_negative"] == 1
    record = _read_jsonl(output)[0]
    assert record["preferred_answer"] == "Revenue was $42 million."
    assert record["rejected_answer"] == "Revenue was $24 million."
    assert record["negative_source"] == "synthetic"


def test_postprocess_rubrics_writes_portable_artifact_and_manifest(tmp_path: Path) -> None:
    generated = tmp_path / "generated.jsonl"
    output = tmp_path / "rubrics.jsonl"
    _write_jsonl(
        generated,
        [
            {
                "id": "1",
                "question": "Q?",
                "context": "The answer is 42.",
                "reference_answer": "42",
                "metadata": {"domain": "test"},
                "source_sha256": "source-hash",
                "context_sha256": "context-hash",
                "rubric_generation": json.dumps(_rubric_payload()),
            }
        ],
    )
    stats = postprocess_rubrics(
        input_file=generated,
        output_file=output,
        method="simple",
        model_id="model-a",
        prompt_id="prompt-a",
    )
    assert stats["valid"] == 1
    assert stats["criteria"] == 2
    record = _read_jsonl(output)[0]
    assert record["rubric_schema_version"] == 1
    assert record["rubric_method"] == "simple"
    assert record["provenance"]["model"] == "model-a"
    manifest = json.loads((tmp_path / "rubrics.manifest.json").read_text(encoding="utf-8"))
    assert manifest["valid_rate"] == 1.0
    assert len(manifest["output_sha256"]) == 64


def test_shared_catalog_resolves_rubric_stages() -> None:
    expected = {
        "rubric.build": RubricGenerationStage,
        "rubric.normalize": NormalizeRubricInputStage,
        "rubric.prepare_method": PrepareRubricMethodStage,
        "rubric.generate": GenerateRubricsStage,
    }
    for name, stage_class in expected.items():
        assert get_stage_type(name).contract_versions == (1,)
        assert resolve_stage_type(name) is stage_class


def test_stage_validation_and_commands() -> None:
    normalize_config = {
        "input_file": "/data/input.jsonl",
        "output_file": "/out/normalized.jsonl",
        "audit_file": "/out/audit.json",
        "method": "simple",
        "input_format": "openai_response",
        "field_map": {"question": "problem"},
        "max_records": 20,
    }
    NormalizeRubricInputStage().validate_config(normalize_config)
    command = build_normalize_command(normalize_config)
    assert "--field_map_json" in command
    assert "--input_format openai_response" in command
    assert "--max_records 20" in command
    assert "nvflow.stages.rubric_generation.normalize" in command

    generation_config = {
        "input_file": "/out/method-input.jsonl",
        "output_file": "/out/rubrics.jsonl",
        "method": "simple",
        "stage_kwargs": {"model": "/models/rubric"},
    }
    GenerateRubricsStage().validate_config(generation_config)
    assert "postprocess_rubrics" in build_rubric_postprocess_command(generation_config)
    assert "++prompt_config=/nemo_run/code/" in rubric_arguments(generation_config)
    assert Path(default_rubric_prompt("simple")).is_file()
    assert Path(default_rubric_prompt("openrubric_contrastive")).is_file()


def test_black_box_stage_composes_private_steps() -> None:
    config = {
        "input_file": "/data/input.jsonl",
        "output_dir": "/out/rubrics",
        "method": "simple",
        "input_format": "openai_response",
        "stage_kwargs": {
            "model": "/models/rubric",
            "server_type": "sglang",
            "server_gpus": 1,
            "num_chunks": 1,
        },
    }
    stage = RubricGenerationStage()
    stage.validate_config(config)
    execution = stage.execution(_planning_context("build_rubrics", config))
    assert execution.disposition == "executable"
    assert all(task.stage == "build_rubrics" for task in execution.tasks)
    assert [task.operation for task in execution.tasks] == [
        "nemo_skills.run_cmd",
        "nemo_skills.run_cmd",
        "nemo_skills.generate",
        "nemo_skills.run_cmd",
    ]
    normalize_config, method_config, generation_config = child_stage_configs(config)
    assert normalize_config["input_format"] == "openai_response"
    assert method_config["input_file"].endswith("internal/normalize/normalized.jsonl")
    assert generation_config["output_file"] == "/out/rubrics/rubrics.jsonl"
    assert execution.tasks[1].dependencies[0].upstream.endswith("-normalize")
    assert execution.tasks[2].dependencies[0].upstream.endswith("-prepare")
    internalized = internalize_dependencies(execution.tasks)
    assert not any(dependency.external for task in internalized for dependency in task.dependencies)


def test_black_box_stage_composes_openrubric_ranking() -> None:
    config = {
        "input_file": "/data/input.jsonl",
        "output_dir": "/out/rubrics",
        "method": "openrubric_contrastive",
        "stage_kwargs": {
            "model": "/models/rubric",
            "server_type": "sglang",
            "server_gpus": 1,
            "num_chunks": 1,
        },
    }
    stage = RubricGenerationStage()
    stage.validate_config(config)
    execution = stage.execution(_planning_context("build_rubrics", config))
    assert [task.operation for task in execution.tasks] == [
        "nemo_skills.run_cmd",
        "nemo_skills.generate",
        "nemo_skills.run_cmd",
        "nemo_skills.generate",
        "nemo_skills.run_cmd",
    ]
    internalized = internalize_dependencies(execution.tasks)
    assert not any(dependency.external for task in internalized for dependency in task.dependencies)


@pytest.mark.parametrize(
    ("filename", "values"),
    [
        (
            "simple.yaml",
            {"question": "Q", "context": "C", "reference_answer": "A"},
        ),
        (
            "openrubric_contrastive.yaml",
            {
                "question": "Q",
                "context": "C",
                "preferred_answer": "A",
                "rejected_answer": "B",
            },
        ),
        (
            "rank_answers.yaml",
            {
                "question": "Q",
                "context": "C",
                "reference_answer": "A",
                "candidate_answers": "[]",
                "candidate_count": "0",
                "candidate_indices": "[]",
            },
        ),
    ],
)
def test_prompt_templates_format_literal_json(filename: str, values: dict[str, str]) -> None:
    prompt_path = Path(default_rubric_prompt("simple")).with_name(filename)
    prompt = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    assert '"rubric"' in prompt["system"].format(**values) or '"ranking"' in prompt[
        "system"
    ].format(**values)
    assert prompt["user"].format(**values)


def test_rubric_stages_produce_exact_execution_contracts() -> None:
    normalize_config = {
        "input_file": "/data/input.jsonl",
        "output_file": "/out/normalized.jsonl",
        "audit_file": "/out/normalize-audit.json",
        "method": "simple",
    }
    direct_config = {
        "input_file": "/out/normalized.jsonl",
        "output_file": "/out/method-input.jsonl",
        "method": "simple",
    }
    generation_config = {
        "input_file": "/out/method-input.jsonl",
        "output_file": "/out/rubrics.jsonl",
        "method": "simple",
        "stage_kwargs": {"model": "/models/rubric", "server_gpus": 1},
    }
    stages: tuple[tuple[str, Any, dict[str, Any]], ...] = (
        ("normalize", NormalizeRubricInputStage(), normalize_config),
        ("prepare_method", PrepareRubricMethodStage(), direct_config),
        ("generate", GenerateRubricsStage(), generation_config),
    )
    for name, stage, config in stages:
        execution = stage.execution(_planning_context(name, config))
        assert execution.disposition == "executable"
        assert execution.tasks
        assert all(task.commands for task in execution.tasks)


def test_openrubric_ranking_stage_requires_model_and_audit() -> None:
    with pytest.raises(ValueError, match="rankings_file"):
        PrepareRubricMethodStage().validate_config(
            {
                "input_file": "/data/normalized.jsonl",
                "output_file": "/out/method-input.jsonl",
                "method": "openrubric_contrastive",
                "strategy": "rank_candidates",
                "stage_kwargs": {"model": "/models/rubric"},
            }
        )


def test_finance_smoke_workflow_uses_only_shared_rubric_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nvflow, "__version__", "1.1.3.dev0")
    monkeypatch.setattr(resolver_module, "__version__", "1.1.3.dev0")
    workflow_path = (
        Path(__file__).parents[1]
        / "projects"
        / "finance"
        / "workflows"
        / "sdg"
        / "rubric-smoke-simple.yaml"
    )
    workflow = load_workflow(workflow_path)
    workflow.validate()
    for stage_name in workflow.pipeline_stages:
        implementation = workflow.stage_implementation(stage_name)
        assert implementation["kind"] == "shared"
        assert implementation["contract_version"] == 1
    assert workflow.pipeline_stages == ["build_rubrics"]
    for stage_name in workflow.pipeline_stages:
        stage = workflow.resolve_stage_class(stage_name)()
        stage.validate_config(workflow.stage_runtime_config(stage_name))


def test_rubric_grpo_smoke_base_is_non_runnable_and_uses_binary_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nvflow, "__version__", "1.1.3.dev0")
    monkeypatch.setattr(resolver_module, "__version__", "1.1.3.dev0")
    workflow_path = (
        Path(__file__).parents[1]
        / "projects"
        / "finance"
        / "workflows"
        / "grpo"
        / "rubric-smoke.yaml"
    )
    workflow = load_workflow(workflow_path)
    workflow.validate()
    assert workflow.config["workflow"]["runnable"] is False
    assert workflow.pipeline_stages == ["training"]
    assert (
        workflow.config["environments"]["finance_sec_search"]["gym"][
            "finance_sec_search_resources_server"
        ]["resources_servers"]["finance_sec_search"]["reward_mode"]
        == "binary"
    )


def test_controlled_rubric_smoke_arms_use_the_same_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nvflow, "__version__", "1.1.3.dev0")
    monkeypatch.setattr(resolver_module, "__version__", "1.1.3.dev0")
    workflows = Path(__file__).parents[1] / "projects" / "finance" / "workflows"

    simple = load_workflow(workflows / "sdg" / "rubric-smoke-simple.yaml")
    contrastive = load_workflow(workflows / "sdg" / "rubric-smoke-openrubric.yaml")
    simple.validate()
    contrastive.validate()
    assert simple.config["method"] == "simple"
    assert contrastive.config["method"] == "openrubric_contrastive"
    for key in (
        "rubric_input_file",
        "rubric_model_checkpoint",
        "rubric_output_root",
        "max_records",
    ):
        assert simple.config[key] == contrastive.config[key]
    assert simple.config["max_records"] == 20


def _criterion(text: str, *, must_pass: bool = False) -> dict[str, object]:
    return {
        "id": text[:8],
        "type": "fact",
        "criterion": text,
        "gold": "",
        "lookalike": "",
        "severity": 2,
        "must_pass": must_pass,
    }


def test_dedup_drops_restated_criteria() -> None:
    rubric = [
        _criterion("States total revenue of $8.9 billion for fiscal 2020.", must_pass=True),
        _criterion("Reports the total revenue of $8.9 billion for fiscal 2020."),
    ]
    kept, relations = dedup_criteria(rubric)
    assert len(kept) == 1
    assert kept[0]["must_pass"] is True
    assert relations["near_duplicate"] == 1
    assert relations["criteria_dropped"] == 1


def test_dedup_keeps_criteria_that_pin_different_figures() -> None:
    """The segment figure and the consolidated total are not duplicates."""
    rubric = [
        _criterion("States total revenue of $8.9 billion for fiscal 2020."),
        _criterion("States total revenue of $33.6 billion for fiscal 2020."),
    ]
    kept, relations = dedup_criteria(rubric)
    assert len(kept) == 2
    assert not relations


def test_dedup_keeps_criteria_that_name_different_entities() -> None:
    rubric = [
        _criterion("Identifies the Allstate Protection segment as the source."),
        _criterion("Identifies the Allstate Life segment as the source."),
    ]
    kept, _ = dedup_criteria(rubric)
    assert len(kept) == 2


def test_dedup_reports_but_keeps_a_second_look_at_one_figure() -> None:
    rubric = [
        _criterion("States the reserve balance is $8.9 billion."),
        _criterion(
            "Derives $8.9 billion by subtracting ceded amounts from the gross "
            "liability disclosed in the annuity footnote."
        ),
    ]
    kept, relations = dedup_criteria(rubric)
    assert len(kept) == 2
    assert relations["regrades_same_number"] == 1


def test_dedup_never_leaves_a_rubric_without_a_must_pass() -> None:
    rubric = [
        _criterion("Reports the total revenue of $8.9 billion for fiscal 2020."),
        _criterion("States total revenue of $8.9 billion for fiscal 2020.", must_pass=True),
    ]
    kept, _ = dedup_criteria(rubric)
    assert any(item["must_pass"] for item in kept)


def test_postprocess_records_dedup_in_the_manifest(tmp_path: Path) -> None:
    generation = json.dumps(
        {
            "rubric": [
                {
                    "id": "c1",
                    "type": "fact",
                    "criterion": "States total revenue of $8.9 billion for fiscal 2020.",
                    "gold": "$8.9 billion",
                    "lookalike": "$33.6 billion",
                    "severity": 3,
                    "must_pass": True,
                },
                {
                    "id": "c2",
                    "type": "fact",
                    "criterion": "Reports the total revenue of $8.9 billion for fiscal 2020.",
                    "gold": "$8.9 billion",
                    "lookalike": "$33.6 billion",
                    "severity": 2,
                    "must_pass": False,
                },
            ]
        }
    )
    input_file = tmp_path / "generated.jsonl"
    input_file.write_text(
        json.dumps({"id": "q1", "question": "q", "rubric_generation": generation}) + "\n",
        encoding="utf-8",
    )
    output_file = tmp_path / "rubrics.jsonl"

    stats = postprocess_rubrics(
        input_file=input_file, output_file=output_file, method="openrubric_contrastive"
    )

    assert stats["dedup_criteria"] is True
    assert stats["criteria_relations"]["near_duplicate"] == 1
    assert stats["criteria"] == 1
    record = json.loads(output_file.read_text(encoding="utf-8").splitlines()[0])
    assert len(record["rubric"]) == 1

    kept_all = tmp_path / "kept.jsonl"
    stats_off = postprocess_rubrics(
        input_file=input_file,
        output_file=kept_all,
        method="openrubric_contrastive",
        dedup_criteria_enabled=False,
    )
    assert stats_off["criteria"] == 2
    assert stats_off["criteria_relations"] == {}
