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

import argparse
import json
from types import SimpleNamespace
from typing import Literal

from nemo_skills.inference.model.vllm import process_image_content
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_none

from nvflow.recipes.multimodal.utils import hopchain_difficulty_filter_preprocess as difficulty_mod
from nvflow.recipes.multimodal.utils import hopchain_reconcile_llm_judges as reconcile_mod
from nvflow.recipes.multimodal.utils import hopchain_run_llm_judge as run_judge_mod
from nvflow.recipes.multimodal.utils.hopchain_sdg_models import (
    GeneratedHopChainQueryMetadata,
    LLMJudgeAnswer,
    LLMJudgeEvaluationRecord,
    ReconciledHopChainQuery,
    VerificationMetadata,
    VerifiedHopChainQuery,
)


def make_candidate(query_id: str) -> VerifiedHopChainQuery:
    """Build a minimal verified candidate record for judge tests."""
    return VerifiedHopChainQuery(
        query_id=query_id,
        image_id=f"image-{query_id}",
        combination_id=f"combination-{query_id}",
        image_file_name="image.jpg",
        image_fullpath="/tmp/image.jpg",
        question="How many objects are visible?",
        hypothetical_answer="2",
        involved_instance_ids=["instance-1", "instance-2", "instance-3"],
        hop_count=2,
        query_metadata=GeneratedHopChainQueryMetadata(
            primary_capability="counting",
            instance_chain="instance-1 -> instance-2 -> instance-3",
            reasoning_hops=[],
            design_rationale="test candidate",
            answer_type="count",
            uses_all_instances=True,
        ),
        raw_generation="raw generation",
        verification_status="accepted",
        rejection_reasons=[],
        verification_metadata=VerificationMetadata(
            numeric_answer=True,
            references_all_instances=True,
        ),
    )


def make_judge_record(
    candidate: VerifiedHopChainQuery,
    *,
    judge_name: str,
    status: Literal["parsed", "parse_error", "api_error"] = "parsed",
    answer: str = "2",
    normalized_answer: str = "2",
    judge_error: str | None = None,
) -> LLMJudgeEvaluationRecord:
    """Build a judge output record for reconciliation and runner tests."""
    return LLMJudgeEvaluationRecord(
        **candidate.model_dump(),
        llm_judge=LLMJudgeAnswer(
            judge_name=judge_name,
            provider="openai",
            model="gpt-test",
            answer=answer,
            normalized_answer=normalized_answer,
            confidence="high" if status == "parsed" else "unknown",
            reasoning="test reasoning" if status == "parsed" else None,
            raw_response="raw response",
        ),
        judge_status=status,
        judge_error=judge_error,
    )


def test_main_converts_unhandled_worker_failure_into_output_record(tmp_path, monkeypatch):
    candidates = [make_candidate("query-1"), make_candidate("query-2")]
    input_path = tmp_path / "candidates.jsonl"
    output_path = tmp_path / "judge_output.jsonl"
    summary_path = tmp_path / "summary.json"
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Question: {question}")
    input_path.write_text("\n".join(candidate.model_dump_json() for candidate in candidates) + "\n")

    monkeypatch.setattr(
        run_judge_mod,
        "parse_args",
        lambda: argparse.Namespace(
            input=str(input_path),
            output=str(output_path),
            summary=str(summary_path),
            prompt=str(prompt_path),
            judge_name="judge-a",
            provider="openai",
            model="gpt-test",
            api_key_name="OPENAI_API_KEY",
            api_base=None,
            max_image_dimension=1536,
            temperature=0.0,
            top_p=1.0,
            reasoning_effort=None,
            timeout_seconds=30.0,
            max_retries=0,
            max_workers=2,
        ),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

    def fake_run_single_judge(**kwargs):
        task = kwargs["task"]
        if task.line_num == 2:
            raise RuntimeError("worker exploded")
        return run_judge_mod.JudgeTaskOutput(
            line_num=task.line_num,
            record=make_judge_record(task.record, judge_name="judge-a"),
        )

    monkeypatch.setattr(run_judge_mod, "run_single_judge", fake_run_single_judge)

    run_judge_mod.main()

    output_records = [
        LLMJudgeEvaluationRecord.model_validate_json(line)
        for line in output_path.read_text().splitlines()
        if line.strip()
    ]
    summary = json.loads(summary_path.read_text())

    assert len(output_records) == 2
    assert [record.query_id for record in output_records] == ["query-1", "query-2"]
    assert [record.judge_status for record in output_records] == ["parsed", "api_error"]
    assert output_records[1].judge_error == "Unhandled worker failure: worker exploded"
    assert summary["total_records_processed"] == 2
    assert summary["status_counts"] == {"parsed": 1, "api_error": 1}


def test_run_single_judge_retries_transient_completion_error(monkeypatch):
    candidate = make_candidate("query-1")
    task = run_judge_mod.JudgeTaskInput(line_num=1, record=candidate)
    attempts = {"count": 0}

    monkeypatch.setattr(
        run_judge_mod, "build_messages", lambda *_args, **_kwargs: [{"role": "user"}]
    )
    monkeypatch.setattr(
        run_judge_mod,
        "build_completion_retryer",
        lambda **kwargs: Retrying(
            retry=retry_if_exception_type(Exception),
            stop=stop_after_attempt(kwargs["max_retries"] + 1),
            wait=wait_none(),
            reraise=True,
        ),
    )

    def fake_completion(**_kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("transient failure")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="judge response"))],
            usage={"completion_tokens": 7},
        )

    monkeypatch.setattr(run_judge_mod, "completion", fake_completion)
    monkeypatch.setattr(
        run_judge_mod,
        "parse_judge_response_text",
        lambda _text: SimpleNamespace(
            final_answer="2",
            normalized_answer="2",
            confidence="high",
            reasoning="retry succeeded",
        ),
    )

    result = run_judge_mod.run_single_judge(
        task=task,
        prompt_template="Question: {question}",
        litellm_model="openai/gpt-test",
        judge_name="judge-a",
        provider="openai",
        model="gpt-test",
        api_key="test-api-key",
        api_base=None,
        max_image_dimension=1536,
        temperature=0.0,
        top_p=1.0,
        reasoning_effort=None,
        timeout_seconds=30.0,
        max_retries=1,
    )

    assert attempts["count"] == 2
    assert result.record.judge_status == "parsed"
    assert result.record.llm_judge.normalized_answer == "2"


def test_reconcile_missing_judge_record_avoids_redundant_incomplete_reason():
    candidate = make_candidate("query-1")
    reconciled = reconcile_mod.reconcile_candidate_record(
        candidate,
        [
            make_judge_record(candidate, judge_name="judge-a"),
            None,
        ],
    )

    assert "missing_judge_record" in reconciled.llm_judge_rejection_reasons
    assert "incomplete_judge_answers" not in reconciled.llm_judge_rejection_reasons


def test_reconcile_invalid_present_judge_answer_keeps_incomplete_reason():
    candidate = make_candidate("query-1")
    reconciled = reconcile_mod.reconcile_candidate_record(
        candidate,
        [
            make_judge_record(candidate, judge_name="judge-a"),
            make_judge_record(
                candidate,
                judge_name="judge-b",
                status="parse_error",
                answer="",
                normalized_answer="",
                judge_error="parse failure",
            ),
        ],
    )

    assert "judge_status_judge-b_parse_error" in reconciled.llm_judge_rejection_reasons
    assert "incomplete_judge_answers" in reconciled.llm_judge_rejection_reasons


def test_difficulty_preprocess_emits_local_path_for_nemo_skills(tmp_path, monkeypatch):
    image_path = tmp_path / "image.jpg"
    image_path.touch()
    candidate = make_candidate("query-1")
    reconciled = ReconciledHopChainQuery(
        **candidate.model_dump(exclude={"image_fullpath"}),
        image_fullpath=str(image_path),
        llm_judge_names=["judge-a"],
        llm_judge_answers=[],
        llm_judge_consensus_answer="2",
        llm_judge_consensus_normalized_answer="2",
        llm_judge_consensus_matches_hypothetical_answer=True,
        llm_judge_reconciliation_status="accepted",
    )
    input_path = tmp_path / "reconciled.jsonl"
    output_path = tmp_path / "difficulty-input.jsonl"
    prompt_path = tmp_path / "prompt.txt"
    input_path.write_text(reconciled.model_dump_json() + "\n")
    prompt_path.write_text("Question: {question}")

    monkeypatch.setattr(
        difficulty_mod,
        "parse_args",
        lambda: argparse.Namespace(
            input=str(input_path),
            output=str(output_path),
            prompt=str(prompt_path),
            k=1,
            enable_thinking=False,
        ),
    )

    difficulty_mod.main()

    request = json.loads(output_path.read_text())
    image_url = request["messages"][0]["content"][0]["image_url"]["url"]
    assert image_url == str(image_path)

    processed_content = process_image_content(
        request["messages"][0]["content"], str(output_path.parent)
    )
    assert isinstance(processed_content, list)
    assert processed_content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
