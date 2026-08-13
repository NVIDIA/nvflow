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
"""Contract tests for the finance grounding-verification stage."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

from nvflow.grounding_verifier.decomposer import RuleBasedDecomposer
from nvflow.grounding_verifier.embedder import DEFAULT_EMBEDDING_MODEL
from nvflow.grounding_verifier.evaluator import GroundingVerifierConfig, GroundingVerifierEvaluator
from nvflow.grounding_verifier.nli import DEFAULT_NLI_MODEL, _normalize_nli_label
from nvflow.grounding_verifier.router import EmbeddingSourceRouter
from nvflow.grounding_verifier.types import EvidenceChunk, NLIResult
from nvflow.recipes.finance.stages.rl.evaluate_grounding import (
    EvaluateGroundingStage,
    build_evaluate_command,
)
from nvflow.recipes.finance.utils.rl.grounding_verifier import (
    FinanceEvaluatorConfig,
    _is_parse_success,
    evaluate_row,
    evaluate_seed,
    parse_rollout_line,
)

FILING_URL = "https://www.sec.gov/Archives/edgar/data/320193/000032019322000108/aapl-20220924.htm"
ANSWER = "Apple had approximately 164,000 full-time equivalent employees in 2022."
EVIDENCE = (
    "As of September 24, 2022, Apple Inc. had approximately 164,000 full-time equivalent employees."
)
CAPTURED_TRACE = Path(__file__).parent / "fixtures/grounding_verifier/native_finance_trace.json"


class _Embedder:
    @property
    def dimension(self) -> int:
        return 8

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[float((sum(map(ord, text)) + index) % 17) for index in range(8)] for text in texts]


class _NLI:
    def __init__(self, label: str = "entailment", *, fail: bool = False) -> None:
        self.label = label
        self.fail = fail

    def score(self, *, premise: str, hypothesis: str) -> NLIResult:
        if self.fail:
            raise RuntimeError("model failure")
        return NLIResult(label=self.label, score=0.99, probabilities=((self.label, 0.99),))


def _evaluator(
    label: str = "entailment",
    *,
    fail: bool = False,
) -> GroundingVerifierEvaluator:
    return GroundingVerifierEvaluator(
        decomposer=RuleBasedDecomposer(),
        router=EmbeddingSourceRouter(_Embedder()),
        nli_scorer=_NLI(label, fail=fail),
    )


def _config() -> FinanceEvaluatorConfig:
    return FinanceEvaluatorConfig(
        grounding_config=GroundingVerifierConfig(),
        routing_model_revision="routing-revision",
        nli_model_revision="nli-revision",
    )


def _call(name: str, call_id: str, **arguments) -> dict:
    return {
        "type": "function_call",
        "name": name,
        "call_id": call_id,
        "arguments": json.dumps(arguments),
    }


def _result(call_id: str, output) -> dict:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": json.dumps(output),
    }


def _trace(
    answer: str = ANSWER,
    *,
    parse_output: str | None = None,
    url: str = FILING_URL,
    key: str = "aapl_10k",
) -> dict:
    parse_output = parse_output or (
        f"SUCCESS: The result has been saved to the data storage under the key: {key}."
    )
    filing = {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "form": "10-K",
        "filing_date": "2022-10-28",
        "report_date": "2022-09-24",
        "accession_number": "0000320193-22-000108",
        "filing_url": url,
    }
    return {
        "uuid": "rollout-1",
        "response": {
            "id": "response-1",
            "output": [
                _call("sec_filing_search", "search", ticker="AAPL", form_types=["10-K"]),
                _result("search", {"results": json.dumps([filing])}),
                _call("parse_html_page", "parse", url=url, key=key),
                _result("parse", {"results": parse_output}),
                _call(
                    "retrieve_information",
                    "retrieve",
                    prompt=f"Find employees {{{{{key}}}}}",
                    input_character_ranges=[{"start": 30000, "end": 45000}],
                ),
                _result("retrieve", {"results": EVIDENCE}),
                _call("submit_final_result", "submit", final_result=answer),
            ],
        },
    }


def _decision(answer: str, label: str = "entailment") -> dict:
    row = _trace(answer)
    return evaluate_row(json.dumps(row), row, 0, _evaluator(label), _config())["verdict"]


def test_captured_native_nvflow_trace_to_atomic_sidecar(tmp_path):
    row = json.loads(CAPTURED_TRACE.read_text())
    source = tmp_path / "output-rs0.jsonl"
    target = tmp_path / "grounding-verifier-rs0.jsonl"
    original = json.dumps(row) + "\n"
    source.write_text(original, encoding="utf-8")
    Path(f"{source}.done").touch()

    assert evaluate_seed(str(source), str(target), 0, _evaluator(), _config()) == 1
    output = json.loads(target.read_text())
    chunk = output["evidence"][0]
    assert output["verdict"]["status"] == "allow"
    assert output["models"]["routing_model_revision"] == "routing-revision"
    assert chunk["source_id"] == (
        "sec:cik=0000320193:accession=000032019322000108:doc=aapl-20220924.htm"
    )
    assert row["fixture_provenance"]["source_sha256"] == (
        "52a99d90119aa18d572c3eca4776c1edd4dff4c65bd79daffc2458b3d328d79b"
    )
    assert chunk["char_range"] == {"start": 30000, "end": 45000}
    assert source.read_text() == original
    assert Path(f"{target}.done").is_file()


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        ("Apple had approximately 165,000 full-time equivalent employees in 2022.", None),
        (
            "Microsoft Corporation (MSFT) had approximately 164,000 full-time "
            "equivalent employees in 2022.",
            "entity_conflation",
        ),
        ("Apple reported revenue of 164,000 in 2022.", "financial_metric_mismatch"),
        ("Apple had approximately 164,000 full-time equivalent employees in 2023.", None),
        (
            "Per SEC accession 0000950170-24-087843, Apple had approximately 164,000 "
            "full-time equivalent employees in 2022.",
            "conflation",
        ),
    ],
)
def test_fabrications_and_conflations_block(answer, reason):
    decision = _decision(answer, "entailment")
    assert decision["status"] == "block"
    if reason:
        assert decision["reason"] == reason


@pytest.mark.parametrize(("label", "status"), [("neutral", "block"), ("contradiction", "block")])
def test_fail_closed_nli_policy(label, status):
    assert _decision("Apple's workforce changed in 2022.", label)["status"] == status


def test_matching_numeric_fact_can_override_neutral():
    assert _decision(ANSWER, "neutral")["status"] == "allow"
    assert _decision(ANSWER.replace("164,000", "165,000"), "neutral")["status"] == "block"


def test_neutral_direct_support_requires_entity_metadata():
    evidence = EvidenceChunk(
        chunk_id="amazon",
        source_id="sec:cik=0001018724:accession=000101872425000004:doc=amzn.htm",
        text="Amazon reported net income of $59,248 million in 2024.",
    )
    decision = _evaluator("neutral").evaluate(
        "Microsoft reported net income of $59,248 million in 2024.", [evidence]
    )
    assert decision.status == "block"


def test_model_failure_is_unavailable():
    row = _trace()
    result = evaluate_row(json.dumps(row), row, 0, _evaluator(fail=True), _config())
    assert result["verdict"]["status"] == "unavailable"
    assert result["verdict"]["errors"]


@pytest.mark.parametrize("raw", ["", "{broken", "[]"])
def test_every_input_line_has_a_fail_closed_sidecar_row(raw):
    result = evaluate_row(raw, {}, 0, _evaluator(), _config())
    assert result["verdict"]["status"] == "unavailable"


def test_incomplete_and_failed_attribution_never_allows():
    failed = _trace(parse_output="HTTPSConnectionPool: timed out")
    sidecar = parse_rollout_line(json.dumps(failed))
    assert sidecar.trace is not None
    assert sidecar.trace.evidence[0].source_id is None
    result = evaluate_row(json.dumps(failed), failed, 0, _evaluator(), _config())
    assert result["verdict"]["status"] == "block"
    assert result["verdict"]["reason"] == "no_source"


def test_failed_parse_retry_preserves_the_last_verified_source():
    row = _trace()
    output = row["response"]["output"]
    output[4:4] = [
        _call("parse_html_page", "retry", url=FILING_URL.replace("108", "109"), key="aapl_10k"),
        _result("retry", {"results": "connection timed out"}),
    ]
    trace = parse_rollout_line(json.dumps(row)).trace
    assert trace is not None
    assert "accession=000032019322000108" in trace.evidence[0].source_id


def test_multiple_search_records_never_mix_filing_fields():
    row = _trace()
    second_url = FILING_URL.replace("000032019322000108", "000032019322000109")
    records = [
        {
            "ticker": "AAPL",
            "cik": "320193",
            "primaryDocument": "aapl-20220924.htm",
            "linkToHtml": FILING_URL,
        },
        {
            "ticker": "AAPL",
            "cik": "320193",
            "accessionNo": "0000320193-22-000109",
            "primaryDocument": "other.htm",
            "linkToHtml": second_url,
        },
    ]
    row["response"]["output"][1] = _result("search", {"results": json.dumps(records)})
    trace = parse_rollout_line(json.dumps(row)).trace
    assert trace is not None
    source_id = trace.evidence[0].source_id
    assert source_id is not None
    assert "accession=000032019322000108" in source_id
    assert "000032019322000109" not in source_id


def test_record_metadata_conflicting_with_its_url_is_unavailable():
    row = _trace()
    record = {
        "ticker": "AAPL",
        "cik": "320193",
        "accessionNo": "0000950170-24-087843",
        "primaryDocument": "aapl-20220924.htm",
        "linkToHtml": FILING_URL,
    }
    row["response"]["output"][1] = _result("search", {"results": json.dumps([record])})
    trace = parse_rollout_line(json.dumps(row)).trace
    assert trace is not None
    assert trace.evidence[0].source_id is None
    assert trace.evidence[0].attribution_state == "unavailable"
    result = evaluate_row(json.dumps(row), row, 0, _evaluator(), _config())
    assert result["verdict"]["status"] == "block"
    assert result["verdict"]["reason"] == "no_source"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            "SUCCESS: The result has been saved to the data storage under the key: aapl_10k.",
            True,
        ),
        ("SUCCESS: saved under key: aapl_10k.", False),
        (
            "SUCCESS: The result has been saved to the data storage under the key: other.",
            False,
        ),
        ("connection timed out", False),
    ],
)
def test_parse_success_marker_is_exact_and_key_bound(payload, expected):
    assert _is_parse_success(payload, "aapl_10k") is expected


def test_sidecar_rerun_is_transactional(tmp_path):
    source = tmp_path / "input.jsonl"
    target = tmp_path / "output.jsonl"
    source.write_text(json.dumps(_trace()) + "\n")
    Path(f"{source}.done").touch()
    target.write_text("prior output")
    Path(f"{target}.done").write_text("stale")

    class _Broken:
        def evaluate(self, answer, evidence):
            raise RuntimeError("crashed")

    with pytest.raises(RuntimeError, match="crashed"):
        evaluate_seed(str(source), str(target), 0, _Broken(), _config())
    assert target.read_text() == "prior output"
    assert not Path(f"{target}.done").exists()


def test_missing_input_marker_preserves_prior_output(tmp_path):
    source = tmp_path / "input.jsonl"
    target = tmp_path / "output.jsonl"
    source.write_text(json.dumps(_trace()) + "\n")
    target.write_text("prior output")
    Path(f"{target}.done").write_text("prior marker")
    with pytest.raises(RuntimeError, match="completion marker"):
        evaluate_seed(str(source), str(target), 0, _evaluator(), _config())
    assert target.read_text() == "prior output"
    assert Path(f"{target}.done").read_text() == "prior marker"


def test_duplicate_rows_still_receive_unique_evaluation_ids(tmp_path):
    source = tmp_path / "input.jsonl"
    target = tmp_path / "output.jsonl"
    line = json.dumps(_trace())
    source.write_text(f"{line}\n{line}\n")
    Path(f"{source}.done").touch()
    evaluate_seed(str(source), str(target), 0, _evaluator(), _config())
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    assert len({row["evaluation_uuid"] for row in rows}) == 2


@pytest.mark.parametrize(
    ("label", "expected"),
    [("Entailment", "entailment"), ("NEUTRAL", "neutral"), ("contradiction", "contradiction")],
)
def test_nli_labels_are_validated(label, expected):
    assert _normalize_nli_label(label) == expected


def test_unknown_nli_label_is_rejected():
    with pytest.raises(ValueError, match="Unknown NLI label"):
        _normalize_nli_label("LABEL_0")


def test_workflow_is_opt_in_and_pins_models():
    path = Path("nvflow/recipes/finance/workflows/grpo/base.yaml")
    data = yaml.safe_load(path.read_text())
    stage = data["stages"]["evaluate_grounding"]
    assert "evaluate_grounding" not in data["pipeline_stages"]
    assert stage["dependencies"] == ["collect_rollouts"]
    assert stage["num_gpus"] == 0
    assert stage["routing_model"] == DEFAULT_EMBEDDING_MODEL
    assert stage["nli_model"] == DEFAULT_NLI_MODEL
    assert "# - evaluate_grounding" in path.read_text()


def test_stage_builds_one_cpu_job_per_environment_and_seed(monkeypatch):
    import nemo_skills.pipeline.cli as pipeline_cli

    import nvflow.lib.rl.helpers as helpers

    submitted = []
    monkeypatch.setattr(pipeline_cli, "wrap_arguments", lambda command: command)
    monkeypatch.setattr(pipeline_cli, "run_cmd", lambda **kwargs: submitted.append(kwargs))
    monkeypatch.setattr(helpers, "resolve_environments", lambda config: {"env_a": {}, "env_b": {}})
    EvaluateGroundingStage().execute(
        {
            "rollouts_dir": "/rollouts",
            "output_dir": "/sidecars",
            "starting_seed": 0,
            "num_random_seeds": 2,
            "seeds": [0, 3],
            "num_gpus": 0,
        },
        cluster="test-cluster",
        expname="pg",
        run_after=["collect-rollouts"],
    )
    assert len(submitted) == 4
    assert all(job["num_gpus"] == 0 for job in submitted)
    assert all(
        "output-rs" in job["ctx"] and "grounding-verifier-rs" in job["ctx"] for job in submitted
    )


def test_stage_command_contains_the_native_paths():
    command = build_evaluate_command(
        "/rollouts/output-rs3.jsonl",
        "/sidecars/grounding-verifier-rs3.jsonl",
        3,
        "finance_sec_search",
    )
    assert "nvflow.recipes.finance.utils.rl.grounding_verifier" in command
    assert "--routing_model" in command and "--nli_model" in command


@pytest.mark.parametrize(
    ("rollouts", "output"),
    [("/tmp/a", "/tmp/a"), ("/tmp/a", "/tmp/a/out"), ("/tmp/a/out", "/tmp/a")],
)
def test_stage_rejects_overlapping_paths(rollouts, output):
    with pytest.raises(ValueError):
        EvaluateGroundingStage().validate_config({"rollouts_dir": rollouts, "output_dir": output})
