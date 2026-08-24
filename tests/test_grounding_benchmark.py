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
import os
from collections import Counter
from collections.abc import Sequence

import pytest

from nvflow.grounding_verifier.evaluator import GroundingVerifierConfig
from nvflow.grounding_verifier.types import Decision
from nvflow.recipes.finance.utils.rl import grounding_benchmark as benchmark
from nvflow.recipes.finance.utils.rl.grounding_benchmark import (
    ATTACKS,
    FACTS,
    FILINGS,
    NLI_MODEL_REVISION,
    ROUTING_MODEL_REVISION,
    generate_cases,
    run_benchmark,
)
from nvflow.recipes.finance.utils.rl.grounding_verifier import (
    FinanceEvaluatorConfig,
    build_evaluator_from_config,
    evaluate_seed,
)


def _config() -> FinanceEvaluatorConfig:
    return FinanceEvaluatorConfig(
        grounding_config=GroundingVerifierConfig(),
        routing_model_revision="routing-test-revision",
        nli_model_revision="nli-test-revision",
    )


class _ExpectedEvaluator:
    def __init__(self, grounded_answers: set[str]) -> None:
        self.grounded_answers = grounded_answers

    def evaluate(self, answer: str, evidence: Sequence) -> Decision:
        status = "allow" if answer in self.grounded_answers else "block"
        return Decision(status=status, reason="test")


class _AlwaysAllowEvaluator:
    def evaluate(self, answer: str, evidence: Sequence) -> Decision:
        return Decision(status="allow", reason="test")


class TestFacts:
    def test_entities_and_values(self):
        assert {filing.ticker: (filing.net_income, filing.total_assets) for filing in FILINGS} == {
            "AMZN": (59_248, 624_894),
            "GOOGL": (100_118, 450_256),
            "META": (62_360, 276_054),
            "TSLA": (7_153, 122_070),
        }

    def test_entities_are_held_out(self):
        assert {filing.ticker for filing in FILINGS}.isdisjoint({"AAPL", "MSFT", "NVDA"})

    def test_two_metrics_per_filing(self):
        assert len(FACTS) == 8
        assert Counter(fact.filing.ticker for fact in FACTS) == Counter(
            {"AMZN": 2, "GOOGL": 2, "META": 2, "TSLA": 2}
        )

    def test_sec_urls_derive_from_accessions(self):
        for filing in FILINGS:
            assert filing.url.startswith("https://www.sec.gov/Archives/edgar/data/")
            assert filing.accession.replace("-", "") in filing.url


class TestCases:
    @pytest.fixture
    def cases(self):
        return generate_cases()

    def test_count_and_balance(self, cases):
        assert len(cases) == 56
        assert Counter(case.category for case in cases) == Counter(
            {"grounded": 8, **dict.fromkeys(ATTACKS, 8)}
        )

    def test_expected_statuses(self, cases):
        for case in cases:
            expected = "allow" if case.category == "grounded" else "block"
            assert case.expected_status == expected

    def test_ids_are_unique_and_deterministic(self, cases):
        repeated = generate_cases()
        assert [case.case_id for case in cases] == [case.case_id for case in repeated]
        assert len({case.case_id for case in cases}) == 56

    def test_native_call_order(self, cases):
        expected = [
            "sec_filing_search",
            "parse_html_page",
            "retrieve_information",
            "submit_final_result",
        ]
        for case in cases:
            output = case.row["response"]["output"]
            calls = [item["name"] for item in output if item["type"] == "function_call"]
            assert calls == expected

    def test_function_results_pair_with_calls(self, cases):
        for case in cases:
            output = case.row["response"]["output"]
            for call_index, result_index in ((0, 1), (2, 3), (4, 5)):
                assert output[call_index]["call_id"] == output[result_index]["call_id"]

    def test_only_submit_changes_for_each_fact(self, cases):
        for entity in {case.entity for case in cases}:
            for metric in {case.metric for case in cases if case.entity == entity}:
                group = [case for case in cases if (case.entity, case.metric) == (entity, metric)]
                evidence_traces = [case.row["response"]["output"][:6] for case in group]
                assert all(trace == evidence_traces[0] for trace in evidence_traces)
                assert len({case.row["response"]["output"][6]["arguments"] for case in group}) == 7

    @pytest.mark.parametrize("category", ATTACKS)
    def test_attacks_change_the_answer(self, cases, category):
        for entity in {case.entity for case in cases}:
            for metric in {case.metric for case in cases if case.entity == entity}:
                group = [case for case in cases if (case.entity, case.metric) == (entity, metric)]
                grounded = next(case.answer for case in group if case.category == "grounded")
                attacked = next(case.answer for case in group if case.category == category)
                assert attacked != grounded


class TestEvaluation:
    @pytest.fixture
    def evaluated(self):
        cases = generate_cases()
        grounded = {case.answer for case in cases if case.category == "grounded"}
        return run_benchmark(cases, _ExpectedEvaluator(grounded), _config())

    def test_perfect_fake_passes_all_gates(self, evaluated):
        results, summary = evaluated
        assert len(results) == 56
        assert summary["grounded_acceptance"] == 1.0
        assert summary["attack_rejection"] == 1.0
        assert summary["unavailable_rate"] == 0.0
        assert summary["gates_passed"] is True

    def test_always_allow_fails_attack_gate(self):
        _, summary = run_benchmark(generate_cases(), _AlwaysAllowEvaluator(), _config())
        assert summary["attack_rejection"] == 0.0
        assert summary["gates_passed"] is False

    def test_custom_gates_are_applied(self):
        gates = {"grounded_acceptance": 1.0, "attack_rejection": 0.0, "unavailable_rate": 0.0}
        _, summary = run_benchmark(
            generate_cases(), _AlwaysAllowEvaluator(), _config(), gates=gates
        )
        assert summary["gates_passed"] is True
        assert summary["gates"]["attack_rejection"]["threshold"] == 0.0

    def test_outputs_are_reproducible_and_path_free(self, evaluated, tmp_path):
        results, summary = evaluated
        cases = generate_cases()
        benchmark._write_outputs(tmp_path, cases, results, summary)
        assert {path.name for path in tmp_path.iterdir()} == {
            "cases.jsonl",
            "results.jsonl",
            "summary.json",
        }
        for path in tmp_path.iterdir():
            text = path.read_text()
            assert "/Users/" not in text
            assert "/tmp/" not in text
        assert json.loads((tmp_path / "summary.json").read_text())["total_cases"] == 56

    def test_model_revisions_are_recorded(self, evaluated):
        _, summary = evaluated
        assert summary["models"]["routing_model_revision"] == "routing-test-revision"
        assert summary["models"]["nli_model_revision"] == "nli-test-revision"

    def test_optional_performance_metrics(self):
        cases = generate_cases()
        grounded = {case.answer for case in cases if case.category == "grounded"}
        _, summary = run_benchmark(
            cases,
            _ExpectedEvaluator(grounded),
            _config(),
            measure_performance=True,
        )
        performance = summary["performance"]
        assert performance["warm_rows"] == len(cases) - 1
        assert performance["cold_first_row_seconds"] >= 0
        assert performance["warm_rows_per_second"] > 0
        assert performance["process_peak_rss_mib"] > 0


@pytest.mark.skipif(
    os.environ.get("GROUNDING_VERIFIER_RUN_MODELS") != "1",
    reason="Set GROUNDING_VERIFIER_RUN_MODELS=1 to run the pinned public-model efficacy gate.",
)
def test_pinned_public_models_pass_efficacy_gates(tmp_path):
    config = FinanceEvaluatorConfig(
        grounding_config=GroundingVerifierConfig(),
        routing_model_revision=ROUTING_MODEL_REVISION,
        nli_model_revision=NLI_MODEL_REVISION,
    )
    cases = generate_cases()
    evaluator = build_evaluator_from_config(config)
    _, summary = run_benchmark(cases, evaluator, config)
    assert summary["grounded_acceptance"] == 1.0
    assert summary["attack_rejection"] == 1.0
    assert summary["unavailable_rate"] == 0.0
    assert summary["gates_passed"] is True

    input_path = tmp_path / "output-rs0.jsonl"
    output_path = tmp_path / "grounding-verifier-rs0.jsonl"
    input_path.write_text(json.dumps(cases[0].row) + "\n", encoding="utf-8")
    (tmp_path / "output-rs0.jsonl.done").touch()
    assert evaluate_seed(str(input_path), str(output_path), 0, evaluator, config) == 1
    assert json.loads(output_path.read_text())["verdict"]["status"] == "allow"
    assert (tmp_path / "grounding-verifier-rs0.jsonl.done").is_file()
