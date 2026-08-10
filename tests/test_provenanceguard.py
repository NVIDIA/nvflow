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
"""Offline tests for ProvenanceGuard Open v1.

All tests use deterministic fakes — no model downloads, no network.
Covers: evaluator decision semantics, NLI label validation, finance trace
extraction, atomic sidecar output, YAML parse, registry discovery, opt-in
ordering, exact paths, marker gating, environment/model/revision/policy
propagation, every-line output, malformed/non-object, unchanged input,
repeat-byte determinism, marker atomicity, mixed error+block unavailable,
known+unknown key, zero-pad URL CIK, real nested response.output trace.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

from nvflow.provenanceguard.decomposer import RuleBasedDecomposer
from nvflow.provenanceguard.embedder import DEFAULT_EMBEDDING_MODEL
from nvflow.provenanceguard.evaluator import (
    ALGORITHM_VERSION,
    ProvenanceGuardConfig,
    ProvenanceGuardEvaluator,
)
from nvflow.provenanceguard.nli import DEFAULT_NLI_MODEL, _normalize_nli_label
from nvflow.provenanceguard.protected_values import (
    check_protected_values,
    extract_protected_values,
)
from nvflow.provenanceguard.router import EmbeddingSourceRouter
from nvflow.provenanceguard.types import EvidenceChunk, NLIResult
from nvflow.recipes.finance.utils.rl.provenanceguard import (
    EVIDENCE_BASIS,
    LIMITATION,
    SCHEMA_VERSION,
    FinanceEvaluatorConfig,
    done_marker,
    evaluate_row,
    evaluate_seed,
    merged_filename,
    parse_rollout_line,
    sidecar_filename,
)

# Cache for the directly-loaded evaluate_provenance stage module.
# Loading via spec_from_file_location avoids the package __init__
# auto-importing every finance stage under the test interpreter.
_eval_stage_mod = None


def _load_eval_stage_module():
    """Load evaluate_provenance.py directly, bypassing __init__ auto-import.

    Cached so the @StageRegistry.register decorator runs only once,
    avoiding duplicate registration errors.
    """
    global _eval_stage_mod
    if _eval_stage_mod is not None:
        return _eval_stage_mod
    import sys

    existing = sys.modules.get("nvflow.recipes.finance.stages.rl.evaluate_provenance")
    if existing is not None:
        _eval_stage_mod = existing
        return _eval_stage_mod
    spec = importlib.util.spec_from_file_location(
        "_pg_eval_stage_private",
        Path("nvflow/recipes/finance/stages/rl/evaluate_provenance.py"),
    )
    _eval_stage_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_eval_stage_mod)
    return _eval_stage_mod


# ---------------------------------------------------------------------------
# Deterministic fakes
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Deterministic embedder using simple character-level hashing."""

    @property
    def dimension(self) -> int:
        return 16

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * 16
            for i, ch in enumerate(text[:16]):
                vec[i] = ord(ch) / 128.0
            vectors.append(vec)
        return vectors


class FakeNLIScorer:
    """NLI scorer that returns a configurable label, with optional per-claim error."""

    def __init__(self, label: str = "entailment") -> None:
        self._label = label
        self._should_error = False
        self._error_on_claim: str | None = None

    def set_error(self, should_error: bool = True) -> None:
        self._should_error = should_error

    def set_error_on_claim(self, claim_text: str) -> None:
        self._error_on_claim = claim_text

    def score(self, *, premise: str, hypothesis: str) -> NLIResult:
        if self._should_error:
            raise RuntimeError("Fake NLI model error")
        if self._error_on_claim and self._error_on_claim in hypothesis:
            raise RuntimeError(f"Fake NLI error for claim: {hypothesis}")
        return NLIResult(
            label=self._label,
            score=0.99,
            probabilities=(
                ("entailment", 0.99 if self._label == "entailment" else 0.01),
                ("neutral", 0.99 if self._label == "neutral" else 0.005),
                ("contradiction", 0.99 if self._label == "contradiction" else 0.005),
            ),
        )


def _make_evaluator(
    nli_label: str = "entailment",
    config: ProvenanceGuardConfig | None = None,
) -> tuple[ProvenanceGuardEvaluator, FakeNLIScorer]:
    nli = FakeNLIScorer(label=nli_label)
    evaluator = ProvenanceGuardEvaluator(
        decomposer=RuleBasedDecomposer(),
        router=EmbeddingSourceRouter(FakeEmbedder()),
        nli_scorer=nli,
        config=config or ProvenanceGuardConfig(),
    )
    return evaluator, nli


def _make_evidence(
    text: str = "Revenue was $1.23 billion in 2024.",
    source_id: str | None = "sec:cik=0001811414:accession=0001811414-25-000010:doc=10-K",
    attribution_state: str = "available",
    source_ids: tuple[str, ...] = (),
) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id="ev:test",
        text=text,
        source_id=source_id,
        source_ids=source_ids,
        attribution_state=attribution_state,
    )


def _make_finance_evaluator(
    nli_label: str = "entailment",
    config: ProvenanceGuardConfig | None = None,
) -> tuple[ProvenanceGuardEvaluator, FakeNLIScorer, FinanceEvaluatorConfig]:
    evaluator, nli = _make_evaluator(nli_label=nli_label, config=config)
    finance_config = FinanceEvaluatorConfig(
        provenance_config=evaluator.config,
        environment="finance_sec_search",
        routing_model_revision="abc123",
        nli_model_revision="def456",
    )
    return evaluator, nli, finance_config


def _make_rollout_row(
    answer: str = "Revenue was $1.23 billion in 2024.",
    evidence_text: str = "Revenue was $1.23 billion in 2024.",
) -> dict:
    return {
        "uuid": "test-uuid-001",
        "response_id": "resp-001",
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "output": [
            {
                "type": "function_call",
                "name": "retrieve_information",
                "call_id": "call_retrieve_1",
                "arguments": json.dumps({"prompt": "What was the revenue?"}),
            },
            {
                "type": "function_call_output",
                "call_id": "call_retrieve_1",
                "output": json.dumps({"result": evidence_text}),
            },
            {
                "type": "function_call",
                "name": "submit_final_result",
                "call_id": "call_submit_1",
                "arguments": json.dumps({"final_result": answer}),
            },
        ],
    }


# ---------------------------------------------------------------------------
# YAML parse tests
# ---------------------------------------------------------------------------


class TestYamlParse:
    def test_base_yaml_parses(self):
        path = Path("nvflow/recipes/finance/workflows/grpo/base.yaml")
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data is not None
        assert data["recipe"] == "finance"
        assert data["workflow"]["name"] == "grpo"

    def test_provenanceguard_eval_directory_present(self):
        path = Path("nvflow/recipes/finance/workflows/grpo/base.yaml")
        with open(path) as f:
            data = yaml.safe_load(f)
        dirs = data["directories"]
        assert "provenanceguard-eval" in dirs
        assert dirs["provenanceguard-eval"] == "${model_output_dir}/provenanceguard-eval"

    def test_evaluate_provenance_stage_config_present(self):
        path = Path("nvflow/recipes/finance/workflows/grpo/base.yaml")
        with open(path) as f:
            data = yaml.safe_load(f)
        stages = data["stages"]
        assert "evaluate_provenance" in stages
        ep = stages["evaluate_provenance"]
        assert ep["dependencies"] == ["collect_rollouts"]
        assert ep["container"] == "nemo-skills"
        assert ep["routing_model"] == DEFAULT_EMBEDDING_MODEL
        assert ep["nli_model"] == DEFAULT_NLI_MODEL

    def test_evaluate_provenance_commented_in_pipeline_stages(self):
        path = Path("nvflow/recipes/finance/workflows/grpo/base.yaml")
        text = path.read_text()
        assert "# - evaluate_provenance" in text
        raw_lines = text.splitlines()
        collect_idx = next(
            i for i, line in enumerate(raw_lines) if line.strip().startswith("- collect_rollouts")
        )
        eval_idx = next(i for i, line in enumerate(raw_lines) if "# - evaluate_provenance" in line)
        assert eval_idx == collect_idx + 1


# ---------------------------------------------------------------------------
# Registry discovery tests
# ---------------------------------------------------------------------------


class TestRegistryDiscovery:
    def test_stage_registered(self):
        from nvflow.core import StageRegistry

        stage_cls = StageRegistry.get(
            recipe="finance",
            workflow="grpo",
            stage="evaluate_provenance",
        )
        assert stage_cls is not None
        assert stage_cls.workflow == "grpo"

    def test_stage_not_in_default_pipeline(self):
        path = Path("nvflow/recipes/finance/workflows/grpo/base.yaml")
        with open(path) as f:
            data = yaml.safe_load(f)
        stages = data["pipeline_stages"]
        assert "evaluate_provenance" not in stages


# ---------------------------------------------------------------------------
# Opt-in ordering tests
# ---------------------------------------------------------------------------


class TestOptInOrdering:
    def test_evaluate_provenance_after_collect_rollouts(self):
        path = Path("nvflow/recipes/finance/workflows/grpo/base.yaml")
        text = path.read_text()
        lines = text.splitlines()
        collect_idx = next(
            i for i, line in enumerate(lines) if line.strip().startswith("- collect_rollouts")
        )
        eval_idx = next(i for i, line in enumerate(lines) if "# - evaluate_provenance" in line)
        assert eval_idx == collect_idx + 1

    def test_no_downstream_dependency_on_evaluate_provenance(self):
        path = Path("nvflow/recipes/finance/workflows/grpo/base.yaml")
        with open(path) as f:
            data = yaml.safe_load(f)
        for stage_name, stage_cfg in data["stages"].items():
            deps = stage_cfg.get("dependencies", [])
            if deps:
                assert "evaluate_provenance" not in deps, (
                    f"Stage '{stage_name}' should not depend on evaluate_provenance"
                )


# ---------------------------------------------------------------------------
# Exact paths tests
# ---------------------------------------------------------------------------


class TestExactPaths:
    def test_input_path_pattern(self):
        path = Path("nvflow/recipes/finance/workflows/grpo/base.yaml")
        text = path.read_text()
        assert "step-5-collect-rollouts}/{env}/rollout/output-rs" in text
        assert "rollout/output-rs<seed>.jsonl.done" in text

    def test_output_path_pattern(self):
        path = Path("nvflow/recipes/finance/workflows/grpo/base.yaml")
        text = path.read_text()
        assert "provenanceguard-eval}/{env}/provenanceguard-rs" in text
        assert "provenanceguard-rs<seed>.jsonl.done" in text

    def test_sidecar_filename(self):
        assert sidecar_filename(0) == "provenanceguard-rs0.jsonl"
        assert sidecar_filename(3) == "provenanceguard-rs3.jsonl"

    def test_merged_filename(self):
        assert merged_filename(0) == "output-rs0.jsonl"
        assert merged_filename(7) == "output-rs7.jsonl"

    def test_done_marker(self):
        assert done_marker(0) == "output-rs0.jsonl.done"
        assert done_marker(3) == "output-rs3.jsonl.done"


# ---------------------------------------------------------------------------
# Marker gating tests
# ---------------------------------------------------------------------------


class TestMarkerGating:
    def test_marker_created_after_successful_write(self, tmp_path):
        evaluator, _, config = _make_finance_evaluator()
        input_file = tmp_path / "rollout" / "output-rs0.jsonl"
        input_file.parent.mkdir(parents=True)
        input_file.write_text(json.dumps(_make_rollout_row()) + "\n")
        Path(str(input_file) + ".done").touch()
        output_file = str(tmp_path / "out" / "provenanceguard-rs0.jsonl")
        count = evaluate_seed(str(input_file), output_file, 0, evaluator, config)
        assert count == 1
        assert os.path.exists(output_file)
        assert os.path.exists(output_file + ".done")

    def test_stale_marker_replaced_with_fresh_empty(self, tmp_path):
        evaluator, _, config = _make_finance_evaluator()
        input_file = tmp_path / "rollout" / "output-rs0.jsonl"
        input_file.parent.mkdir(parents=True)
        input_file.write_text(json.dumps(_make_rollout_row()) + "\n")
        Path(str(input_file) + ".done").touch()
        output_file = str(tmp_path / "out" / "provenanceguard-rs0.jsonl")
        stale_marker = output_file + ".done"
        Path(stale_marker).parent.mkdir(parents=True, exist_ok=True)
        Path(stale_marker).write_text("stale")
        assert os.path.exists(stale_marker)
        evaluate_seed(str(input_file), output_file, 0, evaluator, config)
        assert os.path.exists(stale_marker)
        assert Path(stale_marker).read_text() == ""
        assert os.path.exists(output_file)

    def test_failing_rerun_preserves_output_removes_marker(self, tmp_path):
        """If evaluation fails after valid input gate, old output stays
        but stale .done marker is absent."""
        evaluator, _, config = _make_finance_evaluator()
        input_file = tmp_path / "input.jsonl"
        input_file.write_text(json.dumps(_make_rollout_row()) + "\n")
        Path(str(input_file) + ".done").touch()
        output_file = str(tmp_path / "output.jsonl")
        output_file_path = Path(output_file)
        output_file_path.write_text("prior output content")
        done_path = output_file + ".done"
        Path(done_path).write_text("stale marker")

        # Force evaluation failure by using a broken evaluator.
        class _BrokenEvaluator:
            def evaluate(self, answer, evidence):
                raise RuntimeError("evaluation crashed")

        with pytest.raises(RuntimeError, match="evaluation crashed"):
            evaluate_seed(str(input_file), output_file, 0, _BrokenEvaluator(), config)

        assert output_file_path.read_text() == "prior output content"
        assert not os.path.exists(done_path)

    def test_missing_input_marker_preserves_output_and_marker(self, tmp_path):
        """Missing input .done preserves old output and marker bytes."""
        evaluator, _, config = _make_finance_evaluator()
        input_file = tmp_path / "input.jsonl"
        input_file.write_text(json.dumps(_make_rollout_row()) + "\n")
        # NOTE: no .done marker for input
        output_file = tmp_path / "output.jsonl"
        output_file.write_text("prior output")
        done_path = str(output_file) + ".done"
        Path(done_path).write_text("prior done bytes")
        with pytest.raises(RuntimeError, match="Input completion marker"):
            evaluate_seed(str(input_file), str(output_file), 0, evaluator, config)
        assert output_file.read_text() == "prior output"
        assert Path(done_path).read_text() == "prior done bytes"

    def test_successful_rerun_replaces_stale_marker_fresh_empty(self, tmp_path):
        """Successful valid rerun replaces stale marker with fresh empty marker."""
        evaluator, _, config = _make_finance_evaluator()
        input_file = tmp_path / "input.jsonl"
        input_file.write_text(json.dumps(_make_rollout_row()) + "\n")
        Path(str(input_file) + ".done").touch()
        output_file = str(tmp_path / "output.jsonl")
        done_path = output_file + ".done"
        Path(done_path).write_text("stale content")
        evaluate_seed(str(input_file), output_file, 0, evaluator, config)
        assert os.path.exists(output_file)
        assert os.path.exists(done_path)
        assert Path(done_path).read_text() == ""


# ---------------------------------------------------------------------------
# Environment/model/revision/policy propagation tests
# ---------------------------------------------------------------------------


class TestPropagation:
    def test_environment_propagated(self):
        row = _make_rollout_row()
        raw_line = json.dumps(row)
        evaluator, _, config = _make_finance_evaluator()
        config = FinanceEvaluatorConfig(
            provenance_config=evaluator.config,
            environment="my_custom_env",
        )
        result = evaluate_row(raw_line, row, 0, evaluator, config)
        assert result["environment"] == "my_custom_env"

    def test_model_ids_propagated(self):
        row = _make_rollout_row()
        raw_line = json.dumps(row)
        evaluator, _, config = _make_finance_evaluator()
        result = evaluate_row(raw_line, row, 0, evaluator, config)
        assert result["models"]["routing_model"] == DEFAULT_EMBEDDING_MODEL
        assert result["models"]["nli_model"] == DEFAULT_NLI_MODEL

    def test_revisions_propagated(self):
        row = _make_rollout_row()
        raw_line = json.dumps(row)
        evaluator, _, config = _make_finance_evaluator()
        result = evaluate_row(raw_line, row, 0, evaluator, config)
        assert result["models"]["routing_model_revision"] == "abc123"
        assert result["models"]["nli_model_revision"] == "def456"

    def test_nullable_revisions(self):
        row = _make_rollout_row()
        raw_line = json.dumps(row)
        evaluator, _, _ = _make_finance_evaluator()
        config = FinanceEvaluatorConfig(
            provenance_config=evaluator.config,
            routing_model_revision=None,
            nli_model_revision=None,
        )
        result = evaluate_row(raw_line, row, 0, evaluator, config)
        assert result["models"]["routing_model_revision"] is None
        assert result["models"]["nli_model_revision"] is None

    def test_policy_config_propagated(self):
        row = _make_rollout_row()
        raw_line = json.dumps(row)
        pg_config = ProvenanceGuardConfig(
            evidence_excerpt_length=200,
        )
        evaluator, _, config = _make_finance_evaluator(config=pg_config)
        result = evaluate_row(raw_line, row, 0, evaluator, config)
        assert result["thresholds"]["policy"] == "fixed_fail_closed"
        assert "block_on_contradiction" not in result["thresholds"]
        assert "block_on_neutral" not in result["thresholds"]
        assert "block_on_no_source" not in result["thresholds"]
        assert "block_on_protected_value_mismatch" not in result["thresholds"]
        assert result["thresholds"]["evidence_excerpt_length"] == 200


# ---------------------------------------------------------------------------
# Every line / malformed / unchanged input tests
# ---------------------------------------------------------------------------


class TestEveryLine:
    def test_every_line_produces_sidecar_row(self, tmp_path):
        evaluator, _, config = _make_finance_evaluator()
        input_file = tmp_path / "input.jsonl"
        input_file.write_text(
            json.dumps(_make_rollout_row()) + "\n" + "\n" + "not valid json\n" + "[1, 2, 3]\n"
        )
        Path(str(input_file) + ".done").touch()
        output_file = str(tmp_path / "output.jsonl")
        count = evaluate_seed(str(input_file), output_file, 0, evaluator, config)
        assert count == 4

    def test_blank_line_produces_sidecar_row(self):
        sidecar = parse_rollout_line("")
        assert sidecar.parse_error is not None
        assert sidecar.parse_error == "empty line"
        assert sidecar.raw_line_fingerprint

    def test_malformed_json_produces_sidecar_row(self):
        raw = '{"broken": json'
        sidecar = parse_rollout_line(raw)
        assert sidecar.parse_error is not None
        assert "JSONDecodeError" in sidecar.parse_error
        assert sidecar.raw_line_fingerprint

    def test_non_object_json_produces_sidecar_row(self):
        raw = "[1, 2, 3]"
        sidecar = parse_rollout_line(raw)
        assert sidecar.parse_error is not None
        assert "expected JSON object" in sidecar.parse_error


class TestUnchangedInput:
    def test_input_file_not_modified(self, tmp_path):
        evaluator, _, config = _make_finance_evaluator()
        row = _make_rollout_row()
        raw_line = json.dumps(row) + "\n"
        input_file = tmp_path / "input.jsonl"
        input_file.write_text(raw_line)
        original = input_file.read_bytes()
        Path(str(input_file) + ".done").touch()
        output_file = str(tmp_path / "output.jsonl")
        evaluate_seed(str(input_file), output_file, 0, evaluator, config)
        assert input_file.read_bytes() == original


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_repeat_byte_determinism(self, tmp_path):
        evaluator, _, config = _make_finance_evaluator()
        row = _make_rollout_row()
        raw_line = json.dumps(row) + "\n"
        input_file = tmp_path / "input.jsonl"
        input_file.write_text(raw_line * 3)
        Path(str(input_file) + ".done").touch()
        out1 = str(tmp_path / "out1.jsonl")
        out2 = str(tmp_path / "out2.jsonl")
        evaluate_seed(str(input_file), out1, 0, evaluator, config)
        evaluate_seed(str(input_file), out2, 0, evaluator, config)
        assert Path(out1).read_bytes() == Path(out2).read_bytes()

    def test_no_random_uuid(self):
        row = _make_rollout_row()
        raw_line = json.dumps(row)
        evaluator, _, config = _make_finance_evaluator()
        result1 = evaluate_row(raw_line, row, 0, evaluator, config, line_number=0)
        result2 = evaluate_row(raw_line, row, 0, evaluator, config, line_number=0)
        assert result1["evaluation_uuid"] == result2["evaluation_uuid"]
        config_digest = hashlib.sha256(
            json.dumps(config.to_dict(), sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        expected = hashlib.sha256(
            f"eval:{result1['raw_line_fingerprint']}:0:0:{ALGORITHM_VERSION}:{config_digest}".encode()
        ).hexdigest()
        assert result1["evaluation_uuid"] == expected


# ---------------------------------------------------------------------------
# Marker atomicity tests
# ---------------------------------------------------------------------------


class TestMarkerAtomicity:
    def test_no_marker_on_error(self, tmp_path):
        evaluator, _, config = _make_finance_evaluator()
        input_file = tmp_path / "nonexistent.jsonl"
        output_file = str(tmp_path / "out.jsonl")
        with pytest.raises(FileNotFoundError):
            evaluate_seed(str(input_file), output_file, 0, evaluator, config)
        assert not os.path.exists(output_file)
        assert not os.path.exists(output_file + ".done")

    def test_temp_file_cleaned_on_error(self, tmp_path):
        evaluator, _, config = _make_finance_evaluator()
        output_file = str(tmp_path / "out.jsonl")
        done_path = output_file + ".done"
        Path(done_path).write_text("stale")
        with pytest.raises(FileNotFoundError):
            evaluate_seed(str(tmp_path / "nonexistent.jsonl"), output_file, 0, evaluator, config)
        temp_files = list(tmp_path.glob(".pg_tmp_*"))
        assert len(temp_files) == 0


# ---------------------------------------------------------------------------
# Mixed contradiction-or-neutral + error => unavailable
# ---------------------------------------------------------------------------


class TestMixedVerdicts:
    def test_contradiction_plus_error_is_unavailable(self):
        evidence = [
            _make_evidence(text="Revenue was $1.23 billion in 2024."),
            _make_evidence(text="The sky is blue."),
        ]
        evaluator, nli = _make_evaluator(nli_label="entailment")
        answer = "Revenue was $1.23 billion in 2024. The sky is blue today."
        nli.set_error_on_claim("The sky is blue today")
        decision = evaluator.evaluate(answer, evidence)
        assert decision.status == "unavailable"
        assert decision.reason == "partial_nli_errors"

    def test_neutral_plus_error_is_unavailable(self):
        evidence = [_make_evidence(text="Some unrelated text.")]
        evaluator, nli = _make_evaluator(nli_label="neutral")
        answer = "Revenue was $1.23 billion in 2024."
        nli.set_error_on_claim("Revenue")
        decision = evaluator.evaluate(answer, evidence)
        assert decision.status == "unavailable"


# ---------------------------------------------------------------------------
# Known + unknown key tests
# ---------------------------------------------------------------------------


class TestKnownUnknownKey:
    def test_multiple_keys_yields_unavailable(self):
        row = _make_rollout_row()
        row["output"] = [
            {
                "type": "function_call",
                "name": "parse_html_page",
                "call_id": "call_parse_1",
                "arguments": json.dumps(
                    {
                        "url": "https://www.sec.gov/Archives/edgar/data/1811414/000181141425000010/10-K.htm",
                        "key": "filing_10k",
                    }
                ),
            },
            {
                "type": "function_call",
                "name": "parse_html_page",
                "call_id": "call_parse_2",
                "arguments": json.dumps(
                    {
                        "url": "https://www.sec.gov/Archives/edgar/data/1811414/000181141425000020/10-K.htm",
                        "key": "filing_10k_2",
                    }
                ),
            },
            {
                "type": "function_call",
                "name": "retrieve_information",
                "call_id": "call_retrieve_1",
                "arguments": json.dumps({"prompt": "Search {{filing_10k}} and {{filing_10k_2}}"}),
            },
            {
                "type": "function_call_output",
                "call_id": "call_retrieve_1",
                "output": json.dumps({"result": "Some text"}),
            },
            {
                "type": "function_call",
                "name": "submit_final_result",
                "call_id": "call_submit_1",
                "arguments": json.dumps({"final_result": "Some answer."}),
            },
        ]
        sidecar = parse_rollout_line(json.dumps(row))
        assert sidecar.trace is not None
        assert len(sidecar.trace.evidence) == 1
        chunk = sidecar.trace.evidence[0]
        assert chunk.source_id is None
        assert chunk.attribution_state == "unavailable"
        assert len(chunk.source_ids) == 2

    def test_unknown_key_yields_unavailable(self):
        row = _make_rollout_row()
        row["output"] = [
            {
                "type": "function_call",
                "name": "retrieve_information",
                "call_id": "call_retrieve_1",
                "arguments": json.dumps({"prompt": "Search {{unknown_key}}"}),
            },
            {
                "type": "function_call_output",
                "call_id": "call_retrieve_1",
                "output": json.dumps({"result": "Some text"}),
            },
            {
                "type": "function_call",
                "name": "submit_final_result",
                "call_id": "call_submit_1",
                "arguments": json.dumps({"final_result": "Some answer."}),
            },
        ]
        sidecar = parse_rollout_line(json.dumps(row))
        assert sidecar.trace is not None
        assert len(sidecar.trace.evidence) == 1
        chunk = sidecar.trace.evidence[0]
        assert chunk.source_id is None
        assert chunk.attribution_state == "unavailable"


# ---------------------------------------------------------------------------
# Zero-pad URL CIK tests
# ---------------------------------------------------------------------------


class TestZeroPadCik:
    def test_url_cik_zero_padded(self):
        from nvflow.recipes.finance.utils.rl.provenanceguard import (
            _extract_filing_from_url,
        )

        url = "https://www.sec.gov/Archives/edgar/data/1811414/000181141425000010/10-K.htm"
        filing = _extract_filing_from_url(url)
        assert filing.cik == "0001811414"
        assert len(filing.cik) == 10
        sid = filing.source_id()
        assert sid is not None
        assert "cik=0001811414" in sid

    def test_already_padded_cik_preserved(self):
        from nvflow.recipes.finance.utils.rl.provenanceguard import (
            _extract_filing_from_url,
        )

        url = "https://www.sec.gov/Archives/edgar/data/0001811414/000181141425000010/10-K.htm"
        filing = _extract_filing_from_url(url)
        assert filing.cik == "0001811414"
        assert len(filing.cik) == 10


# ---------------------------------------------------------------------------
# Real nested response.output trace tests
# ---------------------------------------------------------------------------


class TestNestedResponseOutput:
    def test_nested_response_output_trace(self):
        row = {
            "uuid": "test-nested-001",
            "response": {
                "id": "resp-nested-001",
                "output": [
                    {
                        "type": "function_call",
                        "name": "retrieve_information",
                        "call_id": "call_retrieve_nested",
                        "arguments": json.dumps({"prompt": "What was the revenue?"}),
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_retrieve_nested",
                        "output": json.dumps({"result": "Revenue was $1.23 billion in 2024."}),
                    },
                    {
                        "type": "function_call",
                        "name": "submit_final_result",
                        "call_id": "call_submit_nested",
                        "arguments": json.dumps(
                            {"final_result": "Revenue was $1.23 billion in 2024."}
                        ),
                    },
                ],
            },
        }
        sidecar = parse_rollout_line(json.dumps(row))
        assert sidecar.trace is not None
        assert sidecar.trace.has_submit is True
        assert sidecar.trace.answer == "Revenue was $1.23 billion in 2024."
        assert len(sidecar.trace.evidence) == 1
        assert sidecar.trace.evidence[0].text == "Revenue was $1.23 billion in 2024."

    def test_flat_output_trace(self):
        row = _make_rollout_row()
        sidecar = parse_rollout_line(json.dumps(row))
        assert sidecar.trace is not None
        assert sidecar.trace.has_submit is True
        assert sidecar.trace.answer == "Revenue was $1.23 billion in 2024."
        assert len(sidecar.trace.evidence) == 1


# ---------------------------------------------------------------------------
# NLI label validation tests
# ---------------------------------------------------------------------------


class TestNLILabelValidation:
    def test_entailment_label_accepted(self):
        assert _normalize_nli_label("entailment") == "entailment"
        assert _normalize_nli_label("ENTAILMENT") == "entailment"

    def test_neutral_label_accepted(self):
        assert _normalize_nli_label("neutral") == "neutral"
        assert _normalize_nli_label("Neutral") == "neutral"

    def test_contradiction_label_accepted(self):
        assert _normalize_nli_label("contradiction") == "contradiction"

    def test_label_0_rejected(self):
        with pytest.raises(ValueError, match="Refusing to assume"):
            _normalize_nli_label("label_0")

    def test_label_1_rejected(self):
        with pytest.raises(ValueError, match="Refusing to assume"):
            _normalize_nli_label("label_1")

    def test_label_2_rejected(self):
        with pytest.raises(ValueError, match="Refusing to assume"):
            _normalize_nli_label("label_2")

    def test_arbitrary_label_rejected(self):
        with pytest.raises(ValueError, match="Unknown NLI label"):
            _normalize_nli_label("some_random_label")


# ---------------------------------------------------------------------------
# Evaluator decision semantics tests
# ---------------------------------------------------------------------------


class TestEvaluatorSemantics:
    def test_no_evidence_returns_unavailable(self):
        evaluator, _ = _make_evaluator()
        decision = evaluator.evaluate("Some answer.", [])
        assert decision.status == "unavailable"
        assert decision.reason == "no_evidence"

    def test_no_claims_returns_unavailable(self):
        evaluator, _ = _make_evaluator()
        decision = evaluator.evaluate("", [_make_evidence()])
        assert decision.status == "unavailable"
        assert decision.reason == "no_claims_extracted"

    def test_all_entailed_returns_allow(self):
        evaluator, _ = _make_evaluator(nli_label="entailment")
        evidence = [_make_evidence(text="Revenue was $1.23 billion in 2024.")]
        decision = evaluator.evaluate("Revenue was $1.23 billion in 2024.", evidence)
        assert decision.status == "allow"
        assert decision.reason == "all_entailed"

    def test_contradiction_returns_block(self):
        evaluator, _ = _make_evaluator(nli_label="contradiction")
        evidence = [_make_evidence(text="Revenue was $2.00 billion in 2024.")]
        decision = evaluator.evaluate("Revenue was $1.23 billion in 2024.", evidence)
        assert decision.status == "block"
        assert decision.reason == "contradiction"

    def test_neutral_returns_block(self):
        evaluator, _ = _make_evaluator(nli_label="neutral")
        evidence = [_make_evidence(text="Some unrelated text here.")]
        decision = evaluator.evaluate("Revenue was $1.23 billion in 2024.", evidence)
        assert decision.status == "block"
        assert decision.reason == "neutral"

    def test_protected_value_mismatch_returns_block(self):
        evaluator, _ = _make_evaluator(nli_label="entailment")
        evidence = [_make_evidence(text="Revenue was $2.00 billion in 2024.")]
        decision = evaluator.evaluate("Revenue was $1.23 billion in 2024.", evidence)
        assert decision.status == "block"
        assert decision.reason == "protected_value_mismatch"

    def test_nli_error_returns_unavailable(self):
        evaluator, nli = _make_evaluator()
        nli.set_error(True)
        evidence = [_make_evidence(text="Some text.")]
        decision = evaluator.evaluate("Revenue was $1.23 billion.", evidence)
        assert decision.status == "unavailable"


# ---------------------------------------------------------------------------
# Protected values: no year-only date match
# ---------------------------------------------------------------------------


class TestProtectedValuesNoYearOnly:
    def test_year_only_does_not_match(self):
        outcome, missing = check_protected_values(
            "The fiscal year ended December 31, 2024.",
            "In 2024 the company grew.",
        )
        assert outcome == "fail"
        assert len(missing) > 0

    def test_full_date_matches(self):
        outcome, _ = check_protected_values(
            "The fiscal year ended December 31, 2024.",
            "The fiscal year ended December 31, 2024.",
        )
        assert outcome == "pass"


# ---------------------------------------------------------------------------
# Sidecar output schema tests
# ---------------------------------------------------------------------------


class TestSidecarSchema:
    def test_output_contains_all_required_fields(self):
        row = _make_rollout_row()
        raw_line = json.dumps(row)
        evaluator, _, config = _make_finance_evaluator()
        result = evaluate_row(raw_line, row, 0, evaluator, config)
        required_top = [
            "schema_version",
            "algorithm",
            "evidence_basis",
            "limitation",
            "environment",
            "seed",
            "line_number",
            "uuid",
            "_ng_task_index",
            "_ng_rollout_index",
            "response_id",
            "fingerprint",
            "raw_line_fingerprint",
            "evaluation_uuid",
            "thresholds",
            "models",
            "answer",
            "has_submit",
            "submit_call_id",
            "evidence",
            "extraction_errors",
            "verdict",
        ]
        for key in required_top:
            assert key in result, f"Missing required field: {key}"

    def test_output_schema_and_algorithm(self):
        row = _make_rollout_row()
        raw_line = json.dumps(row)
        evaluator, _, config = _make_finance_evaluator()
        result = evaluate_row(raw_line, row, 0, evaluator, config)
        assert result["schema_version"] == SCHEMA_VERSION
        assert result["algorithm"] == ALGORITHM_VERSION
        assert result["evidence_basis"] == EVIDENCE_BASIS
        assert LIMITATION in result["limitation"]

    def test_parse_error_row_has_unavailable_verdict(self):
        raw = '{"broken": json'
        evaluator, _, config = _make_finance_evaluator()
        result = evaluate_row(raw, {}, 0, evaluator, config)
        assert result["verdict"]["status"] == "unavailable"
        assert result["parse_error"] is not None
        assert result["evidence"] == []

    def test_no_submit_row_has_unavailable_verdict(self):
        row = _make_rollout_row()
        row["output"] = [i for i in row["output"] if i.get("name") != "submit_final_result"]
        raw_line = json.dumps(row)
        evaluator, _, config = _make_finance_evaluator()
        result = evaluate_row(raw_line, row, 0, evaluator, config)
        assert result["verdict"]["status"] == "unavailable"
        assert result["verdict"]["reason"] == "no_submit_final_result"

    def test_one_chunk_per_retrieve_result(self):
        row = _make_rollout_row()
        row["output"] = [
            {
                "type": "function_call",
                "name": "retrieve_information",
                "call_id": "call_r1",
                "arguments": json.dumps({"prompt": "query1"}),
            },
            {
                "type": "function_call_output",
                "call_id": "call_r1",
                "output": json.dumps({"result": "Evidence one."}),
            },
            {
                "type": "function_call",
                "name": "retrieve_information",
                "call_id": "call_r2",
                "arguments": json.dumps({"prompt": "query2"}),
            },
            {
                "type": "function_call_output",
                "call_id": "call_r2",
                "output": json.dumps({"result": "Evidence two."}),
            },
            {
                "type": "function_call",
                "name": "submit_final_result",
                "call_id": "call_s1",
                "arguments": json.dumps({"final_result": "Some answer."}),
            },
        ]
        sidecar = parse_rollout_line(json.dumps(row))
        assert sidecar.trace is not None
        assert len(sidecar.trace.evidence) == 2
        assert sidecar.trace.evidence[0].text == "Evidence one."
        assert sidecar.trace.evidence[1].text == "Evidence two."

    def test_last_submit_final_result_used(self):
        row = _make_rollout_row()
        row["output"].append(
            {
                "type": "function_call",
                "name": "submit_final_result",
                "call_id": "call_submit_2",
                "arguments": json.dumps({"final_result": "The later answer."}),
            }
        )
        sidecar = parse_rollout_line(json.dumps(row))
        assert sidecar.trace is not None
        assert sidecar.trace.answer == "The later answer."
        assert sidecar.trace.submit_call_id == "call_submit_2"


# ---------------------------------------------------------------------------
# Realistic nested response.output trace tests
# ---------------------------------------------------------------------------

_FILING_URL = "https://www.sec.gov/Archives/edgar/data/1811414/000181141425000010/10-K.htm"


def _make_realistic_trace() -> dict:
    """A realistic nested response.output trace with all 4 finance tool types."""
    return {
        "uuid": "real-trace-001",
        "response_id": "resp-real-001",
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "output": [
            {
                "type": "function_call",
                "name": "sec_filing_search",
                "call_id": "call_sec_1",
                "arguments": json.dumps({"query": "Apple 10-K", "form_type": "10-K"}),
            },
            {
                "type": "function_call_output",
                "call_id": "call_sec_1",
                "output": json.dumps(
                    {
                        "filings": [
                            {
                                "cik": "1811414",
                                "accessionNo": "0001811414-25-000010",
                                "primaryDocument": "10-K.htm",
                                "linkToHtml": _FILING_URL,
                            }
                        ]
                    }
                ),
            },
            {
                "type": "function_call",
                "name": "parse_html_page",
                "call_id": "call_parse_1",
                "arguments": json.dumps({"url": _FILING_URL, "key": "filing_10k"}),
            },
            {
                "type": "function_call_output",
                "call_id": "call_parse_1",
                "output": json.dumps({"result": "Page stored successfully"}),
            },
            {
                "type": "function_call",
                "name": "retrieve_information",
                "call_id": "call_retrieve_1",
                "arguments": json.dumps({"prompt": "What was the revenue? {{filing_10k}}"}),
            },
            {
                "type": "function_call_output",
                "call_id": "call_retrieve_1",
                "output": json.dumps({"result": "Revenue was $1.23 billion in 2024."}),
            },
            {
                "type": "function_call",
                "name": "submit_final_result",
                "call_id": "call_submit_1",
                "arguments": json.dumps({"final_result": "Revenue was $1.23 billion in 2024."}),
            },
        ],
    }


class TestRealisticTrace:
    def test_realistic_trace_extraction(self):
        row = _make_realistic_trace()
        sidecar = parse_rollout_line(json.dumps(row))
        assert sidecar.trace is not None
        assert sidecar.trace.has_submit is True
        assert sidecar.trace.answer == "Revenue was $1.23 billion in 2024."
        assert len(sidecar.trace.evidence) == 1
        chunk = sidecar.trace.evidence[0]
        assert chunk.text == "Revenue was $1.23 billion in 2024."
        assert chunk.source_id is not None
        assert chunk.attribution_state == "available"
        assert "cik=0001811414" in chunk.source_id
        assert "accession=000181141425000010" in chunk.source_id
        assert "doc=10-K.htm" in chunk.source_id

    def test_realistic_trace_evaluates_allow(self):
        row = _make_realistic_trace()
        raw_line = json.dumps(row)
        evaluator, _, config = _make_finance_evaluator(nli_label="entailment")
        result = evaluate_row(raw_line, row, 0, evaluator, config)
        assert result["verdict"]["status"] == "allow"
        assert result["verdict"]["reason"] == "all_entailed"


# ---------------------------------------------------------------------------
# parse_html_page plain key tests
# ---------------------------------------------------------------------------


class TestParsePlainKey:
    def test_parse_html_page_reads_plain_key(self):
        row = _make_rollout_row()
        row["output"] = [
            {
                "type": "function_call",
                "name": "parse_html_page",
                "call_id": "call_parse_1",
                "arguments": json.dumps({"url": _FILING_URL, "key": "filing_10k"}),
            },
            {
                "type": "function_call_output",
                "call_id": "call_parse_1",
                "output": json.dumps({"result": "Page stored"}),
            },
            {
                "type": "function_call",
                "name": "retrieve_information",
                "call_id": "call_retrieve_1",
                "arguments": json.dumps({"prompt": "Revenue {{filing_10k}}"}),
            },
            {
                "type": "function_call_output",
                "call_id": "call_retrieve_1",
                "output": json.dumps({"result": "Revenue was $1.23 billion in 2024."}),
            },
            {
                "type": "function_call",
                "name": "submit_final_result",
                "call_id": "call_submit_1",
                "arguments": json.dumps({"final_result": "Revenue was $1.23 billion in 2024."}),
            },
        ]
        sidecar = parse_rollout_line(json.dumps(row))
        assert sidecar.trace is not None
        assert len(sidecar.trace.evidence) == 1
        chunk = sidecar.trace.evidence[0]
        assert chunk.source_id is not None
        assert chunk.attribution_state == "available"


# ---------------------------------------------------------------------------
# Error exclusion tests
# ---------------------------------------------------------------------------


class TestErrorExclusion:
    def test_raw_error_string_excluded(self):
        row = _make_rollout_row()
        row["output"] = [
            {
                "type": "function_call",
                "name": "retrieve_information",
                "call_id": "call_r1",
                "arguments": json.dumps({"prompt": "query"}),
            },
            {
                "type": "function_call_output",
                "call_id": "call_r1",
                "output": "Error: Failed to retrieve data",
            },
            {
                "type": "function_call",
                "name": "submit_final_result",
                "call_id": "call_s1",
                "arguments": json.dumps({"final_result": "Some answer."}),
            },
        ]
        sidecar = parse_rollout_line(json.dumps(row))
        assert sidecar.trace is not None
        assert len(sidecar.trace.evidence) == 0
        assert any("error" in e.lower() for e in sidecar.trace.extraction_errors)

    def test_traceback_excluded(self):
        row = _make_rollout_row()
        row["output"] = [
            {
                "type": "function_call",
                "name": "retrieve_information",
                "call_id": "call_r1",
                "arguments": json.dumps({"prompt": "query"}),
            },
            {
                "type": "function_call_output",
                "call_id": "call_r1",
                "output": "Traceback (most recent call last):\n  File ...",
            },
            {
                "type": "function_call",
                "name": "submit_final_result",
                "call_id": "call_s1",
                "arguments": json.dumps({"final_result": "Some answer."}),
            },
        ]
        sidecar = parse_rollout_line(json.dumps(row))
        assert sidecar.trace is not None
        assert len(sidecar.trace.evidence) == 0

    def test_unavailable_string_excluded(self):
        row = _make_rollout_row()
        row["output"] = [
            {
                "type": "function_call",
                "name": "retrieve_information",
                "call_id": "call_r1",
                "arguments": json.dumps({"prompt": "query"}),
            },
            {
                "type": "function_call_output",
                "call_id": "call_r1",
                "output": "unavailable",
            },
            {
                "type": "function_call",
                "name": "submit_final_result",
                "call_id": "call_s1",
                "arguments": json.dumps({"final_result": "Some answer."}),
            },
        ]
        sidecar = parse_rollout_line(json.dumps(row))
        assert sidecar.trace is not None
        assert len(sidecar.trace.evidence) == 0

    def test_dict_error_result_excluded(self):
        """A dict-shaped failed retrieval (success=False, error) must be excluded.

        This test does not skip.  It proves that structured tool results
        are preserved as Any (not coerced to str) so error dicts with
        ``success`` false / ``error`` cannot become evidence.
        """
        row = _make_rollout_row()
        row["output"] = [
            {
                "type": "function_call",
                "name": "retrieve_information",
                "call_id": "call_r1",
                "arguments": json.dumps({"prompt": "query"}),
            },
            {
                "type": "function_call_output",
                "call_id": "call_r1",
                "output": {"success": False, "error": "retrieval failed"},
            },
            {
                "type": "function_call",
                "name": "submit_final_result",
                "call_id": "call_s1",
                "arguments": json.dumps({"final_result": "Some answer."}),
            },
        ]
        sidecar = parse_rollout_line(json.dumps(row))
        assert sidecar.trace is not None
        assert len(sidecar.trace.evidence) == 0
        assert any("error" in e.lower() for e in sidecar.trace.extraction_errors)

        raw_line = json.dumps(row)
        evaluator, _, config = _make_finance_evaluator(nli_label="entailment")
        result = evaluate_row(raw_line, row, 0, evaluator, config)
        assert result["verdict"]["status"] == "unavailable"
        assert result["verdict"]["reason"] == "trace_extraction_errors"
        assert len(result["evidence"]) == 0


# ---------------------------------------------------------------------------
# Normalized accession tests
# ---------------------------------------------------------------------------


class TestNormalizedAccession:
    def test_accession_dashes_stripped(self):
        from nvflow.recipes.finance.utils.rl.provenanceguard import (
            _format_accession,
        )

        assert _format_accession("0001811414-25-000010") == "000181141425000010"

    def test_accession_already_digits(self):
        from nvflow.recipes.finance.utils.rl.provenanceguard import (
            _format_accession,
        )

        assert _format_accession("000181141425000010") == "000181141425000010"

    def test_accession_short_returns_none(self):
        from nvflow.recipes.finance.utils.rl.provenanceguard import (
            _format_accession,
        )

        assert _format_accession("123") is None

    def test_accession_overlong_returns_none(self):
        from nvflow.recipes.finance.utils.rl.provenanceguard import (
            _format_accession,
        )

        assert _format_accession("000181141425000010999") is None
        assert _format_accession("0001811414-25-000010-999") is None

    def test_source_id_uses_digit_only_accession(self):
        from nvflow.recipes.finance.utils.rl.provenanceguard import (
            _extract_filing_from_url,
        )

        filing = _extract_filing_from_url(_FILING_URL)
        assert filing.accession == "000181141425000010"
        sid = filing.source_id()
        assert sid is not None
        assert "accession=000181141425000010" in sid


# ---------------------------------------------------------------------------
# Percentage extraction tests
# ---------------------------------------------------------------------------


class TestPercentageExtraction:
    def test_percentage_at_end_of_string(self):
        result = extract_protected_values("Revenue grew 12.3%")
        assert len(result) == 1
        assert result[0].kind == "percentage"
        assert "12.3%" in result[0].normalized

    def test_percentage_before_word(self):
        result = extract_protected_values("Growth was 12.3% YoY")
        assert len(result) == 1
        assert result[0].kind == "percentage"

    def test_percentage_protected_value_check_pass(self):
        outcome, _ = check_protected_values("Margin was 12.3%", "Margin was 12.3%")
        assert outcome == "pass"

    def test_percentage_protected_value_check_fail(self):
        outcome, missing = check_protected_values("Margin was 12.3%", "Margin was 15.0%")
        assert outcome == "fail"
        assert len(missing) == 1


# ---------------------------------------------------------------------------
# Unattributable chunk blocks no_source without NLI
# ---------------------------------------------------------------------------


class TestUnattributableChunk:
    def test_unavailable_chunk_blocks_no_source_without_nli(self):
        evidence = [
            _make_evidence(
                text="Some text about revenue.",
                source_id=None,
                attribution_state="unavailable",
            ),
        ]
        evaluator, nli = _make_evaluator(nli_label="entailment")
        nli.set_error(True)
        decision = evaluator.evaluate("Revenue was $1.23 billion.", evidence)
        assert decision.status == "block"
        assert decision.reason == "no_source"
        assert len(decision.verdicts) == 1
        assert decision.verdicts[0].final_label == "no_source"
        assert decision.verdicts[0].errors == ()

    def test_composite_chunk_blocks_no_source(self):
        evidence = [
            _make_evidence(
                text="Some text.",
                source_id=None,
                attribution_state="composite",
            ),
        ]
        evaluator, nli = _make_evaluator(nli_label="entailment")
        nli.set_error(True)
        decision = evaluator.evaluate("Revenue was $1.23 billion.", evidence)
        assert decision.status == "block"
        assert decision.reason == "no_source"


# ---------------------------------------------------------------------------
# Top trace failure: extraction errors force unavailable preserving verdicts
# ---------------------------------------------------------------------------


class TestTopTraceFailure:
    def test_extraction_errors_force_unavailable_preserving_verdicts(self):
        row = _make_rollout_row()
        row["output"] = [
            {
                "type": "function_call",
                "name": "retrieve_information",
                "call_id": "call_r1",
                "arguments": json.dumps({"prompt": "query1"}),
            },
            {
                "type": "function_call_output",
                "call_id": "call_r1",
                "output": json.dumps({"result": "Revenue was $1.23 billion in 2024."}),
            },
            {
                "type": "function_call",
                "name": "retrieve_information",
                "call_id": "call_r2",
                "arguments": json.dumps({"prompt": "query2"}),
            },
            {
                "type": "function_call_output",
                "call_id": "call_r2",
                "output": "Error: Failed to retrieve data",
            },
            {
                "type": "function_call",
                "name": "submit_final_result",
                "call_id": "call_s1",
                "arguments": json.dumps({"final_result": "Revenue was $1.23 billion in 2024."}),
            },
        ]
        raw_line = json.dumps(row)
        evaluator, _, config = _make_finance_evaluator(nli_label="entailment")
        result = evaluate_row(raw_line, row, 0, evaluator, config)
        assert result["verdict"]["status"] == "unavailable"
        assert result["verdict"]["reason"] == "trace_extraction_errors"
        assert len(result["verdict"]["verdicts"]) > 0


# ---------------------------------------------------------------------------
# Model adapter tests
# ---------------------------------------------------------------------------


class TestModelAdapters:
    def test_embedder_lazy_loading(self):
        from nvflow.provenanceguard.embedder import HFEmbedder

        emb = HFEmbedder()
        assert emb._model is None
        assert emb._tokenizer is None

    def test_nli_lazy_loading(self):
        from nvflow.provenanceguard.nli import HFNLI

        nli = HFNLI()
        assert nli._model is None
        assert nli._tokenizer is None
        assert nli._label_map is None

    def test_no_model_imports_at_import_time(self):
        import importlib

        mod = importlib.import_module("nvflow.provenanceguard.embedder")
        assert mod is not None
        mod2 = importlib.import_module("nvflow.provenanceguard.nli")
        assert mod2 is not None

    def test_embed_returns_plain_floats(self):
        """HFEmbedder.embed must return list[list[float]], not tensor scalars.

        Uses a testable subclass with fake tokenizer/model to exercise the
        full embed() pipeline (mean pooling, L2 norm, return) without
        loading any real model.  The assertion rejects scalar tensors
        (which the old list(v) approach produced) and proves every
        element is a plain Python float.
        """
        import torch

        from nvflow.provenanceguard.embedder import HFEmbedder

        class _FakeTokenizer:
            def __call__(self, texts, **kwargs):
                batch = len(texts)
                seq_len = 3
                return {
                    "input_ids": torch.zeros(batch, seq_len, dtype=torch.long),
                    "attention_mask": torch.ones(batch, seq_len, dtype=torch.long),
                }

        class _FakeModel:
            class Config:
                hidden_size = 4

            config = Config()

            def eval(self):
                return self

            def __call__(self, **kwargs):
                class _Output:
                    last_hidden_state = torch.randn(2, 3, 4)

                return _Output()

        class _TestableEmbedder(HFEmbedder):
            def _ensure_loaded(self):
                if self._model is not None:
                    return
                self._dim = 4
                self._tokenizer = _FakeTokenizer()
                self._model = _FakeModel()

        emb = _TestableEmbedder()
        result = emb.embed(["hello", "world"])

        assert isinstance(result, list)
        assert len(result) == 2
        for vec in result:
            assert isinstance(vec, list)
            for val in vec:
                assert isinstance(val, float), f"Expected Python float, got {type(val).__name__}"
                assert not hasattr(val, "item"), (
                    "Value is a tensor scalar, not a plain Python float"
                )

        # Sanity: the old list(v) approach would produce tensor scalars.
        old_result = [list(v) for v in torch.randn(2, 3)]
        assert hasattr(old_result[0][0], "item"), (
            "Sanity check: old list(v) approach should produce tensor scalars"
        )


# ---------------------------------------------------------------------------
# Seed interpolation tests
# ---------------------------------------------------------------------------


class TestSeedInterpolation:
    def test_omegaconf_seed_interpolation(self):
        p = Path("nvflow/recipes/finance/workflows/grpo/base.yaml")
        with open(p) as f:
            data = yaml.safe_load(f)
        eval_stage = data["stages"]["evaluate_provenance"]
        assert eval_stage["starting_seed"] == "${stages.collect_rollouts.rollout.starting_seed}"
        assert (
            eval_stage["num_random_seeds"] == "${stages.collect_rollouts.rollout.num_random_seeds}"
        )

    def test_evaluate_provenance_has_num_gpus_zero(self):
        p = Path("nvflow/recipes/finance/workflows/grpo/base.yaml")
        with open(p) as f:
            data = yaml.safe_load(f)
        assert data["stages"]["evaluate_provenance"]["num_gpus"] == 0


# ---------------------------------------------------------------------------
# Build command tests
# ---------------------------------------------------------------------------


class TestBuildCommand:
    def test_build_command_contains_module_and_flags(self):
        mod = _load_eval_stage_module()
        build_evaluate_command = mod.build_evaluate_command
        cmd = build_evaluate_command(
            input_file="/input/r.jsonl",
            output_file="/output/p.jsonl",
            seed=3,
            environment="finance_sec_search",
        )
        assert "python3 -m" in cmd
        assert "nvflow.recipes.finance.utils.rl.provenanceguard" in cmd
        assert "--input_file" in cmd
        assert "--output_file" in cmd
        assert "--seed" in cmd
        assert "--environment" in cmd

    def test_build_command_is_pure(self):
        mod = _load_eval_stage_module()
        build_evaluate_command = mod.build_evaluate_command
        cmd1 = build_evaluate_command(
            input_file="/in.jsonl",
            output_file="/out.jsonl",
            seed=0,
            environment="test",
        )
        cmd2 = build_evaluate_command(
            input_file="/in.jsonl",
            output_file="/out.jsonl",
            seed=0,
            environment="test",
        )
        assert cmd1 == cmd2


# ---------------------------------------------------------------------------
# Path overlap tests
# ---------------------------------------------------------------------------


class TestPathOverlap:
    def _stage(self):
        mod = _load_eval_stage_module()
        return mod.EvaluateProvenanceStage()

    def test_equal_paths_rejected(self, tmp_path):
        stage = self._stage()
        p = str(tmp_path / "same")
        with pytest.raises(ValueError, match="same path"):
            stage.validate_config({"rollouts_dir": p, "output_dir": p})

    def test_output_inside_rollouts_rejected(self, tmp_path):
        stage = self._stage()
        with pytest.raises(ValueError, match="inside rollouts_dir"):
            stage.validate_config(
                {
                    "rollouts_dir": str(tmp_path),
                    "output_dir": str(tmp_path / "sub"),
                }
            )

    def test_rollouts_inside_output_rejected(self, tmp_path):
        stage = self._stage()
        with pytest.raises(ValueError, match="inside output_dir"):
            stage.validate_config(
                {
                    "rollouts_dir": str(tmp_path / "sub"),
                    "output_dir": str(tmp_path),
                }
            )

    def test_non_overlapping_paths_accepted(self, tmp_path):
        stage = self._stage()
        stage.validate_config(
            {
                "rollouts_dir": str(tmp_path / "rollouts"),
                "output_dir": str(tmp_path / "output"),
            }
        )


# ---------------------------------------------------------------------------
# Missing input marker tests
# ---------------------------------------------------------------------------


class TestMissingInputMarker:
    def test_missing_input_marker_preserves_output(self, tmp_path):
        evaluator, _, config = _make_finance_evaluator()
        input_file = tmp_path / "input.jsonl"
        input_file.write_text(json.dumps(_make_rollout_row()) + "\n")
        output_file = tmp_path / "output.jsonl"
        output_file.write_text("prior output")
        done_path = str(output_file) + ".done"
        Path(done_path).write_text("prior done")
        with pytest.raises(RuntimeError, match="Input completion marker"):
            evaluate_seed(str(input_file), str(output_file), 0, evaluator, config)
        assert output_file.read_text() == "prior output"
        assert Path(done_path).read_text() == "prior done"

    def test_missing_input_file_raises(self, tmp_path):
        evaluator, _, config = _make_finance_evaluator()
        output_file = str(tmp_path / "output.jsonl")
        with pytest.raises(FileNotFoundError, match="Input file does not exist"):
            evaluate_seed(
                str(tmp_path / "nonexistent.jsonl"),
                output_file,
                0,
                evaluator,
                config,
            )


# ---------------------------------------------------------------------------
# Fixed policy tests
# ---------------------------------------------------------------------------


class TestFixedPolicy:
    def test_config_has_no_block_on_fields(self):
        cfg = ProvenanceGuardConfig()
        assert not hasattr(cfg, "block_on_contradiction")
        assert not hasattr(cfg, "block_on_neutral")
        assert not hasattr(cfg, "block_on_no_source")
        assert not hasattr(cfg, "block_on_protected_value_mismatch")

    def test_config_rejects_block_on_kwargs(self):
        with pytest.raises(TypeError):
            ProvenanceGuardConfig(block_on_contradiction=False)  # type: ignore[call-arg]

    def test_to_dict_has_no_block_on(self):
        result = FinanceEvaluatorConfig().to_dict()
        thresholds = result["thresholds"]
        assert "block_on_contradiction" not in thresholds
        assert "block_on_neutral" not in thresholds
        assert "block_on_no_source" not in thresholds
        assert "block_on_protected_value_mismatch" not in thresholds
        assert thresholds["policy"] == "fixed_fail_closed"


# ---------------------------------------------------------------------------
# Deterministic per-line IDs for duplicate/invalid lines
# ---------------------------------------------------------------------------


class TestDuplicateLineIDs:
    def test_duplicate_lines_get_different_uuids(self, tmp_path):
        evaluator, _, config = _make_finance_evaluator()
        row = _make_rollout_row()
        raw_line = json.dumps(row)
        input_file = tmp_path / "input.jsonl"
        input_file.write_text(raw_line + "\n" + raw_line + "\n")
        Path(str(input_file) + ".done").touch()
        output_file = str(tmp_path / "output.jsonl")
        evaluate_seed(str(input_file), output_file, 0, evaluator, config)
        lines = Path(output_file).read_text().strip().split("\n")
        r1 = json.loads(lines[0])
        r2 = json.loads(lines[1])
        assert r1["evaluation_uuid"] != r2["evaluation_uuid"]
        assert r1["raw_line_fingerprint"] == r2["raw_line_fingerprint"]
        assert r1["line_number"] == 0
        assert r2["line_number"] == 1

    def test_duplicate_invalid_lines_get_different_uuids(self, tmp_path):
        evaluator, _, config = _make_finance_evaluator()
        input_file = tmp_path / "input.jsonl"
        input_file.write_text("not valid json\nnot valid json\n")
        Path(str(input_file) + ".done").touch()
        output_file = str(tmp_path / "output.jsonl")
        evaluate_seed(str(input_file), output_file, 0, evaluator, config)
        lines = Path(output_file).read_text().strip().split("\n")
        r1 = json.loads(lines[0])
        r2 = json.loads(lines[1])
        assert r1["evaluation_uuid"] != r2["evaluation_uuid"]
        assert r1["raw_line_fingerprint"] == r2["raw_line_fingerprint"]
