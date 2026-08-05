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
"""Tests for prompt_template_applier stale-output cleanup + content equivalence.

Pins the invariant that each successful run leaves only outputs derived
from the *current* input set in ``{output_dir}/{env_name}/``.  Without
this, a rerun where data_transformation produced fewer chunks than a
prior run (e.g. ``num_chunks`` reduced) would silently leave higher-index
output files on disk; the downstream ``responses_api_converter`` globs
``*.jsonl`` from this directory and would mix stale records into the
new dataset.

Mirrors ``tests/test_dataset_transformer.py`` for the parallel cleanup
in ``dataset_transformer.py``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from nvflow.recipes.finance.utils.rl.prompt_template_applier import DateResolver


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _make_prompt_template(tmp_path: Path) -> Path:
    """Minimal prompt template that exercises the format() path."""
    template = tmp_path / "template.yaml"
    template.write_text(
        "user: |\n  Answer the following question.\n\n  Question: {problem}\n",
        encoding="utf-8",
    )
    return template


def _make_input_chunks(input_dir: Path, num_chunks: int, records_per_chunk: int = 3) -> None:
    """Build ``num_chunks`` JSONL chunks shaped like data_transformation output."""
    for chunk_idx in range(1, num_chunks + 1):
        chunk_path = input_dir / f"final_result_chunk{chunk_idx}.jsonl"
        records = [
            {
                "problem": f"What is the revenue in chunk {chunk_idx} record {i}?",
                "context": f"Filing context {chunk_idx}-{i}",
                "generation": f"${chunk_idx}{i}M",
                "uuid": f"uuid-{chunk_idx}-{i}",
                "question_type": "factual",
            }
            for i in range(records_per_chunk)
        ]
        _write_jsonl(chunk_path, records)


def _run_applier(
    input_dir: Path,
    output_dir: Path,
    template: Path,
) -> subprocess.CompletedProcess[str]:
    """Invoke the applier as a subprocess so we exercise main() + argparse."""
    cmd = [
        sys.executable,
        "-m",
        "nvflow.recipes.finance.utils.rl.prompt_template_applier",
        str(input_dir),
        str(output_dir),
        "--prompt_template",
        str(template),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def test_stale_chunk_cleanup_removes_higher_index_outputs(tmp_path: Path) -> None:
    """A rerun with fewer input chunks must delete higher-index stale outputs.

    Reproduces the footgun: data_transformation rerun with smaller
    ``--num_chunks`` shrinks the step-1 chunk set; if step-2 doesn't
    clean up its prior run's outputs, ``responses_api_converter`` would
    glob the union of fresh + stale chunks and mix records.
    """
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    template = _make_prompt_template(tmp_path)

    # First run: 5 input chunks → 5 output chunks at output_dir root.
    _make_input_chunks(input_dir, num_chunks=5)
    _run_applier(input_dir, output_dir, template)
    chunks_after_run1 = sorted(output_dir.glob("final_result_chunk*.jsonl"))
    assert len(chunks_after_run1) == 5, (
        f"first run should have produced 5 outputs, got: {chunks_after_run1}"
    )

    # Second run: 2 input chunks (simulate data_transformation rerun with
    # smaller num_chunks).  Wipe input, regenerate with NEW content
    # (different ``problem`` strings so we can detect run-1 records
    # bleeding through into the surviving chunk files).
    for old in input_dir.glob("*.jsonl"):
        old.unlink()
    for chunk_idx in range(1, 3):
        chunk_path = input_dir / f"final_result_chunk{chunk_idx}.jsonl"
        records = [
            {
                "problem": f"RUN2-CHUNK{chunk_idx}-RECORD{i}",
                "context": f"run2 ctx {chunk_idx}-{i}",
                "generation": f"run2 gen {chunk_idx}-{i}",
                "uuid": f"run2-uuid-{chunk_idx}-{i}",
                "question_type": "factual",
            }
            for i in range(3)
        ]
        _write_jsonl(chunk_path, records)

    result = _run_applier(input_dir, output_dir, template)

    chunks_after_run2 = sorted(output_dir.glob("final_result_chunk*.jsonl"))
    chunk_names = sorted(p.name for p in chunks_after_run2)
    # Chunks 3..5 from run 1 must be deleted -- this is the stale-output
    # bug the cleanup is fixing.
    assert chunk_names == [
        "final_result_chunk1.jsonl",
        "final_result_chunk2.jsonl",
    ], f"only chunks 1..2 should remain, got: {chunk_names}"

    # Surviving chunks 1..2 must contain run-2 records, NOT run-1.  The
    # cleanup deletes-only-if-stale (preserves files in the new input
    # set), so chunks 1..2 are truncate-and-overwritten in place; this
    # check confirms the *content* is fresh even though the inode is
    # reused.
    for chunk in chunks_after_run2:
        with chunk.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                assert rec["problem"].startswith("RUN2-"), (
                    f"{chunk.name} contains stale run-1 record: {rec.get('problem')!r}"
                )

    # Log line is the operator's signal that cleanup ran -- pin it so a
    # silent regression in the cleanup path is loud.
    assert "stale chunk file" in result.stderr or "stale chunk file" in result.stdout, (
        "cleanup log line missing -- operators rely on it to know cleanup ran"
    )


def test_first_run_does_not_emit_stale_cleanup_log(tmp_path: Path) -> None:
    """First-time runs (empty output_dir) must not emit the cleanup log."""
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    template = _make_prompt_template(tmp_path)

    _make_input_chunks(input_dir, num_chunks=3)
    result = _run_applier(input_dir, output_dir, template)

    chunks = sorted(output_dir.glob("final_result_chunk*.jsonl"))
    assert len(chunks) == 3
    # The "Removed N stale chunk file(s)" log line must NOT appear when
    # there's nothing stale to remove -- avoids confusing operators on
    # first runs.
    assert "stale chunk file" not in result.stderr
    assert "stale chunk file" not in result.stdout


def test_cleanup_preserves_errors_jsonl(tmp_path: Path) -> None:
    """``errors.jsonl`` must NOT be swept by the stale-output cleanup.

    Operators triage flaky inputs by reading ``errors.jsonl`` across
    reruns; deleting it on a no-error rerun would erase that audit
    trail.  ``responses_api_converter`` filters records missing the
    ``prompt`` field into its own skipped stream, so a stale
    ``errors.jsonl`` is harmless to downstream correctness.
    """
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    template = _make_prompt_template(tmp_path)

    # First run: 3 input chunks, plant a fake errors.jsonl that the
    # cleanup must not touch.
    _make_input_chunks(input_dir, num_chunks=3)
    output_dir.mkdir(parents=True, exist_ok=True)
    errors_path = output_dir / "errors.jsonl"
    sentinel = "preexisting error from a prior run\n"
    errors_path.write_text(sentinel, encoding="utf-8")

    # Second run: shrink to 1 chunk so the cleanup definitely fires.
    for old in input_dir.glob("*.jsonl"):
        old.unlink()
    _make_input_chunks(input_dir, num_chunks=1)

    # Pre-stage a stale chunk in output_dir so cleanup has work to do.
    (output_dir / "final_result_chunk2.jsonl").write_text(
        '{"prompt": "stale", "expected_answer": "stale"}\n', encoding="utf-8"
    )
    (output_dir / "final_result_chunk3.jsonl").write_text(
        '{"prompt": "stale", "expected_answer": "stale"}\n', encoding="utf-8"
    )

    _run_applier(input_dir, output_dir, template)

    # Stale per-chunk outputs were swept ...
    chunks = sorted(output_dir.glob("final_result_chunk*.jsonl"))
    assert sorted(p.name for p in chunks) == ["final_result_chunk1.jsonl"]

    # ... but errors.jsonl survived untouched.
    assert errors_path.exists(), "errors.jsonl was unexpectedly deleted by cleanup"
    assert errors_path.read_text(encoding="utf-8") == sentinel, (
        "errors.jsonl contents were mutated by cleanup"
    )


def test_chunk_records_preserve_input_content(tmp_path: Path) -> None:
    """Output chunks must contain exactly the records the input described,
    with the prompt field rendered from the template.

    Parses outputs with stdlib ``json.loads`` (the applier writes via
    ``orjson``) to guard the orjson-output ↔ stdlib-input contract that
    downstream stages (``responses_api_converter``, ``ng_prepare_data``)
    rely on.
    """
    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    template = _make_prompt_template(tmp_path)

    _make_input_chunks(input_dir, num_chunks=2, records_per_chunk=4)
    _run_applier(input_dir, output_dir, template)

    out_chunks = sorted(output_dir.glob("final_result_chunk*.jsonl"))
    assert len(out_chunks) == 2

    all_records: list[dict] = []
    for chunk in out_chunks:
        with chunk.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                all_records.append(json.loads(line))

    assert len(all_records) == 8, f"expected 8 records, got {len(all_records)}"

    for rec in all_records:
        # Every output carries the template-rendered prompt with the raw
        # problem substituted in -- this is the entire point of the stage.
        assert "Question:" in rec["prompt"], (
            f"prompt missing template scaffolding: {rec['prompt']!r}"
        )
        assert rec["problem"] in rec["prompt"], (
            f"problem text not substituted into prompt for record: {rec.get('uuid')}"
        )
        # ``context`` is cleared after rendering (per apply_template
        # contract) -- downstream stages must not see the raw context.
        assert rec["context"] == "", "context field must be cleared after templating"
        # ``expected_answer`` falls back to full generation when no
        # ``--answer_prefix`` is provided (this test omits the flag).
        assert rec["expected_answer"] == rec["generation"]
        # uuid is preserved verbatim from data_transformation output.
        assert rec["uuid"].startswith("uuid-")


# ============================================================================
# P1: DateResolver emits long-form dates ("February 23, 2022") to match
# vals-ai/finance-agent eval's verbatim "current date is April 07, 2025"
# style.  Pre-P1 the resolver emitted ISO ("2022-02-23"), which trained the
# policy to expect ISO dates and broke generalization to eval.
# ============================================================================


def _make_resolver(
    tmp_path: Path,
    *,
    accession: str = "0000320193-22-000108",
    problem: str = "What was Apple's revenue?",
    filing_date: str = "2022-09-01",
    fallback: str = "2025-04-07",
    jitter_min: int = 1,
    jitter_max: int = 60,
) -> DateResolver:
    """Build a DateResolver from a minimal in-memory parquet + raw SDG file."""
    parquet_path = tmp_path / "sec_metadata.parquet"
    pd.DataFrame(
        {
            "accession_number": [accession],
            "filing_date": [filing_date],
        }
    ).to_parquet(parquet_path)

    raw_sdg = tmp_path / "raw_sdg.jsonl"
    raw_sdg.write_text(
        json.dumps({"problem": problem, "file_path0": f"a/b/c/{accession}/d.html"}) + "\n",
        encoding="utf-8",
    )

    return DateResolver(
        parquet_path=str(parquet_path),
        raw_sdg_path=str(raw_sdg),
        jitter_min_days=jitter_min,
        jitter_max_days=jitter_max,
        fallback_current_date=fallback,
        parquet_accession_column="accession_number",
        parquet_filing_date_column="filing_date",
    )


def test_resolver_returns_long_form_date(tmp_path: Path) -> None:
    """Resolved dates must be long-form (e.g. "September 02, 2022") so the
    rendered system prompt matches eval's wording verbatim.

    Pre-P1 this returned "%Y-%m-%d" and the policy saw "2022-09-02" while
    eval shows "April 07, 2025" -- a silent train/eval format skew.
    """
    resolver = _make_resolver(tmp_path, filing_date="2022-09-01")

    record = {"problem": "What was Apple's revenue?", "uuid": "fixed-seed-uuid"}
    rendered, source = resolver.resolve(record)

    assert source == "resolved"
    # Long-form check: must contain a written-out month and a 4-digit year,
    # NOT an ISO "YYYY-MM-DD" date.
    assert "-" not in rendered, f"date should be long-form, got: {rendered!r}"
    assert "2022" in rendered
    # The jitter range is [1, 60] days past 2022-09-01, so the month is
    # September or October -- pin to those two so we don't have to mock
    # random.Random just to assert the format.
    assert ("September" in rendered) or ("October" in rendered), f"unexpected month in {rendered!r}"


def test_resolver_fallback_returns_long_form_date(tmp_path: Path) -> None:
    """Fallback path (unknown problem) must also emit long-form so success
    and fallback records both share eval's date wording.

    Pre-P1 the fallback path returned the raw ISO ``fallback_current_date``
    while the success path returned long-form -- a mixed-format dataset
    that the policy could learn to discriminate against.
    """
    resolver = _make_resolver(tmp_path, fallback="2025-04-07")

    record = {"problem": "UNKNOWN PROBLEM NOT IN MAP", "uuid": "irrelevant"}
    rendered, source = resolver.resolve(record)

    assert source == "fallback"
    assert rendered == "April 07, 2025"


def test_resolver_non_iso_fallback_passes_through(tmp_path: Path) -> None:
    """Operators sometimes pass an already-long-form fallback (e.g. when
    copy-pasting from eval).  The init must not crash on that and must
    preserve the value as-is.
    """
    resolver = _make_resolver(tmp_path, fallback="April 07, 2025")

    record = {"problem": "UNKNOWN", "uuid": "u"}
    rendered, source = resolver.resolve(record)

    assert source == "fallback"
    assert rendered == "April 07, 2025"


def test_resolver_resolved_date_is_deterministic(tmp_path: Path) -> None:
    """Same (problem, uuid) → same resolved date.  Seed-derived jitter
    determinism is what lets us shuffle/replay the dataset without
    changing the per-record current_date field.
    """
    resolver = _make_resolver(tmp_path, filing_date="2022-09-01")

    record = {"problem": "What was Apple's revenue?", "uuid": "seed-X"}
    a, _ = resolver.resolve(record)
    b, _ = resolver.resolve(record)
    assert a == b
