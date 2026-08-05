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
"""Tests for dataset_transformer chunk-cleanup behavior.

Pins the invariant that each successful run leaves only the chunks
produced by *this* run in ``{output_dir}/chunks/``.  Without this,
re-running with a smaller ``--num_chunks`` than a previous run would
silently leave higher-index chunk files on disk; downstream stages
(e.g. apply_prompt_template) glob this directory and would mix stale
records into the new dataset.

This invariant is shared by both SFT (stage 0) and GRPO (stage 1)
pipelines -- both call into the same ``dataset_transformer`` module --
so a regression here would corrupt both training paths.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _make_input(tmp_path: Path, num_records: int) -> Path:
    """Build a minimal SDG-shaped JSONL the transformer accepts."""
    records = [
        {
            "problem": f"What was the revenue in fiscal year {i}?",
            "context": f"Acme Corp filing {i}: revenue was ${i}M.",
            "generation": f"${i}M",
            "question_type": "factual",
            # source_format=separated requires non-empty reasoning_content
            # OR per-answer answer_reasoning_content_{idx}.  Use the
            # simpler reasoning_content path.
            "reasoning_content": f"Looking at filing {i}, revenue is reported as ${i}M.",
        }
        for i in range(num_records)
    ]
    input_path = tmp_path / "input.jsonl"
    _write_jsonl(input_path, records)
    return input_path


def _run_transformer(
    input_path: Path,
    output_path: Path,
    num_chunks: int,
) -> subprocess.CompletedProcess[str]:
    """Invoke the transformer as a subprocess so we exercise the same
    code path the Slurm job would (including ``main()`` and argparse).
    """
    cmd = [
        sys.executable,
        "-m",
        "nvflow.recipes.finance.utils.shared.dataset_transformer",
        str(input_path),
        "--output_file",
        str(output_path),
        "--source_format",
        "separated",
        "--reasoning_mode",
        "none",
        "--num_chunks",
        str(num_chunks),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def test_chunk_cleanup_removes_stale_chunks_from_previous_run(tmp_path: Path) -> None:
    """A rerun with smaller --num_chunks must delete higher-index stale chunks.

    Reproduces the pre-fix footgun: if a run with ``--num_chunks=10``
    is followed by a rerun with ``--num_chunks=3``, chunks 4..10 from
    the first run survived on disk and were silently picked up by
    apply_prompt_template's input glob, mixing stale data into the new
    dataset.
    """
    input_path = _make_input(tmp_path, num_records=20)
    output_path = tmp_path / "out" / "final_result.jsonl"
    chunks_dir = output_path.parent / "chunks"

    # First run with 10 chunks -- writes final_result_chunk{1..10}.jsonl
    _run_transformer(input_path, output_path, num_chunks=10)
    chunks_after_run1 = sorted(chunks_dir.glob("final_result_chunk*.jsonl"))
    assert len(chunks_after_run1) == 10, (
        f"first run should have produced 10 chunks, got: {chunks_after_run1}"
    )

    # Second run with 3 chunks -- must delete chunks 4..10 BEFORE writing.
    _run_transformer(input_path, output_path, num_chunks=3)
    chunks_after_run2 = sorted(chunks_dir.glob("final_result_chunk*.jsonl"))

    chunk_names = sorted(p.name for p in chunks_after_run2)
    assert chunk_names == [
        "final_result_chunk1.jsonl",
        "final_result_chunk2.jsonl",
        "final_result_chunk3.jsonl",
    ], f"only chunks 1..3 should remain, got: {chunk_names}"

    # NOTE: we intentionally do NOT assert inode identity to prove
    # "delete-then-rewrite" ordering.  Inode reuse is filesystem-dependent
    # (overlayfs / tmpfs reuse freed inode numbers), which causes false
    # failures in containerized CI.  The behaviour that matters -- no stale
    # chunks from a previous, larger run survive -- is already verified by the
    # chunk_names assertion above.


def test_chunk_cleanup_is_a_no_op_on_first_run(tmp_path: Path) -> None:
    """First-time runs (no chunks_dir, or empty chunks_dir) must not error."""
    input_path = _make_input(tmp_path, num_records=10)
    output_path = tmp_path / "out" / "final_result.jsonl"

    result = _run_transformer(input_path, output_path, num_chunks=5)
    chunks_dir = output_path.parent / "chunks"
    chunks = sorted(chunks_dir.glob("final_result_chunk*.jsonl"))

    assert len(chunks) == 5
    # The "Removed N stale chunk file(s)" log line must NOT appear when
    # there's nothing stale to remove -- we don't want to confuse log
    # readers on first runs.
    assert "stale chunk file" not in result.stderr
    assert "stale chunk file" not in result.stdout


def test_chunk_cleanup_does_not_touch_sibling_artefacts(tmp_path: Path) -> None:
    """``errors.jsonl``, ``duplicates.jsonl``, ``filtered_outliers.jsonl``
    live in ``output_dir/`` (the parent of ``chunks/``) and must not be
    affected by the chunk-cleanup glob.  Also any unrelated file inside
    ``chunks/`` that doesn't match ``final_result_chunk*.jsonl`` must be
    preserved.
    """
    input_path = _make_input(tmp_path, num_records=10)
    output_path = tmp_path / "out" / "final_result.jsonl"
    output_dir = output_path.parent
    chunks_dir = output_dir / "chunks"

    # Pre-populate sibling artefacts (output_dir/) and a non-chunk file
    # inside chunks_dir/ that the cleanup must NOT touch.
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "errors.jsonl").write_text("preexisting errors\n", encoding="utf-8")
    (output_dir / "duplicates.jsonl").write_text("preexisting dups\n", encoding="utf-8")
    (output_dir / "filtered_outliers.jsonl").write_text("preexisting outliers\n", encoding="utf-8")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    (chunks_dir / "README.md").write_text("# chunk dir notes\n", encoding="utf-8")

    _run_transformer(input_path, output_path, num_chunks=2)

    # Sibling artefacts in output_dir/ are untouched.
    assert (output_dir / "errors.jsonl").read_text(encoding="utf-8") == "preexisting errors\n"
    assert (output_dir / "duplicates.jsonl").read_text(encoding="utf-8") == "preexisting dups\n"
    assert (output_dir / "filtered_outliers.jsonl").read_text(
        encoding="utf-8"
    ) == "preexisting outliers\n"

    # Non-matching file inside chunks_dir/ is also untouched.
    assert (chunks_dir / "README.md").read_text(encoding="utf-8") == "# chunk dir notes\n"

    # And the new chunks were written.
    chunks = sorted(chunks_dir.glob("final_result_chunk*.jsonl"))
    assert len(chunks) == 2


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file via stdlib json (the transformer writes via orjson;
    this confirms downstream stdlib parsers still consume the output)."""
    out: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def test_chunk_records_preserve_input_content(tmp_path: Path) -> None:
    """Output chunks must contain exactly the records the input described.

    SFT-shared regression guard: parses the chunk output with stdlib
    ``json.loads`` (the transformer now encodes via ``orjson``), gathers
    every record across chunks, and asserts that the
    ``problem``/``generation``/``context`` triples from the input are
    present in the output.  Verifies:

    1. ``orjson``-encoded output is valid JSON consumable by stdlib parsers
       (which is what downstream stages, including SFT
       ``prepare_for_sft.py``, rely on).
    2. No record loss across chunks.
    3. Field values are byte-for-byte preserved through the transform
       (no whitespace mangling, no key drops).

    Pinned because the move to ``orjson`` changes encoded bytes
    (``{"a":1}`` vs ``{"a": 1}``); this test guarantees the *parsed*
    content is identical.
    """
    input_path = _make_input(tmp_path, num_records=12)
    output_path = tmp_path / "out" / "final_result.jsonl"
    chunks_dir = output_path.parent / "chunks"

    _run_transformer(input_path, output_path, num_chunks=4)

    chunks = sorted(chunks_dir.glob("final_result_chunk*.jsonl"))
    assert len(chunks) == 4

    all_records: list[dict] = []
    for chunk in chunks:
        all_records.extend(_read_jsonl(chunk))

    assert len(all_records) == 12, (
        f"expected 12 transformed records across all chunks, got {len(all_records)}"
    )

    # Build {problem -> record} for fast lookup; the transformer preserves
    # ``problem`` verbatim so we can match input <-> output by it.
    by_problem = {rec["problem"]: rec for rec in all_records}
    for i in range(12):
        problem = f"What was the revenue in fiscal year {i}?"
        assert problem in by_problem, f"missing record for input idx {i}"
        rec = by_problem[problem]
        assert rec["generation"] == f"${i}M"
        assert rec["context"] == f"Acme Corp filing {i}: revenue was ${i}M."
        # UUID is deterministic per (problem, generation) pair via uuid5,
        # so it must be present and non-empty -- this is the field that
        # downstream stages (responses_api_converter, ng_prepare_data)
        # join on, so a missing/blank uuid would silently break the
        # whole pipeline.
        assert rec.get("uuid"), f"record for idx {i} is missing a uuid"


def test_malformed_lines_routed_to_errors_with_line_number(tmp_path: Path) -> None:
    """Malformed JSON lines must land in ``errors.jsonl`` with a 1-based
    ``line_number`` matching their position in the input.

    Pins the contract operators rely on when grepping the source file:
    ``errors.jsonl`` line numbers map directly to ``head -n N | tail -n 1``
    of the input.  Critical because ``iter_jsonl(on_error="yield_error")``
    no longer exposes the line index natively -- the transformer rebuilds
    it via ``enumerate``, and a future refactor that drops the
    ``enumerate`` would silently break this contract.
    """
    # 50 valid records keeps the malformed-line ratio (1/50 = 2%) well
    # under the script's 5% error-rate failure threshold so we exercise
    # the success path; otherwise the script exits non-zero before
    # writing errors.jsonl.
    records = [
        {
            "problem": f"q{i}",
            "context": f"ctx{i}",
            "generation": f"a{i}",
            "question_type": "factual",
            "reasoning_content": f"reason{i}",
        }
        for i in range(50)
    ]
    input_path = tmp_path / "input.jsonl"
    malformed_at_line = 7  # 1-based; lands inside ``records`` at index 6.
    with input_path.open("w", encoding="utf-8") as fh:
        for idx, rec in enumerate(records):
            if idx == malformed_at_line - 1:
                fh.write("{not valid json\n")
            else:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    output_path = tmp_path / "out" / "final_result.jsonl"
    _run_transformer(input_path, output_path, num_chunks=1)

    errors_path = output_path.parent / "errors.jsonl"
    assert errors_path.exists(), "errors.jsonl must be written when malformed input exists"

    error_records = _read_jsonl(errors_path)
    json_decode_errors = [e for e in error_records if "JSON decode error" in e.get("error", "")]
    assert len(json_decode_errors) == 1, (
        f"expected exactly 1 JSON decode error, got {len(json_decode_errors)}: {error_records}"
    )
    err = json_decode_errors[0]
    assert err["line_number"] == malformed_at_line, (
        f"malformed line was at position {malformed_at_line} in the input, "
        f"got line_number={err['line_number']}"
    )
    assert "raw_line" in err, "decode-error records must preserve the raw_line for auditing"
