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
"""Tests for responses_api_converter content equivalence + error routing.

Pins the contract that downstream stages (``ng_prepare_data``,
``aggregate_seeds``) rely on:

- Each apply_prompt_template row is converted *losslessly* -- the
  output retains every input field plus three additions
  (``responses_create_params``, ``question``, ``expected_answer``
  passthrough).
- Rows missing required fields (``prompt`` / ``expected_answer`` /
  ``uuid``) are routed to a sibling ``errors.jsonl`` instead of the
  main output, so a single bad record doesn't poison the whole stage.
- The streaming refactor preserves the historical behaviour of
  reading either a single file or a directory of files (apply_prompt_template
  emits a directory of per-chunk JSONL files at finance-sec-search scale).

Parses outputs with stdlib ``json.loads`` (the converter writes via
``orjson``) to guard the orjson-output <-> stdlib-input contract that
``ng_prepare_data`` relies on -- it reads the converter's output with
stdlib ``json``.
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


def _make_record(idx: int | str, with_response_params: bool = False) -> dict:
    """Shape a single apply_prompt_template-style output row.

    ``idx`` accepts either an int (simple sequential ids) or a string
    (composite ids like ``"chunk1-rec0"``); both are stringified into
    the synthetic record fields.
    """
    rec: dict = {
        "prompt": f"Question {idx}: what is the answer?",
        "problem": f"raw problem {idx}",
        "context": "",
        "generation": f"generation {idx}",
        "expected_answer": f"answer {idx}",
        "uuid": f"uuid-{idx}",
        "question_type": "factual",
    }
    if with_response_params:
        # Mirrors what apply_prompt_template emits for agent-style
        # templates (e.g. finance_sec_search) -- the converter must merge
        # these into responses_create_params alongside the prompt input.
        rec["_response_params"] = {
            "tools": [{"type": "function", "name": "search"}],
            "parallel_tool_calls": False,
        }
    return rec


def _run_converter(input_path: Path, output_file: Path) -> subprocess.CompletedProcess[str]:
    """Invoke the converter as a subprocess so we exercise main() + argparse."""
    cmd = [
        sys.executable,
        "-m",
        "nvflow.recipes.finance.utils.rl.responses_api_converter",
        str(input_path),
        str(output_file),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def test_directory_input_concatenates_all_chunks(tmp_path: Path) -> None:
    """Directory input must read every JSONL file in sorted order.

    apply_prompt_template emits a directory of per-chunk JSONL files;
    the converter is the join point that turns N chunks into a single
    ``final_result.jsonl`` consumed by ng_prepare_data.
    """
    input_dir = tmp_path / "in"
    output_file = tmp_path / "out" / "final_result.jsonl"

    for chunk_idx in range(1, 4):
        _write_jsonl(
            input_dir / f"final_result_chunk{chunk_idx}.jsonl",
            [_make_record(f"chunk{chunk_idx}-rec{i}") for i in range(2)],
        )

    _run_converter(input_dir, output_file)

    with output_file.open(encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    assert len(records) == 6, f"expected 6 records (3 chunks x 2 records), got {len(records)}"
    # Sorted-by-filename ordering: chunk1's records come before chunk2's, etc.
    # ``ng_prepare_data`` doesn't depend on this, but the deterministic
    # ordering makes diffing across reruns possible.
    assert all(rec["uuid"].startswith("uuid-chunk") for rec in records)


def test_content_equivalence_preserves_all_input_fields(tmp_path: Path) -> None:
    """Output rows must contain every input field + the converter's additions.

    The converter is documented as "lossless" -- downstream debugging
    relies on being able to trace a record back to its
    apply_prompt_template output by matching ``problem`` / ``generation``
    / ``uuid``.  A regression that drops a field would silently break
    that audit trail.
    """
    input_file = tmp_path / "in.jsonl"
    output_file = tmp_path / "out.jsonl"

    input_records = [_make_record(i, with_response_params=True) for i in range(5)]
    _write_jsonl(input_file, input_records)

    _run_converter(input_file, output_file)

    with output_file.open(encoding="utf-8") as fh:
        out_records = [json.loads(line) for line in fh if line.strip()]

    assert len(out_records) == 5
    for original, converted in zip(input_records, out_records, strict=True):
        # All original fields are preserved verbatim except _response_params,
        # which is consumed and merged into responses_create_params.
        for field in ("prompt", "problem", "context", "generation", "uuid", "question_type"):
            assert converted[field] == original[field], (
                f"field {field!r} mutated: {original[field]!r} -> {converted[field]!r}"
            )
        # _response_params is consumed (popped) -- downstream must not see it.
        assert "_response_params" not in converted, (
            "_response_params leaked into output (must be merged + popped)"
        )
        # The three converter additions:
        assert converted["question"] == original["problem"], (
            "question must mirror the raw problem (used by eval/aggregate_seeds)"
        )
        assert converted["expected_answer"] == original["expected_answer"]
        rcp = converted["responses_create_params"]
        assert rcp["input"] == [{"role": "user", "content": original["prompt"]}], (
            "responses_create_params.input must wrap the rendered prompt as a user message"
        )
        # _response_params contents merged into responses_create_params.
        assert rcp["tools"] == [{"type": "function", "name": "search"}]
        assert rcp["parallel_tool_calls"] is False


def test_records_missing_required_fields_routed_to_errors(tmp_path: Path) -> None:
    """Rows missing ``prompt`` / ``expected_answer`` / ``uuid`` go to errors.jsonl.

    Without this routing, a single malformed upstream record would
    abort the whole stage and force a manual rerun.  Operators rely
    on ``errors.jsonl`` to triage upstream bugs without losing the
    rest of the dataset.
    """
    input_file = tmp_path / "in.jsonl"
    output_file = tmp_path / "out" / "final_result.jsonl"

    records = [
        _make_record(0),  # valid
        {**_make_record(1), "prompt": ""},  # missing prompt
        {**_make_record(2), "expected_answer": ""},  # missing expected_answer
        {k: v for k, v in _make_record(3).items() if k != "uuid"},  # missing uuid
        _make_record(4),  # valid
    ]
    _write_jsonl(input_file, records)

    _run_converter(input_file, output_file)

    with output_file.open(encoding="utf-8") as fh:
        out_records = [json.loads(line) for line in fh if line.strip()]
    # Only the two valid records reach the main output.
    assert len(out_records) == 2
    assert {rec["uuid"] for rec in out_records} == {"uuid-0", "uuid-4"}

    # The three malformed rows are quarantined in errors.jsonl with
    # their original fields preserved (so an operator can grep).
    errors_file = output_file.parent / "errors.jsonl"
    assert errors_file.exists(), "errors.jsonl must be created when any row is skipped"
    with errors_file.open(encoding="utf-8") as fh:
        err_records = [json.loads(line) for line in fh if line.strip()]
    assert len(err_records) == 3
    # Each error row has a ``reason`` field stamped by the converter.
    assert all("reason" in rec for rec in err_records)


def test_empty_input_writes_empty_output_without_crash(tmp_path: Path) -> None:
    """Empty input must produce empty output (no missing-parent crash).

    Pins the latent bug fix from the streaming refactor: the previous
    code called ``output_file.touch()`` *before* ``mkdir(parents=...)``,
    so an empty input with a non-existent output dir would
    ``FileNotFoundError`` instead of producing the expected empty file.
    Downstream stages distinguish "empty success" from "missing file"
    via ``Path.exists()``, so the empty file matters.
    """
    input_file = tmp_path / "empty.jsonl"
    output_file = tmp_path / "missing_dir" / "out.jsonl"
    input_file.touch()

    _run_converter(input_file, output_file)

    assert output_file.exists(), "empty input must still produce an output file"
    assert output_file.stat().st_size == 0
    # No errors.jsonl on empty input.
    assert not (output_file.parent / "errors.jsonl").exists()


def test_missing_input_path_exits_nonzero_with_clean_message(tmp_path: Path) -> None:
    """A non-existent input path must fail loudly with a single error line.

    The streaming refactor swapped ``sys.exit(1)`` from inside
    ``_read_input`` for a ``raise FileNotFoundError`` caught in
    ``main()``; this test pins the exit-code-1 + clean-log contract
    so a regression that lets the stack trace bubble up (or returns 0
    on missing input) gets caught.
    """
    cmd = [
        sys.executable,
        "-m",
        "nvflow.recipes.finance.utils.rl.responses_api_converter",
        str(tmp_path / "does_not_exist"),
        str(tmp_path / "out.jsonl"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 1, f"expected exit 1 for missing input, got {result.returncode}"
    # The friendly error line must appear in stderr (logger.error path).
    assert "does not exist" in result.stderr or "does not exist" in result.stdout


def test_missing_input_does_not_leave_zombie_output_file(tmp_path: Path) -> None:
    """Failed runs must not create a 0-byte output file or parent dir.

    Reproduces slurm job 12091275: a wrong config pointed step-3 at a
    non-existent input dir; the previous code called
    ``output_file.parent.mkdir(...)`` and opened the writer (which
    creates the file) *before* ``_iter_input``'s validation raised,
    leaving a 0-byte zombie output behind even though ``main()``
    returned 1.

    The zombie was harmful for two reasons:
      1. ``Path.exists()`` health checks read it as "step succeeded".
      2. Reruns of the failed job would see a pre-existing output and
         could skip the stage as already-done.

    The fix hoists ``_resolve_input_files()`` out of ``_iter_input``
    and calls it from ``convert()`` *before* ``mkdir``/``write_jsonl``,
    so the failure path leaves no artefacts behind.
    """
    output_file = tmp_path / "out_dir" / "final_result.jsonl"
    cmd = [
        sys.executable,
        "-m",
        "nvflow.recipes.finance.utils.rl.responses_api_converter",
        str(tmp_path / "does_not_exist"),
        str(output_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 1, f"expected exit 1 for missing input, got {result.returncode}"
    # Neither the output file nor its parent dir should exist.  An empty
    # parent dir would be slightly less harmful than a 0-byte output
    # file (no false ``Path.exists(output_file)`` positive), but the
    # cleanest contract is "failed run leaves zero artefacts behind".
    assert not output_file.exists(), (
        f"missing input must not leave a zombie output: {output_file} exists "
        f"(size={output_file.stat().st_size})"
    )
    assert not output_file.parent.exists(), (
        f"missing input must not create the output parent dir: {output_file.parent} exists"
    )


def test_empty_input_dir_does_not_leave_zombie_output_file(tmp_path: Path) -> None:
    """Empty input *directory* (vs. empty input *file*) is also a failure.

    Pins the contract that a directory containing zero ``*.jsonl``
    files is treated as a failed run, not an empty success.  This is
    distinct from ``test_empty_input_writes_empty_output_without_crash``
    which exercises the empty-file (zero-record) success path.

    An empty directory typically signals an upstream stage failure
    (e.g. ``apply_prompt_template`` produced nothing because its own
    input was empty), so failing fast prevents downstream stages from
    silently consuming an empty dataset.
    """
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    # Plant a non-jsonl file so the dir is non-empty but our glob misses it.
    (input_dir / "README.txt").write_text("not a jsonl file\n")

    output_file = tmp_path / "out_dir" / "final_result.jsonl"
    cmd = [
        sys.executable,
        "-m",
        "nvflow.recipes.finance.utils.rl.responses_api_converter",
        str(input_dir),
        str(output_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 1
    assert "No .jsonl files found" in result.stderr or "No .jsonl files found" in result.stdout
    assert not output_file.exists(), "empty input dir must not leave a zombie output"
    assert not output_file.parent.exists(), "empty input dir must not create the output parent dir"


def test_streaming_handles_large_record_count(tmp_path: Path) -> None:
    """Output record count must equal input -- pins the streaming refactor.

    The previous list-based implementation buffered all rows before
    writing; the streaming refactor pipes rows directly through.
    A regression that breaks the streaming generator (e.g. an
    early ``return`` after the first file) would silently drop
    later chunks.  This test concatenates 5 chunks of 200 records
    each (1000 total) and pins that ALL records reach the output.
    """
    input_dir = tmp_path / "in"
    output_file = tmp_path / "out.jsonl"

    expected_uuids: set[str] = set()
    for chunk_idx in range(1, 6):
        records = [_make_record(f"c{chunk_idx}-r{i}") for i in range(200)]
        expected_uuids.update(rec["uuid"] for rec in records)
        _write_jsonl(input_dir / f"final_result_chunk{chunk_idx}.jsonl", records)

    _run_converter(input_dir, output_file)

    with output_file.open(encoding="utf-8") as fh:
        out_uuids = {json.loads(line)["uuid"] for line in fh if line.strip()}

    assert out_uuids == expected_uuids, (
        f"streaming dropped records: missing {expected_uuids - out_uuids}, "
        f"extra {out_uuids - expected_uuids}"
    )
