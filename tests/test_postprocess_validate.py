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
"""Tests for the validate_questions Phase 2 in-process orchestrator.

Covers integration-equivalence (end artefacts match running parse + apply
separately), failure-mode short-circuit (parse raises -> apply not run),
and sentinel log lines for grep-able post-mortems.
"""

from __future__ import annotations

import re
from pathlib import Path

import orjson
import pytest

from nvflow.recipes.finance.utils.rl.apply_validate_filter import apply_validate_filter
from nvflow.recipes.finance.utils.rl.parse_validate_responses import parse_validate_responses
from nvflow.recipes.finance.utils.rl.postprocess_validate import postprocess_validate

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_sdg_and_llm_input(tmp_path: Path) -> tuple[Path, Path]:
    """Build a small SDG file and a matching LLM-output file.

    Layout: 5 VALID + 4 INVALID + 1 missing-tag (parse failure) = 10 rows.
    Records share their ``problem`` keys with the SDG file so the
    Mode-A SDG-restore path is exercised.
    """
    sdg = tmp_path / "sdg.jsonl"
    llm = tmp_path / "output.jsonl"

    schedule = ["V"] * 5 + ["I"] * 4 + ["F"]
    sdg_lines: list[bytes] = []
    llm_lines: list[bytes] = []
    for idx, code in enumerate(schedule):
        sdg_record = {
            "problem": f"problem_{idx}",
            "company_name": f"Company{idx}",
            "answer": f"answer_{idx}",
            "reasoning_content": f"sdg_reasoning_{idx}",
            "year": 2024,
        }
        if code == "V":
            generation = f"Reason: identifier-and-context check pass (idx={idx}).\nAnswer: VALID"
        elif code == "I":
            generation = f"Reason: vague reference, no identifier (idx={idx}).\nAnswer: INVALID"
        else:
            generation = "rambled, no answer tag"

        llm_record = {
            **sdg_record,
            "generation": generation,
            "finish_reason": "stop",
            "num_generated_tokens": 16,
            # Emulate nemo-skills overwriting reasoning_content; apply_validate_filter
            # restores SDG-original via the raw_sdg_source path.
            "reasoning_content": "(LLM-mutated)",
        }
        sdg_lines.append(orjson.dumps(sdg_record))
        llm_lines.append(orjson.dumps(llm_record))

    sdg.write_bytes(b"\n".join(sdg_lines) + b"\n")
    llm.write_bytes(b"\n".join(llm_lines) + b"\n")
    return sdg, llm


# ---------------------------------------------------------------------------
# Integration: orchestrator vs running scripts separately
# ---------------------------------------------------------------------------


def test_orchestrator_produces_equivalent_artefacts(tmp_path: Path) -> None:
    """Running the orchestrator must produce the same artefacts as running
    parse_validate_responses then apply_validate_filter back-to-back.
    """
    sdg, llm = _make_sdg_and_llm_input(tmp_path)

    # Path 1: orchestrator
    orchestrator_dir = tmp_path / "orchestrator"
    orchestrator_dir.mkdir()
    postprocess_validate(
        llm_output=str(llm),
        parsed_jsonl=str(orchestrator_dir / "parsed.jsonl"),
        final_kept=str(orchestrator_dir / "final_result.jsonl"),
        dropped=str(orchestrator_dir / "dropped.jsonl"),
        stats=str(orchestrator_dir / "stats.json"),
        raw_sdg_source=str(tmp_path),
        raw_sdg_filename="sdg.jsonl",
    )

    # Path 2: separate scripts
    separate_dir = tmp_path / "separate"
    separate_dir.mkdir()
    parse_validate_responses(str(llm), str(separate_dir / "parsed.jsonl"))
    apply_validate_filter(
        input_file=str(separate_dir / "parsed.jsonl"),
        output_kept=str(separate_dir / "final_result.jsonl"),
        output_dropped=str(separate_dir / "dropped.jsonl"),
        stats_file=str(separate_dir / "stats.json"),
        raw_sdg_path=str(sdg),
    )

    for name in ("parsed.jsonl", "final_result.jsonl", "dropped.jsonl"):
        assert (orchestrator_dir / name).read_bytes() == (separate_dir / name).read_bytes(), (
            f"orchestrator vs separate artefact differs: {name}"
        )

    # Stats: ignore path fields (they differ because the call paths differ).
    orch_stats = orjson.loads((orchestrator_dir / "stats.json").read_bytes())
    sep_stats = orjson.loads((separate_dir / "stats.json").read_bytes())
    for path_field in ("input_file", "output_kept", "output_dropped", "raw_sdg_path"):
        orch_stats.pop(path_field, None)
        sep_stats.pop(path_field, None)
    assert orch_stats == sep_stats


def test_orchestrator_kept_count_matches_schedule(tmp_path: Path) -> None:
    """5 VALID + 1 parse-failure (defaulted VALID) = 6 kept; 4 INVALID dropped."""
    sdg, llm = _make_sdg_and_llm_input(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    postprocess_validate(
        llm_output=str(llm),
        parsed_jsonl=str(out / "parsed.jsonl"),
        final_kept=str(out / "final.jsonl"),
        dropped=str(out / "dropped.jsonl"),
        stats=str(out / "stats.json"),
        raw_sdg_source=str(tmp_path),
        raw_sdg_filename="sdg.jsonl",
    )
    stats = orjson.loads((out / "stats.json").read_bytes())
    assert stats["num_total"] == 10
    assert stats["num_kept"] == 6  # 5 VALID + 1 parse-failure recall-bias
    assert stats["num_dropped"] == 4
    assert stats["num_missing_tag"] == 0  # parse-failure -> defaulted VALID, not missing
    assert stats["num_parse_failed_kept_as_valid"] == 1


def test_orchestrator_kept_records_are_byte_identical_to_sdg(tmp_path: Path) -> None:
    """Each kept record must be the EXACT bytes from the SDG file -- this is
    the validate_questions pure-row-filter contract.
    """
    sdg, llm = _make_sdg_and_llm_input(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    postprocess_validate(
        llm_output=str(llm),
        parsed_jsonl=str(out / "parsed.jsonl"),
        final_kept=str(out / "final.jsonl"),
        dropped=str(out / "dropped.jsonl"),
        stats=str(out / "stats.json"),
        raw_sdg_source=str(tmp_path),
        raw_sdg_filename="sdg.jsonl",
    )
    sdg_by_problem: dict[str, bytes] = {}
    for line in sdg.read_bytes().splitlines():
        if not line.strip():
            continue
        obj = orjson.loads(line)
        sdg_by_problem[obj["problem"]] = line.strip()

    for line in (out / "final.jsonl").read_bytes().splitlines():
        if not line.strip():
            continue
        obj = orjson.loads(line)
        # The kept stream emits original SDG bytes; round-tripping through
        # orjson then compare to the source line is the byte-identity check.
        assert line.strip() == sdg_by_problem[obj["problem"]]


# ---------------------------------------------------------------------------
# Failure-mode: parse raises -> apply NOT run
# ---------------------------------------------------------------------------


def test_orchestrator_aborts_when_parse_raises(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    # llm_output points at a non-existent file -> parse_validate_responses
    # raises FileNotFoundError before apply_validate_filter can run.
    nonexistent = tmp_path / "missing.jsonl"
    final = out / "final.jsonl"
    with pytest.raises(FileNotFoundError):
        postprocess_validate(
            llm_output=str(nonexistent),
            parsed_jsonl=str(out / "parsed.jsonl"),
            final_kept=str(final),
            dropped=str(out / "dropped.jsonl"),
            stats=str(out / "stats.json"),
            raw_sdg_source=str(tmp_path),
            raw_sdg_filename="sdg.jsonl",
        )
    # Apply must NOT have run -- final_kept must be absent.
    assert not final.exists(), "apply phase ran despite parse failure"
    assert not (out / "stats.json").exists(), "stats written despite parse failure"


# ---------------------------------------------------------------------------
# Sentinel log lines
# ---------------------------------------------------------------------------


def test_orchestrator_emits_sentinel_logs(tmp_path: Path) -> None:
    """Slurm log readers grep for ``PHASE: parse`` / ``PHASE: apply`` to find
    phase boundaries.  ``setup_logger`` configures ``propagate=False`` and
    binds its handler to ``sys.stdout`` at import time, so neither pytest's
    caplog nor capfd see the messages reliably.  Attach a memory handler
    directly to the orchestrator's logger to capture them.
    """
    import logging

    from nvflow.recipes.finance.utils.rl import postprocess_validate as orchestrator_module

    captured: list[str] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record.getMessage())

    handler = _ListHandler(level=logging.INFO)
    orchestrator_module.logger.addHandler(handler)
    try:
        sdg, llm = _make_sdg_and_llm_input(tmp_path)
        out = tmp_path / "out"
        out.mkdir()
        postprocess_validate(
            llm_output=str(llm),
            parsed_jsonl=str(out / "parsed.jsonl"),
            final_kept=str(out / "final.jsonl"),
            dropped=str(out / "dropped.jsonl"),
            stats=str(out / "stats.json"),
            raw_sdg_source=str(tmp_path),
            raw_sdg_filename="sdg.jsonl",
        )
    finally:
        orchestrator_module.logger.removeHandler(handler)

    text = "\n".join(captured)
    assert re.search(r"PHASE: parse", text), text
    assert re.search(r"PHASE: apply", text), text
    assert re.search(r"PHASE: done", text), text
    # Sentinel ordering matters too: parse before apply before done.
    assert text.index("PHASE: parse") < text.index("PHASE: apply")
    assert text.index("PHASE: apply") < text.index("PHASE: done")
