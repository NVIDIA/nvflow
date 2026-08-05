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
"""Tests for ``apply_validate_filter`` post-S1.

Pins the new contracts:
- ``raw_sdg_path`` is required (TypeError if omitted).
- ``--raw_sdg_source`` CLI flag is required (argparse exits non-zero).
- A VALID row whose ``problem`` is missing from the SDG file raises
  :class:`MissingSdgRecordError` instead of silently emitting LLM bytes.
- An INVALID/missing-tag row whose ``problem`` is missing from the SDG
  file is fine -- those rows go to the dropped stream and don't need
  SDG-original bytes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import orjson
import pytest

from nvflow.recipes.finance.utils.rl.apply_validate_filter import (
    MissingSdgRecordError,
    apply_validate_filter,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_bytes(b"\n".join(orjson.dumps(r) for r in rows) + b"\n")


def _make_sdg_and_parsed(
    tmp_path: Path,
    *,
    sdg_problems: list[str],
    parsed: list[dict],
) -> tuple[Path, Path]:
    sdg = tmp_path / "sdg.jsonl"
    parsed_file = tmp_path / "parsed.jsonl"
    _write_jsonl(
        sdg,
        [
            {
                "problem": p,
                "company_name": f"Co{p}",
                "answer": f"A{p}",
                "reasoning_content": f"R{p}",
            }
            for p in sdg_problems
        ],
    )
    _write_jsonl(parsed_file, parsed)
    return sdg, parsed_file


# ---------------------------------------------------------------------------
# Required raw_sdg_path
# ---------------------------------------------------------------------------


def test_function_call_omitting_raw_sdg_path_raises_typeerror(tmp_path: Path) -> None:
    """Calling the function without raw_sdg_path is a programming bug."""
    parsed_file = tmp_path / "parsed.jsonl"
    parsed_file.write_bytes(b"")
    with pytest.raises(TypeError, match="raw_sdg_path"):
        apply_validate_filter(  # type: ignore[call-arg]
            input_file=str(parsed_file),
            output_kept=str(tmp_path / "kept.jsonl"),
            output_dropped=str(tmp_path / "dropped.jsonl"),
            stats_file=str(tmp_path / "stats.json"),
        )


def test_cli_omitting_raw_sdg_source_exits_nonzero(tmp_path: Path) -> None:
    """The CLI must require --raw_sdg_source so operators can't accidentally
    fall back to a removed Mode B path.
    """
    parsed_file = tmp_path / "parsed.jsonl"
    parsed_file.write_bytes(b"")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "nvflow.recipes.finance.utils.rl.apply_validate_filter",
            "--input_file",
            str(parsed_file),
            "--output_kept",
            str(tmp_path / "kept.jsonl"),
            "--output_dropped",
            str(tmp_path / "dropped.jsonl"),
            "--stats_file",
            str(tmp_path / "stats.json"),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert proc.returncode != 0
    assert "--raw_sdg_source" in proc.stderr


# ---------------------------------------------------------------------------
# Missing-SDG-record contract
# ---------------------------------------------------------------------------


def test_valid_row_missing_from_sdg_raises(tmp_path: Path) -> None:
    """A VALID row with no SDG counterpart is a Phase 1/2 mismatch -- raise."""
    sdg, parsed = _make_sdg_and_parsed(
        tmp_path,
        sdg_problems=["p_in_sdg"],
        parsed=[
            {"problem": "p_NOT_in_sdg", "validate_tag": "VALID", "generation": "x"},
        ],
    )
    with pytest.raises(MissingSdgRecordError, match="p_NOT_in_sdg"):
        apply_validate_filter(
            input_file=str(parsed),
            output_kept=str(tmp_path / "kept.jsonl"),
            output_dropped=str(tmp_path / "dropped.jsonl"),
            stats_file=str(tmp_path / "stats.json"),
            raw_sdg_path=str(sdg),
        )


def test_invalid_row_missing_from_sdg_is_fine(tmp_path: Path) -> None:
    """Non-VALID rows go to the dropped stream regardless of SDG presence."""
    sdg, parsed = _make_sdg_and_parsed(
        tmp_path,
        sdg_problems=["p_in_sdg"],
        parsed=[
            {"problem": "p_NOT_in_sdg", "validate_tag": "INVALID", "generation": "x"},
            {"problem": "p_NOT_in_sdg", "validate_tag": None, "generation": "y"},
        ],
    )
    apply_validate_filter(
        input_file=str(parsed),
        output_kept=str(tmp_path / "kept.jsonl"),
        output_dropped=str(tmp_path / "dropped.jsonl"),
        stats_file=str(tmp_path / "stats.json"),
        raw_sdg_path=str(sdg),
    )
    stats = orjson.loads((tmp_path / "stats.json").read_bytes())
    assert stats["num_total"] == 2
    assert stats["num_kept"] == 0
    assert stats["num_dropped"] == 2
    # No VALID rows -> no missing-record exposure.


# ---------------------------------------------------------------------------
# Happy path: kept rows are byte-identical to SDG
# ---------------------------------------------------------------------------


def test_kept_rows_are_sdg_bytes_verbatim(tmp_path: Path) -> None:
    sdg, parsed = _make_sdg_and_parsed(
        tmp_path,
        sdg_problems=["p1", "p2", "p3"],
        parsed=[
            {"problem": "p1", "validate_tag": "VALID", "generation": "x"},
            {"problem": "p2", "validate_tag": "INVALID", "generation": "y"},
            {"problem": "p3", "validate_tag": "VALID", "generation": "z"},
        ],
    )
    kept_file = tmp_path / "kept.jsonl"
    apply_validate_filter(
        input_file=str(parsed),
        output_kept=str(kept_file),
        output_dropped=str(tmp_path / "dropped.jsonl"),
        stats_file=str(tmp_path / "stats.json"),
        raw_sdg_path=str(sdg),
    )

    sdg_bytes_by_problem: dict[str, bytes] = {}
    for line in sdg.read_bytes().splitlines():
        if line.strip():
            sdg_bytes_by_problem[orjson.loads(line)["problem"]] = line.strip()

    kept_lines = [line for line in kept_file.read_bytes().splitlines() if line.strip()]
    assert len(kept_lines) == 2
    assert kept_lines[0] == sdg_bytes_by_problem["p1"]
    assert kept_lines[1] == sdg_bytes_by_problem["p3"]
