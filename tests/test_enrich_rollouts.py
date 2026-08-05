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
"""Tests for nvflow.recipes.finance.utils.rl.enrich_rollouts.

Pins the atomicity contract introduced as F3 of the collect_rollouts
audit: both the rollouts-file rewrite and the input-file UUID write-back
go through ``tmp + os.replace`` so a SIGKILL mid-write cannot leave a
truncated file behind.

The pre-F3 code did ``with open(path, "w"): ...`` for both writes;
that truncates ``path`` on open, so any kill before completion meant
loss of the original rollouts (or input).  Re-running merge would
either fail (chunk data already deleted) or silently produce wrong
results.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nvflow.recipes.finance.utils.rl.enrich_rollouts import (
    _atomic_write_jsonl,
    _ensure_uuids,
    enrich,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# _atomic_write_jsonl
# ---------------------------------------------------------------------------


def test_atomic_write_publishes_via_replace(tmp_path: Path) -> None:
    """Successful write leaves no ``.tmp`` file behind.  ``os.replace``
    runs as the final step, so the directory should contain only the
    target.
    """
    target = tmp_path / "out.jsonl"
    _atomic_write_jsonl(str(target), [{"a": 1}, {"b": 2}])

    assert target.exists()
    assert not (tmp_path / "out.jsonl.tmp").exists(), "tmp leaked after success"
    assert _read_jsonl(target) == [{"a": 1}, {"b": 2}]


def test_atomic_write_overwrites_existing_target(tmp_path: Path) -> None:
    """Replaces an existing target atomically.  Pin this contract so a
    refactor that switches to ``open("w")`` (non-atomic truncate-then-
    write) would surface as a missing-overwrite test failure.
    """
    target = tmp_path / "out.jsonl"
    target.write_text('{"old": true}\n')

    _atomic_write_jsonl(str(target), [{"new": True}])
    assert _read_jsonl(target) == [{"new": True}]


def test_atomic_write_keeps_target_intact_when_open_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If creating the temp file fails, the target must be untouched -- the
    whole point of the tmp+replace pattern.

    The failure is simulated by monkeypatching the module's ``open`` to raise,
    which exercises the atomic-write logic regardless of OS/user.  (A chmod-based
    simulation silently passes under root -- which ignores file permission bits --
    so it is not portable to root CI environments.)
    """
    target = tmp_path / "out.jsonl"
    target.write_text('{"original": "intact"}\n')
    original_text = target.read_text()

    def _boom(*args: object, **kwargs: object) -> None:
        raise OSError("simulated open failure")

    # Shadow ``open`` only inside the module under test; the test's own
    # read_text() below uses a different path and is unaffected.
    monkeypatch.setattr(
        "nvflow.recipes.finance.utils.rl.enrich_rollouts.open", _boom, raising=False
    )
    with pytest.raises(OSError):
        _atomic_write_jsonl(str(target), [{"new": True}])
    # Target survived because the temp open failed before we touched it.
    assert target.read_text() == original_text


# ---------------------------------------------------------------------------
# _ensure_uuids -- input file write-back is atomic
# ---------------------------------------------------------------------------


def test_ensure_uuids_no_writeback_when_all_have_uuids(tmp_path: Path) -> None:
    """Fast-path: every input already has a uuid → no file write at all
    (atomic or otherwise).  Verifies we don't gratuitously rewrite a
    file that doesn't need updating.
    """
    input_file = tmp_path / "input.jsonl"
    rows = [{"uuid": "abc", "expected_answer": "A"}, {"uuid": "def", "expected_answer": "B"}]
    _write_jsonl(input_file, rows)
    mtime_before = input_file.stat().st_mtime_ns

    generated = _ensure_uuids(rows, input_file=str(input_file))
    assert generated == 0
    assert input_file.stat().st_mtime_ns == mtime_before, (
        "input file rewritten despite no missing uuids"
    )


def test_ensure_uuids_writeback_is_atomic(tmp_path: Path) -> None:
    """Slow-path: at least one input lacks a uuid → input file gets
    rewritten via ``_atomic_write_jsonl``.  No tmp leak after success.
    """
    input_file = tmp_path / "input.jsonl"
    rows = [
        {"expected_answer": "A", "problem": "q1"},  # missing uuid
        {"uuid": "preexisting", "expected_answer": "B", "problem": "q2"},
    ]
    _write_jsonl(input_file, rows)

    generated = _ensure_uuids(rows, input_file=str(input_file))
    assert generated == 1
    assert not (tmp_path / "input.jsonl.tmp").exists(), "tmp leaked"
    rewritten = _read_jsonl(input_file)
    assert "uuid" in rewritten[0]
    assert rewritten[1]["uuid"] == "preexisting"


# ---------------------------------------------------------------------------
# enrich -- rollouts file rewrite is atomic
# ---------------------------------------------------------------------------


def test_enrich_publishes_atomically(tmp_path: Path) -> None:
    """End-to-end: enrich() reads ``rollouts_file``, computes enriched
    rows, and republishes via tmp+replace.  Pin the no-tmp-leak
    contract so a refactor cannot regress to ``open(rollouts_file, "w")``.
    """
    input_file = tmp_path / "input.jsonl"
    rollouts_file = tmp_path / "rollouts.jsonl"

    inputs = [
        {
            "uuid": "u1",
            "expected_answer": "Paris",
            "problem": "Capital of France?",
            "responses_create_params": {"input": [{"content": "Capital of France?"}]},
            "extra_field": "kept",
        },
    ]
    rollouts = [
        {
            "expected_answer": "Paris",
            "responses_create_params": {"input": [{"content": "Capital of France?"}]},
            "response": "Paris.",
            "reward": 1.0,
        },
    ]
    _write_jsonl(input_file, inputs)
    _write_jsonl(rollouts_file, rollouts)

    enrich(str(input_file), str(rollouts_file))

    assert not (tmp_path / "rollouts.jsonl.tmp").exists(), "rollouts tmp leaked"
    enriched = _read_jsonl(rollouts_file)
    assert len(enriched) == 1
    # Output fields preserved
    assert enriched[0]["response"] == "Paris."
    assert enriched[0]["reward"] == 1.0
    # Input fields restored
    assert enriched[0]["extra_field"] == "kept"
    # uuid carried over from input
    assert enriched[0]["uuid"] == "u1"


def test_enrich_preserves_unmatched_rollouts(tmp_path: Path) -> None:
    """An output row without a matching input must still appear in the
    enriched file (with no extra fields restored).  Without the atomic
    publish, this used to be sensitive to write order; now the entire
    enriched list is computed in memory before any write.
    """
    input_file = tmp_path / "input.jsonl"
    rollouts_file = tmp_path / "rollouts.jsonl"

    inputs = [
        {
            "uuid": "u1",
            "expected_answer": "Paris",
            "problem": "q",
            "responses_create_params": {"input": [{"content": "q"}]},
        },
    ]
    rollouts = [
        {
            "expected_answer": "Paris",
            "responses_create_params": {"input": [{"content": "q"}]},
            "response": "ok",
        },
        {
            "expected_answer": "Berlin",  # no matching input
            "responses_create_params": {"input": [{"content": "other"}]},
            "response": "no-match",
        },
    ]
    _write_jsonl(input_file, inputs)
    _write_jsonl(rollouts_file, rollouts)

    enrich(str(input_file), str(rollouts_file))

    enriched = _read_jsonl(rollouts_file)
    assert len(enriched) == 2
    assert enriched[0]["response"] == "ok"
    assert enriched[1]["response"] == "no-match"
    assert "uuid" not in enriched[1]


def test_enrich_no_rollouts_is_noop(tmp_path: Path) -> None:
    """Empty rollouts file → early return with a warning, no rewrite.

    Important: the original code also early-returned, but pre-F3 it
    had already opened the rollouts file in mode ``"w"`` -- which is
    not the case here (we early-return BEFORE opening).  Pin that the
    file is not touched.
    """
    input_file = tmp_path / "input.jsonl"
    rollouts_file = tmp_path / "rollouts.jsonl"
    _write_jsonl(input_file, [{"uuid": "u1", "expected_answer": "A", "problem": "q"}])
    rollouts_file.write_text("")

    enrich(str(input_file), str(rollouts_file))
    # File still exists and is still empty -- no .tmp leaked.
    assert rollouts_file.read_text() == ""
    assert not (tmp_path / "rollouts.jsonl.tmp").exists()
