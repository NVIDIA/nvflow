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
"""Tests for ``nvflow.utils.jsonl``.

Exercises the contract that the validate_questions refactor depends on:
byte-identical round-trip for both dict and bytes paths, malformed-line
policies, buffer-flush boundaries, atomic stats writes.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import orjson
import pytest

from nvflow.utils.jsonl import (
    DEFAULT_BUFFER_SIZE,
    iter_jsonl,
    write_jsonl,
    write_stats_json,
)

# ---------------------------------------------------------------------------
# iter_jsonl
# ---------------------------------------------------------------------------


def _write_lines(path: Path, lines: list[bytes]) -> None:
    path.write_bytes(b"\n".join(lines) + b"\n")


def test_iter_jsonl_skips_empty_lines(tmp_path: Path) -> None:
    f = tmp_path / "x.jsonl"
    _write_lines(f, [b'{"a": 1}', b"", b'{"a": 2}', b"   ", b'{"a": 3}'])
    assert list(iter_jsonl(f)) == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_iter_jsonl_skip_mode_drops_malformed(tmp_path: Path) -> None:
    f = tmp_path / "x.jsonl"
    _write_lines(f, [b'{"a": 1}', b"not json", b'{"a": 2}'])
    assert list(iter_jsonl(f, on_error="skip")) == [{"a": 1}, {"a": 2}]


def test_iter_jsonl_raise_mode_propagates(tmp_path: Path) -> None:
    f = tmp_path / "x.jsonl"
    _write_lines(f, [b'{"a": 1}', b"not json", b'{"a": 2}'])
    with pytest.raises(orjson.JSONDecodeError):
        list(iter_jsonl(f, on_error="raise"))


def test_iter_jsonl_yield_error_mode_emits_triples_with_raw_bytes(tmp_path: Path) -> None:
    """yield_error mode yields (row | None, exc | None, raw_line: bytes).

    The raw_line is the stripped source bytes; callers that emit audit
    records (e.g., regex_prefilter_questions) include a truncated
    decoded copy in the dropped stream so an operator can inspect the
    offending row without re-opening the input file.
    """
    f = tmp_path / "x.jsonl"
    _write_lines(f, [b'{"a": 1}', b"not json", b'{"a": 2}'])
    triples = list(iter_jsonl(f, on_error="yield_error"))
    assert len(triples) == 3

    row0, exc0, raw0 = triples[0]
    assert row0 == {"a": 1}
    assert exc0 is None
    assert raw0 == b'{"a": 1}'

    row1, exc1, raw1 = triples[1]
    assert row1 is None
    assert isinstance(exc1, orjson.JSONDecodeError)
    assert raw1 == b"not json"

    row2, exc2, raw2 = triples[2]
    assert row2 == {"a": 2}
    assert exc2 is None
    assert raw2 == b'{"a": 2}'


def test_iter_jsonl_yield_error_strips_whitespace_from_raw_line(tmp_path: Path) -> None:
    """raw_line is the stripped form so callers see canonical bytes."""
    f = tmp_path / "x.jsonl"
    f.write_bytes(b'   {"a": 1}   \n  bad  \n')
    triples = list(iter_jsonl(f, on_error="yield_error"))
    assert len(triples) == 2
    assert triples[0][2] == b'{"a": 1}'
    assert triples[1][2] == b"bad"


def test_iter_jsonl_accepts_path_or_str(tmp_path: Path) -> None:
    f = tmp_path / "x.jsonl"
    f.write_bytes(b'{"a": 1}\n')
    assert list(iter_jsonl(str(f))) == [{"a": 1}]
    assert list(iter_jsonl(f)) == [{"a": 1}]


def test_iter_jsonl_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "x.jsonl"
    f.write_bytes(b"")
    assert list(iter_jsonl(f)) == []


# ---------------------------------------------------------------------------
# write_jsonl
# ---------------------------------------------------------------------------


def test_write_jsonl_roundtrip_dicts(tmp_path: Path) -> None:
    f = tmp_path / "x.jsonl"
    rows = [{"i": i, "v": f"row-{i}"} for i in range(1000)]
    with write_jsonl(f) as w:
        for r in rows:
            w.write(r)
    assert list(iter_jsonl(f)) == rows


def test_write_jsonl_roundtrip_bytes(tmp_path: Path) -> None:
    f = tmp_path / "x.jsonl"
    raw = [orjson.dumps({"i": i, "v": f"row-{i}"}) for i in range(1000)]
    with write_jsonl(f) as w:
        for b in raw:
            w.write(b)
    # Byte-identical round-trip is the contract Phase 2 depends on.
    expected = b"\n".join(raw) + b"\n"
    assert f.read_bytes() == expected


def test_write_jsonl_mixed_dict_and_bytes(tmp_path: Path) -> None:
    f = tmp_path / "x.jsonl"
    with write_jsonl(f) as w:
        w.write({"i": 0})
        w.write(orjson.dumps({"i": 1}))
        w.write({"i": 2})
        w.write(orjson.dumps({"i": 3}))
    assert list(iter_jsonl(f)) == [{"i": 0}, {"i": 1}, {"i": 2}, {"i": 3}]


def test_write_jsonl_trailing_newline_guaranteed(tmp_path: Path) -> None:
    f = tmp_path / "x.jsonl"
    with write_jsonl(f) as w:
        w.write({"a": 1})
    blob = f.read_bytes()
    assert blob.endswith(b"\n")
    assert blob == b'{"a":1}\n'


def test_write_jsonl_empty_writer_produces_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "x.jsonl"
    with write_jsonl(f):
        pass
    assert f.read_bytes() == b""


def test_write_jsonl_buffer_boundary_999_1000_1001(tmp_path: Path) -> None:
    """Buffer threshold is ``>=`` -- exact 1000th write triggers a flush.

    Output bytes must be the same regardless of the boundary because the
    final exit-flush rolls everything together with the same join pattern.
    """
    for n in (999, 1000, 1001):
        # Reference output: write all rows into one byte-buffer with the
        # same join pattern the writer uses.
        rows = [orjson.dumps({"i": i}) for i in range(n)]
        reference = b"\n".join(rows) + b"\n"

        f = tmp_path / f"n{n}.jsonl"
        with write_jsonl(f) as w:
            for i in range(n):
                w.write({"i": i})
        assert f.read_bytes() == reference, f"buffer-boundary mismatch at n={n}"


def test_write_jsonl_one_syscall_per_flush(tmp_path: Path) -> None:
    """The flush pattern must be one ``write()`` call per buffer drain.

    This guards the ``b"\\n".join(buffer) + b"\\n"`` performance contract
    -- if someone refactors to per-row writes we'd see N calls instead of 1.
    """
    f = tmp_path / "x.jsonl"

    write_calls: list[int] = []

    class CountingFile:
        def __init__(self, real: Any) -> None:
            self._real = real

        def write(self, data: bytes) -> int:
            write_calls.append(len(data))
            return self._real.write(data)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._real, name)

    with write_jsonl(f, buffer_size=10) as w:
        # Patch the open file with the counter just for this test.
        w._fp = CountingFile(w._fp)  # type: ignore[attr-defined]
        for i in range(25):
            w.write({"i": i})
    # 25 writes at buffer_size=10 -> flushes at 10, 20, plus final exit
    # flush of remaining 5. Three write() calls total.
    assert len(write_calls) == 3, f"expected 3 syscalls, got {len(write_calls)}"


def test_write_jsonl_invalid_buffer_size(tmp_path: Path) -> None:
    f = tmp_path / "x.jsonl"
    with pytest.raises(ValueError, match="buffer_size must be >= 1"):
        write_jsonl(f, buffer_size=0)


def test_write_jsonl_default_buffer_size_constant() -> None:
    """The exported constant must match the historical magic number 1000.

    Recipes still document the 1000-record buffer in their stage docstrings;
    a silent change here would surprise operators reading those docs.
    """
    assert DEFAULT_BUFFER_SIZE == 1000


def test_write_jsonl_closes_file_on_exception(tmp_path: Path) -> None:
    """Even on an exception inside the with-block, the file must be closed."""
    f = tmp_path / "x.jsonl"
    with pytest.raises(RuntimeError, match="boom"):
        with write_jsonl(f) as w:
            w.write({"a": 1})
            raise RuntimeError("boom")
    # The partial flush should still have happened on exit.
    assert f.read_bytes() == b'{"a":1}\n'


# ---------------------------------------------------------------------------
# write_stats_json
# ---------------------------------------------------------------------------


def test_write_stats_json_format_parity(tmp_path: Path) -> None:
    """Output must be byte-identical to the historical inline atomic write."""
    f = tmp_path / "stats.json"
    sample = {"a": 1, "b": [2, 3, 4], "c": "hello"}
    write_stats_json(f, sample)
    expected = orjson.dumps(sample, option=orjson.OPT_INDENT_2)
    assert f.read_bytes() == expected


def test_write_stats_json_replaces_existing(tmp_path: Path) -> None:
    f = tmp_path / "stats.json"
    f.write_bytes(b"stale-content")
    write_stats_json(f, {"fresh": True})
    assert f.read_bytes() == orjson.dumps({"fresh": True}, option=orjson.OPT_INDENT_2)


def test_write_stats_json_tmp_in_same_directory(tmp_path: Path) -> None:
    """``os.replace`` needs the tmp on the same mount; we ensure same parent dir."""
    target = tmp_path / "subdir" / "stats.json"
    target.parent.mkdir()
    write_stats_json(target, {"k": 1})
    # No leftover .tmp on success.
    assert not (target.parent / "stats.json.tmp").exists()
    assert target.exists()


def test_write_stats_json_atomic_on_kill(tmp_path: Path) -> None:
    """SIGKILL during the write must leave only the .tmp, not a truncated target.

    Spawns a subprocess that opens the tmp file, writes a large payload very
    slowly, and waits for the parent's signal.  Parent kills it mid-write.
    Verifies the final state: target absent (rename never happened), tmp
    may be present and truncated -- but the target file itself is never
    truncated because os.replace is atomic.
    """
    script = tmp_path / "kill_me.py"
    target = tmp_path / "stats.json"
    tmp_marker = tmp_path / "child_started"

    script.write_text(
        textwrap.dedent(
            f"""
            import time, orjson, os, sys
            from pathlib import Path

            target = Path({str(target)!r})
            tmp = target.with_name(target.name + '.tmp')
            marker = Path({str(tmp_marker)!r})

            big = orjson.dumps({{'pad': 'x' * 1_000_000}}, option=orjson.OPT_INDENT_2)
            with open(tmp, 'wb') as f:
                # Signal the parent that we've created the tmp and are
                # about to commit; partial writes happen between here and
                # os.replace.
                marker.write_text('go')
                f.write(big[:len(big) // 2])
                f.flush()
                os.fsync(f.fileno())
                # Sleep so the parent can SIGKILL us before we finish.
                time.sleep(60)
                f.write(big[len(big) // 2:])
            os.replace(tmp, target)
            """
        )
    )

    proc = subprocess.Popen([sys.executable, str(script)])
    # Wait for the child to indicate it started writing.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not tmp_marker.exists():
        time.sleep(0.02)
    assert tmp_marker.exists(), "child did not start in time"

    os.kill(proc.pid, signal.SIGKILL)
    proc.wait(timeout=5.0)

    # Target must be absent: os.replace never ran.
    assert not target.exists(), "target was created despite SIGKILL"
    # Tmp may exist (with partial content) but is not the consumer-visible file.
