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
"""Shared JSONL read / write helpers for the recipes/finance utilities.

The same buffered JSONL pattern (orjson decode/encode, ``WRITE_BUFFER_SIZE``,
manual flush, error handling) appeared verbatim in eight files under
``recipes/finance/``.  This module is the single source of truth so a fix
or change applies everywhere.

Three primitives:

- :func:`iter_jsonl` -- read a JSONL file lazily, with a choice of
  malformed-line policy (``skip`` / ``raise`` / ``yield_error``).  Empty
  lines are always silently skipped.
- :class:`write_jsonl` -- buffered writer context manager.  ``write()``
  accepts either a ``dict`` (orjson-encoded) or raw ``bytes`` (written
  verbatim).  The bytes pass-through is REQUIRED for the
  validate_questions pure-row-filter contract: ``apply_validate_filter``
  passes raw SDG bytes through to preserve key ordering and float
  formatting from the source file.
- :func:`write_stats_json` -- atomic write of a stats JSON via
  ``tmp + os.replace()``.  Crash-safe: a partial write leaves only
  ``{path}.tmp``, never a truncated ``{path}``.

Performance constraints honoured to keep adoption byte-identical to the
old in-line code:

- Buffer flush uses ``b"\\n".join(buffer) + b"\\n"`` (one syscall per flush).
- Buffer threshold is ``>=`` (matches existing behaviour).
- Last flush appends a trailing newline (file always ends in ``\\n``).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, overload

import orjson

DEFAULT_BUFFER_SIZE = 1000
"""Default number of records to buffer before flushing to disk.

Matches the historical ``WRITE_BUFFER_SIZE`` constant used in the
recipes/finance utilities.  Kept centralised so future tuning needs to
happen in only one place.
"""


@overload
def iter_jsonl(
    path: str | Path,
    *,
    on_error: Literal["skip", "raise"] = ...,
) -> Iterator[dict[str, Any]]: ...


@overload
def iter_jsonl(
    path: str | Path,
    *,
    on_error: Literal["yield_error"],
) -> Iterator[tuple[dict[str, Any] | None, orjson.JSONDecodeError | None, bytes]]: ...


def iter_jsonl(
    path: str | Path,
    *,
    on_error: Literal["skip", "raise", "yield_error"] = "skip",
) -> Iterator[Any]:
    """Iterate JSONL records lazily.

    Empty lines are silently skipped in every mode (universal behaviour
    today).  Trailing whitespace is stripped before parsing.

    Args:
        path: Path to the JSONL file (str or :class:`pathlib.Path`).
        on_error: How to handle malformed JSON lines.

            - ``"skip"`` (default): silently skip the line.
            - ``"raise"``: raise the underlying :class:`orjson.JSONDecodeError`.
            - ``"yield_error"``: yield ``(None, exc, raw_line)`` for
              malformed lines and ``(row, None, raw_line)`` for valid
              ones.  ``raw_line`` is the stripped source bytes -- callers
              that emit audit records (e.g.,
              ``regex_prefilter_questions``) include a truncated decoded
              copy in the dropped stream so an operator can inspect the
              offending source line without re-opening the input file.

    Yields:
        For ``skip`` / ``raise``: ``dict`` per valid line.
        For ``yield_error``:
            ``tuple[dict | None, orjson.JSONDecodeError | None, bytes]``.
    """
    with open(path, "rb") as reader:
        for raw in reader:
            line = raw.strip()
            if not line:
                continue
            try:
                row = orjson.loads(line)
            except orjson.JSONDecodeError as exc:
                if on_error == "skip":
                    continue
                if on_error == "raise":
                    raise
                yield (None, exc, line)
                continue
            if on_error == "yield_error":
                yield (row, None, line)
            else:
                yield row


class write_jsonl:  # noqa: N801 -- callable-style API: pairs with iter_jsonl(path) function
    """Buffered JSONL writer context manager.

    Named in lowercase intentionally so the call site reads as a
    function-style helper paired with :func:`iter_jsonl`::

        with write_jsonl(out_path) as out:
            for row in iter_jsonl(in_path):
                out.write(transform(row))

    The lowercase naming violates :pep:`8` ``N801`` (CapWords for class
    names); the rule is silenced via ``noqa`` because the readability
    win at every call site outweighs the convention deviation, and the
    pair ``iter_jsonl`` / ``write_jsonl`` is the established symmetry.

    Accepts ``dict`` (orjson-encoded with no options) or raw ``bytes``
    (written verbatim, used for byte-preserving pass-through).  Bytes
    must NOT contain a trailing newline -- the writer adds the line
    terminator on flush, matching the historical
    ``b"\\n".join(buffer) + b"\\n"`` pattern.

    Buffer flushes happen when ``len(buffer) >= buffer_size`` (matches
    historical ``>=`` semantics) and once more on context exit if the
    buffer is non-empty.  The final flush appends a trailing newline so
    the file ALWAYS ends in ``\\n`` -- matches historical behaviour and
    ensures downstream tools that split on newlines see the last record.
    """

    def __init__(self, path: str | Path, *, buffer_size: int = DEFAULT_BUFFER_SIZE) -> None:
        if buffer_size < 1:
            raise ValueError(f"buffer_size must be >= 1 (got {buffer_size})")
        self._path = path
        self._buffer_size = buffer_size
        self._buffer: list[bytes] = []
        self._fp: Any = None  # opened in __enter__

    def __enter__(self) -> write_jsonl:
        # Open lazily on enter so the user can construct the writer
        # outside a try/except without leaking file handles.
        self._fp = open(self._path, "wb")
        return self

    def write(self, row: dict[str, Any] | bytes) -> None:
        """Append a record to the write buffer.

        ``dict`` rows are encoded via :func:`orjson.dumps` with no options
        (no indenting, no key sort -- matches historical behaviour).
        ``bytes`` rows are appended verbatim; they MUST be a single JSON
        line WITHOUT a trailing newline (the writer adds line breaks on
        flush via the join pattern).
        """
        if isinstance(row, bytes):
            encoded = row
        else:
            encoded = orjson.dumps(row)
        self._buffer.append(encoded)
        if len(self._buffer) >= self._buffer_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        # Single syscall per flush -- matches historical performance.
        self._fp.write(b"\n".join(self._buffer) + b"\n")
        self._buffer.clear()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            self._flush()
        finally:
            self._fp.close()
            self._fp = None


def write_stats_json(path: str | Path, fields: dict[str, Any]) -> None:
    """Atomically write a stats JSON to ``path``.

    Encodes ``fields`` with :data:`orjson.OPT_INDENT_2` (matches the
    historical pretty-printed stats files), writes to ``{path}.tmp``,
    then renames to ``{path}`` via :func:`os.replace`.  The tmp file
    lives in the same directory as the target, so the rename is atomic
    on POSIX (same-mount requirement satisfied).

    Crash semantics: a process killed mid-write leaves ``{path}.tmp``
    behind but never a truncated ``{path}``.  Downstream tools that
    cache ``{path}.exists()`` as "prior run completed" stay correct.
    """
    target = Path(path)
    tmp = target.with_name(target.name + ".tmp")
    encoded = orjson.dumps(fields, option=orjson.OPT_INDENT_2)
    with open(tmp, "wb") as f:
        f.write(encoded)
    os.replace(tmp, target)
