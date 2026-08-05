#!/usr/bin/env python3
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
"""Lossless conversion of Q&A data to NeMo-Gym Responses API format.

Expected input (from apply_prompt_template)::

    {"prompt": "...", "problem": "...", "expected_answer": "...", "generation": "...",
     "uuid": "...", ...}

Output adds ``responses_create_params``, ``question``, and ``uuid`` while preserving all
original input fields.  The ``prompt`` field is used as the model input, ``problem`` is
preserved as the raw question in ``question``, and ``expected_answer`` is passed through
directly (already extracted by apply_prompt_template).

Accepts a single JSONL file or a directory of JSONL files.

Usage::

    python -m nvflow.recipes.finance.utils.rl.responses_api_converter <input> <output.jsonl>
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from nvflow.utils import setup_logger
from nvflow.utils.jsonl import iter_jsonl, write_jsonl

logger = setup_logger(__name__)


def _convert_row(row: dict) -> dict:
    """Convert an apply_prompt_template output row to Responses API format.

    Requires ``prompt`` (model input), ``problem`` (raw question),
    and ``expected_answer`` (extracted clean answer).

    When ``_response_params`` is present (set by apply_prompt_template for
    agent-style templates), its contents (tools, parallel_tool_calls, etc.) are
    merged into ``responses_create_params``.
    """
    prompt = row.get("prompt", "")
    if not prompt:
        raise KeyError("'prompt' field is required (run apply_prompt_template first)")

    expected_answer = row.get("expected_answer", "")
    if not expected_answer:
        raise KeyError("'expected_answer' field is required (run apply_prompt_template first)")

    result = dict(row)
    rcp: dict = {"input": [{"role": "user", "content": prompt}]}
    response_params = result.pop("_response_params", None)
    if response_params:
        rcp.update(response_params)
    result["responses_create_params"] = rcp
    result["question"] = row.get("problem", "")
    result["expected_answer"] = expected_answer
    # A random uuid4 fallback here would break aggregate_seeds (keyed on uuid).
    if "uuid" not in result:
        raise KeyError(
            "Record missing 'uuid' field -- upstream data_transformation "
            "should have assigned one via uuid5(problem, generation)."
        )
    return result


def _resolve_input_files(path: Path) -> list[Path]:
    """Resolve ``path`` to the list of JSONL files we will read.

    Hoisted out of ``_iter_input`` so ``convert()`` can validate the
    input path *before* opening the output writer.  Without this
    upfront check, ``write_jsonl.__enter__`` would create (and
    immediately close) a 0-byte output file before ``_iter_input``'s
    first iteration raised ``FileNotFoundError`` -- a zombie output
    that would mislead ``Path.exists()`` health checks and reruns of
    the failed job into thinking the stage succeeded.

    Raises ``FileNotFoundError`` when ``path`` does not exist or a
    directory contains no JSONL files; ``main()`` translates this into
    a clean non-zero exit so the Slurm log shows a single error line
    instead of a buried stack trace.
    """
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No .jsonl files found in directory: {path}")
        return files
    raise FileNotFoundError(f"Input path does not exist: {path}")


def _iter_input(path: Path) -> Iterator[dict]:
    """Yield rows lazily from a single JSONL file or a directory of JSONL files.

    Streaming (vs. buffering all rows into a list) keeps peak memory
    constant in the dataset size -- at finance-sec-search scale (~187K
    records, ~13 KB each in-memory) the previous list-based reader held
    ~2.4 GB in heap before the writer started flushing.  The streaming
    contract lets the caller pipe rows straight through to ``write_jsonl``.

    Malformed JSON lines are logged with their source location and
    skipped; this preserves the historical "warn + continue" behaviour
    that operators rely on for triaging flaky inputs without having to
    rerun the whole stage.

    Path validation lives in :func:`_resolve_input_files`; ``convert()``
    calls that helper upfront so a missing input dir cannot leave a
    zombie 0-byte output behind.
    """
    files = _resolve_input_files(path)
    if path.is_file():
        logger.info("Reading file: %s", path)
    else:
        logger.info("Reading %d file(s) from directory: %s", len(files), path)

    for fpath in files:
        for line_num, (row, parse_exc, _raw_line) in enumerate(
            iter_jsonl(fpath, on_error="yield_error"), 1
        ):
            if parse_exc is not None:
                logger.warning("Skipping malformed JSON at %s:%d: %s", fpath, line_num, parse_exc)
                continue
            # ``iter_jsonl(on_error="yield_error")`` contract: ``row`` is
            # ``None`` iff ``parse_exc`` is set; the assert narrows the
            # type for mypy so ``yield row`` doesn't leak ``None`` into
            # the iterator's element type.
            assert row is not None
            yield row


def convert(input_path: Path, output_file: Path) -> None:
    """Convert apply_prompt_template output to Responses API format.

    Streams rows from input -> output without buffering, so peak heap
    is independent of dataset size.  Rows missing required fields
    (``prompt`` / ``expected_answer`` / ``uuid``) are routed to a
    sibling ``errors.jsonl`` for operator triage; the row's original
    fields are preserved so an operator can grep ``errors.jsonl`` to
    pinpoint the offending record without re-parsing the source file.

    The output directory is created up front so the empty-input edge
    case (zero rows in -> zero rows out) still materialises an empty
    output file at the expected path; downstream stages distinguish
    "empty success" from "missing file" using ``Path.exists()``.

    Input validation runs *before* the output directory is created so
    a missing/empty input never leaves a zombie 0-byte output file or
    parent dir behind on the failure path -- this is what we observed
    in slurm job 12091275 (wrong config -> input dir missing -> 0-byte
    output created before the FileNotFoundError surfaced).  The empty
    dir/file pair would mislead ``Path.exists()`` health checks and
    cause reruns to skip the stage as "already done".
    """
    _resolve_input_files(input_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped_rows: list[dict] = []

    with write_jsonl(output_file) as f:
        for row in _iter_input(input_path):
            try:
                f.write(_convert_row(row))
                converted += 1
            except (KeyError, TypeError) as e:
                logger.warning("Skipping row: %s", e)
                skipped_rows.append({"reason": str(e), **row})

    logger.info("Converted %d rows -> %s", converted, output_file)

    if skipped_rows:
        errors_file = output_file.parent / "errors.jsonl"
        with write_jsonl(errors_file) as ef:
            for row in skipped_rows:
                ef.write(row)
        logger.warning("Skipped %d rows -> %s", len(skipped_rows), errors_file)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert apply_prompt_template output to Responses API format"
    )
    parser.add_argument("input_path", help="Input JSONL file or directory")
    parser.add_argument("output_file", help="Output JSONL file path")
    args = parser.parse_args()

    try:
        convert(Path(args.input_path), Path(args.output_file))
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
