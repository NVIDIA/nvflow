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

import argparse
import json
import sys
from pathlib import Path

from nvflow.utils import setup_logger

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


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, skipping blank lines and logging malformed rows."""
    rows: list[dict] = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed JSON at %s:%d: %s", path, line_num, e)
    return rows


def _read_input(path: Path) -> list[dict]:
    """Read rows from a single JSONL file or a directory of JSONL files."""
    if path.is_file():
        logger.info("Reading file: %s", path)
        return _read_jsonl(path)

    if path.is_dir():
        jsonl_files = sorted(path.glob("*.jsonl"))
        if not jsonl_files:
            logger.error("No .jsonl files found in directory: %s", path)
            sys.exit(1)
        logger.info("Reading %d file(s) from directory: %s", len(jsonl_files), path)
        rows: list[dict] = []
        for fpath in jsonl_files:
            rows.extend(_read_jsonl(fpath))
        logger.info("Total rows read: %d", len(rows))
        return rows

    logger.error("Input path does not exist: %s", path)
    sys.exit(1)


def convert(input_path: Path, output_file: Path) -> None:
    """Convert apply_prompt_template output to Responses API format."""
    rows = _read_input(input_path)

    if not rows:
        logger.warning("Empty input -- writing empty output file")
        output_file.touch()
        return

    logger.info("Converting %d rows to Responses API format", len(rows))

    converted = 0
    skipped_rows: list[dict] = []
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        for idx, row in enumerate(rows):
            try:
                result = _convert_row(row)
                f.write(json.dumps(result) + "\n")
                converted += 1
            except (KeyError, TypeError) as e:
                logger.warning("Skipping row %d: %s", idx, e)
                skipped_rows.append({"row_index": idx, "reason": str(e), **row})

    logger.info("Converted %d rows -> %s", converted, output_file)

    if skipped_rows:
        errors_file = output_file.parent / "errors.jsonl"
        with open(errors_file, "w") as ef:
            for row in skipped_rows:
                ef.write(json.dumps(row) + "\n")
        logger.warning("Skipped %d rows -> %s", len(skipped_rows), errors_file)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert apply_prompt_template output to Responses API format"
    )
    parser.add_argument("input_path", help="Input JSONL file or directory")
    parser.add_argument("output_file", help="Output JSONL file path")
    args = parser.parse_args()

    convert(Path(args.input_path), Path(args.output_file))
    return 0


if __name__ == "__main__":
    sys.exit(main())
