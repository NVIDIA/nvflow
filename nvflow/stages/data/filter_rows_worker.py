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
"""Drop the JSONL records whose id appears in another file."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nvflow.stages.data.chunks import atomic_output, atomic_write_json
from nvflow.stages.data.text_keys import field_value


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def filter_rows(
    input_file: Path,
    exclude_file: Path,
    output_dir: Path,
    *,
    id_field: str,
    exclude_id_field: str,
) -> dict[str, int]:
    """Write ``kept.jsonl``, ``dropped.jsonl``, and ``report.json``.

    A row of either file without its id field fails the run.
    """
    excluded_ids: set[str] = set()
    for number, row in enumerate(_rows(exclude_file), 1):
        if (value := field_value(row, exclude_id_field)) is None:
            raise ValueError(f"{exclude_file}: row {number} has no {exclude_id_field!r}")
        excluded_ids.add(str(value))
    matched: set[str] = set()
    kept = dropped = 0
    with (
        atomic_output(output_dir / "kept.jsonl") as kept_stream,
        atomic_output(output_dir / "dropped.jsonl") as dropped_stream,
    ):
        for number, row in enumerate(_rows(input_file), 1):
            if (value := field_value(row, id_field)) is None:
                raise ValueError(f"{input_file}: row {number} has no {id_field!r}")
            line = json.dumps(row, ensure_ascii=False) + "\n"
            if (key := str(value)) in excluded_ids:
                matched.add(key)
                dropped_stream.write(line)
                dropped += 1
            else:
                kept_stream.write(line)
                kept += 1
    report = {
        "input_rows": kept + dropped,
        "kept_rows": kept,
        "dropped_rows": dropped,
        "exclude_ids": len(excluded_ids),
        "unmatched_exclude_ids": len(excluded_ids - matched),
    }
    atomic_write_json(output_dir / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--exclude_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--id_field", required=True)
    parser.add_argument("--exclude_id_field", required=True)
    args = parser.parse_args(argv)
    report = filter_rows(
        args.input_file,
        args.exclude_file,
        args.output_dir,
        id_field=args.id_field,
        exclude_id_field=args.exclude_id_field,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
