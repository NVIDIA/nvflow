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
"""Atomic completion evidence for train/validation and named split outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from nvflow.stages.data.chunks import atomic_output

SPLIT_MARKER_NAME = "split.done"
SPLIT_MARKER_SCHEMA_VERSION = 1
NAMED_SPLIT_MARKER_NAME = "splits.done"
SPLIT_MANIFEST_NAME = "manifest.jsonl"


def _file_evidence(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if line.strip():
                rows += 1
    return {"sha256": digest.hexdigest(), "rows": rows}


def _expected_marker(output_dir: Path) -> dict[str, object]:
    return {
        "schema_version": SPLIT_MARKER_SCHEMA_VERSION,
        "status": "complete",
        "train": _file_evidence(output_dir / "train.jsonl"),
        "validation": _file_evidence(output_dir / "val.jsonl"),
    }


def publish_split_marker(output_dir: Path) -> Path:
    """Atomically publish hashes and row counts for both split files."""
    marker = output_dir / SPLIT_MARKER_NAME
    with atomic_output(marker) as stream:
        stream.write(json.dumps(_expected_marker(output_dir), sort_keys=True) + "\n")
    return marker


def validate_split_marker(output_dir: Path) -> dict[str, object]:
    """Validate that the marker exactly describes both current split files."""
    marker = output_dir / SPLIT_MARKER_NAME
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{marker}: invalid or missing split completion marker") from exc
    expected = _expected_marker(output_dir)
    if value != expected:
        raise ValueError(f"{marker}: split completion evidence does not match output files")
    return expected


def _expected_named_marker(output_dir: Path, names: list[str]) -> dict[str, object]:
    return {
        "schema_version": SPLIT_MARKER_SCHEMA_VERSION,
        "status": "complete",
        "splits": {name: _file_evidence(output_dir / f"{name}.jsonl") for name in names},
        "manifest": _file_evidence(output_dir / SPLIT_MANIFEST_NAME),
    }


def publish_named_split_marker(output_dir: Path, names: list[str]) -> Path:
    """Atomically publish hashes and row counts for named splits and their manifest."""
    marker = output_dir / NAMED_SPLIT_MARKER_NAME
    with atomic_output(marker) as stream:
        stream.write(json.dumps(_expected_named_marker(output_dir, names), sort_keys=True) + "\n")
    return marker


def validate_named_split_marker(output_dir: Path) -> dict[str, object]:
    """Validate that ``splits.done`` exactly describes the current split files."""
    marker = output_dir / NAMED_SPLIT_MARKER_NAME
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
        names = list(value["splits"])
        expected = _expected_named_marker(output_dir, names)
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"{marker}: invalid or missing split completion marker") from exc
    if value != expected:
        raise ValueError(f"{marker}: split completion evidence does not match output files")
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--named",
        action="store_true",
        help=f"Validate named splits ({NAMED_SPLIT_MARKER_NAME}) instead of train and validation.",
    )
    args = parser.parse_args()
    validate = validate_named_split_marker if args.named else validate_split_marker
    print(json.dumps(validate(args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
