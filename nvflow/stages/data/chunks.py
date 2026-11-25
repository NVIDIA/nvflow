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
"""Manifests and completion records for resumable, chunked data stages.

A manifest binds an output directory to its inputs (SHA-256 digest and row
count of each file), the chunk row ranges, and the settings that change results.
Each step writes its outputs atomically and then a completion record holding
their digests, so a rerun skips finished steps and fails closed when the inputs
or settings differ from the manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1


def digest_value(value: Any) -> str:
    """SHA-256 digest of a JSON value in canonical form."""
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_evidence(path: Path) -> dict[str, int | str]:
    """SHA-256 digest, size, and non-empty line count of one file."""
    digest = hashlib.sha256()
    rows = size = 0
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            size += len(line)
            if line.strip():
                rows += 1
    return {"sha256": digest.hexdigest(), "bytes": size, "rows": rows}


@contextmanager
def atomic_output(path: Path, mode: str = "w") -> Iterator[IO[Any]]:
    """Open a temporary sibling of *path*; rename it into place only on success.

    The sibling's name is random, as process ids repeat across nodes and containers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open(mode, encoding=None if "b" in mode else "utf-8") as stream:
            yield stream
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    with atomic_output(path) as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n")


def chunk_ranges(total_rows: int, num_chunks: int) -> list[list[int]]:
    """Contiguous ``[start, end)`` row ranges; later chunks may be empty."""
    size = -(-total_rows // num_chunks) if total_rows else 0
    return [
        [min(index * size, total_rows), min((index + 1) * size, total_rows)]
        for index in range(num_chunks)
    ]


def build_manifest(
    inputs: Sequence[Path],
    num_chunks: int,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    if num_chunks < 1:
        raise ValueError("num_chunks must be at least 1")
    files = [{"path": str(path), **file_evidence(path)} for path in inputs]
    total = sum(int(item["rows"]) for item in files)
    # A JSON round trip makes tuples lists, so a reloaded manifest digests the same.
    normalized = json.loads(json.dumps(dict(settings)))
    return {
        "schema_version": SCHEMA_VERSION,
        "inputs": files,
        "total_rows": total,
        "num_chunks": num_chunks,
        "ranges": chunk_ranges(total, num_chunks),
        "settings": normalized,
        "settings_sha256": digest_value(normalized),
    }


def _differences(existing: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    problems = []
    old_inputs = [(item.get("path"), item.get("sha256")) for item in existing.get("inputs", [])]
    new_inputs = [(item["path"], item["sha256"]) for item in current["inputs"]]
    if old_inputs != new_inputs:
        problems.append("the input files or their contents changed")
    if existing.get("num_chunks") != current["num_chunks"]:
        problems.append(
            f"num_chunks changed from {existing.get('num_chunks')} to {current['num_chunks']}"
        )
    if existing.get("settings_sha256") != current["settings_sha256"]:
        old = existing.get("settings") or {}
        changed = sorted(
            key
            for key in set(old) | set(current["settings"])
            if old.get(key) != current["settings"].get(key)
        )
        problems.append("settings changed: " + ", ".join(changed))
    if existing.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"manifest schema {existing.get('schema_version')} is not {SCHEMA_VERSION}")
    return problems


def prepare_manifest(
    output_dir: Path,
    inputs: Sequence[Path],
    num_chunks: int,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the manifest, or confirm an existing one still describes this run."""
    current = build_manifest(inputs, num_chunks, settings)
    path = output_dir / MANIFEST_NAME
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        problems = _differences(existing, current)
        if problems:
            raise ValueError(
                f"{output_dir} holds results for a different run: {'; '.join(problems)}. "
                "Use a new output_dir, or delete this one to recompute."
            )
        return dict(existing)
    atomic_write_json(path, current)
    return current


def iter_rows(
    manifest: Mapping[str, Any],
    start: int = 0,
    end: int | None = None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(row_index, record)`` for manifest rows in ``[start, end)``."""
    stop = int(manifest["total_rows"]) if end is None else end
    index = 0
    for item in manifest["inputs"]:
        with open(item["path"], encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                if index >= stop:
                    return
                if index >= start:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError(f"{item['path']}: row {index} is not a JSON object")
                    yield index, record
                index += 1


def publish_completion(
    path: Path,
    manifest: Mapping[str, Any],
    outputs: Mapping[str, Path],
    **details: Any,
) -> Path:
    """Record that *outputs* are complete for this manifest."""
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "manifest_sha256": digest_value(dict(manifest)),
        "outputs": {name: file_evidence(output) for name, output in sorted(outputs.items())},
        **details,
    }
    atomic_write_json(path, record)
    return path


def completion_matches(
    path: Path,
    manifest: Mapping[str, Any],
    outputs: Mapping[str, Path],
) -> bool:
    """Whether *path* records exactly these outputs for this manifest."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if record.get("manifest_sha256") != digest_value(dict(manifest)):
        return False
    try:
        actual = {name: file_evidence(output) for name, output in sorted(outputs.items())}
    except OSError:
        return False
    return bool(record.get("outputs") == actual)


__all__ = [
    "MANIFEST_NAME",
    "atomic_output",
    "atomic_write_json",
    "completion_matches",
    "digest_value",
    "file_evidence",
    "iter_rows",
    "prepare_manifest",
    "publish_completion",
]
