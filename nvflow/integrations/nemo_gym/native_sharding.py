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
"""Dependency-free sharding and resume guards for native NeMo Gym workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import BinaryIO, Final, TextIO, TypeAlias, cast

SHARD_MANIFEST_SCHEMA_VERSION: Final[int] = 1
JOB_COMPLETION_SCHEMA_VERSION: Final[int] = 1
TASK_INDEX_KEY: Final[str] = "_ng_task_index"
ROLLOUT_INDEX_KEY: Final[str] = "_ng_rollout_index"

_PathLike: TypeAlias = str | os.PathLike[str]  # noqa: UP040 - mypy compatibility
_PairKey: TypeAlias = tuple[str, str]  # noqa: UP040


class NativeGymShardError(ValueError):
    """Raised when a native Gym shard or resume artifact is unsafe."""


class ShardIdentityError(NativeGymShardError):
    """Raised when an existing shard manifest has a different identity."""


@dataclass(frozen=True, slots=True)
class ShardManifest:
    """Identity and balanced source range for one prepared input shard."""

    source_sha256: str
    effective_row_count: int
    chunk_index: int
    num_chunks: int
    selected_start: int
    selected_end: int
    selected_row_count: int
    num_repeats: int
    rollout_config_sha256: str
    shard_sha256: str
    kind: str = "shard_manifest"
    schema_version: int = SHARD_MANIFEST_SCHEMA_VERSION

    def as_json_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TailRepairResult:
    """Description of the narrow mutation made to a JSONL tail."""

    action: str
    original_size: int
    repaired_size: int

    def as_json_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RolloutCompletion:
    """Deterministic evidence that every materialized rollout succeeded."""

    materialized_inputs_sha256: str
    successes_sha256: str
    materialized_input_count: int
    success_count: int
    behavior_sha256: str | None = None
    artifacts_sha256: dict[str, str] | None = None
    kind: str = "rollout_completion"
    schema_version: int = JOB_COMPLETION_SCHEMA_VERSION

    def as_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.behavior_sha256 is None:
            payload.pop("behavior_sha256")
        if self.artifacts_sha256 is None:
            payload.pop("artifacts_sha256")
        return {"status": "complete", **payload}


@dataclass(frozen=True, slots=True)
class _SourceScan:
    source_sha256: str
    effective_row_count: int


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")


def _strict_json_loads(text: str) -> object:
    return json.loads(text, parse_constant=_reject_nonfinite_json)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_artifact(path: Path) -> str:
    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = sorted(
        (candidate for candidate in path.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(path).as_posix(),
    )
    for child in files:
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        with child.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _load_json_object(path: Path, kind: str) -> dict[str, object]:
    try:
        value = _strict_json_loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise NativeGymShardError(f"{path}: invalid {kind}: {exc}") from exc
    if not isinstance(value, dict):
        raise NativeGymShardError(f"{path}: {kind} must be an object")
    return cast(dict[str, object], value)


def _expected_completion_count(
    shard_manifests: Sequence[Path],
    reference_completion: Path | None,
    source: Path | None = None,
    manifest_behavior_sha256: str | None = None,
    accepted_manifest_behavior_sha256: Collection[str] = (),
) -> int | None:
    if shard_manifests and reference_completion is not None:
        raise NativeGymShardError(
            "completion count accepts shard manifests or a reference marker, not both"
        )
    if reference_completion is not None:
        value = _load_json_object(reference_completion, "completion record")
        count = value.get("materialized_input_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise NativeGymShardError(
                f"{reference_completion}: invalid materialized_input_count {count!r}"
            )
        return count
    if not shard_manifests:
        return None

    source_sha256 = _sha256_file(source) if source is not None else None
    total = 0
    for manifest_path in shard_manifests:
        value = _load_json_object(manifest_path, "shard manifest")
        if value.get("kind") != "shard_manifest" or value.get("schema_version") != 1:
            raise NativeGymShardError(f"{manifest_path}: unsupported shard manifest schema")
        if source_sha256 is not None and value.get("source_sha256") != source_sha256:
            raise ShardIdentityError(f"{manifest_path}: source dataset changed")
        if manifest_behavior_sha256 is not None and value.get("rollout_config_sha256") not in (
            manifest_behavior_sha256,
            *accepted_manifest_behavior_sha256,
        ):
            raise ShardIdentityError(f"{manifest_path}: rollout configuration changed")
        selected = value.get("selected_row_count")
        repeats = value.get("num_repeats")
        if (
            isinstance(selected, bool)
            or not isinstance(selected, int)
            or selected < 0
            or isinstance(repeats, bool)
            or not isinstance(repeats, int)
            or repeats <= 0
        ):
            raise NativeGymShardError(f"{manifest_path}: invalid selected_row_count/num_repeats")
        total += selected * repeats
    return total


def _parse_json_object(raw_line: bytes, path: Path, line_number: int) -> dict[str, object]:
    if not raw_line.strip():
        raise NativeGymShardError(f"{path}:{line_number}: blank JSONL rows are not allowed")
    try:
        value = _strict_json_loads(raw_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise NativeGymShardError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
    if not isinstance(value, dict):
        raise NativeGymShardError(f"{path}:{line_number}: JSONL row must be an object")
    return cast(dict[str, object], value)


def _index_token(value: object, *, path: Path, line_number: int, field: str) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise NativeGymShardError(f"{path}:{line_number}: {field} is not a JSON value") from exc
    return f"{type(value).__name__}:{encoded}"


def _scan_source(source: Path, max_num_samples: int | None) -> _SourceScan:
    digest = hashlib.sha256()
    seen_task_ids: dict[str, int] = {}
    effective_row_count = 0

    with source.open("rb") as stream:
        for source_index, raw_line in enumerate(stream):
            digest.update(raw_line)
            if max_num_samples is not None and effective_row_count >= max_num_samples:
                continue

            row = _parse_json_object(raw_line, source, source_index + 1)
            task_index = row[TASK_INDEX_KEY] if TASK_INDEX_KEY in row else source_index
            token = _index_token(
                task_index,
                path=source,
                line_number=source_index + 1,
                field=TASK_INDEX_KEY,
            )
            first_index = seen_task_ids.get(token)
            if first_index is not None:
                raise NativeGymShardError(
                    f"{source}:{source_index + 1}: duplicate {TASK_INDEX_KEY} "
                    f"{task_index!r}; first used by source row {first_index}"
                )
            seen_task_ids[token] = source_index
            effective_row_count += 1

    return _SourceScan(
        source_sha256=digest.hexdigest(),
        effective_row_count=effective_row_count,
    )


def _sync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        directory_fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    except OSError:
        pass
    finally:
        os.close(directory_fd)


@contextmanager
def _atomic_text_writer(path: Path) -> Iterator[TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    with _atomic_text_writer(path) as stream:
        json.dump(
            payload,
            stream,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")


def _require_distinct_paths(paths: Mapping[str, Path]) -> None:
    resolved: dict[Path, str] = {}
    for label, path in paths.items():
        normalized = path.expanduser().resolve()
        previous = resolved.get(normalized)
        if previous is not None:
            raise NativeGymShardError(f"{label} and {previous} must use different paths: {path}")
        resolved[normalized] = label


def _validate_prepare_options(
    *,
    chunk_index: int,
    num_chunks: int,
    num_repeats: int,
    max_num_samples: int | None,
) -> int | None:
    if num_chunks <= 0:
        raise NativeGymShardError("num_chunks must be greater than zero")
    if chunk_index < 0 or chunk_index >= num_chunks:
        raise NativeGymShardError(f"chunk_index must be in [0, {num_chunks}), got {chunk_index}")
    if num_repeats <= 0:
        raise NativeGymShardError("num_repeats must be greater than zero")
    if max_num_samples is not None and max_num_samples < 0:
        raise NativeGymShardError("max_num_samples cannot be negative")
    return None if max_num_samples in (None, 0) else max_num_samples


def _load_reusable_manifest(
    path: Path,
    identity: ShardManifest,
    accepted_rollout_config_sha256: Collection[str] = (),
) -> ShardManifest:
    try:
        value = _strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ShardIdentityError(f"cannot reuse invalid shard manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ShardIdentityError(f"cannot reuse non-object shard manifest {path}")

    actual = cast(dict[str, object], value)
    expected = identity.as_json_dict()
    identity_keys = expected.keys() - {"shard_sha256"}
    differing = sorted(
        key
        for key in actual.keys() | expected.keys()
        if key not in actual
        or key not in expected
        or (
            key in identity_keys
            and (type(actual[key]) is not type(expected[key]) or actual[key] != expected[key])
        )
    )
    if differing == ["rollout_config_sha256"] and actual.get("rollout_config_sha256") in tuple(
        accepted_rollout_config_sha256
    ):
        differing = []
    if differing:
        raise ShardIdentityError(
            f"refusing to reuse {path}: shard identity differs in {', '.join(differing)}"
        )

    shard_sha256 = actual["shard_sha256"]
    if (
        not isinstance(shard_sha256, str)
        or len(shard_sha256) != 64
        or any(character not in "0123456789abcdef" for character in shard_sha256)
    ):
        raise ShardIdentityError(f"cannot reuse shard manifest {path}: shard_sha256 is invalid")
    return replace(identity, shard_sha256=shard_sha256)


def _write_prepared_shard(
    source: Path,
    shard: Path,
    manifest: ShardManifest,
) -> str:
    written = 0
    source_digest = hashlib.sha256()
    shard_digest = hashlib.sha256()

    with _atomic_text_writer(shard) as destination:
        with source.open("rb") as stream:
            for source_index, raw_line in enumerate(stream):
                source_digest.update(raw_line)
                if source_index < manifest.selected_start:
                    continue
                if source_index >= manifest.selected_end:
                    continue

                row = _parse_json_object(raw_line, source, source_index + 1)
                if TASK_INDEX_KEY not in row:
                    row[TASK_INDEX_KEY] = source_index
                serialized = (
                    json.dumps(
                        row,
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
                destination.write(serialized)
                shard_digest.update(serialized.encode("utf-8"))
                written += 1

        if source_digest.hexdigest() != manifest.source_sha256:
            raise NativeGymShardError(f"{source} changed while preparing the shard")
        if written != manifest.selected_row_count:
            raise NativeGymShardError(
                f"{source} changed while preparing the shard: expected "
                f"{manifest.selected_row_count} selected rows, wrote {written}"
            )
    return shard_digest.hexdigest()


def _verify_prepared_shard(source: Path, shard: Path, manifest: ShardManifest) -> None:
    if not shard.is_file():
        raise NativeGymShardError(f"prepared shard is missing or not a file: {shard}")

    shard_digest = hashlib.sha256()
    row_count = 0
    with source.open("rb") as source_stream, shard.open("rb") as shard_stream:
        source_iterator = iter(source_stream)
        try:
            for _ in range(manifest.selected_start):
                next(source_iterator)
        except StopIteration as exc:
            raise NativeGymShardError(
                f"{source} changed while validating prepared shard {shard}"
            ) from exc

        for line_number, raw_line in enumerate(shard_stream, start=1):
            shard_digest.update(raw_line)
            row = _parse_json_object(raw_line, shard, line_number)
            if row_count < manifest.selected_row_count:
                source_index = manifest.selected_start + row_count
                try:
                    source_row = _parse_json_object(
                        next(source_iterator),
                        source,
                        source_index + 1,
                    )
                except StopIteration as exc:
                    raise NativeGymShardError(
                        f"{source} changed while validating prepared shard {shard}"
                    ) from exc
                expected_task_index = (
                    source_row[TASK_INDEX_KEY] if TASK_INDEX_KEY in source_row else source_index
                )
                if TASK_INDEX_KEY not in row:
                    raise NativeGymShardError(
                        f"{shard}:{line_number}: prepared shard row is missing {TASK_INDEX_KEY}"
                    )
                if _index_token(
                    row[TASK_INDEX_KEY],
                    path=shard,
                    line_number=line_number,
                    field=TASK_INDEX_KEY,
                ) != _index_token(
                    expected_task_index,
                    path=source,
                    line_number=source_index + 1,
                    field=TASK_INDEX_KEY,
                ):
                    raise NativeGymShardError(
                        f"{shard}:{line_number}: prepared shard has unexpected "
                        f"{TASK_INDEX_KEY}; expected {expected_task_index!r}"
                    )
            row_count += 1

    if row_count != manifest.selected_row_count:
        raise NativeGymShardError(
            f"prepared shard {shard} row count differs: "
            f"expected {manifest.selected_row_count}, found {row_count}"
        )
    actual_digest = shard_digest.hexdigest()
    if actual_digest != manifest.shard_sha256:
        raise NativeGymShardError(
            f"prepared shard {shard} digest differs: "
            f"expected {manifest.shard_sha256}, found {actual_digest}"
        )


def prepare_input_shard(
    source_jsonl: _PathLike,
    shard_jsonl: _PathLike,
    manifest_json: _PathLike,
    *,
    chunk_index: int,
    num_chunks: int,
    num_repeats: int,
    rollout_config_sha256: str,
    max_num_samples: int | None = None,
    allow_empty: bool = False,
    accepted_rollout_config_sha256: Collection[str] = (),
) -> ShardManifest:
    """Prepare or safely reuse one deterministic, globally indexed input shard.

    A source without usable rows fails unless *allow_empty*, which prepares an
    empty shard instead. A reused manifest that records one of
    *accepted_rollout_config_sha256* is rewritten with *rollout_config_sha256*.
    """
    source = Path(source_jsonl)
    shard = Path(shard_jsonl)
    manifest_path = Path(manifest_json)
    _require_distinct_paths(
        {
            "source_jsonl": source,
            "shard_jsonl": shard,
            "manifest_json": manifest_path,
        }
    )
    limit = _validate_prepare_options(
        chunk_index=chunk_index,
        num_chunks=num_chunks,
        num_repeats=num_repeats,
        max_num_samples=max_num_samples,
    )
    if len(rollout_config_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in rollout_config_sha256
    ):
        raise NativeGymShardError(
            "rollout_config_sha256 must be a lowercase hexadecimal SHA-256 digest"
        )
    source_scan = _scan_source(source, limit)
    if source_scan.effective_row_count == 0 and not allow_empty:
        raise NativeGymShardError(f"source contains zero usable rows: {source}")
    start = source_scan.effective_row_count * chunk_index // num_chunks
    end = source_scan.effective_row_count * (chunk_index + 1) // num_chunks
    identity = ShardManifest(
        source_sha256=source_scan.source_sha256,
        effective_row_count=source_scan.effective_row_count,
        chunk_index=chunk_index,
        num_chunks=num_chunks,
        selected_start=start,
        selected_end=end,
        selected_row_count=end - start,
        num_repeats=num_repeats,
        rollout_config_sha256=rollout_config_sha256,
        shard_sha256="",
    )

    if manifest_path.exists():
        manifest = _load_reusable_manifest(manifest_path, identity, accepted_rollout_config_sha256)
        _verify_prepared_shard(source, shard, manifest)
        if _load_json_object(manifest_path, "shard manifest") != manifest.as_json_dict():
            _atomic_write_json(manifest_path, manifest.as_json_dict())
        return manifest

    shard_sha256 = _write_prepared_shard(source, shard, identity)
    manifest = replace(identity, shard_sha256=shard_sha256)
    _atomic_write_json(manifest_path, manifest.as_json_dict())
    return manifest


def repair_truncated_jsonl_tail(path: _PathLike) -> TailRepairResult:
    """Append a missing delimiter or remove only an invalid unterminated tail."""
    jsonl_path = Path(path)
    with jsonl_path.open("r+b") as stream:
        stream.seek(0, os.SEEK_END)
        original_size = stream.tell()
        if original_size == 0:
            return TailRepairResult("unchanged", 0, 0)

        stream.seek(-1, os.SEEK_END)
        if stream.read(1) == b"\n":
            return TailRepairResult("unchanged", original_size, original_size)

        tail_start = _final_binary_line_start(stream, original_size)
        stream.seek(tail_start)
        tail = stream.read(original_size - tail_start)
        try:
            _strict_json_loads(tail.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            stream.truncate(tail_start)
            repaired_size = tail_start
            action = "truncated_invalid_tail"
        else:
            stream.seek(0, os.SEEK_END)
            stream.write(b"\n")
            repaired_size = original_size + 1
            action = "appended_newline"

        stream.flush()
        os.fsync(stream.fileno())
        return TailRepairResult(action, original_size, repaired_size)


def _final_binary_line_start(stream: BinaryIO, size: int) -> int:
    position = size
    block_size = 64 * 1024
    while position > 0:
        read_start = max(0, position - block_size)
        stream.seek(read_start)
        block = stream.read(position - read_start)
        newline = block.rfind(b"\n")
        if newline >= 0:
            return read_start + newline + 1
        position = read_start
    return 0


def _pair_key(row: Mapping[str, object], path: Path, line_number: int) -> _PairKey:
    missing = [key for key in (TASK_INDEX_KEY, ROLLOUT_INDEX_KEY) if key not in row]
    if missing:
        raise NativeGymShardError(
            f"{path}:{line_number}: missing required key(s): {', '.join(missing)}"
        )
    return (
        _index_token(
            row[TASK_INDEX_KEY],
            path=path,
            line_number=line_number,
            field=TASK_INDEX_KEY,
        ),
        _index_token(
            row[ROLLOUT_INDEX_KEY],
            path=path,
            line_number=line_number,
            field=ROLLOUT_INDEX_KEY,
        ),
    )


def _index_pair_keys(
    connection: sqlite3.Connection,
    table: str,
    path: Path,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0

    with path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            digest.update(raw_line)
            row = _parse_json_object(raw_line, path, line_number)
            task_index, rollout_index = _pair_key(row, path, line_number)
            try:
                connection.execute(
                    f"INSERT INTO {table}(task_index, rollout_index) VALUES (?, ?)",
                    (task_index, rollout_index),
                )
            except sqlite3.IntegrityError as exc:
                raise NativeGymShardError(
                    f"{path}:{line_number}: duplicate ({TASK_INDEX_KEY}, {ROLLOUT_INDEX_KEY})"
                ) from exc
            count += 1

    return count, digest.hexdigest()


def _completion_evidence(
    materialized_inputs_jsonl: _PathLike,
    successes_jsonl: _PathLike,
    behavior_sha256: str | None = None,
    expected_count: int | None = None,
    artifacts: Mapping[str, _PathLike] | None = None,
) -> RolloutCompletion:
    """Return deterministic evidence after validating exact successful coverage."""
    materialized_inputs = Path(materialized_inputs_jsonl)
    successes = Path(successes_jsonl)
    _require_distinct_paths(
        {
            "materialized_inputs_jsonl": materialized_inputs,
            "successes_jsonl": successes,
        }
    )

    descriptor, raw_index = tempfile.mkstemp(
        prefix=".completion-identity.",
        suffix=".sqlite3",
    )
    os.close(descriptor)
    index_path = Path(raw_index)
    connection = sqlite3.connect(index_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        for table in ("expected", "actual"):
            connection.execute(
                f"CREATE TABLE {table} ("
                "task_index TEXT NOT NULL, rollout_index TEXT NOT NULL, "
                "PRIMARY KEY(task_index, rollout_index))"
            )
        expected_count_found, expected_sha256 = _index_pair_keys(
            connection,
            "expected",
            materialized_inputs,
        )
        if expected_count_found == 0 and expected_count != 0:
            raise NativeGymShardError(
                f"materialized inputs contain zero rollout keys: {materialized_inputs}"
            )
        if expected_count is not None and expected_count_found != expected_count:
            raise NativeGymShardError(
                "materialized input count does not match deterministic source identity: "
                f"expected={expected_count}, found={expected_count_found}"
            )
        actual_count, actual_sha256 = _index_pair_keys(connection, "actual", successes)
        missing = connection.execute(
            "SELECT COUNT(*) FROM expected LEFT JOIN actual "
            "USING(task_index, rollout_index) WHERE actual.task_index IS NULL"
        ).fetchone()[0]
        unexpected = connection.execute(
            "SELECT COUNT(*) FROM actual LEFT JOIN expected "
            "USING(task_index, rollout_index) WHERE expected.task_index IS NULL"
        ).fetchone()[0]
        if missing or unexpected:
            raise NativeGymShardError(
                "successful rollout keys do not exactly match materialized inputs: "
                f"missing={missing}, unexpected={unexpected}"
            )
    finally:
        connection.close()
        index_path.unlink(missing_ok=True)

    return RolloutCompletion(
        materialized_inputs_sha256=expected_sha256,
        successes_sha256=actual_sha256,
        materialized_input_count=expected_count_found,
        success_count=actual_count,
        behavior_sha256=behavior_sha256,
        artifacts_sha256=(
            {name: _sha256_artifact(Path(path)) for name, path in sorted((artifacts or {}).items())}
            or None
        ),
    )


def validate_rollout_completion(
    materialized_inputs_jsonl: _PathLike,
    successes_jsonl: _PathLike,
    marker_json: _PathLike,
    behavior_sha256: str | None = None,
    expected_count: int | None = None,
    artifacts: Mapping[str, _PathLike] | None = None,
    source_path: _PathLike | None = None,
    shard_manifests: Sequence[_PathLike] = (),
    manifest_behavior_sha256: str | None = None,
    accepted_behavior_sha256: Collection[str] = (),
    accepted_manifest_behavior_sha256: Collection[str] = (),
) -> RolloutCompletion:
    """Validate content evidence and, when supplied, expected behavior.

    A marker that records one of *accepted_behavior_sha256* stands in for
    *behavior_sha256*.
    """
    marker_path = Path(marker_json)
    try:
        payload = _strict_json_loads(marker_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise NativeGymShardError(f"{marker_path}: invalid completion record: {exc}") from exc
    marker_behavior = payload.get("behavior_sha256")
    if marker_behavior is not None and not isinstance(marker_behavior, str):
        raise NativeGymShardError(f"{marker_path}: completion behavior_sha256 must be a string")
    if expected_count is None and shard_manifests:
        expected_count = _expected_completion_count(
            [Path(path) for path in shard_manifests],
            None,
            Path(source_path) if source_path is not None else None,
            manifest_behavior_sha256,
            accepted_manifest_behavior_sha256,
        )
    expected_behavior = behavior_sha256 if behavior_sha256 is not None else marker_behavior
    if behavior_sha256 is not None and marker_behavior in accepted_behavior_sha256:
        expected_behavior = marker_behavior
    evidence = _completion_evidence(
        materialized_inputs_jsonl,
        successes_jsonl,
        expected_behavior,
        expected_count,
        artifacts,
    )
    if artifacts is None and isinstance(payload.get("artifacts_sha256"), dict):
        evidence = replace(
            evidence,
            artifacts_sha256=cast(dict[str, str], payload["artifacts_sha256"]),
        )
    if payload != evidence.as_json_dict():
        raise NativeGymShardError(
            f"{marker_path}: completion evidence does not match current rollout artifacts"
        )
    return evidence


def publish_rollout_completion(
    materialized_inputs_jsonl: _PathLike,
    successes_jsonl: _PathLike,
    marker_json: _PathLike,
    behavior_sha256: str | None = None,
    expected_count: int | None = None,
    artifacts: Mapping[str, _PathLike] | None = None,
    source_path: _PathLike | None = None,
    shard_manifests: Sequence[_PathLike] = (),
    manifest_behavior_sha256: str | None = None,
    accepted_manifest_behavior_sha256: Collection[str] = (),
) -> RolloutCompletion:
    """Validate exact successful rollout coverage and atomically publish a marker."""
    marker_path = Path(marker_json)
    _require_distinct_paths(
        {
            "materialized_inputs_jsonl": Path(materialized_inputs_jsonl),
            "successes_jsonl": Path(successes_jsonl),
            "marker_json": marker_path,
        }
    )
    if expected_count is None and shard_manifests:
        expected_count = _expected_completion_count(
            [Path(path) for path in shard_manifests],
            None,
            Path(source_path) if source_path is not None else None,
            manifest_behavior_sha256,
            accepted_manifest_behavior_sha256,
        )
    marker = _completion_evidence(
        materialized_inputs_jsonl,
        successes_jsonl,
        behavior_sha256,
        expected_count,
        artifacts,
    )
    _atomic_write_json(marker_path, marker.as_json_dict())
    return marker


def bind_behavior_manifest(
    manifest_json: _PathLike,
    behavior_sha256: str,
    source_completion_json: _PathLike | None = None,
    accepted_behavior_sha256: Collection[str] = (),
) -> dict[str, object]:
    """Create or validate behavior identity for partial output.

    A manifest that records one of *accepted_behavior_sha256* for the same
    source is rewritten with *behavior_sha256*.
    """
    if len(behavior_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in behavior_sha256
    ):
        raise NativeGymShardError("behavior_sha256 must be a lowercase SHA-256 digest")
    manifest_path = Path(manifest_json)
    source_completion = Path(source_completion_json) if source_completion_json is not None else None
    expected: dict[str, object] = {
        "kind": "behavior_manifest",
        "schema_version": 1,
        "behavior_sha256": behavior_sha256,
    }
    if source_completion is not None:
        expected["source_completion_sha256"] = _sha256_file(source_completion)
    if manifest_path.exists():
        try:
            actual = _strict_json_loads(manifest_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise NativeGymShardError(f"{manifest_path}: invalid behavior manifest: {exc}") from exc
        if actual == expected:
            return expected
        if not isinstance(actual, dict) or actual.get("behavior_sha256") not in (
            accepted_behavior_sha256
        ):
            raise ShardIdentityError(f"refusing to reuse {manifest_path}: rejudge behavior differs")
        if {**actual, "behavior_sha256": behavior_sha256} != expected:
            raise ShardIdentityError(f"refusing to reuse {manifest_path}: rejudge behavior differs")
    _atomic_write_json(manifest_path, expected)
    return expected


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and validate NVFlow-owned native Gym shard artifacts."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="prepare one deterministic input shard")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--chunk-index", type=int, required=True)
    prepare.add_argument("--num-chunks", type=int, required=True)
    prepare.add_argument("--num-repeats", type=int, required=True)
    prepare.add_argument("--rollout-config-sha256", required=True)
    prepare.add_argument("--accept-rollout-config-sha256", action="append", default=[])
    prepare.add_argument("--max-num-samples", type=int)
    prepare.add_argument("--allow-empty", action="store_true")

    repair = commands.add_parser("repair", help="repair only an unterminated JSONL tail")
    repair.add_argument("--jsonl", type=Path, required=True)

    bind = commands.add_parser(
        "bind-behavior",
        help="create or validate behavior identity for partial output",
    )
    bind.add_argument("--manifest", type=Path, required=True)
    bind.add_argument("--behavior-sha256", required=True)
    bind.add_argument("--accept-behavior-sha256", action="append", default=[])
    bind.add_argument("--source-completion", type=Path)

    complete = commands.add_parser(
        "complete",
        help="validate successes and atomically write a completion record",
    )
    complete.add_argument("--materialized-inputs", type=Path, required=True)
    complete.add_argument("--successes", type=Path, required=True)
    complete.add_argument("--completion", type=Path, required=True)
    complete.add_argument("--behavior-sha256")
    complete.add_argument("--shard-manifest", type=Path, action="append", default=[])
    complete.add_argument("--reference-completion", type=Path)
    complete.add_argument("--source", type=Path)
    complete.add_argument("--manifest-behavior-sha256")
    complete.add_argument("--accept-manifest-behavior-sha256", action="append", default=[])
    complete.add_argument("--artifact", action="append", default=[])
    complete.add_argument("--layout-root", type=Path)

    validate = commands.add_parser(
        "validate",
        help="validate that an existing completion record still matches its artifacts",
    )
    validate.add_argument("--materialized-inputs", type=Path, required=True)
    validate.add_argument("--successes", type=Path, required=True)
    validate.add_argument("--completion", type=Path, required=True)
    validate.add_argument("--behavior-sha256")
    validate.add_argument("--accept-behavior-sha256", action="append", default=[])
    validate.add_argument("--shard-manifest", type=Path, action="append", default=[])
    validate.add_argument("--reference-completion", type=Path)
    validate.add_argument("--source", type=Path)
    validate.add_argument("--manifest-behavior-sha256")
    validate.add_argument("--accept-manifest-behavior-sha256", action="append", default=[])
    validate.add_argument("--artifact", action="append", default=[])
    validate.add_argument("--layout-root", type=Path)
    return parser


def _artifact_arguments(
    values: Sequence[str],
    *,
    layout_root: Path | None = None,
    completion: Path | None = None,
) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if (
            not separator
            or not name
            or not raw_path
            or not all(part.isidentifier() for part in name.split("."))
        ):
            raise NativeGymShardError(f"artifact must use a dotted name=path value, got {value!r}")
        if name in artifacts:
            raise NativeGymShardError(f"duplicate completion artifact: {name}")
        artifacts[name] = Path(raw_path)
    if layout_root is not None:
        if completion is None:
            raise NativeGymShardError("layout-root requires a completion path")
        from .native_contract import NativeGymOutputLayout

        payload = _load_json_object(completion, "completion record")
        evidence = payload.get("artifacts_sha256")
        if not isinstance(evidence, dict):
            raise NativeGymShardError(f"{completion}: completion record has no analyzed artifacts")
        resolved = NativeGymOutputLayout.collection(layout_root).resolve_analyzed_artifacts(
            evidence
        )
        for name, path in resolved.items():
            if name in artifacts and artifacts[name] != Path(path):
                raise NativeGymShardError(f"conflicting artifact path for {name}")
            artifacts[name] = Path(path)
    return artifacts


def main(argv: Sequence[str] | None = None) -> int:
    """Run the worker-facing command-line interface."""
    args = _build_parser().parse_args(None if argv is None else list(argv))
    try:
        if args.command == "prepare":
            result = prepare_input_shard(
                args.source,
                args.output,
                args.manifest,
                chunk_index=args.chunk_index,
                num_chunks=args.num_chunks,
                num_repeats=args.num_repeats,
                rollout_config_sha256=args.rollout_config_sha256,
                max_num_samples=args.max_num_samples,
                allow_empty=args.allow_empty,
                accepted_rollout_config_sha256=args.accept_rollout_config_sha256,
            ).as_json_dict()
        elif args.command == "repair":
            result = repair_truncated_jsonl_tail(args.jsonl).as_json_dict()
        elif args.command == "bind-behavior":
            result = bind_behavior_manifest(
                args.manifest,
                args.behavior_sha256,
                args.source_completion,
                args.accept_behavior_sha256,
            )
        elif args.command == "complete":
            expected_count = _expected_completion_count(
                args.shard_manifest,
                args.reference_completion,
                args.source,
                args.manifest_behavior_sha256,
                args.accept_manifest_behavior_sha256,
            )
            result = publish_rollout_completion(
                args.materialized_inputs,
                args.successes,
                args.completion,
                args.behavior_sha256,
                expected_count,
                _artifact_arguments(
                    args.artifact,
                    layout_root=args.layout_root,
                    completion=args.completion,
                ),
            ).as_json_dict()
        else:
            expected_count = _expected_completion_count(
                args.shard_manifest,
                args.reference_completion,
                args.source,
                args.manifest_behavior_sha256,
                args.accept_manifest_behavior_sha256,
            )
            result = validate_rollout_completion(
                args.materialized_inputs,
                args.successes,
                args.completion,
                args.behavior_sha256,
                expected_count,
                _artifact_arguments(
                    args.artifact,
                    layout_root=args.layout_root,
                    completion=args.completion,
                )
                or None,
                accepted_behavior_sha256=args.accept_behavior_sha256,
            ).as_json_dict()
    except (NativeGymShardError, OSError) as exc:
        print(f"native Gym shard error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "JOB_COMPLETION_SCHEMA_VERSION",
    "ROLLOUT_INDEX_KEY",
    "SHARD_MANIFEST_SCHEMA_VERSION",
    "TASK_INDEX_KEY",
    "RolloutCompletion",
    "NativeGymShardError",
    "ShardIdentityError",
    "bind_behavior_manifest",
    "ShardManifest",
    "TailRepairResult",
    "main",
    "publish_rollout_completion",
    "prepare_input_shard",
    "repair_truncated_jsonl_tail",
    "validate_rollout_completion",
]
