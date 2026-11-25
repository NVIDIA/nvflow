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
"""Durable append-only records for asynchronous workflow controllers."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


@dataclass(frozen=True, slots=True)
class RunEvent:
    """One immutable run/task/attempt state transition."""

    sequence: int
    timestamp: str
    run_id: str
    kind: str
    state: str
    task_id: str | None = None
    attempt: int | None = None
    cluster: str | None = None
    submission_id: str | None = None
    message: str | None = None


class RunJournal:
    """Persist graph identity and append-only events below the artifact root."""

    def __init__(self, artifact_root: str | Path, run_id: str) -> None:
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError(f"Invalid run_id: {run_id!r}")
        root = Path(artifact_root)
        if not root.is_absolute():
            raise ValueError("Run journal artifact_root must be absolute")
        self.run_id = run_id
        self.root = root / ".nvflow" / "runs" / run_id
        self.events_path = self.root / "events.jsonl"
        self.state_path = self.root / "state.json"
        self.graph_path = self.root / "execution.json"
        self._repair_partial_tail()
        self._sequence = self._read_last_sequence()

    def _repair_partial_tail(self) -> None:
        """Discard only an interrupted final write; preserve complete JSON."""
        if not self.events_path.exists():
            return
        data = self.events_path.read_bytes()
        repaired = _without_partial_tail(data)
        if repaired == data:
            return
        with self.events_path.open("wb") as stream:
            stream.write(repaired)
            stream.flush()
            os.fsync(stream.fileno())

    def initialize(self, graph: dict[str, Any]) -> None:
        """Create the immutable graph record or validate an existing one."""
        self.root.mkdir(parents=True, exist_ok=True)
        encoded = _encoded(graph)
        if self.graph_path.exists():
            if self.graph_path.read_bytes() != encoded:
                raise ValueError(f"Run {self.run_id!r} execution graph identity changed")
            return
        _atomic_write(self.graph_path, encoded)

    def append(
        self,
        *,
        kind: str,
        state: str,
        task_id: str | None = None,
        attempt: int | None = None,
        cluster: str | None = None,
        submission_id: str | None = None,
        message: str | None = None,
    ) -> RunEvent:
        """Append one event and update the terminal/current snapshot."""
        self.root.mkdir(parents=True, exist_ok=True)
        self._sequence += 1
        event = RunEvent(
            sequence=self._sequence,
            timestamp=datetime.now(UTC).isoformat(),
            run_id=self.run_id,
            kind=kind,
            state=state,
            task_id=task_id,
            attempt=attempt,
            cluster=cluster,
            submission_id=submission_id,
            message=message,
        )
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(event), sort_keys=True, separators=(",", ":")))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _atomic_write(self.state_path, _encoded(asdict(event)))
        return event

    def events(self) -> tuple[RunEvent, ...]:
        """Read every valid persisted event in sequence order."""
        if not self.events_path.exists():
            return ()
        return _parse_events(
            self.events_path.read_text(encoding="utf-8"),
            str(self.events_path),
        )

    def _read_last_sequence(self) -> int:
        if not self.events_path.exists():
            return 0
        events = self.events()
        return events[-1].sequence if events else 0


def run_events_path(artifact_root: str, run_id: str) -> PurePosixPath:
    """Return the events file that RunJournal keeps for one run."""
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError(f"Invalid run_id: {run_id!r}")
    return PurePosixPath(artifact_root) / ".nvflow" / "runs" / run_id / "events.jsonl"


def parse_events(text: str, source: str) -> tuple[RunEvent, ...]:
    """Parse events as RunJournal reads them, without repairing the file.

    An interrupted final write is dropped, exactly as RunJournal repairs it.
    """
    repaired = _without_partial_tail(text.encode("utf-8")).decode("utf-8")
    return _parse_events(repaired, source)


def _without_partial_tail(data: bytes) -> bytes:
    """Drop only an interrupted final write; preserve complete JSON."""
    if not data or data.endswith(b"\n"):
        return data
    boundary = data.rfind(b"\n") + 1
    try:
        RunEvent(**json.loads(data[boundary:].decode("utf-8")))
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError):
        return data[:boundary]
    return data + b"\n"


def _parse_events(text: str, source: str) -> tuple[RunEvent, ...]:
    events = []
    for line_number, line in enumerate(text.splitlines(), 1):
        try:
            value = json.loads(line)
            event = RunEvent(**value)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid run event {source}:{line_number}") from exc
        events.append(event)
    if [event.sequence for event in events] != list(range(1, len(events) + 1)):
        raise ValueError(f"Run event sequence is not contiguous: {source}")
    return tuple(events)


def _encoded(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


__all__ = ["RunEvent", "RunJournal", "parse_events", "run_events_path"]
