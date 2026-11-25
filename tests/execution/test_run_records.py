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
"""Durability contracts for asynchronous execution records."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nvflow.execution.run_records import RunJournal, parse_events, run_events_path


def test_run_journal_persists_graph_and_contiguous_events(tmp_path: Path) -> None:
    journal = RunJournal(tmp_path, "run-123")
    graph = {"digest": "sha256:graph", "tasks": ["first"]}

    journal.initialize(graph)
    journal.append(kind="run", state="running")
    journal.append(
        kind="task",
        state="submitted",
        task_id="first",
        attempt=1,
        cluster="skills",
        submission_id="run-123-first-a1",
    )

    resumed = RunJournal(tmp_path, "run-123")
    events = resumed.events()
    assert [event.sequence for event in events] == [1, 2]
    assert events[-1].submission_id == "run-123-first-a1"
    assert json.loads(resumed.graph_path.read_text()) == graph
    assert json.loads(resumed.state_path.read_text())["sequence"] == 2


def test_run_journal_rejects_changed_graph(tmp_path: Path) -> None:
    journal = RunJournal(tmp_path, "run")
    journal.initialize({"digest": "first"})

    with pytest.raises(ValueError, match="identity changed"):
        journal.initialize({"digest": "second"})


def test_run_journal_recovers_an_interrupted_final_event(tmp_path: Path) -> None:
    journal = RunJournal(tmp_path, "recover")
    journal.append(kind="run", state="running")
    with journal.events_path.open("ab") as stream:
        stream.write(b'{"sequence":2,"state":"sub')

    resumed = RunJournal(tmp_path, "recover")
    resumed.append(kind="task", state="submitted", task_id="next")

    assert [event.sequence for event in resumed.events()] == [1, 2]
    assert resumed.events()[-1].task_id == "next"


def test_parse_events_reads_like_run_journal_without_repairing_the_file(
    tmp_path: Path,
) -> None:
    journal = RunJournal(tmp_path, "remote")
    journal.append(kind="run", state="running")
    path = run_events_path(str(tmp_path), "remote")
    assert Path(path) == journal.events_path
    with journal.events_path.open("ab") as stream:
        stream.write(b'{"sequence":2,"state":"sub')
    interrupted = journal.events_path.read_text(encoding="utf-8")

    events = parse_events(interrupted, str(path))

    assert [event.state for event in events] == ["running"]
    assert journal.events_path.read_text(encoding="utf-8") == interrupted


@pytest.mark.parametrize("run_id", ["", "../escape", "spaces are unsafe"])
def test_run_journal_rejects_unsafe_identity(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError, match="Invalid run_id"):
        RunJournal(tmp_path, run_id)
