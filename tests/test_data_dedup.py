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
"""data.dedup: duplicate rules, chunked resume, fail-closed manifests, and plans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from nvflow.execution import PlanningContext, TargetBinding
from nvflow.stages.data.dedup import DedupStage
from nvflow.stages.data.dedup_worker import (
    DedupSettings,
    chunk_completion,
    compute_signatures,
    find_duplicates,
    main,
)

QUESTION = "What was the total rainfall of the city in summer 2023 according to its weather report?"
SETTINGS: dict[str, Any] = {
    "text_fields": ["q"],
    "id_field": "uuid",
    "near": {"threshold": 0.5, "shingle_size": 3},
    "guard_fields": ["doc"],
    "keep": {"field": "origin", "order": ["curated", "synthetic"]},
    "conflict_field": "answer",
}


def _records() -> list[dict[str, Any]]:
    return [
        {"uuid": "a", "q": QUESTION, "doc": "d1", "origin": "synthetic", "answer": "10"},
        {"uuid": "b", "q": QUESTION.upper()[:-1], "doc": "d1", "origin": "curated", "answer": "10"},
        {
            "uuid": "c",
            "q": QUESTION.replace("its weather", "its official weather"),
            "doc": "d1",
            "origin": "synthetic",
            "answer": "10.1",
        },
        {"uuid": "d", "q": QUESTION, "doc": "d2", "origin": "synthetic", "answer": "7"},
        {"uuid": "e", "q": "", "doc": "d3", "origin": "synthetic", "answer": "1"},
        {
            "uuid": "f",
            "q": "How many employees did the company report at the end of the year?",
            "doc": "d1",
            "origin": "synthetic",
            "answer": "5",
        },
    ]


def _write(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def _ids(path: Path) -> list[str]:
    return [json.loads(line)["uuid"] for line in path.read_text(encoding="utf-8").splitlines()]


def _run(
    step: str, source: Path, output: Path, *extra: str, settings: dict[str, Any] = SETTINGS
) -> int:
    return main(
        [step, str(source), "--output_dir", str(output), "--settings", json.dumps(settings), *extra]
    )


def test_the_near_layer_alone_drops_every_copy_and_near_copy() -> None:
    texts = [QUESTION, QUESTION.upper(), QUESTION + "!", QUESTION.replace("its weather", "its")]
    rows = [(index, {"q": text}) for index, text in enumerate([*texts, "Who audited the firm?"])]
    settings = DedupSettings.from_config(
        {"text_fields": ["q"], "near": {"threshold": 0.5, "shingle_size": 3}, "exact": False}
    )

    decisions, _groups = find_duplicates(compute_signatures(rows, settings), settings)

    assert {row: decision["reason"] for row, decision in decisions.items()} == {
        1: "exact",
        2: "exact",
        3: "near",
    }


def test_an_empty_near_mapping_uses_the_defaults() -> None:
    settings = DedupSettings.from_config({"text_fields": ["q"], "near": {}})

    assert (settings.near_threshold, settings.shingle_size) == (0.8, 5)
    assert DedupSettings.from_config({"text_fields": ["q"]}).near_threshold is None


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({}, "text_fields"),
        ({"text_fields": ["q"], "exact": False}, "enable"),
        ({"text_fields": ["q"], "near": {"threshold": 1.5}}, "threshold"),
        ({"text_fields": ["q"], "near": {"threshold": True}}, "threshold"),
        ({"text_fields": ["q"], "near": {"shingle_size": True}}, "shingle_size"),
        ({"text_fields": ["q"], "near": {"seed": "1"}}, "seed"),
        ({"text_fields": ["q"], "near": {"size": 3}}, "unknown 'near'"),
        ({"text_fields": ["q"], "keep": {"field": "origin"}}, "both"),
        ({"text_fields": ["q"], "on_conflict": "drop_group"}, "conflict_field"),
    ],
)
def test_settings_fail_closed(config: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DedupSettings.from_config(config)


def _dedup(config: dict[str, Any]) -> tuple[list[str], dict[str, Any], list[dict[str, Any]]]:
    records = _records()
    settings = DedupSettings.from_config(config)
    decisions, groups = find_duplicates(compute_signatures(enumerate(records), settings), settings)
    kept = [record["uuid"] for index, record in enumerate(records) if index not in decisions]
    dropped = {records[index]["uuid"]: decision for index, decision in decisions.items()}
    return kept, dropped, groups


def test_exact_and_near_duplicates_group_within_guard_fields() -> None:
    kept, dropped, groups = _dedup(SETTINGS)

    assert kept == ["b", "d", "e", "f"]
    assert dropped["a"] == {"group": 0, "duplicate_of": "b", "reason": "exact"}
    assert dropped["c"]["reason"] == "near"
    assert dropped["c"]["duplicate_of"] == "b"
    assert groups == [
        {
            "group": 0,
            "ids": ["a", "b", "c"],
            "kept": ["b"],
            "conflict": True,
            "conflict_values": ["10", "10.1"],
        }
    ]


def test_conflicts_compare_answers_ignoring_only_case_and_spacing() -> None:
    settings = DedupSettings.from_config({"text_fields": ["q"], "conflict_field": "answer"})
    records = [{"q": QUESTION, "answer": answer} for answer in ("-5%", "5%", " 5%  ")]

    _decisions, [group] = find_duplicates(
        compute_signatures(enumerate(records), settings), settings
    )

    assert group["conflict_values"] == ["-5%", "5%"]


@pytest.mark.parametrize(
    ("policy", "kept"),
    [("drop_group", ["d", "e", "f"]), ("keep_all", ["a", "b", "c", "d", "e", "f"])],
)
def test_conflict_policies(policy: str, kept: list[str]) -> None:
    survivors, _dropped, groups = _dedup({**SETTINGS, "on_conflict": policy})

    assert survivors == kept
    assert groups[0]["conflict"] is True


def test_worker_publishes_outputs_and_skips_complete_work(tmp_path: Path) -> None:
    source = _write(tmp_path / "pool.jsonl", _records())
    output = tmp_path / "dedup"

    assert _run("run", source, output, "--num_chunks", "3") == 0
    assert _ids(output / "kept.jsonl") == ["b", "d", "e", "f"]
    assert _ids(output / "dropped.jsonl") == ["a", "c"]
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["input_rows"] == 6
    assert report["conflict_groups"] == 1
    assert report["dropped_by_reason"] == {"exact": 1, "near": 1}
    assert report["rows_without_text"] == 1
    finished = (output / "kept.jsonl").stat().st_mtime_ns

    assert _run("run", source, output, "--num_chunks", "3") == 0
    assert (output / "kept.jsonl").stat().st_mtime_ns == finished


def test_resume_recomputes_only_the_interrupted_chunk(tmp_path: Path) -> None:
    source = _write(tmp_path / "pool.jsonl", _records())
    clean, resumed = tmp_path / "clean", tmp_path / "resumed"
    assert _run("run", source, clean, "--num_chunks", "3") == 0

    for chunk in (0, 2):
        assert _run("signatures", source, resumed, "--num_chunks", "3", "--chunk", str(chunk)) == 0
    interrupted = resumed / "chunks" / "00001" / "meta.jsonl"
    interrupted.parent.mkdir(parents=True)
    interrupted.write_text('{"row": 2', encoding="utf-8")
    with pytest.raises(ValueError, match=r"incomplete or stale: \[1\]"):
        _run("group", source, resumed, "--num_chunks", "3")

    untouched = (resumed / "chunks" / "00000" / "meta.jsonl").stat().st_mtime_ns
    assert _run("run", source, resumed, "--num_chunks", "3") == 0
    assert (resumed / "chunks" / "00000" / "meta.jsonl").stat().st_mtime_ns == untouched
    for name in ("kept.jsonl", "dropped.jsonl", "groups.jsonl"):
        assert (resumed / name).read_bytes() == (clean / name).read_bytes(), name


def test_changed_inputs_or_settings_fail_closed(tmp_path: Path) -> None:
    source = _write(tmp_path / "pool.jsonl", _records())
    output = tmp_path / "dedup"
    assert _run("run", source, output) == 0

    changed = {**SETTINGS, "near": {"threshold": 0.9, "shingle_size": 3}}
    with pytest.raises(ValueError, match="settings changed: near_threshold"):
        _run("run", source, output, settings=changed)
    _write(source, _records()[:3])
    with pytest.raises(ValueError, match="input files or their contents changed"):
        _run("run", source, output)


class _Filesystem:
    def __init__(self, present: set[str]):
        self.present = present

    def batch_exists(self, paths: list[str]) -> dict[str, bool]:
        return {path: path in self.present for path in paths}


def _context(
    config: dict[str, Any],
    *,
    present: set[str] | None = None,
    run_after: tuple[str, ...] = (),
) -> PlanningContext:
    return PlanningContext(
        stage="dedup",
        config=config,
        launcher_config=config,
        target=TargetBinding.resolve("test"),
        expname="dedup",
        run_after=run_after,
        live=present is not None,
        filesystem=cast(Any, None if present is None else _Filesystem(present)),
        engine=None if present is None else "slurm",
    )


CONFIG = {"input_file": "/data/pool.jsonl", "output_dir": "/out/dedup", **SETTINGS}


def test_one_chunk_plans_one_cpu_task_after_upstream() -> None:
    [task] = DedupStage().plan(_context(CONFIG, run_after=("clean",))).tasks

    assert task.name == "dedup-signatures-group"
    assert task.commands[0].container == "nemo-skills"
    assert task.resources[0].gpus_per_node == 0
    assert [(d.upstream, d.external) for d in task.dependencies] == [("clean", True)]
    assert {output.path for output in task.outputs} >= {
        "/out/dedup/kept.jsonl",
        "/out/dedup/dedup.done",
        "/out/dedup/chunks/00000/signatures.done",
    }


def test_chunks_and_retries_chain_into_the_group_task() -> None:
    tasks = DedupStage().plan(_context({**CONFIG, "num_chunks": 2, "dependent_jobs": 1})).tasks

    assert [task.name for task in tasks] == [
        "dedup-chunk0-signatures",
        "dedup-chunk0-signatures-dep1",
        "dedup-chunk1-signatures",
        "dedup-chunk1-signatures-dep1",
        "dedup-group",
        "dedup-group-dep1",
    ]
    by_name = {task.name: task for task in tasks}
    group = by_name["dedup-group"]
    assert [(d.upstream, d.condition) for d in group.dependencies] == [
        (by_name["dedup-chunk0-signatures-dep1"].task_id, "after_success"),
        (by_name["dedup-chunk1-signatures-dep1"].task_id, "after_success"),
    ]
    retry = by_name["dedup-group-dep1"]
    assert [(d.upstream, d.condition) for d in retry.dependencies] == [(group.task_id, "after_any")]


def test_live_plan_leaves_finished_steps_for_the_worker_to_check() -> None:
    config = {**CONFIG, "num_chunks": 3}
    present = {
        "/data/pool.jsonl",
        "/out/dedup/chunks/00001/signatures.done",
        "/out/dedup/dedup.done",
    }

    names = [task.name for task in DedupStage().plan(_context(config, present=present)).tasks]
    assert names == [
        "dedup-chunk0-signatures",
        "dedup-chunk1-signatures",
        "dedup-chunk2-signatures",
        "dedup-group",
    ]
    with pytest.raises(ValueError, match="/data/pool.jsonl"):
        DedupStage().plan(_context(config, present=set()))


def test_plan_and_worker_agree_on_chunk_completion_paths() -> None:
    tasks = DedupStage().plan(_context({**CONFIG, "num_chunks": 2})).tasks
    planned = {output.path for task in tasks for output in task.outputs}

    for chunk in range(2):
        assert str(chunk_completion(Path("/out/dedup"), chunk)) in planned


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"input_file": "/a.jsonl", **SETTINGS}, "output_dir"),
        ({"output_dir": "/o", **SETTINGS}, "input_file"),
        ({**CONFIG, "input_files": ["/b.jsonl"]}, "exactly one"),
        ({**CONFIG, "num_chunks": 0}, "num_chunks"),
        ({**CONFIG, "dependent_jobs": -1}, "dependent_jobs"),
        ({**CONFIG, "neer": {"threshold": 0.9}}, "unknown data.dedup option"),
    ],
)
def test_stage_config_fails_closed(config: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DedupStage().validate_config(config)
