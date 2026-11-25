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
"""data.filter: drop records by the ids in another file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from nvflow.execution import PlanningContext, TargetBinding
from nvflow.stages.data.filter_rows import FilterRowsStage
from nvflow.stages.data.filter_rows_worker import main


def _write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _ids(path: Path) -> list[str]:
    return [json.loads(line)["uuid"] for line in path.read_text(encoding="utf-8").splitlines()]


def test_rows_whose_id_is_excluded_are_dropped(tmp_path: Path) -> None:
    source = _write(tmp_path / "pool.jsonl", [{"uuid": "a"}, {"uuid": "b"}, {"uuid": "c"}])
    exclude = _write(tmp_path / "failed.jsonl", [{"id": "b", "reference": "x"}, {"id": "b"}])
    output = tmp_path / "filtered"

    arguments = [str(source), "--exclude_file", str(exclude), "--output_dir", str(output)]
    assert main([*arguments, "--id_field", "uuid", "--exclude_id_field", "id"]) == 0

    assert _ids(output / "kept.jsonl") == ["a", "c"]
    assert _ids(output / "dropped.jsonl") == ["b"]
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report == {
        "input_rows": 3,
        "kept_rows": 2,
        "dropped_rows": 1,
        "exclude_ids": 1,
        "unmatched_exclude_ids": 0,
    }


@pytest.mark.parametrize(
    ("pool", "failed", "message"),
    [
        ([{"uuid": "a"}], [{"pair": "b"}], "failed.jsonl: row 1 has no 'id'"),
        ([{"uid": "a"}], [{"id": "b"}], "pool.jsonl: row 1 has no 'uuid'"),
    ],
)
def test_a_row_without_its_id_fails_the_run(
    tmp_path: Path, pool: list[dict[str, Any]], failed: list[dict[str, Any]], message: str
) -> None:
    source, exclude = (
        _write(tmp_path / "pool.jsonl", pool),
        _write(tmp_path / "failed.jsonl", failed),
    )
    arguments = [str(source), "--exclude_file", str(exclude), "--output_dir", str(tmp_path / "out")]

    with pytest.raises(ValueError, match=message):
        main([*arguments, "--id_field", "uuid", "--exclude_id_field", "id"])
    assert not (tmp_path / "out" / "kept.jsonl").exists()


CONFIG = {
    "input_file": "/data/pool.jsonl",
    "exclude_file": "/data/failed.jsonl",
    "output_dir": "/out/filtered",
    "id_field": "uuid",
    "exclude_id_field": "id",
}


def _context(config: dict[str, Any], present: set[str] | None = None) -> PlanningContext:
    class Filesystem:
        def batch_exists(self, paths: list[str]) -> dict[str, bool]:
            return {path: path in (present or set()) for path in paths}

    return PlanningContext(
        stage="filter",
        config=config,
        launcher_config=config,
        target=TargetBinding.resolve("test"),
        expname="filter",
        run_after=(),
        live=present is not None,
        filesystem=cast(Any, None if present is None else Filesystem()),
    )


def test_plan_runs_the_worker_on_one_cpu_task() -> None:
    FilterRowsStage().validate_config(CONFIG)
    setup = "export PATH=/opt/venv/bin:$PATH"
    config = {**CONFIG, "container": "nemo-gym", "installation_command": setup}
    [task] = FilterRowsStage().plan(_context(config)).tasks

    command = task.commands[0].command
    assert command.startswith(f"set -e\n{setup}\n")
    assert "-m nvflow.stages.data.filter_rows_worker /data/pool.jsonl" in command
    assert "--exclude_id_field id" in command
    assert task.commands[0].container == "nemo-gym"
    assert task.resources[0].gpus_per_node == 0
    assert {output.path for output in task.outputs} == {
        "/out/filtered/kept.jsonl",
        "/out/filtered/dropped.jsonl",
        "/out/filtered/report.json",
    }


def test_live_plan_requires_both_files() -> None:
    with pytest.raises(ValueError, match="/data/failed.jsonl"):
        FilterRowsStage().plan(_context(CONFIG, present={"/data/pool.jsonl"}))


def test_a_misspelled_option_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown data.filter option.*exclude_ids_field"):
        FilterRowsStage().validate_config({**CONFIG, "exclude_ids_field": "id"})


@pytest.mark.parametrize("missing", ["input_file", "exclude_file", "output_dir", "id_field"])
def test_stage_config_fails_closed(missing: str) -> None:
    with pytest.raises(ValueError, match=missing):
        FilterRowsStage().validate_config(
            {key: value for key, value in CONFIG.items() if key != missing}
        )
