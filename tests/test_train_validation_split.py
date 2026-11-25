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
"""Train/validation split stages, the generic splitter, and its completion marker."""

from __future__ import annotations

import json
import random
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from nvflow.execution import PlanningContext, TargetBinding
from nvflow.stages.data.dataset_splitter import perform_split
from nvflow.stages.data.split_contract import (
    publish_split_marker,
    validate_named_split_marker,
    validate_split_marker,
)
from nvflow.stages.data.train_validation_split import (
    TrainValidationSplitStage,
    split_command,
    validate_split_options,
)
from nvflow.stages.nemo_gym.train_validation_split import (
    TrainValidationSplitStage as GymTrainValidationSplitStage,
)

GOLDEN = Path(__file__).parent / "fixtures" / "train_validation_split"
GOLDEN_CASES = {
    "sft": {
        "stratify_by": "question_type",
        "random_seed": 42,
        "fields": ["input", "output", "uuid", "total_token_length"],
        "max_token_length": 120,
        "token_length_field": "total_token_length",
    },
    "grpo": {"stratify_by": "question_type", "random_seed": 42},
    "curriculum": {
        "stratify_by": "question_type",
        "random_seed": 7,
        "sort_by": "difficulty_profile.avg_reward",
        "sort_order": "desc",
    },
    "curriculum_asc": {
        "stratify_by": "question_type",
        "random_seed": 3,
        "sort_by": "difficulty_profile.avg_reward",
        "sort_order": "asc",
    },
}


class _Filesystem:
    def __init__(self, present: set[str]):
        self.present = present

    def batch_exists(self, paths: list[str]) -> dict[str, bool]:
        return {path: path in self.present for path in paths}


def _run_split(
    tmp_path: Path,
    config: dict[str, Any],
    *,
    source: Path = GOLDEN / "input.jsonl",
) -> Path:
    output = tmp_path / "split"
    command = split_command(
        input_file=str(source),
        output_dir=str(output),
        config={"val_ratio": 0.2, **config},
    )
    command = command.replace("python -m", f"{shlex.quote(sys.executable)} -m", 1)
    subprocess.run(["bash", "-c", command], check=True, capture_output=True)
    return output


@pytest.mark.parametrize("case", sorted(GOLDEN_CASES))
def test_split_reproduces_golden_outputs(tmp_path: Path, case: str) -> None:
    output = _run_split(tmp_path, GOLDEN_CASES[case])

    for name in ("train.jsonl", "val.jsonl", "split.done"):
        assert (output / name).read_bytes() == (GOLDEN / case / name).read_bytes(), name


def test_unset_stratify_by_splits_one_stratum(tmp_path: Path) -> None:
    output = _run_split(tmp_path, {"val_ratio": 0.15})

    validate_split_marker(output)
    assert (output / "val.jsonl").read_text(encoding="utf-8").count("\n") == 6
    assert (output / "train.jsonl").read_text(encoding="utf-8").count("\n") == 34


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"max_token_length": 10}, "token_length_field"),
        ({"fields": []}, "fields"),
        ({"fields": "input"}, "fields"),
        ({"sort_order": "random"}, "sort_order"),
        ({"val_ratio": 1.5}, "val_ratio"),
        ({"val_ratio": True}, "val_ratio"),
        ({"random_seed": "42"}, "random_seed"),
        ({"max_token_length": True, "token_length_field": "n"}, "max_token_length"),
        ({"id_field": "uuid"}, "only with 'splits'"),
    ],
)
def test_split_options_fail_closed(config: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_split_options(config)


def test_token_filter_without_length_field_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="token_length_field"):
        perform_split(GOLDEN / "input.jsonl", tmp_path, None, 0.2, 42, max_tokens=120)


def test_split_marker_detects_tampered_output(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text('{"id":1}\n', encoding="utf-8")
    (tmp_path / "val.jsonl").write_text('{"id":2}\n', encoding="utf-8")
    marker = publish_split_marker(tmp_path)

    assert validate_split_marker(tmp_path)["status"] == "complete"
    assert marker.name == "split.done"

    (tmp_path / "train.jsonl").write_text('{"id":3}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        validate_split_marker(tmp_path)


def test_splitter_cli_publishes_completion_marker(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "output"
    source.write_text(
        "".join(f'{{"uuid":"{index}","question_type":"type-a"}}\n' for index in range(4)),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "nvflow.stages.data.dataset_splitter",
            str(source),
            "--output_dir",
            str(output),
            "--val_ratio",
            "0.25",
        ],
        check=True,
    )

    assert validate_split_marker(output)["status"] == "complete"


def test_split_check_ignores_named_splits_unless_asked(tmp_path: Path) -> None:
    rows = [{"uuid": f"u{index}"} for index in range(4)]
    output = _run_split(
        tmp_path,
        {"splits": {"a": 0.5, "b": 0.5}},
        source=_write_rows(tmp_path / "rows.jsonl", rows),
    )
    (output / "train.jsonl").write_text('{"partial": ', encoding="utf-8")
    check = [sys.executable, "-m", "nvflow.stages.data.split_contract", "--output-dir", str(output)]

    assert subprocess.run(check, capture_output=True).returncode != 0
    assert subprocess.run([*check, "--named"], capture_output=True).returncode == 0


def _context(
    config: dict[str, Any],
    *,
    live: bool = False,
    filesystem: _Filesystem | None = None,
    run_after: tuple[str, ...] = (),
) -> PlanningContext:
    return PlanningContext(
        stage="train-validation-split",
        config=config,
        launcher_config=config,
        target=TargetBinding.resolve("test"),
        expname="split",
        run_after=run_after,
        live=live,
        filesystem=cast(Any, filesystem),
    )


def test_generic_split_plan_renders_only_configured_options() -> None:
    config = {"input_file": "/data/all.jsonl", "output_dir": "/split"}
    [task] = TrainValidationSplitStage().plan(_context(config)).tasks

    command = task.commands[0].command
    assert "-m nvflow.stages.data.dataset_splitter" in command
    assert "--stratify_by" not in command
    assert "--fields" not in command
    assert task.parameters.get("sbatch_kwargs") is None


def test_live_generic_split_requires_input_without_upstream() -> None:
    config = {"input_file": "/data/all.jsonl", "output_dir": "/split"}
    with pytest.raises(ValueError, match="/data/all.jsonl"):
        TrainValidationSplitStage().plan(_context(config, live=True, filesystem=_Filesystem(set())))


def _gym_context(
    *,
    live: bool,
    filesystem: _Filesystem | None,
    run_after: tuple[str, ...] = (),
) -> PlanningContext:
    return _context(
        {
            "input_dir": "/outputs/profile",
            "output_dir": "/outputs/split",
            "input_filename": "train.jsonl",
            "input_contract": "dataset_selection",
            "environments": {
                "finance": {
                    "raw_train_data": "/unused/source.jsonl",
                }
            },
        },
        live=live,
        filesystem=filesystem,
        run_after=run_after,
    )


def test_gym_split_requires_environments() -> None:
    with pytest.raises(ValueError, match="'environments' is required"):
        GymTrainValidationSplitStage().validate_config(
            {"input_dir": "/outputs/profile", "output_dir": "/outputs/split"}
        )


def test_dependencies_are_scoped_to_each_environment() -> None:
    dependencies = [
        "collect-rollouts-finance",
        "collect-rollouts-math",
        "shared-setup",
    ]

    assert GymTrainValidationSplitStage._scope_dependencies(
        dependencies,
        "finance",
        ("finance", "math"),
    ) == [
        "collect-rollouts-finance",
        "shared-setup",
    ]


def test_static_plan_does_not_inspect_input_files() -> None:
    plan = GymTrainValidationSplitStage().plan(_gym_context(live=False, filesystem=None))

    assert len(plan.tasks) == 1
    assert "workers.validate_dataset_selection" in plan.tasks[0].commands[0].command
    assert "-m nvflow.stages.data.dataset_splitter" in plan.tasks[0].commands[0].command
    assert plan.tasks[0].parameters["sbatch_kwargs"] == {"dependency_type": "afterany"}


def test_live_standalone_plan_requires_committed_input() -> None:
    with pytest.raises(ValueError, match="job-2-build-training-dataset"):
        GymTrainValidationSplitStage().plan(_gym_context(live=True, filesystem=_Filesystem(set())))


def test_live_full_workflow_plan_defers_to_planned_upstream() -> None:
    plan = GymTrainValidationSplitStage().plan(
        _gym_context(
            live=True,
            filesystem=_Filesystem(set()),
            run_after=("collect-rollouts",),
        )
    )

    assert len(plan.tasks) == 1


def _grouped_source(tmp_path: Path) -> Path:
    source = tmp_path / "grouped.jsonl"
    source.write_text(
        "".join(
            json.dumps(
                {
                    "uuid": f"u{index}",
                    "document": f"doc{index // 4}",
                    "kind": "ab"[index % 2],
                    "origin": "xyz"[index % 3],
                }
            )
            + "\n"
            for index in range(240)
        ),
        encoding="utf-8",
    )
    return source


def _split_rows(output: Path, name: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (output / f"{name}.jsonl").read_text().splitlines()]


def test_named_splits_keep_groups_whole_and_publish_a_manifest(tmp_path: Path) -> None:
    output = _run_split(
        tmp_path,
        {
            "splits": {"held_out": 0.1, "pool_a": 0.45, "pool_b": 0.45},
            "group_by": "document",
            "stratify_by": ["kind"],
            "id_field": "uuid",
        },
        source=_grouped_source(tmp_path),
    )

    rows = {name: _split_rows(output, name) for name in ("held_out", "pool_a", "pool_b")}
    assert sum(len(split) for split in rows.values()) == 240
    assert 18 <= len(rows["held_out"]) <= 30
    homes: dict[str, set[str]] = {}
    for name, split in rows.items():
        for row in split:
            homes.setdefault(row["document"], set()).add(name)
    assert all(len(names) == 1 for names in homes.values())
    manifest = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert len(manifest) == 240
    assert {entry["id"]: entry["split"] for entry in manifest}["u0"] in rows
    splits = validate_named_split_marker(output)["splits"]
    assert isinstance(splits, dict)
    assert sorted(splits) == ["held_out", "pool_a", "pool_b"]

    (output / "pool_a.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        validate_named_split_marker(output)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_named_splits_ignore_the_input_order(tmp_path: Path) -> None:
    rows = [{"uuid": f"u{index}", "kind": "abc"[index % 3]} for index in range(300)]
    shuffled = random.Random(0).sample(rows, len(rows))
    config = {
        "splits": {"held_out": 0.1, "pool_a": 0.45, "pool_b": 0.45},
        "stratify_by": ["kind"],
        "id_field": "uuid",
    }
    homes = []
    for index, records in enumerate((rows, shuffled)):
        source = _write_rows(tmp_path / f"input-{index}.jsonl", records)
        output = _run_split(tmp_path / str(index), config, source=source)
        homes.append(
            {row["uuid"]: name for name in config["splits"] for row in _split_rows(output, name)}
        )

    assert len(homes[0]) == 300
    assert homes[0] == homes[1]


def test_named_splits_keep_their_ratios_across_small_strata(tmp_path: Path) -> None:
    rows = [{"uuid": f"u{index}", "kind": f"k{index // 10}"} for index in range(500)]
    output = _run_split(
        tmp_path,
        {
            "splits": {"held_out": 0.05, "pool_a": 0.475, "pool_b": 0.475},
            "stratify_by": ["kind"],
            "id_field": "uuid",
        },
        source=_write_rows(tmp_path / "small-strata.jsonl", rows),
    )

    sizes = {name: len(_split_rows(output, name)) for name in ("held_out", "pool_a", "pool_b")}
    assert sizes["held_out"] == 25
    assert abs(sizes["pool_a"] - sizes["pool_b"]) <= 1


def test_group_by_and_combined_strata_still_write_train_and_validation(tmp_path: Path) -> None:
    output = _run_split(
        tmp_path,
        {"val_ratio": 0.25, "group_by": "document", "stratify_by": ["kind", "origin"]},
        source=_grouped_source(tmp_path),
    )

    validate_split_marker(output)
    train, val = _split_rows(output, "train"), _split_rows(output, "val")
    assert not {row["document"] for row in train} & {row["document"] for row in val}
    assert 48 <= len(val) <= 72


def test_split_command_renders_new_options_only_when_set() -> None:
    plain = split_command(input_file="/in.jsonl", output_dir="/out", config={"stratify_by": "kind"})
    named = split_command(
        input_file="/in.jsonl",
        output_dir="/out",
        config={
            "splits": {"a": 0.5, "b": 0.5},
            "group_by": "document",
            "stratify_by": ["kind", "origin"],
            "id_field": "uuid",
        },
    )

    assert "--stratify_by 'kind' " in plain
    assert "--group_by" not in plain and "--splits" not in plain
    assert "--val_ratio" not in named
    assert "--stratify_by 'kind' 'origin'" in named
    assert named.endswith(" --group_by 'document' --splits 'a=0.5' 'b=0.5' --id_field 'uuid'")


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"splits": {"a": 1.0}}, "at least two"),
        ({"splits": {"a": 0.5, "b": 0.4}}, "sum to 1"),
        ({"splits": {"a": float("nan"), "b": 0.5}}, "non-negative ratio"),
        ({"splits": {"manifest": 0.5, "b": 0.5}}, "invalid split name"),
        ({"splits": {"a/b": 0.5, "b": 0.5}}, "invalid split name"),
        ({"splits": {"a": 0.5, "b": 0.5}, "val_ratio": 0.1}, "not both"),
        ({"splits": {"a": 0.5, "b": 0.5}, "sort_by": "score"}, "sort_by"),
        ({"stratify_by": []}, "stratify_by"),
        ({"group_by": ""}, "group_by"),
    ],
)
def test_named_split_options_fail_closed(config: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_split_options(config)


def test_named_split_plan_declares_every_output() -> None:
    config = {
        "input_file": "/data/all.jsonl",
        "output_dir": "/split",
        "splits": {"held_out": 0.1, "pool": 0.9},
        "container": "nemo-gym",
        "installation_command": "export PATH=/opt/venv/bin:$PATH",
    }
    [task] = TrainValidationSplitStage().plan(_context(config)).tasks

    assert task.commands[0].container == "nemo-gym"
    assert task.commands[0].command.startswith("set -e\nexport PATH=/opt/venv/bin:$PATH\n")
    assert {output.name: output.path for output in task.outputs} == {
        "held_out": "/split/held_out.jsonl",
        "pool": "/split/pool.jsonl",
        "manifest": "/split/manifest.jsonl",
        "completion": "/split/splits.done",
    }


def test_gym_split_rejects_named_splits() -> None:
    with pytest.raises(ValueError, match="'splits' is not supported"):
        GymTrainValidationSplitStage().validate_config(
            {
                "environments": {"example": {}},
                "input_dir": "/in",
                "output_dir": "/out",
                "splits": {"a": 0.5, "b": 0.5},
            }
        )
