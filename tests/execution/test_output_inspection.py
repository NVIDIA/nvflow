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
"""Recipe-independent output-presence contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from nvflow.execution import (
    ArtifactDeclaration,
    DependencyPlan,
    LaunchPlan,
    StagePlan,
    TaskPlan,
)
from nvflow.execution.filesystem import LocalTargetFilesystem
from nvflow.execution.inspection import inspect_declared_outputs
from nvflow.integrations.nemo_skills.filesystem import LazyTargetFilesystem


class RecordingFilesystem:
    remote = False

    def __init__(self, existing: set[str]) -> None:
        self.existing = existing
        self.batches: list[list[str]] = []

    def exists(self, path: str) -> bool:
        return path in self.existing

    def batch_exists(self, paths: list[str]) -> dict[str, bool]:
        self.batches.append(paths)
        return {path: path in self.existing for path in paths}

    def is_nonempty_file(self, path: str) -> bool:
        return path in self.existing

    def ls(self, directory: str, pattern: str) -> list[str]:
        return []

    def count_lines(self, path: str) -> int:
        return 0


def test_lazy_target_filesystem_defers_factory_until_first_use() -> None:
    created = 0
    backing = RecordingFilesystem(existing={"/output"})

    def factory():
        nonlocal created
        created += 1
        return backing

    filesystem = LazyTargetFilesystem(cast(Any, factory))

    assert created == 0
    assert filesystem.batch_exists(["/output"]) == {"/output": True}
    assert created == 1
    assert filesystem.exists("/output") is True
    assert created == 1


def test_artifact_declaration_preserves_custom_kind() -> None:
    artifact = ArtifactDeclaration(name="dataset", path="/output", kind="dataset")

    assert artifact.kind == "dataset"


def _task(stage: str, path: str | None) -> TaskPlan:
    outputs = (
        (ArtifactDeclaration(name="custom", path=path, kind="directory"),)
        if path is not None
        else ()
    )
    return TaskPlan(
        task_id=f"{stage}.task",
        stage=stage,
        name=stage,
        operation="custom",
        outputs=outputs,
    )


def test_output_presence_depends_only_on_artifact_declarations() -> None:
    plan = LaunchPlan(
        plan_version=1,
        project="unfamiliar-external-project",
        workflow="arbitrary-workflow-name",
        target="target",
        stages=(
            StagePlan(
                stage="novel-stage",
                tasks=(_task("novel-stage", "/custom/location/result"),),
            ),
            StagePlan(
                stage="missing-stage",
                tasks=(_task("missing-stage", "/another/custom/location"),),
            ),
            StagePlan(
                stage="no-output-stage",
                tasks=(_task("no-output-stage", None),),
            ),
            StagePlan.opaque("legacy-stage", "execute-only compatibility"),
        ),
        planning_mode="live",
    )
    filesystem = RecordingFilesystem({"/custom/location/result"})

    inspections = inspect_declared_outputs(plan, filesystem)

    assert [(inspection.stage, inspection.presence) for inspection in inspections] == [
        ("novel-stage", "exists"),
        ("missing-stage", "missing"),
        ("no-output-stage", "not_declared"),
        ("legacy-stage", "unavailable"),
    ]
    assert filesystem.batches == [["/another/custom/location", "/custom/location/result"]]


def test_declared_outputs_are_unavailable_without_target_filesystem() -> None:
    plan = LaunchPlan(
        plan_version=1,
        project="external",
        workflow="custom",
        target="target",
        stages=(
            StagePlan(
                stage="custom-stage",
                tasks=(_task("custom-stage", "/custom/output"),),
            ),
        ),
        planning_mode="live",
    )

    inspection = inspect_declared_outputs(plan, None)[0]

    assert inspection.presence == "unavailable"


def test_outputs_hidden_from_this_machine_are_unavailable_instead_of_missing() -> None:
    plan = LaunchPlan(
        plan_version=1,
        project="external",
        workflow="custom",
        target="target",
        stages=(
            StagePlan(stage="hidden", tasks=(_task("hidden", "/workspace/out"),)),
            StagePlan(stage="visible", tasks=(_task("visible", "/shared/out"),)),
        ),
        planning_mode="live",
    )
    filesystem = RecordingFilesystem(set())

    inspections = inspect_declared_outputs(
        plan,
        filesystem,
        visible=lambda path: not path.startswith("/workspace/"),
    )

    assert [(inspection.stage, inspection.presence) for inspection in inspections] == [
        ("hidden", "unavailable"),
        ("visible", "missing"),
    ]
    assert inspections[0].hidden_paths == ("/workspace/out",)
    assert filesystem.batches == [["/shared/out"]]


def test_local_filesystem_mounts_only_paths_whose_top_level_directory_exists(
    tmp_path: Path,
) -> None:
    filesystem = LocalTargetFilesystem()

    assert filesystem.mounted(str(tmp_path / "not-created" / "output"))
    assert not filesystem.mounted("/nvflow-test-unmounted-root/outputs/result")
    assert filesystem.mounted("relative/output")


def test_stage_presence_reports_activity_when_only_one_declared_output_exists() -> None:
    task = TaskPlan(
        task_id="multi-output.task",
        stage="multi-output",
        name="multi-output",
        operation="custom",
        outputs=(
            ArtifactDeclaration(name="intermediate", path="/output/chunk.jsonl"),
            ArtifactDeclaration(name="final", path="/output/final.jsonl"),
        ),
    )
    plan = LaunchPlan(
        plan_version=1,
        project="external",
        workflow="custom",
        target="target",
        stages=(StagePlan(stage="multi-output", tasks=(task,)),),
        planning_mode="live",
    )

    inspection = inspect_declared_outputs(
        plan,
        RecordingFilesystem({"/output/chunk.jsonl"}),
    )[0]

    assert inspection.presence == "exists"
    assert inspection.existing_paths == ("/output/chunk.jsonl",)


def test_stage_presence_ignores_outputs_of_tasks_before_the_final_ones() -> None:
    shared_data = TaskPlan(
        task_id="eval.data",
        stage="eval",
        name="data",
        operation="prepare",
        outputs=(ArtifactDeclaration(name="shared", path="/shared/eval-data.jsonl"),),
    )
    evaluation = TaskPlan(
        task_id="eval.run",
        stage="eval",
        name="run",
        operation="evaluate",
        dependencies=(DependencyPlan("eval.data"),),
        outputs=(ArtifactDeclaration(name="results", path="/run/results"),),
    )
    plan = LaunchPlan(
        plan_version=1,
        project="external",
        workflow="custom",
        target="target",
        stages=(StagePlan(stage="eval", tasks=(shared_data, evaluation)),),
        planning_mode="live",
    )

    inspection = inspect_declared_outputs(
        plan,
        RecordingFilesystem({"/shared/eval-data.jsonl"}),
    )[0]

    assert inspection.presence == "missing"
    assert inspection.declared_paths == ("/run/results",)
