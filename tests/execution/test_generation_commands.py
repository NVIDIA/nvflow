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
"""Executable regressions for generated NeMo-Skills commands."""

from __future__ import annotations

import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

import pytest

from nvflow.execution import (
    CapturedExecutionStage,
    DependencyPlan,
    PlanningContext,
    TargetBinding,
)
from nvflow.execution.operations import _generation_client_command, plan_generate, plan_run_cmd


def _context(stage: str) -> PlanningContext:
    return PlanningContext(
        stage=stage,
        config={},
        launcher_config={},
        target=TargetBinding.resolve("cluster"),
        expname=stage,
        run_after=(),
    )


def _stub_python(tmp_path: Path) -> Path:
    executable = tmp_path / "bin" / "python"
    executable.parent.mkdir()
    executable.write_text(
        """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

if os.environ.get("FAIL_GENERATE"):
    raise SystemExit(17)

options = {}
for argument in sys.argv:
    if argument.startswith("++") and "=" in argument:
        key, value = argument[2:].split("=", 1)
        options[key] = value
output = Path(options["output_file"])
chunk = options.get("chunk_id")
if chunk is not None:
    output = output.with_name(f"{output.stem}_chunk_{chunk}{output.suffix}")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text('{"generated": true}\\n', encoding="utf-8")
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def _chunk_command(
    tmp_path: Path,
    *,
    chunk: int | None = 0,
    postprocess_cmd: str | None = None,
) -> tuple[str, Path]:
    command, output = _generation_client_command(
        output_dir=str(tmp_path / "outputs"),
        input_file=str(tmp_path / "input.jsonl"),
        input_dir=None,
        arguments="",
        seed=0,
        chunk=chunk,
        num_chunks=None if chunk is None else 2,
        preprocess_cmd=None,
        postprocess_cmd=postprocess_cmd,
        setup_command=None,
        server_addresses=[],
        models=[],
        server_types=[],
    )
    return command, Path(output)


def test_chunk_command_tracks_the_filename_written_by_nemo_skills(tmp_path: Path) -> None:
    executable = _stub_python(tmp_path)
    command, output = _chunk_command(tmp_path)
    environment = {
        **os.environ,
        "PATH": f"{executable.parent}:{os.environ['PATH']}",
    }

    result = subprocess.run(["bash", "-c", command], env=environment, check=False)

    assert result.returncode == 0
    assert output.name == "output-rs0_chunk_0.jsonl"
    assert output.read_text(encoding="utf-8") == '{"generated": true}\n'
    assert Path(f"{output}.done").is_file()
    assert "++output_file=" + str(tmp_path / "outputs" / "output-rs0.jsonl") in command
    resumed = subprocess.run(
        ["bash", "-c", command],
        env={**environment, "FAIL_GENERATE": "1"},
        check=False,
    )
    assert resumed.returncode == 0


def test_failed_generation_removes_stale_marker_and_stays_failed(tmp_path: Path) -> None:
    executable = _stub_python(tmp_path)
    command, output = _chunk_command(tmp_path)
    marker = Path(f"{output}.done")
    marker.parent.mkdir(parents=True)
    marker.touch()
    environment = {
        **os.environ,
        "FAIL_GENERATE": "1",
        "PATH": f"{executable.parent}:{os.environ['PATH']}",
    }

    result = subprocess.run(["bash", "-c", command], env=environment, check=False)

    assert result.returncode == 17
    assert not marker.exists()


def test_merged_output_completes_its_chunks(tmp_path: Path) -> None:
    executable = _stub_python(tmp_path)
    command, _output = _chunk_command(tmp_path)
    merged = tmp_path / "outputs" / "output-rs0.jsonl"
    merged.parent.mkdir(parents=True)
    merged.write_text('{"generated": true}\n', encoding="utf-8")
    Path(f"{merged}.done").touch()
    environment = {
        **os.environ,
        "FAIL_GENERATE": "1",
        "PATH": f"{executable.parent}:{os.environ['PATH']}",
    }

    result = subprocess.run(["bash", "-c", command], env=environment, check=False)

    assert result.returncode == 0


def test_single_chunk_generation_is_merged_before_postprocessing(tmp_path: Path) -> None:
    plan = plan_generate(
        _context("generate"),
        "",
        output_dir=str(tmp_path / "outputs"),
        input_file=str(tmp_path / "input.jsonl"),
        model="model",
        server_address="http://server:5000/v1",
        num_chunks=1,
        postprocess_cmd="echo postprocess",
    )

    generation, merge = plan.tasks
    assert "++num_chunks=1 ++chunk_id=0" in generation.commands[0].command
    assert "echo postprocess" not in generation.commands[0].command
    assert merge.dependencies == (DependencyPlan(generation.task_id),)
    assert str(tmp_path / "outputs" / "output_chunk_0.jsonl") in merge.commands[0].command
    assert "echo postprocess" in merge.commands[0].command
    invocations = {task.parameters["generation_invocation"] for task in plan.tasks}
    assert len(invocations) == 1
    assert next(iter(invocations)).startswith("generate:generate:")


def _merge_task(tmp_path: Path, postprocess_cmd: str | None = None):
    plan = plan_generate(
        _context("generate"),
        "",
        output_dir=str(tmp_path / "outputs"),
        input_file=str(tmp_path / "input.jsonl"),
        model="model",
        server_address="http://server:5000/v1",
        num_chunks=2,
        postprocess_cmd=postprocess_cmd,
    )
    return plan.tasks[-1]


class _Markers:
    def __init__(self, existing: set[str]) -> None:
        self.existing = existing

    def batch_exists(self, paths: list[str]) -> dict[str, bool]:
        return {path: path in self.existing for path in paths}


def test_live_plans_leave_out_chunks_nemo_skills_will_skip(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"

    def plan(
        *done: int,
        live: bool = True,
        engine: Literal["slurm", "ray"] | None = None,
        **extra: Any,
    ):
        markers = {str(outputs / f"output_chunk_{chunk}.jsonl.done") for chunk in done}
        context = replace(
            _context("generate"),
            run_after=("upstream",),
            live=live,
            engine=engine,
            filesystem=cast(Any, _Markers(markers)),
        )
        return plan_generate(
            context,
            "",
            output_dir=str(outputs),
            input_file=str(tmp_path / "input.jsonl"),
            model="model",
            server_address="http://server:5000/v1",
            num_chunks=2,
            **extra,
        )

    partial = plan(0)
    finished = plan(0, 1)

    assert [task.name for task in partial.tasks] == ["generate-chunk1", "generate-merge"]
    assert partial.tasks[-1].dependencies == (DependencyPlan(partial.tasks[0].task_id),)
    assert [task.name for task in finished.tasks] == ["generate-merge"]
    assert finished.tasks[0].dependencies == (DependencyPlan("upstream", external=True),)
    assert len(plan(0, 1, live=False).tasks) == 3
    assert len(plan(0, 1, rerun_done=True).tasks) == 3
    assert len(plan(0, 1, engine="ray").tasks) == 3


def test_a_multi_node_vllm_server_on_mp_runs_the_launcher_on_every_node(tmp_path: Path) -> None:
    def plan(**extra: Any):
        parameters: dict[str, Any] = {
            "model": "/models/large",
            "server_type": "vllm",
            "server_gpus": 4,
            "server_nodes": 2,
            **extra,
        }
        [task] = plan_generate(
            _context("generate"),
            "++prompt_config=generic/default",
            output_dir=str(tmp_path / "outputs"),
            input_file=str(tmp_path / "input.jsonl"),
            **parameters,
        ).tasks
        return task

    on_ray = plan()
    assert "--node_rank" not in on_ray.commands[0].command
    assert on_ray.parameters["server_type"] == "vllm"

    without_ray = plan(server_args="--distributed-executor-backend mp")
    service = without_ray.commands[0].command
    assert "--distributed-executor-backend mp --node_rank ${SLURM_PROCID:-0}" in service
    assert without_ray.parameters["server_type"] == ["generic"]
    assert without_ray.parameters["server_container"] == ["vllm"]
    custom = plan(server_args="--distributed-executor-backend mp", server_container="vllm-custom")
    assert custom.parameters["server_container"] == ["vllm-custom"]
    assert without_ray.parameters["server_entrypoint"] == [
        "python3 -m nvflow.integrations.model_servers.serve --server-type vllm"
    ]
    assert without_ray.parameters["arguments"] == (
        "++server.server_type=vllm ++prompt_config=generic/default"
    )
    with pytest.raises(ValueError, match="one model"):
        plan(
            model=["/models/a", "/models/b"],
            server_gpus=[4, 4],
            server_nodes=[2, 2],
            server_args="--distributed-executor-backend mp",
        )


def test_finished_unchunked_generation_leaves_a_noop_stage(tmp_path: Path) -> None:
    class OnlyGenerate(CapturedExecutionStage):
        capture_strategy = "run_cmd"

        def build_tasks(self, config, cluster, expname, run_after=None):
            from nvflow.integrations.nemo_skills.commands import generate, wrap_arguments

            generate(
                ctx=wrap_arguments(""),
                cluster=cluster,
                output_dir=str(tmp_path),
                input_file=str(tmp_path / "input.jsonl"),
                expname=expname,
                model="model",
                server_address="http://server:5000/v1",
            )

    context = replace(
        _context("generate"),
        live=True,
        filesystem=cast(Any, _Markers({str(tmp_path / "output.jsonl.done")})),
    )

    assert OnlyGenerate().execution(context).disposition == "noop"


def test_run_cmd_without_num_gpus_is_a_cpu_task() -> None:
    task = plan_run_cmd(_context("prepare"), "python prepare.py").tasks[0]

    assert task.resources[0].gpus_per_node == 0


def test_merge_fails_when_its_postprocess_fails_partway(tmp_path: Path) -> None:
    executable = tmp_path / "bin" / "python"
    executable.parent.mkdir()
    executable.write_text(
        '#!/bin/sh\nshift 2\nout="$1"; shift\ncat "$@" > "$out" && touch "$out.done" && rm -f "$@"\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    merge = _merge_task(tmp_path, postprocess_cmd="false && true")
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    for chunk in (0, 1):
        (outputs / f"output_chunk_{chunk}.jsonl").write_text(f"{chunk}\n", encoding="utf-8")
        (outputs / f"output_chunk_{chunk}.jsonl.done").touch()
    environment = {**os.environ, "PATH": f"{executable.parent}:{os.environ['PATH']}"}

    result = subprocess.run(["bash", "-c", merge.commands[0].command], env=environment, check=False)

    assert result.returncode != 0


def test_unchunked_postprocess_failure_leaves_no_marker(tmp_path: Path) -> None:
    executable = _stub_python(tmp_path)
    command, output = _chunk_command(tmp_path, chunk=None, postprocess_cmd="false && true")
    environment = {**os.environ, "PATH": f"{executable.parent}:{os.environ['PATH']}"}

    result = subprocess.run(["bash", "-c", command], env=environment, check=False)

    assert result.returncode != 0
    assert not Path(f"{output}.done").exists()


def test_merge_fails_on_a_chunk_marked_done_without_its_file(tmp_path: Path) -> None:
    merge = _merge_task(tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "output_chunk_0.jsonl").write_text('{"generated": true}\n', encoding="utf-8")
    (outputs / "output_chunk_0.jsonl.done").touch()
    (outputs / "output_chunk_1.jsonl.done").touch()

    result = subprocess.run(
        ["bash", "-c", merge.commands[0].command],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert f"incomplete generation chunk: {outputs / 'output_chunk_1.jsonl'}" in result.stderr
    assert not (outputs / "output.jsonl").exists()


def test_distinct_generation_invocations_never_share_a_marker(tmp_path: Path) -> None:
    def invocation(input_name: str) -> str:
        plan = plan_generate(
            _context("generate"),
            "",
            output_dir=str(tmp_path / "outputs"),
            input_file=str(tmp_path / input_name),
            model="model",
            server_address="http://server:5000/v1",
        )
        return plan.tasks[0].parameters["generation_invocation"]

    assert invocation("first.jsonl") != invocation("second.jsonl")
    assert invocation("first.jsonl") == invocation("first.jsonl")


def test_run_cmd_setup_is_part_of_the_backend_neutral_command() -> None:
    plan = plan_run_cmd(
        _context("prepare"),
        "gym eval prepare",
        container="nemo-gym",
        installation_command="export PATH=/opt/nemo_gym_venv/bin:$PATH",
    )

    task = plan.tasks[0]
    assert task.commands[0].command == (
        "set -e\nexport PATH=/opt/nemo_gym_venv/bin:$PATH\ngym eval prepare"
    )
    assert "installation_command" not in task.parameters


def test_run_cmd_output_declarations_are_not_scheduler_arguments() -> None:
    plan = plan_run_cmd(
        _context("split"),
        "python split.py",
        output_dir="/workspace/split",
        log_dir="/workspace/split/logs",
    )

    task = plan.tasks[0]
    assert "output_dir" not in task.parameters
    assert task.parameters["log_dir"] == "/workspace/split/logs"
    assert [output.path for output in task.outputs] == ["/workspace/split"]
