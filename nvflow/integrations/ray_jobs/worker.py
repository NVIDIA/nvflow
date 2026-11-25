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
"""Worker entrypoint for one exact task submitted through Ray Jobs."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO


def run_payload(payload: Mapping[str, Any]) -> None:
    """Execute the ordered commands in one trusted, signed execution payload."""
    task_id = payload.get("task_id")
    commands = payload.get("commands")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("task payload requires a non-empty task_id")
    if not isinstance(commands, list) or not commands:
        raise ValueError(f"task payload {task_id!r} requires commands")
    log_path = payload.get("log_path")
    if not isinstance(log_path, str) or not log_path:
        raise ValueError(f"task payload {task_id!r} requires a log_path")

    code_root = Path.cwd().resolve()
    environment = dict(os.environ)
    environment["NVFLOW_CODE_ROOT"] = str(code_root)
    environment["NVFLOW_ENGINE"] = "ray"
    project_python_root = payload.get("project_python_root")
    python_paths = [str(code_root)]
    if isinstance(project_python_root, str) and project_python_root:
        python_paths.insert(0, str(code_root / project_python_root))
    existing_pythonpath = environment.get("PYTHONPATH")
    if existing_pythonpath:
        python_paths.append(existing_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment.setdefault("RAY_ADDRESS", "auto")
    if payload.get("ray_application"):
        _bind_supplied_ray(environment)
    log_file = Path(log_path.replace("{{NVFLOW_CODE_ROOT}}", str(code_root)))

    for index, command_value in enumerate(commands):
        if not isinstance(command_value, dict):
            raise ValueError(f"task payload {task_id!r} command {index} must be a mapping")
        command = command_value.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError(f"task payload {task_id!r} command {index} is empty")
        if command_value.get("runtime_template"):
            raise ValueError(f"task payload {task_id!r} contains a runtime template")
        _run_command(
            task_id,
            index,
            command.replace("{{NVFLOW_CODE_ROOT}}", str(code_root)),
            environment,
            log_file,
        )


def _bind_supplied_ray(environment: dict[str, str]) -> None:
    """Prove a Ray-native child can attach before launching user code."""
    try:
        import ray

        ray.init(address="auto", log_to_driver=True, include_dashboard=False)
        context = ray.get_runtime_context()
        if not context.gcs_address:
            raise RuntimeError("attached Ray context has no GCS address")
        environment["NVFLOW_EXPECTED_GCS_ADDRESS"] = context.gcs_address
        environment["NVFLOW_REQUIRE_SUPPLIED_RAY"] = "1"
        # Ray adds RAY_worker_niceness to every job's runtime env, and workloads such as
        # NeMo-RL pass their whole environment to ray.init; Ray rejects that overlap
        # unless told to merge the two.
        environment["RAY_OVERRIDE_JOB_RUNTIME_ENV"] = "1"
        strict_site = (
            Path(environment["NVFLOW_CODE_ROOT"])
            / "nvflow"
            / "integrations"
            / "ray_jobs"
            / "strict_site"
        )
        environment["PYTHONPATH"] = os.pathsep.join([str(strict_site), environment["PYTHONPATH"]])
    except Exception as exc:
        raise RuntimeError(
            f"Ray-native task could not attach to supplied cluster: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        try:
            ray.shutdown()
        except (NameError, RuntimeError):
            pass


def _run_command(
    task_id: str,
    index: int,
    command: str,
    environment: Mapping[str, str],
    log_path: Path,
) -> None:
    # Ray keeps Job logs only while its cluster runs; the file stays with the outputs.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            ["/bin/bash", "-c", command],
            cwd=Path.cwd(),
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        copier = threading.Thread(
            target=_copy_output,
            args=(process.stdout.fileno(), log),
            daemon=True,
        )
        copier.start()
        returncode = process.wait()
        # A background child can hold the pipe open; it must not keep the task running.
        copier.join(timeout=5)
    if returncode:
        raise RuntimeError(f"Task {task_id!r} command {index} failed with exit code {returncode}")


def _copy_output(output: int, log: BinaryIO) -> None:
    try:
        for chunk in iter(lambda: os.read(output, 64 * 1024), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            log.write(chunk)
            log.flush()
    except (OSError, ValueError):
        return  # The command exited and closed the log while a background child still wrote.


def _decode_payload(encoded: str) -> Mapping[str, Any]:
    try:
        value = json.loads(base64.urlsafe_b64decode(encoded.encode()).decode())
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid task payload") from exc
    if not isinstance(value, dict):
        raise ValueError("Task payload must be a mapping")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute one NVFlow Ray Jobs task")
    parser.add_argument("--payload", required=True)
    arguments = parser.parse_args()
    run_payload(_decode_payload(arguments.payload))


if __name__ == "__main__":
    main()
