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
"""Worker-boundary tests for exact Ray task commands."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import nvflow.integrations.ray_jobs.worker as worker
from nvflow.integrations.ray_jobs.worker import run_payload


def test_worker_sets_backend_neutral_code_root(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "environment.txt"
    monkeypatch.chdir(tmp_path)

    run_payload(
        {
            "task_id": "stage.0",
            "project_python_root": "projects/demo/src",
            "log_path": str(tmp_path / "task.log"),
            "commands": [
                {
                    "command": (f'printf \'%s\\n%s\' "$NVFLOW_CODE_ROOT" "$PYTHONPATH" > {output}'),
                    "container": "demo",
                    "role": "main",
                    "runtime_template": False,
                }
            ],
        }
    )

    code_root, pythonpath = output.read_text().splitlines()
    assert code_root == str(tmp_path)
    assert pythonpath.split(os.pathsep)[:2] == [
        str(tmp_path / "projects/demo/src"),
        str(tmp_path),
    ]


def test_worker_marks_every_task_with_the_supplied_ray_engine(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "engine.txt"
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NVFLOW_REQUIRE_SUPPLIED_RAY", raising=False)

    run_payload(
        {
            "task_id": "service.0",
            "log_path": str(tmp_path / "task.log"),
            "commands": [
                {
                    "command": (
                        f"printf '%s|%s' \"$NVFLOW_ENGINE\" "
                        f'"${{NVFLOW_REQUIRE_SUPPLIED_RAY:-}}" > {output}'
                    ),
                }
            ],
        }
    )

    assert output.read_text() == "ray|"


def test_worker_copies_command_output_to_its_log_file(
    tmp_path: Path,
    monkeypatch,
    capfd,
) -> None:
    monkeypatch.chdir(tmp_path)
    log_path = tmp_path / "outputs" / "generation-logs" / "main_task_0123_ray.log"

    with pytest.raises(RuntimeError, match="exit code 3"):
        run_payload(
            {
                "task_id": "stage.0",
                "log_path": str(log_path),
                "commands": [{"command": "echo to-stdout; echo to-stderr >&2; exit 3"}],
            }
        )

    assert log_path.read_text().splitlines() == ["to-stdout", "to-stderr"]
    assert capfd.readouterr().out.splitlines() == ["to-stdout", "to-stderr"]


def test_worker_rejects_runtime_template(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="runtime template"):
        run_payload(
            {
                "task_id": "stage.0",
                "log_path": str(tmp_path / "task.log"),
                "commands": [
                    {
                        "command": "builder(<runtime-bindings>)",
                        "runtime_template": True,
                    }
                ],
            }
        )


def test_worker_returns_when_the_command_exits_despite_background_children(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    log_path = tmp_path / "task.log"
    started = time.monotonic()

    run_payload(
        {
            "task_id": "stage.0",
            "log_path": str(log_path),
            "commands": [{"command": "sleep 20 & echo started"}],
        }
    )

    assert time.monotonic() - started < 15
    assert log_path.read_text() == "started\n"


def test_worker_requires_a_log_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="requires a log_path"):
        run_payload({"task_id": "stage.0", "commands": [{"command": "true"}]})


def test_worker_keeps_child_in_ray_job_process_group(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    read_end, write_end = os.pipe()
    os.close(write_end)

    class Process:
        stdout = os.fdopen(read_end, "rb")

        def wait(self) -> int:
            return 0

    def popen(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return Process()

    monkeypatch.setattr(worker.subprocess, "Popen", popen)

    worker._run_command("task", 0, "true", {}, tmp_path / "task.log")
    Process.stdout.close()

    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert "start_new_session" not in kwargs
    assert captured["args"] == (["/bin/bash", "-c", "true"],)


def test_ray_application_installs_strict_attach_boundary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "ray-environment.txt"
    fake_ray = SimpleNamespace(
        init=lambda **_kwargs: None,
        get_runtime_context=lambda: SimpleNamespace(gcs_address="gcs://expected"),
        shutdown=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.chdir(tmp_path)

    run_payload(
        {
            "task_id": "training.0",
            "ray_application": True,
            "log_path": str(tmp_path / "task.log"),
            "commands": [
                {
                    "command": (
                        "printf '%s\\n%s\\n%s\\n%s' \"$NVFLOW_REQUIRE_SUPPLIED_RAY\" "
                        '"$NVFLOW_EXPECTED_GCS_ADDRESS" "$PYTHONPATH" '
                        f'"$RAY_OVERRIDE_JOB_RUNTIME_ENV" > {output}'
                    ),
                    "container": "nemo-rl",
                }
            ],
        }
    )

    required, address, pythonpath, merge_runtime_env = output.read_text().splitlines()
    assert required == "1"
    assert address == "gcs://expected"
    assert pythonpath.split(os.pathsep)[0].endswith("nvflow/integrations/ray_jobs/strict_site")
    # NeMo-RL passes its whole environment, including Ray's RAY_worker_niceness, to ray.init.
    assert merge_runtime_env == "1"


def test_ray_application_attach_failure_keeps_rays_reason(tmp_path: Path, monkeypatch) -> None:
    def refuse(**_kwargs):
        raise ConnectionError("no running Ray instance")

    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(init=refuse, shutdown=lambda: None))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="ConnectionError: no running Ray instance"):
        run_payload(
            {
                "task_id": "training.0",
                "ray_application": True,
                "log_path": str(tmp_path / "task.log"),
                "commands": [{"command": "true", "container": "nemo-rl"}],
            }
        )


_FAKE_RAY = """
import os

attempts = []


def init(*args, **kwargs):
    attempts.append(kwargs.get("address"))
    if os.environ.get("FAKE_RAY_REFUSES") == "1":
        raise ConnectionError("no running Ray instance")


def get_runtime_context():
    class Context:
        gcs_address = "gcs://expected"

    return Context()


def shutdown():
    pass
"""

_ATTACH_TWICE = """
import atexit
import ray

try:
    ray.init()
except RuntimeError as error:
    print(error)
else:
    atexit._run_exitfuncs()
    try:
        ray.init()
    except RuntimeError as error:
        print(error)
print(ray.attempts)
"""


@pytest.mark.parametrize(
    ("refuses", "expected"),
    [
        (
            "1",
            "NVFlow Ray-native workload could not attach to supplied cluster: "
            "ConnectionError: no running Ray instance",
        ),
        ("0", "Ray already shut down as this process exits; NVFlow does not reconnect"),
    ],
)
def test_strict_attach_keeps_rays_reason_and_does_not_reconnect_while_exiting(
    tmp_path: Path, refuses: str, expected: str
) -> None:
    fake = tmp_path / "fake" / "ray"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text(_FAKE_RAY, encoding="utf-8")
    strict_site = Path(worker.__file__).parent / "strict_site"

    completed = subprocess.run(
        [sys.executable, "-c", _ATTACH_TWICE],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(strict_site), str(fake.parent)]),
            "NVFLOW_REQUIRE_SUPPLIED_RAY": "1",
            "NVFLOW_EXPECTED_GCS_ADDRESS": "gcs://expected",
            "FAKE_RAY_REFUSES": refuses,
        },
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.splitlines() == [expected, "['auto']"]
