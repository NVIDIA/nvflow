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

"""Cross-provider contracts for the shared CI test lanes."""

from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
GITLAB_CI = PROJECT_ROOT / ".gitlab-ci.yml"
GITHUB_CI = PROJECT_ROOT / ".github" / "workflows" / "unit-tests.yml"
DRIVER = PROJECT_ROOT / "scripts" / "run_ci_test_lane.sh"


def test_gitlab_and_github_use_each_shared_test_lane_once() -> None:
    for path in (GITLAB_CI, GITHUB_CI):
        text = path.read_text(encoding="utf-8")
        for lane in ("minimal", "nemo"):
            command = f"scripts/run_ci_test_lane.sh {lane}"
            assert text.count(command) == 1, f"{path.relative_to(PROJECT_ROOT)}: {command}"


def test_github_launcher_lanes_pin_python_31314() -> None:
    text = GITHUB_CI.read_text(encoding="utf-8")

    assert text.count("actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065") == 2
    assert text.count('python-version: "3.13.14"') == 2


def test_test_lanes_clone_the_history_that_versions_nvflow() -> None:
    gitlab = GITLAB_CI.read_text(encoding="utf-8")
    for job in ("test", "test-full-deps"):
        body = gitlab.split(f"\n{job}:\n", 1)[1].split("\n\n", 1)[0]
        assert "GIT_DEPTH: 0" in body, job
        assert "git fetch --tags --force" in body, job
    assert GITHUB_CI.read_text(encoding="utf-8").count("fetch-depth: 0") == 2


def test_gitlab_full_dependency_lane_can_compile_editdistance() -> None:
    gitlab = GITLAB_CI.read_text(encoding="utf-8")
    body = gitlab.split("\ntest-full-deps:\n", 1)[1].split("\n\n", 1)[0]
    assert "apt-get install -y -qq git g++" in body


def test_minimal_lane_checks_current_worker_runtime() -> None:
    text = DRIVER.read_text(encoding="utf-8")

    assert text.count("scripts/check_worker_python313.py") == 1
    assert "worker-python313" not in text


def test_shared_test_lane_driver_is_valid_shell() -> None:
    assert DRIVER.stat().st_mode & 0o111
    subprocess.run(["bash", "-n", str(DRIVER)], check=True)
