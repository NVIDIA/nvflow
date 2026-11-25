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
"""Container setup must run on compute nodes with only Enroot available."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/setup_containers.sh"


def _run_setup(tmp_path: Path, manifest: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    config = tmp_path / "containers.yaml"
    config.write_text(manifest)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    enroot = fake_bin / "enroot"
    enroot.write_text("#!/bin/bash\nexit 0\n")
    enroot.chmod(0o755)
    srun = fake_bin / "srun"
    srun.write_text(
        """#!/bin/bash
printf '%s\n' "$*" >> "$SETUP_CONTAINERS_CAPTURE"
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--output" ]; then
    shift
    mkdir -p "$(dirname "$1")"
    : > "$1"
    exit 0
  fi
  shift
done
exit 1
"""
    )
    srun.chmod(0o755)
    capture = tmp_path / "srun.txt"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SETUP_CONTAINERS_CAPTURE": str(capture),
    }
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--config",
            str(config),
            str(tmp_path / "images"),
            "--platform",
            "amd64",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, capture


def test_setup_parses_manifest_without_yq(tmp_path: Path) -> None:
    result, capture = _run_setup(
        tmp_path,
        """
containers:
  nemo-gym: nvcr.io/nvidia/nemo-gym:v0.7.0
  platform-image:
    amd64: registry.example/amd64:v1
    arm64: registry.example/arm64:v1
""",
    )

    output_dir = tmp_path / "images"
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output_dir / "nemo-gym-v0.7.0.sqsh").is_file()
    assert (output_dir / "platform-image-v1.sqsh").is_file()
    assert "registry.example/arm64:v1" not in result.stdout
    submitted = capture.read_text(encoding="utf-8")
    assert "docker://nvcr.io#nvidia/nemo-gym:v0.7.0" in submitted
    assert "docker://registry.example#amd64:v1" in submitted
    assert (
        "yq"
        not in SCRIPT.read_text()
        .split("check_dependencies()", 1)[1]
        .split("parse_containers()", 1)[0]
    )


def test_setup_refuses_placeholders_before_downloading(tmp_path: Path) -> None:
    result, capture = _run_setup(
        tmp_path,
        """
containers:
  nemo-gym: nvcr.io/nvidia/nemo-gym:v0.7.0
  # nvflow-client: <YOUR_REGISTRY>/nvflow-client:<RELEASE_TAG>
  nemo-skills: <YOUR_REGISTRY>/nvflow-nemo-skills:v1.2.0
""",
    )

    assert result.returncode == 1, result.stdout + result.stderr
    output = result.stdout + result.stderr
    assert "Replace the placeholders" in output
    assert "nemo-skills: <YOUR_REGISTRY>/nvflow-nemo-skills:v1.2.0" in output
    assert "nvflow-client" not in output
    assert not capture.exists()
