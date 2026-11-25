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
"""scripts/build_images.sh drives Bake one target at a time against a fake docker."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_TARGETS = [
    "nemo-skills",
    "nemo-rl",
    "nemo-gym",
    "vllm-0-29-0",
    "sglang-0-5-17",
    "nvflow-client",
]
FAKE_DOCKER = r"""#!/usr/bin/env bash
echo "$*" >> "$FAKE_DOCKER_LOG"
state=$FAKE_STATE
case "$1 $2" in
  "buildx version") echo "github.com/docker/buildx v0.30.0 0000000" ;;
  "buildx create")
    echo "$*" > "$state/builder"
    while (($#)); do [[ $1 == --buildkitd-config ]] && cp "$2" "$state/buildkitd.toml"; shift; done ;;
  "buildx inspect")
    [[ -f $state/builder ]] || exit 1
    printf 'Name: nvflow-multiarch\n'
    grep -q -- 'network=host' "$state/builder" && printf 'Driver Options: network="host"\n'
    [[ -f $state/buildkitd.toml ]] && sed 's/^/ > /' "$state/buildkitd.toml"
    printf 'Platforms: %s\n' "$FAKE_PLATFORMS" ;;
  "buildx bake")
    if [[ " $* " == *" --print "* && $3 == vllm ]]; then
      printf '{\n  "target": {\n    "a": {\n      "tags": [\n        "r/a:t"\n      ]\n    },\n'
      printf '    "b": {\n      "tags": [\n        "r/b:t"\n      ]\n    }\n  }\n}\n'
    elif [[ " $* " == *" --print "* ]]; then
      printf '{\n  "target": {\n    "%s": {\n      "tags": [\n        "registry.invalid/ns/%s:tag"\n      ]\n    }\n  }\n}\n' "$3" "$3"
    fi ;;
  "buildx imagetools")
    if [[ $3 == inspect && " $* " == *" --format "* ]]; then
      [[ ${*: -1} == *-arm64 ]] && printf linux/arm64 || printf linux/amd64
    elif [[ $3 == inspect ]]; then
      printf '  Platform:    linux/amd64\n  Platform:    linux/arm64\n'
    fi ;;
esac
exit 0
"""


def _run(
    tmp_path: Path,
    *args: str,
    platforms: str = "linux/amd64, linux/arm64*",
    registry: str = "registry.invalid/ns",
    builder_exists: bool = True,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / "scripts" / "build_images.sh", repo / "scripts")
    git = ["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.invalid"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "init"], check=True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "docker").write_text(FAKE_DOCKER, encoding="utf-8")
    (fake_bin / "docker").chmod(0o755)
    state = tmp_path / "state"
    state.mkdir()
    if builder_exists:
        (state / "builder").write_text("buildx create --name nvflow-multiarch\n", encoding="utf-8")
    calls = tmp_path / "docker-calls.txt"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "REGISTRY": registry,
        "FAKE_DOCKER_LOG": str(calls),
        "FAKE_PLATFORMS": platforms,
        "FAKE_STATE": str(state),
    }
    result = subprocess.run(
        ["bash", str(repo / "scripts" / "build_images.sh"), *args],
        env=env,
        capture_output=True,
        text=True,
    )
    return result, calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []


def _builds(calls: list[str]) -> list[str]:
    return [call for call in calls if call.startswith("buildx bake") and "--print" not in call]


def test_default_targets_exist_in_the_bake_file() -> None:
    bake = (PROJECT_ROOT / "docker-bake.hcl").read_text(encoding="utf-8")
    assert all(f'target "{target}"' in bake for target in DEFAULT_TARGETS)


def test_builds_targets_in_order_and_verifies_both_platforms(tmp_path: Path) -> None:
    result, calls = _run(tmp_path)

    assert result.returncode == 0, result.stderr
    builds = _builds(calls)
    assert [call.split()[2] for call in builds] == DEFAULT_TARGETS
    assert all("--push --provenance=false --sbom=false --progress=plain" in call for call in builds)
    assert all("*.platform" not in call for call in builds)
    assert not [call for call in calls if call.startswith("buildx create")]
    assert [call for call in calls if call.startswith("buildx imagetools")] == [
        f"buildx imagetools inspect --builder nvflow-multiarch registry.invalid/ns/{target}:tag"
        for target in DEFAULT_TARGETS
    ]


def test_serial_builds_one_platform_at_a_time_then_joins_them(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, "--serial", "vllm-0-29-0")

    assert result.returncode == 0, result.stderr
    image = "registry.invalid/ns/vllm-0-29-0:tag"
    amd64, arm64 = _builds(calls)
    assert f"--set *.platform=linux/amd64 --set vllm-0-29-0.tags={image}-amd64" in amd64
    assert f"--set *.platform=linux/arm64 --set vllm-0-29-0.tags={image}-arm64" in arm64
    assert [call for call in calls if call.startswith("buildx imagetools")] == [
        f"buildx imagetools create --builder nvflow-multiarch --tag {image} "
        f"{image}-amd64 {image}-arm64",
        f"buildx imagetools inspect --builder nvflow-multiarch {image}",
    ]


def test_single_platform_test_build_never_replaces_the_dual_arch_tag(tmp_path: Path) -> None:
    result, calls = _run(
        tmp_path, "--platform", "linux/amd64", "nemo-skills", platforms="linux/amd64"
    )

    assert result.returncode == 0, result.stderr
    image = "registry.invalid/ns/nemo-skills:tag"
    [build] = _builds(calls)
    assert f"--set *.platform=linux/amd64 --set nemo-skills.tags={image}-amd64" in build
    assert [call for call in calls if call.startswith("buildx imagetools")] == [
        "buildx imagetools inspect --builder nvflow-multiarch "
        f"--format {{{{.Image.OS}}}}/{{{{.Image.Architecture}}}} {image}-amd64"
    ]


def test_bake_groups_are_rejected_before_any_build(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, "--serial", "nemo-skills", "vllm")

    assert result.returncode == 1
    assert "vllm must be a single Bake target with one image, not a group" in result.stderr
    assert not _builds(calls)


def test_local_registry_gets_a_builder_that_can_reach_it(tmp_path: Path) -> None:
    result, calls = _run(
        tmp_path, "nemo-skills", registry="localhost:5000/nvflow", builder_exists=False
    )

    assert result.returncode == 0, result.stderr
    [create] = [call for call in calls if call.startswith("buildx create")]
    assert "--driver-opt network=host --buildkitd-config" in create
    config = (tmp_path / "state" / "buildkitd.toml").read_text(encoding="utf-8")
    assert config == '[registry."localhost:5000"]\n  http = true\n'


def test_local_registry_rejects_a_builder_without_host_networking(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, registry="localhost:5000/nvflow")

    assert result.returncode == 1
    assert "docker buildx rm nvflow-multiarch" in result.stderr
    assert not _builds(calls)


def test_unknown_platform_and_option_are_rejected(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, "--platform", "linux/riscv64")
    assert result.returncode == 1
    assert "--platform must be linux/amd64 or linux/arm64" in result.stderr
    assert not calls

    result = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / "build_images.sh"), "--amd64-only"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "error: unknown option --amd64-only" in result.stderr


def test_missing_arm64_stops_before_any_build_and_names_the_fix(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, platforms="linux/amd64, linux/amd64/v2, linux/386")

    assert result.returncode == 1
    assert "tonistiigi/binfmt --install all" in result.stderr
    assert "docker buildx stop nvflow-multiarch" in result.stderr
    assert not _builds(calls)
