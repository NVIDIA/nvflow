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
"""Stage-filesystem inspection failure contracts."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from nvflow.execution.filesystem import MutableTargetFilesystem, TargetFilesystem
from nvflow.integrations.nemo_gym.launcher_fs import LauncherFS as LegacyLauncherFS
from nvflow.integrations.nemo_skills.filesystem import (
    LauncherFS,
    NemoSkillsTargetFilesystem,
)


def test_legacy_launcher_filesystem_import_is_a_compatibility_alias() -> None:
    assert LegacyLauncherFS is LauncherFS is NemoSkillsTargetFilesystem
    filesystem = LauncherFS({})
    assert isinstance(filesystem, TargetFilesystem)
    assert isinstance(filesystem, MutableTargetFilesystem)


@pytest.mark.requires_nemo
def test_missing_local_stage_path_is_normal_filesystem_state(tmp_path: Path) -> None:
    assert LauncherFS({"executor": "none"}).exists(str(tmp_path / "missing")) is False


@pytest.mark.requires_nemo
def test_login_node_maps_workspace_through_target_mounts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "workspace-source"
    (source / "outputs").mkdir(parents=True)
    marker = source / "outputs" / "split.done"
    marker.write_text("{}\n")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "pyproject.toml").touch()
    monkeypatch.chdir(checkout)
    filesystem = LauncherFS({"executor": "slurm", "mounts": [f"{source}:/workspace"]})

    assert filesystem.exists("/workspace/outputs/split.done") is True
    assert filesystem.is_nonempty_file("/workspace/outputs/split.done") is True
    assert filesystem.ls("/workspace/outputs", "*.done") == ["split.done"]
    assert filesystem.count_lines("/workspace/outputs/split.done") == 1
    filesystem.rm(["/workspace/outputs/split.done"])
    assert not marker.exists()


@pytest.mark.requires_nemo
def test_login_node_completion_probe_rejects_paths_outside_the_mounts(tmp_path: Path) -> None:
    filesystem = LauncherFS({"executor": "slurm", "mounts": [f"{tmp_path}:/workspace"]})
    probe = {
        "materialized_inputs": "/data/inputs.jsonl",
        "successes": "/data/rollouts.jsonl",
        "marker": "/data/completion.json",
        "behavior_sha256": "a" * 64,
    }

    with pytest.raises(ValueError, match="Could not find a mount path"):
        filesystem.batch_completion_matches([probe])


def test_existence_cache_avoids_duplicate_remote_probe(monkeypatch) -> None:
    filesystem = LauncherFS({"ssh_tunnel": {"host": "cluster"}})
    calls = 0

    def probe(paths: list[str]) -> dict[str, bool]:
        nonlocal calls
        calls += 1
        return dict.fromkeys(paths, True)

    monkeypatch.setattr(filesystem, "_remote_batch_exists", probe)

    assert filesystem.exists("/workspace/output") is True
    assert filesystem.exists("/workspace/output") is True
    assert calls == 1


@pytest.mark.requires_nemo
def test_remove_invalidates_cached_local_existence(tmp_path: Path) -> None:
    output = tmp_path / "output.jsonl"
    output.write_text("{}\n")
    filesystem = LauncherFS({"executor": "none"})

    assert filesystem.exists(str(output)) is True
    filesystem.rm([str(output)])
    assert filesystem.exists(str(output)) is False


@pytest.mark.requires_nemo
def test_local_nonempty_file_probe_distinguishes_missing_empty_and_populated(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty.jsonl"
    populated = tmp_path / "populated.jsonl"
    empty.touch()
    populated.write_text("{}\n")
    filesystem = LauncherFS({"executor": "none"})

    assert filesystem.is_nonempty_file(str(tmp_path / "missing.jsonl")) is False
    assert filesystem.is_nonempty_file(str(empty)) is False
    assert filesystem.is_nonempty_file(str(populated)) is True


@pytest.mark.requires_nemo
def test_local_line_count_handles_missing_and_populated_files(tmp_path: Path) -> None:
    populated = tmp_path / "records.jsonl"
    populated.write_text("one\ntwo\nthree\n")
    filesystem = LauncherFS({"executor": "none"})

    assert filesystem.count_lines(str(tmp_path / "missing.jsonl")) == 0
    assert filesystem.count_lines(str(populated)) == 3


@pytest.mark.parametrize(("response", "expected"), [("Y", True), ("N", False)])
def test_remote_nonempty_file_probe_uses_target_filesystem(
    monkeypatch,
    response: str,
    expected: bool,
) -> None:
    filesystem = LauncherFS({"ssh_tunnel": {"host": "cluster"}})
    tunnel = SimpleNamespace(
        run=lambda *_args, **_kwargs: SimpleNamespace(stdout=f"__NVFLOW_NONEMPTY__:{response}\n")
    )
    monkeypatch.setattr(filesystem, "_tunnel", lambda: tunnel)
    monkeypatch.setattr(filesystem, "_unmounted", lambda path: f"/remote{path}")

    assert filesystem.is_nonempty_file("/workspace/output.jsonl") is expected


def test_incomplete_remote_batch_probe_is_an_error(monkeypatch) -> None:
    filesystem = LauncherFS({"ssh_tunnel": {"host": "cluster"}})
    tunnel = SimpleNamespace(run=lambda *_args, **_kwargs: SimpleNamespace(stdout="0:Y\n"))
    monkeypatch.setattr(filesystem, "_tunnel", lambda: tunnel)
    monkeypatch.setattr(filesystem, "_unmounted", lambda path: f"/remote{path}")

    with pytest.raises(RuntimeError, match="incomplete path probe"):
        filesystem.batch_exists(["/workspace/one", "/workspace/two"])


def test_remote_listing_distinguishes_missing_from_operational_failure(monkeypatch) -> None:
    filesystem = LauncherFS({"ssh_tunnel": {"host": "cluster"}})
    responses = iter(
        [
            "__NVFLOW_LS__:MISSING\n",
            "__NVFLOW_LS__:UNREADABLE\n",
        ]
    )
    tunnel = SimpleNamespace(run=lambda *_args, **_kwargs: SimpleNamespace(stdout=next(responses)))
    monkeypatch.setattr(filesystem, "_tunnel", lambda: tunnel)
    monkeypatch.setattr(filesystem, "_unmounted", lambda path: f"/remote{path}")

    assert filesystem.ls("/workspace/missing", "*.jsonl") == []
    with pytest.raises(RuntimeError, match="unreadable"):
        filesystem.ls("/workspace/private", "*.jsonl")


def test_remote_listing_accepts_only_tagged_filenames(monkeypatch) -> None:
    filesystem = LauncherFS({"ssh_tunnel": {"host": "cluster"}})
    tunnel = SimpleNamespace(
        run=lambda *_args, **_kwargs: SimpleNamespace(
            stdout=(
                "incidental login output\n"
                "__NVFLOW_ENTRY__:rollouts.jsonl\n"
                "__NVFLOW_ENTRY__:output with spaces.jsonl\n"
                "__NVFLOW_LS__:OK\n"
            )
        )
    )
    monkeypatch.setattr(filesystem, "_tunnel", lambda: tunnel)
    monkeypatch.setattr(filesystem, "_unmounted", lambda path: f"/remote{path}")

    assert filesystem.ls("/workspace/outputs", "*.jsonl") == [
        "output with spaces.jsonl",
        "rollouts.jsonl",
    ]


def test_remote_remove_failure_is_not_ignored(monkeypatch) -> None:
    filesystem = LauncherFS({"ssh_tunnel": {"host": "cluster"}})
    tunnel = SimpleNamespace(
        run=lambda *_args, **_kwargs: SimpleNamespace(stdout="__NVFLOW_RM__:ERROR\n")
    )
    monkeypatch.setattr(filesystem, "_tunnel", lambda: tunnel)
    monkeypatch.setattr(filesystem, "_unmounted", lambda path: f"/remote{path}")

    with pytest.raises(RuntimeError, match="could not remove"):
        filesystem.rm(["/workspace/stale.done"])


def test_remote_transport_failure_is_not_treated_as_a_missing_path(monkeypatch) -> None:
    filesystem = LauncherFS({"ssh_tunnel": {"host": "cluster"}})

    def fail_connection():
        raise ConnectionError("authentication failed")

    monkeypatch.setattr(filesystem, "_tunnel", fail_connection)

    with pytest.raises(ConnectionError, match="authentication failed"):
        filesystem.exists("/workspace/outputs/stage")


def test_generation_file_probe_uses_injected_filesystem() -> None:
    from nvflow.integrations.nemo_gym.generation import _target_has_nonempty_file

    observed: dict[str, str] = {}

    class RecordingFilesystem:
        def is_nonempty_file(self, path: str) -> bool:
            observed["path"] = path
            return True

    assert (
        _target_has_nonempty_file(
            "remote-target",
            "/workspace/input.jsonl",
            RecordingFilesystem(),  # type: ignore[arg-type]
        )
        is True
    )
    assert observed == {"path": "/workspace/input.jsonl"}
