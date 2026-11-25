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
"""Remote batching and tunnel-lifecycle contracts."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path
from typing import cast

import pytest

from nvflow.integrations.nemo_gym.native_sharding import (
    bind_behavior_manifest,
    publish_rollout_completion,
)
from nvflow.integrations.nemo_skills.filesystem import (
    _COMPLETION_PROBE_SCRIPT,
    NemoSkillsTargetFilesystem,
    check_remote_mount_directories,
    remote_batch_exists,
    submission_tunnel_scope,
)


class FakeTunnel:
    def __init__(self, output: str) -> None:
        self.output = output
        self.commands: list[str] = []
        self.cleanups = 0

    def run(self, command: str, **_kwargs):
        self.commands.append(command)
        return types.SimpleNamespace(stdout=self.output)

    def cleanup(self) -> None:
        self.cleanups += 1


def _install_nemo_mount_modules(monkeypatch, tunnel: FakeTunnel) -> None:
    for name in ("nemo_skills", "nemo_skills.pipeline", "nemo_skills.pipeline.utils"):
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, name, module)
    cluster = types.ModuleType("nemo_skills.pipeline.utils.cluster")
    cluster.__dict__["get_tunnel"] = lambda _config: tunnel
    mounts = types.ModuleType("nemo_skills.pipeline.utils.mounts")
    mounts.__dict__["get_unmounted_path"] = lambda _config, path: path
    mounts.__dict__["is_mounted_filepath"] = lambda _config, _path: False
    monkeypatch.setitem(sys.modules, "nemo_skills.pipeline.utils.cluster", cluster)
    monkeypatch.setitem(sys.modules, "nemo_skills.pipeline.utils.mounts", mounts)


def test_remote_batch_exists_quotes_paths_and_uses_one_call() -> None:
    tunnel = FakeTunnel("0:Y\n1:N\n")

    result = remote_batch_exists(tunnel, ["/path with spaces", "/missing"])

    assert result == {"/path with spaces": True, "/missing": False}
    assert len(tunnel.commands) == 1
    assert "'/path with spaces'" in tunnel.commands[0]


def test_mount_checks_retain_tunnel_until_submission_scope(monkeypatch) -> None:
    tunnel = FakeTunnel("0:Y\n1:N\n")
    _install_nemo_mount_modules(monkeypatch, tunnel)

    with submission_tunnel_scope():
        check_remote_mount_directories(
            ["/existing", "/missing"],
            {"executor": "slurm"},
            exit_on_failure=False,
        )
        assert tunnel.cleanups == 0

    assert len(tunnel.commands) == 1
    assert tunnel.cleanups == 1


def test_mount_check_reports_missing_and_cleans_without_scope(monkeypatch) -> None:
    tunnel = FakeTunnel("0:N\n")
    _install_nemo_mount_modules(monkeypatch, tunnel)

    with pytest.raises(FileNotFoundError, match="/missing DOES NOT exist"):
        check_remote_mount_directories(["/missing"], {"executor": "slurm"})

    assert tunnel.cleanups == 1


def test_remote_batch_exists_rejects_malformed_response() -> None:
    tunnel = FakeTunnel("unexpected\n")

    with pytest.raises(RuntimeError, match="incomplete path probe"):
        remote_batch_exists(tunnel, ["/path"])


@pytest.mark.requires_nemo
def test_local_completion_probe_validates_content_hashes(tmp_path: Path) -> None:
    compile(_COMPLETION_PROBE_SCRIPT, "<completion-probe>", "exec")
    materialized = tmp_path / "inputs.jsonl"
    successes = tmp_path / "rollouts.jsonl"
    marker = tmp_path / "completion.json"
    row = '{"_ng_task_index":0,"_ng_rollout_index":0}\n'
    materialized.write_text(row, encoding="utf-8")
    successes.write_text(row, encoding="utf-8")
    behavior = "a" * 64
    publish_rollout_completion(materialized, successes, marker, behavior)
    filesystem = NemoSkillsTargetFilesystem({"executor": "none"})
    probe = {
        "materialized_inputs": str(materialized),
        "successes": str(successes),
        "marker": str(marker),
        "behavior_sha256": behavior,
        "behavior_manifest": None,
        "reference_completion": None,
    }

    assert filesystem.batch_completion_matches([probe]) == [True]
    successes.write_text(row + row.replace(":0", ":1", 1), encoding="utf-8")
    assert filesystem.batch_completion_matches([probe]) == [False]


@pytest.mark.requires_nemo
def test_completion_probes_accept_evidence_recorded_under_an_accepted_fingerprint(
    tmp_path: Path,
) -> None:
    materialized = tmp_path / "inputs.jsonl"
    successes = tmp_path / "rollouts.jsonl"
    marker = tmp_path / "completion.json"
    row = '{"_ng_task_index":0,"_ng_rollout_index":0}\n'
    materialized.write_text(row, encoding="utf-8")
    successes.write_text(row, encoding="utf-8")
    legacy, current = "a" * 64, "b" * 64
    publish_rollout_completion(materialized, successes, marker, legacy)
    probe: dict[str, object] = {
        "materialized_inputs": str(materialized),
        "successes": str(successes),
        "marker": str(marker),
        "behavior_sha256": current,
        "behavior_manifest": None,
        "reference_completion": None,
    }
    accepting: dict[str, object] = {**probe, "accepted_behavior_sha256": [legacy]}

    def on_login_node(probes: list[dict[str, object]]) -> list[bool]:
        output = subprocess.run(
            [sys.executable, "-c", _COMPLETION_PROBE_SCRIPT, json.dumps(probes)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return cast(list[bool], json.loads(output.split("__NVFLOW_COMPLETION__:", 1)[1]))

    local = NemoSkillsTargetFilesystem({"executor": "none"})
    assert local.batch_completion_matches([probe, accepting]) == [False, True]
    assert on_login_node([probe, accepting]) == [False, True]


@pytest.mark.requires_nemo
def test_local_completion_probe_accepts_schema_one_behavior_manifest(
    tmp_path: Path,
) -> None:
    materialized = tmp_path / "inputs.jsonl"
    successes = tmp_path / "rollouts.jsonl"
    completion = tmp_path / "completion.json"
    source_completion = tmp_path / "source-completion.json"
    behavior_manifest = tmp_path / "behavior.json"
    row = '{"_ng_task_index":0,"_ng_rollout_index":0}\n'
    materialized.write_text(row, encoding="utf-8")
    successes.write_text(row, encoding="utf-8")
    source_completion.write_text('{"status":"complete"}\n', encoding="utf-8")
    behavior = "a" * 64
    publish_rollout_completion(materialized, successes, completion, behavior)
    bind_behavior_manifest(behavior_manifest, behavior, source_completion)
    filesystem = NemoSkillsTargetFilesystem({"executor": "none"})

    assert filesystem.batch_completion_matches(
        [
            {
                "materialized_inputs": str(materialized),
                "successes": str(successes),
                "marker": str(completion),
                "behavior_sha256": behavior,
                "manifest_behavior_sha256": behavior,
                "behavior_manifest": str(behavior_manifest),
                "reference_completion": str(source_completion),
            }
        ]
    ) == [True]
