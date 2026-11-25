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
"""Contracts for cache-owned uv project launcher environments."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import threading
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from urllib.parse import quote

import pytest
from packaging.requirements import Requirement

from nvflow.cli.project_environment import handoff_to_project_environment
from nvflow.project import ProjectResolver
from nvflow.project import environment as project_environment
from nvflow.project.environment import ProjectEnvironment, ProjectEnvironmentManager
from nvflow.project.install_source import (
    NVFlowInstallSource,
    ProjectEnvironmentError,
    active_nvflow_install_source,
    toml_digest,
)


class _Distribution:
    def __init__(self, direct_url: dict[str, object] | None, version: str = "1.2.3") -> None:
        self.metadata = {"Name": "nvflow"}
        self.version = version
        self._direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        assert filename == "direct_url.json"
        return json.dumps(self._direct_url) if self._direct_url is not None else None


def _file_url(path: Path) -> str:
    return f"file://{quote(str(path))}"


def _project(tmp_path: Path):
    root = tmp_path / "demo"
    (root / "src" / "demo_project").mkdir(parents=True)
    (root / "workflows").mkdir()
    (root / "src" / "demo_project" / "__init__.py").write_text("", encoding="utf-8")
    (root / "project.yaml").write_text(
        "manifest_version: 1\nproject: demo\nrequires_nvflow: '>=0'\npython_source_root: src\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        "[project]\nname='demo-project'\nversion='0.1.0'\ndependencies=['nvflow>=0']\n",
        encoding="utf-8",
    )
    return ProjectResolver(search_paths=(tmp_path,)).resolve_named("demo")


def _editable_source(tmp_path: Path) -> NVFlowInstallSource:
    root = tmp_path / "nvflow"
    root.mkdir()
    pyproject = root / "pyproject.toml"
    pyproject.write_text("[project]\nname='nvflow'\nversion='1.2.3'\n", encoding="utf-8")
    return NVFlowInstallSource(
        identity=f"editable:{root}",
        requirement="nvflow",
        source={"path": str(root), "editable": True},
        metadata_path=pyproject,
    )


def _create_environment_files(environment: ProjectEnvironment, pyproject: str) -> None:
    environment.cache_root.mkdir(parents=True, exist_ok=True)
    environment.pyproject.write_text(pyproject, encoding="utf-8")
    environment.lock.write_text("version = 1\n", encoding="utf-8")
    environment.python.parent.mkdir(parents=True)
    environment.python.write_text("", encoding="utf-8")
    environment.nvflow.write_text("", encoding="utf-8")


def _create_project_path_file(environment: ProjectEnvironment) -> Path:
    path = environment.root / "lib" / "site-packages" / "_nvflow_project_source.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{environment.project_python_root}\n", encoding="utf-8")
    return path


def test_index_install_source_uses_exact_active_version() -> None:
    source = active_nvflow_install_source(_Distribution(None))

    assert source.identity == "index:nvflow==1.2.3"
    assert source.requirement == "nvflow==1.2.3"
    assert source.source is None


def test_editable_install_source_uses_local_checkout(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='nvflow'\n", encoding="utf-8")

    source = active_nvflow_install_source(
        _Distribution({"url": _file_url(tmp_path), "dir_info": {"editable": True}})
    )

    assert source.identity == f"editable:{tmp_path.resolve()}"
    assert source.requirement == "nvflow"
    assert source.source == {"path": str(tmp_path.resolve()), "editable": True}
    assert source.fingerprint_material()["metadata_digest"].startswith("sha256:")


def test_archive_and_vcs_sources_preserve_integrity_metadata(tmp_path: Path) -> None:
    wheel = tmp_path / "nvflow.whl"
    wheel.write_bytes(b"wheel")
    wheel_hash = hashlib.sha256(b"wheel").hexdigest()

    archive = active_nvflow_install_source(
        _Distribution(
            {
                "url": _file_url(wheel),
                "archive_info": {"hash": f"sha256={wheel_hash}"},
            }
        )
    )
    vcs = active_nvflow_install_source(
        _Distribution(
            {
                "url": "https://example.com/repository.git",
                "vcs_info": {"vcs": "git", "commit_id": "a" * 40},
                "subdirectory": "packages/nv flow",
            }
        )
    )

    assert archive.requirement == f"nvflow @ {_file_url(wheel)}#sha256={wheel_hash}"
    assert vcs.requirement == (
        f"nvflow @ git+https://example.com/repository.git@{'a' * 40}"
        "#subdirectory=packages/nv%20flow"
    )


def test_local_archive_without_metadata_gets_computed_hash(tmp_path: Path) -> None:
    wheel = tmp_path / "nvflow.whl"
    wheel.write_bytes(b"wheel")

    source = active_nvflow_install_source(
        _Distribution({"url": _file_url(wheel), "archive_info": {}})
    )

    assert source.content_hash == f"sha256={hashlib.sha256(b'wheel').hexdigest()}"
    assert f"#{source.content_hash}" in source.requirement


def test_file_url_percent_sequences_are_decoded_once(tmp_path: Path) -> None:
    wheel = tmp_path / "nvflow%20literal.whl"
    wheel.write_bytes(b"wheel")

    source = active_nvflow_install_source(
        _Distribution({"url": wheel.as_uri(), "archive_info": {}})
    )

    assert source.local_artifact == wheel


def test_remote_archive_without_hash_is_rejected() -> None:
    with pytest.raises(ProjectEnvironmentError, match="no integrity hash"):
        active_nvflow_install_source(
            _Distribution(
                {
                    "url": "https://example.com/nvflow.whl",
                    "archive_info": {},
                }
            )
        )


def test_source_identity_redacts_credentials() -> None:
    source = active_nvflow_install_source(
        _Distribution(
            {
                "url": "https://user:secret@example.com/nvflow.git",
                "vcs_info": {"vcs": "git", "commit_id": "a" * 40},
            }
        )
    )

    assert "secret" not in source.identity
    assert "<redacted>" in source.identity
    assert "secret" in source.requirement


def test_source_identity_redacts_query_credentials() -> None:
    source = active_nvflow_install_source(
        _Distribution(
            {
                "url": (
                    "https://example.com/nvflow.git?X-Amz-Signature=secret&ref=main"
                    "#private_token=hidden"
                ),
                "vcs_info": {"vcs": "git", "commit_id": "a" * 40},
            }
        )
    )

    assert "secret" not in source.identity
    assert "hidden" not in source.identity
    assert "X-Amz-Signature=<redacted>" in source.identity
    assert "private_token=<redacted>" in source.identity
    assert f"nvflow.git@{'a' * 40}?X-Amz-Signature=secret" in source.requirement


def test_archive_hash_change_is_rejected(tmp_path: Path) -> None:
    wheel = tmp_path / "nvflow.whl"
    wheel.write_bytes(b"original")
    expected = hashlib.sha256(b"original").hexdigest()
    source = active_nvflow_install_source(
        _Distribution(
            {
                "url": _file_url(wheel),
                "archive_info": {"hash": f"sha256={expected}"},
            }
        )
    )
    wheel.write_bytes(b"changed")

    with pytest.raises(ProjectEnvironmentError, match="artifact hash changed"):
        source.validate()


def test_deleted_hashed_archive_can_reuse_complete_environment(tmp_path: Path) -> None:
    wheel = tmp_path / "nvflow.whl"
    wheel.write_bytes(b"wheel")
    wheel_hash = hashlib.sha256(b"wheel").hexdigest()
    metadata = _Distribution(
        {
            "url": _file_url(wheel),
            "archive_info": {"hash": f"sha256={wheel_hash}"},
        }
    )
    source = active_nvflow_install_source(metadata)
    context = _project(tmp_path)
    manager = ProjectEnvironmentManager(
        cache_root=tmp_path / "cache",
        source=source,
    )
    spec = manager._spec(context)
    _create_environment_files(spec.environment, spec.pyproject)
    stamp = {
        "state_version": project_environment._STATE_VERSION,
        "fingerprint": spec.environment.fingerprint,
        "project_root": str(context.root),
        "project_python_root": str(context.launcher_python_root),
        "project_path_file": str(_create_project_path_file(spec.environment)),
        "lock_digest": project_environment.file_digest(spec.environment.lock),
    }
    spec.environment.stamp.write_text(json.dumps(stamp), encoding="utf-8")
    wheel.unlink()

    restored_source = active_nvflow_install_source(metadata)
    restored = ProjectEnvironmentManager(
        cache_root=tmp_path / "cache",
        source=restored_source,
    )

    assert restored.check(context) == spec.environment


def test_fingerprint_ignores_comments_but_tracks_dependencies(tmp_path: Path) -> None:
    context = _project(tmp_path)
    manager = ProjectEnvironmentManager(
        cache_root=tmp_path / "cache", source=_editable_source(tmp_path)
    )
    first = manager._spec(context).environment.fingerprint
    with (context.root / "pyproject.toml").open("a", encoding="utf-8") as project_file:
        project_file.write("\n# comment only\n")
    assert manager._spec(context).environment.fingerprint == first
    with (context.root / "pyproject.toml").open("a", encoding="utf-8") as project_file:
        project_file.write("\n[project.optional-dependencies]\nworker=['pandas']\n")
    assert manager._spec(context).environment.fingerprint == first

    (context.root / "pyproject.toml").write_text(
        "[project]\nname='demo-project'\nversion='0.1.0'\ndependencies=['nvflow>=0', 'requests']\n",
        encoding="utf-8",
    )
    assert manager._spec(context).environment.fingerprint != first


def test_toml_digest_supports_native_date_values(tmp_path: Path) -> None:
    path = tmp_path / "metadata.toml"
    path.write_text("released = 2026-09-05\n", encoding="utf-8")

    assert toml_digest(path).startswith("sha256:")


def test_project_environment_requires_static_dependencies(tmp_path: Path) -> None:
    context = _project(tmp_path)
    (context.root / "pyproject.toml").write_text(
        "[project]\nname='demo-project'\nversion='0.1.0'\ndynamic=['dependencies']\n",
        encoding="utf-8",
    )

    with pytest.raises(ProjectEnvironmentError, match="static project.dependencies"):
        ProjectEnvironmentManager(
            cache_root=tmp_path / "cache",
            source=_editable_source(tmp_path),
        ).check(context)


def test_fingerprint_tracks_resolver_indexes(tmp_path: Path) -> None:
    context = _project(tmp_path)
    source = _editable_source(tmp_path)
    default = ProjectEnvironmentManager(cache_root=tmp_path / "default", source=source)
    private = ProjectEnvironmentManager(
        cache_root=tmp_path / "private",
        source=source,
        environ={"UV_INDEX": "https://user:secret@example.com/simple"},
    )

    assert default._spec(context).environment.fingerprint != (
        private._spec(context).environment.fingerprint
    )


def test_python_fingerprint_uses_stable_interpreter_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nvflow.project.environment.platform.platform",
        lambda: pytest.fail("kernel-dependent platform string entered fingerprint"),
    )

    identity = project_environment._python_identity()

    assert {"implementation", "version", "cache_tag", "abi_flags", "platform_tag", "machine"} == (
        identity.keys()
    )


def test_environment_cache_root_respects_override(tmp_path: Path) -> None:
    assert (
        project_environment.project_environment_cache_root(
            {"NVFLOW_PROJECT_ENV_CACHE": str(tmp_path)}
        )
        == tmp_path
    )


def test_manager_builds_native_uv_project_once_and_reuses_without_uv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _project(tmp_path)
    source = _editable_source(tmp_path)
    project_files = sorted(path.relative_to(context.root) for path in context.root.rglob("*"))
    calls: list[list[str]] = []

    monkeypatch.setattr("nvflow.project.environment.shutil.which", lambda *_args, **_kwargs: "uv")

    def fake_run(command, **_kwargs):
        command = [str(value) for value in command]
        calls.append(command)
        if command == ["uv", "--version"]:
            return subprocess.CompletedProcess(command, 0, stdout="uv 0.11.32\n", stderr="")
        if "-c" in command:
            environment_root = Path(command[0]).parent.parent
            site_packages = environment_root / "lib" / "site-packages"
            site_packages.mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=f"{site_packages}\n",
                stderr="",
            )
        cache_root = Path(command[command.index("--project") + 1])
        if "lock" in command:
            cache_root.mkdir(parents=True, exist_ok=True)
            (cache_root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
        elif "sync" in command:
            environment = ProjectEnvironment(
                cache_root=cache_root,
                fingerprint=cache_root.name,
                project_root=context.root,
                project_python_root=context.launcher_python_root,
            )
            environment.python.parent.mkdir(parents=True)
            environment.python.write_text("", encoding="utf-8")
            environment.nvflow.write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("nvflow.project.environment._run", fake_run)
    manager = ProjectEnvironmentManager(cache_root=tmp_path / "cache", source=source)

    first = manager.ensure(context)
    call_count = len(calls)
    second = manager.ensure(context)

    assert first == second
    assert first.lock.is_file()
    assert first.stamp.is_file()
    assert len(calls) == call_count
    assert sum("lock" in command for command in calls) == 1
    assert sum("sync" in command for command in calls) == 1
    sync_command = next(command for command in calls if "sync" in command)
    assert sync_command[sync_command.index("--python") + 1] == project_environment.sys.executable
    rendered = tomllib.loads(first.pyproject.read_text(encoding="utf-8"))
    assert rendered["tool"]["uv"]["package"] is False
    assert rendered["project"]["dependencies"][0] == "nvflow"
    assert any(
        Requirement(value).name == "nemo-skills" for value in rendered["project"]["dependencies"]
    )
    assert "demo-project" not in rendered["tool"]["uv"]["sources"]
    assert rendered["tool"]["uv"]["sources"]["nvflow"]["editable"] is True
    stamp = json.loads(first.stamp.read_text(encoding="utf-8"))
    assert Path(stamp["project_path_file"]).read_text(encoding="utf-8") == (
        f"{context.launcher_python_root}\n"
    )
    assert (
        sorted(path.relative_to(context.root) for path in context.root.rglob("*")) == project_files
    )
    if os.name != "nt":
        assert stat.S_IMODE(first.cache_root.stat().st_mode) == 0o700


def test_check_is_non_mutating_and_upgrade_publishes_new_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _project(tmp_path)
    source = _editable_source(tmp_path)
    manager = ProjectEnvironmentManager(cache_root=tmp_path / "cache", source=source)
    spec = manager._spec(context)
    _create_environment_files(spec.environment, spec.pyproject)
    stamp = {
        "state_version": project_environment._STATE_VERSION,
        "fingerprint": spec.environment.fingerprint,
        "project_root": str(context.root),
        "project_python_root": str(context.launcher_python_root),
        "project_path_file": str(_create_project_path_file(spec.environment)),
        "lock_digest": project_environment.file_digest(spec.environment.lock),
    }
    spec.environment.stamp.write_text(json.dumps(stamp), encoding="utf-8")

    assert manager.check(context) == spec.environment

    synchronizations = []

    def synchronize(upgraded_spec, *, upgrade):
        assert upgrade is True
        synchronizations.append(upgraded_spec.environment)
        _create_environment_files(upgraded_spec.environment, upgraded_spec.pyproject)
        upgraded_stamp = {
            "state_version": project_environment._STATE_VERSION,
            "fingerprint": upgraded_spec.environment.fingerprint,
            "project_root": str(context.root),
            "project_python_root": str(context.launcher_python_root),
            "project_path_file": str(_create_project_path_file(upgraded_spec.environment)),
            "lock_digest": project_environment.file_digest(upgraded_spec.environment.lock),
        }
        upgraded_spec.environment.stamp.write_text(
            json.dumps(upgraded_stamp),
            encoding="utf-8",
        )

    monkeypatch.setattr(manager, "_synchronize", synchronize)
    upgraded = manager.ensure(context, upgrade=True)

    assert synchronizations == [upgraded]
    assert upgraded != spec.environment
    assert manager.is_valid(spec.environment)
    assert manager.check(context) == upgraded


def test_failed_sync_does_not_publish_complete_stamp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _project(tmp_path)
    manager = ProjectEnvironmentManager(
        cache_root=tmp_path / "cache",
        source=_editable_source(tmp_path),
    )
    monkeypatch.setattr(manager, "_require_uv_version", lambda _executable: None)
    monkeypatch.setattr(manager, "_uv_path", lambda: "uv")
    monkeypatch.setattr(
        "nvflow.project.environment._run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("sync failed")),
    )

    with pytest.raises(RuntimeError, match="sync failed"):
        manager.ensure(context)

    assert not list((tmp_path / "cache").rglob(_COMPLETE_NAME))


def test_concurrent_builders_converge_on_one_lock_and_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _project(tmp_path)
    manager = ProjectEnvironmentManager(
        cache_root=tmp_path / "cache",
        source=_editable_source(tmp_path),
    )
    synchronizations = []

    def synchronize(spec, *, upgrade):
        synchronizations.append(spec.environment.cache_root)
        time.sleep(0.05)
        _create_environment_files(spec.environment, spec.pyproject)
        stamp = {
            "state_version": project_environment._STATE_VERSION,
            "fingerprint": spec.environment.fingerprint,
            "project_root": str(context.root),
            "project_python_root": str(context.launcher_python_root),
            "project_path_file": str(_create_project_path_file(spec.environment)),
            "lock_digest": project_environment.file_digest(spec.environment.lock),
        }
        spec.environment.stamp.write_text(json.dumps(stamp), encoding="utf-8")

    monkeypatch.setattr(manager, "_synchronize", synchronize)
    results = []

    def ensure() -> None:
        results.append(manager.ensure(context))

    threads = [threading.Thread(target=ensure) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(synchronizations) == 1
    assert len(results) == 2
    assert results[0] == results[1]


def test_manager_sanitizes_python_and_foreign_policy_environment(tmp_path: Path) -> None:
    manager = ProjectEnvironmentManager(
        cache_root=tmp_path,
        environ={
            "PATH": os.environ.get("PATH", ""),
            "UV_OFFLINE": "1",
            "UV_NO_SOURCES": "1",
            "UV_OVERRIDE": "/untrusted/overrides.txt",
            "UV_PYTHON": "/unrelated/python",
            "UV_PYTHON_PLATFORM": "aarch64-manylinux",
            "VIRTUAL_ENV": "/unrelated",
            "PYTHONHOME": "/unrelated",
            "PYTHONPATH": "/unrelated",
        },
    )

    clean = manager._clean_uv_environment()

    assert clean["UV_OFFLINE"] == "1"
    assert clean["PYTHONNOUSERSITE"] == "1"
    assert "UV_OVERRIDE" not in clean
    assert "UV_NO_SOURCES" not in clean
    assert "UV_PYTHON" not in clean
    assert "UV_PYTHON_PLATFORM" not in clean
    assert "VIRTUAL_ENV" not in clean
    assert "PYTHONHOME" not in clean
    assert "PYTHONPATH" not in clean


def test_active_context_avoids_environment_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _project(tmp_path)
    environment = ProjectEnvironment(
        cache_root=tmp_path / "fingerprint",
        fingerprint="fingerprint",
        project_root=context.root,
        project_python_root=context.launcher_python_root,
    )
    monkeypatch.setattr("nvflow.project.environment.sys.prefix", str(environment.root))
    manager = ProjectEnvironmentManager(
        cache_root=tmp_path / "cache",
        environ={
            "VIRTUAL_ENV": str(environment.root),
            "NVFLOW_ACTIVE_PROJECT_ROOT": str(context.root),
            "NVFLOW_ACTIVE_PROJECT_ENVIRONMENT": "fingerprint",
        },
    )

    assert manager.active_context(context)
    unrelated = tmp_path / "unrelated" / ".venv"
    manager.environ["VIRTUAL_ENV"] = str(unrelated)
    monkeypatch.setattr("nvflow.project.environment.sys.prefix", str(unrelated))
    assert not manager.active_context(context)


def test_cli_handoff_preserves_arguments_and_uses_project_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _project(tmp_path)
    environment = ProjectEnvironment(
        cache_root=tmp_path / "environment-state",
        fingerprint="fingerprint",
        project_root=context.root,
        project_python_root=context.launcher_python_root,
    )
    environment.nvflow.parent.mkdir(parents=True)
    environment.nvflow.write_text("", encoding="utf-8")
    child_environment = {"PATH": str(environment.nvflow.parent), "MARKER": "value"}

    class Manager:
        def active_context(self, selected):
            assert selected == context
            return False

        def check(self, selected):
            assert selected == context
            return environment

        def ensure(self, selected):
            assert selected == context
            return environment

        def active_for(self, selected):
            assert selected == environment
            return False

        def activation_environment(self, selected):
            assert selected == environment
            return child_environment

    captured: dict[str, Any] = {}

    def fake_execve(executable, arguments, environ):
        captured.update(executable=executable, arguments=arguments, environ=environ)
        raise RuntimeError("exec")

    monkeypatch.delenv("NVFLOW_DISABLE_PROJECT_ENVIRONMENT", raising=False)
    monkeypatch.setattr("nvflow.cli.project_environment.os.execve", fake_execve)

    with pytest.raises(RuntimeError, match="exec"):
        handoff_to_project_environment(
            context,
            argv=["validate", "--workflow", "demo:generate"],
            manager=Manager(),
        )

    assert captured == {
        "executable": str(environment.nvflow),
        "arguments": [
            str(environment.nvflow),
            "validate",
            "--workflow",
            "demo:generate",
        ],
        "environ": child_environment,
    }


def test_bootstrap_handoff_does_not_prepare_a_missing_environment(tmp_path: Path) -> None:
    context = _project(tmp_path)

    class Manager:
        def active_context(self, _context):
            return False

        def check(self, _context):
            return None

        def ensure(self, _context):
            pytest.fail("bootstrap attempted an expensive environment build")

    handoff_to_project_environment(
        context,
        manager=cast(Any, Manager()),
        prepare=False,
    )


def test_cli_handoff_is_disabled_for_in_process_callers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _project(tmp_path)
    monkeypatch.setenv("NVFLOW_DISABLE_PROJECT_ENVIRONMENT", "1")

    handoff_to_project_environment(
        context,
        manager=cast(
            Any,
            SimpleNamespace(
                ensure=lambda _context: pytest.fail("disabled handoff built an environment")
            ),
        ),
    )


_COMPLETE_NAME = ".nvflow-environment.json"
