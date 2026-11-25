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
"""Cache-owned uv projects for isolated NVFlow project launchers."""

from __future__ import annotations

import hashlib
import importlib.resources as resources
import json
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
import sysconfig
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock, Timeout
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from nvflow.project.context import ProjectContext
from nvflow.project.install_source import (
    NVFlowInstallSource,
    ProjectEnvironmentError,
    active_nvflow_install_source,
    file_digest,
    redact_source_credentials,
)

_STATE_VERSION = 3
_COMPLETE_NAME = ".nvflow-environment.json"
_GENERATION_NAME = ".nvflow-generation"
_ACTIVE_PROJECT_ROOT = "NVFLOW_ACTIVE_PROJECT_ROOT"
_ACTIVE_ENVIRONMENT = "NVFLOW_ACTIVE_PROJECT_ENVIRONMENT"
_CACHE_ROOT = "NVFLOW_PROJECT_ENV_CACHE"
_POLICY_PACKAGE = "nvflow.project._environment_policy"
_MINIMUM_UV_VERSION = Version("0.11.32")
_PYTHON_REQUIREMENT = ">=3.13.14,<3.14"
_BUILD_LOCK_TIMEOUT_SECONDS = 600.0
_PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
_TORCH_CPU_MARKER = (
    "(platform_machine == 'x86_64' or platform_machine == 'AMD64' "
    "or platform_machine == 'aarch64') and sys_platform != 'darwin'"
)
_RESOLVER_ENVIRONMENT_KEYS = (
    "PIP_EXTRA_INDEX_URL",
    "PIP_FIND_LINKS",
    "PIP_INDEX_URL",
    "PIP_NO_INDEX",
    "UV_DEFAULT_INDEX",
    "UV_EXCLUDE_NEWER",
    "UV_EXTRA_INDEX_URL",
    "UV_FIND_LINKS",
    "UV_INDEX",
    "UV_INDEX_STRATEGY",
    "UV_INDEX_URL",
    "UV_NO_INDEX",
    "UV_PRERELEASE",
    "UV_RESOLUTION",
)


@dataclass(frozen=True)
class ProjectEnvironment:
    """One cache-owned synthetic uv project and its launcher venv."""

    cache_root: Path
    fingerprint: str
    project_root: Path
    project_python_root: Path

    @property
    def root(self) -> Path:
        """Virtual environment root retained for CLI compatibility."""
        return self.cache_root / ".venv"

    @property
    def python(self) -> Path:
        return _environment_executable(self.root, "python")

    @property
    def nvflow(self) -> Path:
        name = "nvflow.exe" if os.name == "nt" else "nvflow"
        return _environment_executable(self.root, name)

    @property
    def pyproject(self) -> Path:
        return self.cache_root / "pyproject.toml"

    @property
    def lock(self) -> Path:
        return self.cache_root / "uv.lock"

    @property
    def stamp(self) -> Path:
        return self.cache_root / _COMPLETE_NAME


@dataclass(frozen=True)
class _EnvironmentSpec:
    environment: ProjectEnvironment
    source: NVFlowInstallSource
    pyproject: str
    policy_digest: str


StatusCallback = Callable[[str], None]


class ProjectEnvironmentManager:
    """Prepare and reuse one native uv environment per selected project."""

    def __init__(
        self,
        *,
        cache_root: Path | None = None,
        uv_executable: str = "uv",
        environ: Mapping[str, str] | None = None,
        status: StatusCallback | None = None,
        source: NVFlowInstallSource | None = None,
    ) -> None:
        self.cache_root = (cache_root or project_environment_cache_root()).expanduser().resolve()
        self.uv_executable = uv_executable
        self.environ = dict(os.environ if environ is None else environ)
        self.status = status or (lambda _message: None)
        self.source = source

    def check(self, context: ProjectContext) -> ProjectEnvironment | None:
        """Return the ready environment without invoking uv or mutating the cache."""
        spec = self._spec(context)
        return spec.environment if self.is_valid(spec.environment) else None

    def ensure(
        self,
        context: ProjectContext,
        *,
        upgrade: bool = False,
    ) -> ProjectEnvironment:
        """Create or synchronize the selected environment with native uv."""
        if upgrade:
            return self._upgrade(context)
        spec = self._spec(context)
        environment = spec.environment
        if self.is_valid(environment):
            return environment

        self.status(f"Preparing full project execution environment for {context.identity}")
        environment.cache_root.parent.mkdir(parents=True, exist_ok=True)
        _restrict_directory(environment.cache_root.parent)
        lock = FileLock(
            str(environment.cache_root.parent / f".{environment.fingerprint}.lock"),
            timeout=_BUILD_LOCK_TIMEOUT_SECONDS,
        )
        try:
            with lock:
                if self.is_valid(environment):
                    return environment
                self._synchronize(spec, upgrade=False)
        except Timeout as exc:
            raise ProjectEnvironmentError(
                f"Timed out waiting for project environment build: {environment.root}"
            ) from exc
        return environment

    def _upgrade(self, context: ProjectContext) -> ProjectEnvironment:
        parent = self._project_cache_directory(context)
        self.status(f"Preparing full project execution environment for {context.identity}")
        parent.mkdir(parents=True, exist_ok=True)
        _restrict_directory(parent)
        lock = FileLock(
            str(parent / ".upgrade.lock"),
            timeout=_BUILD_LOCK_TIMEOUT_SECONDS,
        )
        try:
            with lock:
                generation = secrets.token_hex(16)
                spec = self._spec(context, generation=generation)
                self._synchronize(spec, upgrade=True)
                _atomic_write(parent / _GENERATION_NAME, generation + "\n")
                return spec.environment
        except Timeout as exc:
            raise ProjectEnvironmentError(
                f"Timed out waiting for project environment upgrade: {parent}"
            ) from exc

    def is_valid(self, environment: ProjectEnvironment) -> bool:
        """Return whether uv completed this exact cache input successfully."""
        if (
            not environment.python.is_file()
            or not environment.nvflow.is_file()
            or not environment.lock.is_file()
            or not environment.pyproject.is_file()
        ):
            return False
        try:
            stamp = json.loads(environment.stamp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            isinstance(stamp, dict)
            and stamp.get("state_version") == _STATE_VERSION
            and stamp.get("fingerprint") == environment.fingerprint
            and stamp.get("project_root") == str(environment.project_root)
            and stamp.get("project_python_root") == str(environment.project_python_root)
            and stamp.get("lock_digest") == file_digest(environment.lock)
            and _project_path_file_valid(environment, stamp)
        )

    def active_for(self, environment: ProjectEnvironment) -> bool:
        """Return whether this process already uses the selected environment."""
        return (
            self.environ.get(_ACTIVE_PROJECT_ROOT) == str(environment.project_root)
            and self.environ.get(_ACTIVE_ENVIRONMENT) == environment.fingerprint
            and Path(sys.prefix).resolve() == environment.root.resolve()
        )

    def active_context(self, context: ProjectContext) -> bool:
        """Fast recursion guard that performs no fingerprint or uv work."""
        active_root = self.environ.get(_ACTIVE_PROJECT_ROOT)
        active_environment = self.environ.get(_ACTIVE_ENVIRONMENT)
        virtual_environment = self.environ.get("VIRTUAL_ENV")
        return (
            active_root == str(context.root)
            and bool(active_environment)
            and bool(virtual_environment)
            and Path(virtual_environment).parent.name == active_environment
            and Path(sys.prefix).resolve() == Path(virtual_environment).resolve()
        )

    def activation_environment(self, environment: ProjectEnvironment) -> dict[str, str]:
        """Return isolated child variables and a recursion marker."""
        child = self._clean_python_environment()
        child[_ACTIVE_PROJECT_ROOT] = str(environment.project_root)
        child[_ACTIVE_ENVIRONMENT] = environment.fingerprint
        child["VIRTUAL_ENV"] = str(environment.root)
        scripts = str(environment.nvflow.parent)
        current_path = child.get("PATH")
        child["PATH"] = os.pathsep.join((scripts, current_path)) if current_path else scripts
        return child

    def _spec(
        self,
        context: ProjectContext,
        *,
        generation: str | None = None,
    ) -> _EnvironmentSpec:
        source = self.source or active_nvflow_install_source()
        policy = _environment_policy()
        project_data = _project_metadata(context)
        if generation is None:
            generation = self._active_generation(context)
        rendered = _render_synthetic_pyproject(
            source=source,
            policy=policy,
            project_dependencies=_project_dependencies(project_data),
        )
        inputs = {
            "state_version": _STATE_VERSION,
            "project_root": str(context.root),
            "project_python_root": str(context.launcher_python_root),
            "project_metadata": _project_fingerprint(project_data),
            "python": _python_identity(),
            "nvflow_source": source.fingerprint_material(),
            "policy": policy,
            "resolver_environment": self._resolver_environment(),
            "generation": generation,
        }
        fingerprint = hashlib.sha256(
            json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        environment = ProjectEnvironment(
            cache_root=self._project_cache_directory(context) / fingerprint,
            fingerprint=fingerprint,
            project_root=context.root,
            project_python_root=context.launcher_python_root,
        )
        policy_digest = hashlib.sha256(
            json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return _EnvironmentSpec(
            environment=environment,
            source=source,
            pyproject=rendered,
            policy_digest=f"sha256:{policy_digest}",
        )

    def _project_cache_directory(self, context: ProjectContext) -> Path:
        path_key = hashlib.sha256(str(context.root).encode()).hexdigest()[:12]
        return self.cache_root / f"{context.identity}-{path_key}"

    def _active_generation(self, context: ProjectContext) -> str:
        path = self._project_cache_directory(context) / _GENERATION_NAME
        try:
            generation = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return ""
        except OSError as exc:
            raise ProjectEnvironmentError(f"Unable to read environment generation {path}") from exc
        if not re.fullmatch(r"[0-9a-f]{32}", generation):
            raise ProjectEnvironmentError(f"Invalid project environment generation: {path}")
        return generation

    def _synchronize(self, spec: _EnvironmentSpec, *, upgrade: bool) -> None:
        environment = spec.environment
        spec.source.validate()
        executable = self._uv_path()
        self._require_uv_version(executable)
        environment.cache_root.mkdir(parents=True, exist_ok=True)
        _restrict_directory(environment.cache_root)
        environment.stamp.unlink(missing_ok=True)
        _atomic_write(environment.pyproject, spec.pyproject)
        clean_env = self._clean_uv_environment()

        lock_command = [
            executable,
            "--no-config",
            "--project",
            str(environment.cache_root),
            "lock",
        ]
        if upgrade:
            lock_command.append("--upgrade")
        _run(lock_command, env=clean_env, capture_output=True)
        _run(
            [
                executable,
                "--no-config",
                "--project",
                str(environment.cache_root),
                "sync",
                "--locked",
                "--no-dev",
                "--python",
                sys.executable,
            ],
            env=clean_env,
            capture_output=True,
        )
        if not environment.python.is_file() or not environment.nvflow.is_file():
            raise ProjectEnvironmentError(
                f"uv did not create a runnable project environment: {environment.root}"
            )
        project_path_file = self._install_project_source_path(environment, clean_env)
        stamp = {
            "state_version": _STATE_VERSION,
            "fingerprint": environment.fingerprint,
            "project_root": str(environment.project_root),
            "project_python_root": str(environment.project_python_root),
            "project_path_file": str(project_path_file),
            "nvflow_source": spec.source.identity,
            "policy_digest": spec.policy_digest,
            "lock_digest": file_digest(environment.lock),
        }
        _atomic_write(
            environment.stamp,
            json.dumps(stamp, indent=2, sort_keys=True) + "\n",
        )

    def _install_project_source_path(
        self,
        environment: ProjectEnvironment,
        clean_env: Mapping[str, str],
    ) -> Path:
        source_root = str(environment.project_python_root)
        if "\n" in source_root or "\r" in source_root:
            raise ProjectEnvironmentError("Project Python source path cannot contain newlines")
        completed = _run(
            [
                str(environment.python),
                "-c",
                "import sysconfig; print(sysconfig.get_paths()['purelib'])",
            ],
            env=clean_env,
            capture_output=True,
        )
        site_packages = Path(completed.stdout.strip()).resolve()
        try:
            site_packages.relative_to(environment.root.resolve())
        except ValueError as exc:
            raise ProjectEnvironmentError(
                f"Project environment reported an invalid site-packages path: {site_packages}"
            ) from exc
        path_file = site_packages / "_nvflow_project_source.pth"
        _atomic_write(path_file, source_root + "\n")
        return path_file

    def _uv_path(self) -> str:
        executable = shutil.which(self.uv_executable, path=self.environ.get("PATH"))
        if executable is None:
            raise ProjectEnvironmentError(
                "uv is required to prepare project environments; install uv and retry"
            )
        return executable

    def _require_uv_version(self, executable: str) -> None:
        rendered = _run(
            [executable, "--version"],
            env=self._clean_uv_environment(),
            capture_output=True,
        ).stdout.strip()
        try:
            version = Version(rendered.split()[1])
        except (IndexError, InvalidVersion) as exc:
            raise ProjectEnvironmentError(f"Unable to parse uv version: {rendered!r}") from exc
        if version < _MINIMUM_UV_VERSION:
            raise ProjectEnvironmentError(
                f"Project environments require uv >= {_MINIMUM_UV_VERSION}; found {version}"
            )

    def _clean_uv_environment(self) -> dict[str, str]:
        clean = self._clean_python_environment()
        for name in (
            "PIP_CONFIG_FILE",
            "PIP_CONSTRAINT",
            "PIP_REQUIREMENT",
            "PIP_TARGET",
            "VIRTUAL_ENV",
            "UV_CONFIG_FILE",
            "UV_CONSTRAINT",
            "UV_MANAGED_PYTHON",
            "UV_NO_MANAGED_PYTHON",
            "UV_NO_SOURCES",
            "UV_NO_SOURCES_PACKAGE",
            "UV_OVERRIDE",
            "UV_PYTHON",
            "UV_PYTHON_PLATFORM",
            "UV_PROJECT",
            "UV_PROJECT_ENVIRONMENT",
            "UV_SYSTEM_PYTHON",
            "UV_TORCH_BACKEND",
            "UV_WORKING_DIR",
        ):
            clean.pop(name, None)
        return clean

    def _clean_python_environment(self) -> dict[str, str]:
        clean = dict(self.environ)
        for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE"):
            clean.pop(name, None)
        clean["PYTHONNOUSERSITE"] = "1"
        return clean

    def _resolver_environment(self) -> dict[str, str]:
        return {
            name: self.environ[name]
            for name in _RESOLVER_ENVIRONMENT_KEYS
            if self.environ.get(name)
        }


def project_environment_cache_root(environ: Mapping[str, str] | None = None) -> Path:
    """Return the platform-appropriate project environment cache root."""
    environ = os.environ if environ is None else environ
    configured = environ.get(_CACHE_ROOT)
    if configured:
        return Path(configured)
    if os.name == "nt":
        parent = Path(environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        parent = Path.home() / "Library" / "Caches"
    else:
        parent = Path(environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return parent / "nvflow" / "project-environments"


def _project_metadata(context: ProjectContext) -> dict:
    pyproject = context.root / "pyproject.toml"
    if not pyproject.is_file():
        raise ProjectEnvironmentError(
            f"Project {context.identity!r} must provide pyproject.toml: {pyproject}"
        )
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ProjectEnvironmentError(
            f"Unable to read project metadata {pyproject}: {exc}"
        ) from exc
    project = data.get("project")
    if not isinstance(project, dict) or not isinstance(project.get("name"), str):
        raise ProjectEnvironmentError(f"Project metadata must declare project.name: {pyproject}")
    dynamic = project.get("dynamic", [])
    if not isinstance(dynamic, list) or not all(isinstance(item, str) for item in dynamic):
        raise ProjectEnvironmentError(f"Project metadata has invalid project.dynamic: {pyproject}")
    if "dependencies" in dynamic:
        raise ProjectEnvironmentError(
            f"Project metadata must declare static project.dependencies: {pyproject}"
        )
    requires_python = project.get("requires-python")
    if requires_python is not None:
        try:
            compatible = Version(platform.python_version()) in SpecifierSet(requires_python)
        except (InvalidSpecifier, TypeError) as exc:
            raise ProjectEnvironmentError(
                f"Project metadata has invalid project.requires-python: {pyproject}"
            ) from exc
        if not compatible:
            raise ProjectEnvironmentError(
                f"Project requires Python {requires_python}, found {platform.python_version()}"
            )
    return data


def _project_dependencies(project_data: dict) -> list[str]:
    dependencies = project_data["project"].get("dependencies", [])
    if not isinstance(dependencies, list) or not all(
        isinstance(requirement, str) and requirement for requirement in dependencies
    ):
        raise ProjectEnvironmentError("Project metadata project.dependencies must be a list")
    return dependencies


def _project_fingerprint(project_data: dict) -> dict[str, object]:
    project = project_data["project"]
    return {
        "name": project["name"],
        "requires_python": project.get("requires-python"),
        "dependencies": _project_dependencies(project_data),
    }


def _environment_policy() -> dict[str, object]:
    policy = resources.files(_POLICY_PACKAGE)
    return {
        "execution": _requirements(policy.joinpath("execution.txt").read_text()),
        "overrides": _requirements(policy.joinpath("overrides.txt").read_text()),
        "constraints": _requirements(policy.joinpath("constraints.txt").read_text()),
        "required_uv": str(_MINIMUM_UV_VERSION),
        "cpu_index": _PYTORCH_CPU_INDEX,
        "cpu_marker": _TORCH_CPU_MARKER,
    }


def _requirements(value: str) -> list[str]:
    return [
        line
        for raw_line in value.splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    ]


def _render_synthetic_pyproject(
    *,
    source: NVFlowInstallSource,
    policy: dict[str, object],
    project_dependencies: list[str],
) -> str:
    execution_requirements = policy["execution"]
    if not isinstance(execution_requirements, list) or not all(
        isinstance(requirement, str) for requirement in execution_requirements
    ):
        raise ProjectEnvironmentError("Environment policy execution must be a list of strings")
    dependencies = [
        source.requirement,
        *execution_requirements,
        *project_dependencies,
    ]
    lines = [
        "[project]",
        'name = "nvflow-project-launcher-environment"',
        'version = "0.0.0"',
        f'requires-python = "{_PYTHON_REQUIREMENT}"',
        *_toml_array("dependencies", dependencies),
        "",
        "[tool.uv]",
        "package = false",
        f'required-version = ">={policy["required_uv"]}"',
        *_toml_array("override-dependencies", policy["overrides"]),
        *_toml_array("constraint-dependencies", policy["constraints"]),
        "",
        "[[tool.uv.index]]",
        'name = "pytorch-cpu"',
        f"url = {_toml_string(policy['cpu_index'])}",
        "explicit = true",
        "",
        "[tool.uv.sources]",
    ]
    if source.source is not None:
        lines.append(f'"nvflow" = {_toml_inline_table(source.source)}')
    lines.extend(
        [
            '"torch" = [',
            (f'    {{ index = "pytorch-cpu", marker = {_toml_string(policy["cpu_marker"])} }},'),
            "]",
            "",
        ]
    )
    return "\n".join(lines)


def _toml_array(name: str, values: object) -> list[str]:
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ProjectEnvironmentError(f"Environment policy {name} must be a list of strings")
    return [f"{name} = [", *(f"    {_toml_string(value)}," for value in values), "]"]


def _toml_inline_table(value: dict[str, str | bool]) -> str:
    fields = []
    for key, item in value.items():
        rendered = "true" if item is True else "false" if item is False else _toml_string(item)
        fields.append(f"{key} = {rendered}")
    return "{ " + ", ".join(fields) + " }"


def _toml_string(value: object) -> str:
    if not isinstance(value, str):
        raise ProjectEnvironmentError(f"Expected a TOML string, got {value!r}")
    return json.dumps(value)


def _python_identity() -> dict[str, str]:
    return {
        "implementation": sys.implementation.name,
        "version": platform.python_version(),
        "cache_tag": sys.implementation.cache_tag or "",
        "abi_flags": getattr(sys, "abiflags", ""),
        "platform_tag": sysconfig.get_platform(),
        "machine": platform.machine().lower(),
    }


def _environment_executable(environment: Path, name: str) -> Path:
    directory = "Scripts" if os.name == "nt" else "bin"
    executable = f"{name}.exe" if os.name == "nt" and not name.endswith(".exe") else name
    return environment / directory / executable


def _project_path_file_valid(
    environment: ProjectEnvironment,
    stamp: dict,
) -> bool:
    value = stamp.get("project_path_file")
    if not isinstance(value, str):
        return False
    path = Path(value)
    try:
        path.resolve().relative_to(environment.root.resolve())
        return path.read_text(encoding="utf-8") == f"{environment.project_python_root}\n"
    except (OSError, ValueError):
        return False


def _atomic_write(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _restrict_directory(path: Path) -> None:
    if os.name != "nt":
        path.chmod(0o700)


def _run(
    command: list[str],
    *,
    env: Mapping[str, str],
    cwd: Path | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=dict(env),
            check=True,
            capture_output=capture_output,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() if exc.stderr else str(exc)
        raise ProjectEnvironmentError(
            "Project environment command failed "
            f"(exit {exc.returncode}): {redact_source_credentials(detail)}"
        ) from exc
