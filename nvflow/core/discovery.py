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
"""Tolerant-but-loud helpers for stage and recipe discovery.

Replaces the unsafe ``except ImportError: pass`` pattern that silently
dropped entire subpackages from the stage registry.  Failures still
don't abort discovery (siblings keep registering), but each failure is
printed to stderr so operators see what broke and why.

Set ``NVFLOW_DEBUG_IMPORTS=1`` for full tracebacks.
"""

import importlib
import os
import sys
import traceback
from pathlib import Path


def import_stage_modules(package: str, current_dir: Path) -> None:
    """Import every ``.py`` file in *current_dir* as a submodule of *package*.

    Files starting with ``_`` or ``.`` are skipped.  Import failures are
    collected and reported on stderr; sibling modules still load.
    """
    failures: list[tuple[str, BaseException]] = []
    for file in sorted(current_dir.glob("*.py")):
        if file.name.startswith(".") or file.stem.startswith("_"):
            continue
        module_path = f"{package}.{file.stem}"
        try:
            importlib.import_module(f".{file.stem}", package=package)
        except Exception as exc:
            failures.append((module_path, exc))
    if failures:
        _report_failures(failures, kind="module")


def import_stage_subpackages(package: str, current_dir: Path) -> None:
    """Import every immediate subdirectory of *current_dir* that has an ``__init__.py``.

    Directories starting with ``_`` or ``.`` are skipped.  Import
    failures are collected and reported on stderr; sibling packages
    still load.
    """
    failures: list[tuple[str, BaseException]] = []
    for subdir in sorted(current_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith(("_", ".")):
            continue
        if not (subdir / "__init__.py").exists():
            continue
        module_path = f"{package}.{subdir.name}"
        try:
            importlib.import_module(f".{subdir.name}", package=package)
        except Exception as exc:
            failures.append((module_path, exc))
    if failures:
        _report_failures(failures, kind="sub-package")


def _is_duplicate_registration(exc: BaseException) -> bool:
    """Detect StageRegistry's 'already registered' ValueError."""
    return isinstance(exc, ValueError) and "already registered" in str(exc)


def _report_failures(failures: list[tuple[str, BaseException]], kind: str) -> None:
    """Print a concise warning for each failed import to stderr.

    Duplicate-registration ValueErrors are surfaced with a CRITICAL
    prefix so the contract violation is obvious in scrollback.
    """
    debug = bool(os.environ.get("NVFLOW_DEBUG_IMPORTS"))
    has_critical = any(_is_duplicate_registration(exc) for _, exc in failures)
    print(
        f"\n[nvflow] {'CRITICAL' if has_critical else 'WARNING'}: failed to "
        f"import {len(failures)} stage {kind}(s); their stages will NOT be "
        "registered:",
        file=sys.stderr,
    )
    for module_path, exc in failures:
        prefix = "[nvflow] CRITICAL: " if _is_duplicate_registration(exc) else "  - "
        print(
            f"{prefix}{module_path}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        if debug:
            traceback.print_exception(exc, file=sys.stderr)
    print(file=sys.stderr)
