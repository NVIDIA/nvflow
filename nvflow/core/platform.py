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
"""Identity of the active NVFlow implementation used during planning."""

from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

from nvflow._version import __version__


def wheel_distribution_identity(
    distribution: metadata.Distribution,
    *,
    fallback_name: str,
) -> str:
    """Return one canonical identity for an installed wheel distribution."""
    name = distribution.metadata.get("Name", fallback_name)
    identity = f"wheel:{name}=={distribution.version}"
    direct_url = distribution.read_text("direct_url.json")
    if not direct_url:
        return identity
    try:
        commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    except (AttributeError, json.JSONDecodeError):
        return identity
    if isinstance(commit, str) and commit:
        return f"{identity}@{commit}"
    return identity


@lru_cache(maxsize=1)
def active_platform_identity() -> dict[str, Any]:
    """Return distribution version plus exact Git or wheel source identity."""
    package_dir = Path(__file__).resolve().parents[1]
    root = package_dir.parent
    source = f"wheel:nvflow=={__version__}"
    try:
        distribution = metadata.distribution("nvflow")
    except metadata.PackageNotFoundError:
        pass
    else:
        source = wheel_distribution_identity(distribution, fallback_name="nvflow")
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        tracked = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--error-unmatch",
                package_dir.relative_to(root).joinpath("__init__.py").as_posix(),
            ],
            capture_output=True,
        )
        if revision.returncode == 0 and tracked.returncode == 0:
            source = f"git:{revision.stdout.strip()}"
    except OSError:
        pass
    return {
        "distribution": "nvflow",
        "version": __version__,
        "source": source,
    }


__all__ = ["active_platform_identity", "wheel_distribution_identity"]
