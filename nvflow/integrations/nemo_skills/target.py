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
"""Lightweight local preflight for NeMo-Skills target configuration."""

from __future__ import annotations

import os
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from pathlib import Path

from nvflow.execution.target import (
    TARGET_CONFIG_DIR_ENV,
    preflight_target_config,
)

_UPSTREAM_CONFIG_DIR_ENV = "NEMO_SKILLS_CONFIG_DIR"


@contextmanager
def bind_target_config(
    config_path: Path,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> Iterator[None]:
    """Expose one preflighted NVFlow directory to the execution backend."""
    environment = os.environ if environ is None else environ
    previous = environment.get(_UPSTREAM_CONFIG_DIR_ENV)
    environment[_UPSTREAM_CONFIG_DIR_ENV] = str(config_path.resolve().parent)
    try:
        yield
    finally:
        if previous is None:
            environment.pop(_UPSTREAM_CONFIG_DIR_ENV, None)
        else:
            environment[_UPSTREAM_CONFIG_DIR_ENV] = previous


def load_target_config(
    target: str,
    *,
    config_dir: str | Path | None = None,
) -> dict:
    """Load a backend target using NVFlow's public config-directory setting."""
    from nemo_skills.pipeline.utils.cluster import get_cluster_config

    directory = config_dir or os.environ.get(TARGET_CONFIG_DIR_ENV)
    if directory is None:
        return get_cluster_config(target)
    return get_cluster_config(target, config_dir=directory)


__all__ = [
    "TARGET_CONFIG_DIR_ENV",
    "bind_target_config",
    "load_target_config",
    "preflight_target_config",
]
