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
"""Helpers for cluster-specific runtime settings."""

from __future__ import annotations

from typing import Any


def resolve_partition(
    config: dict[str, Any],
    cluster: str,
    *,
    cpu: bool = False,
    override_key: str = "partition",
) -> str:
    """Resolve a stage partition from YAML or the selected cluster config."""
    if config.get(override_key):
        return str(config[override_key])

    from nemo_skills.pipeline.utils import get_cluster_config

    cluster_config = get_cluster_config(
        cluster=cluster,
        config_dir=config.get("cluster_config_dir"),
    )
    key = "cpu_partition" if cpu else "partition"
    partition = cluster_config.get(key)
    if partition is None and cpu:
        partition = cluster_config.get("partition")
    if partition is None:
        raise ValueError(f"Cluster config for {cluster!r} does not define {key!r}")
    return str(partition)
