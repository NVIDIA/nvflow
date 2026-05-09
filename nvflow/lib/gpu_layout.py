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
"""Compute Slurm node layout from total GPU count and cluster hardware.

Workflow YAMLs express intent as ``total_gpus`` (how many GPUs the model
needs).  Cluster configs declare hardware as ``gpus_per_node``.  This module
bridges the two so that the same workflow YAML works on clusters with
different GPU-per-node counts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

LOG = logging.getLogger(__name__)

_DEFAULT_GPUS_PER_NODE = 8


@dataclass(frozen=True)
class GpuLayout:
    """Resolved Slurm node layout for a training job."""

    num_nodes: int
    gpus_per_node: int

    @property
    def total_gpus(self) -> int:
        return self.num_nodes * self.gpus_per_node


def resolve_gpu_layout(
    config: dict,
    cluster_config: dict | None = None,
) -> GpuLayout:
    """Compute ``(num_nodes, gpus_per_node)`` from workflow + cluster config.

    Resolution order:

    1. **``total_gpus``** (preferred) -- portable across clusters.
       ``num_nodes = total_gpus // gpus_per_node``.
    2. **``num_nodes`` + optional ``num_gpus``** -- legacy, cluster-specific.
       Passed through directly.
    3. **Neither** -- defaults to a single node.

    Args:
        config: Stage/workflow config dict (may contain ``total_gpus``,
            ``num_nodes``, ``num_gpus``).
        cluster_config: Cluster config dict (may contain ``gpus_per_node``).
            ``None`` is tolerated for local/non-Slurm execution.

    Returns:
        Resolved :class:`GpuLayout`.

    Raises:
        ValueError: If ``total_gpus`` is not evenly divisible by
            ``gpus_per_node``.
    """
    gpus_per_node = _DEFAULT_GPUS_PER_NODE
    if cluster_config:
        gpus_per_node = cluster_config.get("gpus_per_node", _DEFAULT_GPUS_PER_NODE)

    if "total_gpus" in config:
        total = config["total_gpus"]
        if total % gpus_per_node != 0:
            raise ValueError(
                f"total_gpus ({total}) is not evenly divisible by "
                f"gpus_per_node ({gpus_per_node}).  Adjust total_gpus in the "
                f"workflow YAML or gpus_per_node in the cluster config."
            )
        num_nodes = total // gpus_per_node
        return GpuLayout(num_nodes=num_nodes, gpus_per_node=gpus_per_node)

    if "num_nodes" in config:
        num_nodes = config["num_nodes"]
        num_gpus = config.get("num_gpus", gpus_per_node)
        LOG.debug(
            "Using legacy num_nodes=%d / num_gpus=%d (not portable across clusters)",
            num_nodes,
            num_gpus,
        )
        return GpuLayout(num_nodes=num_nodes, gpus_per_node=num_gpus)

    return GpuLayout(num_nodes=1, gpus_per_node=gpus_per_node)
