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
"""How NVFlow starts a vLLM server that spans nodes.

With a model's ``distributed_executor_backend: mp``, vLLM's own multi-node
mode, every node runs NVFlow's launcher with its rank and the first node's
address, and no Ray cluster is involved.
"""

from __future__ import annotations

import shlex

LAUNCHER_MODULE = "nvflow.integrations.model_servers.serve"
VLLM_LAUNCHER = f"python3 -m {LAUNCHER_MODULE} --server-type vllm"
# Each node of a multi-node vLLM server on the mp backend passes its rank and
# the first node's address; the defaults keep the command parseable off Slurm.
MP_NODE_ARGS = "--node_rank ${SLURM_PROCID:-0} --dist_init_addr ${SLURM_MASTER_NODE:-localhost}"


def backend_in_args(server_args: str) -> str | None:
    """Return the ``--distributed-executor-backend`` a vLLM argument string sets, if any."""
    arguments = shlex.split(server_args)
    for index, argument in enumerate(arguments):
        if argument == "--distributed-executor-backend" and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith("--distributed-executor-backend="):
            return argument.split("=", 1)[1]
    return None
