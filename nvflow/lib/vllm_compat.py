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
"""vLLM compatibility helpers for standalone ``vllm serve`` processes.

Routes all vLLM servers through a patched entrypoint that applies runtime
workarounds before the server starts.  See ``scripts/serve_vllm_patched.py``
for the full list of active workarounds.

WORKAROUND(vllm-0.17-hermes, harmony-aarch64)
    Remove this module (and serve_vllm_patched.py) once:
    - vLLM ships hermes thread-safety fix (PR #35034), AND
    - openai_harmony ships a fixed aarch64 binary (issue #71)
"""

from __future__ import annotations

# Path inside the container where the patched wrapper is mounted.
_PATCHED_SERVER_ENTRYPOINT = (
    "/nemo_run/code/scripts/serve_vllm_patched.py"  # WORKAROUND(vllm-0.17-hermes, harmony-aarch64)
)


def get_server_entrypoint() -> str:
    """Return the patched vLLM server entrypoint path."""
    _ensure_ray_ports_patched()  # WORKAROUND(nemo-skills-ray-ports)
    return _PATCHED_SERVER_ENTRYPOINT


def inject_server_entrypoint(kwargs: dict, model_path: str = "", **_ignored) -> dict:
    """Enrich *kwargs* with ``server_entrypoint`` for vLLM servers.

    Only injects when ``server_type`` is ``vllm`` or ``vllm_multimodal``
    (or absent, since vLLM is the nemo-skills default).  Skips sglang /
    trtllm / etc.

    If *kwargs* already contains ``server_entrypoint``, the caller's
    explicit override is preserved.
    """
    _ensure_ray_ports_patched()  # WORKAROUND(nemo-skills-ray-ports)
    server_type = kwargs.get("server_type", "vllm")
    if server_type not in ("vllm", "vllm_multimodal"):
        return kwargs
    if "server_entrypoint" not in kwargs:
        kwargs = {**kwargs, "server_entrypoint": _PATCHED_SERVER_ENTRYPOINT}
    return kwargs


# ---------------------------------------------------------------------------
# WORKAROUND(nemo-skills-ray-ports)
#
# nemo-skills' get_ray_server_cmd hardcodes Ray worker ports at 14349-18349,
# which overlaps with the OS ephemeral port range (9000+ on HSG, 32768+ on
# standard Linux).  vLLM's DP Coordinator allocates ZMQ ports from the
# ephemeral range, causing EADDRINUSE when a Ray worker already holds the
# same port.  Observed on ~25% of multi-node (server_nodes > 1) launches.
#
# Fix: pin Ray worker ports to 6400-6999 — below both ephemeral floors and
# clear of all known services (NFS 2049, Redis/Ray-GCS 6379, vLLM 7000+).
#
# Only affects multi-node vLLM launches.  Single-node configs (num_nodes=1)
# never invoke get_ray_server_cmd (guarded by nemo-skills server.py:174).
#
# Remove when: nemo-skills makes Ray ports configurable upstream.
# ---------------------------------------------------------------------------


def _patched_get_ray_server_cmd(start_cmd):
    """Replacement for nemo_skills get_ray_server_cmd with safe port ranges."""
    ports = (
        "--node-manager-port=1301 "
        "--object-manager-port=1303 "
        "--dashboard-port=8265 "
        "--dashboard-agent-grpc-port=1307 "
        "--runtime-env-agent-port=1305 "
        "--metrics-export-port=1309 "
        "--min-worker-port=6400 "
        "--max-worker-port=6999 "
    )
    return (
        'if [ "${SLURM_PROCID:-0}" = 0 ]; then '
        "    echo 'Starting head node' && "
        "    export RAY_raylet_start_wait_time_s=120 && "
        "    ray start "
        "        --head "
        "        --port=6379 "
        f"       {ports} && "
        f"   {start_cmd} ; "
        "else "
        "    echo 'Starting worker node' && "
        "    export RAY_raylet_start_wait_time_s=120 && "
        '    echo "Connecting to head node at $SLURM_MASTER_NODE" && '
        "    ray start "
        "        --block "
        "        --address=$SLURM_MASTER_NODE:6379 "
        f"       {ports} ;"
        "fi"
    )


_patched_get_ray_server_cmd._nvflow_patched = True


def _ensure_ray_ports_patched():
    """Apply the Ray port fix lazily, on first call. Idempotent."""
    import nemo_skills.pipeline.utils as _ns_utils
    import nemo_skills.pipeline.utils.server as _ns_server

    if getattr(_ns_server.get_ray_server_cmd, "_nvflow_patched", False):
        return
    _ns_server.get_ray_server_cmd = _patched_get_ray_server_cmd
    _ns_utils.get_ray_server_cmd = _patched_get_ray_server_cmd
