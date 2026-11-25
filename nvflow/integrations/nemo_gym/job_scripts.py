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
"""Shared NeMo-Gym job-script builders."""

import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from nemo_skills.pipeline.utils.scripts import BaseJobScript, ServerScript

from nvflow.core.code_root import CODE_ROOT_TOKEN
from nvflow.core.node_address import NODE_ADDRESS_COMMAND
from nvflow.core.vllm_multi_node import LAUNCHER_MODULE, MP_NODE_ARGS, VLLM_LAUNCHER
from nvflow.integrations.model_servers.serve import server_environment

from .operation_config import parse_server_environment
from .shell_snippets import SHELL_FIND_FREE_PORT, SHELL_READ_PORT_FILE
from .vllm_judge import build_vllm_server_args, vllm_serve_overrides

_DYNAMIC_PORT_TOKEN = "__NVFLOW_DYNAMIC_PORT__"


@dataclass(kw_only=True)
class BashScript(BaseJobScript):
    """A ``BaseJobScript`` that runs a bash command string.

    Module-level class so Fiddle/nemo-run can serialize it by import path.
    """

    cmd: str = ""

    def __post_init__(self):
        self.set_inline(self.cmd)
        super().__post_init__()


def _vllm_port_file(log_dir: str, role: str, job_label: str = "") -> str:
    """Return the port file a supplied-Ray service publishes for readiness."""
    suffix = f"_{job_label}" if job_label else ""
    return f"{log_dir}/.vllm_port_{role}{suffix}.txt"


def _vllm_host_file(log_dir: str, role: str, job_label: str = "") -> str:
    """Return the host file a supplied-Ray service publishes for readiness."""
    suffix = f"_{job_label}" if job_label else ""
    return f"{log_dir}/.vllm_host_{role}{suffix}.txt"


def _slurm_vllm_port_file(log_dir: str, role: str, job_label: str = "") -> str:
    """Return the direct-Slurm port file, unique to one run of one Slurm job.

    The client reads the first file it finds, so no run may find an earlier
    run's file: ``$SLURM_JOB_ID`` separates jobs, and ``$SLURM_RESTART_COUNT``
    the runs of a requeued job, which keeps its ID.
    """
    suffix = f"_{job_label}" if job_label else ""
    return f"{log_dir}/.vllm_port_{role}{suffix}_${{SLURM_JOB_ID}}_${{SLURM_RESTART_COUNT:-0}}.txt"


def _wrap_server_with_dynamic_port(
    script: ServerScript,
    role: str,
    log_dir: str,
    job_label: str,
) -> None:
    """Replace the hardcoded port in *script* with runtime-dynamic allocation.

    Wraps the server's inline command so that at runtime on the compute node:
      1. ``find_free_port()`` probes for an available port
      2. The port is written to a port file (shared filesystem)
      3. ``sed`` replaces the hardcoded port in the original command
    """
    port_file = _vllm_port_file(log_dir, role, job_label)
    host_file = _vllm_host_file(log_dir, role, job_label)
    slurm_port_file = _slurm_vllm_port_file(log_dir, role, job_label)
    hardcoded = str(script.port)
    port_argument = f"--port {hardcoded}"
    if port_argument not in script.inline:
        raise ValueError(
            f"Server command does not contain expected port argument {port_argument!r}"
        )
    original_inline = script.inline.replace(
        port_argument,
        f"--port {_DYNAMIC_PORT_TOKEN}",
        1,
    )
    wrapped = (
        f"{SHELL_FIND_FREE_PORT}\n"
        f"VLLM_PORT=$(find_free_port)\n"
        'if [ "${NVFLOW_ENGINE:-}" = "ray" ]; then\n'
        f'    mkdir -p "{log_dir}" || exit 1\n'
        f'    rm -f "{port_file}" "{host_file}"\n'
        f'    echo "$VLLM_PORT" > "{port_file}" || exit 1\n'
        f"    VLLM_HOST=$({NODE_ADDRESS_COMMAND})\n"
        f'    [ -n "$VLLM_HOST" ] || VLLM_HOST=$(hostname)\n'
        f'    printf \'%s\\n\' "$VLLM_HOST" > "{host_file}" || exit 1\n'
        "else\n"
        f'    echo "[dynamic-port] {role} vLLM using port $VLLM_PORT (node ${{SLURM_NODEID:-0}})"\n'
        f'    if [ "${{SLURM_NODEID:-0}}" = "0" ]; then\n'
        f'        echo "$VLLM_PORT" > "{slurm_port_file}"\n'
        "    fi\n"
        "fi\n"
        f"ORIG_CMD=$(cat <<'__NVFLOW_VLLM_CMD__'\n"
        f"{original_inline}\n"
        f"__NVFLOW_VLLM_CMD__\n"
        f")\n"
        f'eval "$(echo "$ORIG_CMD" | sed "s/{_DYNAMIC_PORT_TOKEN}/$VLLM_PORT/g")"\n'
    )
    script.set_inline(wrapped)
    object.__setattr__(
        script,
        "nvflow_readiness",
        f"files:{host_file},{port_file}",
    )


def _build_port_read_preamble(
    log_dir: str,
    job_label: str,
    *,
    policy_host: str | None = None,
    judge_host: str | None = None,
) -> str:
    """Build bash preamble that sets ``<ROLE>_HOST`` and ``<ROLE>_PORT`` per managed server.

    *policy_host* and *judge_host* are the servers' direct-Slurm hostname
    references; supplied Ray reads the host each service publishes instead.
    Returns empty string if neither server is present.
    """
    servers = [
        (variable, label, host)
        for variable, label, host in (
            ("POLICY", "Policy", policy_host),
            ("JUDGE", "Judge", judge_host),
        )
        if host is not None
    ]
    if not servers:
        return ""
    ray_lines = []
    slurm_lines = []
    for variable, label, host in servers:
        role = variable.lower()
        ray_port = _vllm_port_file(log_dir, role, job_label)
        ray_host = _vllm_host_file(log_dir, role, job_label)
        slurm_port = _slurm_vllm_port_file(log_dir, role, job_label)
        ray_lines += [
            f'    {variable}_PORT=$(read_port_file "{ray_port}" "{label} vLLM" 300)',
            f'    {variable}_HOST=$(read_port_file "{ray_host}" "{label} vLLM host" 300)',
        ]
        slurm_lines += [
            f'    {variable}_PORT=$(read_port_file "{slurm_port}" "{label} vLLM" 300)',
            f'    {variable}_HOST="{host}"',
        ]
    return (
        "\n".join(
            [
                SHELL_READ_PORT_FILE,
                'if [ "${NVFLOW_ENGINE:-}" = "ray" ]; then',
                *ray_lines,
                "else",
                *slurm_lines,
                "fi",
            ]
        )
        + "\n"
    )


def make_server_script(
    vllm_cfg: dict[str, Any],
    cluster_config: dict,
    *,
    role: str = "policy",
    server_type: str = "vllm",
    log_dir: str = "",
    job_label: str = "",
    environment: Mapping[str, Any] | None = None,
) -> ServerScript:
    if "num_gpus" not in vllm_cfg:
        raise ValueError("vLLM config must specify 'num_gpus'")
    if vllm_cfg.get("hf_config_overrides") and server_type != "vllm":
        raise ValueError(f"hf_config_overrides is supported only for vLLM, not {server_type}")

    overrides = vllm_serve_overrides(vllm_cfg)
    num_nodes = vllm_cfg.get("server_nodes", 1)
    entrypoint = vllm_cfg.get("server_entrypoint")
    server_args = build_vllm_server_args(overrides)
    # NeMo-Skills wraps every multi-node vLLM server in `ray start`; its generic
    # type runs NVFlow's launcher on each node as is, which the mp backend needs.
    multi_node_without_ray = (
        server_type == "vllm"
        and num_nodes > 1
        and overrides.get("distributed_executor_backend") == "mp"
        and entrypoint is None
    )
    script = ServerScript(
        server_type="generic" if multi_node_without_ray else server_type,
        model_path=vllm_cfg["model_path"],
        cluster_config=cluster_config,
        num_gpus=vllm_cfg["num_gpus"],
        num_nodes=num_nodes,
        server_args=f"{server_args} {MP_NODE_ARGS}" if multi_node_without_ray else server_args,
        server_entrypoint=VLLM_LAUNCHER if multi_node_without_ray else entrypoint,
    )

    inline = str(script.inline).replace("/nemo_run/code", CODE_ROOT_TOKEN)
    if entrypoint is None and not multi_node_without_ray:
        upstream_module = f"nemo_skills.inference.server.serve_{server_type}"
        if upstream_module not in inline:
            raise ValueError(f"Server command does not contain {upstream_module!r}")
        inline = inline.replace(
            upstream_module,
            f"{LAUNCHER_MODULE} --server-type {server_type}",
            1,
        )
    if environment:
        inline = f"{_exports(environment)} && {inline}"
    script.set_inline(inline)

    if log_dir:
        _wrap_server_with_dynamic_port(script, role, log_dir, job_label)

    return script


def _exports(environment: Mapping[str, Any]) -> str:
    """Return shell exports of a server's environment, with every value kept literal."""
    parse_server_environment(environment, "vllm_serve_env_vars")
    return " && ".join(
        f"export {name}={shlex.quote(value)}"
        for name, value in sorted(server_environment(environment).items())
    )


def make_bash_script(
    bash_cmd: str,
    *,
    installation_command: str | None = None,
) -> BashScript:
    return BashScript(cmd=bash_cmd, installation_command=installation_command)
