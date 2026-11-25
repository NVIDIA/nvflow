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
"""Launch a native model server without requiring NeMo-Skills in the image.

Commands match NeMo-Skills' ``serve_vllm`` and ``serve_sglang`` launchers.
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import os
import socket
import subprocess
import sys
import threading
from collections.abc import Mapping
from typing import Any

# Direct-Slurm SGLang peers rendezvous on this fixed port.
_SGLANG_DIST_INIT_PORT = 20000
# Names the server's environment sets, space-separated. Server processes on a
# supplied Ray cluster don't inherit the launcher's environment, so the launcher
# passes these to each node's server.
SERVER_ENV_NAMES = "NVFLOW_SERVER_ENV"


def server_environment(environment: Mapping[str, Any]) -> dict[str, str]:
    """Return a server's environment with the list of its names the launcher passes on."""
    values = {str(name): str(value) for name, value in environment.items()}
    if values:
        values[SERVER_ENV_NAMES] = " ".join(sorted(values))
    return values


def _contains_option(arguments: list[str], *names: str) -> bool:
    return any(
        argument in names or any(argument.startswith(f"{name}=") for name in names)
        for argument in arguments
    )


def _option_value(arguments: list[str], name: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == name and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith(f"{name}="):
            return argument.split("=", 1)[1]
    return None


def _multi_node_vllm_without_ray(server_type: str, num_nodes: int, arguments: list[str]) -> bool:
    """vLLM spans the nodes with its multiprocessing backend instead of a Ray cluster."""
    return (
        server_type == "vllm"
        and num_nodes > 1
        and _option_value(arguments, "--distributed-executor-backend") == "mp"
    )


def _served_endpoint(model: str, port: int, arguments: list[str]) -> list[str]:
    options = []
    if not _contains_option(arguments, "--served-model-name"):
        options.extend(["--served-model-name", model])
    if not _contains_option(arguments, "--trust-remote-code"):
        options.append("--trust-remote-code")
    if not _contains_option(arguments, "--host"):
        options.extend(["--host", "0.0.0.0"])
    return [*options, "--port", str(port)]


def build_command(
    *,
    server_type: str,
    model: str,
    port: int,
    num_gpus: int,
    num_nodes: int,
    arguments: list[str],
    node_rank: int | None = None,
    dist_init_addr: str | None = None,
    master_addr: str | None = None,
    master_port: int | None = None,
) -> list[str]:
    parallel_size = num_gpus * num_nodes
    if server_type == "vllm":
        command = ["vllm", "serve", model, *_served_endpoint(model, port, arguments)]
        if not _contains_option(
            arguments,
            "--tensor-parallel-size",
            "-tp",
        ):
            command.extend(["--tensor-parallel-size", str(parallel_size)])
        if _multi_node_vllm_without_ray(server_type, num_nodes, arguments):
            if node_rank is None or master_addr is None:
                raise ValueError(
                    "Multi-node vLLM on the mp backend requires node_rank and master_addr"
                )
            command.extend(
                [
                    "--nnodes",
                    str(num_nodes),
                    "--node-rank",
                    str(node_rank),
                    "--master-addr",
                    master_addr,
                ]
            )
            if master_port is not None:
                command.extend(["--master-port", str(master_port)])
            if node_rank > 0:
                # Only rank 0 serves the API; the other nodes run its workers.
                command.append("--headless")
        elif num_nodes > 1 and not _contains_option(
            arguments,
            "--distributed-executor-backend",
        ):
            command.extend(["--distributed-executor-backend", "ray"])
        return [*command, *arguments]
    if server_type == "sglang":
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            *_served_endpoint(model, port, arguments),
        ]
        if not _contains_option(arguments, "--tensor-parallel-size", "--tp-size", "--tp"):
            command.extend(["--tensor-parallel-size", str(parallel_size)])
        if num_nodes > 1:
            if node_rank is None or dist_init_addr is None:
                raise ValueError("Multi-node SGLang requires node_rank and dist_init_addr")
            command.extend(
                [
                    "--nnodes",
                    str(num_nodes),
                    "--node-rank",
                    str(node_rank),
                    "--dist-init-addr",
                    dist_init_addr,
                ]
            )
        return [*command, *arguments]
    raise ValueError(f"Unsupported native server type: {server_type!r}")


def parse_args(
    arguments: list[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-type", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--num_gpus", "--num-gpus", dest="num_gpus", type=int, required=True)
    parser.add_argument(
        "--num_nodes",
        "--num-nodes",
        "--nnodes",
        dest="num_nodes",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--node_rank",
        "--node-rank",
        dest="legacy_node_rank",
        type=int,
    )
    parser.add_argument(
        "--dist_init_addr",
        "--dist-init-addr",
        dest="legacy_dist_init_addr",
    )
    return parser.parse_known_args(arguments)


def main() -> None:
    known, remaining = parse_args()
    multi_node_sglang = known.server_type == "sglang" and known.num_nodes > 1
    multi_node_vllm = _multi_node_vllm_without_ray(known.server_type, known.num_nodes, remaining)
    if (multi_node_sglang or multi_node_vllm) and os.environ.get("NVFLOW_ENGINE") == "ray":
        _run_nodes_on_ray(
            server_type=known.server_type,
            model=known.model,
            port=known.port,
            num_gpus=known.num_gpus,
            num_nodes=known.num_nodes,
            arguments=remaining,
        )
        return
    first_node = known.legacy_dist_init_addr
    if (
        (multi_node_sglang or multi_node_vllm)
        and (known.legacy_node_rank or 0) > 0
        and _reaches_only_this_node(first_node)
    ):
        raise SystemExit(
            f"node {known.legacy_node_rank} of a {known.num_nodes}-node {known.server_type} "
            f"server needs the first node's address, got {first_node!r}; "
            "is SLURM_MASTER_NODE set?"
        )
    command = build_command(
        server_type=known.server_type,
        model=known.model,
        port=known.port,
        num_gpus=known.num_gpus,
        num_nodes=known.num_nodes,
        arguments=remaining,
        node_rank=known.legacy_node_rank if multi_node_sglang or multi_node_vllm else None,
        dist_init_addr=(
            f"{known.legacy_dist_init_addr}:{_SGLANG_DIST_INIT_PORT}"
            if multi_node_sglang and known.legacy_dist_init_addr is not None
            else None
        ),
        master_addr=known.legacy_dist_init_addr if multi_node_vllm else None,
    )
    os.execvp(command[0], command)


def _reaches_only_this_node(address: str | None) -> bool:
    if not address or address.lower().startswith("localhost"):
        return True
    try:
        parsed = ipaddress.ip_address(address.strip("[]"))
    except ValueError:
        return False
    return parsed.is_loopback or parsed.is_unspecified


def _run_nodes_on_ray(
    *,
    server_type: str,
    model: str,
    port: int,
    num_gpus: int,
    num_nodes: int,
    arguments: list[str],
) -> None:
    """Start one server process per node of a supplied Ray cluster; rank 0 serves HTTP."""
    import ray
    from ray.util.placement_group import placement_group

    ray.init(address="auto")
    names = os.environ.get(SERVER_ENV_NAMES, "").split()
    environment = {name: os.environ[name] for name in names if name in os.environ}

    @ray.remote(num_cpus=1, num_gpus=num_gpus)
    class ServerNode:
        def address(self) -> str:
            return ray.util.get_node_ip_address()

        def run(self, command: list[str], environment: dict[str, str]) -> int:
            return subprocess.call(command, env={**os.environ, **environment})

    group = placement_group(
        [{"CPU": 1, "GPU": num_gpus} for _ in range(num_nodes)],
        strategy="STRICT_SPREAD",
    )
    ray.get(group.ready())
    actors = [
        ServerNode.options(
            placement_group=group,
            placement_group_bundle_index=index,
        ).remote()
        for index in range(num_nodes)
    ]
    master = ray.get(actors[0].address.remote())
    if master != ray.util.get_node_ip_address():
        # The Job published this node's address, but only rank 0 serves HTTP.
        _relay(port, master, port)
    sglang = server_type == "sglang"
    results = [
        actor.run.remote(
            build_command(
                server_type=server_type,
                model=model,
                port=port,
                num_gpus=num_gpus,
                num_nodes=num_nodes,
                arguments=arguments,
                node_rank=index,
                dist_init_addr=f"{master}:{port + 1}" if sglang else None,
                master_addr=None if sglang else master,
                master_port=None if sglang else port + 1,
            ),
            environment,
        )
        for index, actor in enumerate(actors)
    ]
    # A server runs until it fails, so the first node to stop ends the server.
    [stopped], _ = ray.wait(results, num_returns=1)
    return_code = ray.get(stopped)
    for actor in actors:
        ray.kill(actor)
    if return_code != 0:
        raise RuntimeError(
            f"{server_type} server node {results.index(stopped)} exited with {return_code}"
        )


def _relay(listen_port: int, host: str, port: int) -> None:
    """Relay every connection to ``listen_port`` on this node to ``host:port``."""
    listener = socket.create_server(("", listen_port))

    async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while data := await reader.read(1 << 16):
                writer.write(data)
                await writer.drain()
            writer.write_eof()
        except OSError:
            writer.close()

    async def connect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(host, port)
        except OSError:
            writer.close()
            return
        await asyncio.gather(pipe(reader, upstream_writer), pipe(upstream_reader, writer))
        upstream_writer.close()
        writer.close()

    async def accept() -> None:
        server = await asyncio.start_server(connect, sock=listener)
        await server.serve_forever()

    threading.Thread(target=asyncio.run, args=(accept(),), daemon=True).start()


if __name__ == "__main__":
    main()
