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
"""Native model-service entrypoints for supplied runtime images."""

import socket
import sys
import threading
import types
from collections.abc import Callable
from typing import Any

import pytest

from nvflow.integrations.model_servers import serve
from nvflow.integrations.model_servers.serve import build_command, parse_args


def _value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_vllm_command_matches_nemo_skills_launcher() -> None:
    command = build_command(
        server_type="vllm",
        model="/models/policy",
        port=8000,
        num_gpus=4,
        num_nodes=2,
        arguments=["--dtype", "bfloat16"],
    )

    assert command[:3] == ["vllm", "serve", "/models/policy"]
    assert _value(command, "--served-model-name") == "/models/policy"
    assert "--trust-remote-code" in command
    assert _value(command, "--host") == "0.0.0.0"
    assert _value(command, "--port") == "8000"
    assert _value(command, "--tensor-parallel-size") == "8"
    assert _value(command, "--distributed-executor-backend") == "ray"
    assert command[-2:] == ["--dtype", "bfloat16"]
    assert "nemo_skills" not in " ".join(command)


def test_sglang_command_binds_all_interfaces_by_default() -> None:
    command = build_command(
        server_type="sglang",
        model="/models/policy",
        port=8000,
        num_gpus=4,
        num_nodes=1,
        arguments=[],
    )

    assert command[1:3] == ["-m", "sglang.launch_server"]
    assert _value(command, "--model-path") == "/models/policy"
    assert _value(command, "--served-model-name") == "/models/policy"
    assert "--trust-remote-code" in command
    assert _value(command, "--host") == "0.0.0.0"
    assert _value(command, "--tensor-parallel-size") == "4"


def test_explicit_server_options_are_not_duplicated() -> None:
    command = build_command(
        server_type="sglang",
        model="/models/policy",
        port=8000,
        num_gpus=4,
        num_nodes=1,
        arguments=["--tp", "2", "--host=127.0.0.1", "--served-model-name", "policy"],
    )

    assert command.count("--tp") == 1
    assert "--tensor-parallel-size" not in command
    assert "--host" not in command
    assert command.count("--served-model-name") == 1


def test_multi_node_sglang_requires_and_renders_rank_contract() -> None:
    with pytest.raises(ValueError, match="node_rank"):
        build_command(
            server_type="sglang",
            model="/models/policy",
            port=8000,
            num_gpus=8,
            num_nodes=2,
            arguments=[],
        )

    command = build_command(
        server_type="sglang",
        model="/models/policy",
        port=8000,
        num_gpus=8,
        num_nodes=2,
        arguments=[],
        node_rank=1,
        dist_init_addr="10.0.0.1:8001",
    )

    assert _value(command, "--nnodes") == "2"
    assert _value(command, "--node-rank") == "1"
    assert _value(command, "--dist-init-addr") == "10.0.0.1:8001"


def test_multi_node_vllm_on_mp_renders_each_nodes_rank() -> None:
    arguments = ["--distributed-executor-backend", "mp"]
    with pytest.raises(ValueError, match="node_rank and master_addr"):
        build_command(
            server_type="vllm",
            model="/models/policy",
            port=8000,
            num_gpus=4,
            num_nodes=2,
            arguments=arguments,
        )

    head, worker = (
        build_command(
            server_type="vllm",
            model="/models/policy",
            port=8000,
            num_gpus=4,
            num_nodes=2,
            arguments=arguments,
            node_rank=rank,
            master_addr="head-node",
            master_port=8001,
        )
        for rank in (0, 1)
    )

    for rank, command in enumerate((head, worker)):
        assert _value(command, "--tensor-parallel-size") == "8"
        assert _value(command, "--nnodes") == "2"
        assert _value(command, "--node-rank") == str(rank)
        assert _value(command, "--master-addr") == "head-node"
        assert _value(command, "--master-port") == "8001"
        assert command.count("--distributed-executor-backend") == 1
        assert _value(command, "--distributed-executor-backend") == "mp"
    assert "--headless" not in head
    assert "--headless" in worker


def test_legacy_sglang_rank_flags_are_consumed_not_forwarded() -> None:
    known, remaining = parse_args(
        [
            "--server-type",
            "sglang",
            "--model",
            "/models/policy",
            "--port",
            "8000",
            "--num_gpus",
            "8",
            "--nnodes",
            "2",
            "--node_rank",
            "1",
            "--dist_init_addr",
            "old-master:8001",
            "--dtype",
            "bfloat16",
        ]
    )

    assert known.num_nodes == 2
    assert known.legacy_node_rank == 1
    assert known.legacy_dist_init_addr == "old-master:8001"
    assert remaining == ["--dtype", "bfloat16"]


def _launched(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> list[str]:
    launched: list[str] = []

    def execvp(_executable, command):
        launched.extend(command)

    monkeypatch.setattr(serve.os, "execvp", execvp)
    monkeypatch.setattr(sys, "argv", ["serve", *argv])
    serve.main()
    return launched


def test_direct_slurm_multi_node_sglang_uses_launcher_rank_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NVFLOW_ENGINE", raising=False)
    command = _launched(
        monkeypatch,
        [
            "--server-type",
            "sglang",
            "--model",
            "/models/policy",
            "--port",
            "8000",
            "--num_gpus",
            "8",
            "--num_nodes",
            "2",
            "--dist_init_addr",
            "head-node",
            "--node_rank",
            "1",
        ],
    )

    assert _value(command, "--node-rank") == "1"
    assert _value(command, "--dist-init-addr") == "head-node:20000"


def test_direct_slurm_multi_node_vllm_joins_existing_ray_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NVFLOW_ENGINE", raising=False)
    command = _launched(
        monkeypatch,
        [
            "--server-type",
            "vllm",
            "--model",
            "/models/policy",
            "--port",
            "8000",
            "--num_gpus",
            "8",
            "--num_nodes",
            "2",
        ],
    )

    assert command[:3] == ["vllm", "serve", "/models/policy"]
    assert _value(command, "--distributed-executor-backend") == "ray"


def test_direct_slurm_multi_node_vllm_on_mp_uses_launcher_rank_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NVFLOW_ENGINE", raising=False)
    command = _launched(
        monkeypatch,
        [
            "--server-type",
            "vllm",
            "--model",
            "/models/policy",
            "--port",
            "8000",
            "--num_gpus",
            "4",
            "--num_nodes",
            "2",
            "--distributed-executor-backend",
            "mp",
            "--node_rank",
            "1",
            "--dist_init_addr",
            "head-node",
        ],
    )

    assert _value(command, "--node-rank") == "1"
    assert _value(command, "--master-addr") == "head-node"
    assert "--master-port" not in command
    assert "--headless" in command
    assert "--node_rank" not in command
    assert "--dist_init_addr" not in command


@pytest.mark.parametrize(
    ("server_type", "backend_arguments"),
    [("sglang", []), ("vllm", ["--distributed-executor-backend", "mp"])],
)
@pytest.mark.parametrize(
    "address",
    [
        None,
        "",
        "localhost",
        "localhost.localdomain",
        "127.0.0.1",
        "127.0.1.1",
        "::1",
        "[::1]",
        "0.0.0.0",
    ],
)
def test_a_later_node_without_the_first_nodes_address_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
    server_type: str,
    backend_arguments: list[str],
    address: str | None,
) -> None:
    monkeypatch.delenv("NVFLOW_ENGINE", raising=False)

    def launch(rank: str, first_node: str | None) -> list[str]:
        return _launched(
            monkeypatch,
            [
                "--server-type",
                server_type,
                "--model",
                "/models/policy",
                "--port",
                "8000",
                "--num_gpus",
                "4",
                "--num_nodes",
                "2",
                *backend_arguments,
                "--node_rank",
                rank,
                *([] if first_node is None else ["--dist_init_addr", first_node]),
            ],
        )

    assert launch("0", "localhost")
    assert launch("1", "node-01")
    with pytest.raises(SystemExit, match="needs the first node's address"):
        launch("1", address)


def _fake_ray(
    monkeypatch: pytest.MonkeyPatch,
    worker_lacks: list[str],
    *,
    worker_sets: dict[str, str] | None = None,
    killed: list[Any] | None = None,
) -> None:
    """Install a ray module that runs actors here, with *worker_lacks* missing from their
    environment and *worker_sets* set in it."""

    class Method:
        def __init__(self, method: Callable[..., Any], *, in_worker: bool) -> None:
            self._method = method
            self._in_worker = in_worker

        def remote(self, *args: Any) -> Any:
            if self._in_worker:
                for name in worker_lacks:
                    monkeypatch.delenv(name, raising=False)
                for name, value in (worker_sets or {}).items():
                    monkeypatch.setenv(name, value)
            return self._method(*args)

    def remote(**_resources: Any) -> Callable[[type], Any]:
        def decorate(cls: type) -> Any:
            def start(**_options: Any) -> Any:
                instance = cls()
                handle = types.SimpleNamespace(
                    address=Method(instance.address, in_worker=False),
                    run=Method(instance.run, in_worker=True),
                )
                return types.SimpleNamespace(remote=lambda: handle)

            return types.SimpleNamespace(options=start)

        return decorate

    util = types.SimpleNamespace(get_node_ip_address=lambda: "10.0.0.1")
    ray = types.SimpleNamespace(
        init=lambda **_kwargs: None,
        remote=remote,
        get=lambda value: value,
        # Every node has returned here; Ray reports a server that stopped,
        # which is one that failed, before servers that still run.
        wait=lambda codes, num_returns: (
            sorted(codes, key=lambda code: code == 0)[:num_returns],
            [],
        ),
        kill=(killed if killed is not None else []).append,
        util=util,
    )
    placement = types.SimpleNamespace(
        placement_group=lambda *_args, **_kwargs: types.SimpleNamespace(ready=lambda: None)
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setitem(sys.modules, "ray.util", util)
    monkeypatch.setitem(sys.modules, "ray.util.placement_group", placement)


def test_supplied_ray_nodes_run_with_the_servers_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched: list[dict[str, str]] = []
    monkeypatch.setenv("NCCL_MNNVL_ENABLE", "1")
    monkeypatch.setenv("NCCL_CUMEM_ENABLE", "1")
    monkeypatch.setenv(serve.SERVER_ENV_NAMES, "NCCL_CUMEM_ENABLE NCCL_MNNVL_ENABLE")
    _fake_ray(
        monkeypatch,
        worker_lacks=["NCCL_MNNVL_ENABLE"],
        worker_sets={"NCCL_CUMEM_ENABLE": "0"},
    )

    def call(command: list[str], env: dict[str, str]) -> int:
        launched.append(env)
        return 0

    monkeypatch.setattr(serve.subprocess, "call", call)

    serve._run_nodes_on_ray(
        server_type="vllm",
        model="/models/policy",
        port=8000,
        num_gpus=4,
        num_nodes=2,
        arguments=["--distributed-executor-backend", "mp"],
    )

    assert [environment.get("NCCL_MNNVL_ENABLE") for environment in launched] == ["1", "1"]
    assert [environment.get("NCCL_CUMEM_ENABLE") for environment in launched] == ["1", "1"]


def test_a_supplied_ray_server_stops_every_node_when_one_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    killed: list[Any] = []
    return_codes = iter([0, 1])
    _fake_ray(monkeypatch, worker_lacks=[], killed=killed)
    monkeypatch.setattr(serve.subprocess, "call", lambda *_args, **_kwargs: next(return_codes))

    with pytest.raises(RuntimeError, match="vllm server node 1 exited with 1"):
        serve._run_nodes_on_ray(
            server_type="vllm",
            model="/models/policy",
            port=8000,
            num_gpus=4,
            num_nodes=2,
            arguments=["--distributed-executor-backend", "mp"],
        )

    assert len(killed) == 2


@pytest.mark.parametrize(
    ("server_type", "backend_arguments"),
    [("sglang", []), ("vllm", ["--distributed-executor-backend", "mp"])],
)
def test_supplied_ray_multi_node_servers_use_placement_groups(
    monkeypatch: pytest.MonkeyPatch,
    server_type: str,
    backend_arguments: list[str],
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setenv("NVFLOW_ENGINE", "ray")
    monkeypatch.setattr(serve, "_run_nodes_on_ray", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "serve",
            "--server-type",
            server_type,
            "--model",
            "/models/policy",
            "--port",
            "8000",
            "--num_gpus",
            "8",
            "--num_nodes",
            "2",
            "--node_rank",
            "0",
            "--dist_init_addr",
            "localhost",
            *backend_arguments,
        ],
    )

    serve.main()

    assert calls == [
        {
            "server_type": server_type,
            "model": "/models/policy",
            "port": 8000,
            "num_gpus": 8,
            "num_nodes": 2,
            "arguments": backend_arguments,
        }
    ]


def test_relay_serves_the_published_port_from_rank_zero() -> None:
    rank_zero = socket.create_server(("127.0.0.1", 0))

    def answer() -> None:
        connection, _ = rank_zero.accept()
        with connection:
            connection.sendall(b"rank0 " + connection.recv(1024))

    threading.Thread(target=answer, daemon=True).start()
    with socket.socket() as probe:
        probe.bind(("", 0))
        published_port = probe.getsockname()[1]

    serve._relay(published_port, "127.0.0.1", rank_zero.getsockname()[1])

    with socket.create_connection(("127.0.0.1", published_port), timeout=5) as client:
        client.sendall(b"GET /health")
        response = b""
        while chunk := client.recv(1024):
            response += chunk
    rank_zero.close()

    assert response == b"rank0 GET /health"
