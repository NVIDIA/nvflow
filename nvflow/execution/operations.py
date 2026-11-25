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
"""Public helpers for common project-owned stage plan contributions."""

from __future__ import annotations

import enum
import hashlib
import json
import shlex
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from nvflow.core.node_address import NODE_ADDRESS_COMMAND
from nvflow.core.vllm_multi_node import MP_NODE_ARGS, VLLM_LAUNCHER, backend_in_args
from nvflow.execution.contracts import (
    ArtifactDeclaration,
    CommandPlan,
    DependencyPlan,
    ResourceRequest,
    StagePlan,
    TaskPlan,
)
from nvflow.execution.planner import PlanningContext

_DECLARED_TASK_OUTPUTS: ContextVar[tuple[ArtifactDeclaration, ...] | None] = ContextVar(
    "nvflow_declared_task_outputs",
    default=None,
)


def declared_task_outputs() -> tuple[ArtifactDeclaration, ...] | None:
    """Return outputs scoped around the currently captured invocation."""
    return _DECLARED_TASK_OUTPUTS.get()


def opaque_payload_marker(payload: str, *, label: str = "config") -> str:
    """Return a non-reversible marker for an execution-only payload."""
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"<{label}:sha256:{digest}>"


@contextmanager
def declare_task_outputs(
    *outputs: ArtifactDeclaration,
    output_dir: str | Path | None = None,
    output_file: str | Path | None = None,
) -> Iterator[None]:
    """Attach explicit outputs to one captured helper invocation.

    The context has no execution-side effects. During audited planning, the
    patched helper consumes these declarations for the task created by the
    invocation inside the context. An empty context is a no-op.
    """
    declarations = (
        *outputs,
        *_conventional_outputs(output_dir=output_dir, output_file=output_file),
    )
    if not declarations:
        yield
        return
    token = _DECLARED_TASK_OUTPUTS.set(declarations)
    try:
        yield
    finally:
        _DECLARED_TASK_OUTPUTS.reset(token)


def plan_generate(
    context: PlanningContext,
    arguments: str,
    **parameters: Any,
) -> StagePlan:
    """Build exact generation client/service commands without scheduler calls."""
    tasks = _generation_tasks(context, arguments, parameters)
    declared = declared_task_outputs()
    if declared is not None and tasks:
        tasks = (*tasks[:-1], replace(tasks[-1], outputs=declared))
    return (
        StagePlan(stage=context.stage, tasks=tasks)
        if tasks
        else StagePlan.noop(context.stage, "all generation outputs are complete")
    )


def plan_run_cmd(
    context: PlanningContext,
    arguments: str,
    **parameters: Any,
) -> StagePlan:
    """Plan one NeMo-Skills run_cmd invocation without submitting it."""
    task = task_from_invocation(
        context,
        index=0,
        operation="nemo_skills.run_cmd",
        positional=(),
        keyword=parameters,
        arguments=arguments,
    )
    return StagePlan(stage=context.stage, tasks=(task,))


def _generation_tasks(
    context: PlanningContext,
    arguments: str,
    parameters: dict[str, Any],
) -> tuple[TaskPlan, ...]:
    if parameters.get("with_sandbox"):
        raise ValueError("Generic generation sandbox topology requires an explicit StageExecution")
    models = _list_value(parameters.get("model") or [None])
    server_types = _broadcast(parameters.get("server_type") or "vllm", len(models))
    server_addresses = _broadcast(parameters.get("server_address"), len(models))
    server_gpus = _broadcast(parameters.get("server_gpus"), len(models))
    server_nodes = _broadcast(parameters.get("server_nodes") or 1, len(models))
    server_containers = _broadcast(parameters.get("server_container"), len(models))
    server_args, submitted, submitted_arguments = _launcher_on_every_node(
        parameters,
        arguments,
        models=models,
        server_types=server_types,
        server_addresses=server_addresses,
        server_nodes=server_nodes,
    )
    output_dir = str(parameters["output_dir"])
    input_file = parameters.get("input_file")
    input_dir = parameters.get("input_dir")
    if (input_file is None) == (input_dir is None):
        raise ValueError("Generation requires exactly one of input_file or input_dir")
    seeds = _generation_seeds(parameters)
    chunks = _generation_chunks(parameters)
    chunked = chunks != [None]
    total_chunks = int(parameters.get("num_chunks") or len(chunks))
    dependent_jobs = int(parameters.get("dependent_jobs", 0))
    main_container = str(parameters.get("main_container") or "nemo-skills")
    captured = json.dumps(_json_safe({"arguments": arguments, **parameters}), sort_keys=True)
    invocation = (
        f"{context.stage}:{context.expname}:{hashlib.sha256(captured.encode()).hexdigest()[:12]}"
    )
    upstream = tuple(
        DependencyPlan(name, external=True)
        for name in parameters.get("run_after") or context.run_after
    )
    # On Slurm, NeMo-Skills submits only the chunks without a .done marker, so
    # live plans leave out the rest. Ray runs are planned statically, so Ray plans
    # keep every chunk.
    filesystem = (
        context.filesystem
        if context.live and context.engine != "ray" and not parameters.get("rerun_done")
        else None
    )
    finished = (
        filesystem.batch_exists(
            [
                f"{_generation_output(output_dir, seed, chunk)}.done"
                for seed in seeds
                for chunk in chunks
            ]
        )
        if filesystem is not None
        else {}
    )
    tasks = []
    task_index = 0
    for seed in seeds:
        chunk_terminal_ids = []
        for chunk in chunks:
            if finished.get(f"{_generation_output(output_dir, seed, chunk)}.done"):
                continue
            previous: str | None = None
            output_file = ""
            for attempt in range(dependent_jobs + 1):
                task_id = f"{context.stage}.{task_index}"
                task_index += 1
                suffix = ""
                if seed is not None:
                    suffix += f"-rs{seed}"
                if chunk is not None:
                    suffix += f"-chunk{chunk}"
                if attempt:
                    suffix += f"-dep{attempt}"
                name = f"{context.expname}{suffix}"
                commands = []
                resources = []
                endpoint_variables = []
                for model_index, model in enumerate(models):
                    address = server_addresses[model_index]
                    server_type = _string_value(server_types[model_index])
                    if address:
                        endpoint_variables.append(str(address))
                        continue
                    if model is None:
                        raise ValueError("Self-hosted generation requires a model")
                    gpus = int(server_gpus[model_index] or 0)
                    nodes = int(server_nodes[model_index] or 1)
                    if gpus <= 0:
                        raise ValueError(
                            "Generation requires server_address or positive server_gpus"
                        )
                    role = f"server-{model_index}"
                    host_file = f"{output_dir}/.nvflow/{name}-{role}.host"
                    port_file = f"{output_dir}/.nvflow/{name}-{role}.port"
                    service_command = _generation_server_command(
                        model=str(model),
                        server_type=server_type,
                        server_args=str(server_args[model_index] or ""),
                        host_file=host_file,
                        port_file=port_file,
                        num_gpus=gpus,
                        num_nodes=nodes,
                    )
                    commands.append(
                        CommandPlan(
                            command=service_command,
                            container=str(server_containers[model_index] or server_type),
                            runtime_role=str(server_containers[model_index] or server_type),
                            role=role,
                            group=role,
                            kind="service",
                            readiness=f"files:{host_file},{port_file}",
                        )
                    )
                    resources.append(
                        ResourceRequest(
                            role=role,
                            nodes=nodes,
                            gpus_per_node=gpus,
                            partition=parameters.get("partition"),
                            account=parameters.get("account"),
                            exclusive=parameters.get("exclusive"),
                        )
                    )
                    endpoint_variables.append(_file_backed_endpoint(host_file, port_file))
                client_command, output_file = _generation_client_command(
                    output_dir=output_dir,
                    input_file=(None if input_file is None else str(input_file)),
                    input_dir=(None if input_dir is None else str(input_dir)),
                    arguments=arguments,
                    seed=seed,
                    chunk=chunk,
                    num_chunks=(None if chunk is None else total_chunks),
                    preprocess_cmd=parameters.get("preprocess_cmd"),
                    postprocess_cmd=(None if chunked else parameters.get("postprocess_cmd")),
                    setup_command=parameters.get("installation_command"),
                    server_addresses=endpoint_variables,
                    models=[str(model) for model in models if model is not None],
                    server_types=[_string_value(value) for value in server_types],
                )
                commands.append(
                    CommandPlan(
                        command=client_command,
                        container=main_container,
                        runtime_role=main_container,
                        role="client",
                        group="client",
                    )
                )
                resources.append(
                    ResourceRequest(
                        role="client",
                        nodes=1,
                        gpus_per_node=0,
                        partition=parameters.get("partition"),
                        account=parameters.get("account"),
                    )
                )
                dependencies = (DependencyPlan(previous),) if previous is not None else upstream
                tasks.append(
                    TaskPlan(
                        task_id=task_id,
                        stage=context.stage,
                        name=name,
                        operation="nemo_skills.generate",
                        commands=tuple(commands),
                        resources=tuple(resources),
                        dependencies=dependencies,
                        outputs=(
                            ArtifactDeclaration(
                                name="generation",
                                path=output_file,
                            ),
                        ),
                        parameters={
                            **{
                                key: _json_safe(value)
                                for key, value in submitted.items()
                                if key != "installation_command"
                            },
                            "arguments": submitted_arguments,
                            "experiment_name": context.expname,
                            "generation_invocation": invocation,
                            "group_log_dirs": {
                                resource.role: parameters.get("log_dir")
                                or f"{output_dir}/generation-logs"
                                for resource in resources
                            },
                            **(
                                {
                                    "slurm": {
                                        "installation_command": parameters["installation_command"]
                                    }
                                }
                                if parameters.get("installation_command")
                                else {}
                            ),
                        },
                    )
                )
                previous = task_id
            if previous is not None:
                chunk_terminal_ids.append(previous)
        if chunked:
            final_output = _generation_output(output_dir, seed)
            chunk_outputs = [
                _generation_output(output_dir, seed, index) for index in range(total_chunks)
            ]
            merge_command = (
                "set -euo pipefail\n"
                f"if [ -f {shlex.quote(final_output)} ] && "
                f"[ -f {shlex.quote(final_output + '.done')} ]; then\n"
                "  echo 'merged generation output already complete; skipping'\n"
                "  exit 0\n"
                "fi\n"
                f"rm -f {shlex.quote(final_output + '.done')}\n"
                + "".join(
                    f"{_completed_output(path)} || "
                    f"{{ echo {shlex.quote(f'incomplete generation chunk: {path}')} >&2; exit 1; }}\n"
                    for path in chunk_outputs
                )
                + "python -m nemo_skills.inference.merge_chunks "
                f"{shlex.quote(final_output)} "
                + " ".join(shlex.quote(path) for path in chunk_outputs)
            )
            if parameters.get("postprocess_cmd"):
                # A subshell makes set -e stop on a failure anywhere in an && chain.
                merge_command += f"\n(\n{parameters['postprocess_cmd']}\n)"
            merge_command += (
                f"\ntest -f {shlex.quote(final_output)}"
                f"\ntouch {shlex.quote(final_output + '.done')}"
            )
            merge_id = f"{context.stage}.{task_index}"
            task_index += 1
            tasks.append(
                TaskPlan(
                    task_id=merge_id,
                    stage=context.stage,
                    name=f"{context.expname}-merge" + ("" if seed is None else f"-rs{seed}"),
                    operation="nemo_skills.run_cmd",
                    commands=(
                        CommandPlan(
                            command=merge_command,
                            container=main_container,
                            runtime_role=main_container,
                        ),
                    ),
                    resources=(ResourceRequest(gpus_per_node=0),),
                    dependencies=tuple(DependencyPlan(task_id) for task_id in chunk_terminal_ids)
                    or upstream,
                    outputs=(
                        ArtifactDeclaration(
                            name="generation",
                            path=final_output,
                        ),
                    ),
                    parameters={
                        "experiment_name": context.expname,
                        "generation_invocation": invocation,
                        "log_dir": parameters.get("log_dir") or f"{output_dir}/generation-logs",
                    },
                )
            )
    return tuple(tasks)


def _launcher_on_every_node(
    parameters: dict[str, Any],
    arguments: str,
    *,
    models: list[Any],
    server_types: list[Any],
    server_addresses: list[Any],
    server_nodes: list[Any],
) -> tuple[list[str], dict[str, Any], str]:
    """Return each server's arguments, and the parameters and arguments NeMo-Skills submits.

    A vLLM server that spans nodes on the mp backend runs NVFlow's launcher on
    every node with its rank. NeMo-Skills wraps its own multi-node vLLM servers
    in ``ray start``, so on Slurm it runs the launcher as a generic server, and
    its client is told the server is vLLM.
    """
    count = len(models)
    server_args = [str(value or "") for value in _broadcast(parameters.get("server_args"), count)]
    types = list(server_types)
    containers = _broadcast(parameters.get("server_container"), count)
    entrypoints = _broadcast(parameters.get("server_entrypoint"), count)
    changed = False
    for index, model in enumerate(models):
        nodes = int(server_nodes[index] or 1)
        server_type = _string_value(types[index])
        if (
            model is None
            or server_addresses[index]
            or server_type != "vllm"
            or nodes <= 1
            or backend_in_args(server_args[index]) != "mp"
            or entrypoints[index]
        ):
            continue
        server_args[index] = f"{server_args[index]} {MP_NODE_ARGS}"
        # NeMo-Skills picks a server's container by its type unless given one.
        containers[index] = containers[index] or server_type
        types[index] = "generic"
        entrypoints[index] = VLLM_LAUNCHER
        changed = True
    if not changed:
        return server_args, parameters, arguments
    if count > 1:
        raise ValueError(
            f"vLLM's own multi-node mode runs a generation with one model; this one has {count}"
        )
    submitted = {
        **parameters,
        "server_type": types,
        "server_container": containers,
        "server_entrypoint": entrypoints,
        "server_args": server_args,
    }
    return server_args, submitted, f"++server.server_type=vllm {arguments}".strip()


def _generation_server_command(
    *,
    model: str,
    server_type: str,
    server_args: str,
    host_file: str,
    port_file: str,
    num_gpus: int,
    num_nodes: int,
) -> str:
    if server_type not in {"vllm", "sglang"}:
        raise ValueError(f"Unsupported supplied-Ray generation server type: {server_type}")
    return (
        "set -euo pipefail\n"
        f"mkdir -p {shlex.quote(str(Path(host_file).parent))}\n"
        f"rm -f {shlex.quote(host_file)} {shlex.quote(port_file)}\n"
        "SERVER_PORT=$(python - <<'PY'\n"
        "import socket\n"
        "with socket.socket() as sock:\n"
        "    sock.bind(('', 0))\n"
        "    print(sock.getsockname()[1])\n"
        "PY\n"
        ")\n"
        f"SERVER_HOST=$({NODE_ADDRESS_COMMAND})\n"
        'if [ -z "$SERVER_HOST" ]; then SERVER_HOST=$(hostname); fi\n'
        f"printf '%s\\n' \"$SERVER_HOST\" > {shlex.quote(host_file)}\n"
        f"printf '%s\\n' \"$SERVER_PORT\" > {shlex.quote(port_file)}\n"
        "python -m nvflow.integrations.model_servers.serve "
        f"--server-type {shlex.quote(server_type)} --model {shlex.quote(model)} "
        f'--num_gpus {num_gpus} --num_nodes {num_nodes} --port "$SERVER_PORT" '
        f"{server_args}"
    )


def _generation_client_command(
    *,
    output_dir: str,
    input_file: str | None,
    input_dir: str | None,
    arguments: str,
    seed: int | None,
    chunk: int | None,
    num_chunks: int | None,
    preprocess_cmd: object,
    postprocess_cmd: object,
    setup_command: object,
    server_addresses: list[str],
    models: list[str],
    server_types: list[str],
) -> tuple[str, str]:
    source = f"{input_dir}/output-rs{seed}.jsonl" if input_dir is not None else str(input_file)
    requested_output = _generation_output(output_dir, seed)
    output_file = _generation_output(output_dir, seed, chunk)
    complete = _completed_output(output_file)
    if chunk is not None:
        # merge_chunks deletes the chunk files once the merged output is complete.
        complete = f"{{ {complete}; }} || {{ {_completed_output(requested_output)}; }}"
    command = (
        "set -euo pipefail\n"
        "export HYDRA_FULL_ERROR=1\n"
        f"mkdir -p {shlex.quote(output_dir)}\n"
        f"if {complete}; then\n"
        "  echo 'generation output already complete; skipping'\n"
        "  exit 0\n"
        "fi\n"
        f"rm -f {shlex.quote(output_file + '.done')}\n"
    )
    if setup_command:
        if not isinstance(setup_command, str):
            raise ValueError("installation_command must be a string")
        command += f"{setup_command}\n"
    for index, address in enumerate(server_addresses):
        command += (
            f"NVFLOW_SERVER_{index}={address}\n"
            "for _attempt in $(seq 1 900); do\n"
            f'  curl -fsS -m 5 "$NVFLOW_SERVER_{index}/models" >/dev/null 2>&1 && break\n'
            "  [ \"$_attempt\" -eq 900 ] && { echo 'generation server not ready' >&2; exit 1; }\n"
            "  sleep 1\n"
            "done\n"
        )
    resolved_addresses = [f'"$NVFLOW_SERVER_{index}"' for index in range(len(server_addresses))]
    if preprocess_cmd:
        command += f"{preprocess_cmd}\n"
    command += (
        "python -m nemo_skills.inference.generate "
        f"++skip_filled=True ++input_file={shlex.quote(source)} "
        f"++output_file={shlex.quote(requested_output)} "
    )
    if len(resolved_addresses) == 1:
        command += (
            f"++server.base_url={resolved_addresses[0]} "
            f"++server.model={shlex.quote(models[0])} "
            f"++server.server_type={shlex.quote(server_types[0])} "
        )
    elif resolved_addresses:
        command += (
            f"++server.base_url=[{','.join(resolved_addresses)}] "
            f"++server.model=[{','.join(models)}] "
            f"++server.server_type=[{','.join(server_types)}] "
        )
    if seed is not None and input_dir is None:
        command += (
            f"++inference.random_seed={seed} ++inference.temperature=0.7 "
            "++inference.top_k=-1 ++inference.top_p=0.95 "
        )
    if chunk is not None:
        command += f"++num_chunks={num_chunks} ++chunk_id={chunk} "
    command += arguments
    command += f"\ntest -f {shlex.quote(output_file)}"
    if postprocess_cmd:
        # A subshell makes set -e stop on a failure anywhere in an && chain.
        command += f"\n(\n{postprocess_cmd}\n)"
    command += f"\ntouch {shlex.quote(output_file + '.done')}"
    return command, output_file


def _generation_output(output_dir: str, seed: int | None, chunk: int | None = None) -> str:
    """Return the file NeMo-Skills generation writes for one seed and chunk."""
    base = "output.jsonl" if seed is None else f"output-rs{seed}.jsonl"
    if chunk is not None:
        base = f"{base.removesuffix('.jsonl')}_chunk_{chunk}.jsonl"
    return f"{output_dir.rstrip('/')}/{base}"


def _completed_output(path: str) -> str:
    return f"[ -f {shlex.quote(path)} ] && [ -f {shlex.quote(path + '.done')} ]"


def _file_backed_endpoint(host_file: str, port_file: str) -> str:
    """Return a shell expression that waits for a complete service endpoint."""
    host = shlex.quote(host_file)
    port = shlex.quote(port_file)
    return (
        "$(for _attempt in $(seq 1 300); do "
        f"if [ -s {host} ] && [ -s {port} ]; then break; fi; "
        '[ "$_attempt" -eq 300 ] && { '
        "echo 'generation service endpoint was not published' >&2; exit 1; }; "
        "sleep 1; done; "
        f'printf \'http://%s:%s/v1\' "$(cat {host})" "$(cat {port})")'
    )


def _generation_seeds(parameters: dict[str, Any]) -> list[int | None]:
    explicit = parameters.get("random_seeds")
    count = parameters.get("num_random_seeds")
    if explicit and count:
        raise ValueError("Cannot specify random_seeds and num_random_seeds together")
    if explicit:
        if isinstance(explicit, str):
            return [int(value) for value in explicit.split(",") if value]
        return [int(value) for value in explicit]
    if count:
        start = int(parameters.get("starting_seed", 0))
        return list(range(start, start + int(count)))
    return [None]


def _generation_chunks(parameters: dict[str, Any]) -> list[int | None]:
    count = parameters.get("num_chunks")
    explicit = parameters.get("chunk_ids")
    if explicit:
        if isinstance(explicit, str):
            return [int(value) for value in explicit.split(",") if value]
        return [int(value) for value in explicit]
    if count:
        return list(range(int(count)))
    return [None]


def _list_value(value: object) -> list[Any]:
    return list(value) if isinstance(value, list | tuple) else [value]


def _broadcast(value: object, count: int) -> list[Any]:
    if isinstance(value, list | tuple):
        values = list(value)
        if len(values) == count:
            return values
        if len(values) == 1:
            return values * count
        raise ValueError(f"Cannot broadcast {len(values)} values to {count} models")
    return [value] * count


def _string_value(value: object) -> str:
    return str(getattr(value, "value", value))


def task_from_invocation(
    context: PlanningContext,
    *,
    index: int,
    operation: str,
    positional: tuple[Any, ...],
    keyword: dict[str, Any],
    arguments: str | None = None,
) -> TaskPlan:
    """Convert one captured helper invocation into a backend-neutral task."""
    ctx = keyword.get("ctx")
    if arguments is not None:
        rendered_arguments = arguments
    elif command := keyword.get("command"):
        rendered_arguments = str(command)
    else:
        rendered_arguments = " ".join(getattr(ctx, "args", []))
    if not rendered_arguments:
        raise ValueError(f"Captured {operation} invocation did not expose a command")
    parameters = {
        key: _json_safe(value)
        for key, value in keyword.items()
        if key not in {"ctx", "installation_command", "output_dir", "output_file"}
    }
    parameters["arguments"] = rendered_arguments
    if positional:
        parameters["positional"] = _json_safe(positional)

    stage_kwargs = dict(context.config.get("stage_kwargs", {}))
    resources = _resources(keyword, stage_kwargs)
    dependencies = tuple(
        DependencyPlan(
            str(name),
            condition=_dependency_condition(keyword),
            external=True,
        )
        for name in keyword.get("run_after") or context.run_after
    )
    declared_outputs = _DECLARED_TASK_OUTPUTS.get()
    outputs = (
        declared_outputs
        if declared_outputs is not None
        else _conventional_outputs(
            output_dir=keyword.get("output_dir"),
            output_file=keyword.get("output_file"),
        )
    )
    container = (
        keyword.get("container")
        or keyword.get("main_container")
        or stage_kwargs.get("container")
        or "nemo-skills"
    )
    setup_command = keyword.get("installation_command")
    if setup_command:
        if not isinstance(setup_command, str):
            raise ValueError("installation_command must be a string")
        rendered_arguments = f"set -e\n{setup_command}\n{rendered_arguments}"
    return TaskPlan(
        task_id=f"{context.stage}.{index}",
        stage=context.stage,
        name=str(keyword.get("expname", context.expname)),
        operation=operation,
        commands=(
            CommandPlan(
                command=rendered_arguments or operation,
                container=str(container),
                runtime_role=str(container),
            ),
        ),
        resources=resources,
        dependencies=dependencies,
        outputs=outputs,
        parameters=parameters,
    )


def _dependency_condition(
    keyword: dict[str, Any],
) -> Literal["after_success", "after_start", "after_any"]:
    dependency_type = (keyword.get("sbatch_kwargs") or {}).get("dependency_type")
    if dependency_type == "afterany":
        return "after_any"
    if dependency_type == "after":
        return "after_start"
    return "after_success"


def _conventional_outputs(
    *,
    output_dir: Any = None,
    output_file: Any = None,
) -> tuple[ArtifactDeclaration, ...]:
    values = [
        ("output_dir", output_dir, "directory"),
        ("output_file", output_file, "file"),
    ]
    present = [(key, value, kind) for key, value, kind in values if value]
    return tuple(
        ArtifactDeclaration(
            name="output" if len(present) == 1 else key,
            path=str(value),
            kind=kind,
        )
        for key, value, kind in present
    )


def _resources(
    keyword: dict[str, Any],
    stage_kwargs: dict[str, Any],
) -> tuple[ResourceRequest, ...]:
    source = {**stage_kwargs, **keyword}
    main = ResourceRequest(
        nodes=int(source.get("num_nodes", 1)),
        # A command without num_gpus is a CPU job; NeMo-Skills alone would request
        # one GPU on a cluster without cpu_partition.
        gpus_per_node=int(source.get("num_gpus") or 0),
        tasks_per_node=int(source.get("num_tasks", 1)),
        partition=source.get("partition"),
        account=source.get("account"),
        exclusive=source.get("exclusive"),
    )
    return (main,)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(child) for child in value]
    return str(value)
