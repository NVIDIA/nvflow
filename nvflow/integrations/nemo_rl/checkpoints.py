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
"""Resolve a training checkpoint to a servable HuggingFace directory.

vLLM serves HuggingFace directories; training writes Megatron or FSDP shards.
Anything that scores a checkpoint needs the same bridge, so every evaluation
path goes through here.

Conversion is planned as a separate task in the ``nemo-rl`` container. Its
script is idempotent and reuses an existing HuggingFace export.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nvflow.execution import TaskPlan


@dataclass(frozen=True)
class CheckpointSourceDescriptor:
    """One resolved checkpoint source shared by planning and execution."""

    checkpoint_root: Path
    step: int
    checkpoint_format: str
    source_path: Path
    hf_model_path: Path
    convert_log_dir: Path


def describe_checkpoint_source(
    config: dict[str, Any],
    step: int,
) -> CheckpointSourceDescriptor:
    """Resolve source, output, and log paths once for all conversion paths."""
    from nvflow.integrations.nemo_rl.checkpoint_converter import get_hf_output_paths

    if not isinstance(step, int):
        raise ValueError(f"checkpoint eval steps must be training step numbers, got {step!r}")
    checkpoint_format = config.get("format", "megatron")
    if checkpoint_format not in ("megatron", "fsdp"):
        raise ValueError(
            f"checkpoint format must be 'megatron' or 'fsdp', got {checkpoint_format!r}"
        )

    checkpoint_root = Path(config["checkpoint_path"])
    hf_model_path, convert_log_dir = get_hf_output_paths(checkpoint_root, step)
    source_path = (
        checkpoint_root
        if checkpoint_format == "fsdp"
        else checkpoint_root / "checkpoints" / f"step_{step}"
    )
    return CheckpointSourceDescriptor(
        checkpoint_root=checkpoint_root,
        step=step,
        checkpoint_format=checkpoint_format,
        source_path=source_path,
        hf_model_path=hf_model_path,
        convert_log_dir=convert_log_dir,
    )


def conversion_gpus(
    config: dict[str, Any],
    *,
    gpus_per_node: int | None = None,
) -> int:
    """GPU count for a conversion job.

    Conversion needs one GPU functionally, but a whole node is the default so
    the job clears per-job GRES minimums (e.g. a GB200 QOS floor of gpu:4).
    """
    conversion = config.get("conversion") or {}
    if "num_gpus" in conversion:
        return conversion["num_gpus"]
    if gpus_per_node is not None:
        return gpus_per_node
    return 8


def plan_checkpoint_conversion(
    config: dict[str, Any],
    step: int,
    cluster: str,
    expname: str,
    run_after: list[str] | None,
    *,
    gpus_per_node: int | None = None,
    require_gpu_topology: bool = False,
) -> tuple[str, TaskPlan]:
    """Describe checkpoint conversion without submitting it."""
    from nvflow.execution import (
        ArtifactDeclaration,
        CommandPlan,
        DependencyPlan,
        ResourceRequest,
        TaskPlan,
    )
    from nvflow.integrations.nemo_rl.checkpoint_converter import (
        build_conversion_script,
        build_dcp_conversion_script,
    )

    source = describe_checkpoint_source(config, step)
    conversion = config.get("conversion") or {}
    if require_gpu_topology and gpus_per_node is None and "num_gpus" not in conversion:
        raise ValueError(
            "Ray target route 'nemo-rl' must declare gpus_per_node for checkpoint conversion"
        )
    installation_command = conversion.get("installation_command", "true")
    num_gpus = conversion_gpus(config, gpus_per_node=gpus_per_node)
    if source.checkpoint_format == "fsdp":
        script = build_dcp_conversion_script(
            source.source_path,
            source.step,
            source.hf_model_path,
        )
    else:
        base_model = config.get("base_model")
        if not base_model:
            raise ValueError("base_model is required to plan a Megatron checkpoint conversion")
        script = build_conversion_script(
            source.source_path,
            source.hf_model_path,
            base_model,
        )

    conversion_name = f"{expname}-convert"
    command = f"bash -c '{script}'"
    if installation_command:
        command = f"set -e\n{installation_command}\n{command}"
    return (
        str(source.hf_model_path),
        TaskPlan(
            task_id=f"{expname}.convert",
            stage="checkpoint_conversion",
            name=conversion_name,
            operation="nemo_skills.run_cmd",
            commands=(
                CommandPlan(
                    command=command,
                    container="nemo-rl",
                    runtime_role="nemo-rl",
                ),
            ),
            resources=(ResourceRequest(gpus_per_node=num_gpus),),
            dependencies=tuple(DependencyPlan(name, external=True) for name in run_after or ()),
            outputs=(
                ArtifactDeclaration(
                    name="hf_model",
                    path=str(source.hf_model_path),
                    kind="model",
                ),
            ),
            parameters={
                "arguments": command,
                "cluster": cluster,
                "container": "nemo-rl",
                "num_gpus": num_gpus,
                "log_dir": str(source.convert_log_dir),
                "expname": conversion_name,
                "run_after": run_after,
            },
        ),
    )
