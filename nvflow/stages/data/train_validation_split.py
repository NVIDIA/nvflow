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
"""Split one JSONL dataset into ``train.jsonl``, ``val.jsonl``, and ``split.done``.

With ``splits``, the stage writes one file per named split, ``manifest.jsonl``,
and ``splits.done`` instead.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from nvflow.execution import ArtifactDeclaration, PlannedExecutionStage, declare_task_outputs

if TYPE_CHECKING:
    from nvflow.execution import PlanningContext, StagePlan

_SPLIT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_RESERVED_SPLIT_NAMES = {"manifest", "completion", "splits"}


def split_command(*, input_file: str, output_dir: str, config: Mapping[str, Any]) -> str:
    """Render the splitter command for one input file."""
    command = (
        f"python -m nvflow.stages.data.dataset_splitter "
        f"    '{input_file}' "
        f"    --output_dir '{output_dir}' "
    )
    if not config.get("splits"):
        command += f"    --val_ratio {config.get('val_ratio', 0.1)} "
    if stratify_by := config.get("stratify_by"):
        fields = [stratify_by] if isinstance(stratify_by, str) else list(stratify_by)
        command += "    --stratify_by " + " ".join(f"'{field}'" for field in fields) + " "
    command += f"    --random_seed {config.get('random_seed', 42)}"
    if max_token_length := config.get("max_token_length"):
        command += f" --max_token_length {max_token_length}"
    if token_length_field := config.get("token_length_field"):
        command += f" --token_length_field '{token_length_field}'"
    if fields := config.get("fields"):
        command += " --fields " + " ".join(f"'{field}'" for field in fields)
    if sort_by := config.get("sort_by"):
        command += f" --sort_by '{sort_by}' --sort_order {config.get('sort_order', 'desc')}"
    if group_by := config.get("group_by"):
        command += f" --group_by '{group_by}'"
    if splits := config.get("splits"):
        command += " --splits " + " ".join(f"'{name}={ratio}'" for name, ratio in splits.items())
    if id_field := config.get("id_field"):
        command += f" --id_field '{id_field}'"
    return command


def _field_names(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value)
    return (
        isinstance(value, list | tuple)
        and bool(value)
        and all(isinstance(field, str) and field for field in value)
    )


def _validate_named_splits(config: Mapping[str, Any]) -> None:
    splits = config["splits"]
    if not isinstance(splits, Mapping) or len(splits) < 2:
        raise ValueError("'splits' must map at least two split names to ratios")
    for name, ratio in splits.items():
        if (
            not isinstance(name, str)
            or not _SPLIT_NAME.fullmatch(name)
            or name in _RESERVED_SPLIT_NAMES
        ):
            raise ValueError(f"invalid split name {name!r}")
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, int | float)
            or not math.isfinite(ratio)
            or ratio < 0
        ):
            raise ValueError(f"split {name!r} needs a non-negative ratio")
    if abs(sum(splits.values()) - 1) > 1e-6:
        raise ValueError("'splits' ratios must sum to 1")
    if "val_ratio" in config:
        raise ValueError("set 'splits' or 'val_ratio', not both")
    if config.get("sort_by"):
        raise ValueError("'sort_by' applies to train/validation splits, not 'splits'")


def validate_split_options(config: Mapping[str, Any]) -> None:
    """Validate the splitter options shared by every split stage."""
    val_ratio = config.get("val_ratio", 0.1)
    if (
        isinstance(val_ratio, bool)
        or not isinstance(val_ratio, int | float)
        or not 0 <= val_ratio <= 1
    ):
        raise ValueError(f"'val_ratio' must be between 0 and 1 (inclusive), got {val_ratio!r}")
    seed = config.get("random_seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("'random_seed' must be an integer")
    max_tokens = config.get("max_token_length")
    if max_tokens is not None and (
        isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1
    ):
        raise ValueError("'max_token_length' must be a positive integer")
    fields = config.get("fields")
    if fields is not None and (
        not isinstance(fields, list | tuple)
        or not fields
        or not all(isinstance(field, str) and field for field in fields)
    ):
        raise ValueError("'fields' must be a non-empty list of field names")
    if config.get("max_token_length") and not config.get("token_length_field"):
        raise ValueError("'max_token_length' requires 'token_length_field'")
    if config.get("sort_order", "desc") not in {"asc", "desc"}:
        raise ValueError("'sort_order' must be 'asc' or 'desc'")
    if config.get("stratify_by") is not None and not _field_names(config["stratify_by"]):
        raise ValueError("'stratify_by' must be a field name or a non-empty list of field names")
    for key in ("group_by", "id_field"):
        if config.get(key) is not None and not (isinstance(config[key], str) and config[key]):
            raise ValueError(f"'{key}' must be a field name")
    if config.get("id_field") is not None and config.get("splits") is None:
        raise ValueError("'id_field' applies only with 'splits'")
    if config.get("splits") is not None:
        _validate_named_splits(config)


def _declared_outputs(
    output_dir: str, config: Mapping[str, Any]
) -> tuple[ArtifactDeclaration, ...]:
    if splits := config.get("splits"):
        return (
            *(ArtifactDeclaration(name, f"{output_dir}/{name}.jsonl") for name in splits),
            ArtifactDeclaration("manifest", f"{output_dir}/manifest.jsonl"),
            ArtifactDeclaration("completion", f"{output_dir}/splits.done"),
        )
    return (
        ArtifactDeclaration("train", f"{output_dir}/train.jsonl"),
        ArtifactDeclaration("validation", f"{output_dir}/val.jsonl"),
        ArtifactDeclaration("completion", f"{output_dir}/split.done"),
    )


class TrainValidationSplitStage(PlannedExecutionStage):
    """Split ``input_file`` into train and validation files under ``output_dir``.

    Optional ``stratify_by`` keeps each value's share in both splits, ``fields``
    projects records, ``max_token_length`` drops records whose
    ``token_length_field`` exceeds it, and ``sort_by`` orders the train file
    instead of shuffling it. ``group_by`` keeps records sharing a value in one
    split, and ``splits`` replaces train and validation with named splits.
    """

    def plan(self, context: PlanningContext) -> StagePlan:
        from nvflow.execution import plan_run_cmd

        config = context.config
        input_file = config["input_file"]
        output_dir = config["output_dir"]
        run_after = list(context.run_after) or None
        if context.live and not run_after:
            assert context.filesystem is not None
            if not context.filesystem.batch_exists([input_file]).get(input_file, False):
                raise ValueError(
                    "live train/validation split plan: required input artifact(s) are missing: "
                    + input_file
                )
        with declare_task_outputs(*_declared_outputs(output_dir, config)):
            return plan_run_cmd(
                context,
                split_command(input_file=input_file, output_dir=output_dir, config=config),
                cluster=context.target.cluster,
                container=str(config.get("container") or "nemo-skills"),
                installation_command=config.get("installation_command"),
                log_dir=f"{output_dir}/logs",
                output_dir=output_dir,
                expname=context.expname,
                run_after=run_after,
                sbatch_kwargs=None,
            )

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate that required configuration fields are present."""
        for field in ("input_file", "output_dir"):
            if field not in config:
                raise ValueError(f"'{field}' is required in train_validation_split config")
        validate_split_options(config)
