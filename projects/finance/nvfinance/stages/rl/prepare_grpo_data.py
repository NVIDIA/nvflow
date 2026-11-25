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

"""Attach generated rubrics to finance GRPO records.

The stock finance verifier receives one ``expected_answer`` string.  We encode
the reference answer and instance rubric together in that field and select a
rubric-aware judge prompt through a Gym config overlay.  This keeps the
existing, battle-tested finance tool server and its submit-final-result hard
gate unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import shlex
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nvflow.stages.data.split_contract import publish_split_marker

logger = logging.getLogger(__name__)

if __name__ != "__main__":
    from nvflow.execution import PlannedExecutionStage
    from nvflow.execution.commands import build_python_cmd

    if TYPE_CHECKING:
        from nvflow.execution import PlanningContext, StagePlan


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            records.append(value)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def _problem(record: dict[str, Any]) -> str:
    value = record.get("problem") or record.get("question")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Record has no non-empty problem/question")
    return value.strip()


def _record_id(record: dict[str, Any]) -> str | None:
    """Return the record's stable identifier under either contract's key."""
    value = record.get("uuid") if record.get("uuid") is not None else record.get("id")
    return None if value is None else str(value)


def _validated_rubric(record: dict[str, Any]) -> list[dict[str, Any]]:
    rubric = record.get("rubric")
    if not isinstance(rubric, list) or not rubric:
        raise ValueError(f"Missing rubric for {_problem(record)[:100]}")
    for item in rubric:
        if not isinstance(item, dict) or not isinstance(item.get("criterion"), str):
            raise ValueError("Every rubric item must contain a criterion")
    return rubric


def _task_spec(reference_answer: str, rubric: list[dict[str, Any]]) -> str:
    return json.dumps(
        {"reference_answer": reference_answer, "rubric": rubric},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _replace_question(prompt: str, problem: str) -> str:
    marker = "\nQuestion:\n"
    if marker not in prompt:
        raise ValueError("Template prompt does not contain the expected Question marker")
    return prompt.rsplit(marker, 1)[0] + marker + problem + "\n"


def _from_template(rubric_record: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
    problem = _problem(rubric_record)
    reference = rubric_record.get("reference_answer")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"Missing reference answer for {problem[:100]}")
    rubric = _validated_rubric(rubric_record)
    record = copy.deepcopy(template)
    row_uuid = _record_id(rubric_record) or str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"nvflow-rubric-smoke:{problem}")
    )
    prompt = _replace_question(str(record["prompt"]), problem)
    record.update(
        {
            "uuid": row_uuid,
            "problem": problem,
            "question": problem,
            "prompt": prompt,
            "context": "",
            "generation": "",
            "reasoning_content": "",
            "expected_answer": _task_spec(reference.strip(), rubric),
            "question_type": rubric_record.get("question_type"),
            "difficulty_profile": rubric_record.get("difficulty_profile")
            or record.get("difficulty_profile"),
        }
    )
    params = copy.deepcopy(record["responses_create_params"])
    params["input"] = [{"role": "user", "content": prompt}]
    record["responses_create_params"] = params
    metadata = dict(record.get("metadata") or {})
    metadata.update(
        {
            "uuid": row_uuid,
            "question_type": rubric_record.get("question_type"),
            "rubric": rubric,
            "rubric_method": rubric_record.get("rubric_method"),
            # The template contributes every other metadata field, so this one
            # has to be replaced or every row inherits the template's answer.
            "reference_answer": reference.strip(),
        }
    )
    record["metadata"] = metadata
    return record


def attach_rubric(base: dict[str, Any], rubric_record: dict[str, Any]) -> dict[str, Any]:
    """Encode the base reference answer and instance rubric into ``expected_answer``."""
    rubric = _validated_rubric(rubric_record)
    reference = base.get("expected_answer")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"Missing expected answer for {_problem(base)[:100]}")
    record = copy.deepcopy(base)
    record["expected_answer"] = _task_spec(reference.strip(), rubric)
    metadata = dict(record.get("metadata") or {})
    metadata.update(
        {
            "rubric": rubric,
            "rubric_method": rubric_record.get("rubric_method"),
            "reference_answer": reference.strip(),
        }
    )
    record["metadata"] = metadata
    return record


def _attach_to_base(
    rubric_records: list[dict[str, Any]],
    base_records: list[dict[str, Any]],
    *,
    allow_missing_rubrics: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    by_uuid = {
        _record_id(record): record for record in rubric_records if _record_id(record) is not None
    }
    by_problem = {_problem(record): record for record in rubric_records}
    output: list[dict[str, Any]] = []
    missing: list[str] = []
    matched_rubric_ids: set[int] = set()
    for base in base_records:
        problem = _problem(base)
        record_id = _record_id(base)
        rubric_record = by_uuid.get(record_id) if record_id is not None else None
        if rubric_record is None:
            rubric_record = by_problem.get(problem)
            if rubric_record is not None:
                logger.warning(
                    "Matched rubric by exact problem text because no matching uuid/id "
                    "was found: %s",
                    problem[:100],
                )
        if rubric_record is None:
            missing.append(problem)
            continue
        matched_rubric_ids.add(id(rubric_record))
        output.append(attach_rubric(base, rubric_record))
    unmatched_rubrics = [
        _problem(record) for record in rubric_records if id(record) not in matched_rubric_ids
    ]
    if unmatched_rubrics:
        raise ValueError(
            f"Rubric rows do not match the base data: {len(unmatched_rubrics)}; "
            f"first unmatched question: {unmatched_rubrics[0][:100]}"
        )
    if missing and not allow_missing_rubrics:
        raise ValueError(
            f"Rubric join incomplete: {len(output)}/{len(base_records)}; "
            f"first missing question: {missing[0][:100]}"
        )
    return output, missing


def prepare_grpo_data(
    *,
    rubric_file: str | Path | list[str | Path],
    output_dir: str | Path,
    base_rl_file: str | Path | None = None,
    template_rl_file: str | Path | None = None,
    validation_size: int = 10,
    seed: int = 42,
    allow_missing_rubrics: bool = False,
    env_name: str = "finance_sec_search",
) -> dict[str, int]:
    """Create rubric-aware ``train.jsonl`` and ``val.jsonl`` files.

    ``env_name`` is the NeMo-Gym environment the split is written for; training
    resolves its data as ``{data_source_dir}/{env_name}/train.jsonl``.
    """

    rubric_files = rubric_file if isinstance(rubric_file, list) else [rubric_file]
    rubrics = [record for path in rubric_files for record in _read_jsonl(path)]
    base_record_count: int | None = None
    missing_rubrics: list[str] = []
    if base_rl_file:
        base_records = _read_jsonl(base_rl_file)
        base_record_count = len(base_records)
        records, missing_rubrics = _attach_to_base(
            rubrics,
            base_records,
            allow_missing_rubrics=allow_missing_rubrics,
        )
    elif template_rl_file:
        template = _read_jsonl(template_rl_file)[0]
        records = [_from_template(record, template) for record in rubrics]
    else:
        raise ValueError("base_rl_file or template_rl_file is required")

    if validation_size < 1 or validation_size >= len(records):
        raise ValueError("validation_size must be between 1 and record_count - 1")
    positions = list(range(len(records)))
    random.Random(seed).shuffle(positions)
    validation_positions = set(positions[:validation_size])
    train = [record for index, record in enumerate(records) if index not in validation_positions]
    validation = [record for index, record in enumerate(records) if index in validation_positions]

    output_path = Path(output_dir) / env_name
    output_path.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("val.jsonl", validation)):
        with (output_path / name).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = {
        "rubrics": len(rubrics),
        "records": len(records),
        "train": len(train),
        "val": len(validation),
    }
    if base_record_count is not None:
        stats.update(
            {
                "base_records": base_record_count,
                "dropped_missing_rubric": len(missing_rubrics),
            }
        )
    (output_path / "rubric_data_audit.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    publish_split_marker(output_path)
    print(json.dumps(stats, sort_keys=True))
    return stats


if __name__ != "__main__":

    def build_grpo_data_command(config: dict[str, Any]) -> str:
        """Build the portable worker command for one GRPO hand-off stage."""

        rubric_files = config.get("rubric_files") or [config["rubric_file"]]
        flags: dict[str, Any] = {
            "rubric_file": rubric_files[0],
            "output_dir": config["output_dir"],
            "validation_size": config.get("validation_size", 10),
            "seed": config.get("seed", 42),
            "env_name": config.get("env_name", "finance_sec_search"),
        }
        for key in ("base_rl_file", "template_rl_file"):
            if config.get(key):
                flags[key] = config[key]
        command = build_python_cmd("nvfinance.stages.rl.prepare_grpo_data", **flags)
        for rubric_file in rubric_files[1:]:
            command += f" --rubric_file {shlex.quote(str(rubric_file))}"
        if config.get("allow_missing_rubrics"):
            command += " --allow-missing-rubrics"
        return command

    class PrepareRubricGRPODataStage(PlannedExecutionStage):
        """Attach validated rubrics to GRPO records for rubric-aware judging."""

        capture_strategy = "run_cmd"

        def validate_config(self, config: dict[str, Any]) -> None:
            if not config.get("output_dir"):
                raise ValueError("'output_dir' is required for the rubric GRPO hand-off")
            if bool(config.get("rubric_file")) + bool(config.get("rubric_files")) != 1:
                raise ValueError("Exactly one of rubric_file or rubric_files is required")
            if config.get("rubric_files") and not all(config["rubric_files"]):
                raise ValueError("rubric_files must contain only non-empty paths")
            if bool(config.get("base_rl_file")) + bool(config.get("template_rl_file")) != 1:
                raise ValueError("Exactly one of base_rl_file or template_rl_file is required")

        def _parameters(self, config: dict[str, Any], cluster: str, expname: str) -> dict[str, Any]:
            parameters: dict[str, Any] = {
                "cluster": cluster,
                "num_gpus": 0,
                "log_dir": config.get("log_dir") or f"{config['output_dir']}/logs",
                "expname": expname,
            }
            for key in ("container", "installation_command"):
                if config.get(key):
                    parameters[key] = config[key]
            return parameters

        def plan(self, context: PlanningContext) -> StagePlan:
            from nvflow.execution import ArtifactDeclaration, declare_task_outputs, plan_run_cmd

            config = dict(context.config)
            env_name = config.get("env_name", "finance_sec_search")
            parameters = self._parameters(config, context.target.cluster, context.expname)
            parameters["run_after"] = list(context.run_after) or None
            with declare_task_outputs(
                ArtifactDeclaration("train", f"{config['output_dir']}/{env_name}/train.jsonl"),
                ArtifactDeclaration("validation", f"{config['output_dir']}/{env_name}/val.jsonl"),
                ArtifactDeclaration("completion", f"{config['output_dir']}/{env_name}/split.done"),
            ):
                return plan_run_cmd(context, build_grpo_data_command(config), **parameters)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rubric-file",
        "--rubric_file",
        action="append",
        required=True,
        help="Validated rubric JSONL. Repeat to combine base and retry outputs.",
    )
    parser.add_argument("--output-dir", "--output_dir", required=True)
    parser.add_argument("--base-rl-file", "--base_rl_file")
    parser.add_argument("--template-rl-file", "--template_rl_file")
    parser.add_argument("--validation-size", "--validation_size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--env-name",
        "--env_name",
        default="finance_sec_search",
        help="NeMo-Gym environment name used as the output subdirectory",
    )
    parser.add_argument(
        "--allow-missing-rubrics",
        action="store_true",
        help="Drop base rows whose generated rubric failed validation",
    )
    args = parser.parse_args()
    prepare_grpo_data(
        rubric_file=args.rubric_file,
        output_dir=args.output_dir,
        base_rl_file=args.base_rl_file,
        template_rl_file=args.template_rl_file,
        validation_size=args.validation_size,
        seed=args.seed,
        allow_missing_rubrics=args.allow_missing_rubrics,
        env_name=args.env_name,
    )


if __name__ == "__main__":
    main()
