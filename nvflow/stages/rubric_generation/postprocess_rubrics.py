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

"""Parse, validate, and publish portable rubric artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import METHODS, read_jsonl
from .criteria_dedup import dedup_criteria
from .parsing import RUBRIC_SCHEMA_VERSION, normalize_rubric, parse_json_object


def postprocess_rubrics(
    *,
    input_file: str | Path,
    output_file: str | Path,
    method: str,
    generation_key: str = "rubric_generation",
    prompt_id: str = "default",
    model_id: str = "unknown",
    dedup_criteria_enabled: bool = True,
) -> dict[str, Any]:
    """Write validated rubrics, deterministic rejects, and a provenance manifest."""

    if method not in METHODS:
        raise ValueError(f"method must be one of {sorted(METHODS)}")
    output_path = Path(output_file)
    rejected_path = output_path.with_name(f"{output_path.stem}_rejected.jsonl")
    manifest_path = output_path.with_name(f"{output_path.stem}.manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "schema_version": RUBRIC_SCHEMA_VERSION,
        "method": method,
        "total": 0,
        "valid": 0,
        "rejected": 0,
        "criteria": 0,
        "must_pass_criteria": 0,
        "severity_counts": {"1": 0, "2": 0, "3": 0},
        "errors": {},
        "dedup_criteria": dedup_criteria_enabled,
        "criteria_relations": {},
    }

    with (
        output_path.open("w", encoding="utf-8") as valid_stream,
        rejected_path.open("w", encoding="utf-8") as rejected_stream,
    ):
        for line_number, record in read_jsonl(input_file):
            stats["total"] += 1
            raw_generation = record.get(generation_key)
            try:
                if not isinstance(raw_generation, str):
                    raise ValueError(f"missing string field {generation_key}")
                rubric = normalize_rubric(parse_json_object(raw_generation))
                if dedup_criteria_enabled:
                    rubric, relations = dedup_criteria(rubric)
                    for name, count in relations.items():
                        stats["criteria_relations"][name] = (
                            stats["criteria_relations"].get(name, 0) + count
                        )
                output_record = {
                    "rubric_schema_version": RUBRIC_SCHEMA_VERSION,
                    "id": record.get("id"),
                    "question": record.get("question"),
                    "context": record.get("context"),
                    "reference_answer": record.get("reference_answer"),
                    "rubric_method": method,
                    "rubric": rubric,
                    "metadata": record.get("metadata") or {},
                    "provenance": {
                        "method": method,
                        "model": model_id,
                        "prompt": prompt_id,
                        "input_format": record.get("input_format", "canonical"),
                        "source_sha256": record.get("source_sha256"),
                        "context_sha256": record.get("context_sha256"),
                    },
                    "rubric_generation_raw": raw_generation,
                }
                valid_stream.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                stats["valid"] += 1
                stats["criteria"] += len(rubric)
                stats["must_pass_criteria"] += sum(item["must_pass"] for item in rubric)
                for item in rubric:
                    stats["severity_counts"][str(item["severity"])] += 1
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                error = str(exc)
                stats["rejected"] += 1
                stats["errors"][error] = stats["errors"].get(error, 0) + 1
                rejected_stream.write(
                    json.dumps(
                        {
                            "line_number": line_number,
                            "id": record.get("id"),
                            "question": record.get("question"),
                            "rubric_method": method,
                            "rubric_error": error,
                            "rubric_generation_raw": raw_generation,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    stats["valid_rate"] = stats["valid"] / stats["total"] if stats["total"] else 0.0
    stats["output_sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
    stats["model"] = model_id
    stats["prompt"] = prompt_id
    manifest_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if stats["valid"] == 0:
        raise ValueError("rubric generation produced no structurally valid rubrics")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", "--input_file", required=True)
    parser.add_argument("--output-file", "--output_file", required=True)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--generation-key", "--generation_key", default="rubric_generation")
    parser.add_argument("--prompt-id", "--prompt_id", default="default")
    parser.add_argument("--model-id", "--model_id", default="unknown")
    parser.add_argument(
        "--no-dedup-criteria",
        "--no_dedup_criteria",
        action="store_true",
        help="keep criteria that restate another criterion in the same rubric",
    )
    args = parser.parse_args()
    stats = postprocess_rubrics(
        input_file=args.input_file,
        output_file=args.output_file,
        method=args.method,
        generation_key=args.generation_key,
        prompt_id=args.prompt_id,
        model_id=args.model_id,
        dedup_criteria_enabled=not args.no_dedup_criteria,
    )
    print(json.dumps(stats, sort_keys=True))


if __name__ == "__main__":
    main()
