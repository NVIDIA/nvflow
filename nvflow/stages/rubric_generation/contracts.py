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

"""Canonical input contract used by all rubric-generation backends."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .input_adapters import adapt_input_record, normalize_input_format

METHODS = {"simple", "openrubric_contrastive"}

DEFAULT_FIELD_MAP: dict[str, tuple[str, ...]] = {
    "id": ("id", "uuid", "source_id"),
    "question": ("question", "problem"),
    "context": ("context", "source_context", "evidence"),
    "reference_answer": ("reference_answer", "expected_answer", "answer"),
    "candidate_answers": ("candidate_answers", "solutions", "answers"),
    "preferred_index": ("preferred_index", "selected_index", "chosen_index"),
    "preferred_answer": ("preferred_answer", "chosen_answer"),
    "rejected_answer": ("rejected_answer",),
    "metadata": ("metadata",),
}


def _get_path(record: Mapping[str, Any], path: str) -> Any:
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _field(
    record: Mapping[str, Any],
    canonical: str,
    field_map: Mapping[str, str | list[str] | tuple[str, ...]],
) -> Any:
    configured = field_map.get(canonical, DEFAULT_FIELD_MAP[canonical])
    paths = (configured,) if isinstance(configured, str) else tuple(configured)
    for path in paths:
        value = _get_path(record, str(path))
        if value is not None:
            return value
    return None


def _text(value: Any, field: str, *, allow_structured: bool = False) -> str:
    if allow_structured and isinstance(value, list | dict):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, *, allow_structured: bool = False) -> str:
    if allow_structured and isinstance(value, list | dict):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value.strip() if isinstance(value, str) else ""


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_record(
    record: Mapping[str, Any],
    *,
    method: str,
    input_format: str = "canonical",
    field_map: Mapping[str, str | list[str] | tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Normalize one recipe record into the portable rubric input contract."""

    if method not in METHODS:
        raise ValueError(f"method must be one of {sorted(METHODS)}")
    source_record = record
    normalized_input_format = normalize_input_format(input_format)
    record = adapt_input_record(record, input_format=normalized_input_format)
    mappings = field_map or {}
    question = _text(_field(record, "question", mappings), "question")
    # Not required yet. Contrastive generation grounds itself on the answers it
    # is given, so it can run without a source document; simple cannot. The
    # check moves below, once the method is known.
    context = _optional_text(_field(record, "context", mappings), allow_structured=True)

    preferred_answer = _field(record, "preferred_answer", mappings)
    rejected_answer = _field(record, "rejected_answer", mappings)
    candidate_answers = _field(record, "candidate_answers", mappings)
    preferred_index = _field(record, "preferred_index", mappings)
    reference_answer = _field(record, "reference_answer", mappings)

    if preferred_answer is not None:
        preferred_answer = _text(preferred_answer, "preferred_answer")
    if rejected_answer is not None:
        rejected_answer = _text(rejected_answer, "rejected_answer")

    normalized_candidates: list[str] | None = None
    original_candidate_indices: list[int] = []
    if candidate_answers is not None:
        if not isinstance(candidate_answers, list):
            raise ValueError("candidate_answers must be a list")
        normalized_candidates = []
        for original_index, value in enumerate(candidate_answers):
            if isinstance(value, str) and value.strip():
                normalized_candidates.append(value.strip())
                original_candidate_indices.append(original_index)
        # A question with a single rollout is still usable by contrastive
        # generation: there is nothing to rank, so the negative gets written
        # instead of chosen. simple ignores candidates entirely.
        minimum = 1 if method == "openrubric_contrastive" else 2
        if len(normalized_candidates) < minimum:
            raise ValueError(f"candidate_answers must contain at least {minimum} non-empty answers")

    if preferred_index is not None:
        if not isinstance(preferred_index, int) or isinstance(preferred_index, bool):
            raise ValueError("preferred_index must be an integer")
        if normalized_candidates is None or preferred_index not in original_candidate_indices:
            raise ValueError("preferred_index is outside candidate_answers")
        preferred_index = original_candidate_indices.index(preferred_index)
        preferred_answer = normalized_candidates[preferred_index]

    if reference_answer is None and preferred_answer is not None:
        reference_answer = preferred_answer
    reference = _text(reference_answer, "reference_answer")

    synthetic_negative_required = False
    if method == "openrubric_contrastive":
        has_pair = preferred_answer is not None and rejected_answer is not None
        has_candidates = normalized_candidates is not None and preferred_index is not None
        if has_pair and preferred_answer == rejected_answer:
            raise ValueError("preferred_answer and rejected_answer must differ")
        if has_pair and normalized_candidates is None:
            # Route explicit pairs through the same contrastive preparation path.
            normalized_candidates = [preferred_answer, rejected_answer]
            preferred_index = 0
        elif not has_candidates:
            # A reference-only record is still a valid generic OpenRubric input.
            # The method-preparation model will generate a controlled negative.
            preferred_answer = reference
            normalized_candidates = [reference]
            preferred_index = 0
            synthetic_negative_required = True
        elif len(normalized_candidates) == 1:
            if rejected_answer is not None:
                # A lone candidate plus a supplied negative is just a pair.
                normalized_candidates.append(rejected_answer)
            else:
                synthetic_negative_required = True

    if not context and method != "openrubric_contrastive":
        # simple writes criteria straight from the document, so without one it
        # has nothing to write them from. Contrastive always has an answer to
        # work from, so context is optional there -- see the note on quality in
        # the contrastive prompt.
        raise ValueError("context must be a non-empty string")

    raw_id = _field(record, "id", mappings)
    record_id = str(raw_id).strip() if raw_id is not None else ""
    if not record_id:
        record_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"nvflow-rubric:{question}:{context}"))

    metadata = _field(record, "metadata", mappings)
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be an object when supplied")

    output: dict[str, Any] = {
        "id": record_id,
        "question": question,
        "context": context,
        "reference_answer": reference,
        "metadata": dict(metadata),
        "source_sha256": _sha256(
            json.dumps(source_record, ensure_ascii=False, sort_keys=True, default=str)
        ),
        "context_sha256": _sha256(context),
        "input_format": normalized_input_format,
    }
    if normalized_candidates is not None:
        output["candidate_answers"] = normalized_candidates
        output["candidate_count"] = len(normalized_candidates)
        output["candidate_indices"] = list(range(len(normalized_candidates)))
    if preferred_index is not None:
        output["preferred_index"] = preferred_index
    if preferred_answer is not None:
        output["preferred_answer"] = preferred_answer
    if rejected_answer is not None:
        output["rejected_answer"] = rejected_answer
    if synthetic_negative_required:
        output["synthetic_negative_required"] = True
    return output


def normalize_file(
    *,
    input_file: str | Path,
    output_file: str | Path,
    audit_file: str | Path,
    method: str,
    input_format: str = "canonical",
    field_map: Mapping[str, str | list[str] | tuple[str, ...]] | None = None,
    allow_invalid_records: bool = False,
    max_records: int | None = None,
) -> dict[str, Any]:
    """Normalize JSONL, write deterministic rejects and a machine-readable audit."""

    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be a positive integer")
    normalized_input_format = normalize_input_format(input_format)

    output_path = Path(output_file)
    rejected_path = output_path.with_name(f"{output_path.stem}_rejected.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    stats: dict[str, Any] = {
        "schema_version": 1,
        "method": method,
        "input_format": normalized_input_format,
        "total": 0,
        "valid": 0,
        "rejected": 0,
        "errors": {},
    }

    with (
        Path(input_file).open(encoding="utf-8") as source,
        output_path.open("w", encoding="utf-8") as valid_stream,
        rejected_path.open("w", encoding="utf-8") as rejected_stream,
    ):
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            if max_records is not None and stats["total"] >= max_records:
                break
            stats["total"] += 1
            raw: Any = None
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("record must be a JSON object")
                normalized = normalize_record(
                    raw,
                    method=method,
                    input_format=normalized_input_format,
                    field_map=field_map,
                )
                if normalized["id"] in seen_ids:
                    raise ValueError(f"duplicate id: {normalized['id']}")
                seen_ids.add(normalized["id"])
                valid_stream.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                stats["valid"] += 1
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                error = str(exc)
                stats["rejected"] += 1
                stats["errors"][error] = stats["errors"].get(error, 0) + 1
                rejected_stream.write(
                    json.dumps(
                        {"line_number": line_number, "record": raw, "error": error},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    audit_path = Path(audit_file)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if stats["valid"] == 0:
        raise ValueError("no valid rubric input records were produced")
    if stats["rejected"] and not allow_invalid_records:
        raise ValueError(
            f"rubric input normalization rejected {stats['rejected']}/{stats['total']} records; "
            f"see {rejected_path}"
        )
    return stats


def read_jsonl(path: str | Path) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield object records with their one-based source line number."""

    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield line_number, value
