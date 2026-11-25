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

"""Validate answer rankings and produce contrastive rubric inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .contracts import read_jsonl
from .parsing import parse_json_object


def _write(stream: Any, record: dict[str, Any]) -> None:
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def postprocess_rankings(
    *,
    input_file: str | Path,
    output_file: str | Path,
    rankings_file: str | Path,
    audit_file: str | Path,
    generation_key: str = "ranking_generation",
) -> dict[str, Any]:
    """Emit only materially contrastive pairs with a valid preferred candidate."""

    output_path = Path(output_file)
    rankings_path = Path(rankings_file)
    rejected_path = output_path.with_name(f"{output_path.stem}_rejected.jsonl")
    for path in (output_path, rankings_path, rejected_path, Path(audit_file)):
        path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "total": 0,
        "accepted": 0,
        "no_material_contrast": 0,
        "invalid": 0,
        "preferred_ranked_last": 0,
        "synthetic_negative": 0,
        "synthetic_fallback": 0,
    }
    with (
        output_path.open("w", encoding="utf-8") as output_stream,
        rankings_path.open("w", encoding="utf-8") as rankings_stream,
        rejected_path.open("w", encoding="utf-8") as rejected_stream,
    ):
        for line_number, record in read_jsonl(input_file):
            stats["total"] += 1
            result = dict(record)
            try:
                raw = record.get(generation_key)
                if not isinstance(raw, str):
                    raise ValueError(f"missing string field {generation_key}")
                payload = parse_json_object(raw)
                candidates = record.get("candidate_answers")
                preferred_index = record.get("preferred_index")
                if not isinstance(candidates, list) or not candidates:
                    raise ValueError("candidate_answers must contain at least one answer")
                if not isinstance(preferred_index, int) or isinstance(preferred_index, bool):
                    raise ValueError("preferred_index must be an integer")

                ranking = payload.get("ranking")
                if not isinstance(ranking, list) or any(
                    not isinstance(index, int) or isinstance(index, bool) for index in ranking
                ):
                    raise ValueError("ranking must be a list of integer indices")
                if sorted(ranking) != list(range(len(candidates))):
                    raise ValueError("ranking must be a complete candidate-index permutation")
                rejected_index = payload.get("rejected_index")
                requires_synthetic = record.get("synthetic_negative_required") is True
                synthetic_rejected = payload.get("synthetic_rejected_answer")
                if requires_synthetic:
                    if len(candidates) != 1 or preferred_index != 0:
                        raise ValueError(
                            "synthetic-negative input must contain one preferred answer"
                        )
                    if rejected_index != -1:
                        raise ValueError("synthetic rejected_index must be -1")
                    if not isinstance(synthetic_rejected, str) or not synthetic_rejected.strip():
                        raise ValueError("synthetic_rejected_answer must be a non-empty string")
                    synthetic_rejected = synthetic_rejected.strip()
                    if synthetic_rejected == candidates[preferred_index]:
                        raise ValueError(
                            "synthetic rejected answer must differ from preferred answer"
                        )
                elif rejected_index != ranking[-1]:
                    raise ValueError("rejected_index must equal the final ranking index")

                material_contrast = payload.get("material_contrast") is True

                # Candidates that turn out to be equally correct used to be
                # dropped, and the only way to recover them was to resubmit
                # them as one-answer records by hand -- 1,072 of them in the
                # V2 campaign. If the model wrote a usable negative anyway,
                # take it rather than losing the question.
                synthetic_fallback = False
                if not requires_synthetic and not material_contrast:
                    candidate = (
                        synthetic_rejected.strip() if isinstance(synthetic_rejected, str) else ""
                    )
                    if candidate and candidate != candidates[preferred_index]:
                        synthetic_rejected = candidate
                        synthetic_fallback = True

                result.update(
                    {
                        "ranking": ranking,
                        "preferred_rank": ranking.index(preferred_index),
                        "rejected_index": rejected_index,
                        "material_contrast": material_contrast,
                        "synthetic_fallback": synthetic_fallback,
                        "ranking_reason": str(payload.get("reason") or "").strip(),
                    }
                )
                if not material_contrast and not synthetic_fallback:
                    result["ranking_status"] = "no_material_contrast"
                    stats["no_material_contrast"] += 1
                    _write(rejected_stream, result)
                elif (
                    not requires_synthetic
                    and not synthetic_fallback
                    and preferred_index == rejected_index
                ):
                    result["ranking_status"] = "preferred_ranked_last"
                    stats["preferred_ranked_last"] += 1
                    _write(rejected_stream, result)
                else:
                    result["ranking_status"] = "accepted"
                    prepared = {
                        key: record.get(key)
                        for key in (
                            "id",
                            "question",
                            "context",
                            "reference_answer",
                            "metadata",
                            "input_format",
                            "source_sha256",
                            "context_sha256",
                        )
                    }
                    prepared.update(
                        {
                            "preferred_answer": candidates[preferred_index],
                            "rejected_answer": (
                                synthetic_rejected
                                if requires_synthetic or synthetic_fallback
                                else candidates[rejected_index]
                            ),
                            "ranking": ranking,
                            "ranking_reason": result["ranking_reason"],
                            "negative_source": (
                                "synthetic"
                                if requires_synthetic
                                else "synthetic_fallback"
                                if synthetic_fallback
                                else "candidate"
                            ),
                        }
                    )
                    _write(output_stream, prepared)
                    stats["accepted"] += 1
                    if requires_synthetic:
                        stats["synthetic_negative"] += 1
                    elif synthetic_fallback:
                        stats["synthetic_fallback"] += 1
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                result.update(
                    {
                        "ranking_status": "invalid",
                        "ranking_error": str(exc),
                        "line_number": line_number,
                    }
                )
                stats["invalid"] += 1
                _write(rejected_stream, result)
            _write(rankings_stream, result)

    Path(audit_file).write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if stats["accepted"] == 0:
        raise ValueError("ranking produced no materially contrastive rubric inputs")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", "--input_file", required=True)
    parser.add_argument("--output-file", "--output_file", required=True)
    parser.add_argument("--rankings-file", "--rankings_file", required=True)
    parser.add_argument("--audit-file", "--audit_file", required=True)
    parser.add_argument("--generation-key", "--generation_key", default="ranking_generation")
    args = parser.parse_args()
    stats = postprocess_rankings(
        input_file=args.input_file,
        output_file=args.output_file,
        rankings_file=args.rankings_file,
        audit_file=args.audit_file,
        generation_key=args.generation_key,
    )
    print(json.dumps(stats, sort_keys=True))


if __name__ == "__main__":
    main()
