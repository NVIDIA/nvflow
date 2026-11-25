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
"""SEC-specific callbacks for the DG-SDG pipeline.

These callbacks inject SEC filing domain knowledge into the generic
``nvflow.stages.sdg.document_grounded`` platform functions.
"""

import re
from typing import Any

_CONTENT_KEY = re.compile(r"^content(\d+)$")


def _chunk_indexes(result: dict[str, Any]) -> list[int]:
    """Return the numeric suffixes for all chunks in a relational record."""
    return sorted(
        int(match.group(1)) for key in result if (match := _CONTENT_KEY.fullmatch(key)) is not None
    )


def _file_type_from_path(file_path: str, fallback: str) -> str:
    """Infer the EDGAR form from ``<company>/<form>/<year>/...`` paths."""
    parts = file_path.replace("\\", "/").split("/")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    return fallback


def construct_context(result: dict[str, Any], file_type: str = "10-K") -> str:
    """Build one labelled SEC context block for every sampled chunk.

    Relational SDG can emit one to five chunks and can vary the company, year,
    form, and section per chunk.  Preserve every chunk and prefer its indexed
    metadata; fall back to record-level metadata for single-document records
    and relation modes that share an attribute across all chunks.
    """
    blocks: list[str] = []
    for index in _chunk_indexes(result):
        content = result.get(f"content{index}", "")
        if not content:
            continue

        company = result.get(f"company_name{index}", result.get("company_name", ""))
        year = result.get(f"year{index}", result.get("year", ""))
        chunk_file_type = result.get(f"file_type{index}")
        if not chunk_file_type:
            chunk_file_type = _file_type_from_path(
                str(result.get(f"file_path{index}", "")), file_type
            )
        section = result.get(f"item_section{index}", result.get("item_section0", "unknown section"))

        blocks.append(
            f"**{year} {company} {chunk_file_type}**\n\n**Part of {section}**\n\n{content}"
        )

    return "\n\n".join(blocks) + ("\n\n" if blocks else "")


def sec_context_builder(record: dict[str, Any]) -> str:
    """Context builder callback for ``construct_question_generate_input``.

    Delegates to :func:`construct_context`, using record-level form metadata as
    a fallback when a chunk has neither ``file_typeN`` nor ``file_pathN``.
    """
    file_type = record.get("file_type") or _file_type_from_path(
        str(record.get("file_path0", "")), "SEC filing"
    )
    return construct_context(record, file_type)
