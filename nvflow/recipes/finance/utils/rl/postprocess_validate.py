#!/usr/bin/env python3
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
"""In-process orchestrator for the validate_questions Phase 2 postprocess.

Equivalent to running ``parse_validate_responses`` and then
``apply_validate_filter`` back-to-back, but in a single Python interpreter
so the stage's Slurm step needs only one ``postprocess_cmd``.  Sentinel
log lines (``PHASE: parse``, ``PHASE: apply``, ``PHASE: done``) preserve
grep-able phase boundaries in Slurm logs so post-mortem analysis stays
straightforward even though the two phases share a process.

CLI argument naming is cleaner than the underlying scripts' overloaded
``--input_file`` / ``--output_file`` -- explicit names map to each
artefact.  The standalone CLIs of ``parse_validate_responses`` and
``apply_validate_filter`` remain available for one-off debugging.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nvflow.recipes.finance.utils.rl.apply_validate_filter import apply_validate_filter
from nvflow.recipes.finance.utils.rl.parse_validate_responses import parse_validate_responses
from nvflow.utils import setup_logger

logger = setup_logger(__name__)


_SENTINEL_PARSE = "=== validate_questions postprocess: PHASE: parse ==="
_SENTINEL_APPLY = "=== validate_questions postprocess: PHASE: apply ==="
_SENTINEL_DONE = "=== validate_questions postprocess: PHASE: done ==="


def postprocess_validate(
    *,
    llm_output: str,
    parsed_jsonl: str,
    final_kept: str,
    dropped: str,
    stats: str,
    raw_sdg_source: str,
    raw_sdg_filename: str = "final_result.jsonl",
    keep_tag: str = "VALID",
    high_drop_threshold: float = 0.20,
) -> None:
    """Run parse + apply phases sequentially in this Python process.

    Args mirror the underlying scripts; see their docstrings for details.
    Any unhandled exception inside parse propagates and the apply phase
    is NOT entered -- equivalent to the historical ``parse_cmd && apply_cmd``
    short-circuit semantics.

    The intermediate ``parsed_jsonl`` file is still written to disk for
    audit (matches historical behaviour and lets operators re-run the
    apply phase standalone with different parameters if needed).
    """
    logger.info(_SENTINEL_PARSE)
    parse_validate_responses(llm_output, parsed_jsonl)

    logger.info(_SENTINEL_APPLY)
    raw_sdg_path = str(Path(raw_sdg_source) / raw_sdg_filename)
    apply_validate_filter(
        input_file=parsed_jsonl,
        output_kept=final_kept,
        output_dropped=dropped,
        stats_file=stats,
        keep_tag=keep_tag,
        high_drop_threshold=high_drop_threshold,
        raw_sdg_path=raw_sdg_path,
    )

    logger.info(_SENTINEL_DONE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "validate_questions Phase 2 postprocess orchestrator: "
            "parse LLM responses, then apply VALID/INVALID filter, in one process."
        )
    )
    parser.add_argument(
        "--llm_output",
        required=True,
        help="Input JSONL with nemo-skills 'generation' field (raw LLM output).",
    )
    parser.add_argument(
        "--parsed_jsonl",
        required=True,
        help="Intermediate JSONL with validate_tag attached (audit artefact).",
    )
    parser.add_argument(
        "--final_kept",
        required=True,
        help="Output JSONL of VALID records (consumed by data_transformation).",
    )
    parser.add_argument(
        "--dropped",
        required=True,
        help="Output JSONL of dropped records, annotated with _llm_drop_reason.",
    )
    parser.add_argument(
        "--stats",
        required=True,
        help="Output JSON with filter counts and high_drop_warning flag.",
    )
    parser.add_argument(
        "--raw_sdg_source",
        required=True,
        help=(
            "Directory holding the raw SDG JSONL.  Required: apply_validate_filter "
            "uses it to restore reasoning_content per VALID record so the output "
            "matches the original SDG schema (data_transformation enforces this)."
        ),
    )
    parser.add_argument(
        "--raw_sdg_filename",
        default="final_result.jsonl",
        help="Filename inside --raw_sdg_source (default: final_result.jsonl).",
    )
    parser.add_argument(
        "--keep_tag",
        default="VALID",
        help="Tag value to keep (default VALID).",
    )
    parser.add_argument(
        "--high_drop_threshold",
        type=float,
        default=0.20,
        help="Drop-rate above which the stats file records high_drop_warning=true (default 0.20).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    postprocess_validate(
        llm_output=args.llm_output,
        parsed_jsonl=args.parsed_jsonl,
        final_kept=args.final_kept,
        dropped=args.dropped,
        stats=args.stats,
        raw_sdg_source=args.raw_sdg_source,
        raw_sdg_filename=args.raw_sdg_filename,
        keep_tag=args.keep_tag,
        high_drop_threshold=args.high_drop_threshold,
    )


if __name__ == "__main__":
    main()
