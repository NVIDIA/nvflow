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
"""Post-process document-grounded SDG data: clean fields, rename, emit a single
``final_result.jsonl`` consumed by downstream SFT / RL workflows.

The previous design produced three files (``full_data.jsonl`` for the full set,
``final_result.jsonl`` for an SFT subset, ``hard_rl_data.jsonl`` for an RL
subset). Both subset filters were gated by ``difficulty_score``, which is no
longer computed (the difficulty_estimation stage has been removed). Subset
selection is now the downstream consumer's responsibility; this stage just
emits the cleaned, renamed full set under the canonical name
``final_result.jsonl`` so it slots into ``grpo/base.yaml`` and ``sft/base.yaml``
unchanged.
"""

import argparse
import json
import os
import re
from collections.abc import Set

from nvflow.utils import setup_logger

logger = setup_logger(__name__)

FIELDS_TO_REMOVE = {
    "solutions",
    "generations_list",
    "reasonings_list",
    "num_solutions",
    "max_idx",
    "num_generated_tokens",
    "finish_reason",
    "generation_start_time",
    "generation_end_time",
    "generation_time",
    "genselect_response",
    "selected_index",
    "reasoning_content",
    "serialized_output",
}

_DYNAMIC_FIELD_PATTERNS = re.compile(r"^answer_\d+$|^answer_reasoning_content_\d+$")

FIELDS_TO_RENAME = {
    "reference_reasoning": "reasoning_content",
    "reference_answer": "answer",
    # Restore the Responses-API original form of the genselect-picked answer to its
    # canonical top-level names so ``final_result.jsonl`` is rollout-like.
    "reference_response": "response",
    "reference_responses_create_params": "responses_create_params",
}

OUTPUT_FILENAME = "final_result.jsonl"


def clean_record(
    record: dict,
    *,
    extra_fields_to_remove: Set[str] | None = None,
    extra_fields_to_rename: dict[str, str] | None = None,
) -> dict:
    """Remove unwanted fields and rename fields.

    Args:
        record: Input record dict
        extra_fields_to_remove: Additional field names to remove on top of
            ``FIELDS_TO_REMOVE`` and dynamically matched ``answer_N`` /
            ``answer_reasoning_content_N`` fields.
        extra_fields_to_rename: Additional ``{old: new}`` renames on top of
            ``FIELDS_TO_RENAME``.
    """
    remove_set = FIELDS_TO_REMOVE
    if extra_fields_to_remove:
        remove_set = remove_set | set(extra_fields_to_remove)

    cleaned = {
        k: v
        for k, v in record.items()
        if k not in remove_set and not _DYNAMIC_FIELD_PATTERNS.match(k)
    }

    rename_map = FIELDS_TO_RENAME
    if extra_fields_to_rename:
        rename_map = {**rename_map, **extra_fields_to_rename}

    for old_name, new_name in rename_map.items():
        if old_name in cleaned:
            cleaned[new_name] = cleaned.pop(old_name)

    return cleaned


def dgsdg_post_process(
    input_file: str,
    output_dir: str,
    *,
    extra_fields_to_remove: Set[str] | None = None,
    extra_fields_to_rename: dict[str, str] | None = None,
    seed: int = 42,
):
    """Clean every record from ``input_file`` and write to
    ``{output_dir}/final_result.jsonl``.

    Args:
        input_file: Path to input JSONL file (``aggregated_answers.jsonl``).
        output_dir: Directory to write ``final_result.jsonl``.
        extra_fields_to_remove: Additional field names to remove during cleaning.
        extra_fields_to_rename: Additional ``{old: new}`` renames during cleaning.
        seed: Random seed for reproducibility (currently unused; reserved for
            future shuffling / subsampling extensions).
    """
    logger.info(f"Post processing document grounded sdg data from: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Random seed: {seed}")

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, OUTPUT_FILENAME)

    total_records = 0
    with (
        open(input_file, encoding="utf-8") as f_in,
        open(output_file, "w", encoding="utf-8") as f_out,
    ):
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: JSON decode error: {e}")
                continue

            cleaned = clean_record(
                record,
                extra_fields_to_remove=extra_fields_to_remove,
                extra_fields_to_rename=extra_fields_to_rename,
            )
            # ``expected_answer`` mirrors the final answer text so the record is a
            # drop-in for consumers that key on the Responses-API/RL convention
            # (sibling of ``responses_create_params`` / ``response``), without
            # touching the flat ``answer`` field.
            if "answer" in cleaned:
                cleaned.setdefault("expected_answer", cleaned["answer"])
            f_out.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            total_records += 1

            if line_num % 100000 == 0:
                logger.info(f"  Processed {line_num} lines...")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Document grounded sdg data post processing complete!")
    logger.info(f"  Total records: {total_records}")
    logger.info(f"  Output: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean DG-SDG records and emit a single final_result.jsonl."
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to input JSONL file (typically aggregated_answers.jsonl).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write final_result.jsonl",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        exit(1)

    dgsdg_post_process(args.input_file, args.output_dir, seed=args.seed)
