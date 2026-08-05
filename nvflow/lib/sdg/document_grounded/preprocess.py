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
"""QA pipeline preprocessing for Document-Grounded SDG.

Generic functions for the question generation -> verification -> answer
generation -> answer verification pipeline.  Domain-specific context
construction is injected via the ``context_builder`` callback parameter
in ``construct_question_generate_input``.
"""

import argparse
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nvflow.utils import setup_logger

logger = setup_logger(__name__)


_KEYS_TO_REMOVE = [
    "generation",
    "serialized_output",
    "num_generated_tokens",
    "finish_reason",
    "generation_start_time",
    "generation_end_time",
    "generation_time",
    "reasoning_content",
]


def remove_keys(output_dict: dict[str, Any], keys_to_remove: list[str]) -> dict[str, Any]:
    """Remove keys from a dictionary."""
    for key in keys_to_remove:
        if key in output_dict:
            output_dict.pop(key)
    return output_dict


# ---------------------------------------------------------------------------
# Stage 1: Construct question generation input (requires callback)
# ---------------------------------------------------------------------------


def construct_question_generate_input(
    input_folder: Path,
    output_file: Path,
    *,
    context_builder: Callable[[dict], str],
) -> None:
    """Read JSONL records, call context_builder(record) on each, write enriched JSONL.

    This is the only preprocessing function that requires a domain-specific
    callback.  All other functions in this module are fully generic.

    Args:
        input_folder: Directory containing input JSONL files
        output_file: Path to write output JSONL
        context_builder: ``(record: dict) -> str`` that produces a context
            string from a JSONL record.  For SEC filings this formats
            company/year/section headers; other domains provide their own.
    """
    logger.info(
        "Preprocessing data for question generation from %s to %s", input_folder, output_file
    )
    input_files = list(input_folder.glob("*.jsonl"))
    if not input_files:
        raise FileNotFoundError(f"No input files found in {input_folder}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    # Write to a temp sibling, then atomically replace -- generate_verified_questions's
    # step1 skip-if-exists check treats this exact final path as "already prepared"
    # once it exists and is non-empty. A direct write would let a killed/timed-out
    # job leave a truncated-but-non-empty file that the skip check mistakes for
    # complete, silently feeding a short input into the rest of the Q-pipeline.
    tmp_output_file = output_file.with_suffix(output_file.suffix + ".tmp")
    with tmp_output_file.open("w", encoding="utf-8") as fout:
        for input_file in input_files:
            with input_file.open("r", encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    try:
                        raw_data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    context = context_builder(raw_data)
                    raw_data["context"] = context
                    fout.write(json.dumps(raw_data, ensure_ascii=False) + "\n")
                    count += 1
    os.replace(tmp_output_file, output_file)
    logger.info("Done writing %d records to %s", count, output_file)


# ---------------------------------------------------------------------------
# Stage 2: Construct question verification input (pluggable parser)
# ---------------------------------------------------------------------------


def _default_generation_parser(record: dict) -> list[tuple[str, str]]:
    """Default parser: extract JSON questions from LLM generation output.

    Handles ``generation`` and ``serialized_output`` fields, strips
    ``<|message|>`` tokens, and parses the JSON block containing
    question categories mapped to question lists.

    Returns:
        list of ``(question_type, question_text)`` pairs
    """
    raw_content = ""
    if "generation" in record:
        raw_content = record["generation"]
    elif "serialized_output" in record:
        ser_out = record["serialized_output"]
        if isinstance(ser_out, list) and len(ser_out) > 0:
            raw_content = ser_out[0].get("content", "")

    generation_str = "{}"
    if raw_content:
        content = raw_content
        if "<|message|>" in content:
            parts = content.split("<|message|>")
            content = parts[-1]

        s_idx = content.find("{")
        e_idx = content.rfind("}")
        if s_idx != -1 and e_idx != -1:
            generation_str = content[s_idx : e_idx + 1]
        else:
            generation_str = content

    try:
        questions_data = json.loads(generation_str)
    except (json.JSONDecodeError, TypeError):
        return []

    questions_items: list[tuple[str, str]] = []
    if isinstance(questions_data, dict):
        for k, v in questions_data.items():
            if isinstance(v, list):
                for q in v:
                    questions_items.append((k, q))
            else:
                questions_items.append((k, v))
    elif isinstance(questions_data, list):
        questions_items = [("General", q) for q in questions_data]

    return questions_items


def construct_question_verify_input(
    input_dir: Path,
    output_file: Path,
    *,
    generation_parser: Callable[[dict], list[tuple[str, str]]] | None = None,
):
    """Parse LLM question-generation output and expand to individual questions.

    Args:
        input_dir: Directory containing output-rs*.jsonl or output.jsonl
        output_file: Path to write expanded questions JSONL
        generation_parser: ``(record) -> [(question_type, question_text), ...]``
            that extracts questions from an LLM generation record.
            Defaults to ``_default_generation_parser`` which parses JSON
            question dicts.  Override for different prompt formats.
    """
    if generation_parser is None:
        generation_parser = _default_generation_parser

    logger.info(
        "Preprocessing data for question verification from %s to %s", input_dir, output_file
    )
    input_files = list(input_dir.glob("output-rs*.jsonl"))
    if not input_files:
        if (input_dir / "output.jsonl").exists():
            input_files = [input_dir / "output.jsonl"]
        else:
            raise FileNotFoundError(f"No output files found in {input_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    # Write to a temp sibling, then atomically replace -- generate_verified_questions's
    # step3 skip-if-exists check treats this exact final path as "already prepared"
    # once it exists and is non-empty. A direct write would let a killed/timed-out
    # job leave a truncated-but-non-empty file that the skip check mistakes for
    # complete, silently feeding fewer questions into verification forever.
    tmp_output_file = output_file.with_suffix(output_file.suffix + ".tmp")
    with tmp_output_file.open("w", encoding="utf-8") as fout:
        for input_file in input_files:
            with input_file.open("r", encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    try:
                        result = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    questions_items = generation_parser(result)

                    for q_type, q_text in questions_items:
                        new_record = result.copy()
                        new_record = remove_keys(new_record, _KEYS_TO_REMOVE)
                        new_record["problem"] = q_text
                        new_record["question_type"] = q_type
                        fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                        count += 1
    os.replace(tmp_output_file, output_file)
    logger.info("Total questions prepared: %d", count)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_verification(result: dict[str, Any]) -> bool:
    """Check if a verification result indicates 'Yes' (verified)."""
    generation = result.get("generation") or ""
    if not generation:
        ser_out = result.get("serialized_output", [])
        if isinstance(ser_out, list) and len(ser_out) > 0:
            generation = ser_out[0].get("content") or ""

    if "<|channel|>final<|message|>" in generation:
        final_ans = generation.split("<|channel|>final<|message|>")[-1].strip()
        if "Yes" in final_ans:
            return True
    elif "Yes" in generation and len(generation) < 10:
        return True
    elif "Yes" in generation:
        if generation.rfind("No") > generation.rfind("Yes"):
            return False
        else:
            return True
    return False


def _make_question_key(result: dict[str, Any]) -> str:
    """Create a hash key from context and problem to identify unique questions."""
    problem = result.get("problem", "")
    context = result.get("context", "")
    key_str = f"{context}||{problem}"
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Stage 3: Construct answer generation input (fully generic)
# ---------------------------------------------------------------------------


def construct_answer_generate_input(input_dir: Path, output_file: Path, threshold: float = 0.5):
    """
    Streaming two-pass implementation to avoid OOM on large datasets.

    Pass 1: Count votes per question (only store counts and first verified location)
    Pass 2: Read and write only records that pass threshold
    """
    logger.info(
        "Preprocessing data for answer generation from %s to %s with threshold %s",
        input_dir,
        output_file,
        threshold,
    )
    input_files = sorted(input_dir.glob("output-rs*.jsonl"))
    if not input_files:
        if (input_dir / "output.jsonl").exists():
            input_files = [input_dir / "output.jsonl"]
        else:
            raise FileNotFoundError(f"No output files found in {input_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: Count votes per question key
    logger.info("Pass 1: Counting votes...")
    vote_stats: dict[str, list] = {}

    for file_idx, input_file in enumerate(input_files):
        logger.info("  Scanning %s...", input_file.name)
        with input_file.open("r", encoding="utf-8") as fin:
            for line_num, line in enumerate(fin):
                if not line.strip():
                    continue
                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    continue

                key = _make_question_key(result)
                is_verified = _check_verification(result)

                if key not in vote_stats:
                    vote_stats[key] = [0, 0, None, (file_idx, line_num)]

                vote_stats[key][0] += 1
                if is_verified:
                    vote_stats[key][1] += 1
                    if vote_stats[key][2] is None:
                        vote_stats[key][2] = (file_idx, line_num)

    logger.info("Filtering by threshold...")
    keys_to_write: dict[str, tuple] = {}
    for key, stats in vote_stats.items():
        total, positive, verified_loc, any_loc = stats
        if total > 0 and (positive / total) >= threshold:
            loc = verified_loc if verified_loc is not None else any_loc
            keys_to_write[key] = (loc, total, positive)

    logger.info(
        "  %d questions passed threshold (out of %d total)", len(keys_to_write), len(vote_stats)
    )

    del vote_stats

    # Pass 2: Read only the records we need and write them
    logger.info("Pass 2: Writing verified questions...")

    records_by_file: dict[int, dict[int, tuple]] = {}
    for key, (loc, total, positive) in keys_to_write.items():
        file_idx, line_num = loc
        if file_idx not in records_by_file:
            records_by_file[file_idx] = {}
        records_by_file[file_idx][line_num] = (key, total, positive)

    count = 0
    # Write to a temp sibling, then atomically replace -- generate_answers's
    # a-prep skip-if-exists check treats this exact final path as "already
    # prepared" once it exists and is non-empty. A direct write would let a
    # killed/timed-out job leave a truncated-but-non-empty file that the skip
    # check mistakes for complete, silently feeding fewer answers into A-gen.
    tmp_output_file = output_file.with_suffix(output_file.suffix + ".tmp")
    with tmp_output_file.open("w", encoding="utf-8") as fout:
        for file_idx, line_nums in sorted(records_by_file.items()):
            input_file = input_files[file_idx]
            logger.info("  Reading %d records from %s...", len(line_nums), input_file.name)

            with input_file.open("r", encoding="utf-8") as fin:
                for line_num, line in enumerate(fin):
                    if line_num not in line_nums:
                        continue

                    _key, total, positive = line_nums[line_num]

                    try:
                        result = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    new_record = result.copy()
                    new_record = remove_keys(new_record, _KEYS_TO_REMOVE)

                    new_record["question_voting_pass_rate"] = positive / total
                    new_record["question_voting_total"] = total

                    fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                    count += 1

    os.replace(tmp_output_file, output_file)
    logger.info("Total verified questions prepared: %d", count)


# ---------------------------------------------------------------------------
# Stage 4: Construct answer verification input (fully generic)
# ---------------------------------------------------------------------------


def construct_answer_verify_input(input_dir: Path, output_file: Path):
    """Prepare answer verification input from generated answers."""
    logger.info("Preprocessing data for answer verification from %s to %s", input_dir, output_file)
    input_files = list(input_dir.glob("output-rs*.jsonl"))
    if not input_files:
        if (input_dir / "output.jsonl").exists():
            input_files = [input_dir / "output.jsonl"]
        else:
            raise FileNotFoundError(f"No output files found in {input_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_file.open("w", encoding="utf-8") as fout:
        for input_file in input_files:
            source_name = input_file.name
            with input_file.open("r", encoding="utf-8") as fin:
                file_index = 0
                for line in fin:
                    if not line.strip():
                        continue
                    try:
                        result = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    generation = result.get("generation") or ""
                    if not generation:
                        ser_out = result.get("serialized_output", [])
                        if isinstance(ser_out, list) and len(ser_out) > 0:
                            generation = ser_out[0].get("content") or ""

                    if not generation:
                        file_index += 1
                        continue

                    new_record = result.copy()
                    new_record["answer_reasoning"] = new_record.get("reasoning_content", "")
                    new_record = remove_keys(new_record, _KEYS_TO_REMOVE)
                    new_record["answer"] = generation
                    new_record["question_index"] = file_index
                    new_record["source_file"] = source_name

                    fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                    count += 1
                    file_index += 1
    logger.info("Total answers prepared for verification: %d", count)


# ---------------------------------------------------------------------------
# Stage 5: Filter verified answers (fully generic)
# ---------------------------------------------------------------------------


def filter_verified_answers(input_dir: Path, output_file: Path, threshold: float = 0.5):
    """Filter verified answers based on 'Yes' responses and majority voting.

    Streaming two-pass implementation to avoid OOM on large datasets (mirrors
    ``construct_answer_generate_input``):

    Pass 1: Count votes per question (only store counts + the location of the
            chosen base record), never the record contents.
    Pass 2: Read back and write only the records that pass the threshold.

    Base-record selection matches the prior in-memory version exactly: the first
    *verified* record for the key, or the first record seen if none verified.
    """
    logger.info(
        "Filtering verified answers from %s to %s with threshold %s",
        input_dir,
        output_file,
        threshold,
    )
    input_files = list(input_dir.glob("output-rs*.jsonl"))
    if not input_files:
        if (input_dir / "output.jsonl").exists():
            input_files = [input_dir / "output.jsonl"]
        else:
            raise FileNotFoundError(f"No output files found in {input_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: tally votes per question key, remembering only byte-light locations.
    # vote_stats[key] = [total, positive, verified_loc, first_loc]
    logger.info("Pass 1: Counting votes...")
    vote_stats: dict[tuple[str, int], list] = {}

    for file_idx, input_file in enumerate(input_files):
        logger.info("  Scanning %s...", input_file.name)
        with input_file.open("r", encoding="utf-8") as fin:
            for line_num, line in enumerate(fin):
                if not line.strip():
                    continue
                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    continue

                key = (result.get("source_file", "unknown"), result.get("question_index", -1))
                is_verified = _check_verification(result)

                if key not in vote_stats:
                    vote_stats[key] = [0, 0, None, (file_idx, line_num)]

                vote_stats[key][0] += 1
                if is_verified:
                    vote_stats[key][1] += 1
                    if vote_stats[key][2] is None:
                        vote_stats[key][2] = (file_idx, line_num)

    logger.info("Filtering by threshold...")
    keys_to_write: dict[tuple[str, int], tuple] = {}
    for key, (total, positive, verified_loc, first_loc) in vote_stats.items():
        if total > 0 and (positive / total) >= threshold:
            loc = verified_loc if verified_loc is not None else first_loc
            keys_to_write[key] = (loc, total, positive)

    logger.info(
        "  %d questions passed threshold (out of %d total)", len(keys_to_write), len(vote_stats)
    )

    del vote_stats

    # Pass 2: read back only the chosen records and write them.
    records_by_file: dict[int, dict[int, tuple]] = {}
    for _key, (loc, total, positive) in keys_to_write.items():
        file_idx, line_num = loc
        records_by_file.setdefault(file_idx, {})[line_num] = (total, positive)

    logger.info("Pass 2: Writing verified answers...")
    count = 0
    with output_file.open("w", encoding="utf-8") as fout:
        for file_idx, line_nums in sorted(records_by_file.items()):
            input_file = input_files[file_idx]
            logger.info("  Reading %d records from %s...", len(line_nums), input_file.name)
            with input_file.open("r", encoding="utf-8") as fin:
                for line_num, line in enumerate(fin):
                    if line_num not in line_nums:
                        continue

                    total_votes, positive_votes = line_nums[line_num]
                    try:
                        base_record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    new_record = base_record.copy()
                    new_record = remove_keys(new_record, _KEYS_TO_REMOVE)
                    new_record["voting_pass_rate"] = positive_votes / total_votes
                    new_record["voting_total"] = total_votes

                    fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                    count += 1

    logger.info("Total answers passed majority voting (%s): %d", threshold, count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QA pipeline preprocessing for Document-Grounded SDG."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p2 = subparsers.add_parser(
        "construct_question_verify_input",
        help="Parse question generation output into individual questions",
    )
    p2.add_argument("--input_dir", type=Path, required=True)
    p2.add_argument("--output_file", type=Path, required=True)

    p3 = subparsers.add_parser(
        "construct_answer_generate_input",
        help="Filter verified questions for answer generation",
    )
    p3.add_argument("--input_dir", type=Path, required=True)
    p3.add_argument("--output_file", type=Path, required=True)
    p3.add_argument("--threshold", type=float, default=0.5)

    p4 = subparsers.add_parser(
        "construct_answer_verify_input",
        help="Prepare answer verification input",
    )
    p4.add_argument("--input_dir", type=Path, required=True)
    p4.add_argument("--output_file", type=Path, required=True)

    p5 = subparsers.add_parser(
        "filter_verified_answers",
        help="Filter verified answers by majority voting",
    )
    p5.add_argument("--input_dir", type=Path, required=True)
    p5.add_argument("--output_file", type=Path, required=True)
    p5.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()

    if args.command == "construct_question_verify_input":
        construct_question_verify_input(args.input_dir, args.output_file)
    elif args.command == "construct_answer_generate_input":
        construct_answer_generate_input(args.input_dir, args.output_file, args.threshold)
    elif args.command == "construct_answer_verify_input":
        construct_answer_verify_input(args.input_dir, args.output_file)
    elif args.command == "filter_verified_answers":
        filter_verified_answers(args.input_dir, args.output_file, args.threshold)
