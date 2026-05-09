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
import argparse
import json
from pathlib import Path
from typing import Any

from nvflow.utils import setup_logger

logger = setup_logger(__name__)


# Utils copied/adapted to be self-contained in this script for cluster execution
def construct_context(result: dict[str, Any], file_type: str = "10-k") -> str:
    """Construct the context string by reading HTML files based on record keys."""
    try:
        # Case 4: 2 companies, 3 sections
        if "item_section2" in result:
            return (
                f"**{result.get('year', '')} {result.get('company_name0', '')} {file_type}**\n\n"
                f"**Part of {result['item_section0']}**\n\n"
                f"{result.get('content0', '')}\n\n"
                f"**{result.get('year', '')} {result.get('company_name1', '')} {file_type}**\n\n"
                f"**Part of {result['item_section1']}**\n\n"
                f"{result.get('content1', '')}\n\n"
                f"**Part of {result['item_section2']}**\n\n"
                f"{result.get('content2', '')}\n\n"
            )
        # Case 3: 2 companies, 2 sections
        elif "company_name0" in result:
            return (
                f"**{result.get('year', '')} {result['company_name0']} {file_type}**\n\n"
                f"**Part of {result['item_section0']}**\n\n"
                f"{result.get('content0', '')}\n\n"
                f"**{result.get('year', '')} {result['company_name1']} {file_type}**\n\n"
                f"**Part of {result['item_section1']}**\n\n"
                f"{result.get('content1', '')}\n\n"
            )
        # Case 2: 1 company, 2 sections
        elif "item_section1" in result:
            return (
                f"**{result.get('year', '')} {result.get('company_name', '')} {file_type}**\n\n"
                f"**Part of {result['item_section0']}**\n\n"
                f"{result.get('content0', '')}\n\n"
                f"**Part of {result['item_section1']}**\n\n"
                f"{result.get('content1', '')}\n\n"
            )
        # Case 1: 1 company, 1 section
        elif "item_section0" in result:
            return (
                f"**{result.get('year', '')} {result.get('company_name', '')} {file_type}**\n\n"
                f"**Part of {result['item_section0']}**\n\n"
                f"{result.get('content0', '')}\n\n"
            )
    except Exception:
        return ""
    return ""


def remove_keys(output_dict: dict[str, Any], keys_to_remove: list[str]) -> dict[str, Any]:
    """Remove keys from a dictionary."""
    for key in keys_to_remove:
        if key in output_dict:
            output_dict.pop(key)
    return output_dict


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


def construct_question_generate_input(input_folder: Path, output_file: Path):
    logger.info(
        "Preprocessing data for question generation from %s to %s", input_folder, output_file
    )
    input_files = list(input_folder.glob("*.jsonl"))
    if not input_files:
        raise FileNotFoundError(f"No input files found in {input_folder}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as fout:
        for input_file in input_files:
            with input_file.open("r", encoding="utf-8") as fin:
                file_type = "10-K" if "10-k" in input_file.name else "10-Q"
                for line in fin:
                    if not line.strip():
                        continue
                    try:
                        raw_data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    context = construct_context(raw_data, file_type)
                    raw_data["context"] = context
                    fout.write(json.dumps(raw_data, ensure_ascii=False) + "\n")
    logger.info("Done writing to %s", output_file)


def construct_question_verify_input(input_dir: Path, output_file: Path):
    logger.info(
        "Preprocessing data for question verification from %s to %s", input_dir, output_file
    )
    input_files = list(input_dir.glob("output-rs*.jsonl"))
    if not input_files:
        # Fallback
        if (input_dir / "output.jsonl").exists():
            input_files = [input_dir / "output.jsonl"]
        else:
            raise FileNotFoundError(f"No output files found in {input_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_file.open("w", encoding="utf-8") as fout:
        for input_file in input_files:
            with input_file.open("r", encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    try:
                        result = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Parse generation
                    raw_content = ""
                    if "generation" in result:
                        raw_content = result["generation"]
                    elif "serialized_output" in result:
                        ser_out = result["serialized_output"]
                        if isinstance(ser_out, list) and len(ser_out) > 0:
                            raw_content = ser_out[0].get("content", "")

                    # Attempt to extract JSON
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
                        continue

                    questions_items = []
                    if isinstance(questions_data, dict):
                        for k, v in questions_data.items():
                            if isinstance(v, list):
                                for q in v:
                                    questions_items.append((k, q))
                            else:
                                questions_items.append((k, v))
                    elif isinstance(questions_data, list):
                        questions_items = [("General", q) for q in questions_data]
                    else:
                        continue

                    for q_type, q_text in questions_items:
                        new_record = result.copy()
                        new_record = remove_keys(new_record, _KEYS_TO_REMOVE)
                        new_record["problem"] = q_text
                        new_record["question_type"] = q_type
                        fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                        count += 1
    logger.info("Total questions prepared: %d", count)


def _check_verification(result: dict[str, Any]) -> bool:
    """Check if a verification result indicates 'Yes' (verified)."""
    generation = result.get("generation") or ""
    if not generation:
        ser_out = result.get("serialized_output", [])
        if isinstance(ser_out, list) and len(ser_out) > 0:
            generation = ser_out[0].get("content") or ""

    if not generation:
        return False

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
    import hashlib

    problem = result.get("problem", "")
    context = result.get("context", "")
    # Use hash to avoid storing large context strings in memory
    key_str = f"{context}||{problem}"
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()


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

    # Pass 1: Count votes per question key (memory-efficient: only store counts and locations)
    # vote_stats[key] = (total_votes, positive_votes, first_verified_location, first_any_location)
    # location = (file_index, line_number)
    logger.info("Pass 1: Counting votes...")
    vote_stats: dict[str, list] = {}  # key -> [total, positive, first_verified_loc, first_any_loc]

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
                    # [total, positive, first_verified_loc, first_any_loc]
                    vote_stats[key] = [0, 0, None, (file_idx, line_num)]

                vote_stats[key][0] += 1  # total
                if is_verified:
                    vote_stats[key][1] += 1  # positive
                    if vote_stats[key][2] is None:
                        vote_stats[key][2] = (file_idx, line_num)  # first verified location

    # Determine which keys pass the threshold and where to read them from
    logger.info("Filtering by threshold...")
    keys_to_write: dict[str, tuple] = {}  # key -> (location, total, positive)
    for key, stats in vote_stats.items():
        total, positive, verified_loc, any_loc = stats
        if total > 0 and (positive / total) >= threshold:
            # Prefer verified location, fallback to any location
            loc = verified_loc if verified_loc is not None else any_loc
            keys_to_write[key] = (loc, total, positive)

    logger.info(
        "  %d questions passed threshold (out of %d total)", len(keys_to_write), len(vote_stats)
    )

    # Free memory from vote_stats
    del vote_stats

    # Pass 2: Read only the records we need and write them
    logger.info("Pass 2: Writing verified questions...")

    # Group by file for efficient reading
    records_by_file: dict[
        int, dict[int, tuple]
    ] = {}  # file_idx -> {line_num -> (key, total, positive)}
    for key, (loc, total, positive) in keys_to_write.items():
        file_idx, line_num = loc
        if file_idx not in records_by_file:
            records_by_file[file_idx] = {}
        records_by_file[file_idx][line_num] = (key, total, positive)

    count = 0
    with output_file.open("w", encoding="utf-8") as fout:
        for file_idx, line_nums in sorted(records_by_file.items()):
            input_file = input_files[file_idx]
            logger.info("  Reading %d records from %s...", len(line_nums), input_file.name)

            with input_file.open("r", encoding="utf-8") as fin:
                for line_num, line in enumerate(fin):
                    if line_num not in line_nums:
                        continue

                    key, total, positive = line_nums[line_num]

                    try:
                        result = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    new_record = result.copy()
                    new_record = remove_keys(new_record, _KEYS_TO_REMOVE)

                    # Add voting stats for traceability
                    new_record["question_voting_pass_rate"] = positive / total
                    new_record["question_voting_total"] = total

                    fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                    count += 1

    logger.info("Total verified questions prepared: %d", count)


def construct_answer_verify_input(input_dir: Path, output_file: Path):
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

                    generation = result.get("generation", "")
                    if not generation:
                        ser_out = result.get("serialized_output", [])
                        if isinstance(ser_out, list) and len(ser_out) > 0:
                            generation = ser_out[0].get("content", "")

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


def filter_verified_answers(input_dir: Path, output_file: Path, threshold: float = 0.5):
    """Filter verified answers based on 'Yes' responses and majority voting."""
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

    # Aggregate results by question identifier (source_file + question_index)
    # This assumes question order is deterministic and identical across runs if using multiple seeds
    results_by_question: dict[tuple[str, int], list[dict[str, Any]]] = {}

    for input_file in input_files:
        with input_file.open("r", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Unique identifier for the question
                # We use source_file and question_index which were added in construct_answer_verify_input
                key = (result.get("source_file", "unknown"), result.get("question_index", -1))

                if key not in results_by_question:
                    results_by_question[key] = []

                # Check verification result
                generation = result.get("generation", "")
                if not generation:
                    ser_out = result.get("serialized_output", [])
                    if isinstance(ser_out, list) and len(ser_out) > 0:
                        generation = ser_out[0].get("content", "")

                is_verified = False
                if "<|channel|>final<|message|>" in generation:
                    final_ans = generation.split("<|channel|>final<|message|>")[-1].strip()
                    if "Yes" in final_ans:
                        is_verified = True
                elif "Yes" in generation and len(generation) < 10:
                    is_verified = True
                elif "Yes" in generation:
                    if generation.rfind("No") > generation.rfind("Yes"):
                        is_verified = False
                    else:
                        is_verified = True

                # Store the result for voting
                results_by_question[key].append({"is_verified": is_verified, "data": result})

    count = 0
    with output_file.open("w", encoding="utf-8") as fout:
        for _, votes in results_by_question.items():
            total_votes = len(votes)
            positive_votes = sum(1 for v in votes if v["is_verified"])

            # Majority voting check
            # If pass_rate > threshold, we keep the answer
            # For multiple votes, we take the first answer data as the record base
            # Ideally we'd pick the "best" answer, but for now we assume consistency or pick one
            if total_votes > 0 and (positive_votes / total_votes) >= threshold:
                # Pick a successful result to write, or just the first one if mixed
                # Prefer writing a verified record if available
                base_record = next((v["data"] for v in votes if v["is_verified"]), votes[0]["data"])

                new_record = base_record.copy()
                new_record = remove_keys(new_record, _KEYS_TO_REMOVE)
                # Add voting metadata
                new_record["voting_pass_rate"] = positive_votes / total_votes
                new_record["voting_total"] = total_votes

                fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                count += 1

    logger.info("Total answers passed majority voting (%s): %d", threshold, count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    p1 = subparsers.add_parser("construct_question_generate_input")
    p1.add_argument("--input_folder", type=Path, required=True)
    p1.add_argument("--output_file", type=Path, required=True)

    p2 = subparsers.add_parser("construct_question_verify_input")
    p2.add_argument("--input_dir", type=Path, required=True)
    p2.add_argument("--output_file", type=Path, required=True)

    p3 = subparsers.add_parser("construct_answer_generate_input")
    p3.add_argument("--input_dir", type=Path, required=True)
    p3.add_argument("--output_file", type=Path, required=True)
    p3.add_argument("--threshold", type=float, default=0.5)

    p4 = subparsers.add_parser("construct_answer_verify_input")
    p4.add_argument("--input_dir", type=Path, required=True)
    p4.add_argument("--output_file", type=Path, required=True)

    p5 = subparsers.add_parser("filter_verified_answers")
    p5.add_argument("--input_dir", type=Path, required=True)
    p5.add_argument("--output_file", type=Path, required=True)
    p5.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()

    if args.command == "construct_question_generate_input":
        construct_question_generate_input(args.input_folder, args.output_file)
    elif args.command == "construct_question_verify_input":
        construct_question_verify_input(args.input_dir, args.output_file)
    elif args.command == "construct_answer_generate_input":
        construct_answer_generate_input(args.input_dir, args.output_file, args.threshold)
    elif args.command == "construct_answer_verify_input":
        construct_answer_verify_input(args.input_dir, args.output_file)
    elif args.command == "filter_verified_answers":
        filter_verified_answers(args.input_dir, args.output_file, args.threshold)
