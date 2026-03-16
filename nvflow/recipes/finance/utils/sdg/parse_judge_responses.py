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
import ast
import json
import re

from nvflow.utils import setup_logger

logger = setup_logger(__name__)

remove_keys = [
    "num_generated_tokens",
    "reasoning_content",
    "finish_reason",
    "serialized_output",
    "generation_start_time",
    "generation_end_time",
    "generation_time",
    "generation",
]


def parse_generation_text(generation_text):
    """Extract and parse JSON from generation field."""
    # Remove markdown code blocks
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", generation_text, re.DOTALL)

    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_str = generation_text.strip()

    def validate_judge_response(parsed):
        if "rating" not in parsed:
            return False, "Missing 'rating' field"
        if "reasoning" not in parsed:
            return False, "Missing 'reasoning' field"
        return True, None

    try:
        parsed_json = json.loads(json_str)
        is_valid, error_msg = validate_judge_response(parsed_json)
        if is_valid:
            return parsed_json, None
        else:
            return None, error_msg + ": " + json_str
    except json.JSONDecodeError:
        pass

    # Try Python literal eval (handles single quotes)
    try:
        parsed_json = ast.literal_eval(json_str)
        is_valid, error_msg = validate_judge_response(parsed_json)
        if is_valid:
            return parsed_json, None
        else:
            return None, error_msg + ": " + json_str
    except (ValueError, SyntaxError):
        return None, "Unable to parse JSON: " + json_str


def parse_judge_responses(input_file, output_file):
    log_file = output_file.replace(".jsonl", "_log.txt")
    num_total_entries = 0
    num_failed_to_parse = 0
    num_successfully_parsed = 0

    with (
        open(input_file) as reader,
        open(output_file, "w") as writer,
        open(log_file, "w") as log_writer,
    ):
        for line in reader:
            row = json.loads(line)
            num_total_entries += 1
            try:
                generation = row["generation"]
            except KeyError:
                msg = f"No generation field found for entry {num_total_entries}: {row.get('problem', 'unknown')}"
                # print(msg)
                log_writer.write(msg + "\n")
                num_failed_to_parse += 1
                continue

            judge_response, error_msg = parse_generation_text(generation)

            row["judge_generation"] = generation
            # Remove unnecessary keys
            for key in remove_keys:
                if key in row:
                    del row[key]

            if judge_response:
                row["judge_response"] = judge_response
                num_successfully_parsed += 1
                writer.write(json.dumps(row) + "\n")
            else:
                num_failed_to_parse += 1
                msg = f"Failed to parse generation for entry {num_total_entries}: {error_msg}"
                # print(msg)
                log_writer.write(msg + "\n")

        # Write summary to log
        log_writer.write(f"\n{'=' * 60}\n")
        log_writer.write("SUMMARY\n")
        log_writer.write(f"{'=' * 60}\n")
        log_writer.write(f"Total entries: {num_total_entries}\n")
        log_writer.write(f"Successfully parsed: {num_successfully_parsed}\n")
        log_writer.write(f"Failed to parse: {num_failed_to_parse}\n")
        log_writer.write(
            f"Success rate: {num_successfully_parsed / num_total_entries * 100:.2f}%\n"
        )

        # Print summary to console as well
        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info("Total entries: %d", num_total_entries)
        logger.info("Successfully parsed: %d", num_successfully_parsed)
        logger.info("Failed to parse: %d", num_failed_to_parse)
        logger.info("Success rate: %.2f%%", num_successfully_parsed / num_total_entries * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse judge responses from LLM generation field and extract rating/reasoning"
    )
    parser.add_argument(
        "--input_file", required=True, help="Input JSONL file with generation field"
    )
    parser.add_argument(
        "--output_file", required=True, help="Output JSONL file with judge_response field"
    )
    args = parser.parse_args()

    parse_judge_responses(args.input_file, args.output_file)
