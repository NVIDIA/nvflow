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

import jsonlines

from nvflow.recipes.finance.utils.shared.metadata_utils import nest_metadata_fields
from nvflow.utils import setup_logger

# Initialize logger
logger = setup_logger(__name__)

# Additional keys to remove (not part of metadata)
additional_remove_keys = [
    "ground_truth",
    "context_markdown_with_headers",
    "context_markdown_without_headers",
    "context_html_with_headers",
    "context_html_without_headers",
    "source_info",
    "UUID",
]


def parse_generation_text(generation_text):
    """Extract and parse JSON from generation field."""
    # Remove markdown code blocks
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", generation_text, re.DOTALL)

    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_str = generation_text.strip()

    def validate_questions_list(parsed):
        if "questions" not in parsed:
            return False
        questions_list = parsed["questions"]
        if not isinstance(questions_list, list):
            return False
        return True

    try:
        parsed_json = json.loads(json_str)
        if validate_questions_list(parsed_json):
            return parsed_json, None
        else:
            return None, "Invalid questions list: " + json_str
    except json.JSONDecodeError:
        pass

    # Try Python literal eval (handles single quotes)
    try:
        parsed_json = ast.literal_eval(json_str)
        logger.debug("literal_eval fallback parsed: %s", parsed_json)
        if validate_questions_list(parsed_json):
            return parsed_json, None
        else:
            return None, "Invalid questions list: " + json_str
    except (ValueError, SyntaxError):
        return None, "Unable to parse JSON: " + json_str


def parse_generations(input_file, output_file):
    log_file = output_file.replace(".jsonl", "_log.txt")
    num_original_questions, num_failed_to_parse = 0, 0
    num_generated_questions, num_generated_questions_after_deduplication = 0, 0
    with (
        jsonlines.open(input_file) as reader,
        jsonlines.open(output_file, "w") as writer,
        open(log_file, "w") as log_writer,
    ):
        deduplicated_questions = set()
        for row in reader:
            num_original_questions += 1
            try:
                generation = row["generation"]
            except KeyError:
                msg = f"No generation field found for: {row}"
                logger.error(msg)
                log_writer.write(msg + "\n")
                num_failed_to_parse += 1
                continue

            original_question = row["original_question"]
            deduplicated_questions.add(original_question.lower())

            questions_list_json, un_parsed_text = parse_generation_text(generation)

            if questions_list_json:
                questions_list = questions_list_json["questions"]
                for sdg_question in questions_list:
                    num_generated_questions += 1
                    if sdg_question.lower() in deduplicated_questions:
                        continue
                    num_generated_questions_after_deduplication += 1
                    deduplicated_questions.add(sdg_question.lower())
                    row["problem"] = sdg_question

                    # Remove additional keys (not part of metadata)
                    for key in additional_remove_keys:
                        if key in row:
                            del row[key]

                    # Nest metadata fields before writing
                    row = nest_metadata_fields(row, "generate_questions_metadata")
                    writer.write(row)
            else:
                num_failed_to_parse += 1
                msg = f"Failed to parse generation for: {un_parsed_text}"
                logger.error(msg)
                log_writer.write(msg + "\n")
        log_writer.write(f"Number of original questions: {num_original_questions}\n")
        log_writer.write(f"Number of failed to parse: {num_failed_to_parse}\n")
        log_writer.write(f"Number of generated questions: {num_generated_questions}\n")
        log_writer.write(
            f"Number of generated questions after deduplication: {num_generated_questions_after_deduplication}\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="Input JSONL file")
    parser.add_argument("--output_file", required=True, help="Output JSONL file")
    args = parser.parse_args()

    parse_generations(args.input_file, args.output_file)
