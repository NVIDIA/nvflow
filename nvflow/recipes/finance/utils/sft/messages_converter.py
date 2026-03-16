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
"""Convert Qwen3 chat-templated training data to OpenAI messages format.

Parses Qwen3 chat template special tokens from prepare_for_sft output and converts
to structured messages array format compatible with OpenAI fine-tuning.

Input format (from train_validation_split):
    {"input": "<|im_start|>system\\n...<|im_end|>\\n<|im_start|>user\\n...", "output": "...", "uuid": "..."}

Output format (OpenAI messages):
    {
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "reasoning_content": "...", "content": "..."}
      ],
      "metadata": {"uuid": "...", "sdg_model": "..."}
    }

Usage:
    python -m nvflow.recipes.finance.utils.sft.messages_converter \\
        train.jsonl val.jsonl \\
        --output_dir ./messages
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from nvflow.utils import setup_logger

logger = setup_logger(__name__)

# Qwen3 chat template pattern: <|im_start|>role\ncontent<|im_end|>
QWEN_CHAT_PATTERN = re.compile(r"<\|im_start\|>(\w+)\n(.*?)(?:<\|im_end\|>|$)", re.DOTALL)

# Pattern to extract reasoning from <think> tags
THINK_PATTERN = re.compile(r"<think>(.*?)</think>\s*(.*)", re.DOTALL)

VALID_ROLES = {"system", "user", "assistant"}


# =============================================================================
# Conversion
# =============================================================================


def parse_chat_template(text: str) -> list[dict[str, str]]:
    """Parse Qwen3 chat-formatted string back to messages list."""
    messages = []
    for role, content in QWEN_CHAT_PATTERN.findall(text):
        content = content.strip()
        if content:  # Skip empty messages (like trailing assistant prompt)
            messages.append({"role": role, "content": content})
    return messages


def strip_chat_tokens(text: str) -> str:
    """Strip Qwen3 chat template tokens from text."""
    text = text.strip()
    if text.endswith("<|im_end|>"):
        text = text[:-10].strip()
    return text


def extract_reasoning(output: str) -> tuple[str | None, str]:
    """Extract reasoning content from <think> tags."""
    output = strip_chat_tokens(output)
    match = THINK_PATTERN.search(output)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, output.strip()


def convert_record(
    record: dict[str, Any],
    extract_reasoning_flag: bool = True,
    sdg_model: str | None = None,
) -> dict[str, Any]:
    """Convert a single record to OpenAI messages format."""
    input_text = record.get("input", "")
    output_text = record.get("output", "")
    uuid = record.get("uuid", "")

    # Start with empty system message (required by OpenAI format)
    messages: list[dict[str, str]] = [{"role": "system", "content": ""}]

    # Parse chat template to get user messages
    messages.extend(parse_chat_template(input_text))

    # Build assistant message
    assistant_msg: dict[str, str] = {"role": "assistant"}
    if extract_reasoning_flag:
        reasoning, content = extract_reasoning(output_text)
        if reasoning:
            assistant_msg["reasoning_content"] = reasoning
        assistant_msg["content"] = content
    else:
        assistant_msg["content"] = strip_chat_tokens(output_text)
    messages.append(assistant_msg)

    # Build metadata
    metadata: dict[str, str] = {"uuid": uuid}
    if sdg_model:
        metadata["sdg_model"] = sdg_model

    return {"messages": messages, "metadata": metadata}


def convert_file(
    input_path: Path,
    output_path: Path,
    extract_reasoning_flag: bool = True,
    sdg_model: str | None = None,
) -> tuple[int, int]:
    """Convert a JSONL file to messages format. Returns (success_count, error_count)."""
    success_count = 0
    error_count = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        open(input_path, encoding="utf-8") as f_in,
        open(output_path, "w", encoding="utf-8") as f_out,
    ):
        for line_num, line in enumerate(f_in, 1):
            try:
                record = json.loads(line.strip())
                converted = convert_record(record, extract_reasoning_flag, sdg_model)
                f_out.write(json.dumps(converted, ensure_ascii=False) + "\n")
                success_count += 1
            except json.JSONDecodeError as e:
                logger.warning(f"[{input_path.name}:{line_num}] JSON decode error: {e}")
                error_count += 1
            except Exception as e:
                logger.warning(f"[{input_path.name}:{line_num}] Conversion error: {e}")
                error_count += 1

    return success_count, error_count


# =============================================================================
# Validation
# =============================================================================


def validate_record(record: dict[str, Any], line_num: int) -> list[str]:
    """Validate a converted record. Returns list of issues (empty if valid)."""
    issues: list[str] = []

    # Check messages
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) == 0:
        return [f"Line {line_num}: 'messages' must be a non-empty list"]

    for i, msg in enumerate(messages):
        path = f"messages[{i}]"
        if not isinstance(msg, dict):
            issues.append(f"Line {line_num}: {path} must be an object")
            continue

        role = msg.get("role")
        if role not in VALID_ROLES:
            issues.append(f"Line {line_num}: {path}.role invalid: {role!r}")
        if "content" not in msg:
            issues.append(f"Line {line_num}: {path} missing 'content'")
        elif not isinstance(msg["content"], str):
            issues.append(f"Line {line_num}: {path}.content must be string")
        if "reasoning_content" in msg and role != "assistant":
            issues.append(f"Line {line_num}: {path}.reasoning_content only valid for assistant")

    # Check metadata
    metadata = record.get("metadata")
    if not isinstance(metadata, dict) or not metadata.get("uuid"):
        issues.append(f"Line {line_num}: metadata.uuid is required")

    return issues


def validate_file(file_path: Path) -> tuple[int, int, list[str]]:
    """Validate a JSONL file. Returns (valid_count, invalid_count, sample_issues)."""
    valid_count = 0
    invalid_count = 0
    sample_issues: list[str] = []

    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                issues = validate_record(record, line_num)
                if issues:
                    invalid_count += 1
                    if len(sample_issues) < 10:
                        sample_issues.extend(issues[: 10 - len(sample_issues)])
                else:
                    valid_count += 1
            except json.JSONDecodeError as e:
                invalid_count += 1
                if len(sample_issues) < 10:
                    sample_issues.append(f"Line {line_num}: JSON decode error: {e}")

    return valid_count, invalid_count, sample_issues


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    """Convert training datasets to OpenAI messages format."""
    parser = argparse.ArgumentParser(
        description="Convert Qwen3 chat-templated training data to OpenAI messages format",
    )
    parser.add_argument("input_files", nargs="+", help="Input JSONL file(s)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument(
        "--no_extract_reasoning",
        action="store_true",
        help="Don't extract reasoning from <think> tags",
    )
    parser.add_argument("--sdg_model", help="SDG model name (added to metadata)")
    args = parser.parse_args()

    extract_reasoning_flag = not args.no_extract_reasoning
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log configuration
    logger.info("=" * 60)
    logger.info("CONVERT TO OPENAI MESSAGES FORMAT")
    logger.info(f"Input: {len(args.input_files)} file(s)")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Extract reasoning: {extract_reasoning_flag}")
    if args.sdg_model:
        logger.info(f"SDG model: {args.sdg_model}")
    logger.info("=" * 60)

    # Convert files
    total_converted = 0
    total_errors = 0
    output_files: list[Path] = []

    for input_file in args.input_files:
        input_path = Path(input_file)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            continue

        output_path = output_dir / input_path.name
        output_files.append(output_path)

        logger.info(f"Converting: {input_path.name}")
        success, errors = convert_file(
            input_path, output_path, extract_reasoning_flag, args.sdg_model
        )
        total_converted += success
        total_errors += errors
        logger.info(f"  → {success:,} converted, {errors:,} errors")

    # Validate output files
    total_valid = 0
    total_invalid = 0
    all_issues: list[str] = []

    for output_path in output_files:
        if not output_path.exists():
            continue
        logger.info(f"Validating: {output_path.name}")
        valid, invalid, issues = validate_file(output_path)
        total_valid += valid
        total_invalid += invalid
        all_issues.extend(issues)
        logger.info(f"  → {valid:,} valid, {invalid:,} invalid")

    # Report issues
    if all_issues:
        logger.warning("Sample validation issues:")
        for issue in all_issues[:10]:
            logger.warning(f"  {issue}")

    # Final summary
    logger.info("=" * 60)
    logger.info(f"Total: {total_valid:,} valid, {total_invalid:,} invalid, {total_errors:,} errors")
    if total_invalid == 0 and total_errors == 0:
        logger.info("✓ All records converted and validated successfully")
    else:
        logger.warning("✗ Issues found - see above for details")
    logger.info("=" * 60)

    return 1 if (total_errors > 0 or total_invalid > 0) else 0


if __name__ == "__main__":
    sys.exit(main())
