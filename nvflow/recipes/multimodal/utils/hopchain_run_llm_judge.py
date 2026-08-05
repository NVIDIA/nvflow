#!/usr/bin/env python
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
"""Run a LiteLLM-backed judge over verified HopChain candidate queries."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

os.environ["LITELLM_LOG"] = "WARNING"

from litellm import completion
from pydantic import BaseModel, Field
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm.auto import tqdm

from nvflow.recipes.multimodal.utils.hopchain_llm_judge_common import (
    load_prompt_template,
    parse_judge_response_text,
)
from nvflow.recipes.multimodal.utils.hopchain_sdg_models import (
    LLMJudgeAnswer,
    LLMJudgeEvaluationRecord,
    VerifiedHopChainQuery,
)
from nvflow.recipes.multimodal.utils.image_filter_models import GenerationStats
from nvflow.recipes.multimodal.utils.image_utils import get_image_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
PROGRESS_LOG_EVERY = 10


class JudgeRunSummary(BaseModel):
    """Summary metadata for one LLM judge run."""

    judge_name: str
    provider: str
    model: str
    litellm_model: str
    input_file: str
    output_file: str
    total_records_processed: int = Field(ge=0)
    status_counts: dict[str, int] = Field(default_factory=dict)


class JudgeTaskInput(BaseModel):
    """One task sent to the worker pool."""

    line_num: int = Field(ge=1)
    record: VerifiedHopChainQuery


class JudgeTaskOutput(BaseModel):
    """Worker result used to preserve input ordering."""

    line_num: int = Field(ge=1)
    record: LLMJudgeEvaluationRecord


def build_error_task_output(
    *,
    task: JudgeTaskInput,
    judge_name: str,
    provider: str,
    model: str,
    status: Literal["parse_error", "api_error"],
    error: str,
    raw_response: str,
    generation_stats: GenerationStats | None = None,
) -> JudgeTaskOutput:
    """Build a structured error record for one failed judge task."""
    return JudgeTaskOutput(
        line_num=task.line_num,
        record=LLMJudgeEvaluationRecord(
            **task.record.model_dump(),
            llm_judge=LLMJudgeAnswer(
                judge_name=judge_name,
                provider=provider,
                model=model,
                answer="",
                normalized_answer="",
                confidence="unknown",
                reasoning=None,
                raw_response=raw_response,
                generation_stats=generation_stats or GenerationStats(),
            ),
            judge_status=status,
            judge_error=error,
        ),
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run an LLM judge over HopChain candidates")
    parser.add_argument("--input", required=True, help="VerifiedHopChainQuery JSONL")
    parser.add_argument(
        "--output", required=True, help="Output JSONL with one judge result per input row"
    )
    parser.add_argument("--summary", required=True, help="Summary JSON path")
    parser.add_argument("--prompt", required=True, help="Judge prompt template path")
    parser.add_argument("--judge-name", required=True, help="Logical judge name for output records")
    parser.add_argument("--provider", required=True, help="LiteLLM provider name, e.g. openai")
    parser.add_argument("--model", required=True, help="Provider model name, e.g. gpt-5.4")
    parser.add_argument("--api-key-name", required=True, help="API-key environment variable name")
    parser.add_argument("--api-base", default=None, help="Optional custom API base URL")
    parser.add_argument("--max-image-dimension", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high"),
        default=None,
        help="Optional OpenAI reasoning effort to pass through LiteLLM.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args()


def load_input_tasks(input_path: Path) -> list[JudgeTaskInput]:
    """Load verified candidate records from JSONL."""
    tasks: list[JudgeTaskInput] = []
    with input_path.open("r") as input_file:
        for line_num, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            record = VerifiedHopChainQuery.model_validate(json.loads(line))
            tasks.append(JudgeTaskInput(line_num=line_num, record=record))
    return tasks


def build_litellm_model(provider: str, model: str) -> str:
    """Build a LiteLLM model string."""
    if "/" in model:
        return model
    return f"{provider}/{model}"


def load_api_key(api_key_name: str) -> str:
    """Load a judge API key from the environment."""
    api_key = os.getenv(api_key_name)
    if api_key:
        return api_key
    raise ValueError(f"Credential {api_key_name} is not set in the environment")


def iter_completed_judge_futures(
    futures: dict[Any, JudgeTaskInput], total: int, judge_name: str
) -> Any:
    """Wrap completed judge futures with a progress bar."""
    return tqdm(
        as_completed(futures),
        total=total,
        desc=f"{judge_name} LLM judge",
        unit="query",
        dynamic_ncols=True,
    )


def build_completion_retryer(*, judge_name: str, line_num: int, max_retries: int) -> Retrying:
    """Build the tenacity retry policy for one completion call."""
    max_attempts = max_retries + 1

    def log_before_sleep(retry_state: RetryCallState) -> None:
        sleep_seconds = (
            retry_state.next_action.sleep if retry_state.next_action is not None else 0.0
        )
        logger.warning(
            "Judge %s failed on line %s attempt %s/%s, retrying in %.2fs",
            judge_name,
            line_num,
            retry_state.attempt_number,
            max_attempts,
            sleep_seconds,
        )

    return Retrying(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(multiplier=1, max=10),
        before_sleep=log_before_sleep,
        reraise=True,
    )


def build_messages(
    prompt_template: str, record: VerifiedHopChainQuery, max_image_dimension: int
) -> list[dict[str, Any]]:
    """Build a LiteLLM multimodal message payload."""
    prompt_text = prompt_template.format(question=record.question)
    image_url = get_image_url(
        image_path=record.image_fullpath,
        use_base64=True,
        max_dimension=max_image_dimension,
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def extract_response_text(response: Any) -> str:
    """Flatten LiteLLM response content into plain text."""
    choice = response.choices[0]
    message = choice.message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(content)


def serialize_response(response: Any) -> str:
    """Serialize a LiteLLM response object for debugging."""
    if hasattr(response, "model_dump_json"):
        return response.model_dump_json(indent=2, warnings="none")
    if hasattr(response, "model_dump"):
        return json.dumps(response.model_dump(warnings="none"), indent=2, default=str)
    return json.dumps(response, indent=2, default=str)


def get_completion_tokens(response: Any) -> int | None:
    """Extract completion token count from a LiteLLM response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    if hasattr(usage, "completion_tokens"):
        return usage.completion_tokens
    if isinstance(usage, dict):
        completion_tokens = usage.get("completion_tokens")
        if isinstance(completion_tokens, int):
            return completion_tokens
    return None


def run_single_judge(
    *,
    task: JudgeTaskInput,
    prompt_template: str,
    litellm_model: str,
    judge_name: str,
    provider: str,
    model: str,
    api_key: str,
    api_base: str | None,
    max_image_dimension: int,
    temperature: float,
    top_p: float,
    reasoning_effort: str | None,
    timeout_seconds: float,
    max_retries: int,
) -> JudgeTaskOutput:
    """Run one judge inference with retries and structured output capture."""
    messages = build_messages(prompt_template, task.record, max_image_dimension=max_image_dimension)
    completion_kwargs: dict[str, Any] = {
        "model": litellm_model,
        "messages": messages,
        "api_key": api_key,
        "temperature": temperature,
        "top_p": top_p,
        "timeout": timeout_seconds,
        "drop_params": True,
    }
    if reasoning_effort is not None:
        completion_kwargs["reasoning_effort"] = reasoning_effort
    if api_base:
        completion_kwargs["api_base"] = api_base

    retryer = build_completion_retryer(
        judge_name=judge_name,
        line_num=task.line_num,
        max_retries=max_retries,
    )

    try:

        def request_completion() -> tuple[Any, float]:
            started_at = time.monotonic()
            response = completion(**completion_kwargs)
            generation_time = time.monotonic() - started_at
            return response, generation_time

        response, generation_time = retryer(request_completion)
    except Exception as exc:
        logger.exception("Judge %s failed on line %s: %s", judge_name, task.line_num, exc)
        return build_error_task_output(
            task=task,
            judge_name=judge_name,
            provider=provider,
            model=model,
            status="api_error",
            error=str(exc),
            raw_response=str(exc),
        )

    response_text = extract_response_text(response)
    raw_response = serialize_response(response)
    try:
        parsed = parse_judge_response_text(response_text)
    except Exception as exc:
        logger.exception("Failed to parse judge response for line %s: %s", task.line_num, exc)
        return build_error_task_output(
            task=task,
            judge_name=judge_name,
            provider=provider,
            model=model,
            status="parse_error",
            error=str(exc),
            raw_response=raw_response,
            generation_stats=GenerationStats(
                num_generated_tokens=get_completion_tokens(response),
                generation_time=generation_time,
            ),
        )
    judge_record = LLMJudgeEvaluationRecord(
        **task.record.model_dump(),
        llm_judge=LLMJudgeAnswer(
            judge_name=judge_name,
            provider=provider,
            model=model,
            answer=parsed.final_answer,
            normalized_answer=parsed.normalized_answer,
            confidence=parsed.confidence,
            reasoning=parsed.reasoning,
            raw_response=raw_response,
            generation_stats=GenerationStats(
                num_generated_tokens=get_completion_tokens(response),
                generation_time=generation_time,
            ),
        ),
        judge_status="parsed",
        judge_error=None,
    )
    return JudgeTaskOutput(line_num=task.line_num, record=judge_record)


def main() -> None:
    """Entry point."""
    args = parse_args()
    api_key = load_api_key(args.api_key_name)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_template = load_prompt_template(args.prompt)
    litellm_model = build_litellm_model(args.provider, args.model)

    tasks = load_input_tasks(input_path)
    logger.info("Loaded %s candidate queries from %s", len(tasks), input_path)

    results: list[JudgeTaskOutput] = []
    completed_status_counts: Counter[str] = Counter()
    total_tasks = len(tasks)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {
            executor.submit(
                run_single_judge,
                task=task,
                prompt_template=prompt_template,
                litellm_model=litellm_model,
                judge_name=args.judge_name,
                provider=args.provider,
                model=args.model,
                api_key=api_key,
                api_base=args.api_base,
                max_image_dimension=args.max_image_dimension,
                temperature=args.temperature,
                top_p=args.top_p,
                reasoning_effort=args.reasoning_effort,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
            ): task
            for task in tasks
        }
        for completed_count, future in enumerate(
            iter_completed_judge_futures(
                future_to_task, total=total_tasks, judge_name=args.judge_name
            ),
            start=1,
        ):
            task = future_to_task[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.exception("Unhandled worker failure on line %s: %s", task.line_num, exc)
                result = build_error_task_output(
                    task=task,
                    judge_name=args.judge_name,
                    provider=args.provider,
                    model=args.model,
                    status="api_error",
                    error=f"Unhandled worker failure: {exc}",
                    raw_response=f"Unhandled worker failure: {exc}",
                )
            results.append(result)
            completed_status_counts[result.record.judge_status] += 1
            if (
                completed_count == 1
                or completed_count % PROGRESS_LOG_EVERY == 0
                or completed_count == total_tasks
            ):
                logger.info(
                    "Judge %s progress: %s/%s completed (parsed=%s, parse_error=%s, api_error=%s)",
                    args.judge_name,
                    completed_count,
                    total_tasks,
                    completed_status_counts.get("parsed", 0),
                    completed_status_counts.get("parse_error", 0),
                    completed_status_counts.get("api_error", 0),
                )

    ordered_results = sorted(results, key=lambda item: item.line_num)
    if len(ordered_results) != total_tasks:
        raise RuntimeError(
            f"Expected {total_tasks} judge outputs but wrote {len(ordered_results)}."
        )
    status_counts: dict[str, int] = {}
    with output_path.open("w") as output_file:
        for item in ordered_results:
            status_counts[item.record.judge_status] = (
                status_counts.get(item.record.judge_status, 0) + 1
            )
            output_file.write(item.record.model_dump_json() + "\n")

    summary = JudgeRunSummary(
        judge_name=args.judge_name,
        provider=args.provider,
        model=args.model,
        litellm_model=litellm_model,
        input_file=str(input_path),
        output_file=str(output_path),
        total_records_processed=len(ordered_results),
        status_counts=status_counts,
    )
    Path(args.summary).write_text(summary.model_dump_json(indent=2))
    logger.info("Wrote %s judge records to %s", len(ordered_results), output_path)


if __name__ == "__main__":
    main()
