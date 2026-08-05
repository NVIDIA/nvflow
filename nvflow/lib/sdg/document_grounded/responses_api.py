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
"""Convert between SDG JSONL and NeMo-Gym Responses API format.

Two operations:

``render_and_convert``
    Reads an SDG JSONL file, renders each record's prompt using a YAML
    template, and writes Responses API formatted output.  Combines the
    GRPO pipeline's ``apply_prompt_template`` + ``convert_to_responses_api``
    into a single step.

``extract_generations``
    Reads NeMo-Gym rollout output and extracts the model-generated text
    back into SDG-compatible JSONL fields.

CLI::

    python -m nvflow.lib.sdg.document_grounded.responses_api render_and_convert \\
        --input_file INPUT --output_file OUTPUT --prompt_template TEMPLATE

    python -m nvflow.lib.sdg.document_grounded.responses_api extract_generations \\
        --input_file INPUT --output_file OUTPUT [--generation_key KEY]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from nvflow.utils import setup_logger

logger = setup_logger(__name__)

# ============================================================================
# Prompt rendering
# ============================================================================


def load_prompt_template(template_path: str | Path) -> tuple[str | None, str]:
    """Load the (system, user) prompt template strings from a YAML file.

    Expects a YAML file with a required ``user:`` key and an optional
    ``system:`` key, each holding a template string with ``{variable}``
    placeholders (e.g. ``{context}``, ``{problem}``).

    Returns a tuple ``(system_template, user_template)`` where
    ``system_template`` is ``None`` if the YAML has no ``system:`` block.

    NOTE: Earlier versions of this function silently dropped the ``system:``
    block, which caused the Responses-API path to send prompts without the
    system instructions defined in the prompt YAML (e.g.
    ``document_grounded_verify_questions.yaml`` lost the
    "Respond ONLY with 'Yes' or 'No'." directive).  The model then emitted
    verbose markdown that the downstream Yes/No parser misclassified,
    dropping a large fraction of valid questions versus the legacy
    nemo-skills path.  Always returning both roles fixes that.
    """
    with open(template_path) as f:
        config = yaml.safe_load(f)
    user_template = config.get("user")
    if not user_template:
        raise ValueError(f"Prompt template {template_path} must have a 'user:' key")
    system_template = config.get("system") or None
    return system_template, user_template


def render_prompt(record: dict[str, Any], template: str) -> str:
    """Substitute ``{variable}`` placeholders in *template* with record values.

    Missing keys are replaced with empty strings (no KeyError).
    Double braces ``{{`` / ``}}`` in the template are treated as literal
    braces (standard Python format_map behaviour).
    """

    class _DefaultDict(dict):
        def __missing__(self, key: str) -> str:
            return ""

    return template.format_map(_DefaultDict(record))


# ============================================================================
# SDG JSONL -> Responses API
# ============================================================================


def to_responses_api(
    record: dict[str, Any],
    rendered_prompt: str,
    rendered_system: str | None = None,
    system_role: str = "system",
) -> dict[str, Any]:
    """Wrap a rendered prompt into Responses API format.

    When ``rendered_system`` is provided, a system-instruction message is
    prepended using ``system_role`` (default ``"system"``).

    ``"system"`` is the universally-supported role: every standard chat
    template (Qwen, Gemma, Llama, ... and gpt-oss harmony) accepts it.  The
    OpenAI Responses convention ``"developer"`` is only understood by the
    gpt-oss/harmony template and makes other models' Jinja chat templates
    raise ``TemplateError: Unexpected message role`` (e.g. Qwen3.5).  Pass
    ``system_role="developer"`` explicitly only for harmony-format models.

    Returns a new dict with ``responses_create_params`` added and all
    original fields preserved.
    """
    messages: list[dict[str, str]] = []
    if rendered_system:
        messages.append({"role": system_role, "content": rendered_system})
    messages.append({"role": "user", "content": rendered_prompt})

    result = dict(record)
    result["responses_create_params"] = {"input": messages}
    return result


def _apply_field_mappers(
    record: dict[str, Any],
    field_mappers: dict[str, str] | None,
) -> dict[str, Any]:
    """Copy fields from *record* into a new dict according to *field_mappers*.

    ``field_mappers`` maps target field name -> source field name (or dotted
    path like ``"metadata.expected_answer"``).  Missing source paths are
    silently skipped.
    """
    if not field_mappers:
        return {}
    out: dict[str, Any] = {}
    for target, source in field_mappers.items():
        cur: Any = record
        for part in source.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                cur = None
                break
        if cur is not None:
            out[target] = cur
    return out


def render_and_convert(
    input_file: str | Path,
    output_file: str | Path,
    prompt_template: str | Path,
    *,
    inference_params: dict[str, Any] | None = None,
    extra_record_fields: dict[str, Any] | None = None,
    extra_record_field_mappers: dict[str, str] | None = None,
    system_role: str = "system",
) -> int:
    """Render prompts and convert SDG JSONL to Responses API format.

    Args:
        input_file: Path to SDG input JSONL.
        output_file: Path to write Responses API JSONL.
        prompt_template: Path to YAML prompt template.
        inference_params: Optional dict of inference parameters
            (e.g. ``temperature``, ``top_p``) merged into
            ``responses_create_params``.
        extra_record_fields: Optional static fields merged into every output
            record at top-level (e.g. ``verifier`` for ``format_verification``).
            Existing record fields take precedence.
        extra_record_field_mappers: Optional mapping ``target_field ->
            source_field`` (source supports dotted paths) that copies values
            from each input record into the corresponding top-level field of
            the output record (e.g. ``{"expected_answer": "answer"}`` for
            ``equivalence_llm_judge``).  Applied after ``extra_record_fields``
            and takes precedence over both static fields and existing record
            fields.
        system_role: Chat role used for the system-instruction message
            (default ``"system"``). See :func:`to_responses_api`.

    Returns:
        Number of records converted.
    """
    system_template, user_template = load_prompt_template(prompt_template)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Dedup by rendered prompt (same key as compute_join_id). Identical prompts
    # break the enrich join (ambiguous id) and the rollout/input count check.
    # They occur when the source corpus holds the same content under different
    # provenance (e.g. identical 10-K text at two file_path0 values) and the
    # upstream sampler happens to pick more than one -- nondeterministic across
    # runs. Keeping the first occurrence is safe: the rows differ only in
    # provenance metadata, and generating duplicate questions from identical
    # content is wasteful regardless.
    count = 0
    dropped_dups = 0
    seen: set[str] = set()
    # Write to a temp sibling, then atomically replace -- callers (e.g.
    # submit_gym_generation) skip re-rendering when this output file already
    # exists and is non-empty, so a half-written file from a killed/timed-out
    # job (OOM, walltime, node failure) must never be observable at the final
    # path: it would be mistaken for a complete render and silently feed
    # truncated input downstream.
    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(input_file) as fin, open(tmp_output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            rendered_user = render_prompt(record, user_template)
            rendered_system = render_prompt(record, system_template) if system_template else None
            result = to_responses_api(
                record, rendered_user, rendered_system, system_role=system_role
            )
            if inference_params:
                result["responses_create_params"].update(inference_params)
            join_key = _canonical_input(result["responses_create_params"])
            if join_key in seen:
                dropped_dups += 1
                continue
            seen.add(join_key)
            if extra_record_fields:
                for k, v in extra_record_fields.items():
                    result.setdefault(k, v)
            mapped = _apply_field_mappers(record, extra_record_field_mappers)
            result.update(mapped)
            fout.write(json.dumps(result) + "\n")
            count += 1

    os.replace(tmp_output, output_path)

    if dropped_dups:
        logger.warning(
            "render_and_convert: dropped %d duplicate-prompt row(s) "
            "(identical rendered input; kept first occurrence) -> %d unique of %d total",
            dropped_dups,
            count,
            count + dropped_dups,
        )

    return count


# ============================================================================
# Stable join id (links rollout output rows back to input rows)
# ============================================================================


def _canonical_input(responses_create_params: dict[str, Any]) -> str:
    """Canonical JSON of ``responses_create_params.input`` (the chat messages).

    Only the ``input`` field is hashed.  ``ng_collect_rollouts`` merges global
    ``responses_create_params`` overrides (temperature, max_output_tokens, ...)
    into the *output* rows, but never touches ``input`` -- so hashing ``input``
    alone produces an identical id on both the input file and the rollout
    output, regardless of those overrides.

    Each message is reduced to its stable ``{role, content}`` pair before
    hashing.  The Gym rollout layer round-trips every input message through the
    Responses API schema and re-emits it with extra keys (e.g. ``type:
    "message"``) and a different key order.  Hashing the raw message dict would
    therefore produce different ids on the input file vs. the rollout output
    (observed: 1000/1000 mismatch).  ``role`` + ``content`` are the only fields
    the rollout preserves verbatim, so they form the stable join key.
    """
    rcp = responses_create_params or {}
    messages = rcp.get("input", [])
    normalized: list[Any] = []
    for msg in messages:
        if isinstance(msg, dict):
            normalized.append({"role": msg.get("role"), "content": msg.get("content")})
        else:
            normalized.append(msg)
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False)


def compute_join_id(record: dict[str, Any]) -> str:
    """Stable content hash linking a rollout output row back to its input row.

    Hashes ``responses_create_params.input`` (echoed verbatim in gym output),
    NOT ``_ng_task_index`` -- ng_collect_rollouts assigns ``_ng_task_index``
    per chunk (position in that chunk's remaining-input), so it collides across
    a merged multi-chunk file and is unsafe as a join key.
    """
    payload = _canonical_input(record.get("responses_create_params", {}))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


# ============================================================================
# Responses API (rollout output) -> SDG JSONL
# ============================================================================


def _extract_assistant_text(response: dict[str, Any]) -> str:
    """Extract the assistant's text from a NeMo-Gym response object.

    Handles both simple ``{"output": [{"content": "..."}]}`` and the
    structured ``{"output": [{"type": "message", "content": [...]}]}``
    formats that NeMo-Gym may return.
    """
    output = response.get("output", [])
    if not output:
        return ""

    texts: list[str] = []
    for item in output:
        if isinstance(item, dict):
            if item.get("type") == "message" and item.get("role") == "assistant":
                content = item.get("content", "")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "output_text":
                            texts.append(part.get("text", ""))
                elif isinstance(content, str):
                    texts.append(content)
            elif "content" in item and isinstance(item["content"], str):
                texts.append(item["content"])
            elif "text" in item and item.get("type") != "reasoning":
                # Skip type=reasoning items here; _extract_reasoning_text owns those.
                texts.append(item["text"])

    return "\n".join(texts) if texts else ""


def _extract_reasoning_text(response: dict[str, Any]) -> str:
    """Extract reasoning / chain-of-thought text from a NeMo-Gym response.

    The reasoning field's location depends on the upstream API endpoint and
    the vLLM reasoning_parser configuration.  Probed in priority order:

    1. **Responses API (gpt-oss harmony)** — reasoning lives in
       ``response.output[i]`` where ``type == "reasoning"``, with text
       inside ``summary[j].text`` (most common for gpt-oss) and/or
       ``content[j].text``.  Multiple ``reasoning`` items are concatenated.
    2. **Chat Completions API** (vLLM openai_gptoss/deepseek_r1 parsers) —
       top-level ``response.reasoning_content`` string.
    3. **Top-level fallback** — some providers expose ``response.reasoning``
       as a string.

    Returns an empty string when no reasoning is present.
    """
    texts: list[str] = []

    output = response.get("output", [])
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "reasoning":
                continue
            # gpt-oss / Responses API: text inside summary[].text
            summary = item.get("summary")
            if isinstance(summary, list):
                for part in summary:
                    if isinstance(part, dict):
                        t = part.get("text")
                        if isinstance(t, str) and t:
                            texts.append(t)
            # Alternative shape: text inside content[].text
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        t = part.get("text")
                        if isinstance(t, str) and t:
                            texts.append(t)
            elif isinstance(content, str) and content:
                texts.append(content)
            # Simplified inline shape
            t = item.get("text")
            if isinstance(t, str) and t:
                texts.append(t)

    if texts:
        return "\n".join(texts)

    # Chat Completions API path (vLLM openai_gptoss / deepseek_r1)
    rc = response.get("reasoning_content")
    if isinstance(rc, str) and rc:
        return rc

    # Top-level fallback
    r = response.get("reasoning")
    if isinstance(r, str) and r:
        return r

    return ""


def from_rollout_output(
    record: dict[str, Any],
    generation_key: str = "generation",
) -> dict[str, Any]:
    """Extract generation text from a rollout output record.

    The model's response is extracted and stored under *generation_key*.
    Reasoning content (if any) is stored under ``reasoning_content``.
    ``responses_create_params`` is removed from the output.
    """
    result = dict(record)
    response = result.pop("response", {})
    result.pop("responses_create_params", None)

    generation = _extract_assistant_text(response)
    result[generation_key] = generation

    reasoning = _extract_reasoning_text(response)
    if reasoning:
        result["reasoning_content"] = reasoning

    return result


def extract_generations(
    input_file: str | Path,
    output_file: str | Path,
    *,
    generation_key: str = "generation",
    original_input_file: str | Path | None = None,
) -> int:
    """Extract generations from rollout output JSONL back to SDG format.

    NeMo-Gym rollout output only contains Gym-specific fields (reward,
    verifier, response, etc.) and a ``_ng_task_index`` linking back to the
    input.  Original SDG fields (context, company_name, ...) are lost.

    When *original_input_file* is provided (the Responses-API JSONL that
    was fed to ``ng_collect_rollouts``), this function joins rollout
    results back with their original SDG fields.  The join key is the
    **line index** (0-based position in the original-input file), NOT
    the record's own ``_ng_task_index`` field — NeMo-Gym assigns its own
    line-position-based ``_ng_task_index`` which would not match the
    inherited ``_ng_task_index`` carried over from earlier stages.

    Hard alignment checks (raise ``RuntimeError`` on any mismatch — this
    catches the silent data-corruption mode where stale rollout caches
    get joined with new input data; see SKILL.md Gotcha #13):
      - every rollout's ``_ng_task_index`` must be an int within
        ``[0, len(original_input))``
      - every rollout must successfully join an original record
      - rollout count must equal original-input count (no truncation,
        no spurious duplicates).  ``ng_collect_rollouts`` writes one
        rollout per input record so this is a strict equality.

    Args:
        input_file: Path to NeMo-Gym rollout output JSONL.
        output_file: Path to write SDG-compatible JSONL.
        generation_key: Field name for the extracted generation text.
        original_input_file: Optional path to the Responses-API input
            JSONL.  When given, original fields are merged into each
            output record.

    Returns:
        Number of records processed.
    """
    original_records: dict[int, dict[str, Any]] = {}
    if original_input_file:
        with open(original_input_file) as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec.pop("responses_create_params", None)
                original_records[idx] = rec

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    unmatched_rollouts: list[int] = []
    out_of_range_indices: list[Any] = []
    seen_task_indices: set[int] = set()
    count = 0
    with open(input_file) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            result = from_rollout_output(record, generation_key=generation_key)
            task_idx = result.get("_ng_task_index")

            if original_records:
                if (
                    not isinstance(task_idx, int)
                    or task_idx < 0
                    or task_idx >= len(original_records)
                ):
                    out_of_range_indices.append(task_idx)
                elif task_idx not in original_records:
                    unmatched_rollouts.append(task_idx)
                else:
                    seen_task_indices.add(task_idx)
                    merged = dict(original_records[task_idx])
                    merged.update(result)
                    result = merged

            fout.write(json.dumps(result) + "\n")
            count += 1

    if original_records:
        errors: list[str] = []
        if out_of_range_indices:
            errors.append(
                f"{len(out_of_range_indices)} rollout(s) have _ng_task_index "
                f"outside [0, {len(original_records)}); first few: "
                f"{out_of_range_indices[:5]}"
            )
        if unmatched_rollouts:
            errors.append(
                f"{len(unmatched_rollouts)} rollout(s) have valid-range "
                f"_ng_task_index that didn't match any original record; "
                f"first few: {unmatched_rollouts[:5]}"
            )
        # NeMo-Gym writes one rollout per input record, so any deviation
        # implies stale cache or partial run.  Treat as fatal — the
        # downstream pipeline silently produces wrong data otherwise.
        if count != len(original_records):
            errors.append(
                f"rollout count ({count}) != original input count "
                f"({len(original_records)}); stale rollout cache or "
                f"input changed mid-run. See SKILL.md Gotcha #13."
            )
        if errors:
            raise RuntimeError(
                "extract_generations: alignment check failed.\n  - "
                + "\n  - ".join(errors)
                + f"\n\ninput_file={input_file}\noriginal_input_file={original_input_file}"
            )

    return count


# ============================================================================
# CLI
# ============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert between SDG JSONL and NeMo-Gym Responses API format"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rc = subparsers.add_parser(
        "render_and_convert",
        help="Render prompts and convert SDG JSONL to Responses API format",
    )
    rc.add_argument("--input_file", required=True)
    rc.add_argument("--output_file", required=True)
    rc.add_argument("--prompt_template", required=True)
    rc.add_argument(
        "--extra_record_fields",
        default=None,
        help="JSON string of static fields to merge into every record "
        '(e.g. \'{"verifier": {"type": "regex", "verify_regex": [".*"], '
        '"verify_min_matches": 0}}\').',
    )
    rc.add_argument(
        "--extra_record_field_mappers",
        default=None,
        help="JSON string mapping target field -> source field path "
        '(e.g. \'{"expected_answer": "answer"}\').',
    )
    rc.add_argument(
        "--system_role",
        default="system",
        help='Chat role for system-instruction messages (default "system", '
        'universal). Use "developer" only for gpt-oss/harmony models.',
    )

    ex = subparsers.add_parser(
        "extract_generations",
        help="Extract generations from rollout output back to SDG JSONL",
    )
    ex.add_argument("--input_file", required=True)
    ex.add_argument("--output_file", required=True)
    ex.add_argument("--generation_key", default="generation")
    ex.add_argument(
        "--original_input_file",
        default=None,
        help="Path to Responses-API input JSONL to join original SDG fields back.",
    )

    args = parser.parse_args()

    if args.command == "render_and_convert":
        extra_fields = json.loads(args.extra_record_fields) if args.extra_record_fields else None
        field_mappers = (
            json.loads(args.extra_record_field_mappers) if args.extra_record_field_mappers else None
        )
        n = render_and_convert(
            args.input_file,
            args.output_file,
            args.prompt_template,
            extra_record_fields=extra_fields,
            extra_record_field_mappers=field_mappers,
            system_role=args.system_role,
        )
        print(f"Converted {n} records -> {args.output_file}")
    elif args.command == "extract_generations":
        n = extract_generations(
            args.input_file,
            args.output_file,
            generation_key=args.generation_key,
            original_input_file=args.original_input_file,
        )
        print(f"Extracted {n} records -> {args.output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
