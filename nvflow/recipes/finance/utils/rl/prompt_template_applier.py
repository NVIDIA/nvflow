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
"""Apply a prompt template to SDG data and extract the expected answer.

Reads chunked JSONL from data_transformation, applies a YAML prompt template
to create a ``prompt`` field (merging instruction + context + question +
optional ``current_date``), and extracts the concise answer after a
configurable prefix (e.g. "Answer:") from the ``generation`` field.

Input schema (6-field data_transformation output)::

    {"uuid": "...", "problem": "...", "context": "...",
     "reasoning_content": "...", "generation": "...", "question_type": "..."}

Output schema (8 fields -- adds ``prompt`` and ``expected_answer``)::

    prompt          -> NEW: formatted prompt (instruction + context + question)
    expected_answer -> NEW: extracted answer (after prefix) or full generation if no prefix
    problem         -> unchanged (raw question)
    context         -> "" (absorbed into prompt)
    generation      -> unchanged (original full model output from SDG)

Dynamic ``current_date`` (GRPO only):
If ``--sec_metadata_parquet`` and ``--raw_sdg_source_dir`` are provided, the
template is also formatted with a per-record ``{current_date}`` resolved from
the SDG source filing's ``filing_date`` (from the parquet) plus a deterministic
jitter of ``jitter_min_days..jitter_max_days`` days seeded by ``record["uuid"]``.
Join key is ``problem`` (stable through data_transformation).  Falls back to
``--fallback_current_date`` on any lookup miss.

Usage::

    python -m nvflow.recipes.finance.utils.rl.prompt_template_applier \\
        <input_dir> <output_dir> --prompt_template <template.yaml> \\
        [--answer_prefix "Answer:"] \\
        [--sec_metadata_parquet <parquet_path> --raw_sdg_source_dir <dir> \\
         --raw_sdg_filename <filename>]
"""

import argparse
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from nvflow.utils import setup_logger
from nvflow.utils.jsonl import iter_jsonl, write_jsonl

logger = setup_logger(__name__)

# Long-form date format used for ``current_date`` rendered into prompts,
# matching vals-ai/finance-agent eval's "February 23, 2022" style.
# Note: %d is zero-padded ("April 07, 2025") which matches eval's strftime
# output; relying on %-d is non-portable across libc.
_LONG_DATE_FMT = "%B %d, %Y"


def load_prompt_template(template_path: str) -> dict:
    """Load prompt template and optional response parameters from a YAML file.

    Expects a YAML file with a ``user`` key containing a format string
    with a required ``{problem}`` placeholder.  ``{context}`` is
    optional -- SFT templates include it, while the GRPO agent template
    intentionally omits it (the policy discovers filings via tools
    rather than receiving pre-loaded context).

    Agent-style templates may also include ``tools``,
    ``parallel_tool_calls``, etc.  These are
    bundled into a ``response_params`` dict and passed through to the
    Responses API converter (stage 2).

    Returns a dict with ``user_template`` (str) and optionally
    ``response_params`` (dict).
    """
    with open(template_path) as f:
        config = yaml.safe_load(f)

    template = config.get("user")
    if not template:
        logger.error("Prompt template YAML must have a 'user' key: %s", template_path)
        sys.exit(1)

    if "{problem}" not in template:
        logger.warning(
            "Template is missing required {problem} placeholder: %s",
            template_path,
        )

    result: dict = {"user_template": template}

    response_params: dict = {}
    for key in ("tools", "parallel_tool_calls"):
        if key in config:
            response_params[key] = config[key]
    if response_params:
        result["response_params"] = response_params

    return result


class DateResolver:
    """Per-record ``current_date`` resolver driven by SEC metadata + raw SDG.

    Loads two maps once at construction:

    1. ``accession_to_filing_date``: from the SEC metadata parquet (output of
       workflow-2-download-sec).  Multiple parquet rows per accession
       (primary_document + exhibits) are deduped on first occurrence.
    2. ``problem_to_accession``: from the raw SDG file.  SDG ``file_path0``
       follows ``{ticker}/{form_type}/{year}/{accession}/{section}/{filename}``
       so the accession is the 4th path segment.  Only ``problem`` and
       ``file_path0`` are read to keep memory small; ``content0`` / ``context``
       are large and unused here.

    ``resolve(record)`` returns ``(current_date, "resolved" | "fallback")``
    where ``current_date`` is ``filing_date + random.Random(record.uuid)``
    jitter in ``[jitter_min_days, jitter_max_days]`` days (deterministic per
    record).  On any lookup miss, returns the fallback date tagged as
    ``"fallback"`` so callers can count hits/misses.
    """

    def __init__(
        self,
        *,
        parquet_path: str,
        raw_sdg_path: str,
        jitter_min_days: int,
        jitter_max_days: int,
        fallback_current_date: str,
        parquet_accession_column: str,
        parquet_filing_date_column: str,
    ) -> None:
        import pandas as pd

        df = pd.read_parquet(
            parquet_path,
            columns=[parquet_accession_column, parquet_filing_date_column],
        ).drop_duplicates(subset=[parquet_accession_column])
        accession_to_filing_date: dict[str, str] = {}
        for accession, filing_date in zip(
            df[parquet_accession_column], df[parquet_filing_date_column], strict=False
        ):
            if pd.isna(filing_date):
                continue
            accession_to_filing_date[str(accession)] = str(filing_date)[:10]

        problem_to_accession: dict[str, str] = {}
        # Malformed lines are silently skipped to match historical behaviour
        # -- this is a metadata-extraction pass, not the canonical SDG ingest.
        for obj in iter_jsonl(raw_sdg_path, on_error="skip"):
            problem = obj.get("problem")
            file_path0 = obj.get("file_path0", "")
            if not problem or not file_path0:
                continue
            parts = file_path0.split("/")
            if len(parts) < 4:
                continue
            problem_to_accession.setdefault(problem, parts[3])

        self._accession_to_filing_date = accession_to_filing_date
        self._problem_to_accession = problem_to_accession
        self._jitter_min_days = jitter_min_days
        self._jitter_max_days = jitter_max_days
        # Normalize fallback once: emit long-form (e.g. "April 07, 2025") so
        # success + fallback paths both match vals-ai/finance-agent eval's
        # date style.  If the operator supplied a non-ISO string, keep it
        # as-is rather than crashing.
        try:
            self._fallback = datetime.strptime(fallback_current_date, "%Y-%m-%d").strftime(
                _LONG_DATE_FMT
            )
        except ValueError:
            self._fallback = fallback_current_date

    @property
    def num_accessions(self) -> int:
        return len(self._accession_to_filing_date)

    @property
    def num_problems(self) -> int:
        return len(self._problem_to_accession)

    def resolve(self, record: dict) -> tuple[str, str]:
        problem = record.get("problem", "")
        accession = self._problem_to_accession.get(problem)
        if not accession:
            return self._fallback, "fallback"
        filing_date_str = self._accession_to_filing_date.get(accession)
        if not filing_date_str:
            return self._fallback, "fallback"
        try:
            filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d")
        except ValueError:
            return self._fallback, "fallback"
        seed = record.get("uuid") or problem
        delta = random.Random(seed).randint(self._jitter_min_days, self._jitter_max_days)
        return (filing_date + timedelta(days=delta)).strftime(_LONG_DATE_FMT), "resolved"


@dataclass
class ProcessFileStats:
    """Counters returned by :func:`process_file`."""

    processed: int = 0
    extracted: int = 0
    date_hits: int = 0
    date_fallbacks: int = 0
    errors: list[dict] = field(default_factory=list)


def apply_template(
    record: dict,
    template: str,
    response_params: dict | None = None,
    current_date: str | None = None,
) -> dict:
    """Apply the prompt template to a single record.

    Formats the template with ``context`` and ``problem`` from the record,
    stores the result in a new ``prompt`` field, keeps ``problem`` unchanged
    (raw question), and clears ``context``.

    When *response_params* is provided (e.g. tools, parallel_tool_calls from an
    agent-style template), it is attached as ``_response_params`` so that
    the downstream Responses API converter can merge it into
    ``responses_create_params``.

    When *current_date* is provided, it is also passed to ``template.format()``
    so templates that reference ``{current_date}`` can render a per-record
    "as of X" anchor.  Templates that do not reference ``{current_date}``
    ignore the extra kwarg transparently.
    """
    context = record.get("context", "")
    problem = record.get("problem", "")

    format_kwargs: dict[str, str] = {"context": context, "problem": problem}
    if current_date is not None:
        format_kwargs["current_date"] = current_date
    formatted_prompt = template.format(**format_kwargs)

    result = dict(record)
    result["prompt"] = formatted_prompt
    result["context"] = ""
    if current_date is not None:
        result["current_date"] = current_date
    if response_params:
        result["_response_params"] = response_params
    return result


def extract_answer(generation: str, prefix: str) -> str:
    """Extract the answer after the last occurrence of *prefix*.

    Case-insensitive.  Falls back to the full text when the prefix is
    absent or nothing follows it.
    """
    idx = generation.lower().rfind(prefix.lower())
    if idx >= 0:
        answer = generation[idx + len(prefix) :].strip()
        if answer:
            return answer
    return generation


def process_file(
    input_path: Path,
    output_path: Path,
    template: str,
    answer_prefix: str | None,
    response_params: dict | None = None,
    date_resolver: DateResolver | None = None,
) -> ProcessFileStats:
    """Process a single JSONL file.  ``date_*`` counters stay zero when
    ``date_resolver`` is ``None`` (dynamic-date resolution disabled)."""
    stats = ProcessFileStats()

    # ``yield_error`` keeps the historical behaviour of skipping malformed
    # JSON lines while logging + recording them in the per-file errors
    # stream.  Line numbering is preserved via ``enumerate`` so operators
    # can grep the source file directly using the line stamp in
    # ``errors.jsonl``.
    with write_jsonl(output_path) as fout:
        for line_num, (record, parse_exc, _raw_line) in enumerate(
            iter_jsonl(input_path, on_error="yield_error"), 1
        ):
            if parse_exc is not None:
                logger.warning(
                    "Skipping malformed JSON at %s:%d: %s", input_path, line_num, parse_exc
                )
                stats.errors.append(
                    {"file": str(input_path), "line": line_num, "error": str(parse_exc)}
                )
                continue

            current_date: str | None = None
            if date_resolver is not None:
                current_date, source = date_resolver.resolve(record)
                if source == "resolved":
                    stats.date_hits += 1
                else:
                    stats.date_fallbacks += 1

            result = apply_template(record, template, response_params, current_date=current_date)

            generation = result.get("generation", "")
            if answer_prefix:
                answer = extract_answer(generation, answer_prefix)
                result["expected_answer"] = answer
                if answer != generation:
                    stats.extracted += 1
            else:
                result["expected_answer"] = generation

            fout.write(result)
            stats.processed += 1

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply prompt template and extract expected answer"
    )
    parser.add_argument("input_dir", help="Directory with JSONL chunk files")
    parser.add_argument("output_dir", help="Output directory for processed chunks")
    parser.add_argument("--prompt_template", required=True, help="Path to prompt template YAML")
    parser.add_argument(
        "--answer_prefix",
        default=None,
        help='Prefix to extract answer after (e.g. "Answer:"). If not set, generation is kept as-is.',
    )
    # Dynamic current_date resolution (all optional; only fires when both
    # --sec_metadata_parquet and --raw_sdg_source_dir are provided).
    parser.add_argument(
        "--sec_metadata_parquet",
        default=None,
        help="Path to SEC metadata parquet (produced by workflow-2-download-sec).",
    )
    parser.add_argument(
        "--raw_sdg_source_dir",
        default=None,
        help="Directory containing the raw SDG file (used to build problem -> accession).",
    )
    parser.add_argument(
        "--raw_sdg_filename",
        default="final_result.jsonl",
        help="Filename inside --raw_sdg_source_dir (default: final_result.jsonl).",
    )
    parser.add_argument(
        "--jitter_min_days",
        type=int,
        default=1,
        help="Minimum days added to filing_date for current_date jitter (default: 1).",
    )
    parser.add_argument(
        "--jitter_max_days",
        type=int,
        default=60,
        help="Maximum days added to filing_date for current_date jitter (default: 60).",
    )
    parser.add_argument(
        "--fallback_current_date",
        default="2025-04-07",
        help=(
            "Date used when parquet lookup misses (default: 2025-04-07 matching eval). "
            "Accepts YYYY-MM-DD on input; internally normalized to eval's long-form "
            "'%%B %%d, %%Y' style (e.g. 'April 07, 2025') before rendering into prompts."
        ),
    )
    parser.add_argument(
        "--parquet_accession_column",
        default="accession_number",
        help="Column name for accession in the parquet (default: accession_number).",
    )
    parser.add_argument(
        "--parquet_filing_date_column",
        default="filing_date",
        help="Column name for filing date in the parquet (default: filing_date).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("APPLY PROMPT TEMPLATE")
    logger.info("=" * 70)
    logger.info("Input:    %s", input_dir)
    logger.info("Output:   %s", output_dir)
    logger.info("Template: %s", args.prompt_template)
    logger.info("Answer prefix: %s", args.answer_prefix or "(none -- keep full generation)")

    tmpl = load_prompt_template(args.prompt_template)
    template = tmpl["user_template"]
    response_params = tmpl.get("response_params")
    logger.info("Template loaded (%d chars)", len(template))
    if response_params:
        logger.info("Response params: %s", list(response_params.keys()))

    date_resolver: DateResolver | None = None
    if args.sec_metadata_parquet and args.raw_sdg_source_dir:
        raw_sdg_path = Path(args.raw_sdg_source_dir) / args.raw_sdg_filename
        logger.info("Dynamic current_date enabled:")
        logger.info("  Parquet:          %s", args.sec_metadata_parquet)
        logger.info("  Raw SDG:          %s", raw_sdg_path)
        logger.info("  Jitter range:     [%d, %d] days", args.jitter_min_days, args.jitter_max_days)
        logger.info("  Fallback date:    %s", args.fallback_current_date)
        logger.info("Building DateResolver (parquet + raw SDG scan)...")
        date_resolver = DateResolver(
            parquet_path=args.sec_metadata_parquet,
            raw_sdg_path=str(raw_sdg_path),
            jitter_min_days=args.jitter_min_days,
            jitter_max_days=args.jitter_max_days,
            fallback_current_date=args.fallback_current_date,
            parquet_accession_column=args.parquet_accession_column,
            parquet_filing_date_column=args.parquet_filing_date_column,
        )
        logger.info(
            "  %d accessions, %d unique problems",
            date_resolver.num_accessions,
            date_resolver.num_problems,
        )
    else:
        logger.info("Dynamic current_date: DISABLED (missing parquet or raw_sdg_source_dir)")

    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.error("No .jsonl files found in %s", input_dir)
        return 1

    logger.info("Found %d JSONL file(s)", len(jsonl_files))

    # Remove any stale per-chunk outputs from a previous run before writing
    # new ones.  Without this, a rerun where the upstream stage produced a
    # smaller set of chunks (e.g. data_transformation rerun with a smaller
    # ``--num_chunks``) would silently leave higher-index output chunks on
    # disk; the downstream ``responses_api_converter`` globs ``*.jsonl`` from
    # this directory and would mix stale records into the new dataset.
    # Mirrors the cleanup in ``dataset_transformer.py``.  ``errors.jsonl`` is
    # explicitly excluded from the sweep: it is conditionally written only
    # when this run produces errors, and unconditionally deleting it would
    # erase the prior run's audit trail on a no-error rerun -- operators
    # use ``errors.jsonl`` to triage flaky inputs across reruns.  Stale
    # ``errors.jsonl`` entries are harmless to ``responses_api_converter``,
    # which routes any row missing the ``prompt`` field into its own skipped
    # stream rather than the canonical output.
    expected_outputs = {fpath.name for fpath in jsonl_files}
    stale_outputs = sorted(
        p
        for p in output_dir.glob("*.jsonl")
        if p.name not in expected_outputs and p.name != "errors.jsonl"
    )
    for stale in stale_outputs:
        stale.unlink()
    if stale_outputs:
        logger.info(f"Removed {len(stale_outputs):,} stale chunk file(s) from previous run")

    total = ProcessFileStats()
    for fpath in jsonl_files:
        out_path = output_dir / fpath.name
        stats = process_file(
            fpath,
            out_path,
            template,
            args.answer_prefix,
            response_params,
            date_resolver=date_resolver,
        )
        total.processed += stats.processed
        total.extracted += stats.extracted
        total.date_hits += stats.date_hits
        total.date_fallbacks += stats.date_fallbacks
        total.errors.extend(stats.errors)
        logger.info("  %s: %d records processed", fpath.name, stats.processed)

    if total.errors:
        errors_path = output_dir / "errors.jsonl"
        with write_jsonl(errors_path) as ef:
            for err in total.errors:
                ef.write(err)
        logger.warning("Errors: %d -> %s", len(total.errors), errors_path)

    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("Total processed: %d", total.processed)
    if args.answer_prefix:
        pct = total.extracted / total.processed * 100 if total.processed else 0.0
        logger.info("Answer extracted: %d (%.1f%%)", total.extracted, pct)
        logger.info("Kept full text:   %d (%.1f%%)", total.processed - total.extracted, 100.0 - pct)
    if date_resolver is not None:
        total_dates = total.date_hits + total.date_fallbacks
        pct_hit = total.date_hits / total_dates * 100 if total_dates else 0.0
        logger.info(
            "current_date resolved: %d / %d (%.2f%%)", total.date_hits, total_dates, pct_hit
        )
        logger.info(
            "current_date fallback: %d / %d (%.2f%%)",
            total.date_fallbacks,
            total_dates,
            100.0 - pct_hit,
        )
    logger.info("Errors:           %d", len(total.errors))
    logger.info("Output:           %s", output_dir)
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
