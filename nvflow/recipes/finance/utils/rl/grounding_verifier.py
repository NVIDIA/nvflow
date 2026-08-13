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
"""Extract and evaluate finance rollout provenance as an atomic JSONL sidecar.

Evidence is limited to ``retrieve_information`` excerpts. Stable source IDs
require a canonical SEC CIK, accession, and document. Every input line produces
one deterministic output row, including malformed lines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from nvflow.grounding_verifier.evaluator import (
    ALGORITHM_VERSION,
    GroundingVerifierConfig,
    GroundingVerifierEvaluator,
)
from nvflow.grounding_verifier.types import EvidenceChunk
from nvflow.recipes.finance.utils.rl.finance_support import evaluate_finance_support

SCHEMA_VERSION = "1.2.0"
EVIDENCE_BASIS = "retrieval_model_excerpt"
LIMITATION = (
    "Rule-based attribution and protected-value checks supplement public models. "
    "Evidence is retrieved excerpts, not independent primary SEC filing verification."
)

SEC_SEARCH_NAMES = frozenset({"sec_filing_search", "edgar_search"})
PARSE_HTML_NAME = "parse_html_page"
RETRIEVE_INFO_NAME = "retrieve_information"
SUBMIT_FINAL_NAME = "submit_final_result"

STORAGE_KEY_RE = re.compile(r"\{\{([^{}]+)\}\}")

SEC_FILING_URL_RE = re.compile(
    r"https?://www\.sec\.gov/Archives/edgar/data/"
    r"(?P<cik>\d+)/(?P<accession>\d+)/"
    r"(?P<document>[^\s\"<]+)",
    re.IGNORECASE,
)

_FILING_FIELD_ALIASES = {
    "ticker": ("ticker", "symbol"),
    "company_name": ("company_name", "companyName", "name"),
    "form": ("form", "form_type", "formType"),
    "filing_date": ("filing_date", "filingDate", "filed_at"),
    "report_date": ("report_date", "reportDate", "period_of_report"),
    "cik": ("cik", "CIK", "cik_number"),
    "accession": (
        "accessionNo",
        "accession_no",
        "accession_number",
        "accession",
        "AccessionNumber",
    ),
    "document": ("primaryDocument", "document", "file_name", "filename"),
    "url": ("linkToHtml", "url", "filing_url", "link"),
}


def _render_value(value: object, unit: str) -> str:
    raw = str(value).replace(",", "")
    try:
        number = Decimal(raw)
    except InvalidOperation:
        return str(value)
    if unit.lower() == "usd":
        return f"${int(number):,}" if number == number.to_integral() else f"${number:,}"
    if unit.lower() == "percent":
        return f"{number}%"
    return str(value)


def canonicalize_finance_answer(answer: str | dict) -> str:
    """Preserve structured result values in the text evaluated by the guard."""
    parsed = answer
    if isinstance(answer, str):
        try:
            candidate = json.loads(answer)
        except (json.JSONDecodeError, TypeError):
            return answer
        if not isinstance(candidate, dict):
            return answer
        parsed = candidate

    explanation = str(parsed.get("explanation") or "").strip()
    value = parsed.get("value")
    parts = [explanation.rstrip(".")] if explanation else []
    if value is not None:
        parts.append(
            f"The submitted answer is {_render_value(value, str(parsed.get('unit') or ''))}"
        )
    evidence_ids = parsed.get("evidence_ids") or []
    if isinstance(evidence_ids, list) and evidence_ids:
        parts.append("Cited evidence: " + ", ".join(str(value) for value in evidence_ids))
    if not parts:
        return json.dumps(parsed, sort_keys=True)
    return ". ".join(parts) + "."


@dataclass(frozen=True)
class FilingMetadata:
    """Canonical SEC filing metadata."""

    cik: str | None = None
    accession: str | None = None
    document: str | None = None
    url: str | None = None
    ticker: str | None = None
    company_name: str | None = None
    form: str | None = None
    filing_date: str | None = None
    report_date: str | None = None
    identity_conflict: bool = False

    def source_id(self) -> str | None:
        """Return an ID only for a complete canonical filing."""
        if not self.identity_conflict and self.cik and self.accession and self.document:
            cik = self.cik.zfill(10)
            acc = str(self.accession)
            if len(cik) == 10 and cik.isdigit() and len(acc) == 18 and acc.isdigit():
                return f"sec:cik={cik}:accession={self.accession}:doc={self.document}"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cik": self.cik,
            "accession": self.accession,
            "document": self.document,
            "url": self.url,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "form": self.form,
            "filing_date": self.filing_date,
            "report_date": self.report_date,
            "identity_conflict": self.identity_conflict,
            "source_id": self.source_id(),
        }


@dataclass
class TraceExtraction:
    """Evidence and final answer extracted from one rollout."""

    answer: str | None = None
    evidence: list[EvidenceChunk] = field(default_factory=list)
    extraction_errors: list[str] = field(default_factory=list)
    submit_call_id: str | None = None
    has_submit: bool = False


@dataclass
class SidecarRow:
    """Parsed representation of one rollout line."""

    raw_line_fingerprint: str
    parse_error: str | None = None
    trace: TraceExtraction | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "raw_line_fingerprint": self.raw_line_fingerprint,
        }
        if self.parse_error:
            d["parse_error"] = self.parse_error
        if self.trace:
            d["answer"] = self.trace.answer
            d["has_submit"] = self.trace.has_submit
            d["submit_call_id"] = self.trace.submit_call_id
            d["evidence"] = [c.to_dict() for c in self.trace.evidence]
            d["extraction_errors"] = list(self.trace.extraction_errors)
        return d


@dataclass(frozen=True)
class FinanceEvaluatorConfig:
    """Finance-specific evaluator metadata."""

    grounding_config: GroundingVerifierConfig = field(default_factory=GroundingVerifierConfig)
    environment: str = "finance_sec_search"
    routing_model_revision: str | None = None
    nli_model_revision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "algorithm": ALGORITHM_VERSION,
            "evidence_basis": EVIDENCE_BASIS,
            "limitation": LIMITATION,
            "environment": self.environment,
            "thresholds": {
                "policy": "fixed_fail_closed",
                "evidence_excerpt_length": self.grounding_config.evidence_excerpt_length,
            },
            "models": {
                "routing_model": self.grounding_config.routing_model,
                "routing_model_revision": self.routing_model_revision,
                "nli_model": self.grounding_config.nli_model,
                "nli_model_revision": self.nli_model_revision,
            },
        }


def _raw_line_fingerprint(raw_line: str) -> str:
    """SHA-256 fingerprint (first 16 hex chars) of the raw JSONL line."""
    return hashlib.sha256(raw_line.encode("utf-8")).hexdigest()[:16]


def _iter_conversation_items(row: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield conversation items from a rollout row.

    Tries common locations in order:
      1. row["output"] (Responses API output items list)
      2. row["response"]["output"] (nested under response)
      3. row["messages"] (chat completions format)
      4. row["conversation"] (alternative trace field)
    """
    for source_key in ("output", "response"):
        source = row.get(source_key)
        if isinstance(source, dict):
            items = source.get("output")
            if isinstance(items, list):
                yield from (i for i in items if isinstance(i, dict))
                return
        if isinstance(source, list):
            yield from (i for i in source if isinstance(i, dict))
            return

    for key in ("messages", "conversation"):
        items = row.get(key)
        if isinstance(items, list):
            yield from (i for i in items if isinstance(i, dict))
            return


def _canonicalize_document(document: str | None) -> str | None:
    """Strip a document URL to its basename."""
    if not document:
        return None
    doc = str(document).strip()
    doc = doc.split("?")[0].split("#")[0]
    doc = os.path.basename(doc)
    return doc if doc else None


def _merge_filing(primary: FilingMetadata, fallback: FilingMetadata) -> FilingMetadata:
    return FilingMetadata(
        cik=primary.cik or fallback.cik,
        accession=primary.accession or fallback.accession,
        document=primary.document or fallback.document,
        url=primary.url or fallback.url,
        ticker=primary.ticker or fallback.ticker,
        company_name=primary.company_name or fallback.company_name,
        form=primary.form or fallback.form,
        filing_date=primary.filing_date or fallback.filing_date,
        report_date=primary.report_date or fallback.report_date,
        identity_conflict=primary.identity_conflict or fallback.identity_conflict,
    )


def _extract_filing_records(result: Any) -> list[FilingMetadata]:
    """Parse SEC records independently so fields never cross filing boundaries."""
    if isinstance(result, list):
        records = [value for value in result if isinstance(value, dict)]
    elif isinstance(result, dict):
        nested = result.get("filings")
        records = (
            [value for value in nested if isinstance(value, dict)]
            if isinstance(nested, list)
            else [result]
        )
    else:
        return []

    filings = []
    for record in records:
        values = {
            field: next((record[alias] for alias in aliases if record.get(alias)), None)
            for field, aliases in _FILING_FIELD_ALIASES.items()
        }
        url = values["url"]
        cik = str(values["cik"]).strip().zfill(10) if values["cik"] else None
        accession = _format_accession(str(values["accession"] or "").strip())
        document = _canonicalize_document(values["document"])
        url_metadata = _extract_filing_from_url(url) if url else FilingMetadata()
        identity_conflict = any(
            left and right and left != right
            for left, right in (
                (cik, url_metadata.cik),
                (accession, url_metadata.accession),
                (document, url_metadata.document),
            )
        )
        metadata = FilingMetadata(
            cik=cik,
            accession=accession,
            document=document,
            url=url,
            ticker=str(values["ticker"]).strip() if values["ticker"] else None,
            company_name=(str(values["company_name"]).strip() if values["company_name"] else None),
            form=str(values["form"]).strip() if values["form"] else None,
            filing_date=(str(values["filing_date"]).strip() if values["filing_date"] else None),
            report_date=(str(values["report_date"]).strip() if values["report_date"] else None),
            identity_conflict=identity_conflict,
        )
        filings.append(_merge_filing(metadata, url_metadata) if url else metadata)
    return filings


def _extract_filing_metadata(result: Any) -> FilingMetadata:
    filings = _extract_filing_records(result)
    return filings[0] if filings else FilingMetadata()


def _extract_filing_from_url(url: str) -> FilingMetadata:
    """Parse canonical filing fields from an SEC EDGAR URL."""
    match = SEC_FILING_URL_RE.search(url)
    if not match:
        return FilingMetadata(url=url)

    cik = match.group("cik").zfill(10)
    accession_raw = match.group("accession")
    document = _canonicalize_document(match.group("document"))

    accession = _format_accession(accession_raw)
    return FilingMetadata(cik=cik, accession=accession, document=document, url=url)


def _format_accession(accession_raw: str) -> str | None:
    """Return exactly 18 accession digits, otherwise ``None``."""
    digits = re.sub(r"\D", "", accession_raw)
    if len(digits) == 18:
        return digits
    return None


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    """Parse tool call arguments which may be a JSON string or dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def _parse_tool_result(raw: Any) -> Any:
    """Parse a tool result output which may be a JSON string or already parsed."""
    if isinstance(raw, dict | list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw
    return raw


def _try_parse_json(text: str) -> Any:
    """Attempt to parse a string as JSON; return None on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _decode_tool_envelope(raw: Any) -> tuple[str | None, str | None]:
    """Strictly decode a function_call_output envelope for pinned Gym.

    Returns ``(payload, error_reason)``.  On success, ``payload`` is the
    unwrapped content string and ``error_reason`` is ``None``.  On failure,
    ``payload`` is ``None`` and ``error_reason`` explains why.

    Accepts:
      - Pinned Gym envelope: ``{"results": <string>}`` with no top-level
        ``error`` key, a string payload that does not begin ``ERROR:`` and
        is not a nested JSON ``{"error": ...}`` payload.
      - Legacy producer shape: ``{"success": true, "result": <string>}``
        — accepted only when ``success is True`` plus a string ``result``.

    Rejects (returning ``(None, reason)``):
      - Top-level agent ``{"error": ...}`` (timeout / exception envelope).
      - Missing or unknown envelope keys.
      - Non-string ``results`` / ``result`` payloads.
      - ``results`` payload beginning ``ERROR:``.
      - Nested JSON ``{"error": ...}`` inside ``results`` (time-budget /
        no-company error payloads from the resource server).
      - Legacy ``success is False`` or missing ``success``.
      - All raw strings and other non-dict outputs (unstructured).
    """
    parsed = _parse_tool_result(raw)

    if not isinstance(parsed, dict):
        return None, "unstructured output"

    if "error" in parsed:
        return None, "agent error envelope"

    if "results" in parsed:
        results = parsed["results"]
        if not isinstance(results, str):
            return None, "non-string results payload"

        stripped = results.strip()
        if stripped.upper().startswith("ERROR:"):
            return None, "ERROR: payload"

        nested = _try_parse_json(results)
        if isinstance(nested, dict) and "error" in nested:
            return None, "nested JSON error payload"

        return results, None

    if "success" in parsed:
        if parsed["success"] is not True:
            return None, "legacy shape with success=False"
        result = parsed.get("result")
        if not isinstance(result, str):
            return None, "legacy shape with non-string result"
        return result, None

    return None, "unknown envelope"


def _is_parse_success(payload: str, key: str) -> bool:
    """Check whether a parse_html_page results payload indicates success.

    In pinned Gym, a successful ``parse_html_page`` returns a results
    string whose lines include the exact marker (the
    ``_save_tool_output`` line)::

        SUCCESS: The result has been saved to the data storage under the key: {key}.

    Failed parses return arbitrary ``str(e)`` text without the marker.
    This function checks for the exact marker on any line, bound to the
    expected ``key``, so that the ``WARNING:`` overwrite case (which
    still contains the exact line) is accepted while abbreviated,
    unrelated, or wrong-key ``SUCCESS:`` strings are rejected.
    """
    stripped = payload.strip()
    if not stripped:
        return False
    expected = f"SUCCESS: The result has been saved to the data storage under the key: {key}."
    for line in stripped.splitlines():
        if line.strip() == expected:
            return True
    return False


def _call_id(item: dict[str, Any]) -> str:
    return item.get("call_id") or item.get("id") or item.get("tool_call_id") or ""


def _tool_output(item: dict[str, Any]) -> Any:
    return item.get("output", item.get("result", ""))


def _has_filing_data(filing: FilingMetadata) -> bool:
    return bool(filing.cik or filing.accession or filing.url)


def _record_search_output(
    call_id: str,
    raw_output: Any,
    filings_by_call: dict[str, FilingMetadata],
    filings_by_url: dict[str, FilingMetadata],
) -> None:
    existing = filings_by_call.get(call_id, FilingMetadata())
    payload, _ = _decode_tool_envelope(raw_output)
    decoded = _try_parse_json(payload) if payload is not None else None
    filings = _extract_filing_records(decoded)
    if filings and _has_filing_data(filings[0]):
        filings_by_call[call_id] = _merge_filing(filings[0], existing)
    for filing in filings:
        if filing.url:
            filings_by_url[filing.url] = filing


def _record_parse_output(
    call_id: str,
    raw_output: Any,
    parse_calls: dict[str, tuple[str, str]],
    urls_by_key: dict[str, str],
) -> None:
    key, url = parse_calls[call_id]
    payload, _ = _decode_tool_envelope(raw_output)
    if payload is not None and _is_parse_success(payload, key):
        urls_by_key[key] = url


def _build_key_to_filing_map(
    items: list[dict[str, Any]],
) -> dict[str, FilingMetadata]:
    """Correlate successful parse storage keys with SEC filing metadata."""
    urls_by_key: dict[str, str] = {}
    filings_by_call: dict[str, FilingMetadata] = {}
    filings_by_url: dict[str, FilingMetadata] = {}
    search_calls: set[str] = set()
    parse_call_ids: dict[str, tuple[str, str]] = {}

    for item in items:
        item_type = item.get("type", "")
        name = item.get("name", "")
        call_id = _call_id(item)

        if item_type == "function_call" and name in SEC_SEARCH_NAMES:
            search_calls.add(call_id)
            filing = _extract_filing_metadata(_parse_tool_arguments(item.get("arguments")))
            if _has_filing_data(filing):
                filings_by_call[call_id] = filing
            continue

        if item_type == "function_call" and name == PARSE_HTML_NAME:
            args = _parse_tool_arguments(item.get("arguments"))
            key, url = args.get("key", ""), args.get("url", "")
            if key and url and call_id:
                parse_call_ids[call_id] = (key, url)
            continue

        if item_type not in ("function_call_output", "tool_result") or not call_id:
            continue
        if call_id in search_calls:
            _record_search_output(call_id, _tool_output(item), filings_by_call, filings_by_url)
        elif call_id in parse_call_ids:
            _record_parse_output(call_id, _tool_output(item), parse_call_ids, urls_by_key)

    key_to_filing: dict[str, FilingMetadata] = {}
    for key, url in urls_by_key.items():
        filing = filings_by_url.get(url)
        filing = (
            _merge_filing(filing, _extract_filing_from_url(filing.url))
            if filing and filing.url
            else _extract_filing_from_url(url)
        )
        key_to_filing[key] = filing

    return key_to_filing


def _extract_storage_keys_from_text(text: str) -> list[str]:
    """Extract ``{{key}}`` storage key references from text."""
    return [m.strip() for m in STORAGE_KEY_RE.findall(text)]


def _build_evidence_chunk(
    chunk_id: str,
    text: str,
    keys: list[str],
    key_to_filing: dict[str, FilingMetadata],
    tool_call_id: str | None,
    tool_result_id: str | None,
    char_range: tuple[int, int] | None = None,
) -> EvidenceChunk:
    """Build a chunk, allowing attribution only for one canonical source."""
    filings = [key_to_filing[key] for key in keys if key in key_to_filing]
    common = {
        "chunk_id": chunk_id,
        "text": text,
        "tool_call_id": tool_call_id,
        "tool_result_id": tool_result_id,
        "char_range": char_range,
    }

    if not filings:
        return EvidenceChunk(
            source_id=None,
            source_ids=(),
            attribution_state="unavailable",
            **common,
        )

    if len(keys) == 1 and len(filings) == 1:
        filing = filings[0]
        sid = filing.source_id()
        if sid:
            return EvidenceChunk(
                source_id=sid,
                source_ids=(),
                sec_url=filing.url,
                sec_cik=filing.cik,
                sec_accession=filing.accession,
                sec_document=filing.document,
                sec_ticker=filing.ticker,
                sec_company_name=filing.company_name,
                sec_form=filing.form,
                sec_filing_date=filing.filing_date,
                sec_report_date=filing.report_date,
                storage_keys=tuple(keys),
                attribution_state="available",
                **common,
            )
        return EvidenceChunk(
            source_id=None,
            source_ids=(),
            sec_url=filing.url,
            storage_keys=tuple(keys),
            attribution_state="unavailable",
            **common,
        )

    source_ids = tuple(f.source_id() for f in filings if f.source_id() is not None)
    return EvidenceChunk(
        source_id=None,
        source_ids=source_ids,
        sec_url=None,
        storage_keys=tuple(keys),
        attribution_state="unavailable",
        **common,
    )


def _collect_trace_items(
    items: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], str | None, str | None]:
    outputs: dict[str, Any] = {}
    retrievals: list[dict[str, Any]] = []
    submit_call_id = answer = None

    for item in items:
        item_type, name = item.get("type", ""), item.get("name", "")
        call_id = _call_id(item)
        if item_type in ("function_call_output", "tool_result"):
            if call_id and _tool_output(item) is not None:
                outputs[call_id] = _tool_output(item)
        elif item_type == "function_call" and name == RETRIEVE_INFO_NAME:
            retrievals.append(item)
        elif item_type == "function_call" and name == SUBMIT_FINAL_NAME:
            final_result = _parse_tool_arguments(item.get("arguments")).get("final_result", "")
            if isinstance(final_result, str) and final_result:
                submit_call_id, answer = call_id, final_result
    return outputs, retrievals, submit_call_id, answer


def extract_trace(row: dict[str, Any]) -> TraceExtraction:
    """Extract answer and evidence from a single rollout row."""
    items = list(_iter_conversation_items(row))
    extraction = TraceExtraction()
    key_to_filing = _build_key_to_filing_map(items)
    results, retrieve_calls, submit_call_id, answer = _collect_trace_items(items)

    if answer is not None:
        extraction.answer = answer
        extraction.submit_call_id = submit_call_id
        extraction.has_submit = True
    else:
        extraction.extraction_errors.append("No valid submit_final_result call found in trace")

    for idx, call in enumerate(retrieve_calls):
        call_id = _call_id(call) or f"retrieve_{idx}"
        result_raw = results.get(call_id, "")
        if not result_raw:
            extraction.extraction_errors.append(
                f"retrieve_information call {call_id}: no result output"
            )
            continue

        text, error_reason = _decode_tool_envelope(result_raw)
        if text is None:
            extraction.extraction_errors.append(
                f"retrieve_information call {call_id}: {error_reason}"
            )
            continue

        if not text.strip():
            extraction.extraction_errors.append(
                f"retrieve_information call {call_id}: empty result"
            )
            continue

        prompt = _parse_tool_arguments(call.get("arguments")).get("prompt", "")
        ranges = _parse_tool_arguments(call.get("arguments")).get("input_character_ranges")
        char_range = None
        if isinstance(ranges, list) and len(ranges) == 1 and isinstance(ranges[0], dict):
            start, end = ranges[0].get("start"), ranges[0].get("end")
            if isinstance(start, int) and isinstance(end, int):
                char_range = (start, end)
        chunk = _build_evidence_chunk(
            chunk_id=f"ev:{call_id}",
            text=text,
            keys=_extract_storage_keys_from_text(prompt),
            key_to_filing=key_to_filing,
            tool_call_id=call_id,
            tool_result_id=call_id,
            char_range=char_range,
        )
        extraction.evidence.append(chunk)

    return extraction


def parse_rollout_line(raw_line: str) -> SidecarRow:
    """Parse a single JSONL line and return a SidecarRow.

    Every valid or malformed JSONL line yields exactly one SidecarRow.
    """
    fingerprint = _raw_line_fingerprint(raw_line)
    stripped = raw_line.strip()
    if not stripped:
        return SidecarRow(
            raw_line_fingerprint=fingerprint,
            parse_error="empty line",
        )

    try:
        row = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return SidecarRow(
            raw_line_fingerprint=fingerprint,
            parse_error=f"JSONDecodeError: {exc!s}",
        )

    if not isinstance(row, dict):
        return SidecarRow(
            raw_line_fingerprint=fingerprint,
            parse_error=f"expected JSON object, got {type(row).__name__}",
        )

    trace = extract_trace(row)
    return SidecarRow(
        raw_line_fingerprint=fingerprint,
        trace=trace,
    )


def _response_id(row: dict[str, Any]) -> str | None:
    """Extract response ID from a rollout row."""
    for key in ("response_id", "id", "response_id_str"):
        val = row.get(key)
        if isinstance(val, str) and val:
            return val
    resp = row.get("response")
    if isinstance(resp, dict):
        rid = resp.get("id")
        if isinstance(rid, str) and rid:
            return rid
    return None


def _row_uuid(row: dict[str, Any]) -> str | None:
    """Extract UUID from a rollout row."""
    for key in ("uuid", "task_uuid", "sample_id"):
        val = row.get(key)
        if isinstance(val, str) and val:
            return val
    return None


def _row_indices(row: dict[str, Any]) -> tuple[int | None, int | None]:
    """Extract (_ng_task_index, _ng_rollout_index) from a rollout row."""
    task_idx = row.get("_ng_task_index")
    rollout_idx = row.get("_ng_rollout_index")
    return (
        int(task_idx) if isinstance(task_idx, int | float) else None,
        int(rollout_idx) if isinstance(rollout_idx, int | float) else None,
    )


def _row_question(row: dict[str, Any]) -> str | None:
    """Extract the user question from native Responses API inputs."""
    params = row.get("responses_create_params")
    if not isinstance(params, dict):
        return None
    inputs = params.get("input")
    if not isinstance(inputs, list):
        return None
    for item in reversed(inputs):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        content = item.get("content")
        if isinstance(content, str):
            marker = re.search(r"(?:^|\n)Question:\s*", content, re.IGNORECASE)
            return content[marker.end() :].strip() if marker else content.strip()
    return None


def _row_fingerprint(row: dict[str, Any]) -> str:
    """Deterministic fingerprint of the rollout row content."""
    raw = json.dumps(row, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _evaluation_uuid(
    fingerprint: str,
    seed: int,
    line_number: int,
    config: FinanceEvaluatorConfig,
) -> str:
    """Deterministic evaluation UUID from raw-line fingerprint + seed + line number + algorithm/config.

    Uses the raw-line fingerprint (SHA-256 of the raw JSONL line) rather
    than the parsed dict fingerprint, avoiding collisions on malformed
    lines that all parse to ``{}``.  The physical line number is included
    so duplicate identical rows still get distinguishable IDs by their
    position in the input file.  The config digest captures the
    algorithm version and policy settings so the same row evaluated under
    different configurations yields a different UUID.
    """
    config_digest = hashlib.sha256(
        json.dumps(config.to_dict(), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    raw = f"eval:{fingerprint}:{seed}:{line_number}:{ALGORITHM_VERSION}:{config_digest}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _unavailable(reason: str, errors: Sequence[str]) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": reason,
        "verdicts": [],
        "errors": list(errors),
    }


def _parse_failure(output: dict[str, Any], message: str, reason: str) -> dict[str, Any]:
    output.update(
        parse_error=message,
        verdict=_unavailable(reason, [message]),
        evidence=[],
        extraction_errors=[message],
    )
    return output


def evaluate_row(
    raw_line: str,
    row: dict[str, Any],
    seed: int,
    evaluator: GroundingVerifierEvaluator,
    config: FinanceEvaluatorConfig,
    line_number: int = 0,
) -> dict[str, Any]:
    """Evaluate a single rollout row and produce a complete sidecar dict."""
    sidecar = parse_rollout_line(raw_line)
    trace = sidecar.trace

    task_idx, rollout_idx = _row_indices(row)
    output: dict[str, Any] = {
        **config.to_dict(),
        "seed": seed,
        "line_number": line_number,
        "uuid": _row_uuid(row),
        "_ng_task_index": task_idx,
        "_ng_rollout_index": rollout_idx,
        "response_id": _response_id(row),
        "fingerprint": _row_fingerprint(row),
        "raw_line_fingerprint": sidecar.raw_line_fingerprint,
        "evaluation_uuid": _evaluation_uuid(
            sidecar.raw_line_fingerprint, seed, line_number, config
        ),
    }

    if sidecar.parse_error:
        return _parse_failure(output, sidecar.parse_error, "parse_error")

    if trace is None:
        return _parse_failure(output, "trace extraction returned None", "trace_extraction_failed")

    evidence = trace.evidence
    extraction_errors = list(trace.extraction_errors)
    question = _row_question(row)

    evaluated_answer = (
        canonicalize_finance_answer(trace.answer) if trace.answer is not None else None
    )
    output["answer"] = trace.answer
    output["evaluated_answer"] = evaluated_answer
    output["question"] = question
    output["has_submit"] = trace.has_submit
    output["submit_call_id"] = trace.submit_call_id
    output["evidence"] = [c.to_dict() for c in evidence]
    output["extraction_errors"] = extraction_errors

    if not trace.has_submit or trace.answer is None:
        decision = _unavailable("no_submit_final_result", extraction_errors)
    else:
        assert evaluated_answer is not None
        decision_obj = (
            evaluate_finance_support(
                trace.answer, question, evidence, evaluator.verify_against_premise
            )
            if question
            else None
        )
        if decision_obj is None:
            decision_obj = evaluator.evaluate(evaluated_answer, evidence)
        decision = decision_obj.to_dict()
        if extraction_errors:
            decision["status"] = "unavailable"
            decision["reason"] = "trace_extraction_errors"
            decision["errors"] = extraction_errors + list(decision.get("errors", []))

    output["verdict"] = decision
    return output


def evaluate_seed(
    input_path: str,
    output_path: str,
    seed: int,
    evaluator: GroundingVerifierEvaluator,
    config: FinanceEvaluatorConfig,
) -> int:
    """Evaluate one rollout file and atomically replace its sidecar."""
    done_path = output_path + ".done"

    input_done = input_path + ".done"
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if not os.path.isfile(input_done):
        raise RuntimeError(f"Input completion marker not found: {input_done}")

    if os.path.exists(done_path):
        os.remove(done_path)

    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir or ".", exist_ok=True)

    count = 0
    fd, tmp_path = tempfile.mkstemp(
        dir=out_dir or ".",
        prefix=".pg_tmp_",
        suffix=".jsonl",
    )
    try:
        with (
            os.fdopen(fd, "w", encoding="utf-8") as out_f,
            open(input_path, encoding="utf-8") as in_f,
        ):
            for raw_line in in_f:
                parsed = _try_parse_json(raw_line.strip())
                row = parsed if isinstance(parsed, dict) else {}
                result = evaluate_row(raw_line, row, seed, evaluator, config, count)
                out_f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
                count += 1

        os.replace(tmp_path, output_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    Path(done_path).touch()
    return count


def sidecar_filename(seed: int) -> str:
    """Return the sidecar output filename for a given seed."""
    return f"grounding-verifier-rs{seed}.jsonl"


def merged_filename(seed: int) -> str:
    """Return the merged rollout filename for a given seed."""
    return f"output-rs{seed}.jsonl"


def done_marker(seed: int) -> str:
    """Return the completion marker filename for a given seed."""
    return f"{merged_filename(seed)}.done"


def build_evaluator_from_config(
    config: FinanceEvaluatorConfig,
) -> GroundingVerifierEvaluator:
    """Build a GroundingVerifierEvaluator with lazy HF model collaborators.

    Model loading is deferred; this function constructs the objects but
    does not download or load any model weights.
    """
    from nvflow.grounding_verifier.decomposer import RuleBasedDecomposer
    from nvflow.grounding_verifier.embedder import HFEmbedder
    from nvflow.grounding_verifier.nli import HFNLI
    from nvflow.grounding_verifier.router import EmbeddingSourceRouter

    verifier_config = config.grounding_config
    embedder = HFEmbedder(
        model_id=verifier_config.routing_model,
        revision=config.routing_model_revision,
    )
    nli_scorer = HFNLI(
        model_id=verifier_config.nli_model,
        revision=config.nli_model_revision,
    )
    return GroundingVerifierEvaluator(
        decomposer=RuleBasedDecomposer(),
        router=EmbeddingSourceRouter(embedder),
        nli_scorer=nli_scorer,
        config=verifier_config,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for GroundingVerifier finance evaluation."""
    parser = argparse.ArgumentParser(
        description="Run GroundingVerifier evaluation on a finance rollout file."
    )
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--environment", default="finance_sec_search")
    parser.add_argument(
        "--routing_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--nli_model",
        default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    )
    parser.add_argument("--routing_model_revision", default=None)
    parser.add_argument("--nli_model_revision", default=None)
    parser.add_argument("--evidence_excerpt_length", type=int, default=500)

    args = parser.parse_args(argv)

    verifier_config = GroundingVerifierConfig(
        evidence_excerpt_length=args.evidence_excerpt_length,
        routing_model=args.routing_model,
        nli_model=args.nli_model,
    )
    finance_config = FinanceEvaluatorConfig(
        grounding_config=verifier_config,
        environment=args.environment,
        routing_model_revision=args.routing_model_revision,
        nli_model_revision=args.nli_model_revision,
    )

    evaluator = build_evaluator_from_config(finance_config)

    count = evaluate_seed(
        input_path=args.input_file,
        output_path=args.output_file,
        seed=args.seed,
        evaluator=evaluator,
        config=finance_config,
    )
    print(f"GroundingVerifier: {count} rows written to {args.output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
