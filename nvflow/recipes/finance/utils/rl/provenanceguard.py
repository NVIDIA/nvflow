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
"""Finance ProvenanceGuard sidecar: trace extraction, evaluation, and CLI.

This module lives in the finance recipe layer (not the generic
``nvflow.provenanceguard`` package).  It provides:

1. **Trace extraction** - parses NeMo-Gym rollout JSONL rows and extracts
   structured evidence chunks + the final answer from the tool-call trace.
2. **Deterministic sidecar evaluator** - runs the generic
   :class:`~nvflow.provenanceguard.evaluator.ProvenanceGuardEvaluator`
   on each row and produces a complete sidecar dict with full metadata.
3. **Atomic file writer** - writes the sidecar JSONL using an atomic
   temp-file + ``os.replace`` protocol with strict marker gating.
4. **CLI** - ``python3 -m nvflow.recipes.finance.utils.rl.provenanceguard``.

Extraction rules
~~~~~~~~~~~~~~~~

- **Answer**: last valid ``submit_final_result.final_result``.
- **Evidence**: one ``EvidenceChunk`` per ``retrieve_information`` result,
  paired by ``call_id``.
- **Source correlation**: ``sec_filing_search`` metadata + ``parse_html_page``
  storage keys.
- **Stable IDs**: only when 10-digit zero-padded CIK + accession + document
  are all present.  Never storage-key/URL fallback IDs.
- **Composite chunks**: multiple storage keys, any unknown key, or no
  canonical ID -> ``attribution_state="unavailable"``;
  ``source_ids`` contains only known canonical candidates.

Every physical input line (including blank/malformed/non-object) produces
exactly one sidecar row.  Output is deterministic: no random UUID.
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
from pathlib import Path
from typing import Any

from nvflow.provenanceguard.evaluator import (
    ALGORITHM_VERSION,
    ProvenanceGuardConfig,
    ProvenanceGuardEvaluator,
)
from nvflow.provenanceguard.types import EvidenceChunk

SCHEMA_VERSION = "1.0.0"
EVIDENCE_BASIS = "retrieval_model_excerpt"
LIMITATION = (
    "Uncalibrated open approximation of ProvenanceGuard. "
    "Evidence is model-retrieved trace excerpts, not primary SEC filing "
    "verification. Rule-based claim decomposition may over-merge or "
    "split mid-clause. No paper-faithful calibration."
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



@dataclass(frozen=True)
class FilingMetadata:
    """SEC filing metadata extracted from sec_filing_search or URL parsing."""

    cik: str | None = None
    accession: str | None = None
    document: str | None = None
    url: str | None = None

    def source_id(self) -> str | None:
        """Build stable source ID if CIK + accession + document are present.

        CIK must be exactly 10 digits (zero-padded).  Never falls back to
        storage keys or URLs for the canonical ID.
        """
        if self.cik and self.accession and self.document:
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
            "source_id": self.source_id(),
        }


@dataclass
class TraceExtraction:
    """Result of extracting evidence + answer from one rollout row."""

    answer: str | None = None
    evidence: list[EvidenceChunk] = field(default_factory=list)
    extraction_errors: list[str] = field(default_factory=list)
    submit_call_id: str | None = None
    has_submit: bool = False


@dataclass
class SidecarRow:
    """One output row for the ProvenanceGuard sidecar JSONL file."""

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
    """Configuration for the finance ProvenanceGuard evaluator."""

    provenance_config: ProvenanceGuardConfig = field(default_factory=ProvenanceGuardConfig)
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
                "evidence_excerpt_length": self.provenance_config.evidence_excerpt_length,
            },
            "models": {
                "routing_model": self.provenance_config.routing_model,
                "routing_model_revision": self.routing_model_revision,
                "nli_model": self.provenance_config.nli_model,
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
    """Normalize a document name: strip query/fragment, take basename.

    Ensures URL-derived and metadata-derived document names are identical:
    ``10-K.htm``, ``10-K.htm?output=1``, and ``10-K.htm#part1`` all
    yield ``10-K.htm``.
    """
    if not document:
        return None
    doc = str(document).strip()
    doc = doc.split("?")[0].split("#")[0]
    doc = os.path.basename(doc)
    return doc if doc else None


def _extract_filing_metadata(result: Any) -> FilingMetadata:
    """Extract CIK, accession, document, and URL from a sec_filing_search result."""
    if not isinstance(result, dict | list):
        return FilingMetadata()

    filings: list[dict[str, Any]] = []
    if isinstance(result, list):
        filings = [f for f in result if isinstance(f, dict)]
    elif isinstance(result, dict):
        if "filings" in result and isinstance(result["filings"], list):
            filings = [f for f in result["filings"] if isinstance(f, dict)]
        else:
            filings = [result]

    if not filings:
        return FilingMetadata()

    cik = None
    accession = None
    document = None
    url = None

    for filing in filings:
        if not isinstance(filing, dict):
            continue
        if cik is None:
            cik = filing.get("cik") or filing.get("CIK") or filing.get("cik_number")
        if accession is None:
            accession = (
                filing.get("accessionNo")
                or filing.get("accession_no")
                or filing.get("accession_number")
                or filing.get("accession")
                or filing.get("AccessionNumber")
            )
        if document is None:
            document = (
                filing.get("primaryDocument")
                or filing.get("document")
                or filing.get("file_name")
                or filing.get("filename")
            )
        if url is None:
            url = (
                filing.get("linkToHtml")
                or filing.get("url")
                or filing.get("filing_url")
                or filing.get("link")
            )
        if cik and accession and document and url:
            break

    if cik:
        cik = str(cik).strip()
        if len(cik) < 10:
            cik = cik.zfill(10)
    if accession:
        accession = _format_accession(str(accession).strip())
    if document:
        document = _canonicalize_document(document)

    if url:
        url_filing = _extract_filing_from_url(url)
        cik = cik or url_filing.cik
        accession = accession or url_filing.accession
        document = document or url_filing.document

    return FilingMetadata(cik=cik, accession=accession, document=document, url=url)


def _extract_filing_from_url(url: str) -> FilingMetadata:
    """Parse CIK, accession, and document from a SEC EDGAR Archives URL.

    The CIK in the URL path is zero-padded to 10 digits.  The document
    name is canonicalized (query/fragment stripped, basename taken) so
    that URL-derived and metadata-derived document names are identical.
    """
    match = SEC_FILING_URL_RE.search(url)
    if not match:
        return FilingMetadata(url=url)

    cik = match.group("cik").zfill(10)
    accession_raw = match.group("accession")
    document = _canonicalize_document(match.group("document"))

    accession = _format_accession(accession_raw)
    return FilingMetadata(cik=cik, accession=accession, document=document, url=url)


def _format_accession(accession_raw: str) -> str | None:
    """Strip dashes/non-digits and return the 18-digit canonical form.

    Returns ``None`` unless exactly 18 digits remain after stripping
    all non-digit characters.  Both shorter and longer values are
    rejected (overlong values are not truncated).
    """
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
    expected = (
        "SUCCESS: The result has been saved to the data storage "
        f"under the key: {key}."
    )
    for line in stripped.splitlines():
        if line.strip() == expected:
            return True
    return False


def _build_key_to_filing_map(
    items: list[dict[str, Any]],
) -> dict[str, FilingMetadata]:
    """Build a mapping from storage keys to filing metadata.

    Scans for ``sec_filing_search`` / ``edgar_search`` calls (extracts
    filing metadata from arguments and results) and ``parse_html_page``
    calls (reads the plain ``key`` argument, paired by ``call_id`` to
    its output).  The key-to-URL mapping is updated only after a strict
    successful pinned parse output (``SUCCESS:`` marker); failed retries
    preserve the last successful mapping.
    """
    key_to_url: dict[str, str] = {}
    call_id_to_filing: dict[str, FilingMetadata] = {}
    url_to_filing: dict[str, FilingMetadata] = {}

    sec_search_call_ids: set[str] = set()
    parse_call_ids: dict[str, tuple[str, str]] = {}

    for item in items:
        item_type = item.get("type", "")
        name = item.get("name", "")
        call_id = item.get("call_id", item.get("id", ""))

        if item_type == "function_call" and name in SEC_SEARCH_NAMES:
            sec_search_call_ids.add(call_id)
            args = _parse_tool_arguments(item.get("arguments"))
            filing = _extract_filing_metadata(args)
            if filing.cik or filing.accession or filing.url:
                call_id_to_filing[call_id] = filing
            continue

        if item_type == "function_call" and name == PARSE_HTML_NAME:
            args = _parse_tool_arguments(item.get("arguments"))
            url = args.get("url", "")
            key = args.get("key", "")
            if key and url and call_id:
                parse_call_ids[call_id] = (key, url)
            continue

        if item_type in ("function_call_output", "tool_result"):
            if not call_id:
                continue
            if call_id in sec_search_call_ids:
                existing = call_id_to_filing.get(call_id, FilingMetadata())
                raw_output = item.get("output", item.get("result", ""))
                payload, _error = _decode_tool_envelope(raw_output)
                if payload is not None:
                    decoded = _try_parse_json(payload)
                    if isinstance(decoded, (dict, list)):
                        enriched = _extract_filing_metadata(decoded)
                        if enriched.cik or enriched.accession or enriched.url:
                            filing = FilingMetadata(
                                cik=enriched.cik or existing.cik,
                                accession=enriched.accession or existing.accession,
                                document=enriched.document or existing.document,
                                url=enriched.url or existing.url,
                            )
                            call_id_to_filing[call_id] = filing
                if call_id in call_id_to_filing:
                    filing = call_id_to_filing[call_id]
                    if filing.url:
                        url_to_filing[filing.url] = filing
            elif call_id in parse_call_ids:
                key, url = parse_call_ids[call_id]
                raw_output = item.get("output", item.get("result", ""))
                payload, _error = _decode_tool_envelope(raw_output)
                if payload is not None and _is_parse_success(payload, key):
                    key_to_url[key] = url

    key_to_filing: dict[str, FilingMetadata] = {}
    for key, url in key_to_url.items():
        filing = url_to_filing.get(url)
        if not filing:
            filing = _extract_filing_from_url(url)
        elif filing.url:
            url_filing = _extract_filing_from_url(filing.url)
            filing = FilingMetadata(
                cik=filing.cik or url_filing.cik,
                accession=filing.accession or url_filing.accession,
                document=filing.document or url_filing.document,
                url=filing.url,
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
) -> EvidenceChunk:
    """Build an EvidenceChunk from a retrieve_information output.

    Attribution is ``available`` ONLY when the retrieval requested
    exactly one key AND that key maps to exactly one canonical
    CIK+accession+document source.  Any zero keys, multiple keys,
    unknown key, or noncanonical mapping yields ``source_id=None``,
    ``source_ids`` containing only known canonical IDs, and
    ``attribution_state="unavailable"``.
    """
    filings: list[FilingMetadata] = []
    for key in keys:
        filing = key_to_filing.get(key)
        if filing:
            filings.append(filing)

    if not filings:
        return EvidenceChunk(
            chunk_id=chunk_id,
            text=text,
            source_id=None,
            source_ids=(),
            tool_call_id=tool_call_id,
            tool_result_id=tool_result_id,
            attribution_state="unavailable",
        )

    if len(keys) == 1 and len(filings) == 1:
        filing = filings[0]
        sid = filing.source_id()
        if sid:
            return EvidenceChunk(
                chunk_id=chunk_id,
                text=text,
                source_id=sid,
                source_ids=(),
                sec_url=filing.url,
                sec_cik=filing.cik,
                sec_accession=filing.accession,
                sec_document=filing.document,
                storage_keys=tuple(keys),
                tool_call_id=tool_call_id,
                tool_result_id=tool_result_id,
                attribution_state="available",
            )
        return EvidenceChunk(
            chunk_id=chunk_id,
            text=text,
            source_id=None,
            source_ids=(),
            sec_url=filing.url,
            storage_keys=tuple(keys),
            tool_call_id=tool_call_id,
            tool_result_id=tool_result_id,
            attribution_state="unavailable",
        )

    source_ids = tuple(f.source_id() for f in filings if f.source_id() is not None)
    return EvidenceChunk(
        chunk_id=chunk_id,
        text=text,
        source_id=None,
        source_ids=source_ids,
        sec_url=None,
        storage_keys=tuple(keys),
        tool_call_id=tool_call_id,
        tool_result_id=tool_result_id,
        attribution_state="unavailable",
    )


def extract_trace(row: dict[str, Any]) -> TraceExtraction:
    """Extract answer and evidence from a single rollout row."""
    items = list(_iter_conversation_items(row))
    extraction = TraceExtraction()

    key_to_filing = _build_key_to_filing_map(items)

    call_id_to_result: dict[str, Any] = {}
    for item in items:
        item_type = item.get("type", "")
        if item_type in ("function_call_output", "tool_result"):
            call_id = item.get("call_id", item.get("tool_call_id", ""))
            output = item.get("output", item.get("result", ""))
            if call_id and output is not None:
                call_id_to_result[call_id] = output

    last_submit_call_id: str | None = None
    last_submit_result: str | None = None
    retrieve_calls: list[dict[str, Any]] = []

    for item in items:
        item_type = item.get("type", "")
        name = item.get("name", "")
        call_id = item.get("call_id", item.get("id", ""))

        if item_type == "function_call" and name == SUBMIT_FINAL_NAME:
            args = _parse_tool_arguments(item.get("arguments"))
            final_result = args.get("final_result", "")
            if final_result and isinstance(final_result, str):
                last_submit_call_id = call_id
                last_submit_result = final_result

        if item_type == "function_call" and name == RETRIEVE_INFO_NAME:
            retrieve_calls.append(item)

    if last_submit_result is not None:
        extraction.answer = last_submit_result
        extraction.submit_call_id = last_submit_call_id
        extraction.has_submit = True
    else:
        extraction.extraction_errors.append("No valid submit_final_result call found in trace")

    for idx, call in enumerate(retrieve_calls):
        call_id = call.get("call_id", call.get("id", f"retrieve_{idx}"))
        result_raw = call_id_to_result.get(call_id, "")
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

        args = _parse_tool_arguments(call.get("arguments"))
        prompt = args.get("prompt", "")
        keys = _extract_storage_keys_from_text(prompt)

        chunk_id = f"ev:{call_id}"
        chunk = _build_evidence_chunk(
            chunk_id=chunk_id,
            text=text,
            keys=keys,
            key_to_filing=key_to_filing,
            tool_call_id=call_id,
            tool_result_id=call_id,
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


def evaluate_row(
    raw_line: str,
    row: dict[str, Any],
    seed: int,
    evaluator: ProvenanceGuardEvaluator,
    config: FinanceEvaluatorConfig,
    line_number: int = 0,
) -> dict[str, Any]:
    """Evaluate a single rollout row and produce a complete sidecar dict."""
    sidecar = parse_rollout_line(raw_line)
    trace = sidecar.trace

    task_idx, rollout_idx = _row_indices(row)
    response_id = _response_id(row)
    row_uuid = _row_uuid(row)
    fingerprint = _row_fingerprint(row)

    output: dict[str, Any] = {
        **config.to_dict(),
        "seed": seed,
        "line_number": line_number,
        "uuid": row_uuid,
        "_ng_task_index": task_idx,
        "_ng_rollout_index": rollout_idx,
        "response_id": response_id,
        "fingerprint": fingerprint,
        "raw_line_fingerprint": sidecar.raw_line_fingerprint,
        "evaluation_uuid": _evaluation_uuid(
            sidecar.raw_line_fingerprint, seed, line_number, config
        ),
    }

    if sidecar.parse_error:
        output["parse_error"] = sidecar.parse_error
        output["verdict"] = {
            "status": "unavailable",
            "reason": "parse_error",
            "verdicts": [],
            "errors": [sidecar.parse_error],
        }
        output["evidence"] = []
        output["extraction_errors"] = [sidecar.parse_error]
        return output

    if trace is None:
        output["parse_error"] = "trace extraction returned None"
        output["verdict"] = {
            "status": "unavailable",
            "reason": "trace_extraction_failed",
            "verdicts": [],
            "errors": ["trace extraction returned None"],
        }
        output["evidence"] = []
        output["extraction_errors"] = ["trace extraction returned None"]
        return output

    evidence = trace.evidence
    extraction_errors = list(trace.extraction_errors)

    output["answer"] = trace.answer
    output["has_submit"] = trace.has_submit
    output["submit_call_id"] = trace.submit_call_id
    output["evidence"] = [c.to_dict() for c in evidence]
    output["extraction_errors"] = extraction_errors

    if not trace.has_submit or trace.answer is None:
        decision = {
            "status": "unavailable",
            "reason": "no_submit_final_result",
            "verdicts": [],
            "errors": extraction_errors,
        }
    else:
        decision_obj = evaluator.evaluate(trace.answer, evidence)
        decision = decision_obj.to_dict()
        if extraction_errors:
            decision = {
                "status": "unavailable",
                "reason": "trace_extraction_errors",
                "verdicts": decision.get("verdicts", []),
                "errors": list(extraction_errors) + list(decision.get("errors", [])),
            }

    output["verdict"] = decision
    return output


def evaluate_seed(
    input_path: str,
    output_path: str,
    seed: int,
    evaluator: ProvenanceGuardEvaluator,
    config: FinanceEvaluatorConfig,
) -> int:
    """Evaluate one seed rollout file and write the sidecar JSONL atomically.

    Protocol:
      1. Require BOTH input_path and its sibling .done marker to exist
         before touching any output state.  A missing input gate
         preserves prior output and output .done unchanged.
      2. After both input gates pass, remove any stale output .done
         marker before mkdir/temp/evaluation so a failed rerun cannot
         leave a stale completion marker.
      3. Write all rows to a temp file in the output file directory.
      4. os.replace temp -> final (atomic on same filesystem).
      5. Create a fresh empty .done marker only after successful replace.

    Returns the number of rows written.
    """
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
                stripped = raw_line.strip()
                if not stripped:
                    row: dict[str, Any] = {}
                else:
                    try:
                        row = json.loads(stripped)
                        if not isinstance(row, dict):
                            row = {}
                    except json.JSONDecodeError:
                        row = {}

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
    return f"provenanceguard-rs{seed}.jsonl"


def merged_filename(seed: int) -> str:
    """Return the merged rollout filename for a given seed."""
    return f"output-rs{seed}.jsonl"


def done_marker(seed: int) -> str:
    """Return the completion marker filename for a given seed."""
    return f"{merged_filename(seed)}.done"


def build_evaluator_from_config(
    config: FinanceEvaluatorConfig,
) -> ProvenanceGuardEvaluator:
    """Build a ProvenanceGuardEvaluator with lazy HF model collaborators.

    Model loading is deferred; this function constructs the objects but
    does not download or load any model weights.
    """
    from nvflow.provenanceguard.decomposer import RuleBasedDecomposer
    from nvflow.provenanceguard.embedder import HFEmbedder
    from nvflow.provenanceguard.nli import HFNLI
    from nvflow.provenanceguard.router import EmbeddingSourceRouter

    pg_config = config.provenance_config
    embedder = HFEmbedder(
        model_id=pg_config.routing_model,
        revision=config.routing_model_revision,
    )
    nli_scorer = HFNLI(
        model_id=pg_config.nli_model,
        revision=config.nli_model_revision,
    )
    return ProvenanceGuardEvaluator(
        decomposer=RuleBasedDecomposer(),
        router=EmbeddingSourceRouter(embedder),
        nli_scorer=nli_scorer,
        config=pg_config,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ProvenanceGuard finance evaluation."""
    parser = argparse.ArgumentParser(
        description="Run ProvenanceGuard evaluation on a finance rollout file."
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

    pg_config = ProvenanceGuardConfig(
        evidence_excerpt_length=args.evidence_excerpt_length,
        routing_model=args.routing_model,
        nli_model=args.nli_model,
    )
    finance_config = FinanceEvaluatorConfig(
        provenance_config=pg_config,
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
    print(f"ProvenanceGuard: {count} rows written to {args.output_file}")
    return 0


__all__ = [
    "EVIDENCE_BASIS",
    "LIMITATION",
    "SCHEMA_VERSION",
    "FinanceEvaluatorConfig",
    "build_evaluator_from_config",
    "done_marker",
    "evaluate_row",
    "evaluate_seed",
    "extract_trace",
    "merged_filename",
    "parse_rollout_line",
    "sidecar_filename",
]


if __name__ == "__main__":
    sys.exit(main())
