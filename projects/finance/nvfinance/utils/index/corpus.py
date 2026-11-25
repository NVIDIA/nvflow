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
"""Stream indexable documents out of a downloaded SEC filing corpus.

Reads ``sec_metadata.parquet`` written by the download stage, resolves each
primary document and exhibit on disk, converts it to plain text and reports
everything it had to drop.  Section rows are skipped: they are derived from
primary documents that are indexed already.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import pyarrow.parquet as pq
from bs4 import BeautifulSoup, NavigableString

SEC_URL_RE = re.compile(
    r"^/Archives/edgar/data/(?P<cik>\d+)/(?P<accession>\d+)/(?P<filename>[^/]+)$",
    re.IGNORECASE,
)

REQUIRED_COLUMNS = frozenset(
    {
        "accession_number",
        "form_type",
        "report_date",
        "filing_date",
        "url",
        "file_type",
        "file_location",
        "ticker",
        "company_name",
    }
)

INDEXABLE_FILE_TYPES = frozenset({"primary_document", "exhibit"})

_MAX_EXCLUDED_EXAMPLES = 20


@dataclass(frozen=True)
class SecUrl:
    """Canonical identifiers encoded by an SEC Archives document URL."""

    cik: str
    accession: str
    filename: str

    @property
    def key(self) -> str:
        return f"{int(self.cik)}:{self.accession.replace('-', '')}:{self.filename.lower()}"


@dataclass(frozen=True)
class CorpusDocument:
    """One SEC primary document or exhibit that can be indexed."""

    doc_id: str
    accession_number: str
    cik: str
    ticker: str
    company_name: str
    form_type: str
    document_type: str
    description: str | None
    filing_date: str
    report_date: str
    file_type: str
    url: str
    canonical_url_key: str
    source_path: str
    body: str


@dataclass
class PreflightReport:
    """What the corpus scan indexed, and what it had to drop.

    Persisted into the index so the artifact can be audited without the corpus.
    ``missing_paths`` is the load-bearing field: a row whose file is absent is
    skipped rather than raising, so a truncated corpus otherwise yields a
    smaller index and a successful exit.
    """

    metadata_path: str
    corpus_root: str
    total_rows: int = 0
    candidate_rows: int = 0
    indexed_documents: int = 0
    already_indexed: int = 0
    excluded_sections: int = 0
    duplicate_urls: int = 0
    ambiguous_path_rows: int = 0
    missing_paths: int = 0
    unreadable_files: int = 0
    invalid_urls: int = 0
    invalid_dates: int = 0
    empty_documents: int = 0
    inferred_ciks: int = 0
    inferred_document_types: int = 0
    counts_by_file_type: dict[str, int] | None = None
    counts_by_ticker: dict[str, int] | None = None
    counts_by_form_type: dict[str, int] | None = None
    excluded_examples: list[dict[str, str]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CorpusStream:
    """Stream validated documents from a stable parquet snapshot."""

    def __init__(
        self,
        artifact_root: str | Path,
        *,
        known_url_keys: Iterable[str] = (),
        batch_size: int = 1_000,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.root = Path(artifact_root).resolve()
        self.metadata_path = self.root / "sec_metadata.parquet"
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Metadata parquet not found: {self.metadata_path}")
        self.batch_size = batch_size
        self.known_url_keys = set(known_url_keys)
        self.report = PreflightReport(
            metadata_path=str(self.metadata_path),
            corpus_root=str(self.root),
        )
        self._parquet_file = self._open_parquet()
        self._ambiguous_paths = self._scan_metadata()

    def _open_parquet(self) -> pq.ParquetFile:
        parquet = pq.ParquetFile(self.metadata_path)
        missing = sorted(REQUIRED_COLUMNS - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(f"Metadata parquet is missing columns: {', '.join(missing)}")
        return parquet

    def _rows(self) -> Iterable[dict[str, Any]]:
        for batch in self._parquet_file.iter_batches(batch_size=self.batch_size):
            yield from batch.to_pylist()

    def _scan_metadata(self) -> set[Path]:
        """Find local paths claimed by more than one SEC URL.

        Requires a full pass before indexing: a path is only knowably ambiguous
        once every row that resolves to it has been seen.
        """
        first_url_by_path: dict[Path, str] = {}
        ambiguous_paths: set[Path] = set()
        for row in self._rows():
            self.report.total_rows += 1
            if row["file_type"] not in INDEXABLE_FILE_TYPES:
                self.report.excluded_sections += 1
                continue
            self.report.candidate_rows += 1
            try:
                path = resolve_source_path(str(row["file_location"]), self.root).resolve()
            except ValueError:
                continue
            url = str(row["url"])
            first_url = first_url_by_path.setdefault(path, url)
            if first_url != url:
                ambiguous_paths.add(path)
        return ambiguous_paths

    def __iter__(self) -> Iterable[CorpusDocument]:
        examples: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        file_type_counts: Counter[str] = Counter()
        ticker_counts: Counter[str] = Counter()
        form_type_counts: Counter[str] = Counter()
        for row in self._rows():
            if row["file_type"] not in INDEXABLE_FILE_TYPES:
                continue
            try:
                path = resolve_source_path(str(row["file_location"]), self.root).resolve()
            except ValueError as exc:
                self.report.missing_paths += 1
                _record_example(examples, "unresolvable_path", row, str(exc))
                continue
            if path in self._ambiguous_paths:
                self.report.ambiguous_path_rows += 1
                _record_example(
                    examples,
                    "ambiguous_path",
                    row,
                    "Local path is shared by multiple SEC URLs",
                )
                continue
            if not path.is_file():
                self.report.missing_paths += 1
                _record_example(examples, "missing_path", row, str(path))
                continue
            try:
                parsed_url = parse_sec_url(str(row["url"]))
                self.report.inferred_ciks += 1
            except ValueError as exc:
                self.report.invalid_urls += 1
                _record_example(examples, "invalid_url", row, str(exc))
                continue
            if parsed_url.key in self.known_url_keys:
                self.report.already_indexed += 1
                continue
            if parsed_url.key in seen_urls:
                self.report.duplicate_urls += 1
                continue
            seen_urls.add(parsed_url.key)
            try:
                filing_date, report_date = _filing_dates(row)
            except (TypeError, ValueError) as exc:
                self.report.invalid_dates += 1
                _record_example(examples, "invalid_date", row, str(exc))
                continue
            try:
                html = path.read_text(encoding="utf-8", errors="replace")
                body = html_to_text(html)
            except OSError as exc:
                self.report.unreadable_files += 1
                _record_example(examples, "unreadable_file", row, str(exc))
                continue
            if not body:
                self.report.empty_documents += 1
                _record_example(examples, "empty_document", row, str(path))
                continue
            document_type = _document_type(row, path)
            self.report.inferred_document_types += 1
            document = CorpusDocument(
                doc_id=parsed_url.key,
                accession_number=str(row["accession_number"]),
                cik=parsed_url.cik,
                ticker=str(row["ticker"]),
                company_name=str(row["company_name"]),
                form_type=str(row["form_type"]),
                document_type=document_type,
                description=document_type,
                filing_date=filing_date,
                report_date=report_date,
                file_type=str(row["file_type"]),
                url=str(row["url"]),
                canonical_url_key=parsed_url.key,
                source_path=str(path),
                body=body,
            )
            self.report.indexed_documents += 1
            file_type_counts[document.file_type] += 1
            ticker_counts[document.ticker] += 1
            form_type_counts[document.form_type] += 1
            yield document
        self.report.counts_by_file_type = dict(sorted(file_type_counts.items()))
        self.report.counts_by_ticker = dict(sorted(ticker_counts.items()))
        self.report.counts_by_form_type = dict(sorted(form_type_counts.items()))
        self.report.excluded_examples = examples


def parse_sec_url(url: str) -> SecUrl:
    """Parse document identifiers from an SEC Archives URL."""

    parsed = urlsplit(url)
    if parsed.netloc.lower() not in {"www.sec.gov", "sec.gov"}:
        raise ValueError(f"Not an SEC URL: {url}")
    match = SEC_URL_RE.match(unquote(parsed.path))
    if not match:
        raise ValueError(f"Unsupported SEC Archives URL: {url}")
    return SecUrl(
        cik=str(int(match.group("cik"))),
        accession=match.group("accession").replace("-", ""),
        filename=match.group("filename"),
    )


def resolve_source_path(file_location: str, artifact_root: Path) -> Path:
    """Resolve container-written metadata paths against a local artifact root.

    ``file_location`` is absolute and container-relative, so only the fragment
    below ``data/`` is portable to wherever the corpus is mounted now. The last
    ``data/`` is the corpus's own: the root may itself sit under a ``data``
    directory, while nothing below it is named ``data``.
    """

    normalized = file_location.replace("\\", "/")
    marker = "/data/"
    if marker in normalized:
        relative = normalized.rsplit(marker, 1)[1]
        return artifact_root / "data" / relative
    raw = Path(file_location)
    if raw.exists():
        return raw
    raise ValueError(f"Cannot remap file location without /data/: {file_location}")


def _linearize_tables(soup: BeautifulSoup) -> None:
    for table in soup.find_all("table"):
        rows: list[str] = []
        for row in table.find_all("tr"):
            cells = [
                " ".join(cell.get_text(" ", strip=True).split())
                for cell in row.find_all(["th", "td"])
            ]
            cells = [cell for cell in cells if cell]
            if cells:
                rows.append(" | ".join(cells))
        replacement = "\n".join(rows) if rows else table.get_text(" ", strip=True)
        table.replace_with(NavigableString(f"\n{replacement}\n"))


def html_to_text(html: str) -> str:
    """Extract deterministic searchable text while retaining table boundaries.

    Every byte of this function feeds the FTS index. Changing it changes what
    the tokenizer sees, so an index built by a modified version no longer
    matches one built before it.
    """

    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "noscript", "template"]):
        element.decompose()
    _linearize_tables(soup)

    lines: list[str] = []
    for line in soup.get_text("\n").splitlines():
        normalized = " ".join(line.split())
        if normalized:
            lines.append(normalized)
    return "\n".join(lines)


def _iso_date(value: Any) -> str:
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value.isoformat()
    parsed = date.fromisoformat(str(value)[:10])
    return parsed.isoformat()


def _filing_dates(row: dict[str, Any]) -> tuple[str, str]:
    filing_date = _iso_date(row["filing_date"])
    report_value = row.get("report_date")
    report_date = (
        filing_date
        if report_value is None or str(report_value).strip() == ""
        else _iso_date(report_value)
    )
    return filing_date, report_date


def _document_type(row: dict[str, Any], path: Path) -> str:
    if row["file_type"] == "primary_document":
        return str(row["form_type"])
    return path.stem


def _record_example(
    examples: list[dict[str, str]], reason: str, row: dict[str, Any], detail: str
) -> None:
    if len(examples) >= _MAX_EXCLUDED_EXAMPLES:
        return
    examples.append(
        {
            "reason": reason,
            "url": str(row.get("url", "")),
            "file_location": str(row.get("file_location", "")),
            "detail": detail,
        }
    )
