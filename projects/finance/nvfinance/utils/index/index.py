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
"""Build the SQLite FTS5 index that backs NeMo-Gym's ``edgar_search``.

Write path only.  Queries are served by NeMo-Gym
(``resources_servers/sec_local_index/local_edgar_search.py`` from v0.7.0,
``resources_servers/finance_sec_search/local_edgar_search.py`` before), which
opens this file read-only; the schema below is the contract between the two.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from .corpus import CorpusDocument, CorpusStream, PreflightReport

SCHEMA_VERSION = "2"

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    doc_id TEXT NOT NULL UNIQUE,
    accession_number TEXT NOT NULL,
    cik TEXT NOT NULL,
    ticker TEXT NOT NULL,
    company_name TEXT NOT NULL,
    form_type TEXT NOT NULL,
    document_type TEXT NOT NULL,
    description TEXT,
    filing_date TEXT NOT NULL,
    report_date TEXT NOT NULL,
    file_type TEXT NOT NULL,
    url TEXT NOT NULL,
    canonical_url_key TEXT NOT NULL UNIQUE,
    source_path TEXT NOT NULL,
    body TEXT NOT NULL
);
CREATE INDEX documents_filters
    ON documents(cik, form_type, filing_date);
CREATE VIRTUAL TABLE documents_fts USING fts5(
    body,
    content='documents',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2',
    prefix='2 3 4'
);
CREATE TABLE index_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

REQUIRED_TABLES = frozenset({"documents", "documents_fts", "index_metadata"})

# The metadata companion: the same rows as ``documents`` minus the filing text,
# which NeMo-Gym attaches beside the index so that a search reads a candidate's
# form type and date without pulling its multi-kilobyte body. It must sit at the
# index path plus this suffix, which is where NeMo-Gym looks for it.
COMPANION_SUFFIX = ".metadata"

# NeMo-Gym's SIDECAR_SCHEMA_VERSION. Unrelated to SCHEMA_VERSION above, which
# versions the index itself.
COMPANION_SCHEMA_VERSION = 1

COMPANION_COLUMNS = (
    "id",
    "accession_number",
    "cik",
    "ticker",
    "company_name",
    "form_type",
    "document_type",
    "description",
    "filing_date",
    "url",
)

COMPANION_SCHEMA = """
CREATE TABLE documents_meta (
    id INTEGER PRIMARY KEY,
    accession_number TEXT NOT NULL,
    cik TEXT NOT NULL,
    ticker TEXT NOT NULL,
    company_name TEXT NOT NULL,
    form_type TEXT NOT NULL,
    document_type TEXT NOT NULL,
    description TEXT,
    filing_date TEXT NOT NULL,
    url TEXT NOT NULL
);
CREATE TABLE sidecar_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

COMPANION_INDEXES = (
    "CREATE INDEX documents_meta_cik ON documents_meta(cik, form_type, filing_date)",
    "CREATE INDEX documents_meta_form ON documents_meta(form_type, filing_date)",
    "CREATE INDEX documents_meta_date ON documents_meta(filing_date)",
)

_FINGERPRINT_SAMPLES = 64

_COMPANION_BATCH_SIZE = 20_000


def companion_path_for(index_path: str | Path) -> Path:
    return Path(f"{index_path}{COMPANION_SUFFIX}")


def fingerprint_index(connection: sqlite3.Connection) -> str:
    """Fingerprint an index's row identity, as NeMo-Gym recomputes it on startup.

    Must stay byte-identical to ``fingerprint_source_index`` in NeMo-Gym's
    ``local_edgar_search``: the companion joins to the index on rowid, so
    NeMo-Gym refuses to load a pair whose fingerprints disagree.
    """
    highest = connection.execute("SELECT MAX(id) FROM documents").fetchone()[0]
    if not highest:
        return "empty"
    stride = max(1, highest // _FINGERPRINT_SAMPLES)
    sampled = [1 + offset * stride for offset in range(_FINGERPRINT_SAMPLES)]
    placeholders = ",".join("?" for _ in sampled)
    rows = connection.execute(
        "SELECT id, accession_number, filing_date FROM documents "
        f"WHERE id IN ({placeholders}) ORDER BY id",
        sampled,
    ).fetchall()
    digest = hashlib.sha256()
    digest.update(f"{highest}\x1e".encode())
    for row in rows:
        digest.update("\x1f".join(str(value) for value in row).encode())
        digest.update(b"\x1e")
    return digest.hexdigest()


def write_metadata_companion(
    index_path: str | Path,
    companion_path: str | Path,
    *,
    batch_size: int = _COMPANION_BATCH_SIZE,
) -> int:
    """Project the companion out of a finished index and return its row count."""

    source_path = Path(index_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Search index not found: {source_path}")
    destination_path = Path(companion_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = Path(f"{destination_path}.partial")
    for stale in (partial_path, Path(f"{partial_path}-journal")):
        stale.unlink(missing_ok=True)

    try:
        total = _project_companion(source_path, partial_path, batch_size=batch_size)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise

    os.replace(partial_path, destination_path)
    return total


def _project_companion(source_path: Path, partial_path: Path, *, batch_size: int) -> int:
    source = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
    try:
        total = int(source.execute("SELECT count(*) FROM documents").fetchone()[0])
        destination = sqlite3.connect(partial_path)
        try:
            # Load-bearing, and only settable before the first table exists:
            # 8 KiB pages are half of why splitting these rows out measured
            # faster than reading them from the index.
            destination.execute("PRAGMA page_size = 8192")
            destination.execute("PRAGMA journal_mode = OFF")
            destination.execute("PRAGMA synchronous = OFF")
            destination.executescript(COMPANION_SCHEMA)

            projection = ", ".join(COMPANION_COLUMNS)
            placeholders = ", ".join("?" for _ in COMPANION_COLUMNS)
            cursor = source.execute(f"SELECT {projection} FROM documents ORDER BY id")
            copied = 0
            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                destination.executemany(
                    f"INSERT INTO documents_meta ({projection}) VALUES ({placeholders})",
                    rows,
                )
                destination.commit()
                copied += len(rows)
            if copied != total:
                raise ValueError(f"Copied {copied} rows but the index holds {total}")

            for statement in COMPANION_INDEXES:
                destination.execute(statement)
            destination.executemany(
                "INSERT OR REPLACE INTO sidecar_metadata (key, value) VALUES (?, ?)",
                sorted(
                    {
                        "schema_version": str(COMPANION_SCHEMA_VERSION),
                        "document_count": str(total),
                        "source_fingerprint": fingerprint_index(source),
                        "built_at_unix_seconds": str(int(time.time())),
                    }.items()
                ),
            )
            destination.commit()
            destination.execute("VACUUM")
            destination.commit()
        finally:
            destination.close()
    finally:
        source.close()
    return total


class SearchIndex:
    """Build a local SEC FTS5 index from a downloaded corpus."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @classmethod
    def build(
        cls,
        artifact_root: str | Path,
        index_path: str | Path,
        *,
        batch_size: int = 500,
        progress: Callable[[dict[str, int]], None] | None = None,
    ) -> tuple[SearchIndex, PreflightReport]:
        """Build from scratch, discarding any index already at *index_path*."""

        destination = Path(index_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        for path in (destination, Path(f"{destination}-wal"), Path(f"{destination}-shm")):
            if path.exists():
                path.unlink()
        return cls.update(
            artifact_root,
            destination,
            batch_size=batch_size,
            progress=progress,
        )

    @classmethod
    def update(
        cls,
        artifact_root: str | Path,
        index_path: str | Path,
        *,
        batch_size: int = 500,
        progress: Callable[[dict[str, int]], None] | None = None,
    ) -> tuple[SearchIndex, PreflightReport]:
        """Append newly downloaded documents to an index in bounded batches."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        destination = Path(index_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(destination)
        try:
            if not cls._has_schema(connection):
                connection.executescript(SCHEMA)
            known_url_keys = {
                row[0] for row in connection.execute("SELECT canonical_url_key FROM documents")
            }
            stream = CorpusStream(
                artifact_root,
                known_url_keys=known_url_keys,
                batch_size=batch_size,
            )
            pending: list[CorpusDocument] = []
            committed = 0
            for document in stream:
                pending.append(document)
                if len(pending) >= batch_size:
                    cls._insert_documents(connection, pending)
                    committed += len(pending)
                    pending.clear()
                    if progress:
                        progress(
                            {
                                "new_documents": committed,
                                "already_indexed": stream.report.already_indexed,
                            }
                        )
            if pending:
                cls._insert_documents(connection, pending)
                committed += len(pending)
                if progress:
                    progress(
                        {
                            "new_documents": committed,
                            "already_indexed": stream.report.already_indexed,
                        }
                    )
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "artifact_root": str(Path(artifact_root).resolve()),
                "preflight": json.dumps(stream.report.to_dict(), sort_keys=True),
                "last_update": date.today().isoformat(),
            }
            connection.executemany(
                "INSERT OR REPLACE INTO index_metadata(key, value) VALUES (?, ?)",
                metadata.items(),
            )
            connection.commit()
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            connection.close()
        return cls(destination), stream.report

    @staticmethod
    def _has_schema(connection: sqlite3.Connection) -> bool:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }
        if tables and not REQUIRED_TABLES.issubset(tables):
            raise ValueError("Index exists but does not contain the expected schema")
        return REQUIRED_TABLES.issubset(tables)

    @staticmethod
    def _insert_documents(
        connection: sqlite3.Connection,
        documents: Sequence[CorpusDocument],
    ) -> None:
        try:
            for document in documents:
                values = asdict(document)
                cursor = connection.execute(
                    """
                    INSERT INTO documents (
                        doc_id, accession_number, cik, ticker, company_name,
                        form_type, document_type, description, filing_date,
                        report_date, file_type, url, canonical_url_key,
                        source_path, body
                    ) VALUES (
                        :doc_id, :accession_number, :cik, :ticker, :company_name,
                        :form_type, :document_type, :description, :filing_date,
                        :report_date, :file_type, :url, :canonical_url_key,
                        :source_path, :body
                    )
                    """,
                    values,
                )
                connection.execute(
                    "INSERT INTO documents_fts(rowid, body) VALUES (?, ?)",
                    (cursor.lastrowid, document.body),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def _connect(self) -> sqlite3.Connection:
        if not self.path.is_file():
            raise FileNotFoundError(f"Search index not found: {self.path}")
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def document_count(self) -> int:
        connection = self._connect()
        try:
            return int(connection.execute("SELECT count(*) FROM documents").fetchone()[0])
        finally:
            connection.close()

    def counts_by_form_type(self) -> dict[str, int]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT form_type, count(*) AS documents FROM documents GROUP BY form_type"
            ).fetchall()
        finally:
            connection.close()
        return {str(row["form_type"]): int(row["documents"]) for row in rows}

    def preflight_report(self) -> dict[str, Any]:
        """Return the build-time report stored inside the index."""

        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT value FROM index_metadata WHERE key = 'preflight'"
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise ValueError("Index does not contain a preflight report")
        return json.loads(row["value"])
