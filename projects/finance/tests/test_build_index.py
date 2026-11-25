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
"""Tests for the local SEC full-text search index builder."""

import argparse
import importlib.util
import os
import sqlite3
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from nvfinance.utils.index import build_index
from nvfinance.utils.index.corpus import html_to_text, resolve_source_path
from nvfinance.utils.index.index import (
    SearchIndex,
    companion_path_for,
    fingerprint_index,
)

# The download stage records absolute container paths, so the corpus is only
# readable after remapping the fragment below data/.
CONTAINER_ROOT = "/workspace/outputs/finance/demo/workflow-2-download-sec/step-0-download"

ACCESSION = "0000066740-26-000246"
FILING_DIR = f"MMM/10-Q/2026/{ACCESSION}"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/66740/000006674026000246"


def _row(url: str, relative_path: str, file_type: str, form_type: str = "10-Q") -> dict:
    return {
        "accession_number": ACCESSION,
        "form_type": form_type,
        "report_date": "2026-06-30",
        "filing_date": "2026-07-24",
        "url": url,
        "size": 1024,
        "file_type": file_type,
        "file_location": f"{CONTAINER_ROOT}/data/{relative_path}",
        "ticker": "MMM",
        "company_name": "3M CO",
    }


PRIMARY_ROW = _row(
    f"{ARCHIVE}/mmm-20260630.htm", f"{FILING_DIR}/primary-document.html", "primary_document"
)
EXHIBIT_ROW = _row(
    f"{ARCHIVE}/exhibit101-jvxcreditagreem.htm",
    f"{FILING_DIR}/exhibits/EX-10.1.html",
    "exhibit",
)


def _write_corpus(root: Path, rows: list[dict], present: list[str] | None = None) -> None:
    """Write a corpus whose parquet rows point at *present* files only."""
    for relative_path in present if present is not None else [r["file_location"] for r in rows]:
        relative = relative_path.split("/data/", 1)[-1]
        path = root / "data" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"<html><body><p>Body of {Path(relative).name}</p></body></html>")
    pq.write_table(pa.Table.from_pylist(rows), root / "sec_metadata.parquet")


def _build_args(root: Path, index_path: Path, **overrides) -> argparse.Namespace:
    return argparse.Namespace(
        artifact_root=str(root),
        index_path=str(index_path),
        batch_size=overrides.pop("batch_size", 10),
        incremental=overrides.pop("incremental", False),
        allow_missing_paths=overrides.pop("allow_missing_paths", False),
        **overrides,
    )


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "step-0-download"
    root.mkdir()
    _write_corpus(root, [PRIMARY_ROW, EXHIBIT_ROW])
    return root


class TestBuild:
    def test_primary_documents_and_exhibits_are_both_indexed(self, corpus: Path, tmp_path: Path):
        index_path = tmp_path / "step-1-index" / "demo_sec_fts.sqlite"

        assert build_index.build(_build_args(corpus, index_path)) == 0

        connection = sqlite3.connect(index_path)
        rows = connection.execute(
            "SELECT canonical_url_key, file_type, source_path, body FROM documents ORDER BY file_type"
        ).fetchall()
        connection.close()

        assert [row[1] for row in rows] == ["exhibit", "primary_document"]
        assert [row[0] for row in rows] == [
            "66740:000006674026000246:exhibit101-jvxcreditagreem.htm",
            "66740:000006674026000246:mmm-20260630.htm",
        ]
        # Rewritten to where the corpus actually is, not where it was written.
        assert all(row[2].startswith(str(corpus)) for row in rows)
        assert "Body of EX-10.1.html" in rows[0][3]

    def test_search_matches_indexed_text(self, corpus: Path, tmp_path: Path):
        index_path = tmp_path / "demo_sec_fts.sqlite"
        build_index.build(_build_args(corpus, index_path))

        connection = sqlite3.connect(index_path)
        matches = connection.execute(
            "SELECT count(*) FROM documents_fts WHERE documents_fts MATCH 'primary'"
        ).fetchone()[0]
        connection.close()

        assert matches == 1

    def test_extracted_sections_are_not_indexed(self, tmp_path: Path):
        root = tmp_path / "step-0-download"
        root.mkdir()
        section_row = _row(f"{ARCHIVE}/mmm-20260630.htm", f"{FILING_DIR}/1A.html", "section")
        _write_corpus(root, [PRIMARY_ROW, section_row])
        index_path = tmp_path / "demo_sec_fts.sqlite"

        build_index.build(_build_args(root, index_path))
        report = SearchIndex(index_path).preflight_report()

        assert report["excluded_sections"] == 1
        assert report["indexed_documents"] == 1

    def test_form_type_counts_are_recorded(self, tmp_path: Path):
        root = tmp_path / "step-0-download"
        root.mkdir()
        proxy_row = _row(
            f"{ARCHIVE}/proxy.htm", f"{FILING_DIR}/proxy.html", "primary_document", "DEF 14A"
        )
        _write_corpus(root, [PRIMARY_ROW, proxy_row])
        index_path = tmp_path / "demo_sec_fts.sqlite"

        build_index.build(_build_args(root, index_path))

        assert SearchIndex(index_path).counts_by_form_type() == {"10-Q": 1, "DEF 14A": 1}


class TestMissingPathGate:
    def test_index_is_not_published_when_a_file_is_absent(self, tmp_path: Path, caplog):
        root = tmp_path / "step-0-download"
        root.mkdir()
        _write_corpus(root, [PRIMARY_ROW, EXHIBIT_ROW], present=[PRIMARY_ROW["file_location"]])
        index_path = tmp_path / "demo_sec_fts.sqlite"

        assert build_index.build(_build_args(root, index_path)) == 1
        assert not index_path.exists()
        assert Path(f"{index_path}{build_index.BUILDING_SUFFIX}").exists()

    def test_the_gap_can_be_accepted_explicitly(self, tmp_path: Path):
        root = tmp_path / "step-0-download"
        root.mkdir()
        _write_corpus(root, [PRIMARY_ROW, EXHIBIT_ROW], present=[PRIMARY_ROW["file_location"]])
        index_path = tmp_path / "demo_sec_fts.sqlite"

        assert build_index.build(_build_args(root, index_path, allow_missing_paths=True)) == 0
        assert SearchIndex(index_path).document_count() == 1


class TestIncremental:
    def test_only_new_documents_are_added(self, corpus: Path, tmp_path: Path):
        index_path = tmp_path / "demo_sec_fts.sqlite"
        build_index.build(_build_args(corpus, index_path))

        proxy_row = _row(
            f"{ARCHIVE}/proxy.htm", f"{FILING_DIR}/proxy.html", "primary_document", "DEF 14A"
        )
        _write_corpus(corpus, [PRIMARY_ROW, EXHIBIT_ROW, proxy_row])

        assert build_index.build(_build_args(corpus, index_path, incremental=True)) == 0
        index = SearchIndex(index_path)
        assert index.document_count() == 3
        assert index.preflight_report()["already_indexed"] == 2

    def _interrupted_build(self, tmp_path: Path) -> tuple[Path, Path]:
        """Leave a .building file holding one of the corpus's two documents."""
        root = tmp_path / "step-0-download"
        root.mkdir()
        index_path = tmp_path / "demo_sec_fts.sqlite"
        _write_corpus(root, [PRIMARY_ROW, EXHIBIT_ROW], present=[PRIMARY_ROW["file_location"]])

        assert build_index.build(_build_args(root, index_path)) == 1
        assert Path(f"{index_path}{build_index.BUILDING_SUFFIX}").exists()

        _write_corpus(root, [PRIMARY_ROW, EXHIBIT_ROW])
        return root, index_path

    def test_an_interrupted_build_is_resumed_not_reparsed(self, tmp_path: Path):
        root, index_path = self._interrupted_build(tmp_path)

        assert build_index.build(_build_args(root, index_path, incremental=True)) == 0
        index = SearchIndex(index_path)
        assert index.document_count() == 2
        assert index.preflight_report()["already_indexed"] == 1

    def test_a_full_rebuild_ignores_an_interrupted_build(self, tmp_path: Path):
        root, index_path = self._interrupted_build(tmp_path)

        assert build_index.build(_build_args(root, index_path, incremental=False)) == 0
        assert SearchIndex(index_path).document_count() == 2
        assert SearchIndex(index_path).preflight_report()["already_indexed"] == 0

    def test_a_missing_seed_is_warned_about_and_built_from_scratch(
        self, corpus: Path, tmp_path: Path, monkeypatch
    ):
        index_path = tmp_path / "demo_sec_fts.sqlite"
        warnings: list[str] = []
        monkeypatch.setattr(
            build_index.logger, "warning", lambda message, *args: warnings.append(message % args)
        )

        assert build_index.build(_build_args(corpus, index_path, incremental=True)) == 0
        assert SearchIndex(index_path).document_count() == 2
        assert len(warnings) == 1
        assert str(index_path) in warnings[0]

    def test_a_resumed_build_does_not_warn(self, tmp_path: Path, monkeypatch):
        root, index_path = self._interrupted_build(tmp_path)
        warnings: list[str] = []
        monkeypatch.setattr(
            build_index.logger, "warning", lambda message, *args: warnings.append(message % args)
        )

        assert build_index.build(_build_args(root, index_path, incremental=True)) == 0
        assert warnings == []

    def test_a_full_rebuild_discards_the_previous_index(self, corpus: Path, tmp_path: Path):
        index_path = tmp_path / "demo_sec_fts.sqlite"
        build_index.build(_build_args(corpus, index_path))

        _write_corpus(corpus, [PRIMARY_ROW])

        assert build_index.build(_build_args(corpus, index_path)) == 0
        assert SearchIndex(index_path).document_count() == 1


class TestPublish:
    def test_a_stale_journal_beside_the_destination_is_removed(self, tmp_path: Path):
        source = tmp_path / "index.sqlite.building"
        destination = tmp_path / "index.sqlite"
        source.write_text("new")
        stale_wal = Path(f"{destination}-wal")
        stale_wal.write_text("journal of the index being replaced")

        build_index._publish(source, destination)

        assert destination.read_text() == "new"
        assert not stale_wal.exists()
        assert not source.exists()


class TestMetadataCompanion:
    """The companion NeMo-Gym attaches beside the index; it defines this shape."""

    def test_it_is_published_beside_the_index(self, corpus: Path, tmp_path: Path):
        index_path = tmp_path / "step-1-index" / "demo_sec_fts.sqlite"

        build_index.build(_build_args(corpus, index_path))

        assert companion_path_for(index_path).is_file()

    def test_it_holds_the_metadata_columns_without_the_body(self, corpus: Path, tmp_path: Path):
        index_path = tmp_path / "demo_sec_fts.sqlite"
        build_index.build(_build_args(corpus, index_path))

        connection = sqlite3.connect(companion_path_for(index_path))
        try:
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            columns = [row[1] for row in connection.execute("PRAGMA table_info(documents_meta)")]
            indexes = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
                )
            }
        finally:
            connection.close()

        assert tables == {"documents_meta", "sidecar_metadata"}
        assert columns == [
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
        ]
        assert indexes == {
            "documents_meta_cik",
            "documents_meta_form",
            "documents_meta_date",
        }

    def test_its_page_size_is_the_one_the_split_was_measured_with(
        self, corpus: Path, tmp_path: Path
    ):
        index_path = tmp_path / "demo_sec_fts.sqlite"
        build_index.build(_build_args(corpus, index_path))

        connection = sqlite3.connect(companion_path_for(index_path))
        try:
            assert connection.execute("PRAGMA page_size").fetchone()[0] == 8192
        finally:
            connection.close()

    def test_its_rows_mirror_the_index(self, corpus: Path, tmp_path: Path):
        index_path = tmp_path / "demo_sec_fts.sqlite"
        build_index.build(_build_args(corpus, index_path))

        connection = sqlite3.connect(companion_path_for(index_path))
        try:
            companion_rows = connection.execute(
                "SELECT id, accession_number, url FROM documents_meta ORDER BY id"
            ).fetchall()
        finally:
            connection.close()
        index = sqlite3.connect(index_path)
        try:
            index_rows = index.execute(
                "SELECT id, accession_number, url FROM documents ORDER BY id"
            ).fetchall()
        finally:
            index.close()

        assert companion_rows == index_rows

    def test_its_provenance_pairs_it_to_the_index(self, corpus: Path, tmp_path: Path):
        index_path = tmp_path / "demo_sec_fts.sqlite"
        build_index.build(_build_args(corpus, index_path))

        connection = sqlite3.connect(companion_path_for(index_path))
        try:
            recorded = dict(connection.execute("SELECT key, value FROM sidecar_metadata"))
        finally:
            connection.close()
        index = sqlite3.connect(index_path)
        try:
            expected_fingerprint = fingerprint_index(index)
        finally:
            index.close()

        assert recorded["schema_version"] == "1"
        assert recorded["document_count"] == "2"
        assert recorded["source_fingerprint"] == expected_fingerprint
        assert int(recorded["built_at_unix_seconds"]) > 0

    def test_neither_file_is_published_when_the_gate_fails(self, tmp_path: Path):
        root = tmp_path / "step-0-download"
        root.mkdir()
        _write_corpus(root, [PRIMARY_ROW, EXHIBIT_ROW], present=[PRIMARY_ROW["file_location"]])
        index_path = tmp_path / "demo_sec_fts.sqlite"

        assert build_index.build(_build_args(root, index_path)) == 1

        assert not index_path.exists()
        assert not companion_path_for(index_path).exists()

    def test_a_rebuild_replaces_the_companion(self, corpus: Path, tmp_path: Path):
        index_path = tmp_path / "demo_sec_fts.sqlite"
        build_index.build(_build_args(corpus, index_path))
        _write_corpus(corpus, [PRIMARY_ROW])

        build_index.build(_build_args(corpus, index_path))

        connection = sqlite3.connect(companion_path_for(index_path))
        try:
            recorded = dict(connection.execute("SELECT key, value FROM sidecar_metadata"))
            rows = connection.execute("SELECT count(*) FROM documents_meta").fetchone()[0]
        finally:
            connection.close()

        assert rows == 1
        assert recorded["document_count"] == "1"


def _gym_local_edgar_search():
    """Load NeMo-Gym's search backend, or skip.

    Loaded from a checkout rather than imported so the test runs against the Gym
    the artifacts will actually be consumed by. Point NEMO_GYM_PATH at one, or
    run inside the nemo-gym container where /opt/nemo-gym exists. Gym 0.7.0 keeps
    the backend in sec_local_index; the Gym that NeMo-RL vendors keeps it in
    finance_sec_search.
    """
    root = Path(os.environ.get("NEMO_GYM_PATH", "/opt/nemo-gym"))
    candidates = (
        root / "resources_servers/sec_local_index/local_edgar_search.py",
        root / "resources_servers/finance_sec_search/local_edgar_search.py",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        pytest.skip(f"NeMo-Gym not available at {root}")
    name = "gym_local_edgar_search"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: the module's dataclasses resolve their
    # annotations by looking themselves up in sys.modules.
    sys.modules[name] = module
    # sec_local_index imports its own package by absolute name.
    sys.path.insert(0, str(root))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(root))
    return module


class TestGymConformance:
    """An nvflow-built pair must satisfy the consumer that reads it.

    Gym recomputes the fingerprint and row count from the index on startup and
    refuses a companion that disagrees, so this is what keeps the two writers
    from drifting.
    """

    def test_gym_loads_the_pair_and_searches_through_the_companion(
        self, corpus: Path, tmp_path: Path
    ):
        gym = _gym_local_edgar_search()
        index_path = tmp_path / "demo_sec_fts.sqlite"
        assert build_index.build(_build_args(corpus, index_path)) == 0

        search = gym.LocalEdgarSearch(index_path, max_end_date="2026-12-31")
        try:
            assert search.uses_metadata_sidecar
            results = search.search("primary", end_date="2026-12-31")
        finally:
            search.close()

        assert [row["accessionNo"] for row in results] == [ACCESSION]
        assert results[0]["formType"] == "10-Q"

    def test_gym_rejects_a_companion_built_from_a_different_index(
        self, corpus: Path, tmp_path: Path
    ):
        gym = _gym_local_edgar_search()
        index_path = tmp_path / "demo_sec_fts.sqlite"
        build_index.build(_build_args(corpus, index_path))
        stale_companion = companion_path_for(index_path).read_bytes()

        _write_corpus(corpus, [PRIMARY_ROW])
        build_index.build(_build_args(corpus, index_path))
        companion_path_for(index_path).write_bytes(stale_companion)

        with pytest.raises(ValueError):
            gym.LocalEdgarSearch(index_path, max_end_date="2026-12-31")


class TestHtmlToText:
    def test_tables_are_linearized_and_scripts_dropped(self):
        html = """
        <html><body>
          <script>ignored()</script>
          <p>Revenue   summary</p>
          <table><tr><th>Year</th><th>Sales</th></tr><tr><td>2026</td><td>1,234</td></tr></table>
        </body></html>
        """

        assert html_to_text(html) == "Revenue summary\nYear | Sales\n2026 | 1,234"


class TestResolveSourcePath:
    @pytest.mark.parametrize(
        "recorded_root, current_root",
        [
            pytest.param(CONTAINER_ROOT, CONTAINER_ROOT, id="in-place"),
            pytest.param(CONTAINER_ROOT, "/mnt/shared/sp500/step-0-download", id="moved"),
            pytest.param(
                "/data/sec/step-0-download", "/data/sec/step-0-download", id="root-under-data"
            ),
            pytest.param(CONTAINER_ROOT, "/data/sec/step-0-download", id="moved-under-data"),
        ],
    )
    def test_the_corpus_relative_part_is_rejoined_onto_the_current_root(
        self, recorded_root: str, current_root: str
    ):
        relative = f"{FILING_DIR}/exhibits/EX-10.1.html"

        resolved = resolve_source_path(f"{recorded_root}/data/{relative}", Path(current_root))

        assert resolved == Path(current_root) / "data" / relative

    def test_an_in_place_corpus_under_a_data_directory_is_indexed(self, tmp_path: Path):
        root = tmp_path / "data" / "step-0-download"
        rows = []
        for row in (PRIMARY_ROW, EXHIBIT_ROW):
            relative = row["file_location"].split("/data/", 1)[1]
            path = root / "data" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"<html><body><p>Body of {path.name}</p></body></html>")
            rows.append({**row, "file_location": str(path)})
        pq.write_table(pa.Table.from_pylist(rows), root / "sec_metadata.parquet")
        index_path = tmp_path / "demo_sec_fts.sqlite"

        assert build_index.build(_build_args(root, index_path)) == 0
        assert SearchIndex(index_path).document_count() == 2
