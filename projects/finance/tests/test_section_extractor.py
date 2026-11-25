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
"""Tests for section extraction over a corpus with mixed form types."""

from pathlib import Path

import pandas as pd
import pytest

# The extractors import sec-parser, which cannot be installed alongside this
# project's pandas floor. It ships only in the download container.
pytest.importorskip("sec_parser")

from nvfinance.utils.download import section_extractor  # noqa: E402
from nvfinance.utils.download.section_extractor import (  # noqa: E402
    extract_sections,
    get_extractor_class,
    is_section_supported,
)

# Every form type the S&P 500 config downloads.
ALL_FORMS = [
    "10-K",
    "10-K/A",
    "10-Q",
    "10-Q/A",
    "8-K",
    "8-K/A",
    "20-F",
    "6-K",
    "6-K/A",
    "DEF 14A",
    "DEFM14A",
    "S-1",
    "S-1/A",
    "S-4",
    "S-4/A",
    "FWP",
    "4",
    "4/A",
]


def _row(form_type: str, path: Path) -> dict:
    return {
        "accession_number": f"0000066740-26-{abs(hash(form_type)) % 1000000:06d}",
        "form_type": form_type,
        "report_date": "2026-06-30",
        "filing_date": "2026-07-24",
        "url": "https://www.sec.gov/Archives/edgar/data/66740/x/doc.htm",
        "size": 512,
        "file_type": "primary_document",
        "file_location": str(path),
        "ticker": "MMM",
        "company_name": "3M CO",
    }


def _corpus(tmp_path: Path, form_types: list[str]) -> dict:
    rows = []
    for index, form_type in enumerate(form_types):
        path = tmp_path / "data" / f"filing-{index}" / "primary-document.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<html><body><p>Item 1. Business</p></body></html>")
        rows.append(_row(form_type, path))
    metadata_path = tmp_path / "sec_metadata.parquet"
    pd.DataFrame(rows).to_parquet(metadata_path)
    return {"output_dir": str(tmp_path), "metadata_filename": "sec_metadata.parquet"}


class TestSupportedForms:
    @pytest.mark.parametrize("form_type", ALL_FORMS)
    def test_the_guard_agrees_with_the_extractor_lookup(self, form_type: str):
        if is_section_supported(form_type):
            assert get_extractor_class(form_type) is not None
        else:
            with pytest.raises(ValueError):
                get_extractor_class(form_type)

    def test_amendments_follow_their_base_form(self):
        assert is_section_supported("10-K/A")
        assert not is_section_supported("DEF 14A")


class TestExtractSections:
    def test_a_corpus_of_only_unsupported_forms_is_skipped(self, tmp_path: Path, caplog):
        """Before the guard, the first unsupported form aborted the whole stage,
        after the download had already completed."""
        config = _corpus(tmp_path, ["4", "DEF 14A", "FWP"])

        with caplog.at_level("INFO"):
            extract_sections(config)

        assert "Skipping 3 primary documents" in caplog.text
        metadata = pd.read_parquet(tmp_path / "sec_metadata.parquet")
        assert "section" not in set(metadata["file_type"])

    def test_supported_forms_are_still_processed_alongside_unsupported_ones(
        self, tmp_path: Path, caplog
    ):
        config = _corpus(tmp_path, ["10-K", "4"])

        with caplog.at_level("INFO"):
            extract_sections(config)

        assert "Skipping 1 primary documents" in caplog.text
        assert "Processing 1 new filings" in caplog.text

    def test_unsupported_forms_are_not_counted_as_already_extracted(self, tmp_path: Path, caplog):
        config = _corpus(tmp_path, ["4", "4/A"])

        with caplog.at_level("INFO"):
            extract_sections(config)

        assert "Already extracted sections for 0 filings" in caplog.text


class _OneSectionExtractor:
    def extract_sections(self, html_content: str) -> dict:
        return {"item_1": {"filename": "item_1", "html_content": html_content}}


class TestRelocatedCorpus:
    def test_sections_are_extracted_when_the_corpus_moved_after_download(
        self, tmp_path: Path, monkeypatch
    ):
        config = _corpus(tmp_path, ["10-K"])
        metadata_path = tmp_path / "sec_metadata.parquet"
        metadata = pd.read_parquet(metadata_path)
        metadata["file_location"] = metadata["file_location"].str.replace(
            str(tmp_path), "/workspace/outputs/finance/old-scope/step-0-download", regex=False
        )
        metadata.to_parquet(metadata_path)
        monkeypatch.setattr(
            section_extractor, "get_extractor_class", lambda form_type: _OneSectionExtractor
        )

        extract_sections(config)

        section_file = tmp_path / "data" / "filing-0" / "item_1.html"
        assert section_file.exists()
        sections = pd.read_parquet(metadata_path).query("file_type == 'section'")
        assert sections["file_location"].tolist() == [str(section_file)]

    def test_an_in_place_corpus_under_a_data_directory_is_extracted(
        self, tmp_path: Path, monkeypatch
    ):
        root = tmp_path / "data" / "step-0-download"
        config = _corpus(root, ["10-K"])
        monkeypatch.setattr(
            section_extractor, "get_extractor_class", lambda form_type: _OneSectionExtractor
        )

        extract_sections(config)

        assert (root / "data" / "filing-0" / "item_1.html").exists()
