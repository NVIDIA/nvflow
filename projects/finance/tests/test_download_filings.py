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
"""Tests for SEC filing download, focused on exhibit failure handling."""

import sys
import types
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import pytest

# edgartools only ships in the download container, and the module imports it at
# top level purely for type hints and set_identity.
_edgar = types.ModuleType("edgar")
_edgar.__dict__["Company"] = object
_edgar.__dict__["Filing"] = object
_edgar.__dict__["set_identity"] = lambda *args, **kwargs: None
sys.modules.setdefault("edgar", _edgar)

from nvfinance.utils.download import download_filings as module  # noqa: E402
from nvfinance.utils.download.parquet_io import (  # noqa: E402
    write_parquet_atomic,
)

ACCESSION = "0000066740-26-000246"


class FakeAttachment:
    def __init__(self, document_type: str, body: str, error: Exception | None = None):
        self.document_type = document_type
        self.document = f"{document_type.lower()}.htm"
        self.url = f"https://www.sec.gov/Archives/edgar/data/66740/x/{self.document}"
        self._body = body
        self._error = error
        self.attempts = 0

    def download(self) -> str:
        self.attempts += 1
        if self._error is not None:
            raise self._error
        return self._body


class TransientAttachment(FakeAttachment):
    """Fails until *succeed_on* attempts have been made."""

    def __init__(self, document_type: str, body: str, succeed_on: int):
        super().__init__(document_type, body)
        self._succeed_on = succeed_on

    def download(self) -> str:
        self.attempts += 1
        if self.attempts < self._succeed_on:
            raise httpx.ReadTimeout("timed out")
        return self._body


class FakeSubmission:
    def __init__(self, html: str | None):
        self._html = html

    def html(self) -> str | None:
        return self._html


class FakeFiling:
    def __init__(
        self,
        attachments: list[FakeAttachment],
        *,
        form: str = "10-Q",
        report_date: str = "2026-06-30",
        filing_date: str = "2026-07-24",
        submission_html: str | None = None,
    ):
        self.accession_no = ACCESSION
        self.form = form
        self.report_date = report_date
        self.filing_date = filing_date
        self.filing_url = "https://www.sec.gov/Archives/edgar/data/66740/x/mmm-20260630.htm"
        self.size = 2048
        self.company = "3M CO"
        self.attachments = attachments
        self._submission = FakeSubmission(submission_html)

    def sgml(self) -> FakeSubmission:
        return self._submission

    def html(self) -> str:
        return "<html><body>primary</body></html>"


class IndexPageRefusedFiling(FakeFiling):
    def html(self) -> str:
        raise AttributeError("'NoneType' object has no attribute 'download'")


INLINE_XBRL = "<?xml version='1.0' encoding='ASCII'?>\n<html xmlns=\"http://www.w3.org/1999/xhtml\"><body>10-Q</body></html>"


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch):
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)


def _exhibit_rows(rows: list[dict]) -> dict[str, str]:
    return {row["url"]: row["file_location"] for row in rows if row["file_type"] == "exhibit"}


class TestExhibitFailures:
    def test_a_failed_exhibit_does_not_get_its_neighbour_content(self, tmp_path: Path):
        """The bug this guards: a non-timeout error used to fall through to the
        write, storing the previous exhibit's HTML under this exhibit's name."""
        good = FakeAttachment("EX-10.1", "<html>credit agreement</html>")
        broken = FakeAttachment("EX-99.1", "unused", error=ValueError("malformed attachment"))
        filing = FakeFiling([good, broken])

        rows = module.download_filing(filing, str(tmp_path), "MMM")

        written = _exhibit_rows(rows)
        assert len(written) == 1
        assert "EX-10.1" in next(iter(written.values()))
        assert not (tmp_path / "MMM/10-Q/2026" / ACCESSION / "exhibits/EX-99.1.html").exists()

    def test_the_filing_is_still_recorded_when_an_exhibit_fails(self, tmp_path: Path):
        broken = FakeAttachment("EX-99.1", "unused", error=ValueError("malformed attachment"))
        filing = FakeFiling([broken])

        rows = module.download_filing(filing, str(tmp_path), "MMM")

        assert [row["file_type"] for row in rows] == ["primary_document"]
        assert (tmp_path / "MMM/10-Q/2026" / ACCESSION / "primary-document.html").exists()

    def test_a_timeout_is_retried_then_the_exhibit_is_dropped(self, tmp_path: Path):
        timing_out = FakeAttachment("EX-99.1", "unused", error=httpx.ReadTimeout("timed out"))

        rows = module.download_filing(FakeFiling([timing_out]), str(tmp_path), "MMM")

        assert timing_out.attempts == module._MAX_RETRIES
        assert _exhibit_rows(rows) == {}

    def test_an_exhibit_that_recovers_on_retry_is_written_and_recorded(self, tmp_path: Path):
        flaky = TransientAttachment("EX-99.1", "<html>recovered</html>", succeed_on=2)

        rows = module.download_filing(FakeFiling([flaky]), str(tmp_path), "MMM")

        written = _exhibit_rows(rows)
        assert len(written) == 1
        assert Path(next(iter(written.values()))).read_text() == "<html>recovered</html>"

    def test_a_non_timeout_error_is_not_retried(self, tmp_path: Path):
        broken = FakeAttachment("EX-99.1", "unused", error=ValueError("malformed attachment"))

        module.download_filing(FakeFiling([broken]), str(tmp_path), "MMM")

        assert broken.attempts == 1

    def test_exhibits_after_a_failure_are_still_downloaded(self, tmp_path: Path):
        broken = FakeAttachment("EX-99.1", "unused", error=httpx.ReadTimeout("timed out"))
        good = FakeAttachment("EX-10.1", "<html>credit agreement</html>")

        rows = module.download_filing(FakeFiling([broken, good]), str(tmp_path), "MMM")

        written = _exhibit_rows(rows)
        assert len(written) == 1
        assert Path(next(iter(written.values()))).read_text() == "<html>credit agreement</html>"


class TestPrimaryDocument:
    def test_inline_xbrl_is_read_from_the_submission_without_the_index_page(self, tmp_path: Path):
        filing = IndexPageRefusedFiling([], submission_html=INLINE_XBRL)

        rows = module.download_filing(filing, str(tmp_path), "MMM")

        primary = tmp_path / "MMM/10-Q/2026" / ACCESSION / "primary-document.html"
        assert [row["file_type"] for row in rows] == ["primary_document"]
        assert primary.read_text(encoding="utf-8") == INLINE_XBRL

    def test_plain_html_still_goes_through_edgartools(self):
        filing = FakeFiling([], submission_html="<html><body>raw</body></html>")

        assert module.primary_document_html(filing) == "<html><body>primary</body></html>"

    def test_an_xml_document_that_is_not_html_still_goes_through_edgartools(self):
        filing = FakeFiling(
            [], form="4", submission_html="<?xml version='1.0'?><ownershipDocument/>"
        )

        assert module.primary_document_html(filing) == "<html><body>primary</body></html>"


class TestFilingYear:
    """Which year a filing is filed under on disk."""

    def test_the_report_date_is_used_when_present(self):
        assert module.filing_year(FakeFiling([])) == 2026

    def test_the_filing_date_stands_in_for_an_empty_report_date(self):
        # DEF 14A, FWP and other forms covering no period report no period.
        proxy = FakeFiling([], form="DEF 14A", report_date="", filing_date="2023-05-24")

        assert module.filing_year(proxy) == 2023

    def test_a_date_object_is_accepted(self):
        filing = FakeFiling([], report_date=None, filing_date=date(2024, 2, 21))

        assert module.filing_year(filing) == 2024

    def test_a_filing_dated_neither_way_is_rejected(self):
        with pytest.raises(ValueError):
            module.filing_year(FakeFiling([], report_date="", filing_date=""))


class TestUnusableFiling:
    def test_one_bad_filing_does_not_end_the_run(self, tmp_path: Path, monkeypatch):
        """A crash here used to abandon every ticker still to be downloaded."""
        good = FakeFiling([], form="10-K")
        undated = FakeFiling([], form="FWP", report_date="", filing_date="")
        monkeypatch.setattr(module, "Company", lambda ticker: _FakeCompany([undated, good]))
        monkeypatch.setattr(module, "set_identity", lambda email: None)

        config = {
            "identity_info": {"email_address": "someone@example.com"},
            "output_dir": str(tmp_path),
            "data_dir": "data",
            "metadata_filename": "sec_metadata.parquet",
            "start_year": 2026,
            "end_year": 2026,
            "form_types": ["10-K", "FWP"],
            "tickers": ["MMM"],
        }
        frame = module.download_filings(config)

        assert frame["form_type"].tolist() == ["10-K"]


class _FakeCompany:
    def __init__(self, filings: list[FakeFiling]):
        self._filings = filings

    def get_filings(self, form, year):
        return self._filings


class TestAtomicParquet:
    def test_the_frame_round_trips(self, tmp_path: Path):
        path = tmp_path / "metadata.parquet"

        write_parquet_atomic(pd.DataFrame({"accession_number": [ACCESSION]}), str(path))

        assert pd.read_parquet(path)["accession_number"].tolist() == [ACCESSION]

    def test_a_failed_write_leaves_the_previous_parquet_readable(self, tmp_path: Path):
        path = tmp_path / "metadata.parquet"
        write_parquet_atomic(pd.DataFrame({"accession_number": [ACCESSION]}), str(path))

        class Failing:
            def to_parquet(self, target: str) -> None:
                Path(target).write_bytes(b"PAR1 truncated")
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            write_parquet_atomic(Failing(), str(path))

        assert pd.read_parquet(path)["accession_number"].tolist() == [ACCESSION]
        assert list(tmp_path.iterdir()) == [path]
