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
"""Tests for scheduled Trivy remediation-issue synchronization."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any

import pytest

from scripts.update_trivy_issue import (
    ISSUE_TITLE,
    IssueUpdateError,
    _managed_block,
    _new_description,
    sync_issue,
)

NOW = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)
ENVIRONMENT = {
    "CI_COMMIT_REF_NAME": "dev",
    "CI_COMMIT_SHA": "abc123",
    "CI_PIPELINE_URL": "https://gitlab.example/pipelines/1",
    "TRIVY_SCAN_JOB_URL": "https://gitlab.example/jobs/2",
}


def _finding(severity: str = "HIGH") -> dict[str, Any]:
    return {
        "severity": severity,
        "vulnerability_id": f"CVE-2026-{severity}",
        "package_name": "example",
        "installed_version": "1.0.0",
        "fixed_versions": ["1.0.1"],
        "target": "uv.lock",
    }


def _report(findings: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema_version": 1, "policy": "report", "findings": findings}


class FakeAPI:
    def __init__(self, issues: list[dict[str, Any]] | None = None) -> None:
        self.issues = issues or []
        self.calls: list[tuple[str, str, Any, Any]] = []

    def request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, str] | None = None,
        data: Mapping[str, str] | None = None,
    ) -> Any:
        self.calls.append((method, path, query, data))
        if method == "GET":
            return self.issues
        if method == "POST":
            return {"iid": 10, "web_url": "https://gitlab.example/issues/10"}
        return {"iid": 9, "web_url": "https://gitlab.example/issues/9"}


def test_findings_create_confidential_issue_with_sla() -> None:
    api = FakeAPI()
    status, message = sync_issue(api, _report([_finding()]), ENVIRONMENT, now=NOW)

    assert status == 1
    assert "Created" in message
    _, _, _, data = api.calls[-1]
    assert data["title"] == ISSUE_TITLE
    assert data["confidential"] == "true"
    assert data["due_date"] == "2026-09-08"
    assert "**SLA due:** 2026-09-08" in data["description"]
    assert "automated-trivy" in data["labels"]
    assert "CVE-2026-HIGH" in data["description"]


def test_critical_finding_uses_shorter_sla() -> None:
    api = FakeAPI()
    sync_issue(api, _report([_finding("CRITICAL")]), ENVIRONMENT, now=NOW)

    assert api.calls[-1][3]["due_date"] == "2026-08-28"


def test_new_critical_shortens_due_date_and_preserves_maintainer_notes() -> None:
    initial = _new_description(
        _managed_block(_report([_finding()]), ENVIRONMENT, NOW, date(2026, 9, 8))
    )
    description = initial + "\n\nOwner: Alice\nRemediation MR: !123\n"
    api = FakeAPI(
        [
            {
                "iid": 9,
                "title": ISSUE_TITLE,
                "description": description,
                "due_date": "2026-09-08",
            }
        ]
    )
    status, message = sync_issue(
        api,
        _report([_finding(), _finding("CRITICAL")]),
        ENVIRONMENT,
        now=NOW,
    )

    assert status == 1
    assert "Updated" in message
    data = api.calls[-1][3]
    assert data["due_date"] == "2026-08-28"
    assert "**SLA due:** 2026-08-28" in data["description"]
    assert "Owner: Alice" in data["description"]
    assert "Remediation MR: !123" in data["description"]
    assert "CVE-2026-CRITICAL" in data["description"]


def test_clean_scan_closes_open_issue() -> None:
    description = _new_description(
        _managed_block(_report([_finding()]), ENVIRONMENT, NOW, date(2026, 9, 8))
    )
    api = FakeAPI([{"iid": 9, "title": ISSUE_TITLE, "description": description}])

    status, message = sync_issue(api, _report([]), ENVIRONMENT, now=NOW)

    assert status == 0
    assert "Closed" in message
    assert api.calls[-1][3]["state_event"] == "close"
    assert "No fixable HIGH or CRITICAL" in api.calls[-1][3]["description"]
    assert "**SLA due:** Cleared by this scan" in api.calls[-1][3]["description"]


def test_rescan_never_extends_an_earlier_due_date() -> None:
    description = _new_description(
        _managed_block(_report([_finding("CRITICAL")]), ENVIRONMENT, NOW, date(2026, 8, 26))
    )
    api = FakeAPI(
        [{"iid": 9, "title": ISSUE_TITLE, "description": description, "due_date": "2026-08-26"}]
    )

    status, _ = sync_issue(api, _report([_finding()]), ENVIRONMENT, now=NOW)

    assert status == 1
    data = api.calls[-1][3]
    assert data["due_date"] == "2026-08-26"
    assert "**SLA due:** 2026-08-26" in data["description"]


def test_existing_issue_without_due_date_fails_instead_of_resetting_sla() -> None:
    description = _new_description(
        _managed_block(_report([_finding()]), ENVIRONMENT, NOW, date(2026, 9, 8))
    )
    api = FakeAPI([{"iid": 9, "title": ISSUE_TITLE, "description": description, "due_date": None}])

    with pytest.raises(IssueUpdateError, match="refusing to reset"):
        sync_issue(api, _report([_finding()]), ENVIRONMENT, now=NOW)


def test_clean_scan_without_issue_is_noop() -> None:
    api = FakeAPI()
    status, message = sync_issue(api, _report([]), ENVIRONMENT, now=NOW)

    assert status == 0
    assert "no open managed issue" in message
    assert len(api.calls) == 1


def test_duplicate_managed_issues_fail_closed() -> None:
    api = FakeAPI(
        [
            {"iid": 1, "title": ISSUE_TITLE, "description": "one"},
            {"iid": 2, "title": ISSUE_TITLE, "description": "two"},
        ]
    )
    with pytest.raises(IssueUpdateError, match="2 open issues"):
        sync_issue(api, _report([_finding()]), ENVIRONMENT, now=NOW)


def test_matching_human_issue_without_markers_is_not_overwritten() -> None:
    api = FakeAPI([{"iid": 9, "title": ISSUE_TITLE, "description": "human content"}])
    with pytest.raises(IssueUpdateError, match="does not contain"):
        sync_issue(api, _report([_finding()]), ENVIRONMENT, now=NOW)
