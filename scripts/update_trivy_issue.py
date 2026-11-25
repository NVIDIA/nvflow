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
"""Maintain the confidential GitLab issue for scheduled Trivy findings."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

ISSUE_TITLE = "[Security] Fixable HIGH/CRITICAL findings on dev"
ISSUE_LABELS = "security,dependencies,automated-trivy"
MANAGED_START = "<!-- nvflow-trivy-dev-baseline:managed:start -->"
MANAGED_END = "<!-- nvflow-trivy-dev-baseline:managed:end -->"


class IssueUpdateError(RuntimeError):
    """Raised when issue synchronization cannot be completed safely."""


class IssueAPI(Protocol):
    """Small API surface required by the issue synchronizer."""

    def request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, str] | None = None,
        data: Mapping[str, str] | None = None,
    ) -> Any: ...


class GitLabAPI:
    def __init__(self, api_url: str, project_id: str, token: str) -> None:
        self.api_url = api_url.rstrip("/")
        self.project_id = urllib.parse.quote(project_id, safe="")
        self.token = token

    def request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, str] | None = None,
        data: Mapping[str, str] | None = None,
    ) -> Any:
        url = f"{self.api_url}/projects/{self.project_id}/{path.lstrip('/')}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        encoded = urllib.parse.urlencode(data).encode() if data is not None else None
        request = urllib.request.Request(
            url,
            data=encoded,
            method=method,
            headers={"PRIVATE-TOKEN": self.token, "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            body = error.read(1000).decode("utf-8", errors="replace")
            raise IssueUpdateError(
                f"GitLab API {method} {path} returned HTTP {error.code}: {body}"
            ) from error
        except (OSError, UnicodeError) as error:
            raise IssueUpdateError(f"GitLab API {method} {path} failed: {error}") from error
        try:
            return json.loads(payload)
        except json.JSONDecodeError as error:
            raise IssueUpdateError(f"GitLab API {method} {path} returned invalid JSON") from error


def _load_gate_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise IssueUpdateError(f"{path}: cannot read gate report: {error}") from error
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise IssueUpdateError(f"{path}: unsupported gate report schema")
    if report.get("policy") != "report":
        raise IssueUpdateError(f"{path}: issue updater requires a report-policy result")
    findings = report.get("findings")
    if not isinstance(findings, list) or not all(isinstance(item, dict) for item in findings):
        raise IssueUpdateError(f"{path}: findings must be a list of objects")
    return report


def _required_env(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "").strip()
    if not value:
        raise IssueUpdateError(f"required environment variable {name} is not configured")
    return value


def _positive_days(environment: Mapping[str, str], name: str, default: int) -> int:
    raw = environment.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise IssueUpdateError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise IssueUpdateError(f"{name} must be a positive integer")
    return value


def _markdown(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _finding_table(findings: list[dict[str, Any]]) -> str:
    if not findings:
        return "No fixable HIGH or CRITICAL repository findings were detected."
    lines = [
        "| Severity | ID | Package | Installed | Fixed | Target |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for finding in findings:
        fixed_versions = finding.get("fixed_versions")
        if not isinstance(fixed_versions, list):
            raise IssueUpdateError("gate finding fixed_versions must be a list")
        lines.append(
            "| "
            + " | ".join(
                _markdown(value)
                for value in (
                    finding.get("severity", ""),
                    finding.get("vulnerability_id", ""),
                    finding.get("package_name", ""),
                    finding.get("installed_version", ""),
                    ", ".join(str(value) for value in fixed_versions),
                    finding.get("target", ""),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _managed_block(
    report: dict[str, Any],
    environment: Mapping[str, str],
    now: datetime,
    due_date: date | None,
) -> str:
    findings = report["findings"]
    critical = sum(item.get("severity") == "CRITICAL" for item in findings)
    high = sum(item.get("severity") == "HIGH" for item in findings)
    pipeline_url = environment.get("CI_PIPELINE_URL", "").strip()
    scan_job_url = environment.get("TRIVY_SCAN_JOB_URL", "").strip()
    artifact_url = f"{scan_job_url}/artifacts/file/trivy-repository.json" if scan_job_url else ""
    links = []
    if pipeline_url:
        links.append(f"[pipeline]({pipeline_url})")
    if scan_job_url:
        links.append(f"[scan job]({scan_job_url})")
    if artifact_url:
        links.append(f"[raw JSON artifact]({artifact_url})")
    link_text = " · ".join(links) or "Unavailable outside GitLab CI"
    sla_text = due_date.isoformat() if due_date is not None else "Cleared by this scan"

    return "\n".join(
        (
            MANAGED_START,
            "## Automated scan snapshot",
            "",
            "This section is managed by the scheduled `trivy-repository-scan` pipeline. ",
            "Maintainer notes below the marker are preserved.",
            "",
            f"- **Scanned:** {now.astimezone(UTC).isoformat()}",
            f"- **Ref:** `{environment.get('CI_COMMIT_REF_NAME', 'dev')}`",
            f"- **Commit:** `{environment.get('CI_COMMIT_SHA', 'unknown')}`",
            f"- **Fixable findings:** {critical} Critical, {high} High",
            f"- **SLA due:** {sla_text}",
            f"- **Evidence:** {link_text}",
            "",
            _finding_table(findings),
            MANAGED_END,
        )
    )


def _new_description(managed_block: str) -> str:
    return "\n".join(
        (
            managed_block,
            "",
            "## Maintainer triage",
            "",
            "- [ ] Assign a remediation owner.",
            "- [ ] Link a narrowly scoped security MR or an approved temporary exception.",
            "- [ ] Resolve by the SLA due date in the automated snapshot.",
            "- [ ] Record compensating controls and expiry for any exception.",
            "",
            "Product Security or organizational policy takes precedence when it is stricter.",
        )
    )


def _replace_managed(existing: str, managed_block: str) -> str:
    start, end = _managed_span(existing)
    return existing[:start] + managed_block + existing[end:]


def _managed_span(existing: str) -> tuple[int, int]:
    start = existing.find(MANAGED_START)
    end = existing.find(MANAGED_END)
    if start < 0 and end < 0:
        raise IssueUpdateError("matching issue does not contain the Trivy automation markers")
    if start < 0 or end < start:
        raise IssueUpdateError("matching issue contains malformed Trivy automation markers")
    end += len(MANAGED_END)
    return start, end


def _find_open_issue(api: IssueAPI) -> dict[str, Any] | None:
    response = api.request(
        "GET",
        "issues",
        query={"state": "opened", "search": ISSUE_TITLE, "in": "title", "per_page": "100"},
    )
    if not isinstance(response, list):
        raise IssueUpdateError("GitLab issue search returned a non-list response")
    matches = [issue for issue in response if issue.get("title") == ISSUE_TITLE]
    if len(matches) > 1:
        raise IssueUpdateError(f"found {len(matches)} open issues with the managed title")
    return matches[0] if matches else None


def _effective_due_date(existing: dict[str, Any] | None, proposed: date) -> date:
    """Return the earlier deadline, so rescans can tighten but never extend the SLA."""
    if existing is None:
        return proposed
    raw_due_date = existing.get("due_date")
    if raw_due_date in (None, ""):
        raise IssueUpdateError(
            "matching managed issue has no due_date; refusing to reset the existing SLA"
        )
    if not isinstance(raw_due_date, str):
        raise IssueUpdateError("matching issue has a non-string due_date")
    try:
        current = date.fromisoformat(raw_due_date)
    except ValueError as error:
        raise IssueUpdateError("matching issue has an invalid due_date") from error
    return min(current, proposed)


def sync_issue(
    api: IssueAPI,
    report: dict[str, Any],
    environment: Mapping[str, str],
    *,
    now: datetime,
) -> tuple[int, str]:
    findings = report["findings"]
    existing = _find_open_issue(api)
    if existing is not None:
        existing_description = existing.get("description")
        if not isinstance(existing_description, str):
            raise IssueUpdateError("matching issue has no string description")
        _managed_span(existing_description)
    critical_days = _positive_days(environment, "TRIVY_CRITICAL_SLA_DAYS", 3)
    high_days = _positive_days(environment, "TRIVY_HIGH_SLA_DAYS", 14)
    due_days = (
        critical_days if any(item.get("severity") == "CRITICAL" for item in findings) else high_days
    )
    proposed_due_date = now.date() + timedelta(days=due_days)
    due_date = _effective_due_date(existing, proposed_due_date) if findings else None
    managed_block = _managed_block(report, environment, now, due_date)

    if existing is None:
        if not findings:
            return 0, "No fixable findings and no open managed issue."
        assert due_date is not None
        created = api.request(
            "POST",
            "issues",
            data={
                "title": ISSUE_TITLE,
                "description": _new_description(managed_block),
                "labels": ISSUE_LABELS,
                "confidential": "true",
                "due_date": due_date.isoformat(),
            },
        )
        return 1, f"Created remediation issue {created.get('web_url', created.get('iid'))}."

    issue_iid = str(existing.get("iid", ""))
    if not issue_iid:
        raise IssueUpdateError("matching issue has no iid")
    description = existing.get("description")
    if not isinstance(description, str):
        raise IssueUpdateError("matching issue has no string description")
    updated_description = _replace_managed(description, managed_block)
    data = {
        "description": updated_description,
        "add_labels": ISSUE_LABELS,
        "confidential": "true",
    }
    if findings:
        assert due_date is not None
        data["due_date"] = due_date.isoformat()
    else:
        data["state_event"] = "close"
    updated = api.request("PUT", f"issues/{issue_iid}", data=data)
    action = "Closed" if not findings else "Updated"
    return (
        0 if not findings else 1,
        f"{action} remediation issue {updated.get('web_url', issue_iid)}.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gate_report", type=Path, help="report-policy JSON from trivy_report_diff")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = _load_gate_report(args.gate_report)
        environment = os.environ
        api = GitLabAPI(
            _required_env(environment, "CI_API_V4_URL"),
            _required_env(environment, "CI_PROJECT_ID"),
            _required_env(environment, "TRIVY_ISSUE_TOKEN"),
        )
        status, message = sync_issue(api, report, environment, now=datetime.now(UTC))
        print(message)
        return status
    except IssueUpdateError as error:
        print(f"update_trivy_issue: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
