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
"""Tests for the baseline-aware Trivy repository policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.trivy_report_diff import ReportError, evaluate, load_policy_findings, main


def _vulnerability(
    vulnerability_id: str = "CVE-2026-0001",
    *,
    package: str = "example",
    installed: str = "1.0.0",
    fixed: str = "1.0.1",
    severity: str = "HIGH",
    status: str = "fixed",
    package_path: str | None = None,
) -> dict[str, Any]:
    return {
        "VulnerabilityID": vulnerability_id,
        "PkgID": f"{package}@{installed}",
        "PkgName": package,
        "PkgPath": package_path,
        "PkgIdentifier": {"PURL": f"pkg:pypi/{package}@{installed}", "UID": "ignored"},
        "InstalledVersion": installed,
        "FixedVersion": fixed,
        "Status": status,
        "Severity": severity,
        "Title": f"{vulnerability_id} title",
        "PrimaryURL": f"https://example.invalid/{vulnerability_id}",
        "Fingerprint": "artifact-specific-and-ignored",
    }


def _report(
    vulnerabilities: list[dict[str, Any]] | None = None,
    *,
    target: str = "uv.lock",
    result_class: str = "lang-pkgs",
    package_type: str = "uv",
) -> dict[str, Any]:
    return {
        "SchemaVersion": 2,
        "ArtifactName": "ignored",
        "Results": [
            {
                "Target": target,
                "Class": result_class,
                "Type": package_type,
                "Vulnerabilities": vulnerabilities,
            }
        ],
    }


def _write(path: Path, report: dict[str, Any]) -> Path:
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _run(
    tmp_path: Path,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> tuple[int, dict[str, Any], bytes]:
    candidate_path = _write(tmp_path / "candidate.json", candidate)
    baseline_path = _write(tmp_path / "baseline.json", baseline)
    output = tmp_path / "gate.json"
    status = main(
        [
            str(candidate_path),
            "--baseline",
            str(baseline_path),
            "--policy",
            "delta",
            "--output",
            str(output),
        ]
    )
    return status, json.loads(output.read_text(encoding="utf-8")), output.read_bytes()


def test_identical_reordered_findings_are_inherited(tmp_path: Path) -> None:
    high = _vulnerability()
    critical = _vulnerability("CVE-2026-0002", package="second", severity="CRITICAL")
    status, result, _ = _run(tmp_path, _report([high, critical]), _report([critical, high]))

    assert status == 0
    assert result["candidate_finding_count"] == 2
    assert result["baseline_finding_count"] == 2
    assert result["blocking_finding_count"] == 0


def test_new_findings_fail_and_are_critical_first(tmp_path: Path) -> None:
    high = _vulnerability()
    critical = _vulnerability("CVE-2026-0002", package="second", severity="CRITICAL")
    status, result, _ = _run(tmp_path, _report([high, critical]), _report([]))

    assert status == 1
    assert result["blocking_by_severity"] == {"CRITICAL": 1, "HIGH": 1}
    assert [finding["severity"] for finding in result["findings"]] == ["CRITICAL", "HIGH"]


def test_version_change_is_new_but_fixed_version_change_is_not(tmp_path: Path) -> None:
    baseline = _vulnerability(installed="1.0.0", fixed="1.0.1")
    changed_version = _vulnerability(installed="1.0.0.post1", fixed="1.0.2")
    status, result, _ = _run(tmp_path, _report([changed_version]), _report([baseline]))
    assert status == 1
    assert result["resolved_finding_count"] == 1

    same_version = _vulnerability(installed="1.0.0", fixed="9.9.9")
    status, result, _ = _run(tmp_path, _report([same_version]), _report([baseline]))
    assert status == 0
    assert result["blocking_finding_count"] == 0


def test_only_fixed_high_and_critical_findings_are_eligible(tmp_path: Path) -> None:
    vulnerabilities = [
        _vulnerability("LOW", severity="LOW"),
        _vulnerability("MEDIUM", severity="MEDIUM"),
        _vulnerability("AFFECTED", status="affected"),
        _vulnerability("WONTFIX", status="will_not_fix"),
        _vulnerability("DEFERRED", status="fix_deferred"),
        _vulnerability("EOL", status="end_of_life"),
        _vulnerability("HIGH"),
        _vulnerability("CRITICAL", severity="CRITICAL"),
    ]
    report_path = _write(tmp_path / "report.json", _report(vulnerabilities))

    findings = load_policy_findings(report_path)

    assert {key.vulnerability_id for key in findings} == {"HIGH", "CRITICAL"}


@pytest.mark.parametrize("fixed", ["", "   ", None, 7])
def test_fixed_status_requires_nonblank_string_fixed_version(tmp_path: Path, fixed: Any) -> None:
    vulnerability = _vulnerability()
    vulnerability["FixedVersion"] = fixed
    report_path = _write(tmp_path / "report.json", _report([vulnerability]))

    with pytest.raises(ReportError, match="FixedVersion"):
        load_policy_findings(report_path)


def test_exact_duplicates_collapse(tmp_path: Path) -> None:
    vulnerability = _vulnerability()
    report = _report([vulnerability, dict(vulnerability)])
    report["Results"].append(dict(report["Results"][0]))
    report_path = _write(tmp_path / "report.json", report)

    assert len(load_policy_findings(report_path)) == 1


@pytest.mark.parametrize("results", [None, []])
def test_empty_results_are_valid(tmp_path: Path, results: Any) -> None:
    report = {"SchemaVersion": 2, "Results": results}
    assert load_policy_findings(_write(tmp_path / "report.json", report)) == {}


def test_missing_results_are_valid(tmp_path: Path) -> None:
    assert load_policy_findings(_write(tmp_path / "report.json", {"SchemaVersion": 2})) == {}


def test_location_and_package_identity_changes_are_new(tmp_path: Path) -> None:
    vulnerability = _vulnerability(package_path="one")
    baseline = _report([vulnerability])
    candidate = _report([_vulnerability(package_path="two")], target="nested/uv.lock")
    status, result, _ = _run(tmp_path, candidate, baseline)

    assert status == 1
    assert result["blocking_finding_count"] == 1


@pytest.mark.parametrize(
    "content, match",
    [
        ('{"SchemaVersion":2,"SchemaVersion":2}', "duplicate JSON key"),
        ('{"SchemaVersion":2,"Results":NaN}', "non-finite JSON number"),
        ('{"SchemaVersion":1}', "unsupported SchemaVersion"),
        ("[]", "top-level JSON value"),
        ('{"SchemaVersion":2,"Results":{}}', "Results must be a list"),
    ],
)
def test_malformed_reports_fail_closed(tmp_path: Path, content: str, match: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ReportError, match=match):
        load_policy_findings(path)


def test_output_is_deterministic_for_input_order(tmp_path: Path) -> None:
    first = _vulnerability("CVE-2026-0001", package="zeta")
    second = _vulnerability("CVE-2026-0002", package="alpha", severity="CRITICAL")
    status, _, output_one = _run(tmp_path, _report([first, second]), _report([]))
    assert status == 1

    other_tmp = tmp_path / "other"
    other_tmp.mkdir()
    status, _, output_two = _run(other_tmp, _report([second, first]), _report([]))
    assert status == 1
    assert output_one == output_two


def test_report_policy_never_fails_for_findings(tmp_path: Path) -> None:
    candidate_path = _write(tmp_path / "candidate.json", _report([_vulnerability()]))
    output = tmp_path / "gate.json"

    status = main([str(candidate_path), "--policy", "report", "--output", str(output)])

    assert status == 0
    assert json.loads(output.read_text(encoding="utf-8"))["blocking_finding_count"] == 1


def test_evaluate_rejects_incompatible_policy_inputs() -> None:
    with pytest.raises(ReportError, match="requires a baseline"):
        evaluate({}, policy="delta")
    with pytest.raises(ReportError, match="does not accept"):
        evaluate({}, policy="strict", baseline={})
