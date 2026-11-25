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
"""Apply NVFlow's fixable HIGH/CRITICAL policy to Trivy JSON reports."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

SUPPORTED_SCHEMA_VERSION = 2
POLICY_SEVERITIES = {"HIGH", "CRITICAL"}
SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1}


class ReportError(ValueError):
    """Raised when a report cannot be evaluated safely."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ReportError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ReportError(f"non-finite JSON number: {value}")


def _required_string(value: Any, field_name: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReportError(f"{context}: {field_name} must be a non-empty string")
    return value.strip()


def _optional_string(value: Any, field_name: str, context: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ReportError(f"{context}: {field_name} must be a string or null")
    return value.strip()


def _normalize_target(value: str) -> str:
    normalized = value.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return str(PurePosixPath(normalized))


@dataclass(frozen=True, order=True)
class FindingKey:
    """Fields whose change makes a vulnerability finding MR-related."""

    target: str
    result_class: str
    package_type: str
    vulnerability_id: str
    package_name: str
    installed_version: str
    package_path: str
    package_identifier: str


@dataclass
class Finding:
    key: FindingKey
    severity: str
    fixed_versions: set[str] = field(default_factory=set)
    titles: set[str] = field(default_factory=set)
    urls: set[str] = field(default_factory=set)

    def merge(self, *, severity: str, fixed_version: str, title: str, url: str) -> None:
        if SEVERITY_ORDER[severity] < SEVERITY_ORDER[self.severity]:
            self.severity = severity
        self.fixed_versions.add(fixed_version)
        if title:
            self.titles.add(title)
        if url:
            self.urls.add(url)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "vulnerability_id": self.key.vulnerability_id,
            "package_name": self.key.package_name,
            "installed_version": self.key.installed_version,
            "fixed_versions": sorted(self.fixed_versions),
            "target": self.key.target,
            "result_class": self.key.result_class,
            "package_type": self.key.package_type,
            "package_path": self.key.package_path,
            "package_identifier": self.key.package_identifier,
            "title": min(self.titles, default=""),
            "primary_url": min(self.urls, default=""),
        }


def _finding_sort_key(finding: Finding) -> tuple[Any, ...]:
    return (
        SEVERITY_ORDER[finding.severity],
        finding.key.vulnerability_id,
        finding.key.package_name,
        finding.key.installed_version,
        finding.key.target,
        finding.key,
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ReportError(f"{path}: cannot read report: {error}") from error
    try:
        report = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ReportError) as error:
        raise ReportError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(report, dict):
        raise ReportError(f"{path}: top-level JSON value must be an object")
    if report.get("SchemaVersion") != SUPPORTED_SCHEMA_VERSION:
        raise ReportError(
            f"{path}: unsupported SchemaVersion {report.get('SchemaVersion')!r}; "
            f"expected {SUPPORTED_SCHEMA_VERSION}"
        )
    return report


def load_policy_findings(path: Path) -> dict[FindingKey, Finding]:
    """Load fixable HIGH/CRITICAL findings, deduplicated by stable identity."""
    report = _load_json(path)
    results = report.get("Results")
    if results is None:
        return {}
    if not isinstance(results, list):
        raise ReportError(f"{path}: Results must be a list or null")

    findings: dict[FindingKey, Finding] = {}
    for result_index, result in enumerate(results):
        result_context = f"{path}: Results[{result_index}]"
        if not isinstance(result, dict):
            raise ReportError(f"{result_context} must be an object")
        vulnerabilities = result.get("Vulnerabilities")
        if vulnerabilities is None:
            continue
        if not isinstance(vulnerabilities, list):
            raise ReportError(f"{result_context}.Vulnerabilities must be a list or null")

        target = _required_string(result.get("Target"), "Target", result_context)
        result_class = _optional_string(result.get("Class"), "Class", result_context)
        package_type = _optional_string(result.get("Type"), "Type", result_context)

        for vulnerability_index, vulnerability in enumerate(vulnerabilities):
            context = f"{result_context}.Vulnerabilities[{vulnerability_index}]"
            if not isinstance(vulnerability, dict):
                raise ReportError(f"{context} must be an object")
            severity = _optional_string(vulnerability.get("Severity"), "Severity", context).upper()
            status = _optional_string(vulnerability.get("Status"), "Status", context).lower()
            if severity not in POLICY_SEVERITIES or status != "fixed":
                continue

            fixed_version = _required_string(
                vulnerability.get("FixedVersion"), "FixedVersion", context
            )
            vulnerability_id = _required_string(
                vulnerability.get("VulnerabilityID"), "VulnerabilityID", context
            )
            package_name = _required_string(vulnerability.get("PkgName"), "PkgName", context)
            installed_version = _required_string(
                vulnerability.get("InstalledVersion"), "InstalledVersion", context
            )
            package_path = _optional_string(vulnerability.get("PkgPath"), "PkgPath", context)

            package_identifier_value = vulnerability.get("PkgIdentifier")
            if package_identifier_value is None:
                package_identifier: dict[str, Any] = {}
            elif isinstance(package_identifier_value, dict):
                package_identifier = package_identifier_value
            else:
                raise ReportError(f"{context}: PkgIdentifier must be an object or null")
            purl = _optional_string(package_identifier.get("PURL"), "PkgIdentifier.PURL", context)
            pkg_id = _optional_string(vulnerability.get("PkgID"), "PkgID", context)

            key = FindingKey(
                target=_normalize_target(target),
                result_class=result_class,
                package_type=package_type,
                vulnerability_id=vulnerability_id,
                package_name=package_name,
                installed_version=installed_version,
                package_path=package_path,
                package_identifier=purl or pkg_id,
            )
            title = _optional_string(vulnerability.get("Title"), "Title", context)
            url = _optional_string(vulnerability.get("PrimaryURL"), "PrimaryURL", context)
            if key not in findings:
                findings[key] = Finding(key=key, severity=severity)
            findings[key].merge(
                severity=severity,
                fixed_version=fixed_version,
                title=title,
                url=url,
            )
    return findings


def _serialize(findings: Iterable[Finding]) -> list[dict[str, Any]]:
    return [finding.to_dict() for finding in sorted(findings, key=_finding_sort_key)]


def evaluate(
    candidate: dict[FindingKey, Finding],
    *,
    policy: str,
    baseline: dict[FindingKey, Finding] | None = None,
) -> dict[str, Any]:
    if policy == "delta":
        if baseline is None:
            raise ReportError("delta policy requires a baseline report")
        blocking_keys = set(candidate) - set(baseline)
        resolved_keys = set(baseline) - set(candidate)
    elif policy in {"strict", "report"}:
        if baseline is not None:
            raise ReportError(f"{policy} policy does not accept a baseline report")
        blocking_keys = set(candidate)
        resolved_keys = set()
    else:
        raise ReportError(f"unknown policy: {policy}")

    blocking = [candidate[key] for key in blocking_keys]
    resolved = [baseline[key] for key in resolved_keys] if baseline is not None else []
    severity_counts = {
        severity: sum(finding.severity == severity for finding in blocking)
        for severity in ("CRITICAL", "HIGH")
    }
    return {
        "schema_version": 1,
        "policy": policy,
        "candidate_finding_count": len(candidate),
        "baseline_finding_count": len(baseline or {}),
        "blocking_finding_count": len(blocking),
        "resolved_finding_count": len(resolved),
        "blocking_by_severity": severity_counts,
        "findings": _serialize(blocking),
        "resolved_findings": _serialize(resolved),
    }


def _write_result(result: dict[str, Any], output: Path) -> None:
    try:
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as error:
        raise ReportError(f"{output}: cannot write result: {error}") from error


def _print_result(result: dict[str, Any]) -> None:
    policy = result["policy"]
    count = result["blocking_finding_count"]
    if policy == "delta":
        print(
            "Trivy repository delta: "
            f"candidate={result['candidate_finding_count']}, "
            f"target={result['baseline_finding_count']}, new={count}"
        )
    else:
        print(f"Trivy repository {policy}: fixable HIGH/CRITICAL={count}")
    for finding in result["findings"]:
        detail = {
            "severity": finding["severity"],
            "vulnerability_id": finding["vulnerability_id"],
            "package": finding["package_name"],
            "installed": finding["installed_version"],
            "fixed": finding["fixed_versions"],
            "target": finding["target"],
        }
        print(f"BLOCKING {json.dumps(detail, sort_keys=True, ensure_ascii=True)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="candidate Trivy JSON report")
    parser.add_argument("--baseline", type=Path, help="target-branch Trivy JSON report")
    parser.add_argument(
        "--policy",
        choices=("delta", "strict", "report"),
        required=True,
        help="delta fails on new findings; strict fails on all; report never fails for findings",
    )
    parser.add_argument("--output", type=Path, required=True, help="machine-readable result JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        candidate = load_policy_findings(args.candidate)
        baseline = load_policy_findings(args.baseline) if args.baseline else None
        result = evaluate(candidate, policy=args.policy, baseline=baseline)
        _write_result(result, args.output)
        _print_result(result)
    except ReportError as error:
        print(f"trivy_report_diff: {error}", file=sys.stderr)
        return 2
    if args.policy != "report" and result["blocking_finding_count"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
