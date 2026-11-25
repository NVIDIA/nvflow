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
"""Guard the Trivy CI policy and its first-match rule ordering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CI_TEXT = (ROOT / ".gitlab-ci.yml").read_text(encoding="utf-8")
CI: dict[str, Any] = yaml.safe_load(CI_TEXT)


def _rule(job: dict[str, Any], expression_fragment: str) -> dict[str, Any]:
    matches = [rule for rule in job["rules"] if expression_fragment in str(rule.get("if", ""))]
    assert len(matches) == 1
    return matches[0]


def test_merge_requests_use_delta_policy_before_generic_override() -> None:
    job = CI["trivy-repository-scan"]
    rules = job["rules"]
    mr_index = next(
        index
        for index, rule in enumerate(rules)
        if 'CI_PIPELINE_SOURCE == "merge_request_event"' in rule.get("if", "")
    )
    override_indices = [
        index
        for index, rule in enumerate(rules)
        if 'TRIVY_REPOSITORY_SCAN == "true"' in rule.get("if", "")
        and 'CI_PIPELINE_SOURCE == "schedule"' not in rule.get("if", "")
    ]

    assert override_indices
    assert all(mr_index < index for index in override_indices)
    assert "CI_MERGE_REQUEST_TARGET_BRANCH_SHA" in "\n".join(job["script"])
    assert "--skip-db-update" in "\n".join(job["script"])


def test_dev_pushes_do_not_run_repository_scan() -> None:
    rules = CI["trivy-repository-scan"]["rules"]
    assert not any(rule.get("if") == '$CI_COMMIT_BRANCH == "dev"' for rule in rules)

    scheduled = [
        rule
        for rule in rules
        if 'CI_PIPELINE_SOURCE == "schedule"' in rule.get("if", "") and rule.get("when") != "never"
    ]
    assert len(scheduled) == 2
    assert all('$CI_COMMIT_BRANCH == "dev"' in rule["if"] for rule in scheduled)
    assert all('$CI_COMMIT_REF_PROTECTED == "true"' in rule["if"] for rule in scheduled)

    on_demand = [rule["if"] for rule in rules if 'TRIVY_REPOSITORY_SCAN == "true"' in rule["if"]]
    assert not any(expression == '$TRIVY_REPOSITORY_SCAN == "true"' for expression in on_demand)
    assert all('CI_PIPELINE_SOURCE == "push"' not in expression for expression in on_demand)


def test_scan_policy_cannot_be_downgraded_by_a_custom_variable() -> None:
    job = CI["trivy-repository-scan"]
    script = "\n".join(job["script"])

    assert "TRIVY_SCAN_MODE" not in CI_TEXT
    assert 'SCAN_POLICY="strict"' in script
    assert 'SCAN_POLICY="delta"' in script
    assert 'SCAN_POLICY="report"' in script
    assert 'case "$SCAN_POLICY"' in script


def test_main_and_release_tags_retain_strict_default() -> None:
    job = CI["trivy-repository-scan"]
    assert _rule(job, '$CI_COMMIT_BRANCH == "main"') == {"if": '$CI_COMMIT_BRANCH == "main"'}
    assert "CI_COMMIT_TAG" in _rule(job, "CI_COMMIT_TAG")["if"]


def test_candidate_cannot_supply_trivy_ignore_policy() -> None:
    script = "\n".join(CI["trivy-repository-scan"]["script"])
    assert script.count("--ignorefile /dev/null") == 2
    assert script.count("--config /dev/null") == 2
    assert "--policy delta" in script
    assert "--policy strict" in script
    assert "--policy report" in script


def test_scheduled_issue_job_is_protected_and_consumes_scan_artifacts() -> None:
    job = CI["trivy-repository-remediation"]
    assert job["needs"] == [{"job": "trivy-repository-scan", "artifacts": True}]
    assert job["before_script"] == []
    assert job["resource_group"] == "trivy-dev-repository-remediation"
    assert "update_trivy_issue.py" in "\n".join(job["script"])
    assert len(job["rules"]) == 2
    assert all('$CI_COMMIT_REF_PROTECTED == "true"' in rule["if"] for rule in job["rules"])
    assert "TRIVY_ISSUE_TOKEN" not in "\n".join(CI["trivy-repository-scan"]["script"])


def test_promotion_requires_two_immutable_platform_inputs() -> None:
    job = CI["promotion-container-image-scan"]
    matrix = job["parallel"]["matrix"]
    assert {entry["TRIVY_PLATFORM"] for entry in matrix} == {"linux/amd64", "linux/arm64"}
    script = "\n".join(job["script"])
    assert "TRIVY_IMAGE_AMD64" in script
    assert "TRIVY_IMAGE_ARM64" in script
    assert "@sha256:[0-9a-f]{64}$" in script
    assert 'TRIVY_IMAGE_AMD64" != "$TRIVY_IMAGE_ARM64' in script
    assert script.count("--config /dev/null") == 2
    assert script.count("--ignorefile /dev/null") == 2
    assert "validate_trivy_image_report.py" in script
    assert "--ignore-unfixed --exit-code 1" in script
    assert job["rules"] == [{"if": '$TRIVY_IMAGE_PROMOTION == "true"'}]
