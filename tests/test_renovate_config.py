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
"""Guard the custom dependency extraction in ``renovate.json``."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / "renovate.json").read_text(encoding="utf-8"))


def _python_regex(expression: str) -> re.Pattern[str]:
    """Translate Renovate's JavaScript named-group syntax for Python's re."""
    return re.compile(expression.replace("(?<", "(?P<"))


def _extract(manager: dict[str, Any], content: str) -> list[dict[str, str]]:
    """Apply the ``any`` and ``recursive`` strategies used by this config."""
    if manager.get("matchStringsStrategy", "any") == "any":
        return [
            match.groupdict()
            for expression in manager["matchStrings"]
            for match in _python_regex(expression).finditer(content)
        ]

    assert manager["matchStringsStrategy"] == "recursive"
    scopes = [content]
    matches: list[re.Match[str]] = []
    for expression in manager["matchStrings"]:
        matches = [match for scope in scopes for match in _python_regex(expression).finditer(scope)]
        scopes = [match.group(0) for match in matches]
    return [match.groupdict() for match in matches]


def _manager_with(**expected: str) -> dict[str, Any]:
    matches = [
        manager
        for manager in CONFIG["customManagers"]
        if all(manager.get(key) == value for key, value in expected.items())
    ]
    assert len(matches) == 1
    return matches[0]


def _requirement_parts(requirement: str) -> tuple[str, str]:
    match = re.fullmatch(r"(?P<name>[A-Za-z0-9._-]+)(?:\[[^\]]+\])?(?P<constraint>.+)", requirement)
    assert match is not None
    return match.group("name"), match.group("constraint")


def _replace_current_value(content: str, match: dict[str, str], new_value: str) -> str:
    """Replace exactly the captured value, matching Renovate's default behavior."""
    old_value = match["currentValue"]
    assert content.count(old_value) == 1
    return content.replace(old_value, new_value)


def test_uv_policy_manager_extracts_every_compatibility_override() -> None:
    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    pyproject = tomllib.loads(pyproject_text)
    uv = pyproject["tool"]["uv"]

    manager = _manager_with(depTypeTemplate="tool.uv.override-dependencies")
    actual = {
        (match["depName"], match["currentValue"]) for match in _extract(manager, pyproject_text)
    }
    expected = {
        _requirement_parts(requirement)
        for requirement in uv["override-dependencies"]
        if " @ " not in requirement
    }
    assert actual == expected

    direct_overrides = [
        requirement for requirement in uv["override-dependencies"] if " @ " in requirement
    ]
    assert direct_overrides == [
        "nemo-run @ git+https://github.com/NVIDIA-NeMo/Run@17ae86b64d7f75653351664f5d8c9e466faede00"
    ]

    policy_rules = [
        rule
        for rule in CONFIG["packageRules"]
        if rule.get("matchDepTypes") == ["tool.uv.override-dependencies"]
    ]
    assert len(policy_rules) == 1
    assert policy_rules[0]["dependencyDashboardApproval"] is True
    # Unbounded compatibility ranges such as ``urllib3>=2.6.3`` accept every newer
    # release. Without ``bump``, Renovate detects them but never raises the floor.
    assert policy_rules[0]["rangeStrategy"] == "bump"


def test_uv_policy_manager_preserves_package_name_when_replacing_range() -> None:
    manager = _manager_with(depTypeTemplate="tool.uv.override-dependencies")
    fixture = '[tool.uv]\noverride-dependencies = [\n    "litellm>=1.84.0,<1.85",\n]\n'
    match = _extract(manager, fixture)[0]

    assert match == {"depName": "litellm", "currentValue": ">=1.84.0,<1.85"}
    assert _replace_current_value(fixture, match, ">=1.95.0,<1.96") == (
        '[tool.uv]\noverride-dependencies = [\n    "litellm>=1.95.0,<1.96",\n]\n'
    )


def test_weekly_run_has_no_hourly_throttle_and_weekly_lock_maintenance() -> None:
    assert CONFIG["commitHourlyLimit"] == 0
    assert CONFIG["prHourlyLimit"] == 0
    assert CONFIG["lockFileMaintenance"]["schedule"] == ["* 6-7 * * 1"]


def test_validator_image_is_excluded_from_gitlab_ci_updates() -> None:
    rules = [rule for rule in CONFIG["packageRules"] if rule.get("enabled") is False]
    assert any(
        rule.get("matchManagers") == ["gitlabci"]
        and rule.get("matchPackageNames") == ["renovate/renovate"]
        and rule.get("matchFileNames") == [".gitlab-ci.yml"]
        for rule in rules
    )


def test_trufflehog_image_is_visible_to_gitlab_ci_manager() -> None:
    ci_text = (ROOT / ".gitlab-ci.yml").read_text(encoding="utf-8")

    assert re.search(
        r"(?m)^\.trufflehog-image:\n"
        r"  image:\n"
        r"    name: trufflesecurity/trufflehog:[^\s]+@sha256:[0-9a-f]{64}$",
        ci_text,
    )
    assert ci_text.count("extends: .trufflehog-image") == 2
    assert "&trufflehog-image" not in ci_text
    assert "*trufflehog-image" not in ci_text
