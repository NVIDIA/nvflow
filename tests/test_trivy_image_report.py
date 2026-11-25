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
"""Tests for immutable promotion-image platform validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validate_trivy_image_report import ImageReportError, load_image_platform, main


def _write_report(
    path: Path, *, operating_system: str = "linux", architecture: str = "amd64"
) -> Path:
    path.write_text(
        json.dumps(
            {
                "SchemaVersion": 2,
                "Metadata": {"ImageConfig": {"os": operating_system, "architecture": architecture}},
                "Results": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_loads_native_platform(tmp_path: Path) -> None:
    assert load_image_platform(_write_report(tmp_path / "report.json")) == "linux/amd64"


def test_expected_platform_passes(tmp_path: Path) -> None:
    report = _write_report(tmp_path / "report.json", architecture="arm64")
    assert main([str(report), "--expected-platform", "linux/arm64"]) == 0


def test_wrong_architecture_fails_closed(tmp_path: Path) -> None:
    report = _write_report(tmp_path / "report.json", architecture="amd64")
    assert main([str(report), "--expected-platform", "linux/arm64"]) == 2


@pytest.mark.parametrize(
    "report",
    [
        {},
        {"SchemaVersion": 1},
        {"SchemaVersion": 2, "Metadata": None},
        {"SchemaVersion": 2, "Metadata": {"ImageConfig": {"os": "linux"}}},
    ],
)
def test_malformed_metadata_fails_closed(tmp_path: Path, report: dict[str, object]) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ImageReportError):
        load_image_platform(path)
