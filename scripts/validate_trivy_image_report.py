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
"""Fail closed unless a Trivy image report proves the expected native platform."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SUPPORTED_SCHEMA_VERSION = 2
SUPPORTED_PLATFORMS = ("linux/amd64", "linux/arm64")


class ImageReportError(ValueError):
    """Raised when an image report does not prove the requested platform."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ImageReportError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def load_image_platform(path: Path) -> str:
    try:
        report = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ImageReportError) as error:
        raise ImageReportError(f"{path}: cannot read a valid Trivy JSON report: {error}") from error
    if not isinstance(report, dict):
        raise ImageReportError(f"{path}: top-level JSON value must be an object")
    if report.get("SchemaVersion") != SUPPORTED_SCHEMA_VERSION:
        raise ImageReportError(f"{path}: unsupported Trivy SchemaVersion")
    metadata = report.get("Metadata")
    if not isinstance(metadata, dict):
        raise ImageReportError(f"{path}: Metadata must be an object")
    image_config = metadata.get("ImageConfig")
    if not isinstance(image_config, dict):
        raise ImageReportError(f"{path}: Metadata.ImageConfig must be an object")
    operating_system = image_config.get("os")
    architecture = image_config.get("architecture")
    if not isinstance(operating_system, str) or not operating_system:
        raise ImageReportError(f"{path}: Metadata.ImageConfig.os must be a non-empty string")
    if not isinstance(architecture, str) or not architecture:
        raise ImageReportError(
            f"{path}: Metadata.ImageConfig.architecture must be a non-empty string"
        )
    return f"{operating_system}/{architecture}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Trivy image JSON report")
    parser.add_argument("--expected-platform", choices=SUPPORTED_PLATFORMS, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        actual_platform = load_image_platform(args.report)
        if actual_platform != args.expected_platform:
            raise ImageReportError(
                f"{args.report}: expected {args.expected_platform}, report describes {actual_platform}"
            )
    except ImageReportError as error:
        print(f"validate_trivy_image_report: {error}", file=sys.stderr)
        return 2
    print(f"Validated promotion image platform: {actual_platform}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
