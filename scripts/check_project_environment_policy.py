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
"""Generate or check bundled project-environment policy from root uv policy."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_ROOT = PROJECT_ROOT / "nvflow" / "project" / "_environment_policy"
NEMO_SKILLS_DOCKERFILE = PROJECT_ROOT / "dockerfiles" / "Dockerfile.nemo-skills"
_GENERATED_HEADER = (
    "# Generated from the root pyproject.toml by\n"
    "# `python scripts/check_project_environment_policy.py --write`; do not edit.\n"
)


def _requirements(path: Path) -> list[str]:
    return [
        line
        for raw_line in path.read_text(encoding="utf-8").splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    ]


def _render(requirements: list[str]) -> str:
    return _GENERATED_HEADER + "".join(f"{requirement}\n" for requirement in requirements)


def _nemo_skills_commit(requirements: list[str]) -> str:
    requirement = next(value for value in requirements if value.startswith("nemo-skills @ git+"))
    match = re.search(r"@([0-9a-f]{40})$", requirement)
    if match is None:
        raise SystemExit(f"NVFlow execution uses an invalid NeMo-Skills pin: {requirement}")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="Regenerate bundled policy files from root pyproject.toml",
    )
    args = parser.parse_args()
    root = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    uv = root["tool"]["uv"]
    expected = {
        "execution.txt": root["project"]["optional-dependencies"]["execution"],
        "overrides.txt": uv["override-dependencies"],
        "constraints.txt": uv.get("constraint-dependencies", []),
    }
    nemo_skills_commit = _nemo_skills_commit(expected["execution.txt"])
    if args.write:
        for name, requirements in expected.items():
            (POLICY_ROOT / name).write_text(_render(requirements), encoding="utf-8")
        dockerfile = NEMO_SKILLS_DOCKERFILE.read_text(encoding="utf-8")
        dockerfile = re.sub(
            r"(?m)^ARG NEMO_SKILLS_COMMIT=[0-9a-f]{40}$",
            f"ARG NEMO_SKILLS_COMMIT={nemo_skills_commit}",
            dockerfile,
        )
        NEMO_SKILLS_DOCKERFILE.write_text(dockerfile, encoding="utf-8")
        return

    drift = [
        name
        for name, requirements in expected.items()
        if _requirements(POLICY_ROOT / name) != requirements
    ]
    if drift:
        raise SystemExit(
            "Bundled project environment policy differs from root pyproject.toml: "
            + ", ".join(drift)
            + ". Regenerate it with "
            "`python scripts/check_project_environment_policy.py --write`."
        )
    dockerfile = NEMO_SKILLS_DOCKERFILE.read_text(encoding="utf-8")
    if f"ARG NEMO_SKILLS_COMMIT={nemo_skills_commit}" not in dockerfile:
        raise SystemExit(
            "Dockerfile.nemo-skills differs from the root NVFlow execution pin. "
            "Regenerate it with "
            "`python scripts/check_project_environment_policy.py --write`."
        )


if __name__ == "__main__":
    main()
