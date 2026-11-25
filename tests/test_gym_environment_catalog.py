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
"""Deterministic, explicitly indexed Gym environment catalogs."""

from pathlib import Path

import pytest

from nvflow.integrations.nemo_gym.contracts import GymContractError
from nvflow.integrations.nemo_gym.environments import load_environment_catalog


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_catalog_loads_in_index_order_with_semantic_digest(tmp_path: Path) -> None:
    index = tmp_path / "catalog.yaml"
    first = tmp_path / "first.yaml"
    second = tmp_path / "nested/second.yaml"
    _write(
        index,
        """environments:
  first-environment: first.yaml
  second-environment: nested/second.yaml
""",
    )
    _write(
        first,
        """environment: first-environment
enabled: true
config_paths: [first/config.yaml]
agent_name: first_agent
""",
    )
    _write(
        second,
        """environment: second-environment
enabled: false
config_paths: [second/config.yaml]
agent_name: second_agent
""",
    )

    catalog = load_environment_catalog(index, boundary=tmp_path)

    assert list(catalog.environments) == ["first-environment", "second-environment"]
    assert catalog.environments["first-environment"]["agent_name"] == "first_agent"
    assert catalog.digest.startswith("sha256:")
    assert len(catalog.digest) == len("sha256:") + 64
    assert catalog.files == (index.resolve(), first.resolve(), second.resolve())

    first.write_text(
        """# comments and formatting do not change semantic identity
environment: first-environment
enabled: true
config_paths:
  - first/config.yaml
agent_name: first_agent
""",
        encoding="utf-8",
    )
    assert load_environment_catalog(index, boundary=tmp_path).digest == catalog.digest

    first.write_text(first.read_text().replace("first_agent", "changed_agent"), encoding="utf-8")
    assert load_environment_catalog(index, boundary=tmp_path).digest != catalog.digest


@pytest.mark.parametrize(
    ("index_text", "message"),
    [
        ("other: {}\n", "unknown environment catalog index"),
        ("environments: {}\n", "non-empty environments"),
        (
            """environments:
  duplicate: first.yaml
  duplicate: second.yaml
""",
            "duplicate key",
        ),
        (
            """environments:
  invalid identity: first.yaml
""",
            "safe non-empty identity",
        ),
        (
            """environments:
  valid: ../outside.yaml
""",
            "stay within the catalog",
        ),
        (
            """environments:
  valid: /absolute/environment.yaml
""",
            "stay within the catalog",
        ),
        (
            """environments:
  valid: environment.json
""",
            "name a .yaml or .yml",
        ),
    ],
)
def test_catalog_rejects_ambiguous_index(
    tmp_path: Path,
    index_text: str,
    message: str,
) -> None:
    index = tmp_path / "catalog.yaml"
    _write(index, index_text)

    with pytest.raises(GymContractError, match=message):
        load_environment_catalog(index, boundary=tmp_path)


def test_catalog_rejects_missing_or_mismatched_environment_file(tmp_path: Path) -> None:
    index = tmp_path / "catalog.yaml"
    _write(index, "environments:\n  expected: environment.yaml\n")

    with pytest.raises(GymContractError, match="unable to read environment"):
        load_environment_catalog(index)

    _write(
        tmp_path / "environment.yaml",
        """environment: different
config_paths: [environment.yaml]
agent_name: agent
""",
    )
    with pytest.raises(GymContractError, match="catalog key is 'expected'"):
        load_environment_catalog(index)


def test_catalog_rejects_duplicate_environment_fields_and_path_escape(tmp_path: Path) -> None:
    root = tmp_path / "catalog"
    index = root / "catalog.yaml"
    environment = root / "environment.yaml"
    _write(index, "environments:\n  expected: environment.yaml\n")
    _write(
        environment,
        """environment: expected
agent_name: first
agent_name: second
""",
    )
    with pytest.raises(GymContractError, match="duplicate key"):
        load_environment_catalog(index, boundary=root)

    outside = tmp_path / "outside"
    _write(outside / "catalog.yaml", "environments:\n  expected: environment.yaml\n")
    _write(outside / "environment.yaml", "environment: expected\nagent_name: agent\n")
    with pytest.raises(GymContractError, match="outside catalog boundary"):
        load_environment_catalog(outside / "catalog.yaml", boundary=root)
