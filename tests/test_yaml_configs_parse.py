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
"""A recipe config is only read when a cluster job starts, so a malformed one
fails after the allocation rather than at review time.
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOTS = (REPO_ROOT / "nvflow", REPO_ROOT / "projects")


def _yaml_files() -> list[Path]:
    return sorted(
        path
        for root in CONFIG_ROOTS
        for path in root.rglob("*")
        if path.suffix in (".yaml", ".yml")
    )


@pytest.mark.parametrize("config", _yaml_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_yaml_parses(config: Path):
    yaml.safe_load(config.read_text())
