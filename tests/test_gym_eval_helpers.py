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
"""Tests for the rollout helpers the eval matrix relies on.

Each helper here is shared with the GRPO rollout path, so the tests pin both
the scoped-selection behaviour used by both:

  - ``resolve_scoped_environments`` intersects instead of raising, which is what
    lets a stage scoped to one benchmark sit in a pipeline selected with ``-e``
    on another.
"""

from __future__ import annotations

import pytest

from nvflow.integrations.nemo_gym.environments import (
    resolve_environments,
    resolve_scoped_environments,
)

CATALOG = {
    "secque": {"input_data": "/data/secque.jsonl"},
    "financebench": {"input_data": "/data/financebench.jsonl"},
    "fabv1": {"input_data": "/data/fabv1.jsonl"},
}


# ---------------------------------------------------------------------------
# resolve_scoped_environments
# ---------------------------------------------------------------------------


def test_scoped_no_filter_returns_everything():
    assert resolve_scoped_environments({"environments": CATALOG}) == CATALOG


def test_scoped_string_filter_selects_one():
    got = resolve_scoped_environments({"environments": CATALOG, "_environment": "fabv1"})
    assert list(got) == ["fabv1"]


def test_scoped_list_filter_preserves_selection_order():
    got = resolve_scoped_environments(
        {"environments": CATALOG, "_environment": ["financebench", "secque"]}
    )
    assert list(got) == ["financebench", "secque"]


def test_scoped_drops_names_outside_this_stage():
    """A stage scoped to fabv1 selected with -e secque yields nothing, not an error."""
    got = resolve_scoped_environments(
        {"environments": {"fabv1": CATALOG["fabv1"]}, "_environment": ["secque", "fabv1"]}
    )
    assert list(got) == ["fabv1"]


def test_scoped_returns_empty_when_nothing_matches():
    got = resolve_scoped_environments(
        {"environments": {"fabv1": CATALOG["fabv1"]}, "_environment": "secque"}
    )
    assert got == {}


def test_strict_resolution_still_raises_on_unknown():
    """The GRPO helper keeps its fail-fast behaviour."""
    with pytest.raises(ValueError, match="Unknown environment 'nope'"):
        resolve_environments({"environments": CATALOG, "_environment": "nope"})
