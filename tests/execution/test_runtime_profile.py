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
"""Unified runtime profile sourced from current NeMo-RL and NeMo-Gym main."""

import pytest

from nvflow.execution.runtime_profile import (
    PYTHON_SPEC,
    RAY_VERSION,
    RuntimeProfile,
    RuntimeProfileError,
)
from nvflow.execution.target import RegisteredCluster, TargetResolutionError


def test_current_nemo_runtime_profile_is_accepted() -> None:
    RuntimeProfile(python="3.13.14", ray="2.58.0").validate(location="test")

    assert PYTHON_SPEC == ">=3.13.14,<3.14"
    assert RAY_VERSION == "2.58.0"


@pytest.mark.parametrize(
    ("python", "ray"),
    [
        ("3.13.13", "2.58.0"),
        ("3.14.0", "2.58.0"),
        ("3.13.14", "2.56.1"),
    ],
)
def test_runtime_profile_rejects_version_skew(python: str, ray: str) -> None:
    with pytest.raises(RuntimeProfileError):
        RuntimeProfile(python=python, ray=ray).validate(location="test")


def test_supplied_cluster_rejects_ray_version_skew() -> None:
    with pytest.raises(TargetResolutionError, match="must declare Ray 2.58.0"):
        RegisteredCluster(
            name="cluster",
            jobs_url="http://127.0.0.1:8265",
            expected_python="3.13.14",
            expected_ray="2.56.1",
        )
