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
"""Tests for core functionality."""

import pytest

from nvflow.core import BaseStage


def test_base_stage_exposes_only_exact_execution_api() -> None:
    assert "execute" not in BaseStage.__dict__
    assert "plan" not in BaseStage.__dict__
    assert "execution" in BaseStage.__dict__


def test_stage_validation():
    """Test stage config validation."""

    class ValidationStage(BaseStage):
        def execute(self, config, cluster, expname, run_after=None):
            pass

        def validate_config(self, config):
            if "required_field" not in config:
                raise ValueError("required_field is required")

    stage = ValidationStage()

    # Valid config
    stage.validate_config({"required_field": "value"})

    # Invalid config
    with pytest.raises(ValueError):
        stage.validate_config({})
