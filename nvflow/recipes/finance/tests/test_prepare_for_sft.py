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
"""Tests for PrepareForSFTStage."""

import pytest

from nvflow.recipes.finance.stages.sft.prepare_for_sft import PrepareForSFTStage


class TestPrepareForSFTStage:
    """Test the PrepareForSFTStage functionality."""

    def test_stage_registration(self):
        """Test that the stage is properly registered."""
        stage = PrepareForSFTStage()
        # Stage is registered via decorator, class should be instantiable
        assert stage is not None

    def test_validate_config_success(self):
        """Test config validation with valid configuration."""
        stage = PrepareForSFTStage()
        config = {
            "input_dir": "/path/to/input",
            "output_dir": "/path/to/output",
        }
        # Should not raise any exception
        stage.validate_config(config)

    def test_validate_config_missing_input_dir(self):
        """Test config validation fails when input_dir is missing."""
        stage = PrepareForSFTStage()
        config = {
            "output_dir": "/path/to/output",
        }
        with pytest.raises(ValueError, match="'input_dir' is required"):
            stage.validate_config(config)

    def test_validate_config_missing_output_dir(self):
        """Test config validation fails when output_dir is missing."""
        stage = PrepareForSFTStage()
        config = {
            "input_dir": "/path/to/input",
        }
        with pytest.raises(ValueError, match="'output_dir' is required"):
            stage.validate_config(config)

    def test_validate_config_with_optional_fields(self):
        """Test config validation with optional fields."""
        stage = PrepareForSFTStage()
        config = {
            "input_dir": "/path/to/input",
            "output_dir": "/path/to/output",
            "prepare_data_kwargs": {"ctx_args": "++prompt_config=test"},
        }
        # Should not raise any exception
        stage.validate_config(config)
