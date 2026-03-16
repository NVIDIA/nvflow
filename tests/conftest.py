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
"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary test data directory."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def test_config():
    """Sample test configuration."""
    return {
        "workflow": {"name": "test_workflow", "type": "test"},
        "cluster": "local",
        "pipeline_stages": ["test.stage1", "test.stage2"],
        "stages": {
            "test.stage1": {
                "input_dir": "/test/input",
                "output_dir": "/test/output",
                "dependencies": [],
            },
            "test.stage2": {
                "input_dir": "/test/output",
                "output_dir": "/test/final",
                "dependencies": ["test.stage1"],
            },
        },
    }
