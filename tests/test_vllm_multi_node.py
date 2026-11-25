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
"""Which backend a vLLM server's own arguments choose."""

from __future__ import annotations

import pytest

from nvflow.core.vllm_multi_node import backend_in_args


@pytest.mark.parametrize(
    ("arguments", "backend"),
    [
        ("--max-model-len 4096", None),
        ("--distributed-executor-backend ray", "ray"),
        ("--distributed-executor-backend=mp --max-model-len 4096", "mp"),
    ],
)
def test_server_arguments_name_a_backend_in_either_spelling(
    arguments: str,
    backend: str | None,
) -> None:
    assert backend_in_args(arguments) == backend
