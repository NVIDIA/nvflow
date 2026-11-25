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
"""Backend-neutral packaged source-root token."""

CODE_ROOT_TOKEN = "{{NVFLOW_CODE_ROOT}}"


def render_code_root(value: str, code_root: str) -> str:
    """Resolve the source-root token at engine/worker lowering."""
    return value.replace(CODE_ROOT_TOKEN, code_root)


__all__ = ["CODE_ROOT_TOKEN", "render_code_root"]
