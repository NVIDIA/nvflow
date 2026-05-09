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
"""NVFlow Recipes -- auto-discovered from subdirectories.

Each recipe is a self-contained implementation for a specific domain.
Add a new recipe by creating ``nvflow/recipes/<name>/`` with an
``__init__.py`` that imports its stages tree.  No manual edits to this
file are required.

A recipe whose import fails will be reported on stderr via the
``nvflow.core.discovery`` helper but will not block discovery of the
others.
"""

from pathlib import Path

from nvflow.core.discovery import import_stage_subpackages

import_stage_subpackages(__package__, Path(__file__).parent)
