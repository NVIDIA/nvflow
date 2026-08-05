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
"""Auto-discover all multimodal stages from subdirectories."""

import importlib
from pathlib import Path

_current_dir = Path(__file__).parent
for subdir in _current_dir.iterdir():
    if subdir.is_dir() and not subdir.name.startswith("_"):
        try:
            importlib.import_module(f".{subdir.name}", package=__package__)
        except ImportError:
            pass
