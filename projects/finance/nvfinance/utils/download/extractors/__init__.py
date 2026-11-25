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
"""SEC Extractors for different form types."""

from .base import BaseSecExtractor
from .extractor_8k import Sec8KExtractor
from .extractor_10k import Sec10KExtractor
from .extractor_10q import Sec10QExtractor

__all__ = [
    "BaseSecExtractor",
    "Sec10KExtractor",
    "Sec10QExtractor",
    "Sec8KExtractor",
]
