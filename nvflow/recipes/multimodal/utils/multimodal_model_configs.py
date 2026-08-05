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
"""Typed YAML model configuration for HopChain stages."""

from __future__ import annotations

from nvflow.recipes.multimodal.utils.image_filter_models import VisionModelConfig


def resolve_model_config(stage_config: dict) -> VisionModelConfig:
    """Load a complete model configuration from the workflow YAML."""
    model_config = stage_config.get("model_config")
    if model_config is None:
        raise ValueError("Missing required field: model_config")
    return VisionModelConfig.model_validate(dict(model_config))
