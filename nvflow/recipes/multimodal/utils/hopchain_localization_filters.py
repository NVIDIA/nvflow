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
"""Shared filtering helpers for custom localization backends."""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class MinImageSizeFilterSpec(BaseModel):
    """Filter out localized crops below a minimum width or height."""

    filter_method: Literal["min_image_size"]
    min_w: int = Field(ge=1)
    min_h: int = Field(ge=1)


LocalizationFilterSpec = Annotated[
    MinImageSizeFilterSpec,
    Field(discriminator="filter_method"),
]


def parse_filter_list(filter_list_json: str | None) -> list[LocalizationFilterSpec]:
    """Parse filter configuration from a JSON-encoded CLI argument."""
    if not filter_list_json:
        return []
    raw_value = json.loads(filter_list_json)
    if not isinstance(raw_value, list):
        raise ValueError("filter_list must decode to a list")
    return [MinImageSizeFilterSpec.model_validate(item) for item in raw_value]


def evaluate_localization_filters(
    filters: list[LocalizationFilterSpec],
    *,
    width: int,
    height: int,
) -> str | None:
    """Return the first matching filter method, or None if the instance is kept."""
    for filter_spec in filters:
        if isinstance(filter_spec, MinImageSizeFilterSpec):
            min_w = filter_spec.min_w
            min_h = filter_spec.min_h
            if width < min_w or height < min_h:
                return filter_spec.filter_method
            continue
        raise ValueError(f"Unsupported localization filter method: {filter_spec.filter_method}")
    return None
