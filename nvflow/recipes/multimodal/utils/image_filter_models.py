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
"""Typed models for HopChain image filtering."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator


class ImageDirectorySpec(BaseModel):
    """One image directory entry from the stage input catalog."""

    directory: str = Field(description="Absolute path to a directory containing images")
    start_index: int | None = Field(
        default=None, ge=0, description="Optional inclusive start index"
    )
    end_index: int | None = Field(default=None, ge=0, description="Optional exclusive end index")
    recursive: bool = Field(default=True, description="Recursively scan subdirectories for images")
    shuffle_seed: int | None = Field(
        default=None,
        description="Optional deterministic seed used to shuffle discovered images before slicing",
    )

    @model_validator(mode="after")
    def validate_slice(self) -> ImageDirectorySpec:
        """Ensure the configured image slice is valid."""
        if (
            self.start_index is not None
            and self.end_index is not None
            and self.end_index <= self.start_index
        ):
            raise ValueError("end_index must be greater than start_index when both are provided")
        return self


class ImageDirectoryCatalog(RootModel[list[ImageDirectorySpec]]):
    """Root model for the input JSON catalog."""


class SelectedImageRecord(BaseModel):
    """One selected image entry written to `image_catalog.jsonl`."""

    image_file_name: str
    image_directory: str
    catalog_entry_index: int
    source_image_index: int
    start_index: int | None = None
    end_index: int | None = None
    shuffle_seed: int | None = None


class ComplexObjectAnalysis(BaseModel):
    """Complexity details for one object or scene region."""

    object_name: str
    generalized_name: str
    reason_for_complexity: list[str] = Field(default_factory=list)


class ImageFilterGeneration(BaseModel):
    """Parsed model output for one image."""

    overall_complexity_score: int = Field(ge=1, le=10)
    overall_quality_rating: Literal["High", "Medium", "Low"]
    complexity_analysis: str
    complex_objects: list[ComplexObjectAnalysis] = Field(default_factory=list)


class GenerationStats(BaseModel):
    """Selected runtime stats from the inference backend."""

    num_generated_tokens: int | None = None
    generation_time: float | None = None


class RawInferenceOutput(BaseModel):
    """Raw model response payload preserved for re-thresholding or reparsing."""

    generation: str
    full_generation: str | None = None
    finish_reason: str | None = None
    reasoning_content: str | None = None


class ImageFilterRecord(BaseModel):
    """Final postprocessed record for one image."""

    image_file_name: str
    image_directory: str
    catalog_entry_index: int
    source_image_index: int
    start_index: int | None = None
    end_index: int | None = None
    overall_complexity_score: int
    overall_quality_rating: Literal["High", "Medium", "Low"]
    complexity_analysis: str
    complex_objects: list[ComplexObjectAnalysis] = Field(default_factory=list)
    should_keep: bool
    filter_reason: str
    generation_stats: GenerationStats = Field(default_factory=GenerationStats)
    raw_inference: RawInferenceOutput


class VisionModelConfig(BaseModel):
    """Minimal model configuration needed for nemo-skills generation."""

    model_config = ConfigDict(extra="forbid")

    model: str
    server_type: str
    server_gpus: int
    server_nodes: int
    server_args: str
    max_image_dimension: int | None = None
    tokens_to_generate: int = 2048
    max_concurrent_requests: int = 32
    time_min: int = 45
    use_base64_images: bool = False
