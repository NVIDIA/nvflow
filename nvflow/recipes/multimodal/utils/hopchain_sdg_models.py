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
"""Typed data contracts for the HopChain SDG workflow."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from nvflow.recipes.multimodal.utils.image_filter_models import (
    ComplexObjectAnalysis,
    GenerationStats,
)


class FilteredImageInput(BaseModel):
    """Normalized handoff from image filtering into HopChain SDG."""

    image_id: str
    image_file_name: str
    image_directory: str
    image_path: str
    filter_complexity_score: int = Field(ge=1, le=10)
    filter_quality_rating: Literal["High", "Medium", "Low"]
    filter_complex_objects: list[ComplexObjectAnalysis] = Field(default_factory=list)
    filter_generalized_name_hints: list[str] = Field(default_factory=list)
    filter_analysis: str
    filter_should_keep: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class CategoryLocalizationTarget(BaseModel):
    """Canonical category plus concrete phrases to try during localization."""

    category: str
    localization_phrases: list[str] = Field(default_factory=list)


class CategoryIdentificationRecord(BaseModel):
    """Semantic category identification output for one image."""

    image_id: str
    image_file_name: str
    image_directory: str
    image_path: str
    prior_generalized_name_hints: list[str] = Field(default_factory=list)
    identified_categories: list[str] = Field(default_factory=list)
    localization_targets: list[CategoryLocalizationTarget] = Field(default_factory=list)
    raw_generation: str
    generation_stats: GenerationStats = Field(default_factory=GenerationStats)


class BoundingBoxNorm1000(BaseModel):
    """Bounding box in a normalized 0-1000 coordinate system."""

    x1: int = Field(ge=0, le=1000)
    y1: int = Field(ge=0, le=1000)
    x2: int = Field(ge=0, le=1000)
    y2: int = Field(ge=0, le=1000)

    @model_validator(mode="after")
    def validate_order(self) -> BoundingBoxNorm1000:
        """Ensure the normalized box has positive area."""
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("BoundingBoxNorm1000 requires x2 > x1 and y2 > y1")
        return self


class BoundingBoxPixels(BaseModel):
    """Bounding box in absolute pixel coordinates."""

    x1: int = Field(ge=0)
    y1: int = Field(ge=0)
    x2: int = Field(ge=0)
    y2: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_order(self) -> BoundingBoxPixels:
        """Ensure the pixel box has positive area."""
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("BoundingBoxPixels requires x2 > x1 and y2 > y1")
        return self


class CropContext(BaseModel):
    """Extra crop metadata for a localized instance."""

    padding_ratio: float = Field(default=0.0, ge=0.0)
    image_width: int = Field(ge=1)
    image_height: int = Field(ge=1)


class LocalizedInstance(BaseModel):
    """One concrete localized object instance."""

    instance_id: str
    category: str
    object_name: str
    bbox_xyxy: BoundingBoxPixels
    bbox_norm_1000: BoundingBoxNorm1000
    mask_path: str | None = None
    crop_path: str
    crop_bbox_context: CropContext | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LocalizedInstanceRecord(BaseModel):
    """Localization output for one image."""

    image_id: str
    image_file_name: str
    image_directory: str
    image_path: str
    localizer_backend: str
    identified_categories: list[str] = Field(default_factory=list)
    instances: list[LocalizedInstance] = Field(default_factory=list)
    raw_generation: str
    generation_stats: GenerationStats = Field(default_factory=GenerationStats)


class InstanceCombinationRecord(BaseModel):
    """A sampled combination of instances for query generation."""

    image_id: str
    image_file_name: str
    image_directory: str
    image_path: str
    combination_id: str
    instance_ids: list[str] = Field(min_length=3)
    instances: list[LocalizedInstance] = Field(min_length=3)
    combination_size: int
    sampling_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("instances")
    @classmethod
    def validate_instances(cls, value: list[LocalizedInstance]) -> list[LocalizedInstance]:
        """Require consistent instance IDs inside a combination record."""
        if len({instance.instance_id for instance in value}) != len(value):
            raise ValueError("InstanceCombinationRecord instances must be unique")
        return value

    @model_validator(mode="after")
    def validate_lengths(self) -> InstanceCombinationRecord:
        """Ensure declared size matches actual content."""
        if (
            len(self.instance_ids) != self.combination_size
            or len(self.instances) != self.combination_size
        ):
            raise ValueError("combination_size must match instance_ids and instances length")
        return self


class ReasoningHop(BaseModel):
    """One structured hop emitted by query generation."""

    hop_number: int = Field(ge=1)
    hop_type: str
    from_instance: str | None = None
    to_instance: str | None = None
    description: str
    objects_involved: list[str] = Field(default_factory=list)
    output: str


class GeneratedHopChainQueryMetadata(BaseModel):
    """Internal-only metadata from query generation."""

    primary_capability: str
    instance_chain: str
    reasoning_hops: list[ReasoningHop] = Field(default_factory=list)
    design_rationale: str
    answer_type: Literal["numeric", "count", "arithmetic", "other"] = "numeric"
    uses_all_instances: bool = False
    generator_prompt_version: str | None = None


class GeneratedHopChainQuery(BaseModel):
    """Candidate query synthesized for a specific instance combination."""

    query_id: str
    image_id: str
    combination_id: str
    image_file_name: str
    image_fullpath: str
    question: str
    hypothetical_answer: str
    involved_instance_ids: list[str] = Field(default_factory=list)
    hop_count: int = Field(ge=1)
    query_metadata: GeneratedHopChainQueryMetadata
    raw_generation: str
    generation_stats: GenerationStats = Field(default_factory=GenerationStats)


class VerificationMetadata(BaseModel):
    """Machine verification metadata for a candidate query."""

    numeric_answer: bool = False
    forbidden_reference_terms: list[str] = Field(default_factory=list)
    instance_identifier_terms: list[str] = Field(default_factory=list)
    references_all_instances: bool = False
    verified_by: str = "machine_rules"


class VerifiedHopChainQuery(GeneratedHopChainQuery):
    """Candidate query after machine-side verification."""

    verification_status: Literal["accepted", "rejected"]
    rejection_reasons: list[str] = Field(default_factory=list)
    verification_metadata: VerificationMetadata


class LLMJudgeAnswer(BaseModel):
    """One LLM judge's answer for a candidate query."""

    judge_name: str
    provider: str
    model: str
    answer: str
    normalized_answer: str
    confidence: Literal["high", "medium", "low", "unknown"] = "unknown"
    reasoning: str | None = None
    raw_response: str
    generation_stats: GenerationStats = Field(default_factory=GenerationStats)


class LLMJudgeEvaluationRecord(VerifiedHopChainQuery):
    """Candidate query annotated with one LLM judge output."""

    llm_judge: LLMJudgeAnswer
    judge_status: Literal["parsed", "parse_error", "api_error"] = "parsed"
    judge_error: str | None = None


class LLMJudgeAnswerSummary(BaseModel):
    """Compact judge answer summary carried into reconciled outputs."""

    judge_name: str
    provider: str
    model: str
    answer: str
    normalized_answer: str


class ReconciledHopChainQuery(VerifiedHopChainQuery):
    """Candidate query after cross-judge reconciliation."""

    llm_judge_names: list[str] = Field(default_factory=list)
    llm_judge_answers: list[LLMJudgeAnswerSummary] = Field(default_factory=list)
    llm_judge_consensus_answer: str
    llm_judge_consensus_normalized_answer: str
    llm_judge_consensus_matches_hypothetical_answer: bool = False
    llm_judge_reconciliation_status: Literal["accepted", "rejected"]
    llm_judge_rejection_reasons: list[str] = Field(default_factory=list)


class SFTReasoningTraceCandidate(BaseModel):
    """One model-generated reasoning trace candidate for SFT."""

    query_id: str
    sft_sample_id: str
    sample_index: int = Field(ge=0)
    k: int = Field(ge=1)
    model: str
    question: str
    hypothetical_answer: str
    final_answer: str
    normalized_hypothetical_answer: str
    normalized_final_answer: str
    answer_matches_hypothetical: bool
    reasoning_trace: str
    raw_generation: str
    source_record: dict[str, Any] = Field(default_factory=dict)
    generation_stats: GenerationStats = Field(default_factory=GenerationStats)


class SFTTraceJudgeDecision(BaseModel):
    """Parsed LLM-judge decision for one SFT reasoning trace candidate."""

    verdict: Literal["pass", "fail"]
    reasoning: str = ""
    raw_response: str
    generation_stats: GenerationStats = Field(default_factory=GenerationStats)
