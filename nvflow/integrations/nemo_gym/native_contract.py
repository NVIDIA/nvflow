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
"""Portable, dependency-free contracts for the native NeMo Gym interface."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Final

NATIVE_GYM_SOURCE_ROOT: Final[str] = "/opt/nemo-gym"
GYM_DEFAULT_ROLLOUT_ATTEMPTS: Final[int] = 3

_PathLike = str | os.PathLike[str]


def _output_path(output_jsonl: _PathLike) -> PurePosixPath:
    return PurePosixPath(output_jsonl)


def materialized_inputs_path_for(output_jsonl: _PathLike) -> PurePosixPath:
    """Return Gym's materialized-input sidecar for a rollout output."""
    output = _output_path(output_jsonl)
    return output.with_stem(f"{output.stem}_materialized_inputs").with_suffix(".jsonl")


def failures_path_for(output_jsonl: _PathLike) -> PurePosixPath:
    """Return Gym's failed-rollout sidecar for a rollout output."""
    output = _output_path(output_jsonl)
    return output.with_name(f"{output.stem}_failures.jsonl")


def aggregate_metrics_path_for(output_jsonl: _PathLike) -> PurePosixPath:
    """Return Gym's aggregate-metrics artifact for a rollout output."""
    output = _output_path(output_jsonl)
    return output.with_stem(f"{output.stem}_aggregate_metrics").with_suffix(".json")


def reward_profiling_path_for(output_jsonl: _PathLike) -> PurePosixPath:
    """Return Gym's reward-profiling artifact for a rollout output."""
    output = _output_path(output_jsonl)
    return output.with_stem(f"{output.stem}_reward_profiling").with_suffix(".jsonl")


def agent_metrics_path_for(output_jsonl: _PathLike) -> PurePosixPath:
    """Return Gym's agent-level reward-profile metrics artifact."""
    output = _output_path(output_jsonl)
    return output.with_stem(f"{output.stem}_agent_metrics").with_suffix(".json")


def repeat_level_metrics_path_for(output_jsonl: _PathLike) -> PurePosixPath:
    """Return Gym's repeat-level reward-profile metrics artifact."""
    output = _output_path(output_jsonl)
    return output.with_stem(f"{output.stem}_repeat_level_metrics").with_suffix(".json")


def quality_summary_path_for(output_jsonl: _PathLike) -> PurePosixPath:
    """Return Gym's run-scoped rollout-health summary artifact."""
    return _output_path(output_jsonl).parent / "quality_summary.json"


def rollout_verdicts_path_for(output_jsonl: _PathLike) -> PurePosixPath:
    """Return Gym's run-scoped per-rollout verdict artifact."""
    return _output_path(output_jsonl).parent / "rollout_verdicts.jsonl"


def rollout_attempt_limit(dependent_jobs: int) -> int:
    """Preserve Gym's retry floor while covering every configured resume job."""
    return max(GYM_DEFAULT_ROLLOUT_ATTEMPTS, dependent_jobs + 1)


@dataclass(frozen=True, slots=True)
class NativeGymArtifactPaths:
    """Canonical artifact set rooted at one rollout output."""

    rollouts: PurePosixPath
    materialized_inputs: PurePosixPath
    failures: PurePosixPath
    aggregate_metrics: PurePosixPath
    reward_profiling: PurePosixPath
    agent_metrics: PurePosixPath
    repeat_level_metrics: PurePosixPath
    quality_summary: PurePosixPath
    rollout_verdicts: PurePosixPath
    completion: PurePosixPath

    def as_json_dict(self) -> dict[str, str]:
        """Return a stable, JSON-native representation."""
        return {
            "rollouts": self.rollouts.as_posix(),
            "materialized_inputs": self.materialized_inputs.as_posix(),
            "failures": self.failures.as_posix(),
            "aggregate_metrics": self.aggregate_metrics.as_posix(),
            "reward_profiling": self.reward_profiling.as_posix(),
            "agent_metrics": self.agent_metrics.as_posix(),
            "repeat_level_metrics": self.repeat_level_metrics.as_posix(),
            "quality_summary": self.quality_summary.as_posix(),
            "rollout_verdicts": self.rollout_verdicts.as_posix(),
            "completion": self.completion.as_posix(),
        }


def artifact_paths_for(
    output_jsonl: _PathLike,
    completion_json: _PathLike,
) -> NativeGymArtifactPaths:
    """Return every canonical artifact path for a native Gym rollout."""
    output = _output_path(output_jsonl)
    return NativeGymArtifactPaths(
        rollouts=output,
        materialized_inputs=materialized_inputs_path_for(output),
        failures=failures_path_for(output),
        aggregate_metrics=aggregate_metrics_path_for(output),
        reward_profiling=reward_profiling_path_for(output),
        agent_metrics=agent_metrics_path_for(output),
        repeat_level_metrics=repeat_level_metrics_path_for(output),
        quality_summary=quality_summary_path_for(output),
        rollout_verdicts=rollout_verdicts_path_for(output),
        completion=PurePosixPath(completion_json),
    )


@dataclass(frozen=True, slots=True)
class NativeGymOutputLayout:
    """Job-oriented artifact paths for one environment output cell."""

    root: str
    initial_job_name: str

    @classmethod
    def collection(cls, root: _PathLike) -> NativeGymOutputLayout:
        return cls(PurePosixPath(root).as_posix(), "job-0-collect-rollouts")

    @classmethod
    def rejudge(cls, root: _PathLike) -> NativeGymOutputLayout:
        return cls(PurePosixPath(root).as_posix(), "job-0-rejudge-rollouts")

    @property
    def root_path(self) -> PurePosixPath:
        return PurePosixPath(self.root)

    @property
    def jobs_dir(self) -> PurePosixPath:
        return self.root_path / "jobs"

    @property
    def initial_job_dir(self) -> PurePosixPath:
        return self.jobs_dir / self.initial_job_name

    @property
    def initial_outputs_dir(self) -> PurePosixPath:
        return self.initial_job_dir / "outputs"

    @property
    def initial_logs_dir(self) -> PurePosixPath:
        return self.initial_job_dir / "logs"

    @property
    def initial_artifacts(self) -> NativeGymArtifactPaths:
        return artifact_paths_for(
            self.initial_outputs_dir / "rollouts.jsonl",
            self.initial_job_dir / "completion.json",
        )

    @property
    def initial_behavior_manifest(self) -> PurePosixPath:
        return self.initial_job_dir / "behavior.json"

    def chunk_dir(self, chunk_index: int) -> PurePosixPath:
        return self.initial_job_dir / "chunks" / f"chunk-{chunk_index}"

    def chunk_outputs_dir(self, chunk_index: int) -> PurePosixPath:
        return self.chunk_dir(chunk_index) / "outputs"

    def chunk_artifacts(self, chunk_index: int) -> NativeGymArtifactPaths:
        chunk = self.chunk_dir(chunk_index)
        return artifact_paths_for(
            chunk / "outputs" / "rollouts.jsonl",
            chunk / "completion.json",
        )

    def chunk_manifest(self, chunk_index: int) -> PurePosixPath:
        return self.chunk_dir(chunk_index) / "manifest.json"

    def attempt_logs_dir(self, chunk_index: int, attempt: int) -> PurePosixPath:
        return self.chunk_dir(chunk_index) / "attempts" / f"attempt-{attempt}" / "logs"

    def rejudge_attempt_logs_dir(self, attempt: int) -> PurePosixPath:
        return self.initial_job_dir / "attempts" / f"attempt-{attempt}" / "logs"

    @property
    def analyze_job_dir(self) -> PurePosixPath:
        return self.jobs_dir / "job-1-analyze-rollouts"

    @property
    def analyze_outputs_dir(self) -> PurePosixPath:
        return self.analyze_job_dir / "outputs"

    @property
    def analyze_logs_dir(self) -> PurePosixPath:
        return self.analyze_job_dir / "logs"

    @property
    def analyze_analysis_dir(self) -> PurePosixPath:
        return self.analyze_job_dir / "analysis"

    @property
    def analyzed_artifacts(self) -> NativeGymArtifactPaths:
        return artifact_paths_for(
            self.analyze_outputs_dir / "rollouts.jsonl",
            self.analyze_job_dir / "completion.json",
        )

    def analyzed_completion_artifacts(
        self,
        *,
        include_aggregate: bool = True,
        include_profiles: bool = False,
        include_evaluation: bool = False,
        include_blade: bool = False,
        include_compare: bool = False,
    ) -> dict[str, PurePosixPath]:
        artifacts = self.analyzed_artifacts
        result = {
            "failures": artifacts.failures,
            "quality_summary": artifacts.quality_summary,
            "rollout_verdicts": artifacts.rollout_verdicts,
        }
        if include_aggregate:
            result["aggregate_metrics"] = artifacts.aggregate_metrics
        if include_profiles:
            result.update(
                {
                    "reward_profiles": artifacts.reward_profiling,
                    "agent_metrics": artifacts.agent_metrics,
                    "repeat_metrics": artifacts.repeat_level_metrics,
                }
            )
        if include_evaluation:
            result["evaluation_analysis"] = self.evaluation_analysis_dir
        if include_blade:
            result["blade_analysis"] = self.blade_dir
        if include_compare:
            result.update(
                {
                    "compare_json": self.compare_analysis_dir / "compare_report.json",
                    "compare_markdown": self.compare_analysis_dir / "compare_report.md",
                }
            )
        return result

    @property
    def analyzed_artifact_catalog(self) -> dict[str, PurePosixPath]:
        artifacts = self.analyzed_artifacts
        return {
            "failures": artifacts.failures,
            "aggregate_metrics": artifacts.aggregate_metrics,
            "reward_profiles": artifacts.reward_profiling,
            "agent_metrics": artifacts.agent_metrics,
            "repeat_metrics": artifacts.repeat_level_metrics,
            "quality_summary": artifacts.quality_summary,
            "rollout_verdicts": artifacts.rollout_verdicts,
            "evaluation_analysis": self.evaluation_analysis_dir,
            "blade_analysis": self.blade_dir,
            "compare_json": self.compare_analysis_dir / "compare_report.json",
            "compare_markdown": self.compare_analysis_dir / "compare_report.md",
        }

    def resolve_analyzed_artifacts(
        self,
        names: Iterable[str],
    ) -> dict[str, PurePosixPath]:
        catalog = self.analyzed_artifact_catalog
        unknown = sorted(set(names) - set(catalog))
        if unknown:
            raise ValueError("unknown analyzed completion artifact(s): " + ", ".join(unknown))
        return {name: catalog[name] for name in sorted(set(names))}

    @property
    def blade_dir(self) -> PurePosixPath:
        return self.analyze_analysis_dir / "blade"

    @property
    def evaluation_analysis_dir(self) -> PurePosixPath:
        return self.analyze_analysis_dir / "evaluation"

    @property
    def compare_analysis_dir(self) -> PurePosixPath:
        return self.analyze_analysis_dir / "compare"

    @property
    def dataset_job_dir(self) -> PurePosixPath:
        return self.jobs_dir / "job-2-build-training-dataset"

    @property
    def dataset_logs_dir(self) -> PurePosixPath:
        return self.dataset_job_dir / "logs"

    @property
    def dataset_diagnostics_dir(self) -> PurePosixPath:
        return self.dataset_job_dir / "diagnostics"

    @property
    def profiled_dataset(self) -> PurePosixPath:
        return self.root_path / "profiled.jsonl"

    @property
    def training_dataset(self) -> PurePosixPath:
        return self.root_path / "train.jsonl"

    @property
    def dataset_selection_report(self) -> PurePosixPath:
        return self.dataset_job_dir / "selection_report.json"

    @property
    def dataset_completion(self) -> PurePosixPath:
        return self.dataset_job_dir / "completion.json"

    @property
    def unprofiled_samples(self) -> PurePosixPath:
        return self.dataset_diagnostics_dir / "unprofiled_samples.jsonl"

    @property
    def legacy_paths(self) -> tuple[PurePosixPath, ...]:
        """Flat-layout paths that must not be mixed with this contract."""
        return (
            self.root_path / "chunks",
            self.root_path / "logs",
            self.root_path / "state",
            self.root_path / "analysis",
            self.root_path / "rollouts.jsonl",
            self.root_path / "rollouts_materialized_inputs.jsonl",
            self.root_path / "rollouts.jsonl.done",
            self.root_path / "train.jsonl.done",
            self.root_path / "profile_selection_report.json",
        )


__all__ = [
    "GYM_DEFAULT_ROLLOUT_ATTEMPTS",
    "NATIVE_GYM_SOURCE_ROOT",
    "NativeGymArtifactPaths",
    "NativeGymOutputLayout",
    "agent_metrics_path_for",
    "aggregate_metrics_path_for",
    "artifact_paths_for",
    "failures_path_for",
    "materialized_inputs_path_for",
    "quality_summary_path_for",
    "reward_profiling_path_for",
    "repeat_level_metrics_path_for",
    "rollout_attempt_limit",
    "rollout_verdicts_path_for",
]
