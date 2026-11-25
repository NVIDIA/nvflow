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
"""Lazy, narrow NeMo-Skills command entrypoints used by NVFlow stages."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any


@dataclass
class ArgumentsContext:
    """Minimal Typer context consumed by NeMo-Skills command functions."""

    args: list[str]
    obj: None = None


@dataclass
class SubmissionCapture:
    """Authoritative handles returned by commands in one workflow submission."""

    _handles: dict[str, list[str]]

    def record(self, experiment: str, result: Any) -> None:
        handles = self._handles.setdefault(experiment, [])
        for handle in _submission_handles(result):
            if handle not in handles:
                handles.append(handle)

    def handles_by_experiment(self) -> dict[str, tuple[str, ...]]:
        return {name: tuple(handles) for name, handles in self._handles.items()}


_ACTIVE_CAPTURE: ContextVar[SubmissionCapture | None] = ContextVar(
    "nvflow_nemo_skills_submission_capture",
    default=None,
)


@contextmanager
def capture_submissions() -> Iterator[SubmissionCapture]:
    """Collect command results without reconstructing live experiments."""
    capture = SubmissionCapture(_handles={})
    token = _ACTIVE_CAPTURE.set(capture)
    try:
        yield capture
    finally:
        _ACTIVE_CAPTURE.reset(token)


def wrap_arguments(arguments: str) -> ArgumentsContext:
    """Match NeMo-Skills' whitespace-splitting programmatic CLI context."""
    return ArgumentsContext(args=[argument for argument in arguments.split(" ") if argument])


def generate(*args: Any, **kwargs: Any) -> Any:
    """Call only the NeMo-Skills generation leaf module."""
    from nemo_skills.pipeline.generate import generate as nemo_generate

    result = nemo_generate(*args, **kwargs)
    _record_submission(kwargs, result)
    return result


def run_cmd(*args: Any, **kwargs: Any) -> Any:
    """Call only the NeMo-Skills command leaf module."""
    from nemo_skills.pipeline.run_cmd import run_cmd as nemo_run_cmd

    result = nemo_run_cmd(*args, **kwargs)
    _record_submission(kwargs, result)
    return result


def _record_submission(kwargs: dict[str, Any], result: Any) -> None:
    capture = _ACTIVE_CAPTURE.get()
    experiment = kwargs.get("expname")
    if capture is not None and isinstance(experiment, str) and experiment:
        capture.record(experiment, result)


def _submission_handles(result: Any) -> tuple[str, ...]:
    if result is None:
        return ()
    if isinstance(result, list | tuple):
        return tuple(str(handle) for handle in result if handle)
    jobs = getattr(result, "jobs", ())
    if not isinstance(jobs, list | tuple):
        return ()
    handles: list[str] = []
    for job in jobs:
        handle = getattr(job, "handle", None)
        if handle:
            handles.append(str(handle))
        group_handles = getattr(job, "handles", ())
        if isinstance(group_handles, list | tuple):
            handles.extend(str(item) for item in group_handles if item)
    return tuple(dict.fromkeys(handles))
