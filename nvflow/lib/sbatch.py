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
"""Plumb ``cluster_config['extra_sbatch_args']`` into every Slurm submission.

Cluster configs may declare global Slurm flags via ``extra_sbatch_args``
(e.g. ``--qos=xlarge_qos``, ``--exclusive``).  These must reach the
generated sbatch script for every stage — SDG, SFT, eval, and GRPO.

This module patches ``nemo_skills.pipeline.utils.exp.get_executor``, the
single funnel through which all ``SlurmExecutor`` instances are created.
One wrapper, two rebindings (``exp`` module + ``utils`` package re-export),
complete coverage.

:func:`parse_extra_sbatch_args` is also used by ``nvflow/lib/rl/rollout.py``
which previously inlined the same parsing loop.
"""

from __future__ import annotations

import functools
from typing import Any

__all__ = [
    "apply_sbatch_args_autopatch",
    "parse_extra_sbatch_args",
]


def parse_extra_sbatch_args(cluster_config: dict | None) -> dict[str, Any]:
    """Parse ``extra_sbatch_args`` list into a kwargs dict.

    Accepts entries like ``--qos=xlarge_qos`` (value) or ``--exclusive``
    (boolean flag).  Leading dashes are stripped.  Empty / non-string
    entries are skipped so malformed YAML doesn't crash submission.
    """
    if not cluster_config:
        return {}
    out: dict[str, Any] = {}
    for arg in cluster_config.get("extra_sbatch_args") or []:
        if not isinstance(arg, str):
            continue
        key, sep, val = arg.lstrip("-").partition("=")
        if not key:
            continue
        out[key] = val if sep else True
    return out


_PATCHED = False


def apply_sbatch_args_autopatch() -> None:
    """Wrap ``get_executor`` so every Slurm submission honours ``extra_sbatch_args``.

    Patches both the home module (``exp``) and the ``utils`` package
    re-export so callers that import via either path see the wrapper.

    Safe to call repeatedly; subsequent calls are no-ops.  Silently does
    nothing if ``nemo-skills`` isn't importable.
    """
    global _PATCHED
    if _PATCHED:
        return

    try:
        import nemo_skills.pipeline.utils as utils_mod
        import nemo_skills.pipeline.utils.exp as exp_mod
    except ImportError:
        return

    _PATCHED = True

    _original_get_executor = exp_mod.get_executor

    @functools.wraps(_original_get_executor)
    def _patched_get_executor(cluster_config, *args, **kwargs):
        extras = parse_extra_sbatch_args(cluster_config)
        if extras:
            existing = kwargs.get("sbatch_kwargs") or {}
            merged = {**extras, **existing}
            kwargs["sbatch_kwargs"] = merged
        return _original_get_executor(cluster_config, *args, **kwargs)

    exp_mod.get_executor = _patched_get_executor
    utils_mod.get_executor = _patched_get_executor


def _reset_for_tests() -> None:
    """Test-only hook: clear the idempotency flag and undo monkey-patches."""
    global _PATCHED
    if not _PATCHED:
        return
    try:
        import nemo_skills.pipeline.utils as utils_mod
        import nemo_skills.pipeline.utils.exp as exp_mod

        for mod in (exp_mod, utils_mod):
            wrapped = getattr(mod, "get_executor", None)
            original = getattr(wrapped, "__wrapped__", None)
            if original is not None:
                mod.get_executor = original
    except ImportError:
        pass
    _PATCHED = False
