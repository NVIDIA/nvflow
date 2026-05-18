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
"""Tests for ``nvflow.lib.sbatch``.

Covers the ``parse_extra_sbatch_args`` parser and the ``get_executor``
autopatch that plumbs cluster-level ``extra_sbatch_args`` into every
Slurm submission.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from nvflow.lib import sbatch
from nvflow.lib.sbatch import parse_extra_sbatch_args

# ---------------------------------------------------------------------------
# parse_extra_sbatch_args
# ---------------------------------------------------------------------------


def test_parse_value_form():
    cfg = {"extra_sbatch_args": ["--qos=xlarge_qos", "--partition=gb200"]}
    assert parse_extra_sbatch_args(cfg) == {"qos": "xlarge_qos", "partition": "gb200"}


def test_parse_boolean_form():
    cfg = {"extra_sbatch_args": ["--exclusive", "--requeue"]}
    assert parse_extra_sbatch_args(cfg) == {"exclusive": True, "requeue": True}


def test_parse_strips_dashes():
    cfg = {"extra_sbatch_args": ["---qos=xlarge_qos"]}
    assert parse_extra_sbatch_args(cfg) == {"qos": "xlarge_qos"}


def test_parse_skips_garbage_entries():
    cfg = {"extra_sbatch_args": ["", "--qos=xlarge_qos", None, 42, "---"]}  # type: ignore[list-item]
    assert parse_extra_sbatch_args(cfg) == {"qos": "xlarge_qos"}


@pytest.mark.parametrize(
    "cluster_config",
    [None, {}, {"extra_sbatch_args": None}, {"extra_sbatch_args": []}],  # type: ignore[dict-item]
)
def test_parse_empty(cluster_config):
    assert parse_extra_sbatch_args(cluster_config) == {}


_REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


def test_rollout_uses_shared_parser():
    """rollout.py must use parse_extra_sbatch_args, not an inline loop.

    Reads the file directly so the test does not require ``nemo_skills``
    (which ``rollout`` imports at module load) to be installed.
    """
    src = (_REPO_ROOT / "nvflow" / "lib" / "rl" / "rollout.py").read_text()
    assert "parse_extra_sbatch_args" in src


def test_autopatch_is_installed_by_workflow_runner_not_cli():
    """Autopatch must be in WorkflowRunner.run(), not CLI startup.

    Reads the files directly so the test does not require ``typer``
    (a CLI-only dep) to be installed.
    """
    cli_src = (_REPO_ROOT / "nvflow" / "cli" / "main.py").read_text()
    runner_src = (_REPO_ROOT / "nvflow" / "core" / "workflow_runner.py").read_text()

    assert "apply_sbatch_args_autopatch" not in cli_src
    assert "apply_sbatch_args_autopatch" in runner_src


# ---------------------------------------------------------------------------
# apply_sbatch_args_autopatch
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_nemo_skills(monkeypatch):
    """Minimal fake nemo_skills so the autopatch can target get_executor."""
    calls: list[dict[str, Any]] = []

    def _make(name: str, **attrs):
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
        return mod

    def _original_get_executor(cluster_config, *args, **kwargs):
        calls.append({"cluster_config": cluster_config, "args": args, "kwargs": kwargs})

    _make("nemo_skills")
    _make("nemo_skills.pipeline")
    utils_mod = _make("nemo_skills.pipeline.utils", get_executor=_original_get_executor)
    exp_mod = _make("nemo_skills.pipeline.utils.exp", get_executor=_original_get_executor)

    sbatch._reset_for_tests()
    try:
        yield exp_mod, utils_mod, calls
    finally:
        sbatch._reset_for_tests()
        for mod_name in [
            "nemo_skills.pipeline.utils.exp",
            "nemo_skills.pipeline.utils",
            "nemo_skills.pipeline",
            "nemo_skills",
        ]:
            sys.modules.pop(mod_name, None)


def test_autopatch_injects_qos_into_get_executor(fake_nemo_skills):
    exp_mod, _, calls = fake_nemo_skills

    sbatch.apply_sbatch_args_autopatch()

    exp_mod.get_executor(
        {"extra_sbatch_args": ["--qos=xlarge_qos"]},
        "container",
        num_nodes=1,
    )

    assert len(calls) == 1
    assert calls[0]["kwargs"]["sbatch_kwargs"] == {"qos": "xlarge_qos"}


def test_autopatch_existing_kwargs_win(fake_nemo_skills):
    exp_mod, _, calls = fake_nemo_skills

    sbatch.apply_sbatch_args_autopatch()

    exp_mod.get_executor(
        {"extra_sbatch_args": ["--qos=xlarge_qos", "--time=02:00:00"]},
        "container",
        sbatch_kwargs={"qos": "explicit_override"},
    )

    assert calls[0]["kwargs"]["sbatch_kwargs"] == {
        "qos": "explicit_override",
        "time": "02:00:00",
    }


def test_autopatch_noop_when_no_extras(fake_nemo_skills):
    exp_mod, _, calls = fake_nemo_skills

    sbatch.apply_sbatch_args_autopatch()

    exp_mod.get_executor({"extra_sbatch_args": []}, "container")

    assert len(calls) == 1
    assert calls[0]["kwargs"].get("sbatch_kwargs") is None


def test_autopatch_patches_utils_reexport(fake_nemo_skills):
    _, utils_mod, calls = fake_nemo_skills

    sbatch.apply_sbatch_args_autopatch()

    utils_mod.get_executor(
        {"extra_sbatch_args": ["--qos=test"]},
        "container",
    )

    assert len(calls) == 1
    assert calls[0]["kwargs"]["sbatch_kwargs"] == {"qos": "test"}


def test_autopatch_is_idempotent(fake_nemo_skills):
    exp_mod, utils_mod, _ = fake_nemo_skills

    sbatch.apply_sbatch_args_autopatch()
    first = (exp_mod.get_executor, utils_mod.get_executor)

    sbatch.apply_sbatch_args_autopatch()
    second = (exp_mod.get_executor, utils_mod.get_executor)

    assert first == second


def test_autopatch_skips_when_nemo_skills_absent():
    """No crash when nemo-skills isn't importable."""
    sbatch._reset_for_tests()

    for mod_name in list(sys.modules):
        if mod_name == "nemo_skills" or mod_name.startswith("nemo_skills."):
            sys.modules[mod_name] = None  # type: ignore[assignment]

    sbatch.apply_sbatch_args_autopatch()  # must not raise
