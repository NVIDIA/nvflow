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
"""Tests for eval-prep benchmark filtering (air-gap safety).

``stages.prepare_data.dataset_names`` in ``eval/base.yaml`` is independent
of the recipe's ``benchmarks`` block, so a benchmark disabled in the recipe
(``financebench: null``) would still be fetched at prep time — and its
``prepare.py`` runs an unguarded ``load_dataset()`` that fails under air-gap.
The evaluator must intersect the prep list with the *enabled* benchmarks so
disabled ones are never prepared.
"""

from nvflow.recipes.finance.stages.evaluation import evaluate as ev
from nvflow.recipes.finance.stages.evaluation.evaluate import (
    EmbeddedEvalStage,
    _enabled_benchmark_names,
)


class TestEnabledBenchmarkNames:
    def test_dict_filters_null_entries(self):
        cfg = {"secque": {"seeds": 5}, "financebench": None}
        assert _enabled_benchmark_names(cfg) == {"secque"}

    def test_dict_all_enabled(self):
        cfg = {"secque": {"seeds": 5}, "financebench": {"seeds": 1}}
        assert _enabled_benchmark_names(cfg) == {"secque", "financebench"}

    def test_preformatted_list(self):
        assert _enabled_benchmark_names(["secque:5", "finqa:1"]) == {"secque", "finqa"}

    def test_empty_and_unknown_types(self):
        assert _enabled_benchmark_names({}) == set()
        assert _enabled_benchmark_names(None) == set()  # type: ignore[arg-type]


class _CapturingStage:
    """Stand-in for PrepareFinanceBenchmarksStage that records its config."""

    captured: dict | None = None

    def execute(self, config, cluster, expname, run_after=None):
        type(self).captured = config


class TestPrepFilter:
    BASE = {
        "stages": {
            "prepare_data": {
                "dataset_names": ["secque", "financebench"],
                "output_dir": "/workspace/datasets",
            }
        }
    }

    def _run(self, monkeypatch, enabled):
        _CapturingStage.captured = None
        monkeypatch.setattr(
            "nvflow.recipes.finance.stages.evaluation.prepare_data.PrepareFinanceBenchmarksStage",
            _CapturingStage,
        )
        # _prepare_benchmark_data does not use `self`; a dummy receiver is fine.
        return EmbeddedEvalStage._prepare_benchmark_data(
            None,
            base_config=self.BASE,
            cluster="c",
            expname="grpo-eval",
            enabled_benchmarks=enabled,
        )

    def test_disabled_benchmark_dropped_from_prep(self, monkeypatch):
        """financebench:null in the recipe excludes it from the prep list."""
        expname = self._run(monkeypatch, {"secque"})
        assert expname == "grpo-eval-prepare-data"
        assert _CapturingStage.captured["dataset_names"] == ["secque"]

    def test_all_enabled_keeps_full_list(self, monkeypatch):
        self._run(monkeypatch, {"secque", "financebench"})
        assert _CapturingStage.captured["dataset_names"] == ["secque", "financebench"]

    def test_none_enabled_skips_submission(self, monkeypatch):
        """No enabled benchmark to prepare → no job submitted, empty expname."""
        expname = self._run(monkeypatch, set())
        assert expname == ""
        assert _CapturingStage.captured is None

    def test_no_filter_when_enabled_is_none(self, monkeypatch):
        """Backward compat: enabled_benchmarks=None → original full list."""
        _CapturingStage.captured = None
        monkeypatch.setattr(
            "nvflow.recipes.finance.stages.evaluation.prepare_data.PrepareFinanceBenchmarksStage",
            _CapturingStage,
        )
        EmbeddedEvalStage._prepare_benchmark_data(
            None,
            base_config=self.BASE,
            cluster="c",
            expname="grpo-eval",
            enabled_benchmarks=None,
        )
        # prep_config passed through unmodified (dataset_names untouched)
        assert _CapturingStage.captured["dataset_names"] == ["secque", "financebench"]

    def test_missing_prep_config_returns_empty(self, monkeypatch):
        _CapturingStage.captured = None
        monkeypatch.setattr(
            "nvflow.recipes.finance.stages.evaluation.prepare_data.PrepareFinanceBenchmarksStage",
            _CapturingStage,
        )
        expname = EmbeddedEvalStage._prepare_benchmark_data(
            None,
            base_config={},
            cluster="c",
            expname="grpo-eval",
            enabled_benchmarks={"secque"},
        )
        assert expname == ""
        assert _CapturingStage.captured is None


def test_module_exposes_helpers():
    """Guard against accidental rename of the public-ish helpers under test."""
    assert hasattr(ev, "_enabled_benchmark_names")
    assert hasattr(ev, "_format_benchmarks")
