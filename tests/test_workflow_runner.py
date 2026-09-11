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
"""Tests for nvflow.core.workflow_runner.WorkflowRunner.

WorkflowRunner is the most complex piece of core orchestration logic in
the repo (config loading + ``_base_`` inheritance, dynamic stage
expansion from ``models``/``checkpoints`` sections, dependency-closure
validation, and stage execution) and previously had no dedicated test
file -- it was only exercised incidentally by tests written for
unrelated features.

Two construction strategies are used below:

- Tests of config loading / ``_base_`` inheritance instantiate
  ``WorkflowRunner`` normally against real YAML files under ``tmp_path``,
  since that file I/O *is* the thing being tested.
- Tests of everything downstream of config loading (dynamic expansion,
  dependency validation, ``run()``) use ``_runner_from_config()``, which
  builds a ``WorkflowRunner`` from an in-memory dict via
  ``__new__`` + the same post-load steps ``__init__`` performs, skipping
  file I/O.  This keeps each test's fixture config small and focused
  instead of needing a real YAML file per case.

``StageRegistry`` is a process-wide class-level registry, so any test
that registers stages uses the ``isolated_registry`` fixture, which
swaps ``StageRegistry._stages`` for an empty dict and restores the
original afterward -- registrations never leak between tests or into
the real registry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from nvflow.core.base_stage import BaseStage
from nvflow.core.stage_registry import StageRegistry
from nvflow.core.workflow_runner import WorkflowRunner


def _runner_from_config(config: dict[str, Any]) -> WorkflowRunner:
    """Build a WorkflowRunner from an in-memory config, bypassing file I/O.

    Mirrors the post-load portion of ``__init__``: dynamic stage
    expansion, then extracting recipe/workflow_name/workflow_type/cluster.
    """
    runner = WorkflowRunner.__new__(WorkflowRunner)
    runner.config_path = Path("<in-memory-test-config>")
    runner.config = config
    runner._expand_dynamic_stages()
    runner.recipe = runner.config["recipe"]
    runner.workflow_name = runner.config["workflow"]["name"]
    runner.workflow_type = runner.config["workflow"].get("type", "unknown")
    runner.cluster = runner.config["cluster"]
    return runner


def _cfg(runner: WorkflowRunner) -> dict[str, Any]:
    """Typed accessor for ``runner.config``.

    ``WorkflowRunner.config`` is assigned from ``OmegaConf.to_container()``,
    whose return type is a loose union (``dict | list | str | None``), so
    mypy can't chain-index it outside the constructor's own local flow.
    The runtime value is always a dict for a valid workflow config.
    """
    return runner.config  # type: ignore[return-value]


@pytest.fixture
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give StageRegistry a fresh, empty backing dict for the duration of a test."""
    monkeypatch.setattr(StageRegistry, "_stages", {})


def _register(recipe: str, workflow: str, stage: str, stage_class: type[BaseStage]) -> None:
    StageRegistry.register(recipe=recipe, workflow=workflow, stage=stage)(stage_class)


class _RecordingStage(BaseStage):
    """A stage whose execute() records its call args for assertions."""

    calls: list[dict[str, Any]] = []
    validate_calls: list[dict[str, Any]] = []

    def validate_config(self, config: dict[str, Any]) -> None:
        self.validate_calls.append(config)

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        self.calls.append(
            {"config": config, "cluster": cluster, "expname": expname, "run_after": run_after}
        )


def _make_recording_stage() -> type[_RecordingStage]:
    """Fresh _RecordingStage subclass with its own isolated call log per test."""

    class _Stage(_RecordingStage):
        calls: list[dict[str, Any]] = []
        validate_calls: list[dict[str, Any]] = []

    return _Stage


# ---------------------------------------------------------------------------
# Config loading & _base_ inheritance
# ---------------------------------------------------------------------------


class TestConfigLoadingAndInheritance:
    def _write(self, path: Path, config: dict[str, Any]) -> Path:
        path.write_text(yaml.dump(config))
        return path

    def test_missing_config_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            WorkflowRunner(str(tmp_path / "does_not_exist.yaml"))

    def test_loads_simple_config(self, tmp_path: Path) -> None:
        config_path = self._write(
            tmp_path / "wf.yaml",
            {
                "recipe": "example",
                "workflow": {"name": "sdg_simple", "type": "sdg"},
                "cluster": "my_cluster",
                "pipeline_stages": [],
                "stages": {},
            },
        )
        runner = WorkflowRunner(str(config_path))
        assert runner.recipe == "example"
        assert runner.workflow_name == "sdg_simple"
        assert runner.workflow_type == "sdg"
        assert runner.cluster == "my_cluster"

    def test_workflow_type_defaults_to_unknown(self, tmp_path: Path) -> None:
        config_path = self._write(
            tmp_path / "wf.yaml",
            {
                "recipe": "example",
                "workflow": {"name": "sdg_simple"},
                "cluster": "my_cluster",
                "pipeline_stages": [],
                "stages": {},
            },
        )
        runner = WorkflowRunner(str(config_path))
        assert runner.workflow_type == "unknown"

    def test_base_inheritance_child_overrides_base(self, tmp_path: Path) -> None:
        self._write(
            tmp_path / "base.yaml",
            {
                "recipe": "finance",
                "workflow": {"name": "sft", "type": "training"},
                "cluster": "base_cluster",
                "pipeline_stages": ["sft"],
                "stages": {"sft": {"num_nodes": 8}},
            },
        )
        child_path = self._write(
            tmp_path / "child.yaml",
            {"_base_": "base.yaml", "stages": {"sft": {"num_nodes": 32}}},
        )
        runner = WorkflowRunner(str(child_path))
        assert _cfg(runner)["stages"]["sft"]["num_nodes"] == 32
        # Non-overridden base fields survive the merge.
        assert runner.cluster == "base_cluster"
        assert _cfg(runner)["pipeline_stages"] == ["sft"]

    def test_base_inheritance_chained(self, tmp_path: Path) -> None:
        self._write(
            tmp_path / "grandparent.yaml",
            {
                "recipe": "finance",
                "workflow": {"name": "sft"},
                "cluster": "gp_cluster",
                "pipeline_stages": [],
                "stages": {},
                "base_output_dir": "/gp",
            },
        )
        self._write(
            tmp_path / "parent.yaml",
            {"_base_": "grandparent.yaml", "cluster": "parent_cluster"},
        )
        child_path = self._write(
            tmp_path / "child.yaml",
            {"_base_": "parent.yaml", "base_output_dir": "/child"},
        )
        runner = WorkflowRunner(str(child_path))
        # cluster comes from parent (overrides grandparent), base_output_dir
        # comes from child (overrides grandparent) -- proves both hops merged.
        assert runner.cluster == "parent_cluster"
        assert _cfg(runner)["base_output_dir"] == "/child"

    def test_missing_base_config_raises(self, tmp_path: Path) -> None:
        child_path = self._write(
            tmp_path / "child.yaml",
            {"_base_": "does_not_exist.yaml", "cluster": "x"},
        )
        with pytest.raises(FileNotFoundError, match="Base config not found"):
            WorkflowRunner(str(child_path))

    def test_base_path_resolved_relative_to_child(self, tmp_path: Path) -> None:
        base_dir = tmp_path / "bases"
        base_dir.mkdir()
        self._write(
            base_dir / "base.yaml",
            {
                "recipe": "finance",
                "workflow": {"name": "sft"},
                "cluster": "base_cluster",
                "pipeline_stages": [],
                "stages": {},
            },
        )
        child_path = self._write(tmp_path / "child.yaml", {"_base_": "bases/base.yaml"})
        runner = WorkflowRunner(str(child_path))
        assert runner.cluster == "base_cluster"

    def test_interpolation_is_resolved(self, tmp_path: Path) -> None:
        config_path = self._write(
            tmp_path / "wf.yaml",
            {
                "recipe": "example",
                "workflow": {"name": "sdg_simple"},
                "cluster": "my_cluster",
                "base_data_dir": "/data/root",
                "pipeline_stages": [],
                "stages": {"a": {"output_dir": "${base_data_dir}/step-1"}},
            },
        )
        runner = WorkflowRunner(str(config_path))
        assert _cfg(runner)["stages"]["a"]["output_dir"] == "/data/root/step-1"


# ---------------------------------------------------------------------------
# Dynamic stage expansion (models / checkpoints)
# ---------------------------------------------------------------------------


class TestDynamicStageExpansion:
    def test_models_section_expands_into_stages(self) -> None:
        runner = _runner_from_config(
            {
                "recipe": "finance",
                "workflow": {"name": "sft"},
                "cluster": "c",
                "pipeline_stages": ["qwen3_14b"],
                "stages": {},
                "models": {"qwen3_14b": {"num_nodes": 32}},
            }
        )
        assert _cfg(runner)["stages"]["qwen3_14b"]["num_nodes"] == 32
        assert _cfg(runner)["stages"]["qwen3_14b"]["_source_section"] == "models"
        assert _cfg(runner)["stages"]["qwen3_14b"]["_source_name"] == "qwen3_14b"

    def test_workflow_defaults_forwarded_but_entry_overrides(self) -> None:
        runner = _runner_from_config(
            {
                "recipe": "finance",
                "workflow": {"name": "sft"},
                "cluster": "c",
                "pipeline_stages": ["qwen3_14b"],
                "stages": {},
                "base_output_dir": "/shared",
                "num_nodes": 8,
                "models": {"qwen3_14b": {"num_nodes": 32}},
            }
        )
        stage_cfg = _cfg(runner)["stages"]["qwen3_14b"]
        assert stage_cfg["base_output_dir"] == "/shared"  # forwarded default
        assert stage_cfg["num_nodes"] == 32  # entry overrides default

    def test_structural_keys_not_forwarded_as_defaults(self) -> None:
        runner = _runner_from_config(
            {
                "recipe": "finance",
                "workflow": {"name": "sft"},
                "cluster": "c",
                "pipeline_stages": ["qwen3_14b"],
                "stages": {},
                "models": {"qwen3_14b": {}},
            }
        )
        stage_cfg = _cfg(runner)["stages"]["qwen3_14b"]
        assert "recipe" not in stage_cfg
        assert "cluster" not in stage_cfg
        assert "models" not in stage_cfg

    def test_existing_stage_config_not_overwritten_by_expansion(self) -> None:
        runner = _runner_from_config(
            {
                "recipe": "finance",
                "workflow": {"name": "sft"},
                "cluster": "c",
                "pipeline_stages": ["qwen3_14b"],
                "stages": {"qwen3_14b": {"num_nodes": 999}},
                "models": {"qwen3_14b": {"num_nodes": 32}},
            }
        )
        # Explicit stages: entry wins; expansion must not clobber it.
        assert _cfg(runner)["stages"]["qwen3_14b"]["num_nodes"] == 999

    def test_checkpoints_single_int_eval_step(self) -> None:
        runner = _runner_from_config(
            {
                "recipe": "finance",
                "workflow": {"name": "eval"},
                "cluster": "c",
                "pipeline_stages": ["sft-run"],
                "stages": {},
                "checkpoints": {"sft-run": {"eval_steps": 1000}},
            }
        )
        assert _cfg(runner)["pipeline_stages"] == ["sft-run-1000"]
        assert _cfg(runner)["stages"]["sft-run-1000"]["_step"] == 1000

    def test_checkpoints_multiple_eval_steps_expand_pipeline_stages(self) -> None:
        runner = _runner_from_config(
            {
                "recipe": "finance",
                "workflow": {"name": "eval"},
                "cluster": "c",
                "pipeline_stages": ["sft-run"],
                "stages": {},
                "checkpoints": {"sft-run": {"eval_steps": [1000, 2000]}},
            }
        )
        assert _cfg(runner)["pipeline_stages"] == ["sft-run-1000", "sft-run-2000"]
        assert _cfg(runner)["stages"]["sft-run-1000"]["_step"] == 1000
        assert _cfg(runner)["stages"]["sft-run-2000"]["_step"] == 2000

    def test_checkpoints_leaves_non_checkpoint_pipeline_stages_untouched(self) -> None:
        runner = _runner_from_config(
            {
                "recipe": "finance",
                "workflow": {"name": "eval"},
                "cluster": "c",
                "pipeline_stages": ["prepare_data", "sft-run"],
                "stages": {"prepare_data": {}},
                "checkpoints": {"sft-run": {"eval_steps": [1000]}},
            }
        )
        assert _cfg(runner)["pipeline_stages"] == ["prepare_data", "sft-run-1000"]

    def test_no_expandable_sections_is_a_noop(self) -> None:
        runner = _runner_from_config(
            {
                "recipe": "example",
                "workflow": {"name": "sdg_simple"},
                "cluster": "c",
                "pipeline_stages": ["generate_answer"],
                "stages": {"generate_answer": {}},
            }
        )
        assert _cfg(runner)["pipeline_stages"] == ["generate_answer"]
        assert set(_cfg(runner)["stages"].keys()) == {"generate_answer"}


# ---------------------------------------------------------------------------
# _get_expname / _resolve_env_names / _get_run_after_names
# ---------------------------------------------------------------------------


class TestGetExpname:
    def test_without_run_name(self) -> None:
        runner = _runner_from_config(
            {
                "recipe": "finance",
                "workflow": {"name": "training_sft"},
                "cluster": "c",
                "pipeline_stages": [],
                "stages": {},
            }
        )
        assert runner._get_expname("sft", {}) == "training_sft-sft"

    def test_with_run_name(self) -> None:
        runner = _runner_from_config(
            {
                "recipe": "finance",
                "workflow": {"name": "training_sft"},
                "cluster": "c",
                "pipeline_stages": [],
                "stages": {},
            }
        )
        assert (
            runner._get_expname("sft", {"run_name": "lr5e6-bs128"})
            == "training_sft-sft-lr5e6-bs128"
        )


class TestResolveEnvNames:
    def test_no_environments_returns_empty(self) -> None:
        assert WorkflowRunner._resolve_env_names({}, None) == []

    def test_no_filter_returns_all_env_names(self) -> None:
        stage_config: dict[str, Any] = {"environments": {"mcqa": {}, "equivalence_llm_judge": {}}}
        assert set(WorkflowRunner._resolve_env_names(stage_config, None)) == {
            "mcqa",
            "equivalence_llm_judge",
        }

    def test_filter_returns_only_requested_envs_present_in_config(self) -> None:
        stage_config: dict[str, Any] = {"environments": {"mcqa": {}, "equivalence_llm_judge": {}}}
        result = WorkflowRunner._resolve_env_names(stage_config, ["equivalence_llm_judge", "nope"])
        assert result == ["equivalence_llm_judge"]


class TestGetRunAfterNames:
    def _runner(self, stages: dict[str, Any]) -> WorkflowRunner:
        return _runner_from_config(
            {
                "recipe": "finance",
                "workflow": {"name": "grpo"},
                "cluster": "c",
                "pipeline_stages": list(stages.keys()),
                "stages": stages,
            }
        )

    def test_no_dependencies_returns_none(self) -> None:
        runner = self._runner({"a": {}})
        assert runner._get_run_after_names([], None) is None

    def test_dependency_without_environments(self) -> None:
        runner = self._runner({"download": {}, "prepare": {}})
        assert runner._get_run_after_names(["download"], None) == ["grpo-download"]

    def test_dependency_with_environments_expands_per_env(self) -> None:
        runner = self._runner(
            {
                "collect_rollouts": {"environments": {"mcqa": {}, "equivalence_llm_judge": {}}},
                "train": {},
            }
        )
        result = runner._get_run_after_names(["collect_rollouts"], None)
        assert result is not None
        assert set(result) == {
            "grpo-collect_rollouts-mcqa",
            "grpo-collect_rollouts-equivalence_llm_judge",
        }

    def test_dependency_with_environments_filtered(self) -> None:
        runner = self._runner(
            {
                "collect_rollouts": {"environments": {"mcqa": {}, "equivalence_llm_judge": {}}},
                "train": {},
            }
        )
        result = runner._get_run_after_names(["collect_rollouts"], ["mcqa"])
        assert result == ["grpo-collect_rollouts-mcqa"]


# ---------------------------------------------------------------------------
# _validate_stages / validate_config
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("isolated_registry")
class TestValidateStages:
    def _runner(self, stages: dict[str, Any], pipeline_stages: list[str]) -> WorkflowRunner:
        return _runner_from_config(
            {
                "recipe": "test_recipe",
                "workflow": {"name": "test_workflow"},
                "cluster": "c",
                "pipeline_stages": pipeline_stages,
                "stages": stages,
            }
        )

    def test_valid_registered_stage_passes(self) -> None:
        _register("test_recipe", "test_workflow", "a", _make_recording_stage())
        runner = self._runner({"a": {}}, ["a"])
        runner._validate_stages(["a"], ["a"])  # does not raise

    def test_stage_not_in_pipeline_config_raises(self) -> None:
        runner = self._runner({"a": {}}, ["a"])
        with pytest.raises(ValueError, match="not found in workflow config"):
            runner._validate_stages(["b"], ["a"])

    def test_unregistered_stage_raises(self) -> None:
        runner = self._runner({"a": {}}, ["a"])
        with pytest.raises(ValueError, match="not registered"):
            runner._validate_stages(["a"], ["a"])

    def test_stage_missing_config_block_raises(self) -> None:
        _register("test_recipe", "test_workflow", "a", _make_recording_stage())
        # "a" is in pipeline_stages but has no entry under stages:
        runner = self._runner({}, ["a"])
        with pytest.raises(ValueError, match="no configuration"):
            runner._validate_stages(["a"], ["a"])

    def test_dependency_closure_walks_transitively(self) -> None:
        stage_cls = _make_recording_stage()
        _register("test_recipe", "test_workflow", "a", stage_cls)
        _register("test_recipe", "test_workflow", "b", stage_cls)
        _register("test_recipe", "test_workflow", "c", stage_cls)
        runner = self._runner(
            {"a": {"dependencies": ["b"]}, "b": {"dependencies": ["c"]}, "c": {}},
            ["a", "b", "c"],
        )
        runner._validate_stages(["a"], ["a", "b", "c"])  # transitively OK, does not raise

    def test_dependency_missing_config_block_raises(self) -> None:
        stage_cls = _make_recording_stage()
        _register("test_recipe", "test_workflow", "a", stage_cls)
        runner = self._runner({"a": {"dependencies": ["ghost"]}}, ["a"])
        with pytest.raises(ValueError, match="no config block"):
            runner._validate_stages(["a"], ["a"])

    def test_dependency_not_registered_raises(self) -> None:
        stage_cls = _make_recording_stage()
        _register("test_recipe", "test_workflow", "a", stage_cls)
        runner = self._runner({"a": {"dependencies": ["b"]}, "b": {}}, ["a", "b"])
        with pytest.raises(ValueError, match="not registered"):
            runner._validate_stages(["a"], ["a", "b"])


@pytest.mark.usefixtures("isolated_registry")
class TestValidateConfigPublic:
    @pytest.mark.parametrize(
        "missing_field", ["recipe", "workflow", "cluster", "pipeline_stages", "stages"]
    )
    def test_missing_required_field_raises(self, missing_field: str) -> None:
        config = {
            "recipe": "test_recipe",
            "workflow": {"name": "test_workflow"},
            "cluster": "c",
            "pipeline_stages": [],
            "stages": {},
        }
        del config[missing_field]
        runner = WorkflowRunner.__new__(WorkflowRunner)
        runner.config_path = Path("<test>")
        runner.config = config
        with pytest.raises(ValueError, match=f"Missing required field.*{missing_field}"):
            runner.validate_config()

    def test_valid_config_passes(self) -> None:
        stage_cls = _make_recording_stage()
        _register("test_recipe", "test_workflow", "a", stage_cls)
        runner = _runner_from_config(
            {
                "recipe": "test_recipe",
                "workflow": {"name": "test_workflow"},
                "cluster": "c",
                "pipeline_stages": ["a"],
                "stages": {"a": {}},
            }
        )
        runner.validate_config()  # does not raise


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("isolated_registry")
class TestRun:
    def test_runs_all_stages_in_order_and_calls_validate_then_execute(self) -> None:
        stage_a, stage_b = _make_recording_stage(), _make_recording_stage()
        _register("test_recipe", "test_workflow", "a", stage_a)
        _register("test_recipe", "test_workflow", "b", stage_b)
        runner = _runner_from_config(
            {
                "recipe": "test_recipe",
                "workflow": {"name": "test_workflow"},
                "cluster": "my_cluster",
                "pipeline_stages": ["a", "b"],
                "stages": {"a": {}, "b": {"dependencies": ["a"]}},
            }
        )
        runner.run()
        assert len(stage_a.calls) == 1
        assert len(stage_b.calls) == 1
        assert len(stage_a.validate_calls) == 1
        assert stage_a.calls[0]["cluster"] == "my_cluster"
        assert stage_a.calls[0]["expname"] == "test_workflow-a"
        assert stage_b.calls[0]["run_after"] == ["test_workflow-a"]

    def test_runs_only_requested_subset(self) -> None:
        stage_a, stage_b = _make_recording_stage(), _make_recording_stage()
        _register("test_recipe", "test_workflow", "a", stage_a)
        _register("test_recipe", "test_workflow", "b", stage_b)
        runner = _runner_from_config(
            {
                "recipe": "test_recipe",
                "workflow": {"name": "test_workflow"},
                "cluster": "c",
                "pipeline_stages": ["a", "b"],
                "stages": {"a": {}, "b": {}},
            }
        )
        runner.run(stages=["b"])
        assert len(stage_a.calls) == 0
        assert len(stage_b.calls) == 1

    def test_dependencies_outside_this_run_are_dropped_from_run_after(self) -> None:
        stage_a, stage_b = _make_recording_stage(), _make_recording_stage()
        _register("test_recipe", "test_workflow", "a", stage_a)
        _register("test_recipe", "test_workflow", "b", stage_b)
        runner = _runner_from_config(
            {
                "recipe": "test_recipe",
                "workflow": {"name": "test_workflow"},
                "cluster": "c",
                "pipeline_stages": ["a", "b"],
                # "b" depends on "a", but only "b" is being submitted this run.
                "stages": {"a": {}, "b": {"dependencies": ["a"]}},
            }
        )
        runner.run(stages=["b"])
        assert stage_b.calls[0]["run_after"] is None

    def test_unknown_environment_raises(self) -> None:
        stage_a = _make_recording_stage()
        _register("test_recipe", "test_workflow", "a", stage_a)
        runner = _runner_from_config(
            {
                "recipe": "test_recipe",
                "workflow": {"name": "test_workflow"},
                "cluster": "c",
                "pipeline_stages": ["a"],
                "stages": {"a": {}},
                "environments": {"mcqa": {}},
            }
        )
        with pytest.raises(ValueError, match="Unknown environment"):
            runner.run(environment=["nope"])
        assert len(stage_a.calls) == 0

    def test_preflight_warns_on_unregistered_sibling_stage(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        stage_a = _make_recording_stage()
        _register("test_recipe", "test_workflow", "a", stage_a)
        # "b" is declared in pipeline_stages/stages but deliberately never
        # registered -- simulates a sibling stage whose import failed.
        runner = _runner_from_config(
            {
                "recipe": "test_recipe",
                "workflow": {"name": "test_workflow"},
                "cluster": "c",
                "pipeline_stages": ["a", "b"],
                "stages": {"a": {}, "b": {}},
            }
        )
        runner.run(stages=["a"])
        captured = capsys.readouterr()
        assert "not currently registered" in captured.err
        assert "test_recipe.test_workflow.b" in captured.err

    def test_invalid_requested_stage_raises_before_any_execution(self) -> None:
        stage_a = _make_recording_stage()
        _register("test_recipe", "test_workflow", "a", stage_a)
        runner = _runner_from_config(
            {
                "recipe": "test_recipe",
                "workflow": {"name": "test_workflow"},
                "cluster": "c",
                "pipeline_stages": ["a"],
                "stages": {"a": {}},
            }
        )
        with pytest.raises(ValueError, match="not found in workflow config"):
            runner.run(stages=["ghost"])
        assert len(stage_a.calls) == 0
