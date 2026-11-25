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
"""Execution-contract tests for embedded Gym checkpoint evaluation."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest
from nvfinance.stages.rl.embedded_eval import GymEmbeddedEvalStage
from nvfinance.stages.rl.eval_model import (
    GymCheckpointEvaluatorStage,
    GymModelEvaluatorStage,
)

from nvflow.execution import PlanningContext, StagePlan, TargetBinding
from nvflow.stages.nemo_gym import PrepareBenchmarkStage

CAMPAIGN = {
    "data_root": "/out",
    "container": "nemo-gym",
    "vllm_container": "vllm",
    "rollout": {
        "models": {
            "policy": {
                "deployment": "gym",
                "num_gpus": 4,
                "gym_model": {"vllm_serve_kwargs": {"max_model_len": 40960}},
            },
        },
        "num_repeats": 5,
        "num_chunks": 1,
    },
    "model_presets": {
        "qwen3_4b": {
            "model": "/hf_models/Qwen3-4B",
            "num_gpus": 4,
            "gym_model": {"vllm_serve_kwargs": {"max_model_len": 40960}},
        }
    },
    "benchmarks": {
        "secque": {
            "input_data": "/data/secque.jsonl",
            "config_paths": ["resources_servers/equivalence/config.yaml"],
            "agent_name": "secque_agent",
            "repeats": 5,
        },
        "financebench": {
            "input_data": "/data/financebench.jsonl",
            "config_paths": ["resources_servers/equivalence/config.yaml"],
            "agent_name": "financebench_agent",
            "repeats": 5,
        },
    },
}


def _eval_config(**overrides: Any) -> dict[str, Any]:
    config = {
        "eval_steps": [20],
        "eval_data_root": "/campaign/step-1-eval-data",
        "eval_output_dir": "/out/step-9-eval",
        "base_output_dir": "/out",
        "preset": "qwen3_4b",
        "benchmarks": ["secque", "financebench"],
        "format": "fsdp",
    }
    config.update(overrides)
    return config


def _context(
    config: dict[str, Any],
    *,
    run_after: tuple[str, ...] = (),
) -> PlanningContext:
    return PlanningContext(
        stage="embedded-eval",
        config=config,
        launcher_config=config,
        target=TargetBinding.resolve("cluster"),
        expname="grpo-eval",
        run_after=run_after,
    )


def _plan_with_captured_entries(
    monkeypatch: pytest.MonkeyPatch,
    *,
    campaign: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    run_after: tuple[str, ...] = (),
    configure_setup: Callable[[dict[str, list[PlanningContext]]], None] | None = None,
) -> dict[str, list[PlanningContext]]:
    from nvfinance.stages.rl import embedded_eval

    selected_campaign = copy.deepcopy(campaign or CAMPAIGN)
    monkeypatch.setattr(
        embedded_eval,
        "_load_eval_campaign",
        lambda _root, _campaign=None, **_kwargs: copy.deepcopy(selected_campaign),
    )
    captured: dict[str, list[PlanningContext]] = {
        "checkpoint": [],
        "baseline": [],
        "prepare": [],
    }

    def capture_checkpoint(_self, context: PlanningContext) -> StagePlan:
        captured["checkpoint"].append(context)
        return StagePlan.noop(context.stage, "captured checkpoint plan")

    def capture_baseline(_self, context: PlanningContext) -> StagePlan:
        captured["baseline"].append(context)
        return StagePlan.noop(context.stage, "captured baseline plan")

    monkeypatch.setattr(GymCheckpointEvaluatorStage, "plan", capture_checkpoint)
    monkeypatch.setattr(GymModelEvaluatorStage, "plan", capture_baseline)
    if configure_setup is not None:
        configure_setup(captured)

    GymEmbeddedEvalStage().plan(_context(config or _eval_config(), run_after=run_after))
    return captured


def test_training_environment_does_not_filter_benchmarks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _plan_with_captured_entries(
        monkeypatch,
        config=_eval_config(_environment="equivalence_llm_judge"),
    )

    assert set(captured["checkpoint"][0].config["environments"]) == {
        "secque",
        "financebench",
    }


def test_benchmarks_come_from_eval_block(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _plan_with_captured_entries(
        monkeypatch,
        config=_eval_config(
            benchmarks=["secque"],
            _environment="finance_sec_search",
        ),
    )

    assert set(captured["checkpoint"][0].config["environments"]) == {"secque"}


def test_campaign_settings_reach_checkpoint_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _plan_with_captured_entries(monkeypatch)

    policy = captured["checkpoint"][0].config["rollout"]["models"]["policy"]
    assert policy["deployment"] == "gym"
    assert policy["num_gpus"] == 4


def test_checkpoint_evaluation_reads_prepared_data_without_preparing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign: dict[str, Any] = copy.deepcopy(CAMPAIGN)
    campaign["stages"] = {
        "eval_data": {
            "contract_version": 1,
            "output_dir": "/prepared",
            "gym_path": "/opt/nemo-gym",
            "container": "nemo-gym",
        }
    }

    def configure_setup(captured: dict[str, list[PlanningContext]]) -> None:
        def capture_prepare(_self, context: PlanningContext) -> StagePlan:
            captured["prepare"].append(context)
            return StagePlan.noop(context.stage, "captured preparation plan")

        monkeypatch.setattr(PrepareBenchmarkStage, "plan", capture_prepare)

    captured = _plan_with_captured_entries(
        monkeypatch,
        campaign=campaign,
        run_after=("training",),
        configure_setup=configure_setup,
    )

    assert captured["prepare"] == []
    assert captured["checkpoint"][0].run_after == ("training",)


def test_eval_data_root_reaches_the_campaign(monkeypatch: pytest.MonkeyPatch) -> None:
    from nvfinance.stages.rl import embedded_eval

    roots: list[str] = []

    def load_campaign(_root: str, _campaign: str, **kwargs: Any) -> dict[str, Any]:
        roots.append(kwargs["eval_data_root"])
        return copy.deepcopy(CAMPAIGN)

    monkeypatch.setattr(embedded_eval, "_load_eval_campaign", load_campaign)
    monkeypatch.setattr(
        GymCheckpointEvaluatorStage,
        "plan",
        lambda _self, context: StagePlan.noop(context.stage, "captured checkpoint plan"),
    )

    GymEmbeddedEvalStage().plan(_context(_eval_config()))

    assert roots == ["/campaign/step-1-eval-data"]


def test_campaign_benchmarks_read_from_eval_data_root() -> None:
    from nvfinance.stages.rl.embedded_eval import _load_eval_campaign

    campaign = _load_eval_campaign(
        "/out/step-9-eval/step-20",
        "embedded.yaml",
        eval_data_root="/campaign/step-1-eval-data",
    )

    assert campaign["benchmarks"]["secque"]["input_data"] == (
        "/campaign/step-1-eval-data/secque/eval.jsonl"
    )


def test_checkpoint_evaluation_requires_eval_data_root() -> None:
    config = _eval_config(checkpoint_path="/checkpoints")
    del config["eval_data_root"]

    with pytest.raises(ValueError, match="eval_data_root is required"):
        GymEmbeddedEvalStage().validate_config(config)


def test_eval_settings_follow_the_environment_that_trains_alone() -> None:
    from nvfinance.stages.rl.embedded_eval import _for_training_environment

    config = _eval_config(
        environments={
            "equivalence_llm_judge": {"raw_train_data": "/raw"},
            "finance_sec_search": {"raw_train_data": "/raw", "eval": {"checkpoint_path": "/sec"}},
            "mcqa": {},
        },
    )

    assert "checkpoint_path" not in _for_training_environment(config)
    selected = {**config, "_environment": ["finance_sec_search", "mcqa"]}
    assert _for_training_environment(selected)["checkpoint_path"] == "/sec"


@pytest.mark.parametrize("key", ["dependencies", "environments"])
def test_environment_eval_rejects_keys_it_cannot_apply(key: str) -> None:
    config = _eval_config(environments={"finance_sec_search": {"eval": {key: []}}})

    with pytest.raises(ValueError, match=rf"environments\.finance_sec_search\.eval\.{key}"):
        GymEmbeddedEvalStage().validate_config(config)
