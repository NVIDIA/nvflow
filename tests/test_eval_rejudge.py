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
"""First-class eval rejudging over existing canonical Gym rollouts."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

from nvflow.execution import PlanningContext, TargetBinding
from nvflow.integrations.nemo_gym.operations import (
    normalize_reward_operation,
    parse_worker_extensions,
)
from nvflow.stages.nemo_gym import CollectRolloutsStage, RejudgeRolloutsStage


def _config() -> dict[str, Any]:
    return {
        "contract_version": 1,
        "output_dir": "/runs/qwen/rejudge/gpt-5",
        "rollouts_dir": "/runs/qwen/collect-rollouts",
        "container": "nemo-gym",
        "vllm_container": "vllm",
        "environments": {
            "secque": {
                "enabled": True,
                "input_data": "/prepared/secque/train.jsonl",
                "agent_name": "equivalence_llm_judge_simple_agent",
                "config_paths": [
                    "resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml"
                ],
                "models": {
                    "judge": {
                        "deployment": "nvflow",
                        "model": "/models/old-judge",
                        "num_gpus": 4,
                        "gym_model": {"vllm_serve_kwargs": {"tensor_parallel_size": 4}},
                    }
                },
                "gym": {
                    "observability_enabled": True,
                    "model_call_capture_dir": "/runs/qwen/collect-rollouts/secque/captures/model-calls",
                },
                "blade": {"command": "blade secque"},
            },
            "fabv1": {
                "enabled": True,
                "input_data": "/prepared/fabv1/train.jsonl",
                "output_name": "fabv1_noweb",
                "agent_name": "finance_agent",
                "config_paths": [
                    "resources_servers/finance_sec_search/configs/finance_sec_search.yaml"
                ],
            },
        },
        "verification": {
            "mode": "fallback",
            "models": {
                "judge": {
                    "deployment": "external",
                    "model_type": "openai_model",
                    "model": "gpt-5",
                    "base_url": "https://judge.example/v1",
                    "api_key_env": "JUDGE_API_KEY",
                }
            },
        },
        "blade": {"enabled": True, "command": "blade analyze"},
        "extensions": {
            "report": {
                "module": "nvfinance.utils.rl.eval_aggregate",
                "contract_version": 1,
            }
        },
    }


def test_eval_rejudge_is_bound_as_a_first_class_stage() -> None:
    script = """
from nvflow import load_workflow
definition = load_workflow("projects/finance/workflows/eval/smoke.yaml")
assert definition.resolve_stage_class("rejudge").__name__ == "RejudgeRolloutsStage"
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_eval_rejudge_uses_existing_benchmark_artifact_names() -> None:
    configurations = dict(RejudgeRolloutsStage.target_configs(_config()))

    assert configurations["secque"]["rollouts_dir"] == ("/runs/qwen/collect-rollouts/secque")
    assert configurations["secque"]["output_dir"] == ("/runs/qwen/rejudge/gpt-5/secque")
    assert configurations["fabv1_noweb"]["rollouts_dir"] == (
        "/runs/qwen/collect-rollouts/fabv1_noweb"
    )
    assert "filter" not in configurations["secque"]
    assert configurations["secque"]["blade"]["enabled"] is True
    assert configurations["secque"]["blade"]["command"] == "blade secque"
    assert configurations["secque"]["environments"]["secque"]["gym"]["model_call_capture_dir"] == (
        "/runs/qwen/rejudge/gpt-5/secque/jobs/job-0-rejudge-rollouts/outputs/captures/model-calls"
    )
    assert (
        configurations["secque"]["rejudge"]["models"]["judge"]
        == (_config()["verification"]["models"]["judge"])
    )


def test_eval_rejudge_config_uses_shared_reward_contract() -> None:
    RejudgeRolloutsStage().validate_config(_config())


@pytest.mark.requires_nemo
def test_eval_rejudge_plans_without_training_postprocessing() -> None:
    config = _config()
    plan = RejudgeRolloutsStage().plan(
        PlanningContext(
            stage="rejudge",
            config=config,
            launcher_config=config,
            target=TargetBinding.resolve("cluster"),
            expname="eval-rejudge",
            run_after=("eval-source",),
        )
    )

    assert any("secque" in task.task_id for task in plan.tasks)
    assert any("fabv1_noweb" in task.task_id for task in plan.tasks)
    extensions = parse_worker_extensions(config["extensions"])
    operations = [
        normalize_reward_operation(target, extensions)
        for _name, target in RejudgeRolloutsStage.target_configs(config)
    ]
    assert all(
        operation.report_module == "nvfinance.utils.rl.eval_aggregate" for operation in operations
    )


def test_training_environment_can_override_campaign_blade() -> None:
    config = {
        "contract_version": 1,
        "output_dir": "/runs/collect",
        "blade": {"enabled": True, "command": "blade default"},
        "rollout": {"models": {}},
        "environments": {
            "finance": {
                "enabled": True,
                "input_data": "/runs/prepare/finance/train.jsonl",
                "config_paths": ["resources_servers/finance/config.yaml"],
                "agent_name": "finance_agent",
                "blade": {"enabled": False},
            }
        },
    }

    environment = dict(CollectRolloutsStage.target_configs(config))["finance"]

    assert environment["blade"] == {
        "enabled": False,
        "command": "blade default",
    }
