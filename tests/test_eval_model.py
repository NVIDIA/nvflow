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
"""Tests for the models x benchmarks eval matrix.

The matrix layers three sources of serving config -- workflow defaults, the
model's profile, then the benchmark's -- and a mistake in that order is only
visible as a model served with the wrong context window or GPU count. These
tests pin the precedence, plus three behaviours that silently corrupt results
when they break:

  - the benchmark catalog must not be mutated, or one benchmark's per-model
    override leaks into the next model;
  - ``output_name`` must name the output subtree, so two runs of one benchmark
    with different tool sets never share a directory;
  - backend detection must follow the ``_base_`` chain, or a config that
    inherits ``workflow.name`` registers on the wrong backend.
"""

from __future__ import annotations

import copy

import pytest
from nvfinance.stages.rl.eval_model import (
    build_gym_model_config,
    run_gym_eval_matrix,
)

CATALOG = {
    "secque": {
        "input_data": "/data/secque.jsonl",
        "config_paths": ["resources_servers/equivalence/config.yaml"],
        "agent_name": "secque_agent",
        "repeats": 5,
    },
    "fabv1": {
        "input_data": "/data/fabv1.jsonl",
        "config_paths": ["resources_servers/finance/config.yaml"],
        "agent_name": "finance_agent",
        "repeats": 3,
        "output_name": "fabv1_noweb",
        "models": {"policy": {"gym_model": {"vllm_serve_kwargs": {"max_model_len": 131072}}}},
    },
}


def _model_config(**overrides):
    config = {
        "_source_name": "qwen3-4b",
        "data_root": "/out",
        "gym_path": "/opt/nemo-gym",
        "container": "nemo-gym",
        "vllm_container": "vllm",
        "environments": copy.deepcopy(CATALOG),
        "rollout": {
            "models": {
                "policy": {
                    "deployment": "nvflow",
                    "num_gpus": 4,
                    "gym_model": {"vllm_serve_kwargs": {"max_model_len": 40960}},
                },
            },
            "num_repeats": 5,
            "num_chunks": 1,
        },
        "model": {"model": "/hf_models/Qwen3-4B"},
    }
    config.update(overrides)
    return config


# ---------------------------------------------------------------------------
# build_gym_model_config
# ---------------------------------------------------------------------------


def test_model_identity_lands_in_the_policy_block():
    got = build_gym_model_config(_model_config())
    assert got["rollout"]["models"]["policy"]["model"] == "/hf_models/Qwen3-4B"
    assert got["output_dir"] == "/out/qwen3-4b/collect-rollouts"


def test_model_output_dir_can_follow_pipeline_step_naming():
    got = build_gym_model_config(_model_config(model_output_dir="/out/step-2-qwen3-4b"))
    assert got["output_dir"] == "/out/step-2-qwen3-4b/collect-rollouts"


def test_model_policy_overrides_beat_the_workflow_defaults():
    got = build_gym_model_config(_model_config(model={"model": "/m", "num_gpus": 8}))
    policy = got["rollout"]["models"]["policy"]
    assert policy["num_gpus"] == 8
    assert policy["gym_model"]["vllm_serve_kwargs"]["max_model_len"] == 40960


def test_gym_model_contract_comes_directly_from_model_preset():
    got = build_gym_model_config(
        _model_config(
            model={
                "model": "/m",
                "gym_model": {
                    "uses_reasoning_parser": True,
                    "vllm_serve_kwargs": {"reasoning_parser": "qwen3"},
                },
            }
        )
    )
    policy = got["rollout"]["models"]["policy"]

    assert policy["gym_model"]["vllm_serve_kwargs"]["reasoning_parser"] == "qwen3"
    assert policy["gym_model"]["uses_reasoning_parser"] is True


def test_per_benchmark_overrides_merge_into_that_benchmark_only():
    got = build_gym_model_config(
        _model_config(
            model={
                "model": "/m",
                "benchmark_models": {
                    "fabv1": {"gym_model": {"vllm_serve_kwargs": {"tool_call_parser": "hermes"}}}
                },
            }
        )
    )
    fabv1 = got["environments"]["fabv1"]["models"]["policy"]["gym_model"]["vllm_serve_kwargs"]
    assert fabv1["tool_call_parser"] == "hermes"
    assert fabv1["max_model_len"] == 131072
    assert "models" not in got["environments"]["secque"]


def test_the_shared_catalog_is_never_mutated():
    """Otherwise one model's override would leak into every later model."""
    config = _model_config(
        model={
            "model": "/m",
            "benchmark_models": {
                "fabv1": {"gym_model": {"vllm_serve_kwargs": {"tool_call_parser": "hermes"}}}
            },
        }
    )
    build_gym_model_config(config)
    assert config["environments"]["fabv1"]["models"]["policy"]["gym_model"] == {
        "vllm_serve_kwargs": {"max_model_len": 131072}
    }


def test_chat_template_kwargs_are_carried_on_the_policy_block():
    got = build_gym_model_config(
        _model_config(
            model={
                "model": "/m",
                "gym_model": {"chat_template_kwargs": {"reasoning_effort": "high"}},
            }
        )
    )
    assert got["rollout"]["models"]["policy"]["gym_model"]["chat_template_kwargs"] == {
        "reasoning_effort": "high"
    }


def test_benchmark_filter_is_forwarded_to_the_matrix():
    got = build_gym_model_config(_model_config(_environment=["fabv1"]))
    assert got["_environment"] == ["fabv1"]


def test_no_filter_means_no_key_so_the_full_catalog_runs():
    assert "_environment" not in build_gym_model_config(_model_config())


def test_campaign_blade_config_is_forwarded_to_shared_rollout_engine():
    blade = {"enabled": True, "command": "blade analyze"}
    got = build_gym_model_config(_model_config(blade=blade))

    assert got["blade"] == blade


# ---------------------------------------------------------------------------
# run_gym_eval_matrix
# ---------------------------------------------------------------------------


def _capture_rollout(monkeypatch):
    """Replace rollout() with a recorder and return the list it appends to."""
    pytest.importorskip("nemo_skills")  # rollout imports it at call time
    import nvflow.integrations.nemo_gym.native_rollout as rollout_module

    calls: list[dict] = []

    def _fake_rollout(config, cluster, expname, run_after=None, **kwargs):
        calls.append({"config": config, "expname": expname, "kwargs": kwargs})
        return []

    monkeypatch.setattr(rollout_module, "rollout", _fake_rollout)
    return calls


@pytest.mark.requires_nemo
def test_output_name_names_the_directory_and_the_job(monkeypatch):
    calls = _capture_rollout(monkeypatch)
    run_gym_eval_matrix(build_gym_model_config(_model_config()), "cluster", "exp")

    by_name = {c["expname"]: c for c in calls}
    assert set(by_name) == {"exp-secque", "exp-fabv1_noweb"}
    assert by_name["exp-fabv1_noweb"]["config"]["output_dir"].endswith("/fabv1_noweb")
    # The catalog key is what the environment is still called inside the job.
    assert list(by_name["exp-fabv1_noweb"]["config"]["environments"]) == ["fabv1"]


@pytest.mark.requires_nemo
def test_benchmark_repeat_count_does_not_change_outer_chunks(monkeypatch):
    calls = _capture_rollout(monkeypatch)
    run_gym_eval_matrix(build_gym_model_config(_model_config()), "cluster", "exp")

    repeats = {c["expname"]: c["config"]["rollout"]["num_repeats"] for c in calls}
    assert repeats == {"exp-secque": 5, "exp-fabv1_noweb": 3}
    assert {c["expname"]: c["config"]["rollout"]["num_chunks"] for c in calls} == {
        "exp-secque": 1,
        "exp-fabv1_noweb": 1,
    }


@pytest.mark.requires_nemo
def test_benchmark_can_override_or_disable_campaign_blade(monkeypatch):
    calls = _capture_rollout(monkeypatch)
    config = build_gym_model_config(
        _model_config(blade={"enabled": True, "command": "blade default"})
    )
    config["environments"]["secque"]["blade"] = {"enabled": False}
    config["environments"]["fabv1"]["blade"] = {"command": "blade fabv1"}

    run_gym_eval_matrix(config, "cluster", "exp")
    by_name = {call["expname"]: call["config"]["blade"] for call in calls}

    assert by_name["exp-secque"] == {
        "enabled": False,
        "command": "blade default",
    }
    assert by_name["exp-fabv1_noweb"] == {
        "enabled": True,
        "command": "blade fabv1",
    }


@pytest.mark.requires_nemo
def test_benchmark_policy_profile_layers_over_the_model(monkeypatch):
    calls = _capture_rollout(monkeypatch)
    run_gym_eval_matrix(build_gym_model_config(_model_config()), "cluster", "exp")

    policies = {c["expname"]: c["config"]["rollout"]["models"]["policy"] for c in calls}
    assert policies["exp-fabv1_noweb"]["gym_model"]["vllm_serve_kwargs"]["max_model_len"] == 131072
    assert policies["exp-secque"]["gym_model"]["vllm_serve_kwargs"]["max_model_len"] == 40960
    assert policies["exp-fabv1_noweb"]["num_gpus"] == 4  # model default is kept


@pytest.mark.requires_nemo
def test_eval_uses_canonical_native_report_for_one_repeat(monkeypatch):
    calls = _capture_rollout(monkeypatch)
    config = build_gym_model_config(_model_config())
    config["environments"] = {
        "secque": {
            **CATALOG["secque"],
            "input_data": "/d.jsonl",
            "repeats": 1,
        }
    }
    run_gym_eval_matrix(config, "cluster", "exp")

    assert calls[0]["kwargs"]["report_module"].endswith(".eval_aggregate")


@pytest.mark.parametrize("max_output_tokens", [65536, 128000])
@pytest.mark.requires_nemo
def test_output_budget_that_crowds_out_the_prompt_is_rejected(monkeypatch, max_output_tokens):
    """vLLM 400s every such request, and the run would score zero rather than fail."""
    calls = _capture_rollout(monkeypatch)
    config = build_gym_model_config(
        _model_config(
            rollout={
                "models": {
                    "policy": {
                        "deployment": "nvflow",
                        "num_gpus": 4,
                        "gym_model": {"vllm_serve_kwargs": {"max_model_len": 65536}},
                    },
                },
                "num_repeats": 1,
                "num_chunks": 1,
                "responses_create_params": {"max_output_tokens": max_output_tokens},
            }
        )
    )

    with pytest.raises(ValueError, match="no room for the prompt"):
        run_gym_eval_matrix(config, "cluster", "exp")
    assert calls == []


@pytest.mark.requires_nemo
def test_output_budget_inside_the_window_is_allowed(monkeypatch):
    calls = _capture_rollout(monkeypatch)
    config = build_gym_model_config(
        _model_config(
            rollout={
                "models": {
                    "policy": {
                        "deployment": "nvflow",
                        "num_gpus": 4,
                        "gym_model": {"vllm_serve_kwargs": {"max_model_len": 65536}},
                    },
                },
                "num_repeats": 1,
                "num_chunks": 1,
                "responses_create_params": {"max_output_tokens": 32768},
            }
        )
    )

    run_gym_eval_matrix(config, "cluster", "exp")
    assert len(calls) == 2


@pytest.mark.requires_nemo
def test_budget_is_checked_against_the_benchmark_window_not_the_model_default(monkeypatch):
    """fabv1 widens the window, so a budget too large for QA is fine there."""
    calls = _capture_rollout(monkeypatch)
    config = build_gym_model_config(
        _model_config(
            environments={
                "fabv1": {
                    **CATALOG["fabv1"],
                    "models": {
                        "policy": {"gym_model": {"vllm_serve_kwargs": {"max_model_len": 262144}}}
                    },
                }
            },
            rollout={
                "models": {
                    "policy": {
                        "deployment": "nvflow",
                        "num_gpus": 4,
                        "gym_model": {"vllm_serve_kwargs": {"max_model_len": 65536}},
                    },
                },
                "num_repeats": 1,
                "num_chunks": 1,
                "responses_create_params": {"max_output_tokens": 128000},
            },
        )
    )

    run_gym_eval_matrix(config, "cluster", "exp")
    assert len(calls) == 1


@pytest.mark.requires_nemo
def test_benchmark_without_input_data_is_rejected(monkeypatch):
    _capture_rollout(monkeypatch)
    config = build_gym_model_config(_model_config())
    missing_input = dict(CATALOG["secque"])
    missing_input.pop("input_data")
    config["environments"] = {"secque": missing_input}

    with pytest.raises(ValueError, match="requires input_data"):
        run_gym_eval_matrix(config, "cluster", "exp")


@pytest.mark.requires_nemo
def test_out_of_scope_model_submits_nothing(monkeypatch):
    calls = _capture_rollout(monkeypatch)
    config = build_gym_model_config(_model_config())
    config["environments"] = {"secque": CATALOG["secque"]}
    config["_environment"] = "fabv1"

    run_gym_eval_matrix(config, "cluster", "exp")
    assert calls == []
