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
"""Tests for independent native Gym reward recomputation."""

from __future__ import annotations

import asyncio
import importlib
import json
import subprocess
import sys
import types
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

pytestmark = pytest.mark.requires_nemo
pytest.importorskip("nemo_skills")

import nvflow.integrations.nemo_gym.native_rewards as native_rewards  # noqa: E402
from nvflow.integrations.nemo_gym.native_contract import NativeGymOutputLayout  # noqa: E402
from nvflow.integrations.nemo_gym.native_rewards import (  # noqa: E402
    build_reward_jobs,
    parse_reward_config,
)
from nvflow.integrations.nemo_gym.native_rollout import (  # noqa: E402
    NativeRolloutConfigError,
)


def _config(*, mode: str = "native", judge: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "output_dir": "/runs/rewards/finance",
        "rollouts_dir": "/runs/collect/finance",
        "container": "nemo-gym",
        "dataset_selection_container": "nemo-skills",
        "vllm_container": "vllm",
        "environments": {
            "finance": {
                "config_paths": ["finance.yaml"],
                "agent_name": "finance_agent",
            }
        },
        "rejudge": {
            "mode": mode,
            "num_samples_in_parallel": 16,
            "dependent_jobs": 2,
            "models": {
                "judge": judge
                or {
                    "deployment": "external",
                    "model_type": "openai_model",
                    "model": "judge-v2",
                    "base_url": "https://judge.example/v1",
                    "api_key_env": "JUDGE_API_KEY",
                }
            },
        },
    }


def _render(job: dict[str, Any]) -> str:
    groups = job.get("groups") or [job["group"]]
    commands = []
    for group in groups:
        for command in group.commands:
            inline = command.script.inline
            commands.append(inline() if callable(inline) else str(inline))
    return "\n".join(commands)


def _load_verify_worker(monkeypatch: pytest.MonkeyPatch):
    module_name = "nvflow.integrations.nemo_gym.workers.verify"
    if module_name not in sys.modules:
        monkeypatch.setitem(sys.modules, "nemo_gym", types.ModuleType("nemo_gym"))
        config_types = types.ModuleType("nemo_gym.config_types")
        config_types.__dict__["BaseServerConfig"] = type("BaseServerConfig", (), {})
        monkeypatch.setitem(sys.modules, "nemo_gym.config_types", config_types)
        server_utils = types.ModuleType("nemo_gym.server_utils")
        server_utils.__dict__["ServerClient"] = type("ServerClient", (), {})
        monkeypatch.setitem(sys.modules, "nemo_gym.server_utils", server_utils)
    return importlib.import_module(module_name)


def test_a_resumed_rejudge_may_change_its_parallelism_and_follow_up_jobs() -> None:
    def fingerprint(config: dict[str, Any]) -> str:
        return native_rewards._rejudge_config_sha256(parse_reward_config(config))

    rescheduled = _config()
    rescheduled["rejudge"]["num_samples_in_parallel"] = 64
    rescheduled["rejudge"]["dependent_jobs"] = 3
    regraded = _config()
    regraded["rejudge"]["models"]["judge"]["model"] = "judge-v3"
    params = parse_reward_config(_config())
    legacy = native_rewards._legacy_rejudge_config_sha256(params)
    rendered = "\n".join(_render(job) for job in build_reward_jobs(params, {}, "rewards"))

    assert fingerprint(rescheduled) == fingerprint(_config())
    assert fingerprint(regraded) != fingerprint(_config())
    # What rejudges started before resizing on resume recorded for this configuration.
    assert legacy == "014e41438751065d1306a92368cb7c7a83bf07addb7dcc5d56ee66096b47a4f5"
    assert f"--accept-behavior-sha256 {legacy}" in rendered


def test_native_rejudge_is_derived_resumable_and_never_forced() -> None:
    config = _config()
    config["dataset_selection"] = {
        "source": "/runs/prepare/finance/train.jsonl",
        "mode": "scalar_variance",
        "unprofiled_samples": "exclude",
    }
    config["installation_command"] = "echo INSTALL_GYM"
    config["source_responses_create_params"] = {"max_output_tokens": 32768}
    params = parse_reward_config(config)
    source_layout = NativeGymOutputLayout.collection("/runs/collect/finance")
    layout = NativeGymOutputLayout.rejudge("/runs/rewards/finance")
    jobs = build_reward_jobs(params, {}, "rewards", ["collect-exp"])

    assert [job["name"] for job in jobs] == [
        "rewards-job-0-rejudge-rollouts-attempt-0",
        "rewards-job-0-rejudge-rollouts-attempt-1",
        "rewards-job-0-rejudge-rollouts-attempt-2",
        "rewards-job-1-analyze-rollouts",
        "rewards-job-2-build-training-dataset",
    ]
    assert jobs[0]["dependencies"] == ["collect-exp"]
    assert jobs[1]["dependencies"] == [jobs[0]]
    assert jobs[2]["dependencies"] == [jobs[1]]
    assert jobs[3]["dependencies"] == [jobs[2]]
    assert jobs[4]["dependencies"] == [jobs[3]]
    assert (jobs[0]["group"].hardware.sbatch_kwargs or {}).get("dependency_type") != "afterany"
    assert jobs[1]["group"].hardware.sbatch_kwargs["dependency_type"] == "afterany"
    assert jobs[2]["group"].hardware.sbatch_kwargs["dependency_type"] == "afterany"
    assert jobs[3]["group"].hardware.sbatch_kwargs == {"dependency_type": "afterany"}
    selection_command = _render(jobs[4])
    assert "--responses-create-params-overrides" in selection_command
    assert '"max_output_tokens":32768' in selection_command

    rejudge_command = _render(jobs[0])
    assert "gym eval reverify" in rejudge_command
    assert "--resume" in rejudge_command
    assert "--force" not in rejudge_command
    assert f"--rollouts {source_layout.analyzed_artifacts.rollouts}" in rejudge_command
    assert f"--output {layout.initial_artifacts.rollouts}" in rejudge_command
    assert f"--inputs {layout.initial_artifacts.materialized_inputs}" in rejudge_command
    assert "--model-url" in rejudge_command
    assert "--model-api-key" in rejudge_command
    assert "+policy_model_name=" not in rejudge_command
    assert "+policy_base_url=" not in rejudge_command
    assert "+policy_api_key=" not in rejudge_command
    assert "NVFLOW_REJUDGE_HEAD_PORT=$(find_free_port)" in rejudge_command
    assert '+head_server.host="127.0.0.1"' in rejudge_command
    assert "+head_server.port=${NVFLOW_REJUDGE_HEAD_PORT}" in rejudge_command
    assert "head_server.port=11000" not in rejudge_command
    assert rejudge_command.index("NVFLOW_REJUDGE_HEAD_PORT=$(find_free_port)") < (
        rejudge_command.index("gym eval reverify")
    )
    assert (
        f'cp {source_layout.analyzed_artifacts.materialized_inputs} "$_inputs_tmp"'
        in rejudge_command
    )
    assert "${JUDGE_API_KEY}" in rejudge_command
    assert "NEMO_GYM_MAX_ROLLOUT_ATTEMPTS=3" in rejudge_command
    assert "gym eval profile" not in rejudge_command
    assert "native_sharding bind-behavior" in rejudge_command
    assert str(layout.initial_behavior_manifest) in rejudge_command
    assert str(layout.initial_artifacts.completion) in rejudge_command
    assert f"--source-completion {source_layout.analyzed_artifacts.completion}" in rejudge_command
    assert rejudge_command.index("native_sharding validate") < rejudge_command.index(
        "native_sharding bind-behavior"
    )
    syntax = subprocess.run(["bash", "-n"], input=rejudge_command, text=True, capture_output=True)
    assert syntax.returncode == 0, syntax.stderr

    analyze = _render(jobs[3])
    assert "gym eval profile" in analyze
    assert "gym eval health-check" in analyze
    assert str(layout.analyzed_artifacts.completion) in analyze
    assert "--behavior-sha256" in analyze
    assert "gym eval compare" in analyze
    assert str(layout.compare_analysis_dir) in analyze
    assert "rebuilding invalid rejudge completion evidence" in analyze
    assert f"--reference-completion {source_layout.analyzed_artifacts.completion}" in analyze
    assert jobs[3]["group"].commands[0].script.installation_command == ("echo INSTALL_GYM")
    postprocess_syntax = subprocess.run(
        ["bash", "-n"],
        input=analyze,
        text=True,
        capture_output=True,
    )
    assert postprocess_syntax.returncode == 0, postprocess_syntax.stderr


def test_eval_rejudge_regenerates_evaluation_analysis() -> None:
    params = parse_reward_config(
        _config(),
        report_module="nvfinance.utils.rl.eval_aggregate",
    )
    command = _render(build_reward_jobs(params, {}, "rewards")[-1])

    assert "python3 -m nvfinance.utils.rl.eval_aggregate" in command
    assert "/analysis/evaluation" in command
    assert "gym eval compare" in command


def test_optional_blade_runs_inside_analyze_before_dataset_publication() -> None:
    config = _config()
    config["blade"] = {"enabled": True, "command": "blade-tool analyze"}
    config["dataset_selection"] = {
        "source": "/runs/prepare/finance/train.jsonl",
        "mode": "scalar_variance",
        "unprofiled_samples": "exclude",
    }
    params = parse_reward_config(config)

    jobs = build_reward_jobs(params, {}, "rewards")

    assert [job["name"] for job in jobs[-2:]] == [
        "rewards-job-1-analyze-rollouts",
        "rewards-job-2-build-training-dataset",
    ]
    assert jobs[-1]["dependencies"] == [jobs[-2]]
    analyze = _render(jobs[-2])
    assert "NVFLOW_REWARD_PROFILE=" in analyze
    assert "job-1-analyze-rollouts/outputs" in analyze
    assert "job-1-analyze-rollouts/analysis/blade" in analyze
    assert analyze.index("blade-tool analyze") < analyze.index("gym eval compare")
    assert analyze.index("gym eval compare") < analyze.index("native_sharding complete")


@pytest.mark.parametrize("mode", ["native", "fallback"])
def test_minimal_rejudge_binding_is_resolved_by_gym_runtime(mode: str) -> None:
    config = _config(mode=mode)
    config["environments"] = {
        "dataset-target": {
            "config_paths": [
                "resources_servers/finance_sec_search/configs/finance_sec_search.yaml"
            ],
            "agent_name": "finance_agent",
        }
    }

    params = parse_reward_config(config)
    command = _render(build_reward_jobs(params, {}, "rewards")[0])

    assert params.agent_name == "finance_agent"
    assert "nvflow.integrations.nemo_gym.runtime_binding" in command
    assert str(params.layout.initial_job_dir / "gym_binding_overlay.yaml") in command
    assert "dataset-target.resources_servers" not in command
    if mode == "fallback":
        assert '"$NVFLOW_GYM_RESOURCES_SERVER_INSTANCE"' in command
    syntax = subprocess.run(["bash", "-n"], input=command, text=True, capture_output=True)
    assert syntax.returncode == 0, syntax.stderr


def test_shared_dataset_selection_uses_rejudged_native_profile() -> None:
    config = _config()
    config["dataset_selection"] = {
        "source": "/runs/prepare/finance/train.jsonl",
        "mode": "scalar_variance",
        "min_std": 1e-6,
        "unprofiled_samples": "exclude",
        "success_threshold": 1.0,
    }
    params = parse_reward_config(config)

    jobs = build_reward_jobs(params, {}, "rewards")
    command = _render(jobs[-1])
    analyze = _render(jobs[-2])

    assert jobs[-1]["name"] == "rewards-job-2-build-training-dataset"
    assert "workers.build_training_dataset" in command
    assert "--materialized-inputs" in command
    assert params.analyzed_artifacts.reward_profiling.as_posix() in command
    assert params.dataset_source is not None
    assert params.dataset_source in command
    assert "difficulty.jsonl" not in command
    assert "dataset selection requires at least two Gym repeats" in analyze


def test_rejudge_without_selection_removes_stale_dataset_outputs() -> None:
    config = _config()
    params = parse_reward_config(config)
    postprocess = _render(build_reward_jobs(params, {}, "rewards")[-1])

    assert "rm -f" in postprocess
    assert f"{params.output_dir}/profiled.jsonl" in postprocess
    assert f"{params.output_dir}/train.jsonl" in postprocess
    assert "selection_report.json" in postprocess


def test_unknown_safety_fallback_uses_materialized_inputs_and_preserves_source() -> None:
    params = parse_reward_config(_config(mode="fallback"))
    command = _render(build_reward_jobs(params, {}, "rewards")[0])

    assert "gym env start" in command
    assert "NVFLOW_REJUDGE_HEAD_PORT=$(find_free_port)" in command
    assert "+head_server.port=${NVFLOW_REJUDGE_HEAD_PORT}" in command
    assert "head_server.port=11000" not in command
    assert "nvflow.integrations.nemo_gym.workers.verify" in command
    assert params.source.materialized_inputs.as_posix() in command
    assert params.artifacts.materialized_inputs.as_posix() in command
    assert params.source.rollouts.as_posix() in command
    assert params.artifacts.rollouts.as_posix() in command
    assert "gym eval aggregate" in command
    environment_start = next(line for line in command.splitlines() if "gym env start" in line)
    aggregate = next(
        line for line in command.splitlines() if line.strip().startswith("gym eval aggregate")
    )
    assert "+policy_model_name=" not in environment_start
    assert "+policy_base_url=" not in environment_start
    assert "+policy_api_key=" not in environment_start
    assert "+policy_model_name=" in aggregate
    assert "+policy_base_url=" in aggregate
    assert "+policy_api_key=" in aggregate
    assert "--force" not in command
    syntax = subprocess.run(["bash", "-n"], input=command, text=True, capture_output=True)
    assert syntax.returncode == 0, syntax.stderr


@pytest.mark.parametrize(
    ("judge", "expected_gpus"),
    [
        (
            {
                "deployment": "external",
                "model_type": "openai_model",
                "model": "judge",
                "base_url": "https://judge.example/v1",
            },
            0,
        ),
        (
            {
                "deployment": "gym",
                "model": "/models/judge",
                "num_gpus": 4,
                "gym_model": {
                    "return_token_id_information": False,
                    "uses_reasoning_parser": False,
                    "vllm_serve_env_vars": {"VLLM_RAY_DP_PACK_STRATEGY": "strict"},
                    "vllm_serve_kwargs": {
                        "data_parallel_size": 1,
                        "tensor_parallel_size": 4,
                        "pipeline_parallel_size": 1,
                    },
                },
            },
            4,
        ),
        (
            {
                "deployment": "nvflow",
                "model": "/models/judge",
                "num_gpus": 4,
                "gym_model": {
                    "vllm_serve_kwargs": {"tensor_parallel_size": 4},
                },
            },
            4,
        ),
    ],
)
def test_reward_endpoint_uses_same_hosted_native_resource_contract(
    monkeypatch: pytest.MonkeyPatch,
    judge: dict[str, Any],
    expected_gpus: int,
) -> None:
    class _Server:
        inline = "serve"
        port = 12345

        def set_inline(self, value: str) -> None:
            self.inline = value

        def hostname_ref(self) -> str:
            return "judge-host"

    monkeypatch.setattr(
        native_rewards,
        "make_server_script",
        lambda *args, **kwargs: _Server(),
    )
    params = parse_reward_config(_config(judge=judge))
    [first, *_] = build_reward_jobs(params, {}, "rewards")

    assert first["group"].hardware.num_gpus == expected_gpus


def test_completed_rejudge_is_validated_inside_guarded_attempt() -> None:
    config = _config()
    config["rejudge"]["dependent_jobs"] = 0
    jobs = build_reward_jobs(parse_reward_config(config), {}, "rewards")

    assert [job["name"] for job in jobs] == [
        "rewards-job-0-rejudge-rollouts-attempt-0",
        "rewards-job-1-analyze-rollouts",
    ]
    attempt = _render(jobs[0])
    assert (
        "if [ -f /runs/rewards/finance/jobs/job-1-analyze-rollouts/completion.json ]; then"
    ) in attempt
    assert "removing invalid completion marker" in attempt
    analyze = _render(jobs[1])
    assert "gym eval profile" in analyze
    assert "gym eval compare" in analyze


def test_rejudge_without_retry_chain_inherits_afterok_policy() -> None:
    config = _config()
    config["rejudge"]["dependent_jobs"] = 0
    jobs = build_reward_jobs(parse_reward_config(config), {}, "rewards")

    assert jobs[0]["group"].hardware.sbatch_kwargs is None
    assert jobs[-1]["group"].hardware.sbatch_kwargs is None


def test_rejudge_completion_fingerprint_changes_with_judge_behavior() -> None:
    first = parse_reward_config(_config())
    changed_config = _config()
    changed_config["rejudge"]["models"]["judge"]["gym_model"] = {
        "responses_create_params": {"temperature": 0.2}
    }
    changed = parse_reward_config(changed_config)

    assert native_rewards._rejudge_config_sha256(  # noqa: SLF001
        first
    ) != native_rewards._rejudge_config_sha256(changed)  # noqa: SLF001


def test_rejudge_analysis_fingerprint_tracks_report_and_blade() -> None:
    original = parse_reward_config(_config(), report_module="reports.v1")
    changed_config = _config()
    changed_config["blade"] = {"enabled": True, "command": "blade analyze"}
    changed = parse_reward_config(changed_config, report_module="reports.v2")

    assert native_rewards._rejudge_config_sha256(  # noqa: SLF001
        original
    ) == native_rewards._rejudge_config_sha256(changed)  # noqa: SLF001
    assert native_rewards._rejudge_analysis_sha256(  # noqa: SLF001
        original
    ) != native_rewards._rejudge_analysis_sha256(changed)  # noqa: SLF001


@pytest.mark.parametrize(
    ("judge", "container_affects_fingerprint"),
    [
        (
            {
                "deployment": "external",
                "model_type": "openai_model",
                "model": "judge",
                "base_url": "https://judge.example/v1",
            },
            False,
        ),
        (
            {
                "deployment": "gym",
                "model": "/models/judge",
                "num_gpus": 1,
                "gym_model": {
                    "return_token_id_information": False,
                    "uses_reasoning_parser": False,
                    "vllm_serve_env_vars": {
                        "VLLM_RAY_DP_PACK_STRATEGY": "strict",
                    },
                    "vllm_serve_kwargs": {
                        "tensor_parallel_size": 1,
                        "pipeline_parallel_size": 1,
                        "data_parallel_size": 1,
                    },
                },
            },
            False,
        ),
        (
            {
                "deployment": "nvflow",
                "server_type": "sglang",
                "model": "/models/judge",
                "num_gpus": 1,
            },
            False,
        ),
        (
            {
                "deployment": "nvflow",
                "model": "/models/judge",
                "num_gpus": 1,
            },
            True,
        ),
        (
            {
                "deployment": "nvflow",
                "container": "judge-vllm",
                "model": "/models/judge",
                "num_gpus": 1,
            },
            False,
        ),
    ],
)
def test_rejudge_fingerprint_hashes_only_effective_vllm_container(
    judge: dict[str, Any],
    container_affects_fingerprint: bool,
) -> None:
    first_config = _config(judge=judge)
    first_config["vllm_container"] = "vllm-a"
    second_config = deepcopy(first_config)
    second_config["vllm_container"] = "vllm-b"

    first = native_rewards._rejudge_config_sha256(parse_reward_config(first_config))
    second = native_rewards._rejudge_config_sha256(parse_reward_config(second_config))

    assert (first != second) is container_affects_fingerprint


def test_live_rejudge_plan_keeps_guarded_judge_attempt() -> None:
    class _Filesystem:
        def batch_exists(self, paths: list[str]) -> dict[str, bool]:
            return dict.fromkeys(paths, True)

    planned = native_rewards.rejudge(
        _config(),
        "cluster",
        "rewards",
        inspect_live_state=True,
        filesystem=cast(Any, _Filesystem()),
    )

    assert [job["name"] for job in planned] == [
        "rewards-job-0-rejudge-rollouts-attempt-0",
        "rewards-job-0-rejudge-rollouts-attempt-1",
        "rewards-job-0-rejudge-rollouts-attempt-2",
        "rewards-job-1-analyze-rollouts",
    ]


@pytest.mark.parametrize("server_nodes", [1, 2, 4, 8])
def test_live_rejudge_plan_matches_static_plan_without_building_executors(
    monkeypatch: pytest.MonkeyPatch,
    server_nodes: int,
) -> None:
    import nemo_skills
    import nemo_skills.pipeline.utils.declarative as declarative
    import nemo_skills.pipeline.utils.exp as exp

    from nvflow.integrations.nemo_skills import stage_plan_from_jobs

    def no_executor(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("planning built a scheduler executor")

    monkeypatch.setattr(exp, "get_executor", no_executor)
    monkeypatch.setattr(declarative, "get_executor", no_executor)

    class _MissingFilesystem:
        def batch_exists(self, paths: list[str]) -> dict[str, bool]:
            return dict.fromkeys(paths, False)

        def batch_completion_matches(self, probes: list[dict[str, Any]]) -> list[bool]:
            return [False] * len(probes)

    judge = {
        "deployment": "nvflow",
        "model": "/models/judge",
        "num_gpus": 8,
        "server_nodes": server_nodes,
    }

    def plan(*, live: bool) -> tuple[Any, ...]:
        jobs = native_rewards.rejudge(
            _config(judge=judge),
            "cluster",
            "multi-node",
            inspect_live_state=live,
            filesystem=cast(Any, _MissingFilesystem()) if live else None,
        )
        return stage_plan_from_jobs("rejudge", jobs, output_dir="/runs/rewards").tasks

    live = plan(live=True)

    assert live == plan(live=False)
    assert server_nodes in {resource.nodes for task in live for resource in task.resources}
    commands = "\n".join(command.command for task in live for command in task.commands)
    assert "/nemo_run/code" in commands
    assert str(Path(nemo_skills.__file__).parents[1]) not in commands


def test_live_rejudge_prunes_only_source_bound_validated_output() -> None:
    params = parse_reward_config(_config())

    class _ValidatedFilesystem:
        def batch_completion_matches(self, probes):
            assert [probe["marker"] for probe in probes] == [
                params.source.completion.as_posix(),
                params.analyzed_artifacts.completion.as_posix(),
                params.rejudge_marker.as_posix(),
            ]
            assert all(
                probe["behavior_manifest"] == params.behavior_manifest.as_posix()
                for probe in probes[1:]
            )
            return [True, False, True]

    assert native_rewards._rejudge_is_complete(
        params,
        cast(Any, _ValidatedFilesystem()),
    )
    jobs = build_reward_jobs(params, {}, "rewards", rejudge_complete=True)
    assert [job["name"] for job in jobs] == ["rewards-job-1-analyze-rollouts"]


def test_reward_recomputation_requires_explicit_safety_mode_and_dedicated_judge() -> None:
    missing_mode = _config()
    del missing_mode["rejudge"]["mode"]
    with pytest.raises(NativeRolloutConfigError, match="rejudge.mode"):
        parse_reward_config(missing_mode)

    policy_judge = _config(judge={"reuse": "policy"})
    with pytest.raises(NativeRolloutConfigError, match="dedicated judge"):
        parse_reward_config(policy_judge)


def test_legacy_repeat_axis_is_rejected() -> None:
    config = _config()
    config["rejudge"]["num_repeats"] = 8

    with pytest.raises(NativeRolloutConfigError, match="unknown rejudge field"):
        parse_reward_config(config)


def test_dataset_selection_source_requires_policy() -> None:
    config = _config()
    config["dataset_selection"] = {"source": "/data/train.jsonl"}

    with pytest.raises(NativeRolloutConfigError, match="explicit selection policy"):
        parse_reward_config(config)


def test_removed_postprocess_container_points_to_dataset_selection_container() -> None:
    config = _config()
    config["postprocess_container"] = "nemo-skills"

    with pytest.raises(NativeRolloutConfigError, match="dataset_selection_container"):
        parse_reward_config(config)


def test_managed_rejudge_vllm_container_must_come_from_workflow_config() -> None:
    config = _config(
        judge={
            "deployment": "nvflow",
            "model": "/models/judge",
            "num_gpus": 1,
        }
    )
    config.pop("vllm_container")

    with pytest.raises(NativeRolloutConfigError, match="'vllm_container' is required"):
        parse_reward_config(config)

    external = _config()
    external.pop("vllm_container")
    assert parse_reward_config(external).vllm_container is None


def test_fallback_worker_joins_materialized_question_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verify_worker = _load_verify_worker(monkeypatch)

    materialized = tmp_path / "inputs.jsonl"
    rollouts = tmp_path / "rollouts.jsonl"
    output = tmp_path / "reverified.jsonl"
    failures = tmp_path / "reverified_failures.jsonl"
    identity = {"_ng_task_index": 4, "_ng_rollout_index": 2}
    materialized.write_text(
        json.dumps(
            {
                **identity,
                "question": "What was revenue?",
                "expected_answer": "$10",
                "responses_create_params": {"input": [{"role": "user", "content": "question"}]},
            }
        )
        + "\n"
    )
    rollouts.write_text(
        json.dumps(
            {
                **identity,
                "response": {"output": [{"role": "assistant", "content": "$10"}]},
                "ng_trajectory": {
                    "schema_version": 1,
                    "rollout_id": "task-4-repeat-2",
                    "turns": [{"role": "assistant", "content": "$10"}],
                },
                "reward": 0.0,
            }
        )
        + "\n"
    )
    payloads: list[dict[str, Any]] = []

    class Response:
        status = 200

        async def json(self) -> dict[str, Any]:
            return {"reward": 1.0, "judge_evaluations": [{"rating": 1}]}

    class Client:
        async def post(self, **kwargs: Any) -> Response:
            payloads.append(kwargs["json"])
            return Response()

    async def ready(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(verify_worker, "_wait_for_server_client", lambda *args: Client())
    monkeypatch.setattr(verify_worker, "_wait_for_verify_endpoint", ready)
    arguments = (
        str(materialized),
        str(rollouts),
        str(output),
        str(failures),
        "127.0.0.1",
        11000,
        "finance_resources",
        1,
    )

    asyncio.run(verify_worker.verify_rollouts(*arguments))
    asyncio.run(verify_worker.verify_rollouts(*arguments))

    [result] = [json.loads(line) for line in output.read_text().splitlines()]
    assert result["reward"] == 1.0
    assert result | identity == result
    assert result["response"] == {"output": [{"role": "assistant", "content": "$10"}]}
    assert result["ng_trajectory"]["rollout_id"] == "task-4-repeat-2"
    assert payloads[0]["question"] == "What was revenue?"
    assert len(payloads) == 1
    assert not output.with_name(f".{output.name}.join.sqlite3").exists()


def test_unique_prompts_apply_to_fallback_rejudges_and_change_the_fingerprint() -> None:
    fresh = _config(mode="fallback")
    fresh["rejudge"]["unique_prompts"] = True
    params = parse_reward_config(fresh)
    command = _render(build_reward_jobs(params, {}, "rewards")[0])
    plain = parse_reward_config(_config(mode="fallback"))

    assert params.unique_prompts
    assert command.count("--unique-prompts") == 1
    assert "+merge_shards=false" in command
    assert "--unique-prompts" not in _render(build_reward_jobs(plain, {}, "rewards")[0])
    assert native_rewards._rejudge_config_sha256(params) != (
        native_rewards._rejudge_config_sha256(plain)
    )
    # No rejudge recorded before resizing on resume had unique prompts.
    every_job = "\n".join(_render(job) for job in build_reward_jobs(params, {}, "rewards"))
    plain_jobs = "\n".join(_render(job) for job in build_reward_jobs(plain, {}, "rewards"))
    assert native_rewards._legacy_rejudge_config_sha256(params) is None
    assert "--accept-behavior-sha256" not in every_job
    assert "--accept-behavior-sha256" in plain_jobs
    native = _config()
    native["rejudge"]["unique_prompts"] = True
    with pytest.raises(NativeRolloutConfigError, match="applies to fallback mode"):
        parse_reward_config(native)


def test_unique_prompts_retry_a_failed_verification_with_a_new_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verify_worker = _load_verify_worker(monkeypatch)
    identity = {"_ng_task_index": 0, "_ng_rollout_index": 0}
    question = {"input": [{"role": "user", "content": "What was revenue?"}]}
    materialized = tmp_path / "inputs.jsonl"
    rollouts = tmp_path / "rollouts.jsonl"
    output = tmp_path / "reverified.jsonl"
    failures = tmp_path / "reverified_failures.jsonl"
    materialized.write_text(json.dumps({**identity, "responses_create_params": question}) + "\n")
    rollouts.write_text(json.dumps({**identity, "response": {"output": []}}) + "\n")
    sent: list[str] = []

    class Response:
        status = 200

        def __init__(self, body: dict[str, Any]) -> None:
            self.body = body

        async def json(self) -> dict[str, Any]:
            failed: dict[str, Any] = (
                {"_ng_failure_class": "judge_failed"} if len(sent) == 1 else {"reward": 1.0}
            )
            return {
                "responses_create_params": deepcopy(self.body["responses_create_params"]),
                **failed,
            }

    class Client:
        async def post(self, **kwargs: Any) -> Response:
            sent.append(kwargs["json"]["responses_create_params"]["input"][0]["content"])
            return Response(kwargs["json"])

    async def instantly(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(verify_worker, "_wait_for_server_client", lambda *args: Client())
    monkeypatch.setattr(verify_worker, "_wait_for_verify_endpoint", instantly)
    monkeypatch.setattr(verify_worker.asyncio, "sleep", instantly)

    asyncio.run(
        verify_worker.verify_rollouts(
            str(materialized),
            str(rollouts),
            str(output),
            str(failures),
            "127.0.0.1",
            11000,
            "finance_resources",
            1,
            unique_prompts=True,
        )
    )

    assert len(sent) == 2 and sent[0] != sent[1]
    assert all(
        text.startswith("Request ") and text.endswith("\n\nWhat was revenue?") for text in sent
    )
    [result] = [json.loads(line) for line in output.read_text().splitlines()]
    assert result["reward"] == 1.0
    assert result["responses_create_params"] == question
    assert failures.read_text() == ""


def test_a_request_line_marks_only_a_copy_of_the_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    verify_worker = _load_verify_worker(monkeypatch)
    response = {"output": [{"type": "message"}]}
    request = {
        "responses_create_params": {"input": [{"role": "user", "content": "What was revenue?"}]},
        "response": response,
    }

    marked, line = verify_worker.with_request_line(request)

    assert marked["responses_create_params"]["input"][0]["content"] == (
        f"{line}\n\nWhat was revenue?"
    )
    assert request["responses_create_params"]["input"][0]["content"] == "What was revenue?"
    assert marked["response"] is response
    verify_worker.without_request_line(marked, line)
    assert marked["responses_create_params"] == request["responses_create_params"]


def test_a_request_without_a_text_question_is_sent_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verify_worker = _load_verify_worker(monkeypatch)
    parts = [{"type": "input_text", "text": "What was revenue?"}]
    request = {"responses_create_params": {"input": [{"role": "user", "content": parts}]}}
    echoed = {"responses_create_params": {"input": [{"role": "user", "content": "\n\nkept"}]}}

    assert verify_worker.with_request_line(request) == (request, "")
    verify_worker.without_request_line(echoed, "")
    assert echoed["responses_create_params"]["input"][0]["content"] == "\n\nkept"


def test_fallback_worker_routes_judge_failures_without_reusing_stale_reward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verify_worker = _load_verify_worker(monkeypatch)
    identity = {"_ng_task_index": 0, "_ng_rollout_index": 0}
    materialized = tmp_path / "inputs.jsonl"
    rollouts = tmp_path / "rollouts.jsonl"
    output = tmp_path / "reverified.jsonl"
    failures = tmp_path / "reverified_failures.jsonl"
    materialized.write_text(json.dumps({**identity, "question": "q"}) + "\n")
    rollouts.write_text(
        json.dumps(
            {
                **identity,
                "response": {"output": []},
                "reward": 1.0,
                "rubric_old": 99,
            }
        )
        + "\n"
    )

    class Response:
        status = 200

        async def json(self) -> dict[str, Any]:
            return {
                "reward": 0.0,
                "_ng_failure_class": "judge_failed",
                "_ng_failure_judge_error": "unavailable",
            }

    class Client:
        async def post(self, **_kwargs: Any) -> Response:
            return Response()

    async def ready(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(verify_worker, "_wait_for_server_client", lambda *_args: Client())
    monkeypatch.setattr(verify_worker, "_wait_for_verify_endpoint", ready)

    with pytest.raises(RuntimeError, match="rejudge requests failed"):
        asyncio.run(
            verify_worker.verify_rollouts(
                str(materialized),
                str(rollouts),
                str(output),
                str(failures),
                "127.0.0.1",
                11000,
                "finance_resources",
                1,
            )
        )

    assert output.read_text() == ""
    [failed] = [json.loads(line) for line in failures.read_text().splitlines()]
    assert failed["_ng_failure_class"] == "judge_failed"
    assert failed["response"] == {"output": []}
    assert "rubric_old" not in failed


def test_fallback_worker_flushes_each_completed_rejudge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verify_worker = _load_verify_worker(monkeypatch)
    materialized = tmp_path / "inputs.jsonl"
    rollouts = tmp_path / "rollouts.jsonl"
    output = tmp_path / "reverified.jsonl"
    failures = tmp_path / "reverified_failures.jsonl"
    input_rows = [
        {
            "_ng_task_index": index,
            "_ng_rollout_index": 0,
            "question": question,
        }
        for index, question in enumerate(("fast", "blocked"))
    ]
    rollout_rows = [
        {
            "_ng_task_index": row["_ng_task_index"],
            "_ng_rollout_index": 0,
            "response": {"output": []},
        }
        for row in input_rows
    ]
    materialized.write_text("".join(json.dumps(row) + "\n" for row in input_rows))
    rollouts.write_text("".join(json.dumps(row) + "\n" for row in rollout_rows))

    class Response:
        status = 200

        async def json(self) -> dict[str, Any]:
            return {"reward": 1.0}

    class Client:
        async def post(self, **kwargs: Any) -> Response:
            if kwargs["json"]["question"] == "blocked":
                await asyncio.Event().wait()
            return Response()

    async def ready(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(verify_worker, "_wait_for_server_client", lambda *args: Client())
    monkeypatch.setattr(verify_worker, "_wait_for_verify_endpoint", ready)

    async def interrupt() -> None:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                verify_worker.verify_rollouts(
                    str(materialized),
                    str(rollouts),
                    str(output),
                    str(failures),
                    "127.0.0.1",
                    11000,
                    "finance_resources",
                    2,
                ),
                timeout=0.2,
            )

    asyncio.run(interrupt())

    persisted = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["_ng_task_index"] for row in persisted] == [0]
    assert not output.with_name(f".{output.name}.join.sqlite3").exists()
