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
"""Semantic plan tests for the native Gym rollout control plane."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

pytestmark = pytest.mark.requires_nemo
pytest.importorskip("nemo_skills")

import nvflow.integrations.nemo_gym.native_rollout as native_rollout  # noqa: E402
from nvflow.integrations.nemo_gym.generation import build_rollout_config  # noqa: E402
from nvflow.integrations.nemo_gym.job_scripts import (  # noqa: E402
    _build_port_read_preamble,
    _wrap_server_with_dynamic_port,
)
from nvflow.integrations.nemo_gym.native_contract import NativeGymOutputLayout  # noqa: E402
from nvflow.integrations.nemo_gym.native_rollout import (  # noqa: E402
    NativeRolloutConfigError,
    build_rollout_jobs,
    parse_rollout_config,
)
from nvflow.integrations.nemo_gym.native_sharding import prepare_input_shard  # noqa: E402
from nvflow.integrations.nemo_gym.vllm_judge import build_vllm_server_args  # noqa: E402


def _config(**rollout_overrides: Any) -> dict[str, Any]:
    rollout = {
        "input_data": "/data/finance.jsonl",
        "num_repeats": 3,
        "num_chunks": 2,
        "dependent_jobs": 1,
        "max_num_samples": 12,
        "num_samples_in_parallel": 32,
        "responses_create_params": {
            "temperature": 0.7,
            "metadata": {"purpose": "difficulty"},
        },
        "models": {
            "policy": {
                "deployment": "external",
                "model": "/models/policy",
                "base_url": "https://policy.example/v1",
            },
        },
    }
    rollout.update(rollout_overrides)
    return {
        "output_dir": "/outputs/collect/finance",
        "container": "nemo-gym",
        "dataset_selection_container": "nemo-gym",
        "vllm_container": "vllm",
        "environments": {
            "finance": {
                "config_paths": ["env.yaml", "sandbox.yaml"],
                "agent_name": "finance_agent",
            }
        },
        "rollout": rollout,
    }


_POLICY_DEPLOYMENTS = [
    {
        "deployment": "gym",
        "model": "/models/policy",
        "num_gpus": 2,
        "gym_model": {"vllm_serve_kwargs": {"tensor_parallel_size": 2}},
    },
    {
        "deployment": "nvflow",
        "model": "/models/policy",
        "num_gpus": 4,
    },
    {
        "deployment": "external",
        "model": "policy",
        "base_url": "https://policy.example/v1",
    },
]


def test_sdg_outer_chunks_are_independent_of_gym_repeats() -> None:
    config = build_rollout_config(
        input_file="/data/input.jsonl",
        output_dir="/data/output",
        gym_config_paths=["environment.yaml"],
        gym_agent_name="agent",
        container="nemo-gym",
        vllm_container="vllm",
        installation_command=None,
        model_path="/models/policy",
        num_gpus=4,
        num_chunks=2,
        num_repeats=5,
    )

    assert config["rollout"]["num_chunks"] == 2
    assert config["rollout"]["num_repeats"] == 5


def _groups(job: dict[str, Any]) -> list[Any]:
    return list(job.get("groups") or [job["group"]])


def _rendered_commands(job: dict[str, Any]) -> list[str]:
    rendered = []
    for group in _groups(job):
        for command in group.commands:
            inline = command.script.inline
            rendered.append(inline() if callable(inline) else str(inline))
    return rendered


def test_topology_keeps_repeats_inside_each_chunk_and_chains_attempts() -> None:
    params = parse_rollout_config(_config())
    jobs = build_rollout_jobs(params, {}, "native", ["prepare-exp"])

    assert [job["name"] for job in jobs] == [
        "native-job-0-collect-rollouts-chunk-0-attempt-0",
        "native-job-0-collect-rollouts-chunk-0-attempt-1",
        "native-job-0-collect-rollouts-chunk-1-attempt-0",
        "native-job-0-collect-rollouts-chunk-1-attempt-1",
        "native-job-1-analyze-rollouts",
    ]
    assert jobs[0]["dependencies"] == ["prepare-exp"]
    assert jobs[1]["dependencies"] == [jobs[0]]
    assert jobs[2]["dependencies"] == ["prepare-exp"]
    assert jobs[3]["dependencies"] == [jobs[2]]
    assert jobs[4]["dependencies"] == [jobs[1], jobs[3]]
    assert all(
        (group.hardware.sbatch_kwargs or {}).get("dependency_type") != "afterany"
        for group in _groups(jobs[0])
    )
    assert all(
        group.hardware.sbatch_kwargs["dependency_type"] == "afterany" for group in _groups(jobs[1])
    )
    assert _groups(jobs[4])[0].hardware.sbatch_kwargs == {"dependency_type": "afterany"}

    first = "\n".join(_rendered_commands(jobs[0]))
    assert "gym eval run" in first
    assert "--num-repeats 3" in first
    assert "--resume" in first
    assert "NEMO_GYM_MAX_ROLLOUT_ATTEMPTS=3" in first
    assert "+num_repeats_add_seed=true" in first
    assert "+route_failures_to_sidecar=true" in first
    assert "NVFLOW_ROLLOUT_HEAD_PORT=$(find_free_port)" in first
    assert '+head_server.host="127.0.0.1"' in first
    assert "+head_server.port=${NVFLOW_ROLLOUT_HEAD_PORT}" in first
    assert "head_server.port=11000" not in first
    assert first.index("NVFLOW_ROLLOUT_HEAD_PORT=$(find_free_port)") < first.index("gym eval run")
    assert "--chunk-index 0" in first
    assert "--num-chunks 2" in first
    assert "--max-num-samples 12" in first
    assert "--behavior-sha256" in first
    assert "output-rs" not in first
    assert ".complete" not in first


def test_collection_uses_a_generated_dataset_config_for_e2e_serving() -> None:
    params = parse_rollout_config(_config())
    command = "\n".join(_rendered_commands(build_rollout_jobs(params, {}, "native")[0]))

    assert (
        "/outputs/collect/finance/jobs/job-0-collect-rollouts/chunks/chunk-0/"
        "outputs/rollouts/preprocessed_datasets/train.jsonl" in command
    )
    assert (
        "--config /outputs/collect/finance/jobs/job-0-collect-rollouts/"
        "chunks/chunk-0/outputs/dataset_config.yaml"
    ) in command
    gym_run = command[command.index("gym eval run") :]
    assert "--input" not in gym_run
    assert "--agent finance_agent" in gym_run
    assert "--split train" in gym_run
    assert "+reuse_existing_data_preparation=true" in gym_run
    assert "datasets=[{" not in gym_run
    assert "--disable-aggregation" in gym_run
    assert "--no-health-check" in gym_run
    assert command.count("--shard-manifest") == 3


def test_minimal_provider_config_defers_topology_to_gym_runtime() -> None:
    config = _config()
    config["environments"] = {
        "dataset-target": {
            "config_paths": [
                "resources_servers/finance_sec_search/configs/finance_sec_search.yaml"
            ],
            "agent_name": "finance_agent",
        }
    }

    params = parse_rollout_config(config)
    command = "\n".join(_rendered_commands(build_rollout_jobs(params, {}, "native")[0]))

    assert params.agent_name == "finance_agent"
    assert "nvflow.integrations.nemo_gym.runtime_binding" in command
    assert "--agent finance_agent" in command
    assert "dataset-target.resources_servers" not in command
    syntax = subprocess.run(["bash", "-n"], input=command, text=True, capture_output=True)
    assert syntax.returncode == 0, syntax.stderr


def test_policy_as_judge_route_is_generated_from_gym_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def capture_binding(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "true"

    monkeypatch.setattr(
        native_rollout,
        "render_runtime_binding_command",
        capture_binding,
    )

    native_rollout._write_dataset_config(parse_rollout_config(_config()), 0)

    assert captured["judge_model_name"] == "policy_model"


def test_collection_refuses_orphan_rollouts_without_identity_manifest() -> None:
    params = parse_rollout_config(_config())
    command = "\n".join(_rendered_commands(build_rollout_jobs(params, {}, "native")[0]))

    assert "rollout artifacts exist without identity manifest" in command
    assert "/chunks/chunk-0/manifest.json" in command
    assert "/chunks/chunk-0/outputs/rollouts.jsonl" in command


def test_collection_rejects_legacy_flat_output_tree() -> None:
    params = parse_rollout_config(_config())
    command = "\n".join(_rendered_commands(build_rollout_jobs(params, {}, "native")[0]))

    assert "legacy flat Gym artifacts exist" in command
    assert f"{params.output_dir}/rollouts.jsonl" in command
    assert f"{params.output_dir}/train.jsonl.done" in command


def test_analyze_job_validates_every_shard_and_uses_native_analysis_commands() -> None:
    config = _config()
    config["installation_command"] = "echo INSTALL_GYM"
    params = parse_rollout_config(
        config,
        report_module="finance.report",
    )
    analyze_job = build_rollout_jobs(params, {}, "native")[-1]
    group = _groups(analyze_job)[0]
    analyze = group.commands[0].script.cmd

    assert analyze.count("native_sharding prepare") == 2
    assert analyze.count("native_sharding validate") == 3  # two shards + root guard
    assert analyze.count("--shard-manifest") == 6
    assert _groups(analyze_job)[0].hardware.num_gpus == 0
    assert "gym env start" in analyze
    assert "model_endpoint_readiness_timeout_seconds=0" in analyze
    chunks = "/outputs/collect/finance/jobs/job-0-collect-rollouts/chunks"
    root = "/outputs/collect/finance/jobs/job-1-analyze-rollouts/outputs/rollouts.jsonl"
    join = f'cat {chunks}/chunk-0/outputs/rollouts.jsonl {chunks}/chunk-1/outputs/rollouts.jsonl > "$_tmp"'
    [aggregate] = [line for line in analyze.splitlines() if "gym eval aggregate" in line]
    [environment] = [line for line in analyze.splitlines() if "gym env start" in line]
    assert analyze.index(join) < analyze.index("gym eval aggregate")
    assert f"--input-glob {root} --output {root}" in aggregate
    assert "+merge_shards=false" in aggregate
    assert "merge_shards" not in environment
    assert "gym eval profile" in analyze
    assert "gym eval health-check" in analyze
    assert "python3 -m finance.report" in analyze
    assert "/jobs/job-1-analyze-rollouts/analysis/evaluation" in analyze
    assert "/jobs/job-1-analyze-rollouts/completion.json" in analyze
    assert "rebuilding invalid root rollout completion evidence" in analyze
    root_guard = analyze.split("else", 1)[0]
    assert "--behavior-sha256" in root_guard
    assert group.log_dir.endswith("/jobs/job-1-analyze-rollouts/logs")
    assert group.commands[0].script.installation_command == "echo INSTALL_GYM"
    assert analyze.index("gym eval profile") < analyze.index("finance.report")
    syntax = subprocess.run(["bash", "-n"], input=analyze, text=True, capture_output=True)
    assert syntax.returncode == 0, syntax.stderr


def test_a_gym_readiness_wait_reaches_the_rollouts_but_never_the_aggregation() -> None:
    config = _config()
    config["environments"]["finance"]["gym"] = {"model_endpoint_readiness_timeout_seconds": 3600}
    jobs = build_rollout_jobs(parse_rollout_config(config), {}, "native")

    collect = "\n".join(_rendered_commands(jobs[0]))
    analyze = _groups(jobs[-1])[0].commands[0].script.cmd

    assert "model_endpoint_readiness_timeout_seconds=3600" in collect
    assert "model_endpoint_readiness_timeout_seconds=0" in analyze
    assert "model_endpoint_readiness_timeout_seconds=3600" not in analyze


@pytest.mark.parametrize("policy", _POLICY_DEPLOYMENTS)
def test_live_plan_keeps_guarded_attempts_for_completed_chunks(
    policy: dict[str, Any],
) -> None:
    class _CompletedFilesystem:
        def batch_exists(self, paths: list[str]) -> dict[str, bool]:
            return dict.fromkeys(paths, True)

    config = _config(models={"policy": deepcopy(policy)})
    planned = native_rollout.rollout(
        config,
        "test",
        "execute-resume",
        inspect_live_state=True,
        filesystem=cast(Any, _CompletedFilesystem()),
    )

    assert len([job for job in planned if "-attempt-" in job["name"]]) == 4
    attempt = "\n".join(_rendered_commands(planned[0]))
    assert "completion.json" in attempt
    assert "native_sharding validate" in attempt
    assert "removing invalid completion marker" in attempt


@pytest.mark.parametrize("server_nodes", [1, 2, 4, 8])
def test_live_plan_matches_static_plan_without_building_executors(
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

    policy = {
        "deployment": "nvflow",
        "model": "/models/policy",
        "num_gpus": 8,
        "server_nodes": server_nodes,
    }

    def plan(*, live: bool) -> tuple[Any, ...]:
        jobs = native_rollout.rollout(
            _config(models={"policy": policy}),
            "test",
            "multi-node",
            inspect_live_state=live,
            filesystem=cast(Any, _MissingFilesystem()) if live else None,
        )
        return stage_plan_from_jobs("collect_rollouts", jobs, output_dir="/outputs/collect").tasks

    live = plan(live=True)

    assert live == plan(live=False)
    assert server_nodes in {resource.nodes for task in live for resource in task.resources}
    commands = "\n".join(command.command for task in live for command in task.commands)
    assert "/nemo_run/code" in commands
    assert str(Path(nemo_skills.__file__).parents[1]) not in commands


def test_live_resume_prunes_only_content_validated_chunks() -> None:
    params = parse_rollout_config(_config())

    class _ValidatedFilesystem:
        def batch_completion_matches(self, probes):
            assert len(probes) == 2
            assert all(probe["behavior_sha256"] for probe in probes)
            assert all(probe["source"] == params.input_data for probe in probes)
            assert [probe["shard_manifest"] for probe in probes] == [
                params.layout.chunk_manifest(index).as_posix() for index in range(2)
            ]
            assert all(
                probe["manifest_behavior_sha256"] == native_rollout._rollout_config_sha256(params)
                for probe in probes
            )
            return [True, False]

    completed = native_rollout._completed_chunks(
        params,
        cast(Any, _ValidatedFilesystem()),
    )
    jobs = build_rollout_jobs(
        params,
        {},
        "resume",
        ["prepare-exp"],
        completed_chunks=completed,
    )

    assert completed == frozenset({0})
    assert [job["name"] for job in jobs] == [
        "resume-job-0-collect-rollouts-chunk-1-attempt-0",
        "resume-job-0-collect-rollouts-chunk-1-attempt-1",
        "resume-job-1-analyze-rollouts",
    ]


def test_resume_checks_source_identity_before_skipping_installation() -> None:
    config = _config()
    config["installation_command"] = "echo INSTALL_GYM"
    params = parse_rollout_config(config)
    script = native_rollout._collection_script(
        params,
        0,
        params.policy.base_url,
        None,
        port_preamble="",
    )

    completion_guard = (
        "if [ -f /outputs/collect/finance/jobs/job-0-collect-rollouts/"
        "chunks/chunk-0/completion.json"
    )
    assert script.index("native_sharding prepare") < script.index(completion_guard)
    assert script.index(completion_guard) < script.index("INSTALL_GYM")
    assert (
        "--shard-manifest /outputs/collect/finance/jobs/"
        "job-0-collect-rollouts/chunks/chunk-0/manifest.json"
    ) in script
    assert "chunks/chunk-0/outputs/manifest.json" not in script


def test_empty_chunk_guard_completes_before_installing_or_running_gym() -> None:
    config = _config(num_chunks=4)
    config["installation_command"] = "echo INSTALL_GYM"
    params = parse_rollout_config(config)
    script = native_rollout._collection_script(
        params,
        2,
        params.policy.base_url,
        None,
        port_preamble="",
    )

    assert "selected_row_count" in script
    assert script.index("selected_row_count") < script.index("INSTALL_GYM")
    assert script.index("selected_row_count") < script.index("gym eval run")
    assert "--shard-manifest" in script


def test_allowed_empty_input_publishes_empty_analysis_instead_of_running_gym() -> None:
    allowed = parse_rollout_config(_config(num_repeats=1, allow_empty_input=True))
    default = parse_rollout_config(_config(num_repeats=1))
    jobs = build_rollout_jobs(allowed, {}, "native")
    collect, analyze = ("\n".join(_rendered_commands(job)) for job in (jobs[0], jobs[-1]))

    assert "--allow-empty" in collect and "--allow-empty" in analyze
    assert (
        analyze.index("gym eval health-check")
        < analyze.index("No input rows")
        < (
            analyze.index(
                "native_sharding complete --materialized-inputs /outputs/collect/finance/jobs/job-1"
            )
        )
    )
    default_analyze = "\n".join(_rendered_commands(build_rollout_jobs(default, {}, "native")[-1]))
    assert "--allow-empty" not in default_analyze and "No input rows" not in default_analyze
    explicit_off = parse_rollout_config(_config(num_repeats=1, allow_empty_input=False))
    sha = native_rollout._rollout_config_sha256
    assert sha(default) == sha(explicit_off) != sha(allowed)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"rollout": {"allow_empty_input": "yes"}}, "true or false"),
        ({"blade": {"enabled": True, "command": "blade-tool analyze"}}, "cannot be combined"),
    ],
)
def test_allow_empty_input_fails_closed(extra: dict[str, Any], message: str) -> None:
    config = _config(allow_empty_input=True)
    config["rollout"].update(extra.pop("rollout", {}))
    config.update(extra)

    with pytest.raises(NativeRolloutConfigError, match=message):
        parse_rollout_config(config)


def test_single_repeat_skips_unsupported_profile_command() -> None:
    params = parse_rollout_config(_config(num_repeats=1))
    postprocess = "\n".join(_rendered_commands(build_rollout_jobs(params, {}, "native")[-1]))

    assert "gym eval profile" not in postprocess
    assert "gym eval health-check" in postprocess


def test_single_chunk_aggregates_before_gym_shuts_down() -> None:
    params = parse_rollout_config(_config(num_chunks=1))
    jobs = build_rollout_jobs(params, {}, "single")
    attempt = "\n".join(_rendered_commands(jobs[0]))
    postprocess = "\n".join(_rendered_commands(jobs[-1]))

    assert "--disable-aggregation" not in attempt
    assert "rollouts_aggregate_metrics.json" in postprocess
    assert "if [ -s " in postprocess
    assert "gym env start" in postprocess  # fallback for older/incomplete shards only


def test_dataset_selection_requires_multiple_repeats() -> None:
    config = _config(num_repeats=1)
    config["dataset_selection"] = {
        "mode": "scalar_variance",
        "unprofiled_samples": "exclude",
    }

    with pytest.raises(NativeRolloutConfigError, match="num_repeats.*>= 2"):
        parse_rollout_config(config)


def test_optional_blade_runs_inside_analyze_before_dataset_publication() -> None:
    config = _config()
    config["blade"] = {"enabled": True, "command": "blade-tool analyze"}
    config["dataset_selection"] = {
        "mode": "scalar_variance",
        "unprofiled_samples": "exclude",
    }
    params = parse_rollout_config(config)

    jobs = build_rollout_jobs(params, {}, "native")

    assert [job["name"] for job in jobs[-2:]] == [
        "native-job-1-analyze-rollouts",
        "native-job-2-build-training-dataset",
    ]
    assert jobs[-1]["dependencies"] == [jobs[-2]]
    analyze = "\n".join(_rendered_commands(jobs[-2]))
    assert "NVFLOW_REWARD_PROFILE=" in analyze
    assert "NVFLOW_BLADE_OUTPUT_DIR=" in analyze
    assert analyze.index("gym eval profile") < analyze.index("blade-tool analyze")
    assert analyze.index("blade-tool analyze") < analyze.index("native_sharding complete")
    assert _groups(jobs[-2])[0].log_dir.endswith("/jobs/job-1-analyze-rollouts/logs")
    selection_command = "\n".join(_rendered_commands(jobs[-1]))
    assert "nvflow.integrations.nemo_gym.workers.build_training_dataset" in selection_command
    assert _groups(jobs[-1])[0].log_dir.endswith("/jobs/job-2-build-training-dataset/logs")


def test_shared_dataset_selection_uses_native_profile_without_difficulty_file() -> None:
    config = _config()
    config["dataset_selection"] = {
        "mode": "scalar_variance",
        "min_std": 1e-6,
        "unprofiled_samples": "exclude",
        "success_threshold": 1.0,
    }
    params = parse_rollout_config(config)

    jobs = build_rollout_jobs(params, {}, "native")
    command = "\n".join(_rendered_commands(jobs[-1]))

    assert jobs[-1]["name"] == "native-job-2-build-training-dataset"
    assert "workers.build_training_dataset" in command
    assert "--materialized-inputs" in command
    assert "--responses-create-params-overrides" in command
    assert '"temperature":0.7' in command
    assert params.input_data in command
    assert params.output_dir in command
    assert "difficulty.jsonl" not in command


def test_disabling_dataset_selection_removes_stale_selected_data() -> None:
    params = parse_rollout_config(_config())
    postprocess = "\n".join(_rendered_commands(build_rollout_jobs(params, {}, "native")[-1]))

    assert "rm -f" in postprocess
    assert f"{params.output_dir}/profiled.jsonl" in postprocess
    assert f"{params.output_dir}/train.jsonl" in postprocess
    assert "job-2-build-training-dataset/completion.json" in postprocess
    assert "selection_report.json" in postprocess


@pytest.mark.parametrize(
    ("model", "expected_gpus", "expected_fragment"),
    [
        (
            {
                "deployment": "gym",
                "model": "/models/native",
                "num_gpus": 4,
                "gym_model": {"vllm_serve_kwargs": {"tensor_parallel_size": 4}},
            },
            4,
            "local_vllm_model.vllm_serve_kwargs.tensor_parallel_size=4",
        ),
        (
            {
                "deployment": "external",
                "model_type": "openai_model",
                "model": "gpt-5",
                "base_url": "https://api.example/v1",
                "api_key_env": "HOSTED_API_KEY",
            },
            0,
            "${HOSTED_API_KEY}",
        ),
    ],
)
def test_gym_and_external_policy_deployments_route_resources_and_endpoints(
    model: dict[str, Any],
    expected_gpus: int,
    expected_fragment: str,
) -> None:
    params = parse_rollout_config(_config(models={"policy": model}))
    job = build_rollout_jobs(params, {}, "native")[0]

    assert sum(group.hardware.num_gpus or 0 for group in _groups(job)) == expected_gpus
    assert expected_fragment in "\n".join(_rendered_commands(job))


def test_gym_deployed_endpoint_gpu_topology_is_explicit_and_consistent() -> None:
    defaulted = parse_rollout_config(
        _config(
            models={
                "policy": {
                    "deployment": "gym",
                    "model": "/models/policy",
                    "num_gpus": 4,
                    "gym_model": {"vllm_serve_kwargs": {"tensor_parallel_size": 4}},
                },
            }
        )
    )
    assert defaulted.policy.gym_model["vllm_serve_kwargs"]["tensor_parallel_size"] == 4

    with pytest.raises(NativeRolloutConfigError, match="topology reserves 2 GPUs"):
        parse_rollout_config(
            _config(
                models={
                    "policy": {
                        "deployment": "gym",
                        "model": "/models/policy",
                        "num_gpus": 4,
                        "gym_model": {"vllm_serve_kwargs": {"tensor_parallel_size": 2}},
                    },
                }
            )
        )

    with pytest.raises(NativeRolloutConfigError, match="uses_reasoning_parser=true"):
        parse_rollout_config(
            _config(
                models={
                    "policy": {
                        "deployment": "gym",
                        "model": "/models/policy",
                        "num_gpus": 2,
                        "gym_model": {
                            "vllm_serve_kwargs": {
                                "tensor_parallel_size": 2,
                                "reasoning_parser": "qwen3",
                            }
                        },
                    },
                }
            )
        )


def test_gym_deployed_policy_and_judge_share_one_four_gpu_allocation() -> None:
    params = parse_rollout_config(
        _config(
            models={
                "policy": {
                    "deployment": "gym",
                    "model": "/models/qwen3-4b",
                    "num_gpus": 2,
                    "gym_model": {"vllm_serve_kwargs": {"tensor_parallel_size": 2}},
                },
                "judge": {
                    "deployment": "gym",
                    "model": "/models/gpt-oss-120b",
                    "num_gpus": 2,
                    "gym_model": {
                        "return_token_id_information": False,
                        "uses_reasoning_parser": False,
                        "vllm_serve_env_vars": {"VLLM_RAY_DP_PACK_STRATEGY": "strict"},
                        "vllm_serve_kwargs": {
                            "data_parallel_size": 1,
                            "tensor_parallel_size": 2,
                            "pipeline_parallel_size": 1,
                        },
                    },
                },
            }
        )
    )
    first = build_rollout_jobs(params, {}, "quickstart")[0]
    command = "\n".join(_rendered_commands(first))

    assert [group.hardware.num_gpus for group in _groups(first)] == [4]
    assert "policy_model.responses_api_models.local_vllm_model" in command
    assert "judge_model.responses_api_models.local_vllm_model" in command
    assert (
        "judge_model.responses_api_models.local_vllm_model.return_token_id_information=false"
        in command
    )
    assert (
        "judge_model.responses_api_models.local_vllm_model."
        'vllm_serve_env_vars.VLLM_RAY_DP_PACK_STRATEGY="strict"' in command
    )
    assert (
        "judge_model.responses_api_models.local_vllm_model."
        "vllm_serve_kwargs.pipeline_parallel_size=1" in command
    )
    assert "nvflow.integrations.nemo_gym.runtime_binding" in command
    assert "finance_resources.resources_servers.finance.judge_model_server.name" not in command


def test_gym_deployed_judge_requires_ray_pack_strategy() -> None:
    judge = {
        "deployment": "gym",
        "model": "/models/judge",
        "num_gpus": 2,
        "gym_model": {
            "return_token_id_information": False,
            "uses_reasoning_parser": False,
            "vllm_serve_env_vars": {},
            "vllm_serve_kwargs": {
                "data_parallel_size": 1,
                "tensor_parallel_size": 2,
                "pipeline_parallel_size": 1,
            },
        },
    }

    with pytest.raises(NativeRolloutConfigError, match="VLLM_RAY_DP_PACK_STRATEGY"):
        parse_rollout_config(
            _config(
                models={
                    "policy": _POLICY_DEPLOYMENTS[0],
                    "judge": judge,
                }
            )
        )


def test_policy_as_judge_uses_gym_resource_server_reference() -> None:
    config = _config()
    config["environments"]["finance"]["gym"] = {
        "finance_resources": {
            "resources_servers": {"finance": {"judge_model_server": {"name": "policy_model"}}}
        }
    }
    params = parse_rollout_config(config)
    command = "\n".join(_rendered_commands(build_rollout_jobs(params, {}, "reuse-policy")[0]))

    assert params.judge is None
    assert (
        'finance_resources.resources_servers.finance.judge_model_server.name="policy_model"'
        in command
    )
    assert "judge_model.responses_api_models" not in command


def test_analyze_job_aggregates_with_cpu_only_gym_environment() -> None:
    params = parse_rollout_config(
        _config(
            models={
                "policy": {
                    "deployment": "gym",
                    "model": "/models/policy",
                    "num_gpus": 2,
                    "gym_model": {"vllm_serve_kwargs": {"tensor_parallel_size": 2}},
                },
                "judge": {
                    "deployment": "external",
                    "model": "judge-v2",
                    "model_type": "openai_model",
                    "base_url": "https://judge.example/v1",
                },
            }
        )
    )
    jobs = build_rollout_jobs(params, {}, "aggregate")
    analyze = next(job for job in jobs if job["name"] == "aggregate-job-1-analyze-rollouts")
    command = "\n".join(_rendered_commands(analyze))

    assert "gym env start" in command
    assert "gym eval aggregate" in command
    assert "responses_api_models/vllm_model/configs/vllm_model.yaml" in command
    assert "responses_api_models/local_vllm_model/configs/local_vllm_model.yaml" not in command
    assert "+model_endpoint_readiness_timeout_seconds=0" in command
    assert '+head_server.host="127.0.0.1"' in command
    assert "+head_server.port=${NVFLOW_AGGREGATE_HEAD_PORT}" in command
    assert 'judge_model.responses_api_models.vllm_model.model="judge-v2"' in command
    assert command.index("gym env start") < command.index("gym eval aggregate")


def test_gym_policy_and_external_judge_use_one_role_contract() -> None:
    params = parse_rollout_config(
        _config(
            models={
                "policy": {
                    "deployment": "gym",
                    "model": "/models/policy",
                    "num_gpus": 4,
                    "gym_model": {"vllm_serve_kwargs": {"tensor_parallel_size": 4}},
                },
                "judge": {
                    "deployment": "external",
                    "model_type": "openai_model",
                    "model": "judge-v2",
                    "base_url": "https://judge.example/v1",
                    "api_key_env": "JUDGE_KEY",
                },
            }
        )
    )
    command = "\n".join(_rendered_commands(build_rollout_jobs(params, {}, "native")[0]))

    assert 'judge_model.responses_api_models.openai_model.openai_model="judge-v2"' in command
    assert "nvflow.integrations.nemo_gym.runtime_binding" in command
    assert "finance_resources.resources_servers.finance.judge_model_server.name" not in command
    assert "${JUDGE_KEY}" in command


def test_gym_model_parameters_are_identical_across_deployments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gym_model = {
        "uses_reasoning_parser": True,
        "chat_template_kwargs": {"enable_thinking": True},
        "vllm_serve_kwargs": {
            "tensor_parallel_size": 2,
            "max_model_len": 32768,
            "reasoning_parser": "qwen3",
            "tool_call_parser": "hermes",
        },
    }
    gym_deployed = parse_rollout_config(
        _config(
            models={
                "policy": {
                    "deployment": "gym",
                    "model": "/models/policy",
                    "num_gpus": 2,
                    "gym_model": deepcopy(gym_model),
                },
            }
        )
    )
    nvflow_deployed = parse_rollout_config(
        _config(
            models={
                "policy": {
                    "deployment": "nvflow",
                    "model": "/models/policy",
                    "num_gpus": 2,
                    "gym_model": deepcopy(gym_model),
                },
            }
        )
    )
    captured: list[dict[str, Any]] = []

    def _capture_server(config, cluster, *, role, **kwargs):
        captured.append(config)
        return _FakeServer(role)

    monkeypatch.setattr(native_rollout, "make_server_script", _capture_server)
    gym_command = "\n".join(_rendered_commands(build_rollout_jobs(gym_deployed, {}, "gym")[0]))
    build_rollout_jobs(nvflow_deployed, {}, "nvflow")

    assert gym_deployed.policy.gym_model == nvflow_deployed.policy.gym_model == gym_model
    assert captured[0]["max_model_len"] == 32768
    assert captured[0]["reasoning_parser"] == "qwen3"
    assert "vllm_serve_kwargs.max_model_len=32768" in gym_command
    assert "uses_reasoning_parser=true" in gym_command


def test_standalone_vllm_serializes_gym_native_mapping_values() -> None:
    arguments = build_vllm_server_args({"override_generation_config": {"temperature": 0.0}})

    assert arguments == "--override-generation-config '{\"temperature\":0.0}'"


def test_standalone_vllm_shell_quotes_scalar_values() -> None:
    arguments = build_vllm_server_args({"served_model_name": "model name; $(touch /tmp/not-run)"})

    assert shlex.split(arguments) == [
        "--served-model-name",
        "model name; $(touch /tmp/not-run)",
    ]


def test_hf_config_overrides_reach_vllm_without_writing_beside_the_model() -> None:
    from nvflow.integrations.nemo_gym.job_scripts import make_server_script

    script = make_server_script(
        {
            "model_path": "/hf_models/Qwen/Qwen3-4B",
            "num_gpus": 1,
            "max_model_len": 131072,
            "hf_config_overrides": {"rope_scaling": {"rope_type": "yarn", "factor": 3.2}},
        },
        {"executor": "none", "containers": {"vllm": "vllm-image"}},
    )

    arguments = shlex.split(str(script.inline).split("model_servers.serve", 1)[1])
    assert arguments[arguments.index("--model") + 1] == "/hf_models/Qwen/Qwen3-4B"
    overrides = arguments[arguments.index("--hf-overrides") + 1]
    assert json.loads(overrides) == {"rope_scaling": {"rope_type": "yarn", "factor": 3.2}}
    assert "overlay" not in str(script.inline)


def test_multi_node_vllm_on_mp_runs_the_launcher_on_every_node_without_ray() -> None:
    from nvflow.integrations.nemo_gym.job_scripts import make_server_script

    config = {"model_path": "/hf_models/large", "num_gpus": 4, "server_nodes": 2}
    cluster = {"executor": "none", "containers": {"vllm": "vllm-image"}}
    with_ray = make_server_script(config, cluster)
    without_ray = make_server_script({**config, "distributed_executor_backend": "mp"}, cluster)

    assert "ray start" in str(with_ray.inline)
    inline = str(without_ray.inline)
    assert "ray start" not in inline
    arguments = shlex.split(inline.split("model_servers.serve", 1)[1])
    assert arguments[:2] == ["--server-type", "vllm"]
    assert arguments[arguments.index("--num_nodes") + 1] == "2"
    assert arguments[arguments.index("--distributed-executor-backend") + 1] == "mp"
    assert arguments[arguments.index("--node_rank") + 1] == "${SLURM_PROCID:-0}"
    assert arguments[arguments.index("--dist_init_addr") + 1] == "${SLURM_MASTER_NODE:-localhost}"


def test_an_nvflow_deployed_server_exports_its_environment() -> None:
    from nvflow.integrations.nemo_gym.job_scripts import make_server_script

    config = {"model_path": "/hf_models/large", "num_gpus": 4, "server_nodes": 2}
    cluster = {"executor": "none", "containers": {"vllm": "vllm-image"}}
    environment = {"NCCL_MNNVL_ENABLE": 1, "HF_MODULES_CACHE": "/tmp/hf modules"}

    inline = str(make_server_script(config, cluster, environment=environment).inline)

    assert inline.startswith(
        "export HF_MODULES_CACHE='/tmp/hf modules' && export NCCL_MNNVL_ENABLE=1 && "
        "export NVFLOW_SERVER_ENV='HF_MODULES_CACHE NCCL_MNNVL_ENABLE' && "
    )
    with pytest.raises(ValueError, match="invalid variable names: BAD-NAME"):
        make_server_script(config, cluster, environment={"BAD-NAME": "1"})


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"BAD-NAME": "1"}, "invalid variable names: BAD-NAME"),
        ({"NCCL_MNNVL_ENABLE": True}, "NCCL_MNNVL_ENABLE' must be a string or number, not bool"),
        ({"HF_HOME": None}, "HF_HOME' must be a string or number, not NoneType"),
        ({"NVFLOW_SERVER_ENV": "HF_HOME"}, "can't set NVFLOW_SERVER_ENV; NVFlow sets it"),
    ],
)
def test_a_servers_environment_is_checked_when_the_config_is_parsed(
    environment: dict[str, Any], message: str
) -> None:
    policy = {
        "deployment": "nvflow",
        "model": "/models/policy",
        "num_gpus": 4,
        "gym_model": {"vllm_serve_env_vars": environment},
    }

    with pytest.raises(ValueError, match=message):
        parse_rollout_config(_config(models={"policy": policy}))


def test_rollouts_export_the_policy_servers_vllm_serve_env_vars() -> None:
    from nvflow.integrations.nemo_skills import portable_cluster_config

    policy = {
        "deployment": "nvflow",
        "model": "/models/policy",
        "num_gpus": 4,
        "gym_model": {
            "vllm_serve_kwargs": {"tensor_parallel_size": 4},
            "vllm_serve_env_vars": {"NCCL_MNNVL_ENABLE": "1"},
        },
    }
    params = parse_rollout_config(_config(models={"policy": policy}))

    first = build_rollout_jobs(params, portable_cluster_config(), "native")[0]

    assert "export NCCL_MNNVL_ENABLE=1 && " in "\n".join(_rendered_commands(first))


def test_hf_config_overrides_are_refused_for_sglang_servers() -> None:
    from nvflow.integrations.nemo_gym.job_scripts import make_server_script

    with pytest.raises(ValueError, match="only for vLLM"):
        make_server_script(
            {
                "model_path": "/hf_models/Qwen/Qwen3-4B",
                "num_gpus": 1,
                "hf_config_overrides": {"rope_scaling": {"rope_type": "yarn", "factor": 3.2}},
            },
            {"executor": "none", "containers": {"sglang": "sglang-image"}},
            server_type="sglang",
        )


@pytest.mark.parametrize(
    ("policy_deployment", "judge_deployment", "url_variable"),
    [
        ("gym", "nvflow", "NVFLOW_JUDGE_URL"),
        ("nvflow", "gym", "NVFLOW_POLICY_URL"),
    ],
)
def test_mixed_gym_and_nvflow_deployments_use_heterogeneous_groups(
    monkeypatch: pytest.MonkeyPatch,
    policy_deployment: str,
    judge_deployment: str,
    url_variable: str,
) -> None:
    monkeypatch.setattr(
        native_rollout,
        "make_server_script",
        lambda config, cluster, *, role, **kwargs: _FakeServer(role),
    )
    policy = {
        "deployment": policy_deployment,
        "model": "/models/policy",
        "num_gpus": 2 if policy_deployment == "gym" else 4,
        "gym_model": {
            "vllm_serve_kwargs": {"tensor_parallel_size": 2 if policy_deployment == "gym" else 4}
        },
    }
    judge = {
        "deployment": judge_deployment,
        "model": "/models/judge",
        "num_gpus": 2 if judge_deployment == "gym" else 4,
        "gym_model": {
            "return_token_id_information": False,
            "uses_reasoning_parser": False,
            "vllm_serve_env_vars": {"VLLM_RAY_DP_PACK_STRATEGY": "strict"},
            "vllm_serve_kwargs": {
                "data_parallel_size": 1,
                "tensor_parallel_size": 2 if judge_deployment == "gym" else 4,
                "pipeline_parallel_size": 1,
            },
        },
    }
    params = parse_rollout_config(_config(models={"policy": policy, "judge": judge}))

    first = build_rollout_jobs(params, {}, "mixed")[0]
    command = "\n".join(_rendered_commands(first))

    assert len(_groups(first)) == 2
    assert sorted(group.hardware.num_gpus for group in _groups(first)) == [2, 4]
    assert url_variable in command


def test_workflow_owned_gym_overrides_are_rendered_without_overlay_files() -> None:
    config = _config()
    config["environments"]["finance"]["gym"] = {
        "finance_resources": {
            "resources_servers": {
                "finance": {
                    "reward_mode": "scaled",
                    "optional_settings": {},
                    "retrieval_responses_create_params": {"temperature": 1.0},
                }
            }
        }
    }
    command = "\n".join(
        _rendered_commands(build_rollout_jobs(parse_rollout_config(config), {}, "native")[0])
    )

    assert '+finance_resources.resources_servers.finance.reward_mode="scaled"' in command
    assert (
        "+finance_resources.resources_servers.finance."
        "retrieval_responses_create_params.temperature=1.0"
    ) in command
    assert "+finance_resources.resources_servers.finance.optional_settings={}" in command


@pytest.mark.parametrize("field", ["agent_map", "fan_out", "num_repeats"])
def test_gym_overrides_cannot_change_native_rollout_cardinality(field: str) -> None:
    config = _config()
    config["environments"]["finance"]["gym"] = {field: {}}

    with pytest.raises(NativeRolloutConfigError, match="cardinality"):
        parse_rollout_config(config)


class _FakeServer:
    def __init__(self, role: str) -> None:
        self.role = role
        self.inline = f"serve-{role}"
        self.het_group_index = None
        self.span_group_nodes = True

    def set_inline(self, value: str) -> None:
        self.inline = value

    def hostname_ref(self) -> str:
        return f"{self.role}-host"


def test_nvflow_deployed_policy_and_judge_keep_separate_resource_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        native_rollout,
        "make_server_script",
        lambda config, cluster, *, role, **kwargs: _FakeServer(role),
    )
    params = parse_rollout_config(
        _config(
            models={
                "policy": {
                    "deployment": "nvflow",
                    "model": "/models/policy",
                    "num_gpus": 4,
                    "gym_model": {"vllm_serve_kwargs": {"tensor_parallel_size": 4}},
                },
                "judge": {
                    "deployment": "nvflow",
                    "model": "/models/judge",
                    "num_gpus": 2,
                    "gym_model": {"vllm_serve_kwargs": {"tensor_parallel_size": 2}},
                },
            }
        )
    )
    first = build_rollout_jobs(params, {}, "native")[0]

    assert [group.hardware.num_gpus for group in _groups(first)] == [4, 2]
    command = "\n".join(_rendered_commands(first))
    assert 'NVFLOW_POLICY_URL="http://$POLICY_HOST:$POLICY_PORT/v1"' in command
    assert 'NVFLOW_JUDGE_URL="http://$JUDGE_HOST:$JUDGE_PORT/v1"' in command


def test_service_port_files_are_scoped_per_engine() -> None:
    class Script:
        port = 8000
        inline = "python server.py --port 8000"

        def set_inline(self, command: str) -> None:
            self.inline = command

    script = Script()
    _wrap_server_with_dynamic_port(cast(Any, script), "policy", "/shared/logs", "job-0")

    assert "VLLM_PORT=$(find_free_port)" in script.inline
    assert 'if [ "${SLURM_NODEID:-0}" = "0" ]; then' in script.inline
    assert (
        'echo "$VLLM_PORT" > '
        '"/shared/logs/.vllm_port_policy_job-0_${SLURM_JOB_ID}_${SLURM_RESTART_COUNT:-0}.txt"'
    ) in script.inline
    assert 'echo "$VLLM_PORT" > "/shared/logs/.vllm_port_policy_job-0.txt"' in script.inline
    assert cast(Any, script).nvflow_readiness == (
        "files:/shared/logs/.vllm_host_policy_job-0.txt,/shared/logs/.vllm_port_policy_job-0.txt"
    )


def _served_port_script(tmp_path: Path, log_dir: Path) -> Any:
    class Script:
        port = 8000
        inline = f"echo --port 8000 > {tmp_path}/served.txt"

        def set_inline(self, command: str) -> None:
            self.inline = command

    script = Script()
    _wrap_server_with_dynamic_port(cast(Any, script), "policy", str(log_dir), "job-0")
    return script


def test_supplied_ray_service_publishes_readiness_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nvflow.integrations.ray_jobs.worker import run_payload

    log_dir = tmp_path / "attempts" / "attempt-0" / "logs"
    script = _served_port_script(tmp_path, log_dir)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_NODEID", raising=False)

    run_payload(
        {
            "task_id": "rollout.0",
            "log_path": str(tmp_path / "rollout.log"),
            "commands": [{"command": script.inline}],
        }
    )

    port = (log_dir / ".vllm_port_policy_job-0.txt").read_text(encoding="utf-8").strip()
    assert (tmp_path / "served.txt").read_text(encoding="utf-8").strip() == f"--port {port}"
    assert (log_dir / ".vllm_host_policy_job-0.txt").read_text(encoding="utf-8").strip()


def test_supplied_ray_service_fails_fast_when_readiness_cannot_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nvflow.integrations.ray_jobs.worker import run_payload

    blocked = tmp_path / "not-a-directory"
    blocked.write_text("", encoding="utf-8")
    script = _served_port_script(tmp_path, blocked / "logs")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="exit code 1"):
        run_payload(
            {
                "task_id": "rollout.0",
                "log_path": str(tmp_path / "rollout.log"),
                "commands": [{"command": script.inline}],
            }
        )
    assert not (tmp_path / "served.txt").exists()


def _direct_slurm_environment(**slurm: str) -> dict[str, str]:
    inherited = {
        key: value
        for key, value in os.environ.items()
        if key not in {"NVFLOW_ENGINE", "SLURM_RESTART_COUNT"}
    }
    return inherited | {"SLURM_JOB_ID": "4242", "SLURM_NODEID": "0", **slurm}


def test_direct_slurm_service_publishes_per_job_port(tmp_path: Path) -> None:
    script = _served_port_script(tmp_path, tmp_path)

    subprocess.run(["bash", "-c", script.inline], env=_direct_slurm_environment(), check=True)

    port = (tmp_path / ".vllm_port_policy_job-0_4242_0.txt").read_text(encoding="utf-8").strip()
    assert (tmp_path / "served.txt").read_text(encoding="utf-8").strip() == f"--port {port}"
    assert not (tmp_path / ".vllm_host_policy_job-0.txt").exists()


def test_requeued_slurm_client_waits_for_its_own_run_port(tmp_path: Path) -> None:
    # A requeued job keeps its ID, and its client can start before its server.
    (tmp_path / ".vllm_port_policy_job-0_4242_0.txt").write_text("1111\n", encoding="utf-8")
    preamble = _build_port_read_preamble(str(tmp_path), "job-0", policy_host="localhost")
    client_port = tmp_path / "client_port.txt"
    requeued = _direct_slurm_environment(SLURM_RESTART_COUNT="1")

    client = subprocess.Popen(
        ["bash", "-c", f'{preamble}echo "$POLICY_PORT" > {shlex.quote(str(client_port))}'],
        env=requeued,
    )
    subprocess.run(
        ["bash", "-c", _served_port_script(tmp_path, tmp_path).inline], env=requeued, check=True
    )
    assert client.wait(timeout=30) == 0

    port = (tmp_path / ".vllm_port_policy_job-0_4242_1.txt").read_text(encoding="utf-8").strip()
    assert client_port.read_text(encoding="utf-8").strip() == port != "1111"
    assert (tmp_path / "served.txt").read_text(encoding="utf-8").strip() == f"--port {port}"


def test_endpoint_preamble_resolves_hosts_per_engine() -> None:
    preamble = _build_port_read_preamble(
        "/shared/logs",
        "job-0",
        judge_host="${SLURM_MASTER_NODE_HET_GROUP_1:-localhost}",
    )
    supplied_ray, direct_slurm = preamble.split("\nelse\n")

    assert "POLICY_" not in preamble
    assert '"/shared/logs/.vllm_port_judge_job-0.txt"' in supplied_ray
    assert '"/shared/logs/.vllm_host_judge_job-0.txt"' in supplied_ray
    assert (
        '"/shared/logs/.vllm_port_judge_job-0_${SLURM_JOB_ID}_${SLURM_RESTART_COUNT:-0}.txt"'
    ) in direct_slurm
    assert 'JUDGE_HOST="${SLURM_MASTER_NODE_HET_GROUP_1:-localhost}"' in direct_slurm
    assert _build_port_read_preamble("/shared/logs", "job-0") == ""


def test_nvflow_deployment_selects_server_engine_and_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected: dict[str, str] = {}

    def _capture_server(config, cluster, *, role, server_type, **kwargs):
        selected.update(role=role, server_type=server_type)
        return _FakeServer(role)

    monkeypatch.setattr(native_rollout, "make_server_script", _capture_server)
    params = parse_rollout_config(
        _config(
            models={
                "policy": {
                    "deployment": "nvflow",
                    "server_type": "sglang",
                    "container": "sglang-custom",
                    "model": "/models/policy",
                    "num_gpus": 4,
                }
            }
        )
    )
    first = build_rollout_jobs(params, {}, "sglang")[0]

    assert selected == {"role": "policy", "server_type": "sglang"}
    assert _groups(first)[0].commands[0].container == "sglang-custom"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("num_random_seeds", 3, "legacy Gym rollout fields"),
        ("starting_seed", 0, "legacy Gym rollout fields"),
        ("rerun_done", False, "legacy Gym rollout fields"),
        ("policy_vllm", {}, "legacy Gym rollout fields"),
    ],
)
def test_removed_legacy_fields_fail_with_migration_error(
    field: str,
    value: object,
    message: str,
) -> None:
    config = _config()
    config["rollout"][field] = value
    with pytest.raises(NativeRolloutConfigError, match=message):
        parse_rollout_config(config)


def test_removed_postprocess_container_points_to_dataset_selection_container() -> None:
    config = _config()
    config["postprocess_container"] = "nemo-skills"

    with pytest.raises(NativeRolloutConfigError, match="dataset_selection_container"):
        parse_rollout_config(config)


@pytest.mark.parametrize(
    ("removed", "replacement"),
    [
        ("profile_selection", "dataset_selection"),
        ("profile_selection_container", "dataset_selection_container"),
    ],
)
def test_removed_profile_selection_contract_has_no_alias(
    removed: str,
    replacement: str,
) -> None:
    config = _config()
    config[removed] = {}

    with pytest.raises(NativeRolloutConfigError, match=replacement):
        parse_rollout_config(config)


def test_managed_vllm_container_must_come_from_workflow_config() -> None:
    config = _config(
        models={
            "policy": {
                "deployment": "nvflow",
                "model": "/models/policy",
                "num_gpus": 1,
            }
        }
    )
    config.pop("vllm_container")

    with pytest.raises(NativeRolloutConfigError, match="'vllm_container' is required"):
        parse_rollout_config(config)

    external = _config()
    external.pop("vllm_container")
    assert parse_rollout_config(external).vllm_container is None


def test_removed_prepare_data_dir_is_rejected() -> None:
    config = _config()
    config["prepare_data_dir"] = "/prepared"
    with pytest.raises(NativeRolloutConfigError, match="was removed"):
        parse_rollout_config(config)


def test_removed_model_mode_is_not_accepted_as_a_deployment_alias() -> None:
    config = _config(
        models={
            "policy": {
                "mode": "native",
                "model": "/models/policy",
                "num_gpus": 2,
            }
        }
    )

    with pytest.raises(NativeRolloutConfigError, match=r"unknown .* field.*mode"):
        parse_rollout_config(config)


def test_static_job_rendering_is_deterministic() -> None:
    config = _config()
    first = build_rollout_jobs(parse_rollout_config(config), {}, "native")
    second = build_rollout_jobs(parse_rollout_config(deepcopy(config)), {}, "native")

    assert [job["name"] for job in first] == [job["name"] for job in second]
    assert [_rendered_commands(job) for job in first] == [_rendered_commands(job) for job in second]


def test_rollout_fingerprint_is_deterministic_and_behavior_sensitive() -> None:
    original_config = _config()
    original = parse_rollout_config(original_config)
    equivalent = parse_rollout_config(deepcopy(original_config))
    changed_config = deepcopy(original_config)
    changed_config["environments"]["finance"]["gym"] = {
        "observability_enabled": True,
        "model_call_capture_dir": "/outputs/collect/finance/captures/model-calls",
    }
    changed = parse_rollout_config(changed_config)

    assert native_rollout._rollout_config_sha256(original) == (
        native_rollout._rollout_config_sha256(equivalent)
    )
    assert native_rollout._rollout_config_sha256(original) != (
        native_rollout._rollout_config_sha256(changed)
    )


def test_a_resumed_rollout_may_resize_or_reschedule_without_changing_its_fingerprint() -> None:
    def policy(**serve: object) -> dict[str, Any]:
        return {
            "deployment": "nvflow",
            "model": "/models/policy",
            "num_gpus": 4,
            "gym_model": {
                "vllm_serve_kwargs": {"tensor_parallel_size": 4, "max_model_len": 32768, **serve}
            },
        }

    def fingerprint(config: dict[str, Any]) -> str:
        return native_rollout._rollout_config_sha256(parse_rollout_config(config))

    base = fingerprint(_config(models={"policy": policy()}))
    resized: dict[str, Any] = {
        **policy(tensor_parallel_size=8, max_num_seqs=128),
        "server_nodes": 2,
    }
    resized["gym_model"]["vllm_serve_env_vars"] = {"NCCL_MNNVL_ENABLE": "1"}
    waiting = _config(models={"policy": policy()})
    waiting["environments"]["finance"]["gym"] = {"model_endpoint_readiness_timeout_seconds": 3600}

    rescheduled = _config(
        models={"policy": policy()}, num_samples_in_parallel=128, dependent_jobs=3
    )
    assert fingerprint(rescheduled) == base
    assert fingerprint(_config(models={"policy": resized})) == base
    assert fingerprint(waiting) == base
    assert fingerprint(_config(models={"policy": policy(max_model_len=65536)})) != base
    assert fingerprint(_config(models={"policy": policy(load_format="dummy")})) != base
    resampled = _config(models={"policy": policy()}, responses_create_params={"temperature": 1.0})
    assert fingerprint(resampled) != base


def test_runs_recorded_under_the_legacy_fingerprint_keep_resuming() -> None:
    params = parse_rollout_config(_config())
    legacy = native_rollout._legacy_rollout_config_sha256(params)
    jobs = build_rollout_jobs(params, {}, "native")
    collect = "\n".join(_rendered_commands(jobs[0]))
    analyze = _groups(jobs[-1])[0].commands[0].script.cmd
    probes: list[dict[str, Any]] = []

    class _RecordingFilesystem:
        def batch_completion_matches(self, batch: list[dict[str, Any]]) -> list[bool]:
            probes.extend(batch)
            return [False] * len(batch)

    native_rollout._completed_chunks(params, cast(Any, _RecordingFilesystem()))

    # What runs started before resizing on resume recorded for this configuration.
    assert legacy == "a4206f8ac7fd02ff451681713811efed0ba9ed218fe6c5bc897bda3749ef699d"
    assert legacy != native_rollout._rollout_config_sha256(params)
    assert f"--accept-rollout-config-sha256 {legacy}" in collect
    assert f"--accept-behavior-sha256 {legacy}" in collect
    assert f"--accept-manifest-behavior-sha256 {legacy}" in analyze
    analysis_legacy = native_rollout._analysis_config_sha256(params, legacy)
    assert f"--accept-behavior-sha256 {analysis_legacy}" in analyze
    assert [probe["accepted_behavior_sha256"] for probe in probes] == [[legacy], [legacy]]
    assert [probe["accepted_manifest_behavior_sha256"] for probe in probes] == [[legacy], [legacy]]


def test_report_options_reach_the_report_and_its_analysis_fingerprint() -> None:
    plain = parse_rollout_config(_config(num_repeats=1), report_module="reports.split")
    strict = parse_rollout_config(
        _config(num_repeats=1), report_module="reports.split", report_options={"pass_rate": 0.5}
    )
    analyze = "\n".join(_rendered_commands(build_rollout_jobs(strict, {}, "native")[-1]))

    assert """--options '{"pass_rate": 0.5}'""" in analyze
    assert "--options" not in "\n".join(
        _rendered_commands(build_rollout_jobs(plain, {}, "native")[-1])
    )
    sha = native_rollout._analysis_config_sha256
    assert sha(plain) == sha(
        parse_rollout_config(_config(num_repeats=1), report_module="reports.split")
    )
    assert sha(plain) != sha(strict)


def test_analysis_fingerprint_tracks_report_and_blade_configuration() -> None:
    original = parse_rollout_config(_config(), report_module="reports.v1")
    changed = _config()
    changed["blade"] = {"enabled": True, "command": "blade analyze"}
    updated = parse_rollout_config(changed, report_module="reports.v2")

    assert native_rollout._rollout_config_sha256(original) == (
        native_rollout._rollout_config_sha256(updated)
    )
    assert native_rollout._analysis_config_sha256(original) != (
        native_rollout._analysis_config_sha256(updated)
    )


@pytest.mark.parametrize(
    ("model", "container_affects_fingerprint"),
    [
        (
            {
                "deployment": "external",
                "model": "policy",
                "base_url": "https://policy.example/v1",
            },
            False,
        ),
        (
            {
                "deployment": "gym",
                "model": "/models/policy",
                "num_gpus": 1,
                "gym_model": {
                    "vllm_serve_kwargs": {
                        "tensor_parallel_size": 1,
                    }
                },
            },
            False,
        ),
        (
            {
                "deployment": "nvflow",
                "server_type": "sglang",
                "model": "/models/policy",
                "num_gpus": 1,
            },
            False,
        ),
        (
            {
                "deployment": "nvflow",
                "model": "/models/policy",
                "num_gpus": 1,
            },
            True,
        ),
        (
            {
                "deployment": "nvflow",
                "container": "policy-vllm",
                "model": "/models/policy",
                "num_gpus": 1,
            },
            False,
        ),
    ],
)
def test_rollout_fingerprint_hashes_only_effective_vllm_container(
    model: dict[str, Any],
    container_affects_fingerprint: bool,
) -> None:
    first_config = _config(models={"policy": model})
    first_config["vllm_container"] = "vllm-a"
    second_config = deepcopy(first_config)
    second_config["vllm_container"] = "vllm-b"

    first = native_rollout._rollout_config_sha256(parse_rollout_config(first_config))
    second = native_rollout._rollout_config_sha256(parse_rollout_config(second_config))

    assert (first != second) is container_affects_fingerprint


def test_generated_leaf_command_executes_and_resumes_with_fake_gym(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text('{"uuid":"a"}\n{"uuid":"b"}\n', encoding="utf-8")
    output_dir = tmp_path / "run"
    config = _config(
        input_data=str(source),
        num_repeats=2,
        num_chunks=1,
        dependent_jobs=1,
        max_num_samples=None,
    )
    config["output_dir"] = str(output_dir)
    params = parse_rollout_config(config)
    monkeypatch.setattr(native_rollout, "CONTAINER_CODE_DIR", str(Path.cwd()))

    def fake_runtime_binding(**kwargs: Any) -> str:
        output = Path(kwargs["output_config"])
        content = (
            "finance_agent:\n"
            "  responses_api_agents:\n"
            "    finance_agent:\n"
            "      datasets:\n"
            "        - name: nvflow\n"
            "          type: train\n"
            "          jsonl_fpath: generated\n"
        )
        return (
            f"mkdir -p {shlex.quote(str(output.parent))} && "
            f"printf %s {shlex.quote(content)} > {shlex.quote(str(output))}"
        )

    monkeypatch.setattr(
        native_rollout,
        "render_runtime_binding_command",
        fake_runtime_binding,
    )

    layout = NativeGymOutputLayout.collection(output_dir)
    partial = Path(layout.chunk_artifacts(0).rollouts)
    partial.parent.mkdir(parents=True)
    prepare_input_shard(
        source,
        partial.with_suffix("") / "preprocessed_datasets" / "train.jsonl",
        Path(layout.chunk_manifest(0)),
        chunk_index=0,
        num_chunks=1,
        num_repeats=2,
        rollout_config_sha256=native_rollout._rollout_config_sha256(params),
    )
    partial.write_text(
        '{"_ng_task_index":0,"_ng_rollout_index":0,"reward":0.25}\n',
        encoding="utf-8",
    )

    fake_gym = tmp_path / "fake_gym.py"
    fake_gym.write_text(
        """\
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
assert args[:2] == ["eval", "run"], args
output = Path(args[args.index("--output") + 1])
repeats = int(args[args.index("--num-repeats") + 1])
prepared = output.with_suffix("") / "preprocessed_datasets" / "train.jsonl"
materialized = output.with_stem(output.stem + "_materialized_inputs").with_suffix(".jsonl")
rows = [json.loads(line) for line in prepared.read_text().splitlines()]
expanded = []
for source_index, row in enumerate(rows):
    task_index = row.get("_ng_task_index", source_index)
    for rollout_index in range(repeats):
        expanded.append({**row, "_ng_task_index": task_index, "_ng_rollout_index": rollout_index})
materialized.parent.mkdir(parents=True, exist_ok=True)
materialized.write_text("".join(json.dumps(row) + "\\n" for row in expanded))
existing = {}
if output.exists():
    existing = {
        (row["_ng_task_index"], row["_ng_rollout_index"]): row
        for row in map(json.loads, output.read_text().splitlines())
    }
for row in expanded:
    existing.setdefault(
        (row["_ng_task_index"], row["_ng_rollout_index"]),
        {**row, "reward": 1.0},
    )
output.write_text("".join(json.dumps(row) + "\\n" for row in existing.values()))
with open(os.environ["FAKE_GYM_LOG"], "a") as log:
    log.write("run\\n")
""",
        encoding="utf-8",
    )
    script = native_rollout._collection_script(
        params,
        0,
        params.policy.base_url,
        None,
        port_preamble="",
    )
    log = tmp_path / "gym.log"
    shell = (
        f'gym() {{ python3 {fake_gym} "$@"; }}\n'
        'cd() { command cd "$NVFLOW_TEST_CWD"; }\n' + script
    )
    environment = {
        **os.environ,
        "FAKE_GYM_LOG": str(log),
        "NVFLOW_TEST_CWD": str(Path.cwd()),
    }

    subprocess.run(["bash", "-c", shell], check=True, env=environment)
    subprocess.run(["bash", "-c", shell], check=True, env=environment)

    rollouts = Path(layout.chunk_artifacts(0).rollouts)
    dataset_config = (Path(layout.chunk_outputs_dir(0)) / "dataset_config.yaml").read_text()
    assert "finance_agent:" in dataset_config
    assert "responses_api_agents:" in dataset_config
    assert "jsonl_fpath:" in dataset_config
    assert len(rollouts.read_text().splitlines()) == 4
    assert '"reward": 0.25' in rollouts.read_text()
    assert Path(layout.chunk_artifacts(0).completion).is_file()
    assert log.read_text().splitlines() == ["run"]


def _run_with_startup_retry(tmp_path: Path, then: str) -> tuple[int, int, str]:
    from pathlib import PurePosixPath

    rollouts = tmp_path / "rollouts.jsonl"
    attempts = tmp_path / "attempts"
    command = (
        f"( n=$(( $(cat {attempts} 2>/dev/null || echo 0) + 1 )); echo $n > {attempts}; "
        + then.format(rollouts=rollouts)
        + " )"
    )
    script = (
        "set -euo pipefail\nfind_free_port() { echo 12345; }\n"
        + native_rollout._with_startup_retry(command, PurePosixPath(rollouts.as_posix()))
    )
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "NVFLOW_GYM_RETRY_DELAY": "0"},
        capture_output=True,
        text=True,
    )
    return result.returncode, int(attempts.read_text()), result.stderr


def test_gym_that_exits_during_startup_is_started_again(tmp_path: Path) -> None:
    status, attempts, stderr = _run_with_startup_retry(tmp_path, '[ "$n" -ge 3 ]')

    assert (status, attempts) == (0, 3)
    assert "starting it again (attempt 3 of 3)" in stderr


def test_gym_that_fails_after_collecting_is_not_started_again(tmp_path: Path) -> None:
    status, attempts, _stderr = _run_with_startup_retry(
        tmp_path, "echo '{{}}' >> {rollouts}; exit 7"
    )

    assert (status, attempts) == (7, 1)


def test_gym_startup_gives_up_after_three_attempts(tmp_path: Path) -> None:
    status, attempts, _stderr = _run_with_startup_retry(tmp_path, "exit 5")

    assert (status, attempts) == (5, 3)


def _start_aggregate_environment(tmp_path: Path, start: str) -> tuple[int, int, str]:
    from pathlib import PurePosixPath

    from nvflow.integrations.nemo_gym.shell_snippets import SHELL_STOP_BACKGROUND_PROCESS

    attempts = tmp_path / "attempts"
    command = "bash -c " + shlex.quote(
        f"n=$(( $(cat {attempts} 2>/dev/null || echo 0) + 1 )); echo $n > {attempts}; {start}"
    )
    log = PurePosixPath((tmp_path / "environment.log").as_posix())
    script = (
        "set -euo pipefail\nfind_free_port() { echo 12345; }\n"
        + SHELL_STOP_BACKGROUND_PROCESS
        + native_rollout._start_aggregate_environment(command, log)
        + '\nkill -9 "$aggregate_env_pid"\ntrap - EXIT'
    )
    result = subprocess.run(
        ["bash", "-c", script],
        env={**os.environ, "NVFLOW_GYM_RETRY_DELAY": "0"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.returncode, int(attempts.read_text()), result.stderr


def test_an_aggregation_environment_that_exits_during_startup_is_started_again(
    tmp_path: Path,
) -> None:
    status, attempts, stderr = _start_aggregate_environment(
        tmp_path, "[ $n -ge 3 ] || exit 4; echo 'All 2 / 2 servers ready!'; exec sleep 30"
    )

    assert (status, attempts) == (0, 3)
    assert "starting it again (attempt 3 of 3)" in stderr


def test_aggregation_startup_gives_up_after_three_attempts(tmp_path: Path) -> None:
    status, attempts, stderr = _start_aggregate_environment(tmp_path, "exit 4")

    assert (status, attempts) == (1, 3)
    assert "exited during startup" in stderr
