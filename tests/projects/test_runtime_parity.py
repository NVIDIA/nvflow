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
"""Scheduler-free regression tests for finance project runtime parity."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from nvfinance.stages.download.download_sec_filings import (
    DownloadSecFilingsStage,
)
from nvfinance.stages.download.prefetch_cache import PrefetchCacheStage
from nvfinance.stages.rl.embedded_eval import _load_eval_campaign
from nvfinance.stages.rl.eval_model import build_gym_model_config

from nvflow import load_workflow
from nvflow.execution import PlanningContext, TargetBinding
from nvflow.integrations.nemo_gym.environments import (
    collect_target_config,
    selected_environments,
    training_environments,
)
from nvflow.integrations.nemo_gym.operations import normalize_rollout_operation
from nvflow.integrations.nemo_rl.checkpoint_converter import build_dcp_conversion_script
from nvflow.stages.nemo_gym import CollectRolloutsStage
from nvflow.stages.nemo_rl.grpo import GRPOStage

PROJECT_ROOT = Path(__file__).parents[2]
FINANCE_ROOT = PROJECT_ROOT / "projects" / "finance"
WORKER_ROOT = "{{NVFLOW_CODE_ROOT}}/projects/finance"


def _finance_definition(monkeypatch, relative_config: str):
    monkeypatch.setenv("SEC_IDENTITY_EMAIL", "nvflow-tests@example.com")
    monkeypatch.setenv("SEC_IDENTITY_COMPANY", "NVFlow Tests")
    return load_workflow(FINANCE_ROOT / relative_config)


def test_finance_resolves_shared_gym_stages_declaratively(monkeypatch) -> None:
    definition = _finance_definition(monkeypatch, "workflows/grpo/qwen3_4b.yaml")
    assert definition.resolve_stage_class("collect_rollouts") is CollectRolloutsStage
    assert definition.resolve_stage_class("prepare_data").__module__.startswith(
        "nvflow.stages.nemo_gym"
    )
    assert definition.resolve_stage_class("rejudge").__module__.startswith("nvflow.stages.nemo_gym")

    evaluation = _finance_definition(monkeypatch, "workflows/eval/smoke.yaml")
    for stage_name in ("eval_data", "rejudge", "compare"):
        assert evaluation.resolve_stage_class(stage_name).__module__.startswith(
            "nvflow.stages.nemo_gym"
        )


def test_maintained_finance_workflows_defer_gym_deployed_models() -> None:
    offenders: list[str] = []

    def visit(value: object, path: Path) -> None:
        if isinstance(value, dict):
            if value.get("deployment") == "gym":
                offenders.append(path.relative_to(FINANCE_ROOT).as_posix())
            for nested in value.values():
                visit(nested, path)
        elif isinstance(value, list):
            for nested in value:
                visit(nested, path)

    for workflow_path in (FINANCE_ROOT / "workflows").rglob("*.yaml"):
        visit(yaml.safe_load(workflow_path.read_text(encoding="utf-8")), workflow_path)

    assert offenders == []


def test_finance_workflows_do_not_mutate_provider_server_inventory() -> None:
    offenders: list[str] = []

    def visit(value: object, path: Path) -> None:
        if isinstance(value, dict):
            if "_delete_key" in value or "drop_servers" in value:
                offenders.append(path.relative_to(FINANCE_ROOT).as_posix())
            for nested in value.values():
                visit(nested, path)
        elif isinstance(value, list):
            for nested in value:
                visit(nested, path)

    for workflow_path in (FINANCE_ROOT / "workflows").rglob("*.yaml"):
        visit(yaml.safe_load(workflow_path.read_text(encoding="utf-8")), workflow_path)

    assert offenders == []


def test_quickstart_and_production_use_nvflow_managed_models(
    monkeypatch,
) -> None:
    demo = _finance_definition(monkeypatch, "workflows/grpo/qwen3_4b.yaml")
    production = _finance_definition(
        monkeypatch,
        "workflows/grpo/qwen3_4b_template_production.yaml",
    )

    demo_rollout = demo.config["stages"]["collect_rollouts"]["rollout"]
    assert demo_rollout["models"]["policy"]["deployment"] == "nvflow"
    assert isinstance(
        demo_rollout["models"]["policy"]["gym_model"]["vllm_serve_kwargs"]["compilation_config"],
        dict,
    )
    assert (
        demo.config["environments"]["equivalence_llm_judge"]["models"]["judge"]["deployment"]
        == "nvflow"
    )
    production_rollout = production.config["stages"]["collect_rollouts"]["rollout"]
    assert production_rollout["models"]["policy"]["deployment"] == "nvflow"
    assert (
        production.config["environments"]["equivalence_llm_judge"]["models"]["judge"]["deployment"]
        == "nvflow"
    )


def test_eval_reasoning_presets_declare_matching_gym_contract(monkeypatch) -> None:
    definition = _finance_definition(monkeypatch, "workflows/eval/smoke.yaml")

    for name, preset in definition.config["model_presets"].items():
        gym_model = preset.get("gym_model") or {}
        has_parser = bool((gym_model.get("vllm_serve_kwargs") or {}).get("reasoning_parser"))
        assert bool(gym_model.get("uses_reasoning_parser")) == has_parser, name


def test_vals_agentic_thinking_options_live_on_the_policy_model(monkeypatch) -> None:
    definition = _finance_definition(
        monkeypatch,
        "workflows/sdg/vals-agentic-trajectories.yaml",
    )
    rollout = definition.config["stages"]["collect_rollouts_deepseek"]["rollout"]

    assert "chat_template_kwargs" not in rollout["responses_create_params"]
    assert rollout["models"]["policy"]["gym_model"]["chat_template_kwargs"] == {
        "enable_thinking": True
    }


def test_every_finance_eval_environment_uses_the_shared_normalizer(monkeypatch) -> None:
    definition = _finance_definition(monkeypatch, "workflows/eval/demo.yaml")
    campaign = _load_eval_campaign(
        "/workspace/outputs/finance/conformance",
        "demo.yaml",
        eval_data_root="/workspace/outputs/finance/conformance/step-1-eval-data",
        project_context=definition.project_context,
    )
    base_config = build_gym_model_config(
        {
            **campaign,
            "_source_name": "qwen3-4b",
            "model": campaign["model_presets"]["qwen3_4b"],
        }
    )

    operations = {
        environment.name: normalize_rollout_operation(
            collect_target_config(base_config, environment)
        )
        for environment in selected_environments(base_config)
    }

    assert set(operations) == {"fabv1", "secque", "financebench"}
    assert all(operation.policy.nvflow_deployed for operation in operations.values())
    assert all(operation.agent_name for operation in operations.values())
    assert all(operation.config_paths for operation in operations.values())


def test_eval_smoke_enables_persisted_trajectory_capture(monkeypatch) -> None:
    definition = _finance_definition(monkeypatch, "workflows/eval/smoke.yaml")
    secque = definition.config["benchmarks"]["secque"]
    gym = secque["gym"]

    assert gym["observability_enabled"] is True
    assert gym["model_call_capture_dir"] == (
        "/workspace/outputs/finance/smoke/qwen3-4b/collect-rollouts/secque/"
        "jobs/job-0-collect-rollouts/outputs/captures/model-calls"
    )
    assert secque["blade"]["enabled"] is True
    assert "NVFLOW_BLADE_OUTPUT_DIR" in secque["blade"]["command"]
    assert definition.config["rollout"]["num_chunks"] == 2
    assert definition.config["rollout"]["models"]["policy"]["deployment"] == "nvflow"
    assert secque["models"]["judge"]["deployment"] == "nvflow"


def test_rollout_reads_each_prepared_environment_and_skips_placeholders(monkeypatch) -> None:
    definition = _finance_definition(monkeypatch, "workflows/grpo/qwen3_4b.yaml")
    configurations = dict(
        CollectRolloutsStage.target_configs(dict(definition.config["stages"]["collect_rollouts"]))
    )

    assert "mcqa" not in configurations
    assert {config["rollout"]["num_chunks"] for config in configurations.values()} == {2}
    for environment, config in configurations.items():
        expected = f"/step-4-prepare-data/{environment}"
        prepared_dir = config["rollout"]["input_data"].removesuffix("/train.jsonl")
        assert prepared_dir.endswith(expected)
        assert "prepare_data_dir" not in config

    [(_label, training, _expname)] = GRPOStage._resolve_run_configs(
        dict(definition.config["stages"]["training"]),
        "training",
    )
    assert set(training["environments"]) == set(configurations)


@pytest.mark.parametrize(
    "name",
    [
        "qwen3_4b.yaml",
        "qwen3_4b_template_production.yaml",
        "qwen3_30b_a3b.yaml",
        "nemotron_nano_30b_a3b.yaml",
    ],
)
def test_rollout_and_training_model_protocols_do_not_drift(
    monkeypatch,
    name: str,
) -> None:
    definition = _finance_definition(monkeypatch, f"workflows/grpo/{name}")
    rollout_stage = dict(definition.config["stages"]["collect_rollouts"])
    training_stage = definition.project_context.render_launcher_paths(
        dict(definition.config["stages"]["training"])
    )
    presets = GRPOStage._load_presets(training_stage)
    environments = training_environments(training_stage)
    assert environments
    for environment in environments:
        [(_, rollout)] = CollectRolloutsStage.target_configs(
            {**rollout_stage, "_environment": environment}
        )
        serve = rollout["rollout"]["models"]["policy"]["gym_model"]["vllm_serve_kwargs"]
        [(_label, run, _expname)] = GRPOStage._resolve_run_configs(
            {**training_stage, "_environment": environment}, "training"
        )
        policy = GRPOStage()._resolve_nemo_rl_config(run, presets)["policy"]
        chat = policy["generation"]["vllm_cfg"].get("http_server_serving_chat_kwargs") or {}

        # Nemotron uses NeMo-RL's model-specific nano_v3 plugin while Gym/vLLM
        # uses the compatible built-in deepseek_r1 protocol.
        if name == "nemotron_nano_30b_a3b.yaml":
            assert (serve.get("reasoning_parser"), chat.get("reasoning_parser")) == (
                "deepseek_r1",
                "nano_v3",
            )
        else:
            assert serve.get("reasoning_parser") == chat.get("reasoning_parser"), environment
        if chat.get("tool_parser") is not None:
            assert serve.get("tool_call_parser") == chat["tool_parser"], environment


@pytest.mark.requires_nemo
def test_download_reads_launcher_config_but_emits_worker_command(monkeypatch) -> None:
    definition = _finance_definition(monkeypatch, "workflows/download/demo.yaml")
    raw_config = dict(definition.config["stages"]["download"])
    launcher_config = definition.project_context.render_launcher_paths(raw_config)
    worker_config = definition.project_context.render_execution_paths(raw_config)
    stage = DownloadSecFilingsStage()
    plan = stage.plan(
        PlanningContext(
            stage="download",
            config=worker_config,
            launcher_config=launcher_config,
            target=TargetBinding.resolve("test-cluster"),
            expname="download-demo",
            run_after=(),
            project_context=definition.project_context,
        )
    )

    assert Path(launcher_config["config"]).is_file()
    assert worker_config["config"].startswith(WORKER_ROOT)
    assert len(plan.tasks) == 1
    command = plan.tasks[0].commands[0].command
    assert "--tickers" in command
    assert "NVDA" in command
    assert str(FINANCE_ROOT) not in command


def test_embedded_campaign_renders_dynamically_loaded_project_paths(monkeypatch) -> None:
    definition = _finance_definition(monkeypatch, "workflows/sft/qwen3_4b.yaml")

    campaign = _load_eval_campaign(
        "/workspace/outputs/finance/test/eval",
        eval_data_root="/workspace/outputs/finance/test/step-1-eval-data",
        project_context=definition.project_context,
    )

    assert "project://" not in repr(campaign)
    assert campaign["benchmarks"]["secque"]["config_paths"] == [
        "resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml"
    ]
    assert (
        campaign["benchmarks"]["secque"]["gym"]["equivalence_llm_judge"]["resources_servers"][
            "equivalence_llm_judge"
        ]["judge_prompt_template_fpath"]
        == f"{WORKER_ROOT}/workflows/eval/prompts/judge_user.txt"
    )
    assert campaign["stages"]["eval_data"]["benchmarks"][0]["prompt_config"] == (
        f"{WORKER_ROOT}/workflows/eval/prompts/secque.yaml"
    )


def _prefetch_command(monkeypatch, **overrides: Any) -> str:
    calls: list[dict[str, Any]] = []
    config: dict[str, Any] = {
        "gym_path": "/opt/nemo-gym",
        "container": "nemo-gym",
        "script": "resources_servers/finance_sec_search/scripts/prefetch_sec_metadata.py",
        "cache_dir": "/workspace/cache",
        "ticker_config": f"{WORKER_ROOT}/configs/sp500.yaml",
        **overrides,
    }

    monkeypatch.setattr(
        "nvflow.integrations.nemo_skills.commands.wrap_arguments", lambda command: command
    )
    monkeypatch.setattr(
        "nvflow.integrations.nemo_skills.commands.run_cmd",
        lambda **kwargs: calls.append(kwargs),
    )

    stage = PrefetchCacheStage()
    stage.build_tasks(config, "test-cluster", "prefetch")

    assert len(calls) == 1
    return calls[0]["ctx"]


@pytest.mark.requires_nemo
def test_prefetch_uses_packaged_ticker_config(monkeypatch) -> None:
    command = _prefetch_command(monkeypatch)

    assert f"{WORKER_ROOT}/configs/sp500.yaml" in command
    assert "python /workspace/recipes/" not in command


@pytest.mark.requires_nemo
def test_prefetch_passes_supplementary_tickers_only_when_configured(monkeypatch) -> None:
    overlay = "/opt/nemo-gym/benchmarks/finance_sec_search/data/supplementary_tickers.json"

    assert "--supplementary_tickers" not in _prefetch_command(monkeypatch)
    assert "--supplementary_tickers" not in _prefetch_command(
        monkeypatch, supplementary_tickers=None
    )

    with_overlay = _prefetch_command(monkeypatch, supplementary_tickers=overlay)
    assert f"--supplementary_tickers {overlay}" in with_overlay


def test_fsdp_conversion_runs_only_nemo_tools() -> None:
    script = build_dcp_conversion_script("/checkpoints", 20, "/hf/step_20")

    # NeMo-RL's DCP converter refuses an existing output directory.
    assert 'mkdir -p "$HF_OUTPUT"' not in script
    assert "nvflow" not in script
    assert '--cast-dtype "$PRECISION"' in script


@pytest.mark.requires_nemo
@pytest.mark.parametrize("checkpoint_format", ["megatron", "fsdp"])
def test_checkpoint_conversion_command_is_valid_bash(checkpoint_format: str) -> None:
    from nvflow.integrations.nemo_rl import checkpoints

    config: dict[str, Any] = {
        "checkpoint_path": "/workspace/training/model-run",
        "format": checkpoint_format,
        "base_model": "/hf_models/Qwen/Qwen3-4B",
        "conversion": {"num_gpus": 1},
    }
    _, planned = checkpoints.plan_checkpoint_conversion(
        config, 20, "test-cluster", "checkpoint-eval", None
    )

    # The converter script is nested inside bash -c '...', so a stray quote breaks it.
    parsed = subprocess.run(
        ["bash", "-n", "-c", planned.commands[0].command], capture_output=True, text=True
    )
    assert parsed.returncode == 0, parsed.stderr


@pytest.mark.requires_nemo
def test_checkpoint_plan_uses_resolved_megatron_source(monkeypatch) -> None:
    from nvflow.integrations.nemo_rl import checkpoints

    config: dict[str, Any] = {
        "checkpoint_path": "/workspace/training/model-run",
        "format": "megatron",
        "base_model": "/hf_models/Qwen/Qwen3-4B",
        "conversion": {"num_gpus": 1},
    }
    expected_source = Path(config["checkpoint_path"]) / "checkpoints" / "step_20"

    monkeypatch.setattr(
        "nvflow.integrations.nemo_rl.checkpoint_converter.build_conversion_script",
        lambda source, output, model: f"convert {source} {output} {model}",
    )
    _, planned = checkpoints.plan_checkpoint_conversion(
        config,
        20,
        "test-cluster",
        "checkpoint-eval",
        None,
    )
    assert str(expected_source) in planned.commands[0].command
    assert checkpoints.describe_checkpoint_source(config, 20).source_path == expected_source


@pytest.mark.requires_nemo
@pytest.mark.parametrize(("step", "checkpoint_format"), [("final", "megatron"), (20, "hf")])
def test_checkpoint_source_requires_a_step_number_and_training_format(
    step: Any, checkpoint_format: str
) -> None:
    from nvflow.integrations.nemo_rl import checkpoints

    config = {"checkpoint_path": "/workspace/training/model-run", "format": checkpoint_format}
    with pytest.raises(ValueError):
        checkpoints.describe_checkpoint_source(config, step)


def test_qwen3_14b_enables_sequence_packing_for_context_parallelism() -> None:
    config = yaml.safe_load((FINANCE_ROOT / "workflows/sft/qwen3_14b.yaml").read_text())
    policy = config["stages"]["training"]["overrides"]["policy"]

    assert policy["megatron_cfg"]["context_parallel_size"] == 8
    assert policy["sequence_packing"] == {
        "enabled": True,
        "train_mb_tokens": policy["max_total_sequence_length"],
    }


def test_sec_identity_is_only_ever_read_from_the_environment() -> None:
    """A hardcoded contact would reach sec.gov as a forged User-Agent."""
    for relative_path, stage_name, default in (
        ("workflows/download/base.yaml", "download", ""),
        # Defaulted to null so the other SDG stages load without an identity; the
        # stage rejects the null before it can reach EDGAR.
        ("workflows/sdg/template-based-sdg.yaml", "create_seed_data", ",null"),
    ):
        config = yaml.safe_load((FINANCE_ROOT / relative_path).read_text())
        stage = config["stages"][stage_name]
        for key, variable in (
            ("sec_identity_email", "SEC_IDENTITY_EMAIL"),
            ("sec_identity_company", "SEC_IDENTITY_COMPANY"),
        ):
            assert stage[key] == "${oc.env:" + variable + default + "}"
