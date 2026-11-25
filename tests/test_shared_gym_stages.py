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
"""Project-neutral contracts for the shared NeMo Gym stage suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from nvflow.cli.workflow_inspection import inspect_workflow_structure
from nvflow.core.workflow_definition import WorkflowDefinition
from nvflow.integrations.nemo_gym.contracts import (
    GymContractError,
    WorkerExtensionSpec,
)
from nvflow.integrations.nemo_gym.environments import resolve_environments, training_environments
from nvflow.integrations.nemo_gym.native_contract import NativeGymOutputLayout
from nvflow.integrations.nemo_gym.operations import (
    normalize_rollout_operation,
    parse_worker_extensions,
)
from nvflow.stages.catalog import get_stage_type, list_stage_types
from nvflow.stages.nemo_gym import (
    CollateDatasetStage,
    CollectRolloutsStage,
    RejudgeRolloutsStage,
)
from nvflow.stages.nemo_gym._common import dataset_selection_paths
from nvflow.stages.nemo_gym.build_training_dataset import (
    build_training_dataset_command,
)


def _config() -> dict[str, Any]:
    return {
        "contract_version": 1,
        "output_dir": "/runs",
        "container": "nemo-gym",
        "vllm_container": "vllm",
        "environments": {
            "alpha": {
                "enabled": True,
                "input_data": "/data/alpha.jsonl",
                "output_name": "alpha-output",
                "config_paths": ["resources_servers/alpha/config.yaml"],
                "agent_name": "alpha_agent",
                "models": {
                    "policy": {
                        "gym_model": {
                            "vllm_serve_kwargs": {"max_model_len": 8192},
                        }
                    }
                },
                "responses_create_params": {"temperature": 0.7},
            },
            "disabled": {
                "enabled": False,
                "input_data": "",
                "config_paths": ["resources_servers/disabled/config.yaml"],
                "agent_name": "disabled_agent",
            },
        },
        "rollout": {
            "num_repeats": 2,
            "num_chunks": 1,
            "dependent_jobs": 0,
            "responses_create_params": {"top_p": 0.9},
            "models": {
                "policy": {
                    "deployment": "nvflow",
                    "model": "policy",
                    "num_gpus": 4,
                    "gym_model": {
                        "vllm_serve_kwargs": {"dtype": "bfloat16"},
                    },
                }
            },
        },
    }


def test_collect_stage_resolves_explicit_environments_and_overrides() -> None:
    targets = CollectRolloutsStage.target_configs(_config())
    assert [name for name, _target in targets] == ["alpha-output"]
    target = targets[0][1]
    assert target["output_dir"] == "/runs/alpha-output"
    assert target["rollout"]["input_data"] == "/data/alpha.jsonl"
    assert target["rollout"]["responses_create_params"] == {
        "top_p": 0.9,
        "temperature": 0.7,
    }
    assert target["rollout"]["models"]["policy"]["gym_model"]["vllm_serve_kwargs"] == {
        "dtype": "bfloat16",
        "max_model_len": 8192,
    }
    operation = normalize_rollout_operation(target)
    assert operation.agent_name == "alpha_agent"
    assert operation.num_repeats == 2


@pytest.mark.parametrize(
    (
        "name",
        "agent_name",
        "benchmark_config",
        "gym_overrides",
    ),
    [
        (
            "swebench_verified",
            "swebench_verified_opencode_sandboxed_agent",
            "benchmarks/swebench/verified/opencode.yaml",
            {},
        ),
        (
            "swebench_pro",
            "swebench_pro_opencode_sandboxed_agent",
            "benchmarks/swebench/pro/opencode.yaml",
            {},
        ),
        (
            "deepswe",
            "deepswe_opencode_sandboxed_agent",
            "benchmarks/deepswe/opencode.yaml",
            {"use_absolute_ip": True},
        ),
    ],
)
def test_pinned_swe_environments_fit_the_shared_collect_contract(
    name: str,
    agent_name: str,
    benchmark_config: str,
    gym_overrides: dict[str, Any],
) -> None:
    """Compile representative Gym 41232d0 SWE topologies without executing them."""
    config = _config()
    config["environments"] = {
        name: {
            "enabled": True,
            "input_data": f"/data/{name}.jsonl",
            "output_name": name,
            "config_paths": [
                benchmark_config,
                "nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml",
            ],
            "agent_name": agent_name,
            "gym": gym_overrides,
        }
    }

    [(output_name, target)] = CollectRolloutsStage.target_configs(config)
    operation = normalize_rollout_operation(target)

    assert output_name == name
    assert operation.agent_name == agent_name
    assert operation.judge is None
    assert benchmark_config in operation.config_paths
    assert operation.gym_overrides == gym_overrides


def test_curated_stage_catalog_is_stable_and_lazy() -> None:
    names = [stage_type.name for stage_type in list_stage_types()]
    assert names == sorted(names)
    assert "nemo_gym.collect" in names
    assert get_stage_type("nemo_gym.collect").class_name == "CollectRolloutsStage"
    with pytest.raises(ValueError, match="unknown shared stage type"):
        get_stage_type("arbitrary.module")


def test_collect_stage_applies_environment_selection() -> None:
    config = _config()
    config["_environment"] = "disabled"
    assert CollectRolloutsStage.target_configs(config) == []


def test_rejudge_carries_effective_source_response_overrides() -> None:
    config = _config()
    config["rollouts_dir"] = "/runs/source"
    config["source_responses_create_params"] = {"top_p": 0.9}

    [(name, target)] = RejudgeRolloutsStage.target_configs(config)

    assert name == "alpha-output"
    assert target["source_responses_create_params"] == {
        "top_p": 0.9,
        "temperature": 0.7,
    }


def test_example_validation_collates_and_declares_example_dataset() -> None:
    config = _config()
    config.update(
        {
            "gym_path": "/opt/nemo-gym",
            "container": "nemo-gym",
            "mode": "example_validation",
        }
    )
    config["environments"]["alpha"]["source_data"] = "/data/example.jsonl"

    [operation] = CollateDatasetStage.operations(config)

    assert "+mode=example_validation" in operation["command"]
    assert [output.path for output in operation["outputs"]] == [
        "/runs/alpha-output/example.jsonl",
        "/runs/alpha-output/gym_binding.json",
    ]
    assert "nvflow.integrations.nemo_gym.runtime_binding" in operation["command"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("models", [], "models must be a mapping"),
        ("models", {"polciy": {}}, "unknown model role"),
        ("models", {"policy": []}, "models.policy must be a mapping"),
        ("output_name", 1, "output_name must be a safe"),
        ("config_paths", [1], "requires config_paths"),
    ],
)
def test_environment_contract_rejects_implicit_coercion(
    field: str,
    value: object,
    message: str,
) -> None:
    config = _config()
    config["environments"]["alpha"][field] = value

    with pytest.raises(GymContractError, match=message):
        CollectRolloutsStage.target_configs(config)


@pytest.mark.parametrize(
    "field",
    ["agent_type", "resources_server_name", "resources_server_inner_name"],
)
def test_environment_contract_rejects_provider_topology_fields(field: str) -> None:
    config = _config()
    config["environments"]["alpha"][field] = "legacy-value"

    with pytest.raises(GymContractError, match="must let Gym resolve topology"):
        CollectRolloutsStage.target_configs(config)


def test_environment_contract_rejects_the_removed_agent_config_overlay() -> None:
    config = _config()
    config["environments"]["alpha"]["agent_config_overlay"] = "/prepared/overlay.yaml"

    with pytest.raises(GymContractError, match="agent_config_overlay was removed"):
        CollectRolloutsStage.target_configs(config)


def test_disabled_environments_neither_prepare_data_nor_train() -> None:
    config = _config()
    config["environments"]["alpha"]["raw_train_data"] = "/data/alpha.jsonl"
    config["environments"]["disabled"]["raw_train_data"] = "/data/disabled.jsonl"

    assert list(resolve_environments(config)) == ["alpha"]
    assert list(resolve_environments({**config, "_environment": ["alpha", "disabled"]})) == [
        "alpha"
    ]
    assert list(training_environments(config)) == ["alpha"]


@pytest.mark.parametrize("enabled", [0, None, "", "no"])
def test_environment_enabled_must_be_a_boolean(enabled: object) -> None:
    config = _config()
    config["environments"]["disabled"]["enabled"] = enabled

    with pytest.raises(GymContractError, match="enabled must be a boolean"):
        resolve_environments(config)


def test_falsey_non_mapping_stage_sections_are_rejected() -> None:
    with pytest.raises(ValueError, match="dataset_selection.*mapping"):
        build_training_dataset_command({"dataset_selection": []})

    config = _config()
    config["rollouts_dir"] = "/runs/source"
    config["verification"] = []
    with pytest.raises(ValueError, match="verification must be a mapping"):
        RejudgeRolloutsStage.target_configs(config)


def test_standalone_dataset_selection_forwards_response_overrides() -> None:
    command = build_training_dataset_command(
        {
            "source": "/source.jsonl",
            "profile": "/profile.jsonl",
            "materialized_inputs": "/materialized.jsonl",
            "rollouts": "/rollouts.jsonl",
            "rollout_completion": "/completion.json",
            "output_dir": "/selected",
            "dataset_selection": {},
            "responses_create_params_overrides": {"max_output_tokens": 32768},
        }
    )

    assert "--responses-create-params-overrides" in command
    assert '"max_output_tokens":32768' in command


def test_dataset_selection_resume_guard_binds_requested_policy() -> None:
    base = {
        "source": "/source.jsonl",
        "profile": "/profile.jsonl",
        "materialized_inputs": "/materialized.jsonl",
        "rollouts": "/rollouts.jsonl",
        "rollout_completion": "/completion.json",
        "output_dir": "/selected",
        "responses_create_params_overrides": {},
    }
    all_rows = build_training_dataset_command({**base, "dataset_selection": {"mode": "all"}})
    variable_rows = build_training_dataset_command(
        {
            **base,
            "dataset_selection": {
                "mode": "scalar_variance",
                "min_std": 1e-6,
            },
        }
    )

    all_hash = all_rows.split("--selection-sha256 ", 1)[1].split(";", 1)[0]
    variable_hash = variable_rows.split("--selection-sha256 ", 1)[1].split(";", 1)[0]
    assert all_hash != variable_hash


def test_worker_extensions_are_typed_and_versioned() -> None:
    extensions = parse_worker_extensions(
        {
            "report": {
                "contract_version": 1,
                "module": "project.reporting",
            }
        }
    )
    assert extensions["report"] == WorkerExtensionSpec(
        kind="report",
        module="project.reporting",
    )
    with pytest.raises(GymContractError, match="contract_version"):
        parse_worker_extensions({"report": {"contract_version": 2, "module": "project.reporting"}})
    with pytest.raises(GymContractError, match="contract_version"):
        parse_worker_extensions(
            {"report": {"contract_version": True, "module": "project.reporting"}}
        )
    with_options = parse_worker_extensions(
        {"report": {"module": "project.reporting", "options": {"format": "json"}}}
    )
    assert with_options["report"].options == {"format": "json"}
    with pytest.raises(GymContractError, match="options must be a mapping"):
        parse_worker_extensions({"report": {"module": "project.reporting", "options": "json"}})
    with pytest.raises(GymContractError, match="unknown extensions.report field"):
        parse_worker_extensions(
            {
                "report": {
                    "module": "project.reporting",
                    "arguments": ["--format", "json"],
                }
            }
        )
    with pytest.raises(GymContractError, match="unknown Gym extension"):
        parse_worker_extensions({"enrich": {"module": "project.enrich"}})


def test_contract_version_and_duplicate_outputs_fail_early() -> None:
    config = _config()
    config["contract_version"] = 2
    with pytest.raises(GymContractError, match="contract_version"):
        CollectRolloutsStage.target_configs(config)
    config["contract_version"] = True
    with pytest.raises(GymContractError, match="contract_version"):
        CollectRolloutsStage.target_configs(config)

    config = _config()
    config["environments"]["beta"] = {
        **config["environments"]["alpha"],
        "output_name": "alpha-output",
        "agent_name": "beta_agent",
    }
    with pytest.raises(GymContractError, match="unique output_name"):
        CollectRolloutsStage.target_configs(config)


def test_dataset_selection_artifacts_have_canonical_names() -> None:
    layout = NativeGymOutputLayout.collection("/runs/alpha")
    assert dataset_selection_paths(layout, {}) == (
        "/runs/alpha/profiled.jsonl",
        "/runs/alpha/train.jsonl",
        "/runs/alpha/jobs/job-2-build-training-dataset/completion.json",
        "/runs/alpha/jobs/job-2-build-training-dataset/selection_report.json",
        "/runs/alpha/jobs/job-2-build-training-dataset/diagnostics/unprofiled_samples.jsonl",
    )


def test_external_style_project_resolves_shared_stages_without_wrappers() -> None:
    root = Path(__file__).parent / "fixtures" / "projects" / "shared_gym_project"
    definition = WorkflowDefinition(root / "workflows" / "profile.yaml")
    assert definition.resolve_stage_class("collect_rollouts") is CollectRolloutsStage
    implementation = definition.stage_implementation("collect_rollouts")
    assert implementation["kind"] == "shared"
    assert implementation["name"] == "nemo_gym.collect"
    assert implementation["contract_version"] == 1
    assert implementation["provider"].endswith(".CollectRolloutsStage")
    assert implementation["platform"]["distribution"] == "nvflow"
    runtime = definition.stage_runtime_config("collect_rollouts")
    assert "uses" not in runtime


def test_stage_without_uses_is_rejected() -> None:
    root = Path(__file__).parent / "fixtures" / "projects" / "shared_gym_project"
    definition = WorkflowDefinition(root / "workflows" / "profile.yaml")
    definition.config["stages"]["collect_rollouts"]["uses"] = None
    with pytest.raises(KeyError, match="must declare a literal 'uses'"):
        definition.resolve_stage_class("collect_rollouts")


def test_shared_and_project_stages_coexist_without_registration(tmp_path: Path) -> None:
    project = tmp_path / "mixed"
    workflows = project / "workflows"
    package = project / "mixed_project"
    workflows.mkdir(parents=True)
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "stage.py").write_text(
        "from nvflow.core import BaseStage\n"
        "from nvflow.execution import StageExecution\n"
        "class CustomStage(BaseStage):\n"
        "    def execution(self, context):\n"
        "        return StageExecution.noop(context.stage, 'fixture')\n",
        encoding="utf-8",
    )
    (project / "project.yaml").write_text(
        "manifest_version: 1\nproject: mixed\npython_source_root: .\n",
        encoding="utf-8",
    )
    common = (
        "workflow:\n"
        "  name: profile\n"
        "cluster: test\n"
        "pipeline_stages: [collect, custom]\n"
        "stages:\n"
        "  collect:\n"
        "    uses: nemo_gym.collect\n"
        "  custom:\n"
        "    uses: project:mixed_project.stage:CustomStage\n"
    )
    workflow = workflows / "mixed.yaml"
    workflow.write_text(common, encoding="utf-8")
    definition = WorkflowDefinition(workflow)

    assert definition.resolve_stage_class("collect") is CollectRolloutsStage
    assert definition.resolve_stage_class("custom").__name__ == "CustomStage"


def test_declarative_use_rejects_unknown_stage_type() -> None:
    root = Path(__file__).parent / "fixtures" / "projects" / "shared_gym_project"
    definition = WorkflowDefinition(root / "workflows" / "profile.yaml")
    definition.config["stages"]["collect_rollouts"]["uses"] = "arbitrary.module"
    with pytest.raises(ValueError, match="unknown shared stage type"):
        definition.resolve_stage_class("collect_rollouts")


def test_stage_mapping_interpolation_cannot_select_implementation(tmp_path: Path) -> None:
    project = tmp_path / "interpolated"
    workflows = project / "workflows"
    workflows.mkdir(parents=True)
    (project / "project.yaml").write_text(
        "manifest_version: 1\nproject: interpolated\n",
        encoding="utf-8",
    )
    workflow = workflows / "profile.yaml"
    workflow.write_text(
        "workflow:\n"
        "  name: profile\n"
        "cluster: test\n"
        "pipeline_stages: [collect]\n"
        "bindings:\n"
        "  collect:\n"
        "    uses: nemo_gym.collect\n"
        "stages:\n"
        "  collect: ${bindings.collect}\n",
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="must be an explicit mapping"):
        WorkflowDefinition(workflow)
    with pytest.raises(TypeError, match="must be an explicit mapping"):
        inspect_workflow_structure(workflow)
