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
"""Tests for operator-selected single-target binding."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from nvflow.cli import main as cli
from nvflow.execution import TargetBinding, WorkflowRunner
from nvflow.execution.receipts import SubmissionReceipt
from nvflow.execution.target import TargetResolutionError, resolve_target
from nvflow.integrations.nemo_skills.target import (
    bind_target_config,
    preflight_target_config,
)
from nvflow.project import ProjectResolver

PROJECT_ROOT = Path(__file__).parents[2]


def test_target_binding_prefers_explicit_override() -> None:
    default = TargetBinding.resolve("workflow-cluster")
    override = TargetBinding.resolve("workflow-cluster", "operator-cluster")

    assert default == TargetBinding(cluster="workflow-cluster", source="workflow")
    assert override == TargetBinding(cluster="operator-cluster", source="override")


@pytest.mark.parametrize("value", ["", "   "])
def test_target_binding_rejects_empty_cluster(value: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        TargetBinding.resolve("workflow-cluster", value)


def test_resolved_slurm_target_has_no_ray_state(tmp_path: Path) -> None:
    target = tmp_path / "slurm.yaml"
    target.write_text("executor: slurm\npartition: batch\n")

    resolved = resolve_target(TargetBinding.resolve("slurm"), target)

    assert resolved.engine == "slurm"
    assert resolved.ray is None
    assert resolved.fingerprint.startswith("sha256:")


def test_resolved_ray_target_routes_only_declared_roles(tmp_path: Path) -> None:
    target = tmp_path / "ray.yaml"
    target.write_text(
        """
executor: ray
ray:
  control_cluster: skills
  artifact_root: /shared/nvflow
  clusters:
    skills:
      jobs_url: http://127.0.0.1:8265
      expected_image: nemo-skills
      expected_ray: "2.58.0"
      resources:
        CPU: 32
    training:
      jobs_url: ${oc.env:RAY_TRAINING_URL}
      controller_jobs_url: ${oc.env:RAY_TRAINING_INTERNAL_URL}
      credential_env: RAY_TRAINING_TOKEN
      controller_credential_env: RAY_TRAINING_INTERNAL_TOKEN
      expected_image: nemo-rl-v0.7
      gpus_per_node: 4
      allow_insecure: true
  routes:
    nemo-skills: skills
    nemo-rl: training
"""
    )

    resolved = resolve_target(
        TargetBinding.resolve("ray"),
        target,
        environ={
            "RAY_TRAINING_URL": "http://127.0.0.1:18265",
            "RAY_TRAINING_INTERNAL_URL": "http://training.internal:8265",
        },
    )

    assert resolved.engine == "ray"
    assert resolved.ray is not None
    assert resolved.ray.controller.name == "skills"
    assert resolved.ray.cluster_for_role("nemo-rl").jobs_url == "http://127.0.0.1:18265"
    assert (
        resolved.ray.cluster_for_role("nemo-rl").controller_jobs_url
        == "http://training.internal:8265"
    )
    assert resolved.ray.cluster_for_role("nemo-rl").credential_env == "RAY_TRAINING_TOKEN"
    assert (
        resolved.ray.cluster_for_role("nemo-rl").controller_credential_env
        == "RAY_TRAINING_INTERNAL_TOKEN"
    )
    assert resolved.ray.cluster_for_role("nemo-rl").gpus_per_node == 4
    assert resolved.ray.cluster_for_role("nemo-skills").resources["CPU"] == 32
    with pytest.raises(TargetResolutionError, match="No supplied Ray cluster route"):
        resolved.ray.cluster_for_role("unknown")


def test_resolved_ray_target_requires_secure_remote_endpoint(tmp_path: Path) -> None:
    target = tmp_path / "ray.yaml"
    target.write_text(
        """
executor: ray
ray:
  control_cluster: skills
  artifact_root: /shared/nvflow
  clusters:
    skills:
      jobs_url: http://ray.internal:8265
      expected_image: nemo-skills
  routes:
    nemo-skills: skills
"""
    )

    with pytest.raises(TargetResolutionError, match="https or allow_insecure"):
        resolve_target(TargetBinding.resolve("ray"), target)


_RAY_TARGET = """
executor: ray
{ssh_tunnel}ray:
  control_cluster: skills
  artifact_root: /shared/nvflow
  clusters:
    skills:
      jobs_url: {jobs_url}
      allow_insecure: true
      expected_image: nemo-skills
  routes:
    nemo-skills: skills
"""
_SSH_TUNNEL = "ssh_tunnel:\n  host: login.example\n  user: operator\n  identity: ~/.ssh/id\n"


def test_ray_target_uses_the_slurm_ssh_tunnel_block_without_changing_run_identity(
    tmp_path: Path,
) -> None:
    tunneled = tmp_path / "tunneled.yaml"
    tunneled.write_text(_RAY_TARGET.format(ssh_tunnel=_SSH_TUNNEL, jobs_url="http://10.0.0.1:8265"))
    direct = tmp_path / "direct.yaml"
    direct.write_text(_RAY_TARGET.format(ssh_tunnel="", jobs_url="http://10.0.0.1:8265"))

    resolved = resolve_target(TargetBinding.resolve("ray"), tunneled)

    assert resolved.ray is not None
    assert resolved.ray.ssh_tunnel == {
        "host": "login.example",
        "user": "operator",
        "identity": "~/.ssh/id",
    }
    assert resolved.fingerprint == resolve_target(TargetBinding.resolve("ray"), direct).fingerprint


@pytest.mark.parametrize(
    ("ssh_tunnel", "jobs_url", "message"),
    [
        (_SSH_TUNNEL, "https://ray.example", "must be http to use ssh_tunnel"),
        ("ssh_tunnel:\n  host: login.example\n", "http://10.0.0.1:8265", "ssh_tunnel.user"),
    ],
)
def test_ray_ssh_tunnel_rejects_unusable_settings(
    tmp_path: Path, ssh_tunnel: str, jobs_url: str, message: str
) -> None:
    target = tmp_path / "ray.yaml"
    target.write_text(_RAY_TARGET.format(ssh_tunnel=ssh_tunnel, jobs_url=jobs_url))

    with pytest.raises(TargetResolutionError, match=message):
        resolve_target(TargetBinding.resolve("ray"), target)


def test_resolved_ray_target_requires_image_identity(tmp_path: Path) -> None:
    target = tmp_path / "ray.yaml"
    target.write_text(
        """
executor: ray
ray:
  control_cluster: skills
  artifact_root: /shared/nvflow
  clusters:
    skills:
      jobs_url: http://127.0.0.1:8265
  routes:
    nemo-skills: skills
"""
    )

    with pytest.raises(TargetResolutionError, match="expected_image is required"):
        resolve_target(TargetBinding.resolve("ray"), target)


@pytest.mark.parametrize(
    ("environ", "unset"),
    [
        ({}, "RAY_TRAINING_URL"),
        ({"RAY_TRAINING_URL": "http://127.0.0.1:18265"}, "RAY_TRAINING_INTERNAL_URL"),
    ],
)
def test_unset_endpoint_variable_leaves_its_cluster_unconfigured_without_fallback(
    tmp_path: Path, environ: dict[str, str], unset: str
) -> None:
    target = tmp_path / "ray.yaml"
    target.write_text(
        """
executor: ray
ray:
  control_cluster: skills
  artifact_root: /shared/nvflow
  clusters:
    skills:
      jobs_url: http://127.0.0.1:8265
      expected_image: nemo-skills
    training:
      jobs_url: ${oc.env:RAY_TRAINING_URL}
      controller_jobs_url: ${oc.env:RAY_TRAINING_INTERNAL_URL}
      allow_insecure: true
      expected_image: nemo-rl
  routes:
    nemo-skills: skills
    nemo-rl: training
"""
    )
    binding = TargetBinding.resolve("ray")

    resolved = resolve_target(binding, target, environ=environ)

    assert resolved.ray is not None
    training = resolved.ray.clusters["training"]
    assert (training.jobs_url, training.controller_jobs_url) == (None, None)
    assert training.unset_variable == unset
    configured = resolve_target(
        binding,
        target,
        environ={
            "RAY_TRAINING_URL": "http://127.0.0.1:18265",
            "RAY_TRAINING_INTERNAL_URL": "http://training.internal:8265",
        },
    )
    assert configured.ray is not None
    assert configured.ray.clusters["training"].jobs_url_variable == "RAY_TRAINING_URL"
    assert resolved.fingerprint == configured.fingerprint


def test_ray_target_fingerprint_ignores_launcher_connection_endpoint(
    tmp_path: Path,
) -> None:
    target = tmp_path / "ray.yaml"
    target.write_text(
        """
executor: ray
ray:
  control_cluster: shared
  artifact_root: /shared/nvflow
  clusters:
    shared:
      jobs_url: ${oc.env:RAY_JOBS_URL}
      allow_insecure: true
      expected_image: project
  routes:
    project: shared
"""
    )
    binding = TargetBinding.resolve("ray")

    first = resolve_target(
        binding,
        target,
        environ={"RAY_JOBS_URL": "http://first.internal:8265"},
    )
    second = resolve_target(
        binding,
        target,
        environ={"RAY_JOBS_URL": "http://second.internal:8265"},
    )

    assert first.ray is not None
    assert second.ray is not None
    assert first.ray.controller.jobs_url != second.ray.controller.jobs_url
    assert first.fingerprint == second.fingerprint


def test_target_config_preflight_uses_config_environment(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "team-targets"
    config_dir.mkdir()
    target = config_dir / "target-a.yaml"
    target.write_text("executor: slurm\nsecret: do-not-render\n")

    location = preflight_target_config(
        "target-a",
        environ={"NVFLOW_CONFIG_DIR": str(config_dir)},
        cwd=tmp_path,
    )

    assert location == target.resolve()


def test_target_config_preflight_bridges_nvflow_directory_to_execution_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target-a.yaml"
    target.write_text("executor: slurm\n")
    monkeypatch.setenv("NVFLOW_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("NEMO_SKILLS_CONFIG_DIR", raising=False)

    config_path = preflight_target_config("target-a")

    with bind_target_config(config_path):
        assert os.environ["NEMO_SKILLS_CONFIG_DIR"] == str(tmp_path.resolve())
    assert "NEMO_SKILLS_CONFIG_DIR" not in os.environ


def test_target_config_preflight_resolves_relative_config_directory(tmp_path: Path) -> None:
    config_dir = tmp_path / "targets"
    config_dir.mkdir()
    target = config_dir / "target-a.yaml"
    target.write_text("executor: slurm\n")

    location = preflight_target_config(
        "target-a",
        environ={"NVFLOW_CONFIG_DIR": "targets"},
        cwd=tmp_path,
    )

    assert location == target.resolve()


def test_target_config_preflight_defaults_to_cwd_cluster_configs(tmp_path: Path) -> None:
    config_dir = tmp_path / "cluster_configs"
    config_dir.mkdir()
    target = config_dir / "target-a.yaml"
    target.write_text("executor: slurm\n")

    location = preflight_target_config("target-a", environ={}, cwd=tmp_path)

    assert location == target.resolve()


@pytest.mark.parametrize(
    "target",
    ["", " ", " target-a ", ".", "..", "../secret", "nested/target"],
)
def test_target_config_preflight_rejects_empty_or_path_like_names(target: str) -> None:
    with pytest.raises(TargetResolutionError):
        preflight_target_config(target, environ={})


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("- not\n- a\n- mapping\n", "must contain a YAML mapping"),
        ("secret: [do-not-render\n", "is not valid YAML"),
    ],
)
def test_target_config_preflight_rejects_invalid_yaml_without_leaking_contents(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    target = tmp_path / "target-a.yaml"
    target.write_text(contents)

    with pytest.raises(TargetResolutionError, match=message) as caught:
        preflight_target_config(
            "target-a",
            environ={"NVFLOW_CONFIG_DIR": str(tmp_path)},
        )

    assert "do-not-render" not in str(caught.value)


def test_target_config_preflight_reports_actionable_missing_config(tmp_path: Path) -> None:
    with pytest.raises(TargetResolutionError) as caught:
        preflight_target_config(
            "missing",
            environ={"NVFLOW_CONFIG_DIR": str(tmp_path)},
        )

    message = str(caught.value)
    assert "NVFLOW_CONFIG_DIR" in message
    assert "cluster_configs/template-slurm.yaml" in message
    assert "docs/operations/cluster-configuration.md" in message


@pytest.mark.requires_nemo
def test_live_plan_requires_runtime_target_fields_but_static_plan_does_not(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "target-a.yaml").write_text("executor: slurm\n")
    monkeypatch.setenv("NVFLOW_CONFIG_DIR", str(tmp_path))
    base = [
        "plan",
        "--workflow",
        "starter:generate",
        "--cluster",
        "target-a",
    ]

    static = CliRunner().invoke(cli.app, base)
    live = CliRunner().invoke(cli.app, [*base, "--live"])

    assert static.exit_code == 0, static.output
    assert live.exit_code == 1
    assert "job_dir must be provided" in live.stderr


def test_runner_override_does_not_change_workflow_definition(tmp_path: Path) -> None:
    config = PROJECT_ROOT / "projects" / "starter" / "workflows" / "generate.yaml"
    target = tmp_path / "operator-cluster.yaml"
    target.write_text("executor: slurm\n")

    runner = WorkflowRunner(
        str(config),
        cluster="operator-cluster",
        target_config_path=target,
    )

    assert runner.cluster == "operator-cluster"
    assert runner.target.source == "override"
    assert runner.definition.cluster == runner.config["cluster"]
    assert runner.definition.cluster != runner.cluster


def test_same_runner_cli_contract_dispatches_supplied_ray_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = PROJECT_ROOT / "projects" / "starter" / "workflows" / "generate.yaml"
    target = tmp_path / "ray.yaml"
    target.write_text(
        f"""
executor: ray
ray:
  control_cluster: shared
  artifact_root: {tmp_path / "artifacts"}
  clusters:
    shared:
      jobs_url: http://127.0.0.1:8265
      expected_image: starter
      expected_python: "3.13.14"
      expected_ray: "2.58.0"
  routes:
    nemo-skills: shared
    vllm: shared
"""
    )
    captured = {}

    class FakeEngine:
        def submit(self, graph, context):
            captured["graph"] = graph
            captured["target"] = context.target
            return SubmissionReceipt(
                plan_digest=graph.to_plan().digest,
                target=graph.target,
                plan_complete=True,
                opaque_stages=(),
                tasks=(),
                experiments=(),
                engine="ray",
                execution_digest=graph.digest,
                target_fingerprint=context.target.fingerprint,
                controller_handle="rayjobs://shared/controller",
            )

    monkeypatch.setattr("nvflow.execution.runner.engine_for", lambda _target: FakeEngine())
    provenance = ProjectResolver().resolve_reference("starter", config).source_provenance()
    monkeypatch.setattr(
        "nvflow.project.context.ProjectContext.source_provenance",
        lambda _self, **_kwargs: replace(
            provenance,
            clean=True,
            submittable=True,
            dirty_paths=(),
            reason=None,
        ),
    )

    receipt = WorkflowRunner(
        str(config),
        cluster="ray",
        target_config_path=target,
    ).run()

    assert receipt.engine == "ray"
    assert captured["graph"].engine == "ray"
    assert captured["target"].ray.control_cluster == "shared"


@pytest.mark.parametrize(
    ("selection", "stages"),
    [
        (["--all"], None),
        (["generate_answer"], ["generate_answer"]),
    ],
)
def test_cli_threads_cluster_override(
    selection: list[str],
    stages: list[str] | None,
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeRunner:
        def __init__(
            self,
            config: str,
            project_dir: str | None = None,
            *,
            cluster: str | None = None,
            project_search_paths: tuple[str, ...] = (),
            target_config_path: Path | None = None,
        ) -> None:
            self.project_context = SimpleNamespace(
                manifest=SimpleNamespace(entrypoint=None),
            )
            calls.append(
                {
                    "config": config,
                    "project_dir": project_dir,
                    "cluster": cluster,
                    "project_search_paths": project_search_paths,
                    "target_config_path": target_config_path,
                }
            )

        def run(
            self,
            stages: list[str] | None = None,
            environment: list[str] | None = None,
        ) -> None:
            calls[-1]["stages"] = stages
            calls[-1]["environment"] = environment

    monkeypatch.setattr(cli, "WorkflowRunner", FakeRunner)
    (tmp_path / "operator-cluster.yaml").write_text("executor: slurm\n")
    monkeypatch.setenv("NVFLOW_CONFIG_DIR", str(tmp_path))
    arguments = ["run", *selection]
    arguments.extend(
        [
            "--config",
            "workflow.yaml",
            "--cluster",
            "operator-cluster",
        ]
    )

    result = CliRunner().invoke(cli.app, arguments)

    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "config": "workflow.yaml",
            "project_dir": None,
            "cluster": "operator-cluster",
            "project_search_paths": (),
            "target_config_path": (tmp_path / "operator-cluster.yaml").resolve(),
            "stages": stages,
            "environment": None,
        }
    ]


@pytest.mark.parametrize(
    "arguments",
    [["plan"], ["run", "generate_answer"], ["run", "--all"]],
)
def test_cli_requires_explicit_cluster(arguments: list[str]) -> None:
    arguments = list(arguments)
    arguments.extend(["--workflow", "starter:generate"])

    result = CliRunner().invoke(cli.app, arguments)

    assert result.exit_code == 2
    assert "--cluster" in result.output
