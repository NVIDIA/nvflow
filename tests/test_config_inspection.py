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
"""Tests for safe workflow configuration inspection."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import yaml
from typer.testing import CliRunner

from nvflow.cli import main as cli
from nvflow.cli.main import app
from nvflow.core import load_workflow


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _inspection_project(tmp_path: Path) -> Path:
    root = tmp_path / "inspection-project"
    workflows = root / "workflows"
    workflows.mkdir(parents=True)
    _write_yaml(
        root / "project.yaml",
        {"manifest_version": 1, "project": "inspection-project"},
    )
    _write_yaml(
        workflows / "base.yaml",
        {
            "workflow": {"name": "base", "type": "test"},
            "cluster": "default-cluster",
            "api_key": "${oc.env:NVFLOW_INSPECTION_SECRET,null}",
            "fallback_api_key": ("${oc.env:NVFLOW_FALLBACK_SECRET,${defaults.token}-suffix}"),
            "composite_api_key": "${oc.env:NVFLOW_COMPOSITE_SECRET,hard-coded}-suffix",
            "literal_token": "literal-secret-value",
            "train_hf_token": "train-secret-value",
            "tokens": "plural-secret-value",
            "dollar_api_key": "$literal-dollar-secret",
            "password": 1234,
            "self_token": "self_token",
            "tokenizer": "shared-tokenizer",
            "model_label": "shared-tokenizer",
            "api_key_name": "OPENAI_API_KEY",
            "endpoint": "${oc.env:BASE_ENDPOINT,https://example.com}",
            "defaults": {"token": "hard-coded-fallback"},
            "nested": {"unchanged": 1, "changed": "base"},
            "pipeline_stages": ["inspect", "other"],
            "stages": {
                "inspect": {
                    "api_key": "${oc.env:NVFLOW_INSPECTION_SECRET,null}",
                    "literal_token": "literal-secret-value",
                    "max_output_tokens": 32768,
                    "train_mb_tokens": 131072,
                    "logprob_mb_tokens": 131072,
                    "profiling_datasets": {
                        "dataset-a": {"source_data": "/data/a.jsonl"},
                        "dataset-b": {"source_data": "/data/b.jsonl"},
                    },
                    "environments": {
                        "dataset-a": {"source_data": "/data/a.jsonl"},
                        "dataset-b": {"source_data": "/data/b.jsonl"},
                    },
                    "value": "${nested.changed}",
                    "dependencies": [],
                },
                "other": {"value": "other", "dependencies": []},
            },
        },
    )
    child = workflows / "child.yaml"
    _write_yaml(
        child,
        {
            "_base_": "base.yaml",
            "workflow": {"name": "child"},
            "nested": {"changed": "child"},
            "endpoint": "${oc.env:ENDPOINT,user:password}",
        },
    )
    return child


def test_config_views_preserve_composition_and_redact_secrets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _inspection_project(tmp_path)
    monkeypatch.setenv("NVFLOW_INSPECTION_SECRET", "sentinel-secret-value")

    definition = load_workflow(config)

    assert definition.config_view("source")["_base_"] == "base.yaml"
    assert "project" not in definition.config_view("source")
    assert definition.config_view("merged")["api_key"] == "${oc.env:NVFLOW_INSPECTION_SECRET,null}"
    assert "project" not in definition.config_view("merged")
    assert definition.config_view("merged")["literal_token"] == "<redacted:literal_token>"
    assert definition.config_view("merged")["train_hf_token"] == "<redacted:train_hf_token>"
    assert definition.config_view("merged")["tokens"] == "<redacted:tokens>"
    assert definition.config_view("merged")["dollar_api_key"] == "<redacted:dollar_api_key>"
    assert definition.config_view("merged")["password"] == "<redacted:password>"
    assert definition.config_view("merged")["self_token"] == "<redacted:self_token>"
    assert definition.config_view("merged")["tokenizer"] == "shared-tokenizer"
    assert definition.config_view("merged")["model_label"] == "shared-tokenizer"
    assert definition.config_view("resolved")["api_key"] == (
        "<redacted:oc.env:NVFLOW_INSPECTION_SECRET>"
    )
    assert definition.config_view("resolved")["project"] == "inspection-project"
    assert definition.config_view("resolved")["dollar_api_key"] == "<redacted:dollar_api_key>"
    assert definition.config_view("resolved")["tokens"] == "<redacted:tokens>"
    assert definition.config_view("resolved")["password"] == "<redacted:password>"
    assert definition.config_view("resolved")["tokenizer"] == "shared-tokenizer"
    assert definition.config_view("resolved")["model_label"] == "shared-tokenizer"
    assert definition.config_view("resolved")["stages"]["inspect"]["max_output_tokens"] == 32768
    for name in ("train_mb_tokens", "logprob_mb_tokens"):
        assert definition.config_view("resolved")["stages"]["inspect"][name] == 131072
    assert "131072" in definition.redact_runtime_values("--max-model-len 131072")
    assert "train-secret-value" not in definition.redact_runtime_values(
        "--token train-secret-value"
    )
    assert definition.config_view("resolved")["api_key_name"] == "OPENAI_API_KEY"
    assert "sentinel-secret-value" not in yaml.safe_dump(definition.config_view("resolved"))
    redacted_plan = definition.redact_runtime_values(
        {
            "command": (
                "python worker.py --token=sentinel-secret-value "
                "--literal=literal-secret-value --tokenizer=shared-tokenizer"
            )
        }
    )
    assert "sentinel-secret-value" not in redacted_plan["command"]
    assert "literal-secret-value" not in redacted_plan["command"]
    assert "--tokenizer=shared-tokenizer" in redacted_plan["command"]
    assert "<redacted:" in redacted_plan["command"]
    assert definition.config_view("diff") == {
        "workflow": {"name": "child"},
        "nested": {"changed": "child"},
        "endpoint": "<redacted:oc.env-fallback>",
    }


def test_show_config_never_prints_resolved_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _inspection_project(tmp_path)
    monkeypatch.setenv("NVFLOW_INSPECTION_SECRET", "sentinel-secret-value")

    result = CliRunner().invoke(
        app,
        ["show-config", "--config", str(config), "--view", "resolved"],
    )

    assert result.exit_code == 0, result.output
    assert "sentinel-secret-value" not in result.output
    assert "hard-coded-fallback" not in result.output
    assert "<redacted:oc.env:NVFLOW_INSPECTION_SECRET>" in result.output


def test_unresolved_cli_views_need_no_secret_and_redact_literals(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _inspection_project(tmp_path)
    monkeypatch.delenv("NVFLOW_INSPECTION_SECRET", raising=False)
    runner = CliRunner()

    source = runner.invoke(app, ["show-config", "-c", str(config), "-V", "source"])
    merged = runner.invoke(app, ["show-config", "-c", str(config), "-V", "merged"])

    assert source.exit_code == 0, source.output
    assert "_base_: base.yaml" in source.output
    assert merged.exit_code == 0, merged.output
    assert "${oc.env:NVFLOW_INSPECTION_SECRET,null}" in merged.output
    assert "hard-coded-fallback" not in merged.output
    assert "<redacted:fallback_api_key>" in merged.output
    assert "hard-coded}-suffix" not in merged.output
    assert "<redacted:composite_api_key>" in merged.output
    assert "literal-secret-value" not in merged.output
    assert "literal-dollar-secret" not in merged.output
    assert "1234" not in merged.output
    assert "<redacted:literal_token>" in merged.output
    assert "<redacted:dollar_api_key>" in merged.output
    assert "<redacted:self_token>" in merged.output
    assert "tokenizer: shared-tokenizer" in merged.output
    assert "model_label: shared-tokenizer" in merged.output
    assert "api_key_name: OPENAI_API_KEY" in merged.output
    assert "user:password" not in merged.output
    assert "<redacted:oc.env-fallback>" in merged.output


def test_stage_config_view_is_effective_and_redacted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _inspection_project(tmp_path)
    monkeypatch.setenv("NVFLOW_INSPECTION_SECRET", "sentinel-secret-value")

    definition = load_workflow(config)
    view = definition.config_view(
        "resolved",
        stages=["inspect", "other", "inspect"],
    )

    assert list(view["stages"]) == ["inspect", "other"]
    assert view["stages"]["inspect"]["value"] == "child"
    assert view["stages"]["inspect"]["api_key"] == "<redacted:api_key>"
    assert view["stages"]["inspect"]["literal_token"] == "<redacted:literal_token>"
    assert view["stages"]["inspect"]["max_output_tokens"] == 32768
    assert "environments" not in view["stages"]["inspect"]
    assert list(view["stages"]["inspect"]["profiling_datasets"]) == [
        "dataset-a",
        "dataset-b",
    ]
    assert "sentinel-secret-value" not in yaml.safe_dump(view)


def test_stage_config_view_selects_environment_and_hides_duplicate_alias(
    tmp_path: Path,
) -> None:
    config = _inspection_project(tmp_path)
    definition = load_workflow(config)

    view = definition.config_view(
        "resolved",
        stages=["inspect"],
        environment=["dataset-b"],
    )

    stage = view["stages"]["inspect"]
    assert list(stage["profiling_datasets"]) == ["dataset-b"]
    assert "environments" not in stage
    assert (
        definition.config_view(
            "resolved",
            stages=["other"],
            environment=["dataset-b"],
        )["stages"]["other"]["value"]
        == "other"
    )

    result = CliRunner().invoke(
        app,
        [
            "show-config",
            "inspect",
            "--config",
            str(config),
            "-e",
            "dataset-b",
        ],
    )
    assert result.exit_code == 0, result.output
    rendered = yaml.safe_load(result.output)["stages"]["inspect"]
    assert list(rendered["profiling_datasets"]) == ["dataset-b"]
    assert "environments" not in rendered
    assert rendered["max_output_tokens"] == 32768

    unknown = CliRunner().invoke(
        app,
        [
            "show-config",
            "inspect",
            "--config",
            str(config),
            "-e",
            "missing",
        ],
    )
    assert unknown.exit_code == 1
    assert "Unknown environment(s): missing" in unknown.output


def test_show_config_accepts_resolved_stages_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _inspection_project(tmp_path)
    monkeypatch.setenv("NVFLOW_INSPECTION_SECRET", "sentinel-secret-value")
    runner = CliRunner()

    selected = runner.invoke(app, ["show-config", "inspect", "--config", str(config)])
    source = runner.invoke(
        app,
        ["show-config", "inspect", "--config", str(config), "--view", "source"],
    )
    unscoped = runner.invoke(
        app,
        ["show-config", "--config", str(config), "-e", "dataset-a"],
    )

    assert selected.exit_code == 0, selected.output
    assert list(yaml.safe_load(selected.output)["stages"]) == ["inspect"]
    assert "sentinel-secret-value" not in selected.output
    assert source.exit_code == 2
    assert "Positional stage names require --view resolved" in source.output
    assert unscoped.exit_code == 2
    assert "--environment requires positional stage" in unscoped.output


def test_stage_config_view_includes_dynamically_expanded_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "dynamic-project"
    workflows = root / "workflows"
    workflows.mkdir(parents=True)
    _write_yaml(root / "project.yaml", {"manifest_version": 1, "project": "dynamic-project"})
    config = workflows / "models.yaml"
    _write_yaml(
        config,
        {
            "workflow": {"name": "dynamic", "type": "test"},
            "cluster": "cluster-a",
            "pipeline_stages": ["model-a"],
            "stages": {},
            "models": {
                "model-a": {
                    "model_path": "/models/a",
                    "api_key": "${oc.env:NVFLOW_DYNAMIC_SECRET,null}",
                    "dependencies": [],
                }
            },
        },
    )
    monkeypatch.setenv("NVFLOW_DYNAMIC_SECRET", "dynamic-secret-value")

    view = load_workflow(config, project_dir=root).config_view(
        "resolved",
        stages=["model-a"],
    )

    assert view["stages"]["model-a"]["model_path"] == "/models/a"
    assert view["stages"]["model-a"]["api_key"] == "<redacted:api_key>"
    assert "dynamic-secret-value" not in yaml.safe_dump(view)


def test_plan_json_redacts_resolved_secrets(
    monkeypatch,
    tmp_path: Path,
) -> None:
    secret = "plan-secret-value"
    plan_arguments: dict[str, Any] = {}
    definition = types.SimpleNamespace(
        cluster="cluster-a",
        project_context=types.SimpleNamespace(
            manifest=types.SimpleNamespace(entrypoint=None),
        ),
        redact_runtime_values=lambda value: {
            **value,
            "command": value["command"].replace(secret, "<redacted>"),
        },
    )
    plan = types.SimpleNamespace(
        complete=True,
        to_dict=lambda: {"command": f"python worker.py --token={secret}"},
    )
    monkeypatch.setattr(cli, "load_workflow", lambda *_args, **_kwargs: definition)

    def build_graph(_resolved_target, **kwargs):
        plan_arguments.update(kwargs)
        return types.SimpleNamespace(to_plan=lambda **_kwargs: plan)

    monkeypatch.setattr(
        cli,
        "WorkflowPlanner",
        lambda *_args, **_kwargs: types.SimpleNamespace(execution_graph=build_graph),
    )
    (tmp_path / "cluster-a.yaml").write_text("executor: slurm\n")
    monkeypatch.setenv("NVFLOW_CONFIG_DIR", str(tmp_path))

    result = CliRunner().invoke(
        app,
        [
            "plan",
            "training",
            "eval",
            "--config",
            "workflow.yaml",
            "--cluster",
            "cluster-a",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert plan_arguments["stages"] == ["training", "eval"]
    assert secret not in result.output
    assert "<redacted>" in result.output


def test_show_config_rejects_unknown_view(tmp_path: Path) -> None:
    config = _inspection_project(tmp_path)

    result = CliRunner().invoke(
        app,
        ["show-config", "--config", str(config), "--view", "everything"],
    )

    assert result.exit_code != 0
    assert "--view must be one of" in result.output
