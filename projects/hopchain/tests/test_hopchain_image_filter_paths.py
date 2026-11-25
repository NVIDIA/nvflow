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
"""Regression tests for image paths passed from filtering into SDG."""

from __future__ import annotations

import argparse
import ast
import json
import shlex
from pathlib import Path
from typing import Any, cast

import pytest
from nvhopchain.stages.image_filter.image_filter import ImageFilterStage
from nvhopchain.utils import image_filter_preprocess
from nvhopchain.utils.inference import build_generate_kwargs, resolve_project_path
from nvhopchain.utils.multimodal_model_configs import resolve_model_config
from nvhopchain.utils.prepare_filtered_image_inputs import build_image_id

from nvflow.execution import WorkflowRunner

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_code_root_paths_are_not_rebased_onto_the_project_root() -> None:
    """Rendered ``project://`` paths already start at the worker code root."""
    rendered = "{{NVFLOW_CODE_ROOT}}/projects/hopchain/prompts/prompt.txt"

    assert resolve_project_path(rendered, "{{NVFLOW_CODE_ROOT}}") == rendered
    assert resolve_project_path("prompts/prompt.txt", "/code") == "/code/prompts/prompt.txt"
    assert resolve_project_path("/data/prompt.txt", "/code") == "/data/prompt.txt"


def test_recursive_catalog_preserves_paths_and_unique_ids(tmp_path, monkeypatch) -> None:
    """Nested files with the same basename must remain distinct and readable."""
    image_root = tmp_path / "images"
    first_image = image_root / "Chart" / "same.png"
    second_image = image_root / "Document" / "nested" / "same.png"
    first_image.parent.mkdir(parents=True)
    second_image.parent.mkdir(parents=True)
    first_image.write_bytes(b"chart")
    second_image.write_bytes(b"document")

    output = tmp_path / "messages.jsonl"
    catalog_output = tmp_path / "resolved_catalog.jsonl"
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Score this image.")
    monkeypatch.setattr(
        image_filter_preprocess,
        "parse_args",
        lambda: argparse.Namespace(
            input_catalog=None,
            input_catalog_json=json.dumps([{"directory": str(image_root), "recursive": True}]),
            output=str(output),
            catalog_output=str(catalog_output),
            prompt=str(prompt),
            use_base64=False,
            max_image_dimension=None,
        ),
    )

    image_filter_preprocess.main()

    rows = [json.loads(line) for line in catalog_output.read_text().splitlines()]
    assert {row["image_file_name"] for row in rows} == {
        "Chart/same.png",
        "Document/nested/same.png",
    }
    reconstructed = [Path(row["image_directory"]) / row["image_file_name"] for row in rows]
    assert all(path.exists() for path in reconstructed)
    assert (
        len({build_image_id(row["image_directory"], row["image_file_name"]) for row in rows}) == 2
    )


def test_hopchain_demo_workflows_are_yaml_first(monkeypatch) -> None:
    """Documented demos must resolve with no NVFLOW environment configuration."""
    project_root = REPO_ROOT
    monkeypatch.setenv("PWD", str(project_root))
    for variable in (
        "NVFLOW_CLUSTER",
        "NVFLOW_PROJECT_ROOT",
        "NVFLOW_HOPCHAIN_DATA_DIR",
        "NVFLOW_HOPCHAIN_IMAGE_CATALOG",
        "NVFLOW_HOPCHAIN_SOURCE_IMAGES",
        "NVFLOW_GPU_PARTITION",
        "NVFLOW_CPU_PARTITION",
        "NVFLOW_QWEN_MODEL_PATH",
        "NVFLOW_QWEN_SERVER_CONTAINER",
        "NVFLOW_OMNI_MODEL_PATH",
        "NVFLOW_OMNI_SERVER_CONTAINER",
        "NVFLOW_SAM_CONTAINER",
        "NVFLOW_CONFIG_DIR",
    ):
        monkeypatch.delenv(variable, raising=False)

    image_config_path = project_root / "projects/hopchain/workflows/image_filter/demo.yaml"
    image_config = cast(dict[str, Any], WorkflowRunner(str(image_config_path)).config)
    assert image_config["cluster"] == "my_cluster"
    assert image_config["launcher_project_root"] == str(project_root)
    assert image_config["worker_code_dir"] == "{{NVFLOW_CODE_ROOT}}"
    assert image_config["worker_data_dir"] == "/workspace"
    assert image_config["project_root"] == "{{NVFLOW_CODE_ROOT}}"
    assert image_config["cluster_config_dir"] == str(project_root / "cluster_configs")
    assert image_config["stages"]["image_filter"]["project_root"] == "{{NVFLOW_CODE_ROOT}}"
    assert image_config["stages"]["image_filter"]["cluster_config_dir"] == str(
        project_root / "cluster_configs"
    )
    assert image_config["execution_id"] == "demo"
    assert image_config["stages"]["image_filter"]["image_directories"] == [
        {
            "directory": "/workspace/data/images",
            "recursive": True,
            "end_index": 100,
        }
    ]
    assert (
        "/hf_models/Qwen/Qwen3.5-397B-A17B" in image_config["model_profiles"]["qwen"]["server_args"]
    )

    sdg_config_path = project_root / "projects/hopchain/workflows/sdg/demo.yaml"
    sdg_config = cast(dict[str, Any], WorkflowRunner(str(sdg_config_path)).config)
    assert sdg_config["cluster"] == "my_cluster"
    assert sdg_config["execution_id"] == "demo"
    assert sdg_config["image_filter_execution_id"] == "demo"
    assert sdg_config["project_root"] == "{{NVFLOW_CODE_ROOT}}"
    assert sdg_config["cluster_config_dir"] == str(project_root / "cluster_configs")
    assert all(
        stage.get("project_root") == "{{NVFLOW_CODE_ROOT}}"
        for stage in sdg_config["stages"].values()
    )
    assert all(
        stage.get("cluster_config_dir") == str(project_root / "cluster_configs")
        for stage in sdg_config["stages"].values()
    )
    assert sdg_config["source_kept_images_file"] == (
        "/workspace/outputs/hopchain/image_filter/execution/demo/image-filter/kept_images.jsonl"
    )
    assert "judge_candidate_queries_openai" not in sdg_config["pipeline_stages"]
    assert all("partition" not in stage for stage in sdg_config["stages"].values())


def test_hopchain_workflows_honor_launcher_cluster_config_env(monkeypatch) -> None:
    """The airgapped client must use the directory mounted by its launcher."""
    monkeypatch.setenv("PWD", str(REPO_ROOT))
    monkeypatch.setenv("NVFLOW_CONFIG_DIR", "/work")

    workflow_paths = (
        REPO_ROOT / "projects/hopchain/workflows/image_filter/full.yaml",
        REPO_ROOT / "projects/hopchain/workflows/sdg/full.yaml",
    )
    for workflow_path in workflow_paths:
        config = cast(dict[str, Any], WorkflowRunner(str(workflow_path)).config)
        assert config["cluster_config_dir"] == "/work"
        assert all(
            stage.get("cluster_config_dir") == "/work" for stage in config["stages"].values()
        )


def test_full_workflows_use_full_execution_namespace(monkeypatch) -> None:
    project_root = REPO_ROOT
    monkeypatch.setenv("PWD", str(project_root))

    image_config_path = project_root / "projects/hopchain/workflows/image_filter/full.yaml"
    image_config = cast(dict[str, Any], WorkflowRunner(str(image_config_path)).config)
    assert image_config["execution_id"] == "full"
    assert image_config["directories"]["image-filter"] == (
        "/workspace/outputs/hopchain/image_filter/execution/full/image-filter"
    )

    sdg_config_path = project_root / "projects/hopchain/workflows/sdg/full.yaml"
    sdg_config = cast(dict[str, Any], WorkflowRunner(str(sdg_config_path)).config)
    assert sdg_config["execution_id"] == "full"
    assert sdg_config["image_filter_execution_id"] == "full"
    assert sdg_config["source_kept_images_file"] == (
        "/workspace/outputs/hopchain/image_filter/execution/full/image-filter/kept_images.jsonl"
    )


def test_inline_image_directories_are_sent_to_worker_without_local_io() -> None:
    """Inline image directories must be resolved for the worker, not the launcher."""
    catalog_arg = ImageFilterStage()._build_input_catalog_arg(
        {
            "project_root": "{{NVFLOW_CODE_ROOT}}",
            "image_directories": [
                {
                    "directory": "/workspace/data/images",
                    "recursive": True,
                    "end_index": 25,
                }
            ],
        }
    )

    args = shlex.split(catalog_arg)
    assert args[0] == "--input-catalog-json"
    assert json.loads(args[1]) == [
        {
            "directory": "/workspace/data/images",
            "start_index": None,
            "end_index": 25,
            "recursive": True,
            "shuffle_seed": None,
        }
    ]


def test_worker_image_directory_does_not_require_launcher_access() -> None:
    config = {
        "output_dir": "/workspace/outputs/hopchain/image-filter",
        "project_root": "{{NVFLOW_CODE_ROOT}}",
        "prompt_file": "prompts/hopchain_image_filter.txt",
        "image_directories": [{"directory": "/workspace/data/images"}],
        "model_config": {
            "model": "qwen",
            "server_type": "sglang",
            "server_nodes": 1,
            "server_gpus": 1,
            "server_args": "--model-path /hf_models/qwen",
        },
    }

    ImageFilterStage().validate_config(config)


def test_generate_submissions_create_remote_output_directories(monkeypatch) -> None:
    """NeMo-Skills must create fresh workspace output and log directories remotely."""
    from nvflow.integrations.nemo_skills import commands

    monkeypatch.setattr(commands, "wrap_arguments", lambda arguments: arguments)

    model_config = resolve_model_config(
        {
            "model_config": {
                "model": "qwen",
                "server_type": "sglang",
                "server_nodes": 1,
                "server_gpus": 1,
                "server_args": "--model-path /hf_models/qwen",
            }
        }
    )
    generate_kwargs = build_generate_kwargs(
        ctx_args="",
        cluster="my_cluster",
        model_config=model_config,
        output_dir="/workspace/outputs/hopchain/image-filter",
        input_file="/workspace/outputs/hopchain/image-filter/input.jsonl",
        expname="image-filter",
        preprocess_cmd=None,
        postprocess_cmd="true",
        stage_config={"partition": "gpu"},
    )

    assert generate_kwargs["check_mounted_paths"] is True


def test_all_run_cmd_submissions_create_remote_log_directories() -> None:
    """Every direct HopChain run_cmd call must create its remote log path."""
    stages_root = REPO_ROOT / "projects/hopchain/nvhopchain/stages"
    run_cmd_count = 0
    missing_flags: list[str] = []

    for source_path in stages_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "run_cmd":
                continue

            run_cmd_count += 1
            flag = next(
                (
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg == "check_mounted_paths"
                ),
                None,
            )
            if not isinstance(flag, ast.Constant) or flag.value is not True:
                missing_flags.append(f"{source_path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert run_cmd_count > 0
    assert not missing_flags, f"check_mounted_paths=True missing from: {missing_flags}"


def test_model_and_partition_configuration_comes_from_yaml() -> None:
    """Project and cluster YAML are the only model and partition sources."""
    model_config = resolve_model_config(
        {
            "model_config": {
                "model": "private-qwen",
                "server_type": "sglang",
                "server_nodes": 1,
                "server_gpus": 4,
                "server_args": "--model-path /hf_models/private-qwen --tp 4",
            },
        }
    )
    assert model_config.server_nodes == 1
    assert model_config.server_gpus == 4
    assert model_config.server_args == "--model-path /hf_models/private-qwen --tp 4"
    with pytest.raises(ValueError, match="Missing required field: model_config"):
        resolve_model_config({})

    def partition(stage_config: dict[str, Any]) -> str | None:
        return build_generate_kwargs(
            ctx_args="",
            cluster="my_cluster",
            model_config=model_config,
            output_dir="/workspace/outputs/hopchain/image-filter",
            input_file="/workspace/outputs/hopchain/image-filter/input.jsonl",
            expname="image-filter",
            preprocess_cmd=None,
            postprocess_cmd="true",
            stage_config=stage_config,
        )["partition"]

    # The Slurm target supplies its partition at submission; Ray targets have none.
    assert partition({}) is None
    assert partition({"partition": "debug"}) == "debug"
