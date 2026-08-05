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
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

from nvflow.core.workflow_runner import WorkflowRunner
from nvflow.recipes.multimodal.stages.image_filter.image_filter import ImageFilterStage
from nvflow.recipes.multimodal.utils import image_filter_preprocess
from nvflow.recipes.multimodal.utils.multimodal_model_configs import resolve_model_config
from nvflow.recipes.multimodal.utils.prepare_filtered_image_inputs import build_image_id
from nvflow.recipes.multimodal.utils.runtime_env import resolve_partition


def test_recursive_catalog_preserves_paths_and_unique_ids(tmp_path, monkeypatch) -> None:
    """Nested files with the same basename must remain distinct and readable."""
    image_root = tmp_path / "images"
    first_image = image_root / "Chart" / "same.png"
    second_image = image_root / "Document" / "nested" / "same.png"
    first_image.parent.mkdir(parents=True)
    second_image.parent.mkdir(parents=True)
    first_image.write_bytes(b"chart")
    second_image.write_bytes(b"document")

    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps([{"directory": str(image_root), "recursive": True}]))
    output = tmp_path / "messages.jsonl"
    catalog_output = tmp_path / "resolved_catalog.jsonl"
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Score this image.")
    monkeypatch.setattr(
        image_filter_preprocess,
        "parse_args",
        lambda: argparse.Namespace(
            input_catalog=str(catalog),
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


def test_multimodal_demo_workflows_are_yaml_first(monkeypatch) -> None:
    """Documented demos must resolve with no NVFLOW environment configuration."""
    project_root = Path(__file__).parents[1]
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
    ):
        monkeypatch.delenv(variable, raising=False)

    image_config_path = (
        project_root
        / "nvflow/recipes/multimodal/workflows/image_filter/hopchain-image-filter-demo.yaml"
    )
    image_config = cast(dict[str, Any], WorkflowRunner(str(image_config_path)).config)
    assert image_config["cluster"] == "my_cluster"
    assert image_config["project_root"] == str(project_root)
    assert image_config["cluster_config_dir"] == str(project_root / "cluster_configs")
    assert image_config["execution_id"] == "demo"
    assert image_config["stages"]["image_filter"]["image_directories"] == [
        {
            "directory": str(project_root / "data/images"),
            "recursive": True,
            "end_index": 100,
        }
    ]
    assert (
        "/hf_models/Qwen/Qwen3.5-397B-A17B" in image_config["model_profiles"]["qwen"]["server_args"]
    )

    sdg_config_path = (
        project_root / "nvflow/recipes/multimodal/workflows/sdg/hopchain-sdg-demo.yaml"
    )
    sdg_config = cast(dict[str, Any], WorkflowRunner(str(sdg_config_path)).config)
    assert sdg_config["cluster"] == "my_cluster"
    assert sdg_config["execution_id"] == "demo"
    assert sdg_config["image_filter_execution_id"] == "demo"
    assert sdg_config["source_kept_images_file"] == str(
        project_root / "outputs/hopchain/image_filter/execution/demo/image-filter/kept_images.jsonl"
    )
    assert "judge_candidate_queries_openai" not in sdg_config["pipeline_stages"]
    assert all("partition" not in stage for stage in sdg_config["stages"].values())


def test_full_workflows_use_full_execution_namespace(monkeypatch) -> None:
    project_root = Path(__file__).parents[1]
    monkeypatch.setenv("PWD", str(project_root))

    image_config_path = (
        project_root / "nvflow/recipes/multimodal/workflows/image_filter/hopchain-image-filter.yaml"
    )
    image_config = cast(dict[str, Any], WorkflowRunner(str(image_config_path)).config)
    assert image_config["execution_id"] == "full"
    assert image_config["directories"]["image-filter"] == str(
        project_root / "outputs/hopchain/image_filter/execution/full/image-filter"
    )

    sdg_config_path = project_root / "nvflow/recipes/multimodal/workflows/sdg/hopchain-sdg.yaml"
    sdg_config = cast(dict[str, Any], WorkflowRunner(str(sdg_config_path)).config)
    assert sdg_config["execution_id"] == "full"
    assert sdg_config["image_filter_execution_id"] == "full"
    assert sdg_config["source_kept_images_file"] == str(
        project_root / "outputs/hopchain/image_filter/execution/full/image-filter/kept_images.jsonl"
    )


def test_inline_image_directories_materialize_a_catalog(tmp_path) -> None:
    """The simple image-folder input must produce the runtime catalog automatically."""
    project_root = tmp_path / "project"
    image_dir = project_root / "data/images"
    image_dir.mkdir(parents=True)

    catalog_path = ImageFilterStage()._resolve_input_catalog(
        {
            "project_root": str(project_root),
            "image_directories": [
                {
                    "directory": "data/images",
                    "recursive": True,
                    "end_index": 25,
                }
            ],
        },
        output_dir=tmp_path / "output/temp",
    )

    assert json.loads(Path(catalog_path).read_text()) == [
        {
            "directory": str(image_dir),
            "start_index": None,
            "end_index": 25,
            "recursive": True,
            "shuffle_seed": None,
        }
    ]


def test_model_and_partition_configuration_comes_from_yaml(monkeypatch) -> None:
    """Recipe and cluster YAML are the only model and partition sources."""
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

    nemo_skills_module = ModuleType("nemo_skills")
    pipeline_module = ModuleType("nemo_skills.pipeline")
    utils_module = ModuleType("nemo_skills.pipeline.utils")
    utils_module.get_cluster_config = lambda cluster, config_dir: {  # type: ignore[attr-defined]
        "partition": "gpu",
        "cpu_partition": "cpu",
    }
    monkeypatch.setitem(sys.modules, "nemo_skills", nemo_skills_module)
    monkeypatch.setitem(sys.modules, "nemo_skills.pipeline", pipeline_module)
    monkeypatch.setitem(sys.modules, "nemo_skills.pipeline.utils", utils_module)
    config = {"cluster_config_dir": "/workspace/nvflow/cluster_configs"}
    assert resolve_partition(config, "my_cluster") == "gpu"
    assert resolve_partition(config, "my_cluster", cpu=True) == "cpu"
    assert resolve_partition({**config, "partition": "debug"}, "my_cluster") == "debug"
