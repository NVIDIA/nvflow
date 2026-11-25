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
"""Contracts for NVFlow's Gym image, nvflow-nemo-gym."""

from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).parents[1]
DOCKERFILE = ROOT / "dockerfiles/Dockerfile.nemo-gym"
GRPO_BASE = ROOT / "projects/finance/workflows/grpo/base.yaml"
DOWNLOAD_BASE = ROOT / "projects/finance/workflows/download/base.yaml"
TRIVY_IMAGES = ROOT / "ci/trivy-images.txt"


def test_gym_image_builds_on_the_nemo_gym_release() -> None:
    dockerfile = DOCKERFILE.read_text()

    assert "ARG NEMO_GYM_IMAGE=nvcr.io/nvidia/nemo-gym:v0.7.0" in dockerfile
    assert "FROM ${NEMO_GYM_IMAGE}" in dockerfile
    assert 'LABEL org.opencontainers.image.base.name="${NEMO_GYM_IMAGE}"' in dockerfile
    assert "git clone" not in dockerfile
    assert "python:" not in dockerfile
    assert "ENTRYPOINT []" in dockerfile
    assert "nvidian" not in dockerfile
    trivy_images = TRIVY_IMAGES.read_text()
    assert "nvflow-nemo-gym:v0.7.0" in trivy_images


def test_gym_image_bakes_required_finance_environments() -> None:
    dockerfile = DOCKERFILE.read_text()

    assert "WORKDIR /opt/nemo-gym" in dockerfile
    assert "/opt/nemo_gym_venv/bin" in dockerfile
    assert "local_vllm_model/configs/local_vllm_model.yaml" in dockerfile
    assert 'rm -rf "$local_component/.venv"' in dockerfile
    assert "UV_TORCH_BACKEND=cu130 UV_OVERRIDE=/tmp/gym-vllm.txt gym env prefetch" in dockerfile
    assert '"vllm": "0.25.1"' in dockerfile
    assert '"ray": "2.58.0"' in dockerfile
    assert '"openai": "2.44.0"' in dockerfile
    assert '(torch.version.cuda or "").startswith("13.")' in dockerfile
    assert '"nemotron_v3" in ReasoningParserManager.list_registered()' in dockerfile
    assert dockerfile.index("local_component=responses_api_models/local_vllm_model") < (
        dockerfile.index("COPY nvflow/integrations/nemo_gym/configs/format_verification.yaml")
    )
    assert dockerfile.count("unset UV_OFFLINE") == 2
    assert "export UV_OFFLINE=false" not in dockerfile
    for component in (
        "responses_api_models/vllm_model",
        "responses_api_models/local_vllm_model",
        "responses_api_models/openai_model",
        "responses_api_agents/simple_agent",
        "responses_api_agents/finance_agent",
        "resources_servers/equivalence_llm_judge",
        "resources_servers/finance_sec_search",
        "resources_servers/finance_agent_v2",
        "resources_servers/mcqa",
        "resources_servers/format_verification",
    ):
        assert component in dockerfile
    assert 'test -x "$component/.venv/bin/python"' in dockerfile
    assert "gym eval compare --help" in dockerfile


def test_gym_image_keeps_only_functional_runtime_checks() -> None:
    dockerfile = DOCKERFILE.read_text()

    assert "timeout_worker_healthcheck" in dockerfile
    assert "wandb-core-builder" not in dockerfile
    assert "WANDB_MODE=offline WANDB_SILENT=true" not in dockerfile
    assert "ray_dist.jar" not in dockerfile


def test_workflows_use_the_gym_image_top_level_environment() -> None:
    config = OmegaConf.load(GRPO_BASE)

    stage = config.stages["prepare_data"]
    assert stage.container == "nemo-gym"
    assert stage.gym_path == "/opt/nemo-gym"
    assert "/opt/nemo_gym_venv/bin" in stage.installation_command
    assert "local_vllm_model/.venv" not in stage.installation_command
    assert "gym env prefetch" not in stage.installation_command
    assert "uv sync" not in stage.installation_command

    # prefetch_cache belongs to the download workflow, not GRPO. It starts no
    # Gym server, so it needs the CLI on PATH and nothing else.
    prefetch = OmegaConf.load(DOWNLOAD_BASE).stages.prefetch_cache
    assert prefetch.container == "nemo-gym"
    assert prefetch.gym_path == "/opt/nemo-gym"
    assert "/opt/nemo_gym_venv/bin" in prefetch.installation_command
    assert "uv sync" not in prefetch.installation_command

    for stage_name in ("collect_rollouts", "rejudge"):
        stage = config.stages[stage_name]
        assert stage.container == "nemo-gym"
        assert "gym_uv_venv_dir" not in stage
