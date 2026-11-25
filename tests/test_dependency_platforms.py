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
"""Regression checks for platform-specific dependency routing."""

from __future__ import annotations

import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

from omegaconf import OmegaConf
from packaging.requirements import Requirement

from nvflow.project import environment as project_environment


def test_cpu_torch_routing_and_intel_macos_policy() -> None:
    """Route Linux x86/arm64 and Windows x86 torch to CPU wheels, leaving Intel macOS unmatched."""
    repo_root = Path(__file__).parents[1]
    config = tomllib.loads((repo_root / "pyproject.toml").read_text())

    assert config["project"]["requires-python"] == ">=3.13.14,<3.14"
    assert project_environment._PYTHON_REQUIREMENT == config["project"]["requires-python"]
    ray_requirement = Requirement(config["project"]["optional-dependencies"]["ray"][0])
    assert str(ray_requirement.specifier) == "==2.58.0"
    assert config["tool"]["uv"]["sources"]["torch"] == [
        {
            "index": "pytorch-cpu",
            "marker": (
                "(platform_machine == 'x86_64' or platform_machine == 'AMD64' "
                "or platform_machine == 'aarch64') and sys_platform != 'darwin'"
            ),
        }
    ]
    base_dependencies = {
        Requirement(value).name.lower() for value in config["project"]["dependencies"]
    }
    execution_dependencies = config["project"]["optional-dependencies"]["execution"]
    execution_names = {Requirement(value).name for value in execution_dependencies}
    assert {"nemo-skills", "torch"} <= execution_names
    assert {"nemo-skills", "torch"}.isdisjoint(base_dependencies)
    assert "pyyaml" in base_dependencies
    nemo_run = next(
        Requirement(value)
        for value in config["tool"]["uv"]["override-dependencies"]
        if Requirement(value).name == "nemo-run"
    )
    assert nemo_run.url is not None
    assert nemo_run.url.endswith("@17ae86b64d7f75653351664f5d8c9e466faede00")
    assert {"httpx", "nemo-run", "ray", "urllib3"} <= {
        Requirement(value).name for value in config["tool"]["uv"]["override-dependencies"]
    }
    assert project_environment._PYTORCH_CPU_INDEX == config["tool"]["uv"]["index"][0]["url"]
    assert (
        project_environment._TORCH_CPU_MARKER
        == (config["tool"]["uv"]["sources"]["torch"][0]["marker"])
    )
    readme = (repo_root / "README.md").read_text()
    assert "Python 3.13.14" in readme
    assert "Intel macOS is not supported" in readme


def test_projects_are_not_root_workspace_members() -> None:
    repo_root = Path(__file__).parents[1]
    root = tomllib.loads((repo_root / "pyproject.toml").read_text())
    project_files = sorted((repo_root / "projects").glob("*/pyproject.toml"))

    assert "workspace" not in root["tool"]["uv"]
    assert "nvflow" not in root["tool"]["uv"]["sources"]
    assert {path.parent.name for path in project_files} == {"finance", "hopchain", "starter"}
    launcher_dependencies = {
        "finance": {"nvflow"},
        "hopchain": {"nvflow", "pydantic"},
        "starter": {"nvflow"},
    }
    for path in project_files:
        project = tomllib.loads(path.read_text())
        assert project["project"]["requires-python"] == ">=3.13.14,<3.14"
        assert set(project["tool"]["uv"]) == {"sources"}
        assert project["tool"]["uv"]["sources"]["nvflow"] == {
            "git": "https://github.com/NVIDIA/nvflow.git",
            "rev": "v1.2.0",
        }
        assert {
            Requirement(value).name for value in project["project"]["dependencies"]
        } == launcher_dependencies[path.parent.name]
        nvflow = next(
            Requirement(value)
            for value in project["project"]["dependencies"]
            if Requirement(value).name == "nvflow"
        )
        if path.parent.name == "finance":
            assert not nvflow.specifier.contains("1.1.2")
            assert nvflow.specifier.contains("1.1.3.dev999", prereleases=True)
        else:
            assert nvflow.specifier.contains("1.1")
    assert tomllib.loads((repo_root / "projects/finance/pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]["worker"]
    assert tomllib.loads((repo_root / "projects/hopchain/pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]["worker"]

    client_dockerfile = (repo_root / "dockerfiles" / "Dockerfile.nvflow").read_text()
    dockerignore = (repo_root / ".dockerignore").read_text()
    assert "cluster_configs/*.yaml" in dockerignore
    for tracked_template in ("containers.yaml", "template-ray.yaml", "template-slurm.yaml"):
        assert f"!cluster_configs/{tracked_template}" in dockerignore
    assert "must be built from a normal Git clone" in client_dockerfile
    assert "uv sync --frozen --no-dev --extra execution --extra ray" in client_dockerfile
    assert "NVFLOW_PROJECT_ENV_CACHE=/opt/nvflow-project-environments" in client_dockerfile
    assert 'ARG NVFLOW_PREWARM_PROJECTS="all"' in client_dockerfile
    assert ".venv/bin/nvflow sync-project --all" in client_dockerfile
    assert "nvflow sync-project --all --check" in client_dockerfile
    assert "scripts/materialize_git_snapshot.py" in client_dockerfile
    assert "tar --null --files-from=/tmp/tracked" in client_dockerfile
    for excluded in (
        ".agents",
        ".codex",
        "tests/golden/recipe_config",
        "tests/test_recipe_config_golden.py",
    ):
        assert f"--exclude {excluded}" in client_dockerfile
    assert "! -name .venv ! -name .baked_commit -exec rm -rf" in client_dockerfile
    assert "ARG PYTHON_VERSION=3.13.14" in client_dockerfile
    assert 'uv python install "${PYTHON_VERSION}"' in client_dockerfile
    assert "COPY --from=builder /opt/python /opt/python" in client_dockerfile
    assert "PYTHON_DEB_VERSION" not in client_dockerfile
    assert "OPENSSH_CLIENT_VERSION" not in client_dockerfile
    assert "GIT_VERSION" not in client_dockerfile
    assert "RSYNC_VERSION" not in client_dockerfile
    assert "CA_CERTIFICATES_VERSION" not in client_dockerfile
    assert "Ray/private aiohttp" not in client_dockerfile
    assert "wandb/bin/wandb-core" not in client_dockerfile
    assert (
        "COPY --from=builder /opt/nvflow-project-environments /opt/nvflow-project-environments"
    ) in client_dockerfile

    skills_dockerfile = (repo_root / "dockerfiles" / "Dockerfile.nemo-skills").read_text()
    assert "ARG NEMO_SKILLS_COMMIT=bcf059af55c20a89f797724598f9908d126153e6" in skills_dockerfile
    assert "ARG PYTHON_VERSION=3.13.14" in skills_dockerfile
    assert "'ray[default]==2.58.0'" in skills_dockerfile
    assert "17ae86b64d7f75653351664f5d8c9e466faede00" in skills_dockerfile
    assert "RUN rm -rf /usr/local /opt/python" in skills_dockerfile
    managed_install = skills_dockerfile.index('uv python install "${PYTHON_VERSION}"')
    unmarked = skills_dockerfile.index("/EXTERNALLY-MANAGED", managed_install)
    assert unmarked < skills_dockerfile.index("python -m ensurepip", managed_install)
    assert "ARG SDP_COMMIT=29b9b1ec0ceaf3ffa441c1d01297371b3f8e11d2" in skills_dockerfile
    assert "pip install git+https://github.com/NVIDIA/NeMo-speech-data-processor" not in (
        skills_dockerfile
    )
    installer, runtime = skills_dockerfile.split("FROM runtime-base AS dependency-installer", 1)[
        1
    ].split("\nFROM runtime-base\n", 1)
    assert "grep -v -x 'numpy==1.26' /tmp/sdp/requirements/main.txt" in installer
    assert "uv pip install --system --no-cache-dir --no-deps /tmp/sdp" in installer
    assert "uv pip install --system --no-cache-dir --torch-backend cpu" in installer
    overrides = skills_dockerfile.split("> /tmp/resolution-overrides.txt", 1)[0].rsplit(
        "RUN printf", 1
    )[1]
    assert "leptonai" not in overrides
    constraints = skills_dockerfile.split("> /tmp/resolution-constraints.txt", 1)[0].rsplit(
        "printf", 1
    )[1]
    assert "'leptonai<0.47'" in constraints
    assert "--constraint /tmp/resolution-constraints.txt" in installer
    assert "apt-get install -y --no-install-recommends g++" in installer
    assert "g++" not in runtime
    assert "ENV RAY_USAGE_STATS_ENABLED=0" in runtime
    assert "assert version('torch').endswith('+cpu')" in runtime
    assert "assert not cuda, cuda" in runtime
    assert "python3.10" not in skills_dockerfile
    for obsolete_override in (
        "wandb-core-builder",
        "WANDB_CORE_COMMIT",
        "starlette>=1.3.1",
        "lxml>=6.1.0",
        "aiohttp>=3.14.3",
        "GitPython>=3.1.58",
        "datamodel-code-generator>=0.64.0",
        "ray_dist.jar",
        "CVE floors",
    ):
        assert obsolete_override not in skills_dockerfile

    rl_dockerfile = (repo_root / "dockerfiles" / "Dockerfile.nemo-rl").read_text()
    assert "ARG NEMO_RL_IMAGE=nvcr.io/nvidia/nemo-rl:v0.8.0" in rl_dockerfile
    assert "FROM ${NEMO_RL_IMAGE}" in rl_dockerfile
    assert 'LABEL org.opencontainers.image.base.name="${NEMO_RL_IMAGE}"' in rl_dockerfile
    assert "apt-get upgrade" not in rl_dockerfile
    assert "ray_dist.jar" not in rl_dockerfile

    container_matrix = (repo_root / "cluster_configs" / "containers.yaml").read_text()
    assert "<YOUR_REGISTRY>/nvflow-nemo-rl:v0.8.0" in container_matrix
    assert "<YOUR_REGISTRY>/nvflow-nemo-gym:v0.7.0" in container_matrix
    assert "nvcr.io/nvidia/nemo-rl" not in container_matrix
    assert "nvcr.io/nvidia/nemo-gym" not in container_matrix
    assert "nvflow-nemo-skills:v1.2.0" in container_matrix
    assert "nvflow-vllm:v0.29.0" in container_matrix
    assert "nvflow-sglang:v0.5.17-ray2.58.0" in container_matrix

    slurm_template = (repo_root / "cluster_configs" / "template-slurm.yaml").read_text()
    ray_template = (repo_root / "cluster_configs" / "template-ray.yaml").read_text()
    for image_identity in (
        "v0.8.0",
        "v0.7.0",
        "v0.29.0",
        "v0.5.17-ray2.58.0",
    ):
        assert image_identity in slurm_template
        assert image_identity in ray_template
    assert "nemo-skills-v1.2.0.sqsh" in slurm_template
    assert "nvflow-nemo-skills:v1.2.0" in ray_template
    assert "nvflow-nemo-rl:v0.8.0" in ray_template
    assert "nvflow-nemo-gym:v0.7.0" in ray_template
    for template in (slurm_template, ray_template):
        assert any(line.strip().startswith("sglang:") for line in template.splitlines())

    scan_images = (repo_root / "ci" / "trivy-images.txt").read_text()
    assert "nvflow-nemo-rl:v0.8.0" in scan_images
    assert "nvflow-nemo-gym:v0.7.0" in scan_images
    assert "nvcr.io/nvidia/" not in scan_images
    assert "nvflow-nemo-skills:v1.2.0" in scan_images
    assert "nvflow-vllm:v0.29.0" in scan_images
    assert "nvflow-sglang:v0.5.17-ray2.58.0" in scan_images

    build_guide = (repo_root / "dockerfiles" / "docker_instructions.md").read_text()
    assert "-f dockerfiles/Dockerfile.nemo-rl" not in build_guide
    assert "-f dockerfiles/Dockerfile.nemo-gym" not in build_guide
    assert "scripts/build_images.sh nemo-skills nemo-rl nemo-gym vllm-0-29-0" in build_guide
    assert "-airgap" not in build_guide

    bake = (repo_root / "docker-bake.hcl").read_text()
    assert "-airgap" not in bake
    gym = bake.split('target "nemo-gym" {', 1)[1].split("\n}", 1)[0]
    assert 'dockerfile = "dockerfiles/Dockerfile.nemo-gym"' in gym
    assert "NEMO_GYM_IMAGE = NEMO_GYM_IMAGE" in gym
    assert '"${REGISTRY}/nvflow-nemo-gym:v0.7.0"' in gym
    rl = bake.split('target "nemo-rl" {', 1)[1].split("\n}", 1)[0]
    assert 'dockerfile = "dockerfiles/Dockerfile.nemo-rl"' in rl
    assert "NEMO_RL_IMAGE = NEMO_RL_IMAGE" in rl
    assert '"${REGISTRY}/nvflow-nemo-rl:v0.8.0"' in rl
    groups = bake.split('target "_multiarch"', 1)[0]
    for group in ('group "v1_2"', 'group "workers"'):
        targets = groups.split(group, 1)[1].split("}", 1)[0]
        assert '"nemo-rl"' in targets
        assert '"nemo-gym"' in targets

    for gemma_config in (
        "gemma3_1b_it.yaml",
        "gemma3_4b_it.yaml",
        "gemma3_27b_it.yaml",
    ):
        config_text = (
            repo_root / "projects" / "finance" / "workflows" / "sft" / gemma_config
        ).read_text()
        assert "automodel_cfg:" in config_text
        assert "dtensor_cfg" not in config_text
        assert "_v2" not in config_text

    vllm_dockerfile = (repo_root / "dockerfiles" / "Dockerfile.vllm").read_text()
    for pin in (
        "ARG CUDA_IMAGE=nvidia/cuda:13.0.3-base-ubuntu24.04@sha256:",
        "ARG VLLM_VERSION=0.29.0",
        "ARG PYTHON_VERSION=3.13.14",
        "ARG RAY_VERSION=2.58.0",
        "ARG CUPY_VERSION=14.0.1",
        "FROM ghcr.io/astral-sh/uv:0.12.9@sha256:",
    ):
        assert pin in vllm_dockerfile
    assert "FROM vllm/vllm-openai" not in vllm_dockerfile
    assert "add-apt-repository" not in vllm_dockerfile
    assert "UV_PYTHON_INSTALL_DIR=/opt/python uv python install" in vllm_dockerfile
    final_stage = vllm_dockerfile.split("\nFROM ${CUDA_IMAGE}", 1)[1]
    shared_layers_end = final_stage.index('uv python install "${PYTHON_VERSION}"')
    for version_arg in (
        "TARGETARCH",
        "VLLM_VERSION",
        "RAY_VERSION",
        "CUPY_VERSION",
        "FLASHINFER_VERSION",
    ):
        assert final_stage.index(f"\nARG {version_arg}\n") > shared_layers_end
    assert "COPY dockerfiles/requirements/vllm-${VLLM_VERSION}-${TARGETARCH}.txt" in vllm_dockerfile
    assert '"https://wheels.vllm.ai/${VLLM_COMMIT}/${wheel}"' in vllm_dockerfile
    assert vllm_dockerfile.count("sha256sum -c -") == 2
    assert '"flashinfer-cubin==${FLASHINFER_VERSION}"' in vllm_dockerfile
    assert '"flashinfer-jit-cache==${FLASHINFER_VERSION}+${cuda_index}"' in vllm_dockerfile
    assert '"nvidia-cutlass-dsl-libs-cu13==${cutlass}"' in vllm_dockerfile
    assert "vllm.__version__ == '${VLLM_VERSION}'" in vllm_dockerfile
    assert "'cupy-cuda12x' not in installed" in vllm_dockerfile
    assert "import cupy" in vllm_dockerfile
    assert "cupy.cuda.runtime.runtimeGetVersion()" in vllm_dockerfile
    assert "--keep-git-dir" not in vllm_dockerfile
    assert "ray.dashboard.optional_deps" in vllm_dockerfile
    assert "ray.experimental.compiled_dag_ref" in vllm_dockerfile
    assert "RAY_USAGE_STATS_ENABLED=0" in vllm_dockerfile
    assert "platform.python_version() == '${PYTHON_VERSION}'" in vllm_dockerfile
    assert 'ENTRYPOINT ["vllm", "serve"]' in vllm_dockerfile
    for unexpected_override in (
        "VLLM_V1_USE_OUTLINES_CACHE",
        "aiohttp",
        "cryptography",
        "httpx",
        "mcp==",
        "pillow",
        "pyasn1",
        "python-multipart",
        "qwen-vl-utils",
        "ray_dist.jar",
        "Security hardening",
        "starlette",
        "vllm[audio]",
    ):
        assert unexpected_override not in vllm_dockerfile

    assert not (repo_root / "scripts" / "build_vllm_image.sh").exists()
    bake = (repo_root / "docker-bake.hcl").read_text()

    def bake_target_settings(name: str) -> dict[str, str]:
        body = bake.split(f'target "{name}" {{', 1)[1].split("\n}", 1)[0]
        return {
            key.strip(): value.strip()
            for key, value in (line.split("=", 1) for line in body.splitlines() if "=" in line)
        }

    requirements = repo_root / "dockerfiles" / "requirements"
    for version, commit, target_suffix, flashinfer in (
        ("0.29.0", "98dff2a81d747d1dba01a47f939f48c3526d4206", "0-29-0", "0.6.18"),
    ):
        assert commit in bake
        assert f'"${{REGISTRY}}/nvflow-vllm:v{version}"' in bake
        target = bake_target_settings(f"vllm-{target_suffix}")
        assert target["dockerfile"] == '"dockerfiles/Dockerfile.vllm"'
        assert target["VLLM_VERSION"] == f'"{version}"'
        assert target["VLLM_COMMIT"] == f"VLLM_{version.replace('.', '')}_COMMIT"
        assert target["FLASHINFER_VERSION"] == f'"{flashinfer}"'
        for arch in ("AMD64", "ARM64"):
            digest = target[f"VLLM_WHEEL_SHA256_{arch}"].strip('"')
            assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")
        lock_input = (requirements / f"vllm-{version}.in").read_text()
        assert f"https://wheels.vllm.ai/{commit}/vllm-{version}-" in lock_input
        for arch in ("amd64", "arm64"):
            pins = set((requirements / f"vllm-{version}-{arch}.txt").read_text().splitlines())
            assert "--extra-index-url https://download.pytorch.org/whl/cu130" in pins
            assert any(line.startswith("#    uv pip compile --no-config ") for line in pins)
            assert {
                "ray==2.58.0",
                "cupy-cuda13x==14.0.1",
                "torch==2.13.0+cu130",
                "nvidia-nccl-cu13==2.30.7",
                f"flashinfer-python=={flashinfer}",
            } <= pins
            assert not any(line.startswith(("vllm==", "vllm @", "cupy-cuda12x")) for line in pins)
    assert not (repo_root / "dockerfiles" / "Dockerfile.vllm-rust").exists()
    assert "vllm-upstream" not in bake
    assert 'platforms = ["linux/amd64", "linux/arm64"]' in bake
    assert "provenance =" not in bake
    assert "sbom =" not in bake
    assert "--provenance=false --sbom=false" in build_guide


def test_sglang_image_uses_official_release_on_nvflow_python() -> None:
    repo_root = Path(__file__).parents[1]
    dockerfile = (repo_root / "dockerfiles" / "Dockerfile.sglang").read_text()
    for pin in (
        "ARG CUDA_IMAGE=nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04@sha256:",
        "ARG SGLANG_VERSION=0.5.17",
        "ARG PYTHON_VERSION=3.13.14",
        "ARG RAY_VERSION=2.58.0",
        "ARG OUTLINES_CORE_VERSION=0.1.26",
        "ARG FLASHINFER_VERSION=0.6.15.post1",
        "ARG RUST_TOOLCHAIN=1.98.1",
        "FROM ghcr.io/astral-sh/uv:0.12.9@sha256:",
    ):
        assert pin in dockerfile
    assert "sglang[all]" not in dockerfile
    assert "lmsysorg/sglang" not in dockerfile
    assert "libnccl" not in dockerfile
    assert "COPY dockerfiles/requirements/sglang-${SGLANG_VERSION}-${TARGETARCH}.txt" in dockerfile
    assert "--no-binary outlines-core" in dockerfile
    assert "--find-links /tmp/wheels -r /tmp/requirements.txt" in dockerfile
    assert "EXTERNALLY-MANAGED" in dockerfile
    assert "RAY_USAGE_STATS_ENABLED=0" in dockerfile
    requirements = repo_root / "dockerfiles" / "requirements"
    lock_input = (requirements / "sglang-0.5.17.in").read_text()
    assert "sglang[http2,tracing]==0.5.17" in lock_input
    assert "cuda-python==13.2.0" in lock_input
    for arch in ("amd64", "arm64"):
        pins = set((requirements / f"sglang-0.5.17-{arch}.txt").read_text().splitlines())
        assert "--extra-index-url https://download.pytorch.org/whl/cu130" in pins
        assert any(line.startswith("#    uv pip compile --no-config ") for line in pins)
        assert {
            "sglang==0.5.17",
            "outlines-core==0.1.26",
            "ray==2.58.0",
            "torch==2.11.0+cu130",
        } <= pins
    assert dockerfile.count("sha256sum -c -") == 2
    for gate in (
        "platform.python_version() == '${PYTHON_VERSION}'",
        "ray.__version__ == '${RAY_VERSION}'",
        "sglang.__version__ == '${SGLANG_VERSION}'",
        "version('outlines-core') == '${OUTLINES_CORE_VERSION}'",
        "torch.version.cuda.startswith('13.')",
        "load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)",
        "pip check",
    ):
        assert gate in dockerfile

    bake = (repo_root / "docker-bake.hcl").read_text()
    target = bake.split('target "sglang-0-5-17" {', 1)[1].split("\n}", 1)[0]
    assert 'dockerfile = "dockerfiles/Dockerfile.sglang"' in target
    assert '"${REGISTRY}/nvflow-sglang:v0.5.17-ray2.58.0"' in target
    assert 'SGLANG_VERSION = "0.5.17"' in target
    build_guide = (repo_root / "dockerfiles" / "docker_instructions.md").read_text()
    assert "`nvflow-sglang:v0.5.17-ray2.58.0`" in build_guide


def test_gpu_image_pip_check_ignores_only_known_reports(tmp_path: Path) -> None:
    repo_root = Path(__file__).parents[1]
    cusparselt = "nvidia-cusparselt-cu13 0.8.0 is not supported on this platform"
    nccl = (
        "torch 2.13.0+cu130 has requirement nvidia-nccl-cu13==2.29.7; "
        'platform_system == "Linux", but you have nvidia-nccl-cu13 2.30.7.'
    )
    conflict = "foo 1.0 has requirement bar>=2, but you have bar 1.0."
    for dockerfile, command, known in (
        ("Dockerfile.vllm", "python -m pip check", [cusparselt, nccl]),
        ("Dockerfile.sglang", "pip check", [cusparselt]),
    ):
        reports: list[tuple[list[str], bool]] = [
            ([], True),
            (known, True),
            ([*known, conflict], False),
            ([conflict], False),
        ]
        if nccl not in known:
            reports.append(([nccl], False))
        text = (repo_root / "dockerfiles" / dockerfile).read_text()
        block = []
        for line in text[text.index(f"RUN {command} > /tmp/pipcheck.txt") :].splitlines():
            block.append(line)
            if not line.endswith("\\"):
                break
        script = "\n".join(block).removeprefix("RUN ")
        script = script.replace("/tmp/pipcheck.txt", str(tmp_path / "pipcheck.txt"))
        for report, passes in reports:
            output = "".join(f"{line}\n" for line in report)
            fake = f"{{ printf %s {shlex.quote(output)}; [ {len(report)} -eq 0 ]; }}"
            result = subprocess.run(
                ["sh", "-c", script.replace(command, fake, 1)], capture_output=True, text=True
            )
            assert (result.returncode == 0) is passes, (dockerfile, report, result.stdout)


def test_nemo_rl_required_config_contracts() -> None:
    repo_root = Path(__file__).parents[1]
    workflow_root = repo_root / "projects" / "finance" / "workflows"
    for path in workflow_root.rglob("*.yaml"):
        assert "dtensor_cfg" not in path.read_text(), path

    grpo = OmegaConf.load(workflow_root / "grpo" / "grpo_presets.yaml")
    grpo_policy = grpo.presets["grpo-base"].policy
    generation = grpo_policy.generation
    assert generation.val_temperature == generation.temperature
    assert generation.val_top_p == generation.top_p
    assert generation.val_top_k == generation.top_k
    assert grpo_policy.automodel_cfg.checkpoint == {
        "model_save_format": "safetensors",
        "save_consolidated": "false",
    }

    sft = OmegaConf.load(workflow_root / "sft" / "sft_presets.yaml")
    sft_policy = sft.presets["sft-base"].policy
    assert sft_policy.automodel_cfg.checkpoint == {
        "model_save_format": "safetensors",
        "save_consolidated": "false",
    }
    for policy in (grpo_policy, sft_policy):
        assert policy.offload_policy_before_refit is False
        assert policy.offload_optimizer_for_refit is True


def test_bundled_environment_policy_matches_root_uv_policy() -> None:
    repo_root = Path(__file__).parents[1]
    root = tomllib.loads((repo_root / "pyproject.toml").read_text())
    policy_root = repo_root / "nvflow" / "project" / "_environment_policy"

    execution = [
        line
        for line in (policy_root / "execution.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    overrides = [
        line
        for line in (policy_root / "overrides.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    constraints = [
        line
        for line in (policy_root / "constraints.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ]

    assert root["tool"]["uv"]["required-version"] == (
        f">={project_environment._MINIMUM_UV_VERSION}"
    )
    assert {Requirement(value) for value in overrides} == {
        Requirement(value) for value in root["tool"]["uv"]["override-dependencies"]
    }
    assert {Requirement(value) for value in constraints} == {
        Requirement(value) for value in root["tool"]["uv"].get("constraint-dependencies", [])
    }
    assert {Requirement(value) for value in execution} == {
        Requirement(value) for value in root["project"]["optional-dependencies"]["execution"]
    }
    for name in ("execution.txt", "overrides.txt", "constraints.txt"):
        assert (policy_root / name).read_text().startswith("# Generated from")
    subprocess.run(
        [sys.executable, "scripts/check_project_environment_policy.py"],
        cwd=repo_root,
        check=True,
    )


def test_root_lock_excludes_project_packages() -> None:
    repo_root = Path(__file__).parents[1]
    lock = tomllib.loads((repo_root / "uv.lock").read_text())
    locked = {package["name"] for package in lock["package"]}

    assert not {"nvflow-project-finance", "nvflow-project-hopchain", "nvflow-project-starter"} & (
        locked
    )

    subprocess.run(["uv", "lock", "--check"], cwd=repo_root, check=True)
    subprocess.run(
        ["uv", "sync", "--locked", "--dry-run"],
        cwd=repo_root,
        check=True,
    )
