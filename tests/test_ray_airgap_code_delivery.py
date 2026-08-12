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

"""Airgap contract for delivering NVFlow code to Ray Jobs.

The release path must package the committed NVFlow tree in ``nvflow-client``
and let Ray extract it as the job cwd. A live checkout mount, source
``PYTHONPATH`` overlay, or runtime dependency install invalidates that path.
"""

from __future__ import annotations

import base64
import io
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import yaml
from omegaconf import OmegaConf

from nvflow.recipes.finance.stages.rl.training import GRPOStage

REPO_ROOT = Path(__file__).resolve().parents[1]
RAY_CODE_ARCHIVE = "/opt/nvflow-ray-code.zip"
NEMO_SKILLS_COMMIT = "1fd0a9f34a5f213a1d0990f0d587fadabb80ba45"


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_client_dockerfile_builds_tracked_only_ray_archive():
    dockerfile = _read("dockerfiles/Dockerfile.nvflow")

    assert f"git archive --format=zip --output={RAY_CODE_ARCHIVE} HEAD" in dockerfile
    assert f"COPY --from=builder {RAY_CODE_ARCHIVE} {RAY_CODE_ARCHIVE}" in dockerfile
    assert f"NVFLOW_RAY_WORKING_DIR={RAY_CODE_ARCHIVE}" in dockerfile
    assert 'assert "nvflow/__init__.py" in names' in dockerfile
    assert 'assert "nvflow/_version.py" in names' in dockerfile
    assert "bundle.writestr(version_info, version_file.read_bytes())" in dockerfile
    assert 'assert b"__version__" in bundle.read("nvflow/_version.py")' in dockerfile
    assert 'assert not any(name.startswith((".git/", ".venv/"))' in dockerfile
    assert dockerfile.count("python${PYTHON_VERSION} - <<'PY'") == 2
    assert "python3 - <<'PY'" not in dockerfile
    assert "COPY --from=builder /opt/nvflow /opt/nvflow" not in dockerfile
    assert "materialized == archived" in dockerfile
    assert "git diff --quiet" in dockerfile
    assert "git diff --cached --quiet" in dockerfile
    assert "git rev-parse 'HEAD^{tree}' > /tmp/original-tree" in dockerfile
    assert 'test "$(git rev-parse \'HEAD^{tree}\')" = "$(cat /tmp/original-tree)"' in dockerfile
    assert "COPY --from=builder /opt/nvflow/.venv /opt/nvflow/.venv" in dockerfile
    assert "COPY --from=builder /opt/nvflow/.git /opt/nvflow/.git" in dockerfile
    assert "COPY --from=builder /opt/nvflow/.baked_commit /opt/nvflow/.baked_commit" in dockerfile


def test_docker_context_excludes_local_bundle_artifacts():
    dockerignore = _read(".dockerignore").splitlines()

    assert "artifacts/" in dockerignore
    assert "*.bundle" in dockerignore


def test_skills_image_pins_stable_roots_and_bakes_complete_asset_source():
    dockerfile = _read("dockerfiles/Dockerfile.nemo-skills")
    project = _read("pyproject.toml")
    lock = _read("uv.lock")

    assert f"ARG NEMO_SKILLS_COMMIT={NEMO_SKILLS_COMMIT}" in dockerfile
    assert f"NeMo-Skills.git@{NEMO_SKILLS_COMMIT}" in project
    assert f"rev={NEMO_SKILLS_COMMIT}#{NEMO_SKILLS_COMMIT}" in lock
    for asset in (
        "nemo_skills/prompt/config/generic/default.yaml",
        "nemo_skills/dataset/aime24/test.txt",
        "nemo_skills/pipeline/nemo_rl/ray_templates/nemo_skills_sandbox_ray.sub.j2",
        "nemo_skills/mcp/servers/exclude_domains_hle_opus.json",
    ):
        assert f"test -s /opt/NeMo-Skills/{asset}" in dockerfile


def test_git_archive_is_exactly_the_committed_file_set():
    """Exercise the same git-archive primitive used by the image build."""

    archive = subprocess.run(
        ["git", "archive", "--format=zip", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    tracked = set(
        subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.splitlines()
    )

    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        archived_files = {name for name in bundle.namelist() if not name.endswith("/")}
        executable_modes = {
            name: bundle.getinfo(name).external_attr >> 16
            for name in archived_files
            if name.startswith("scripts/")
        }

    assert archived_files == tracked
    assert "nvflow/__init__.py" in archived_files
    assert "nvflow/cli/main.py" in archived_files
    assert "scripts/run_flow.py" in archived_files
    assert executable_modes["scripts/run_flow.py"] & 0o111
    assert "nvflow/recipes/finance/prompts/sec_judge.yaml" in archived_files
    assert not any(path.startswith((".git/", ".venv/")) for path in archived_files)


def test_ray_template_uses_archive_without_source_overlay():
    template_text = _read("cluster_configs/template-ray.yaml")
    template = yaml.safe_load(template_text)

    assert template["executor"] == "none"
    assert template["backend"]["name"] == "ray"
    assert template["backend"]["working_dir"] == RAY_CODE_ARCHIVE
    assert all(not str(item).startswith("PYTHONPATH=") for item in template["env_vars"])
    assert all(not str(item).endswith(":/workspace") for item in template["mounts"])

    # Slurm must not pay for or inherit Ray-only code delivery.
    slurm_template = yaml.safe_load(_read("cluster_configs/template-slurm.yaml"))
    assert slurm_template["executor"] == "slurm"
    assert "working_dir" not in slurm_template
    assert "backend" not in slurm_template or "working_dir" not in slurm_template["backend"]


def test_ray_docs_do_not_require_live_source_overlay():
    docs = "\n".join(
        [
            _read("INSTALL-RAY.md"),
            _read("docs/recipes/finance/quick-start-ray.md"),
            _read("cluster_configs/template-ray.yaml"),
        ]
    )

    assert "PYTHONPATH=/workspace" not in docs
    assert "<CLUSTER_PATH_TO_NVFLOW_REPO>:/workspace" not in docs
    assert "backend.working_dir" in docs
    assert RAY_CODE_ARCHIVE in docs
    assert "NEMO_RUN_CODE_DIR" in docs
    assert "NEMO_SKILLS_CONFIG_DIR" in docs
    assert "uv run --no-sync nflow" in docs


def test_ray_recipe_overlays_use_delivered_tracked_prompts():
    overlays = [
        "nvflow/recipes/finance/workflows/eval/demo_ray.yaml",
        "nvflow/recipes/finance/workflows/sdg/template-based-sdg-demo_ray.yaml",
        "nvflow/recipes/finance/workflows/sft/qwen3_4b_smoke_ray.yaml",
    ]

    for relative_path in overlays:
        text = _read(relative_path)
        assert "/lustre/<you>/nvflow/recipes/finance/prompts" not in text
        assert "nvflow/recipes/finance/prompts/" in text


def test_grpo_base64_config_rewrites_code_paths_only_for_ray(monkeypatch):
    """Opaque config payloads need their own Ray-only stable-root rewrite."""

    legacy_path = (
        "/nemo_run/code/nvflow/recipes/finance/workflows/grpo/overlays/finance_sec_search_env.yaml"
    )
    prepared = SimpleNamespace(
        output_dir="/lustre/output",
        nemo_rl_config={"env": {"nemo_gym": {"config_paths": [legacy_path]}}},
    )
    stage = GRPOStage.__new__(GRPOStage)

    def decoded(snippet: str) -> dict:
        encoded = snippet.split("echo ", 1)[1].split(" | base64 -d", 1)[0]
        return yaml.safe_load(base64.b64decode(encoded))

    slurm_snippet, _ = stage._config_shell_snippet(prepared)
    ray_snippet, _ = stage._config_shell_snippet(prepared, is_ray=True)
    slurm_config = decoded(slurm_snippet)
    ray_config = decoded(ray_snippet)

    assert slurm_config["env"]["nemo_gym"]["config_paths"] == [legacy_path]
    assert ray_config["env"]["nemo_gym"]["config_paths"] == [
        "${oc.env:NEMO_RUN_CODE_DIR}"
        "/nvflow/recipes/finance/workflows/grpo/overlays/finance_sec_search_env.yaml"
    ]
    monkeypatch.setenv("NEMO_RUN_CODE_DIR", "/tmp/ray-working-dir")
    resolved_ray_config = OmegaConf.to_container(OmegaConf.create(ray_config), resolve=True)
    assert resolved_ray_config["env"]["nemo_gym"]["config_paths"] == [
        "/tmp/ray-working-dir"
        "/nvflow/recipes/finance/workflows/grpo/overlays/finance_sec_search_env.yaml"
    ]
    assert "/nemo_run/code" not in ray_snippet
    assert prepared.nemo_rl_config["env"]["nemo_gym"]["config_paths"] == [legacy_path]


def test_gym_uploaded_overlays_resolve_code_root_for_slurm_and_ray(monkeypatch):
    """Uploaded YAML is not covered by Ray's command-string path rewrite.

    These prompt paths are resolved by Gym after the task changes cwd to
    ``/opt/Gym``. The env interpolation must retain Slurm's legacy path when
    unset and use the extracted Ray working directory when set.
    """

    openqa = REPO_ROOT / "nvflow/recipes/finance/prompts/finance_openqa_judge_overlay.yaml"
    search = (
        REPO_ROOT / "nvflow/recipes/finance/workflows/grpo/overlays/finance_sec_search_env.yaml"
    )

    def resolved_paths() -> tuple[str, str, str]:
        openqa_cfg = OmegaConf.load(openqa)
        search_cfg = OmegaConf.load(search)
        search_server = (
            search_cfg.finance_sec_search_resources_server.resources_servers.finance_sec_search
        )
        return (
            openqa_cfg.equivalence_llm_judge.resources_servers.equivalence_llm_judge.judge_prompt_template_fpath,
            search_server.judge_prompt_template_fpath,
            search_server.retrieval_system_prompt_fpath,
        )

    suffixes = (
        "/nvflow/recipes/finance/prompts/finance_openqa_judge.txt",
        "/nvflow/recipes/finance/prompts/finance_sec_search_judge.yaml",
        "/nvflow/recipes/finance/prompts/finance_sec_search_retrieval.yaml",
    )

    monkeypatch.delenv("NEMO_RUN_CODE_DIR", raising=False)
    assert resolved_paths() == tuple(f"/nemo_run/code{suffix}" for suffix in suffixes)

    monkeypatch.setenv("NEMO_RUN_CODE_DIR", "/tmp/ray-working-dir")
    assert resolved_paths() == tuple(f"/tmp/ray-working-dir{suffix}" for suffix in suffixes)


def test_gym_uploaded_overlays_have_no_unwrapped_legacy_code_paths():
    """Prevent future post-cd Gym paths from bypassing the stable Ray root."""

    overlays = (
        "nvflow/recipes/finance/prompts/finance_openqa_judge_overlay.yaml",
        "nvflow/recipes/finance/workflows/grpo/overlays/finance_sec_search_env.yaml",
    )

    def strings(value):
        if isinstance(value, dict):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
        elif isinstance(value, str):
            yield value

    for relative_path in overlays:
        values = strings(yaml.safe_load(_read(relative_path)))
        for value in values:
            if "/nemo_run/code" in value:
                assert "${oc.env:NEMO_RUN_CODE_DIR,/nemo_run/code}" in value
