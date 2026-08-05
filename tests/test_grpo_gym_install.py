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
"""Regression guard for the GRPO Gym installation_command dispatcher.

All GRPO Gym stages -- ``prepare_data``, ``prefetch_cache``,
``collect_rollouts``, ``compute_rewards`` -- run on the CPU **nemo-gym**
container and share the ``&gym_install_cpu`` dispatcher. The nemo-gym image
bakes the Gym CLI (``/opt/gym-cli-venv``) and per-component venvs
(``/opt/gym-venvs``), so the command only exposes the baked CLI entry points on
PATH: no venv build, no network, no ``.venv`` activation.

(The old nemo-rl ``&gym_install`` dispatcher -- baked-``.venv`` source / mounted
dev ``/tmp`` build -- was removed once the Gym rollout/verify clients moved to
nemo-gym. Only ``training`` still uses the nemo-rl image, with
``installation_command: "true"``.)

The ``collect_rollouts`` / ``compute_rewards`` env-start path additionally needs
``gym_uv_venv_dir: /opt/gym-venvs`` so ``ng_run`` finds the baked component
venvs (Gym otherwise defaults to ``/opt/Gym/<component>/.venv``). The
``+uv_venv_dir=$UV_VENV_DIR`` wiring itself is snapshot-tested in
``tests/test_rollout.py`` / ``tests/test_verify.py``; here we guard the config.

OmegaConf-load coverage stays: bare ``${VAR}`` would raise
``UnsupportedInterpolationType`` at load, and the resolved command must pass
``bash -n`` both bare and inside the nemo-skills ``echo 'Installing packages: …'``
wrapper.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
GRPO_BASE_YAML = REPO_ROOT / "nvflow" / "recipes" / "finance" / "workflows" / "grpo" / "base.yaml"

# All Gym stages share &gym_install_cpu.
GYM_STAGES = ("prepare_data", "prefetch_cache", "collect_rollouts", "compute_rewards")
# Env-start stages that must point ng_run at the baked component venvs.
ENV_START_STAGES = ("collect_rollouts", "compute_rewards")
CPU_CLI_BIN = "/opt/gym-cli-venv/bin"
BAKED_VENV_DIR = "/opt/gym-venvs"


def _load_dispatcher(stage: str) -> str:
    """Load a stage's installation_command via OmegaConf (catches ${VAR} regressions)."""
    cfg = OmegaConf.load(GRPO_BASE_YAML)
    cmd = cfg.stages[stage].installation_command
    return cmd if isinstance(cmd, str) else str(cmd)


def _run(cmd: str, env: dict[str, str] | None = None):
    base_env = {"PATH": "/usr/bin:/bin"}
    if env:
        base_env.update(env)
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, env=base_env)


# ---------------------------------------------------------------------------
# Dispatcher: single &gym_install_cpu shared by all Gym stages
# ---------------------------------------------------------------------------


def test_dispatcher_loads_via_omegaconf():
    """Touching the dispatcher forces OmegaConf to resolve interpolations.

    A regression to a bare ``${VAR}`` would raise ``UnsupportedInterpolationType``
    here. We avoid ``to_container(resolve=True)`` so unrelated mandatory
    interpolations (set only in per-model YAMLs) don't fail the load.
    """
    cfg = OmegaConf.load(GRPO_BASE_YAML)
    cmd = cfg.stages.prepare_data.installation_command
    assert isinstance(cmd, str)
    assert "gym-cli-venv" in cmd


@pytest.mark.parametrize("stage", GYM_STAGES)
def test_all_gym_stages_share_gym_install_cpu(stage: str):
    """Every Gym stage must resolve to the same &gym_install_cpu body."""
    assert _load_dispatcher(stage) == _load_dispatcher("prepare_data"), (
        f"stage={stage} drifted from &gym_install_cpu anchor"
    )


def test_exposes_baked_cli_on_path_no_venv_build():
    """The dispatcher only prepends the baked Gym CLI bin to PATH.

    nemo-gym bakes the CLI (/opt/gym-cli-venv) + per-component venvs
    (/opt/gym-venvs), so the command must NOT source a ``.venv`` or run uv --
    it just exposes the entry points (ng_prepare_data, ng_run, …).
    """
    cmd = _load_dispatcher("prepare_data")
    assert f"export PATH={CPU_CLI_BIN}" in cmd
    assert "uv sync" not in cmd
    assert "uv venv" not in cmd
    assert "activate" not in cmd


def test_dispatcher_passes_bash_syntax_and_wrapper():
    """Dispatcher passes ``bash -n`` bare and inside the nemo-skills echo wrapper.

    The wrapper (``echo 'Installing packages: <cmd>'``) catches single-quote
    breakage; ``bash -n`` catches YAML block-scalar formatting regressions.
    """
    cmd = _load_dispatcher("prepare_data")
    assert subprocess.run(["bash", "-n", "-c", cmd], capture_output=True, text=True).returncode == 0
    wrapped = f"echo 'Installing packages: {cmd}'; if {cmd}; then echo ok; else echo fail; fi"
    r = subprocess.run(["bash", "-n", "-c", wrapped], capture_output=True, text=True)
    assert r.returncode == 0, f"wrapped dispatcher failed bash -n:\n{r.stderr}"


def test_dispatcher_runs_clean_and_emits_marker():
    """Dispatcher is a pure PATH prepend -- succeeds even without /opt present,
    and emits the ``[gym_install] nemo-gym`` grep anchor used by the runbook."""
    cmd = _load_dispatcher("prepare_data")
    r = _run(cmd)
    assert r.returncode == 0, f"dispatcher must succeed; stderr={r.stderr}"
    assert "[gym_install] nemo-gym" in r.stdout


# ---------------------------------------------------------------------------
# Env-start stages must reuse the baked per-component venvs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", ENV_START_STAGES)
def test_env_start_stages_point_ng_run_at_baked_venvs(stage: str):
    """collect_rollouts / compute_rewards run ``gym env start`` (``ng_run``), so
    they must set ``gym_uv_venv_dir=/opt/gym-venvs``. Without it, Gym defaults
    ``uv_venv_dir`` to ``/opt/Gym/<component>/.venv`` and would rebuild the
    (baked) venvs on every job -- slow online, fatal air-gapped."""
    cfg = OmegaConf.load(GRPO_BASE_YAML)
    st = cfg.stages[stage]
    assert st.get("container") == "nemo-gym"
    assert st.get("gym_path") == "/opt/Gym"
    assert st.get("gym_uv_venv_dir") == BAKED_VENV_DIR, (
        f"{stage} must set gym_uv_venv_dir={BAKED_VENV_DIR} so ng_run reuses baked venvs"
    )


def test_pure_cpu_stages_do_not_need_uv_venv_dir():
    """prepare_data / prefetch_cache run ``ng_prepare_data`` in the CLI venv (no
    component servers), so they don't set gym_uv_venv_dir -- guard that they stay
    on nemo-gym CPU regardless."""
    cfg = OmegaConf.load(GRPO_BASE_YAML)
    for stage in ("prepare_data", "prefetch_cache"):
        st = cfg.stages[stage]
        assert st.get("container") == "nemo-gym"
        assert int(st.get("num_gpus", -1)) == 0
