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
"""Shared helpers for RL rollout and reward orchestration.

Recipe-agnostic utilities used by ``nvflow.lib.rl.rollout`` and
``nvflow.lib.rl.verify``.

General utilities:
  - ``resolve_host_path``: maps /workspace/ container paths to host paths.
  - ``check_launcher_cwd``: preflight check -- fails fast if cwd is not the
    nvflow project root (required by ``resolve_host_path``).
  - ``build_config_paths_str``: assembles NeMo-Gym config_paths with overlay.

vLLM server configuration:
  - ``build_vllm_server_args``: converts vLLM YAML config to ``--key value``
    CLI args suitable for ``nemo_skills.pipeline.utils.scripts.ServerScript(server_args=...)``.
  - ``compute_num_gpus``: auto-compute Slurm GPU request from per-endpoint num_gpus.
  - ``_overlay_path`` / ``_build_overlay_setup_cmd``: content-addressed model
    overlay directories for per-environment HF config overrides (e.g. YaRN).
    The overlay is created inside the Slurm job via :mod:`nvflow.lib.rl.create_overlay`.

Judge configuration:
  - ``determine_judge_mode``, ``validate_judge_config``: mode detection & validation.
  - ``build_judge_ng_run_overrides``: ng_run CLI overrides for the judge
    (for collect_rollouts / compute_rewards -- shell command strings).
  - ``build_judge_nemo_gym_config``: dict fragment for NeMo-Gym
    ``initial_global_config_dict`` (for GRPO training -- in-memory config).
  - ``log_judge_details``: console output for judge config.

Shell / script templates:
  - ``SHELL_WAIT_FOR_SERVER``: reusable bash function for health-check polling.
  - ``CONTAINER_WORKSPACE``: mount point for the nvflow project root inside
    Slurm containers (used for ``PYTHONPATH`` in inline bash scripts).
"""

import hashlib
import json
import shlex
from pathlib import Path
from typing import Any

# ============================================================================
# General utilities
# ============================================================================


CONTAINER_WORKSPACE = "/workspace"
"""Mount point for the nvflow project root inside Slurm containers."""

CONTAINER_CODE_DIR = "/nemo_run/code"
"""Snapshot of the nvflow codebase inside Slurm containers (nemo-run packager)."""

_WORKSPACE_PREFIX = CONTAINER_WORKSPACE + "/"


def resolve_host_path(container_path: str) -> Path:
    """Map a ``/workspace/`` container path to the host filesystem.

    The Slurm container mounts the nvflow project root at
    :data:`CONTAINER_WORKSPACE` (``/workspace``).  On the submission host
    that path doesn't exist, so the prefix is replaced with ``./`` which
    resolves to the same Lustre directory -- provided the launcher is run
    from the nvflow project root.

    **Launcher requirement**: the process calling this function must be
    running on a host that has Lustre mounted and the cwd must be the
    nvflow project root (``uv run nflow ...`` enforces this by default).
    Call :func:`check_launcher_cwd` early to fail fast with a clear
    message if this assumption is violated.
    """
    if container_path.startswith(_WORKSPACE_PREFIX) and not Path(CONTAINER_WORKSPACE).exists():
        return Path(container_path.replace(_WORKSPACE_PREFIX, "./"))
    return Path(container_path)


def check_launcher_cwd() -> None:
    """Validate that the launcher is running from the nvflow project root.

    :func:`resolve_host_path` maps container paths to ``./`` relative
    paths, which only works when the cwd is the nvflow project root on a
    Lustre-mounted host.  This function checks for the ``pyproject.toml``
    marker file and raises early with a clear message if it's missing.
    """
    marker = Path("pyproject.toml")
    if not marker.exists():
        raise RuntimeError(
            f"Launcher must run from the nvflow project root "
            f"(expected '{marker}' in cwd={Path.cwd()}). "
            f"Use 'uv run nflow ...' which enforces this automatically."
        )


def launcher_is_remote(cluster_config: dict) -> bool:
    """True when launching off-cluster via an ``ssh_tunnel`` cluster config.

    In that mode the launch host has no local Lustre mount, so resume/status
    filesystem checks must run on the cluster over SSH (see :class:`LauncherFS`).
    When ``ssh_tunnel`` is absent, ``nemo_run``/``nemo_skills`` assume the
    launcher already runs on the cluster, and local ``Path`` ops are used --
    identical to the historical behaviour.
    """
    return "ssh_tunnel" in (cluster_config or {})


class LauncherFS:
    """Filesystem probes for the launcher that work both on- and off-cluster.

    - Within the cluster (no ``ssh_tunnel``): uses local :class:`pathlib.Path`
      operations via :func:`resolve_host_path` -- byte-for-byte the historical
      behaviour, so there is no regression for the common case.
    - Off-cluster (``ssh_tunnel`` set): runs ``test``/``rm``/``ls`` on the
      cluster over the tunnel (``get_tunnel`` + ``get_unmounted_path``), the
      same mechanism ``nemo_skills.get_remaining_jobs`` uses, so the launcher
      needs no local Lustre mount.

    All paths passed in are *container* paths (e.g. ``/workspace/...``); the
    remote backend maps them to the real cluster path automatically.
    """

    def __init__(self, cluster_config: dict):
        self.cluster_config = cluster_config
        self.remote = launcher_is_remote(cluster_config)

    # -- remote helpers -----------------------------------------------------
    def _tunnel(self):
        from nemo_skills.pipeline.utils.cluster import get_tunnel

        return get_tunnel(self.cluster_config)

    def _unmounted(self, container_path: str) -> str:
        from nemo_skills.pipeline.utils import get_unmounted_path

        return str(get_unmounted_path(self.cluster_config, str(container_path)))

    # -- public API (container paths in) -----------------------------------
    def exists(self, container_path: str) -> bool:
        if not self.remote:
            return resolve_host_path(str(container_path)).exists()
        return self.batch_exists([str(container_path)]).get(str(container_path), False)

    def batch_exists(self, container_paths: list[str]) -> dict[str, bool]:
        paths = [str(p) for p in container_paths]
        if not paths:
            return {}
        if not self.remote:
            return {p: resolve_host_path(p).exists() for p in paths}
        result: dict[str, bool] = {}
        tunnel = self._tunnel()
        batch = 40
        for i in range(0, len(paths), batch):
            group = paths[i : i + batch]
            lines = []
            for idx, cp in enumerate(group):
                hp = shlex.quote(self._unmounted(cp))
                lines.append(f'if [ -e {hp} ]; then echo "{idx}:Y"; else echo "{idx}:N"; fi')
            out = tunnel.run("; ".join(lines), hide=True, warn=True).stdout
            parsed: dict[int, bool] = {}
            for line in out.splitlines():
                line = line.strip()
                key, sep, val = line.partition(":")
                if sep and key.isdigit():
                    parsed[int(key)] = val.strip().endswith("Y")
            for idx, cp in enumerate(group):
                result[cp] = parsed.get(idx, False)
        return result

    def rm(self, container_paths: list[str]) -> None:
        paths = [str(p) for p in container_paths]
        if not paths:
            return
        if not self.remote:
            for p in paths:
                resolve_host_path(p).unlink(missing_ok=True)
            return
        tunnel = self._tunnel()
        batch = 40
        for i in range(0, len(paths), batch):
            group = paths[i : i + batch]
            quoted = " ".join(shlex.quote(self._unmounted(p)) for p in group)
            tunnel.run(f"rm -f {quoted}", hide=True, warn=True)

    def ls(self, container_dir: str, pattern: str) -> list[str]:
        """Return sorted basenames matching ``pattern`` directly under ``container_dir``."""
        if not self.remote:
            d = resolve_host_path(str(container_dir))
            return sorted(p.name for p in d.glob(pattern)) if d.exists() else []
        hp = shlex.quote(self._unmounted(str(container_dir)))
        # pattern is a controlled literal glob (e.g. output-rs*.jsonl); leave it
        # unquoted so the remote shell expands it.
        out = (
            self._tunnel()
            .run(
                f"cd {hp} 2>/dev/null && ls -1 {pattern} 2>/dev/null || true", hide=True, warn=True
            )
            .stdout
        )
        return sorted(ln.strip() for ln in out.splitlines() if ln.strip())

    def count_lines(self, container_path: str) -> int:
        """Line count of a file (local only; callers skip this when remote)."""
        hp = resolve_host_path(str(container_path))
        if not hp.exists():
            return 0
        with open(hp) as f:
            return sum(1 for _ in f)


VLLM_MODEL = "responses_api_models/vllm_model/configs/vllm_model.yaml"
VLLM_MODEL_FOR_TRAINING = "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml"
VLLM_CONTAINER = "vllm-grpo"
"""Default container key for GRPO vLLM server jobs (matches cluster_configs key).

This is only a fallback default: the value is configurable per stage via the
``vllm_container`` key in the workflow YAML (see base.yaml
``collect_rollouts`` / ``compute_rewards``), read in :func:`rollout._parse_rollout_config`
and :func:`verify` as ``config.get("vllm_container", VLLM_CONTAINER)``.

Defaults to ``vllm-grpo`` (vLLM v0.20.0) rather than ``vllm`` (v0.22.0) so the
rollout-collection and reward (judge) servers run the SAME vLLM version the GRPO
training stage generates with -- keeping tokenization, sampling, and
tool/reasoning-parser behavior consistent between rollout data and training.
The ``vllm`` (v0.22.0) container remains for SDG / eval. Both are built from
dockerfiles/Dockerfile.vllm; ``VLLM_VERSION`` selects the base tag.
"""


def resolve_environments(config: dict[str, Any]) -> dict[str, Any]:
    """Return the environments to process based on the ``_environment`` filter.

    ``config["_environment"]`` may be a single name (``str``), a list of
    names (``list[str]``), or ``None``.  When set, only the matching
    entries are returned (preserving order).  When ``None``, all
    environments from ``config["environments"]`` are returned.

    Raises ``ValueError`` if a requested environment doesn't exist.
    """
    envs = config.get("environments", {})
    if not envs:
        raise ValueError("'environments' dict is required but missing from config")
    selected = config.get("_environment")
    if selected:
        names = [selected] if isinstance(selected, str) else list(selected)
        for name in names:
            if name not in envs:
                available = ", ".join(envs.keys())
                raise ValueError(f"Unknown environment '{name}'. Available: {available}")
        return {name: envs[name] for name in names}
    return envs


def get_env_from_environments(config: dict[str, Any]) -> tuple[str, str, str]:
    """Derive NeMo-Gym identifiers from the ``environments`` dict.

    Returns ``(resources_server_name, env_inner_name, agent_name)`` where:

    - *resources_server_name*: NeMo-Gym top-level config key.  Defaults to
      the env dict key but can be overridden via ``resources_server_name``
      in the environment config (needed when the NeMo-Gym YAML uses a
      different top-level key, e.g. ``finance_sec_search_resources_server``).
    - *env_inner_name*: the env dict key, which always matches the inner
      ``resources_servers`` key in NeMo-Gym configs.
    - *agent_name*: for single-env returns the configured agent name; for
      multi-env returns empty (triggers ``agent_ref`` routing).
    """
    environments = config["environments"]
    env_names = list(environments.keys())
    if len(env_names) == 1:
        env_cfg = environments[env_names[0]]
        rs_name = env_cfg.get("resources_server_name", env_names[0])
        return rs_name, env_names[0], env_cfg.get("agent_name", f"{env_names[0]}_simple_agent")
    env_cfg = environments[env_names[0]]
    rs_name = env_cfg.get("resources_server_name", env_names[0])
    return rs_name, env_names[0], ""


def build_config_paths_str(config: dict[str, Any]) -> str:
    """Build the comma-separated NeMo-Gym config_paths string.

    Prepends ``vllm_model.yaml`` and combines all environment
    config_paths from the ``environments`` dict.  Appends the agent
    config overlay from ``prepare_data_dir`` if set.
    """
    environments = config["environments"]
    config_paths = [VLLM_MODEL]
    for env_cfg in environments.values():
        config_paths.extend(env_cfg.get("config_paths", []))
    prepare_data_dir = config.get("prepare_data_dir")
    if prepare_data_dir:
        config_paths.append(f"{prepare_data_dir}/agent_config_overlay.yaml")
    return ",".join(config_paths)


# ============================================================================
# vLLM & judge configuration
# ============================================================================

NON_VLLM_KEYS = frozenset(
    {
        "num_gpus",
        "server_nodes",
        "model_path",
        "base_url",
        "openai_base_url",
        "openai_model",
        "openai_api_key",
        "uses_reasoning_parser",
        "tensor_parallel_size",
        "trust_remote_code",
        "hf_config_overrides",
        "server_entrypoint",
        "num_samples_in_parallel",
        "dependent_jobs",
    }
)
"""Keys in vLLM config dicts that are NOT ``vllm serve`` CLI flags.

Used by both policy and judge vLLM configs -- callers strip these before
passing the remaining keys to :func:`build_vllm_server_args`.
``hf_config_overrides`` is handled separately via model overlay (see :func:`_overlay_path`).
"""


def _judge_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("judge_vllm") or {}


def _policy_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("policy_vllm") or {}


def compute_num_gpus(config: dict[str, Any], *, has_policy: bool = True) -> int:
    """Compute total Slurm GPU request from per-endpoint num_gpus.

    Args:
        config: Stage config with ``policy_vllm`` and/or ``judge_vllm`` blocks.
        has_policy: If True (collect_rollouts), include policy GPUs.
            If False (compute_rewards), only count judge GPUs.
    """
    pcfg = _policy_cfg(config)
    jcfg = _judge_cfg(config)
    policy_gpus = pcfg.get("num_gpus", 0) if has_policy else 0
    judge_gpus = jcfg.get("num_gpus", 0)
    return policy_gpus + judge_gpus


def determine_judge_mode(
    config: dict[str, Any],
    *,
    allow_policy_as_judge: bool = True,
) -> str:
    """Return 'local_vllm', 'external_vllm', 'openai', or 'policy_as_judge'.

    Reads from ``config.judge_vllm`` sub-config.

    - ``base_url`` set → ``external_vllm`` (pre-launched vLLM server)
    - ``model_path`` set (no base_url) → ``local_vllm`` (launch server in job)
    - ``openai_base_url`` set → ``openai`` (OpenAI-compatible API)
    - None of the above → ``policy_as_judge`` (if allowed)
    """
    jcfg = _judge_cfg(config)
    if jcfg.get("base_url"):
        return "external_vllm"
    if jcfg.get("model_path"):
        return "local_vllm"
    if jcfg.get("openai_base_url"):
        return "openai"
    if allow_policy_as_judge:
        return "policy_as_judge"
    raise ValueError(
        "A judge configuration is required. "
        "Set judge_vllm.model_path (local vLLM), "
        "judge_vllm.base_url (external vLLM), or "
        "judge_vllm.openai_base_url (OpenAI API)."
    )


def validate_judge_config(config: dict[str, Any]) -> None:
    """Validate judge-related config fields (call after determine_judge_mode)."""
    jcfg = _judge_cfg(config)
    if jcfg.get("openai_base_url") and not jcfg.get("openai_model"):
        raise ValueError(
            "'judge_vllm.openai_model' is required when 'judge_vllm.openai_base_url' is set"
        )


def log_judge_details(console_obj: Any, config: dict[str, Any], judge_mode: str) -> None:
    jcfg = _judge_cfg(config)
    console_obj.detail("Judge mode", judge_mode)
    if judge_mode == "local_vllm":
        console_obj.detail("Judge model", jcfg["model_path"])
        console_obj.detail("Judge GPUs", str(jcfg.get("num_gpus", 0)))
    elif judge_mode == "external_vllm":
        console_obj.detail("Judge endpoint", jcfg["base_url"])
    elif judge_mode == "openai":
        console_obj.detail("Judge endpoint", jcfg["openai_base_url"])
        console_obj.detail("Judge model", jcfg.get("openai_model", "(default)"))
        if jcfg.get("openai_api_key"):
            console_obj.detail("Judge API key", "from workflow YAML (explicit override)")
        else:
            console_obj.detail("Judge API key", "$OPENAI_API_KEY from container env")


def build_vllm_server_args(overrides: dict[str, Any]) -> str:
    """Convert a dict of vLLM overrides into CLI flags.

    Every key in *overrides* is emitted as a flag -- the caller is responsible
    for stripping non-vLLM keys before calling this function.

    Underscores are converted to hyphens (``max_model_len`` → ``--max-model-len``).
    Boolean ``True`` emits a bare flag; ``False`` suppresses it.
    """
    parts: list[str] = []
    for key, value in overrides.items():
        cli_key = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                parts.append(cli_key)
        elif isinstance(value, str) and value.startswith("{"):
            parts.append(f"{cli_key} '{value}'")
        else:
            parts.append(f"{cli_key} {value}")
    return " ".join(parts)


def _overlay_path(model_path: str, hf_config_overrides: dict[str, Any]) -> str:
    """Compute the deterministic overlay directory path for *model_path*.

    The overlay name is content-addressed: it includes a hash of the
    serialized *hf_config_overrides* so a new overlay is created only
    when the overrides change.
    """
    override_json = json.dumps(hf_config_overrides, sort_keys=True)
    digest = hashlib.sha256(override_json.encode()).hexdigest()[:12]
    model_name = model_path.rstrip("/").rsplit("/", 1)[-1]
    overlay_name = f"{model_name}-overlay-{digest}"
    return model_path.rstrip("/").rsplit("/", 1)[0] + "/" + overlay_name


def _build_overlay_setup_cmd(
    model_path: str,
    overlay_path: str,
    hf_config_overrides: dict[str, Any],
) -> str:
    """Return a bash snippet that creates a symlinked model overlay.

    The snippet is designed to run **inside the Slurm job** (where
    container mounts are available) before the vLLM server starts.
    It invokes :mod:`nvflow.lib.rl.create_overlay` which is mounted
    at ``/nemo_run/code/`` inside the container.
    """
    overrides_json = json.dumps(hf_config_overrides, sort_keys=True)
    return (
        f"PYTHONPATH={CONTAINER_CODE_DIR} python3 -m nvflow.lib.rl.create_overlay"
        f" --model-path {model_path}"
        f" --overlay-path {overlay_path}"
        f" --overrides '{overrides_json}'"
    )


def build_judge_ng_run_overrides(
    config: dict[str, Any],
    judge_mode: str,
    *,
    judge_url_var: str = "$JUDGE_VLLM_URL",
) -> str:
    """Build ng_run CLI overrides that configure the judge model server.

    Reads from ``config["judge_vllm"]``, ``config["environment_name"]``
    (NeMo-Gym top-level key), and ``config["environment_inner_name"]``
    (inner ``resources_servers`` key, defaults to ``environment_name``).
    For policy_as_judge: returns empty string (judge uses policy_model).

    Args:
        judge_url_var: URL or shell variable for the judge vLLM endpoint.
    """
    jcfg = _judge_cfg(config)
    env_name = config["environment_name"]
    inner_name = config.get("environment_inner_name", env_name)
    judge_server_override = (
        f'    "+{env_name}.resources_servers.{inner_name}.judge_model_server.name=judge_model" \\\n'
    )

    if judge_mode in ("local_vllm", "external_vllm"):
        judge_model = jcfg.get("model_path", "")
        uses_reasoning_parser = str(jcfg.get("uses_reasoning_parser", False)).lower()
        if judge_mode == "external_vllm":
            judge_url_var = jcfg["base_url"]
        return (
            '    "+judge_model.responses_api_models.vllm_model.entrypoint=app.py" \\\n'
            f'    "+judge_model.responses_api_models.vllm_model.base_url={judge_url_var}" \\\n'
            '    "+judge_model.responses_api_models.vllm_model.api_key=EMPTY" \\\n'
            f'    "+judge_model.responses_api_models.vllm_model.model={judge_model}" \\\n'
            '    "+judge_model.responses_api_models.vllm_model.return_token_id_information=false" \\\n'
            f'    "+judge_model.responses_api_models.vllm_model.uses_reasoning_parser={uses_reasoning_parser}" \\\n'
            + judge_server_override
        )

    if judge_mode == "openai":
        base_url = jcfg["openai_base_url"]
        model = jcfg["openai_model"]
        api_key = jcfg.get("openai_api_key", "$OPENAI_API_KEY")
        lines = [
            '    "+judge_model.responses_api_models.openai_model.entrypoint=app.py" \\\n',
            f'    "+judge_model.responses_api_models.openai_model.openai_base_url={base_url}" \\\n',
            f'    "+judge_model.responses_api_models.openai_model.openai_api_key={api_key}" \\\n',
            f'    "+judge_model.responses_api_models.openai_model.openai_model={model}" \\\n',
            judge_server_override,
        ]
        return "".join(lines)

    # policy_as_judge: no judge overrides needed
    return ""


def build_ng_run_invocation(
    *,
    step_label: str,
    policy_base_url: str,
    policy_model: str,
    judge_ng_run_overrides: str = "",
    include_port_range: bool = False,
) -> str:
    """Render the shared ``ng_run`` invocation block (rollout + verify).

    Common to ``collect_rollouts`` (``rollout.py``) and ``compute_rewards``
    (``verify.py``).  Emits the step-header echo, launches ``gym env start`` in
    the background with policy-model + head-server overrides, captures
    ``NG_RUN_PID`` for trap-on-cleanup, and probes the head server via
    ``wait_for_server``.

    The caller is responsible for everything *outside* the invocation
    block:
      - ``HEAD_SERVER_PORT=$(find_free_port)`` (rollout allocates this
        just before ng_run to minimise the TOCTOU window; verify
        allocates earlier in setup so the port is also visible to the
        port-read preamble).
      - ``cd "$GYM_PATH"`` (the Gym CLI is provided by the stage's
        ``installation_command`` -- on PATH -- not sourced here).
      - ``SHELL_WAIT_FOR_SERVER`` definition (must be in scope before
        this block runs; both call sites already include it in their
        setup segment).

    Args:
        step_label: Step header text, e.g. ``"[Step 2/3]"`` (rollout has
            three steps: ng_run, collect, finalize) or ``"[Step 1/2]"``
            (verify has two: ng_run, re-judge).
        policy_base_url: Bash literal or shell variable used for
            ``vllm_model.base_url``.  Rollout uses ``"$VLLM_URL"``
            (the live policy server); verify uses
            ``"http://localhost:0/v1"`` (a stub -- re-judge does not
            invoke the policy).
        policy_model: Bash literal or shell variable used for
            ``vllm_model.model``.  Rollout uses ``"$MODEL_PATH"``;
            verify uses ``"unused"``.
        judge_ng_run_overrides: Pre-formatted overrides string from
            :func:`build_judge_ng_run_overrides`.  Empty string for
            policy-as-judge mode.
        include_port_range: When ``True``, emits the
            ``+port_range_low=1024 +port_range_high=8999`` workaround
            for the NeMo-Gym ephemeral-port collision issue.  Required
            for rollout (where local vLLM lives on the same node and
            collides with NeMo-Gym's default port range).  Not needed
            for re-judge where no policy vLLM runs.
    """
    port_range_block = (
        '    "+port_range_low=1024" \\\n    "+port_range_high=8999" \\\n'
        if include_port_range
        else ""
    )
    return (
        'echo ""\n'
        f'echo "{step_label} Starting NeMo-Gym servers ..."\n'
        'gym env start "+config_paths=[$CONFIG_PATHS]" \\\n'
        f'    "+policy_model.responses_api_models.vllm_model.base_url={policy_base_url}" \\\n'
        '    "+policy_model.responses_api_models.vllm_model.api_key=EMPTY" \\\n'
        f'    "+policy_model.responses_api_models.vllm_model.model={policy_model}" \\\n'
        '    "+head_server.host=127.0.0.1" \\\n'
        '    "+head_server.port=$HEAD_SERVER_PORT" \\\n'
        f"{port_range_block}"
        '    "+skip_venv_if_present=true" \\\n'
        # Reuse the baked per-component venvs. UV_VENV_DIR is set by the caller's
        # variables segment: /opt/gym-venvs for the nemo-gym image, else $GYM_PATH
        # (== Gym's default PARENT_DIR), which is behavior-preserving.
        '    "+uv_venv_dir=$UV_VENV_DIR" \\\n'
        f"{judge_ng_run_overrides}"
        '    > "$OUTPUT_DIR/logs/ng_run_$JOB_LABEL.log" 2>&1 &\n'
        "NG_RUN_PID=$!\n"
        "\n"
        'wait_for_server "http://127.0.0.1:$HEAD_SERVER_PORT/" "NeMo-Gym" $NG_RUN_PID 60 "$OUTPUT_DIR/logs/ng_run_$JOB_LABEL.log"\n'
    )


def build_judge_nemo_gym_config(
    config: dict[str, Any],
    judge_mode: str,
    *,
    environment_name: str,
    environment_inner_name: str = "",
    judge_url_var: str = "",
) -> dict[str, Any]:
    """Build a dict fragment to merge into NeMo-Gym ``initial_global_config_dict``.

    This is the in-memory equivalent of :func:`build_judge_ng_run_overrides`
    (which produces CLI strings for ``ng_run``).  Used by the GRPO training
    stage where the config is passed as a Python dict, not shell arguments.

    Returns an empty dict for ``policy_as_judge`` mode (no-op).

    For ``local_vllm`` / ``external_vllm`` / ``openai`` modes, returns a dict
    with two top-level keys:

    - ``judge_model``: a new ``responses_api_models`` entry (vllm_model or
      openai_model adapter) that NeMo-Gym's ``RunHelper`` will launch.
    - ``<environment_name>``: override of ``judge_model_server.name`` to
      route judge requests to the new ``judge_model`` server.

    Args:
        config: Stage config containing ``judge_vllm`` sub-config.
        judge_mode: One of the modes returned by :func:`determine_judge_mode`.
        environment_name: NeMo-Gym top-level config key (may differ from
            the inner ``resources_servers`` key for environments that use
            the ``_resources_server`` suffix convention).
        environment_inner_name: Inner ``resources_servers`` key.  Defaults
            to *environment_name* when empty (backward compatible with
            environments where both keys are the same).
        judge_url_var: For ``local_vllm`` mode, the URL (or shell variable)
            where the judge vLLM engine will be reachable.  Ignored for other
            modes.
    """
    if judge_mode == "policy_as_judge":
        return {}

    jcfg = _judge_cfg(config)
    inner_name = environment_inner_name or environment_name

    judge_server_name_override = {
        environment_name: {
            "resources_servers": {
                inner_name: {
                    "judge_model_server": {"name": "judge_model"},
                },
            },
        },
    }

    if judge_mode in ("local_vllm", "external_vllm"):
        judge_model = jcfg.get("model_path", "")
        uses_reasoning_parser = jcfg.get("uses_reasoning_parser", False)
        base_url = jcfg["base_url"] if judge_mode == "external_vllm" else judge_url_var
        return {
            "judge_model": {
                "responses_api_models": {
                    "vllm_model": {
                        "entrypoint": "app.py",
                        "base_url": base_url,
                        "api_key": "EMPTY",
                        "model": judge_model,
                        "return_token_id_information": False,
                        "uses_reasoning_parser": uses_reasoning_parser,
                    },
                },
            },
            **judge_server_name_override,
        }

    # judge_mode == "openai"
    return {
        "judge_model": {
            "responses_api_models": {
                "openai_model": {
                    "entrypoint": "app.py",
                    "openai_base_url": jcfg["openai_base_url"],
                    "openai_api_key": jcfg.get("openai_api_key", ""),
                    "openai_model": jcfg["openai_model"],
                },
            },
        },
        **judge_server_name_override,
    }


# ============================================================================
# Shell script templates
# ============================================================================

# Reusable bash snippets concatenated into inline command strings built
# by _build_client_cmd() and similar builders.

SHELL_FIND_FREE_PORT = """\
find_free_port() {
    python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}
"""

SHELL_READ_PORT_FILE = """\
read_port_file() {
    local path="$1" name="$2" timeout="${3:-120}"
    local elapsed=0
    while [ ! -f "$path" ]; do
        sleep 1
        elapsed=$((elapsed + 1))
        if [ $elapsed -ge $timeout ]; then
            echo "ERROR: $name port file not found after ${timeout}s: $path" >&2
            exit 1
        fi
    done
    cat "$path"
}
"""

SHELL_WAIT_FOR_SERVER = """\
wait_for_server() {
    local url="$1" name="$2" pid="$3" max_attempts="$4" log="$5"
    echo "  Waiting for $name at $url ..."
    for i in $(seq 1 $max_attempts); do
        if curl -s -m 5 "$url" > /dev/null 2>&1; then
            echo "  $name ready after $((i * 5))s"
            return 0
        fi
        if ! kill -0 $pid 2>/dev/null; then
            echo "ERROR: $name died. Check $log"
            exit 1
        fi
        sleep 5
    done
    echo "ERROR: $name did not start within $((max_attempts * 5))s"
    exit 1
}
"""
