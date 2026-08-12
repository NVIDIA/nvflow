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
"""Collect rollouts against a NeMo-Gym environment.

Provides :func:`rollout` -- the RL equivalent of nemo-skills'
``generate()``.  Handles Slurm orchestration (single or heterogeneous
jobs for dual-server setups), vLLM lifecycle, chunking, seeding,
merge, and cross-seed aggregation.

URL resolution uses the same lazy ``set_inline(callable)`` pattern as
nemo-skills' ``GenerationClientScript`` so that ``hostname_ref()``
resolves correctly after the Pipeline assigns het-group indices.
"""

import shlex
from dataclasses import dataclass, field
from typing import Any

from nemo_skills.pipeline.utils.scripts import BaseJobScript, ServerScript

from nvflow.core import console
from nvflow.lib.executor import is_ray_backend

from .helpers import (
    CONTAINER_CODE_DIR,
    NON_VLLM_KEYS,
    SHELL_FIND_FREE_PORT,
    SHELL_READ_PORT_FILE,
    SHELL_WAIT_FOR_SERVER,
    VLLM_CONTAINER,
    LauncherFS,
    _build_overlay_setup_cmd,
    _overlay_path,
    build_config_paths_str,
    build_judge_ng_run_overrides,
    build_ng_run_invocation,
    build_vllm_server_args,
    check_launcher_cwd,
    compute_num_gpus,
    determine_judge_mode,
    get_env_from_environments,
    log_judge_details,
)


@dataclass(kw_only=True)
class BashScript(BaseJobScript):
    """A ``BaseJobScript`` that runs a bash command string.

    Module-level class so Fiddle/nemo-run can serialize it by import path.
    """

    cmd: str = ""

    def __post_init__(self):
        self.set_inline(self.cmd)
        super().__post_init__()


@dataclass(kw_only=True)
class RolloutClientScript(BaseJobScript):
    """Client script for NeMo-Gym rollout collection with lazy URL resolution.

    Uses the same lazy ``set_inline(callable)`` pattern as nemo-skills'
    ``GenerationClientScript``.  The callable is evaluated by the Pipeline
    **after** ``het_group_index`` has been assigned to all scripts, so
    ``hostname_ref()`` returns the correct Slurm shell variable for
    cross-node communication in heterogeneous jobs.

    vLLM ports are allocated dynamically on the compute node at runtime.
    The server writes its port to a file on the shared filesystem; the
    client waits for that file and reads the port before constructing URLs.
    """

    policy_server: ServerScript | None = None
    judge_server: ServerScript | None = None
    policy_base_url: str = ""
    config: dict | None = field(default=None, repr=False)
    judge_mode: str = ""
    judge_ng_run_overrides: str = ""

    output_dir: str = ""
    gym_path: str = ""
    uv_venv_dir: str = ""
    model_path: str = ""
    agent_name: str = ""
    input_data: str = ""
    output_file: str = ""
    done_file: str = ""
    config_paths: str = ""
    num_parallel: int = 4
    job_label: str = ""
    max_num_samples: int = 0
    chunk_id: int = 0
    num_chunks: int = 1
    responses_create_params: dict = field(default_factory=dict)
    log_dir: str = ""
    # Per-rollout-attempt Python-baked constant; must equal the ``attempt_id``
    # given to the paired ``make_server_script`` so the reader's port-file path
    # matches the writer's (see :func:`_vllm_port_file`).
    attempt_id: str = ""
    # True on the Ray Jobs backend.  Gates the ``ray_head_node_address``
    # override so the gym attaches to the pre-provisioned cluster instead of
    # forking its own (see :func:`_build_client_cmd`).  Slurm: stays False so
    # the generated command is byte-identical to the validated Slurm path.
    is_ray: bool = False
    # Shared-FS path the client touches in its cleanup trap (Ray only) so the
    # paired serve's watcher stops vLLM and exits 0.  Must equal the
    # ``stop_sentinel`` given to the paired ``make_server_script`` (see
    # :func:`_server_stop_sentinel`).
    stop_sentinel: str = ""

    def __post_init__(self):
        def build_cmd() -> str:
            if self.policy_server is not None:
                hostname = self.policy_server.hostname_ref()
                policy_url = f"http://{hostname}:$POLICY_PORT/v1"
            else:
                policy_url = self.policy_base_url

            if self.judge_server is not None:
                judge_hostname = self.judge_server.hostname_ref()
                judge_url = f"http://{judge_hostname}:$JUDGE_PORT/v1"
                judge_overrides = build_judge_ng_run_overrides(
                    self.config, self.judge_mode, judge_url_var=judge_url
                )
            else:
                judge_url = ""
                judge_overrides = self.judge_ng_run_overrides

            cmd = _build_client_cmd(
                output_dir=self.output_dir,
                gym_path=self.gym_path,
                uv_venv_dir=self.uv_venv_dir,
                model_path=self.model_path,
                agent_name=self.agent_name,
                input_data=self.input_data,
                output_file=self.output_file,
                done_file=self.done_file,
                config_paths=self.config_paths,
                num_parallel=self.num_parallel,
                job_label=self.job_label,
                policy_vllm_url=policy_url,
                judge_vllm_url=judge_url,
                judge_ng_run_overrides=judge_overrides,
                max_num_samples=self.max_num_samples,
                chunk_id=self.chunk_id,
                num_chunks=self.num_chunks,
                responses_create_params=self.responses_create_params,
                is_ray=self.is_ray,
                stop_sentinel=self.stop_sentinel,
            )

            preamble = _build_port_read_preamble(
                self.log_dir,
                self.job_label,
                has_policy=self.policy_server is not None,
                has_judge=self.judge_server is not None,
                attempt_id=self.attempt_id,
            )
            return preamble + cmd

        self.set_inline(build_cmd)
        super().__post_init__()


# ---------------------------------------------------------------------------
# Script helpers
# ---------------------------------------------------------------------------


def _vllm_port_file(log_dir: str, role: str, job_label: str = "", attempt_id: str = "") -> str:
    """Return the shared-filesystem path for the dynamic vLLM port file.

    Suffixed with a per-job id so each job gets a unique file, avoiding a race
    where the client's ``rm -f`` of stale port files deletes the one the server
    just wrote.  Falls back ``$SLURM_JOB_ID`` -> ``$RAY_JOB_ID`` -> ``$$`` so the
    name stays unique under the Ray Jobs backend, where ``$SLURM_JOB_ID`` is unset.

    *attempt_id* is a Python-baked, per-rollout-attempt constant (e.g. a uuid4
    hex generated once in the command builder) that is interpolated identically
    into BOTH the server (writer) and client (reader) paths for a single attempt.
    It MUST be a constant baked into both generated command strings — never a
    bash-runtime value like ``$$`` or ``$RANDOM`` — because the writer and reader
    are *different processes* (and, in multi-node jobs, different nodes), so a
    runtime-random value would differ between them and they would never agree on
    the path.  Why it matters: on the Ray Jobs backend ``$SLURM_JOB_ID`` is
    constant for the pre-provisioned head's whole life, so *every* rollout attempt
    (driver re-runs and retries) would otherwise reuse the SAME port file and a
    client could read a STALE port written by a prior attempt and wait forever on
    a dead server.  The Python-baked *attempt_id* makes the path unique per attempt
    while keeping the paired writer+reader in agreement.
    """
    suffix = f"_{job_label}" if job_label else ""
    if not attempt_id:
        # Preserve the established Slurm command byte-for-byte.  Ray Jobs need
        # the wider fallback chain below because they do not define
        # SLURM_JOB_ID, but ordinary Slurm jobs must not pay for or inherit any
        # of that backend-specific machinery.
        return f"{log_dir}/.vllm_port_{role}{suffix}_${{SLURM_JOB_ID}}.txt"
    return (
        f"{log_dir}/.vllm_port_{role}{suffix}_{attempt_id}_"
        "${SLURM_JOB_ID:-${RAY_JOB_ID:-$$}}.txt"
    )


def _new_ray_attempt_id() -> str:
    """Return a per-attempt nonce without importing UUID on the Slurm path."""
    import uuid

    return uuid.uuid4().hex[:12]


def _python_module_prefix(is_ray: bool) -> str:
    """Return a Slurm-only import prefix for an in-tree Python module.

    Ray's working-dir runtime env already prepends the extracted archive root
    to ``PYTHONPATH``. Reassigning it in every command is redundant and risks
    diverging from Ray's own code-delivery lifecycle.
    """
    return "" if is_ray else f"PYTHONPATH={CONTAINER_CODE_DIR} "


def _server_stop_sentinel(log_dir: str, job_label: str = "", attempt_id: str = "") -> str:
    """Return the shared-FS path for the rollout-client → server stop sentinel.

    Modelled on :func:`_vllm_port_file` (same *log_dir* + per-attempt *attempt_id*
    so the writer [client] and reader [server] agree on one path).  It is
    role-agnostic ("stop"): the client touches a SINGLE sentinel for an attempt
    and BOTH the policy and judge servers (when present) poll the same path.

    Critically — UNLIKE the port file — the path does NOT embed
    ``$SLURM_JOB_ID``/``$RAY_JOB_ID``: the client and server run as SEPARATE Ray
    jobs under the Ray Jobs backend (the policy serve is its own het-group job), so
    a per-job-id suffix would differ between writer and reader and they would
    never rendezvous.  The Python-baked *attempt_id* alone keeps the path unique
    per rollout attempt while keeping the paired writer+reader in agreement.
    """
    suffix = f"_{job_label}" if job_label else ""
    attempt = f"_{attempt_id}" if attempt_id else ""
    return f"{log_dir}/.server_stop{suffix}{attempt}.sentinel"


def _wrap_server_with_dynamic_port(
    script: ServerScript,
    role: str,
    port_file: str,
    *,
    is_ray: bool = False,
    stop_sentinel: str = "",
) -> None:
    """Replace the hardcoded port in *script* with runtime-dynamic allocation.

    Wraps the server's inline command so that at runtime on the compute node:
      1. ``find_free_port()`` probes for an available port
      2. The port is written to *port_file* (shared filesystem)
      3. ``sed`` replaces the hardcoded port in the original command

    On the Ray Jobs backend (*is_ray* True, *stop_sentinel* set) the vLLM
    process is run in the BACKGROUND under its own process group/session leader
    (``setsid``) and a watcher polls for *stop_sentinel*.  When the paired
    rollout client finishes it touches the sentinel (in its cleanup trap, on
    BOTH success and failure) and the watcher signals the ENTIRE vLLM process
    GROUP — ``kill -TERM -<pgid>`` then, after a short grace, ``kill -KILL
    -<pgid>`` — so vLLM's multiprocessing children (EngineCore, ``Worker_TP*``)
    die too and the GPU memory they pin (~74 GiB observed live) is released
    BEFORE this serve Ray job exits.  The watcher then ``exit 0`` so the job
    completes SUCCEEDED, and the driver advances past the ``collect_rollouts``
    gate.  Because ``setsid`` makes the child a new session leader, its PGID
    equals its PID, so ``-$_NVFLOW_VLLM_PID`` targets the whole tree without
    killing this watcher script.

    Killing only the launcher (the pre-refinement behaviour) left those workers
    ORPHANED holding GPU memory for >75s after the job was already SUCCEEDED,
    OOM-ing the next colocated stage (GRPO training on the same single node) at
    engine init.  On Slurm the het-group allocation teardown frees the GPUs
    between stages, so this only bit the Ray single-node path.

    If vLLM instead dies on its own (a real failure) the watcher reaps any
    surviving group members and exits with vLLM's non-zero code, so genuine
    crashes are NOT masked.

    On Slurm (*is_ray* False) the serve stays FOREGROUND and the het-group's
    allocation teardown reaps it — the emitted command is byte-identical to the
    validated Slurm path (the sentinel branch is never built).
    """
    hardcoded = str(script.port)
    original_inline = script.inline
    if is_ray and stop_sentinel:
        run_vllm = (
            f"ORIG_CMD=$(cat <<'__NVFLOW_VLLM_CMD__'\n"
            f"{original_inline}\n"
            f"__NVFLOW_VLLM_CMD__\n"
            f")\n"
            f'VLLM_RUN_CMD=$(echo "$ORIG_CMD" | sed "s/{hardcoded}/$VLLM_PORT/g")\n'
            # WORKAROUND(nemo-rl-vllm-separate-venv), Ray only: on the nemo-rl
            # image vLLM lives in a per-actor venv (/opt/ray_venvs/*), NOT on
            # PATH, while PATH's python3 (nemo_rl_venv) has nemo_skills/nvflow
            # but no vllm.  Stock serve_vllm spawns the engine with a literal
            # ``python3 -m vllm...`` through the shell, so PATH decides the
            # ENGINE interpreter.  Bridge: resolve a vllm-capable bin dir
            # (env override -> current python3 -> glob probe), prepend it to
            # PATH for the engine spawn, and re-anchor the nemo_skills/nvflow
            # module invocations to the ORIGINAL interpreter so they keep
            # importing.  No-op when PATH python3 already imports vllm (e.g. a
            # dedicated vllm-image cluster) — and always a no-op on Slurm
            # (this branch is Ray-only).
            f"_NVFLOW_OUTER_PY=$(command -v python3)\n"
            f'_NVFLOW_VLLM_BIN=""\n'
            f'if [ -n "${{NVFLOW_VLLM_PYTHON:-}}" ]; then\n'
            f'    _NVFLOW_VLLM_BIN="$(dirname "$NVFLOW_VLLM_PYTHON")"\n'
            f'elif [ -n "${{NVFLOW_VLLM_VENV:-}}" ]; then\n'
            f'    _NVFLOW_VLLM_BIN="$NVFLOW_VLLM_VENV/bin"\n'
            f"elif ! python3 -c 'import vllm' >/dev/null 2>&1; then\n"
            f"    for _nvflow_p in ${{NVFLOW_VLLM_VENV_GLOB:-/opt/ray_venvs/*/bin}}; do\n"
            f'        if [ -x "$_nvflow_p/python3" ] && "$_nvflow_p/python3" -c \'import vllm\' >/dev/null 2>&1; then\n'
            f'            _NVFLOW_VLLM_BIN="$_nvflow_p"\n'
            f"            break\n"
            f"        fi\n"
            f"    done\n"
            f"fi\n"
            f'if [ -n "$_NVFLOW_VLLM_BIN" ]; then\n'
            f'    echo "[serve] vLLM interpreter bridge: engine python from $_NVFLOW_VLLM_BIN (launcher stays $_NVFLOW_OUTER_PY)"\n'
            f'    export PATH="$_NVFLOW_VLLM_BIN:$PATH"\n'
            f'    VLLM_RUN_CMD=$(echo "$VLLM_RUN_CMD" | sed -e "s|python3 -m nemo_skills\\.|$_NVFLOW_OUTER_PY -m nemo_skills.|g" -e "s|python3 -m nvflow\\.|$_NVFLOW_OUTER_PY -m nvflow.|g")\n'
            f"fi\n"
            f'rm -f "{stop_sentinel}"\n'
            f'echo "[server-stop] {role} vLLM running in background (own process '
            f'group via setsid); polling for stop sentinel {stop_sentinel}"\n'
            # ``setsid`` makes the child a session/process-group leader so its
            # PGID == its PID; we can then signal the whole group with a negative
            # PID without killing this watcher.  Fall back to a plain background
            # launch if setsid is unavailable (rare; util-linux ships it on
            # essentially every Linux image).
            f"if command -v setsid >/dev/null 2>&1; then\n"
            f'    setsid bash -c "$VLLM_RUN_CMD" &\n'
            f"    _NVFLOW_VLLM_PID=$!\n"
            f"    _NVFLOW_VLLM_GROUP=1\n"
            f"else\n"
            f'    echo "[server-stop] setsid not found; falling back to '
            f'single-process kill (orphaned vLLM workers may linger on GPU)" >&2\n'
            f'    eval "$VLLM_RUN_CMD" &\n'
            f"    _NVFLOW_VLLM_PID=$!\n"
            f"    _NVFLOW_VLLM_GROUP=0\n"
            f"fi\n"
            f"_nvflow_kill_vllm() {{\n"
            f'    if [ "$_NVFLOW_VLLM_GROUP" = "1" ]; then\n'
            f'        kill -TERM -"$_NVFLOW_VLLM_PID" 2>/dev/null || true\n'
            f"        sleep 5\n"
            f'        kill -KILL -"$_NVFLOW_VLLM_PID" 2>/dev/null || true\n'
            f"    else\n"
            f'        kill -TERM "$_NVFLOW_VLLM_PID" 2>/dev/null || true\n'
            f"        sleep 5\n"
            f'        kill -KILL "$_NVFLOW_VLLM_PID" 2>/dev/null || true\n'
            f"    fi\n"
            f'    wait "$_NVFLOW_VLLM_PID" 2>/dev/null || true\n'
            # Brief settle so CUDA actually frees the worker memory before this
            # job exits and the next (colocated) stage starts its engine init.
            f"    sleep 5\n"
            f"}}\n"
            f"while true; do\n"
            f'    if [ -f "{stop_sentinel}" ]; then\n'
            f'        echo "[server-stop] stop sentinel seen — shutting down '
            f'{role} vLLM process group (pid/pgid $_NVFLOW_VLLM_PID)"\n'
            f"        _nvflow_kill_vllm\n"
            f'        rm -f "{stop_sentinel}"\n'
            f"        exit 0\n"
            f"    fi\n"
            f'    if ! kill -0 "$_NVFLOW_VLLM_PID" 2>/dev/null; then\n'
            f"        _NVFLOW_VLLM_RC=0\n"
            f'        wait "$_NVFLOW_VLLM_PID" || _NVFLOW_VLLM_RC=$?\n'
            # Reap any worker children that outlived the launcher so they do not
            # linger on the GPU, but PRESERVE vLLM's real exit code (genuine
            # crash must still surface) — the group-kill here is best-effort.
            f'        if [ "$_NVFLOW_VLLM_GROUP" = "1" ]; then\n'
            f'            kill -KILL -"$_NVFLOW_VLLM_PID" 2>/dev/null || true\n'
            f"        fi\n"
            f'        echo "[server-stop] {role} vLLM exited on its own '
            f'(code $_NVFLOW_VLLM_RC) before any stop sentinel" >&2\n'
            f"        exit $_NVFLOW_VLLM_RC\n"
            f"    fi\n"
            f"    sleep 5\n"
            f"done\n"
        )
    else:
        run_vllm = (
            f"ORIG_CMD=$(cat <<'__NVFLOW_VLLM_CMD__'\n"
            f"{original_inline}\n"
            f"__NVFLOW_VLLM_CMD__\n"
            f")\n"
            f'eval "$(echo "$ORIG_CMD" | sed "s/{hardcoded}/$VLLM_PORT/g")"\n'
        )
    wrapped = (
        f"{SHELL_FIND_FREE_PORT}\n"
        f"VLLM_PORT=$(find_free_port)\n"
        f'echo "[dynamic-port] {role} vLLM using port $VLLM_PORT (node ${{SLURM_NODEID:-0}})"\n'
        f'if [ "${{SLURM_NODEID:-0}}" = "0" ]; then\n'
        f'    echo "$VLLM_PORT" > "{port_file}"\n'
        f"fi\n"
        f"{run_vllm}"
    )
    script.set_inline(wrapped)


def _build_port_read_preamble(
    log_dir: str,
    job_label: str,
    *,
    has_policy: bool = False,
    has_judge: bool = False,
    attempt_id: str = "",
) -> str:
    """Build bash preamble that reads dynamic vLLM ports from port files.

    Returns empty string if neither server is present.

    *attempt_id* must be the SAME Python-baked constant passed to
    :func:`make_server_script` for the paired server in this attempt, so the
    reader's path matches the writer's path (see :func:`_vllm_port_file`).
    """
    if not has_policy and not has_judge:
        return ""
    parts = [SHELL_READ_PORT_FILE]
    if has_policy:
        pf = _vllm_port_file(log_dir, "policy", job_label, attempt_id)
        parts.append(f'POLICY_PORT=$(read_port_file "{pf}" "Policy vLLM" 300)')
    if has_judge:
        jf = _vllm_port_file(log_dir, "judge", job_label, attempt_id)
        parts.append(f'JUDGE_PORT=$(read_port_file "{jf}" "Judge vLLM" 300)')
    return "\n".join(parts) + "\n"


def make_server_script(
    vllm_cfg: dict[str, Any],
    cluster_config: dict,
    *,
    role: str = "policy",
    log_dir: str = "",
    job_label: str = "",
    attempt_id: str = "",
    is_ray: bool = False,
    stop_sentinel: str = "",
) -> ServerScript:
    if "num_gpus" not in vllm_cfg:
        raise ValueError("vLLM config must specify 'num_gpus'")

    model_path = vllm_cfg["model_path"]
    hf_overrides = vllm_cfg.get("hf_config_overrides")
    if hf_overrides:
        overlay = _overlay_path(model_path, hf_overrides)
    else:
        overlay = None

    vllm_overrides = {k: v for k, v in vllm_cfg.items() if k not in NON_VLLM_KEYS}
    if overlay:
        # serve_vllm.py sets --served-model-name to --model (the overlay path).
        # NeMo-Gym's proxy sends requests using the original model name, so we
        # override served-model-name to keep the original identity.
        vllm_overrides["served_model_name"] = model_path
    script = ServerScript(
        server_type="vllm",
        model_path=overlay or model_path,
        cluster_config=cluster_config,
        num_gpus=vllm_cfg["num_gpus"],
        num_nodes=vllm_cfg.get("server_nodes", 1),
        server_args=build_vllm_server_args(vllm_overrides),
        server_entrypoint=vllm_cfg.get("server_entrypoint"),
    )

    if overlay:
        setup_cmd = _build_overlay_setup_cmd(model_path, overlay, hf_overrides, is_ray=is_ray)
        script.set_inline(f"{setup_cmd} && {script.inline}")

    if log_dir:
        port_file = _vllm_port_file(log_dir, role, job_label, attempt_id)
        _wrap_server_with_dynamic_port(
            script, role, port_file, is_ray=is_ray, stop_sentinel=stop_sentinel
        )

    return script


def make_bash_script(
    bash_cmd: str,
    *,
    installation_command: str | None = None,
) -> BashScript:
    return BashScript(cmd=bash_cmd, installation_command=installation_command)


def _ray_postprocess_installation_command(
    installation_command: str | None,
    *,
    is_ray: bool,
) -> str | None:
    """Select the baked runtime for Ray-only post-processing jobs.

    Slurm post-processing runs in ``postprocess_container`` and must retain
    its existing environment. Ray Jobs, however, all execute on the already
    provisioned Ray head; the container field is metadata rather than a
    per-job container switch. Reusing the stage installation command there
    selects the baked Gym venv for merge/aggregate/filter utilities without
    installing packages at runtime.
    """
    return installation_command if is_ray else None


# ---------------------------------------------------------------------------
# Filename conventions
# ---------------------------------------------------------------------------


def _output_filename(seed: int, chunk_id: int) -> str:
    return f"rs{seed}/chunk_{chunk_id}.jsonl"


def _merged_filename(seed: int) -> str:
    return f"output-rs{seed}.jsonl"


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------


def _get_remaining_jobs(
    fs: LauncherFS,
    rollout_dir: str,
    seeds: list[int],
    chunk_ids: list[int],
    rerun_done: bool,
) -> list[tuple[int, int]]:
    """Return ``(seed, chunk)`` pairs that still need to run.

    Two-level integrity check (see Fix 4 in the data-loss plan):

    1. **Merged level** — if merged ``.done`` + merged data file both exist,
       the seed is complete.  Skip it even if chunk output files were
       deleted by merge cleanup (that is the expected post-merge state).
       If ``.done`` exists without a data file, the marker is stale —
       delete it and fall through to chunk-level checks.

    2. **Chunk level** — if a chunk ``.done`` exists but the chunk output
       file is missing (and no successful merge), the marker is stale.
       Delete it and re-schedule the chunk.

    Filesystem access goes through :class:`LauncherFS`, so this resume scan
    works both within the cluster (local ``Path`` ops -- unchanged behaviour)
    and off-cluster (existence/rm run on the cluster over the ssh tunnel).
    """
    if rerun_done:
        rm_paths: list[str] = []
        for s in seeds:
            for c in chunk_ids:
                base = f"{rollout_dir}/{_output_filename(s, c)}"
                rm_paths += [f"{base}.done", f"{base}-async", f"{base}-async.prev", base]
        fs.rm(rm_paths)
        return [(s, c) for s in seeds for c in chunk_ids]

    # Batch all existence checks (a single round-trip group when remote).
    probe: list[str] = []
    for s in seeds:
        mf = _merged_filename(s)
        probe += [f"{rollout_dir}/{mf}.done", f"{rollout_dir}/{mf}"]
        for c in chunk_ids:
            base = f"{rollout_dir}/{_output_filename(s, c)}"
            probe += [f"{base}.done", base]
    ex = fs.batch_exists(probe)

    remaining: list[tuple[int, int]] = []
    stale: list[str] = []
    for s in seeds:
        mf = _merged_filename(s)
        merge_done = f"{rollout_dir}/{mf}.done"
        merge_file = f"{rollout_dir}/{mf}"

        if ex.get(merge_done):
            if ex.get(merge_file):
                continue
            console.warning(f"Merged .done exists but {mf} is missing — resetting merge marker")
            stale.append(merge_done)

        for c in chunk_ids:
            base = f"{rollout_dir}/{_output_filename(s, c)}"
            done = f"{base}.done"
            if ex.get(done) and not ex.get(base):
                console.warning(f"Stale .done for {_output_filename(s, c)} — re-scheduling")
                stale.append(done)
                remaining.append((s, c))
            elif not ex.get(done):
                remaining.append((s, c))

    if stale:
        fs.rm(stale)

    return remaining


# ---------------------------------------------------------------------------
# Inline command builders
# ---------------------------------------------------------------------------


def _build_vllm_wait_snippet(policy_url: str, judge_url: str = "") -> str:
    """Return bash snippet that polls vLLM servers until ready.

    Reuses the ``wait_for_server`` bash function (from SHELL_WAIT_FOR_SERVER)
    which is always emitted earlier in the client command.
    Uses ``$$`` (current shell PID) as a dummy -- vLLM runs in a separate
    het-group so we cannot check its PID, but ``kill -0 $$`` always succeeds,
    effectively skipping the "process died" early-exit while keeping curl polling.
    """
    snippet = f'wait_for_server "{policy_url}/models" "Policy vLLM" $$ 400 /dev/null\n'
    if judge_url:
        snippet += f'wait_for_server "{judge_url}/models" "Judge vLLM" $$ 400 /dev/null\n'
    return snippet


# ---------------------------------------------------------------------------
# Per-segment helpers for _build_client_cmd
# ---------------------------------------------------------------------------
# Each helper renders one labeled section of the rollout client bash script.
# Composed by _build_client_cmd in the documented order; do NOT change order.
#
# Conventions:
#   - Constant segments take no arguments.
#   - Parameterized segments take only the values they interpolate.
#   - Output is concatenated by ``+`` -- each helper is responsible for its
#     own leading/trailing newlines.  The snapshot tests in tests/test_rollout
#     pin the byte-exact concatenation, so any whitespace change here will
#     surface as a fixture diff.


def _bash_dq(value: str) -> str:
    """Emit *value* as a bash double-quoted literal.

    Unlike :func:`shlex.quote` (which uses single quotes when special
    characters are present and thereby suppresses bash variable
    expansion), this helper emits a double-quoted string so embedded
    bash variables (``$VAR`` and ``${VAR:-default}``) and command
    substitution (``$(cmd)``, ``` `cmd` ```) are evaluated at
    assignment time.

    Use ONLY for fields whose values legitimately contain bash
    parameter expansions that must be resolved at runtime -- e.g., URL
    templates built from :meth:`ServerScript.hostname_ref` and dynamic
    port variables (``${SLURM_MASTER_NODE_HET_GROUP_0:-localhost}``,
    ``$POLICY_PORT``).  For paths and arbitrary user-supplied strings,
    keep using :func:`shlex.quote`.

    Escapes backslash, double-quote, and backtick to preserve them as
    literals; ``$`` is NOT escaped (so expansions still apply -- that
    is the entire point of this helper).
    """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
    return f'"{escaped}"'


def _build_variables_segment(
    *,
    output_dir: str,
    gym_path: str,
    uv_venv_dir: str = "",
    model_path: str,
    agent_name: str,
    input_data: str,
    output_file: str,
    done_file: str,
    config_paths: str,
    num_parallel: int,
    job_label: str,
    policy_vllm_url: str,
    judge_vllm_url: str,
    chunk_id: int,
    num_chunks: int,
) -> str:
    """Render shell variable declarations consumed by every later segment.

    Every path-typed value flows through ``shlex.quote`` so a path or
    label containing whitespace, ``$``, ``"``, or backticks cannot break
    the rendered script.  Integer fields are emitted bare since
    ``shlex.quote`` would add unnecessary quotes around their string form.

    URL fields (``VLLM_URL`` and ``JUDGE_URL``) flow through
    :func:`_bash_dq` instead of :func:`shlex.quote` because their values
    are URL templates assembled from
    :meth:`ServerScript.hostname_ref` and dynamic port shell variables
    -- e.g.
    ``http://${SLURM_MASTER_NODE_HET_GROUP_0:-localhost}:$POLICY_PORT/v1``.
    ``shlex.quote`` would single-quote the string and freeze the
    bash variables as literal characters, which then leak through
    ``$VLLM_URL`` into the ng_run CLI.  OmegaConf parses the leaked
    ``${SLURM_MASTER_NODE_HET_GROUP_0:-localhost}`` as an interpolation
    and raises ``UnsupportedInterpolationType`` (silent rollout
    failure).  Double-quoting preserves bash expansion at assignment
    time; the resolved URL flows correctly into both ``wait_for_server``
    and the ``ng_run +policy_model.responses_api_models.vllm_model.base_url``
    override.
    """
    return (
        "set -e\n"
        "\n"
        f"OUTPUT_DIR={shlex.quote(output_dir)}\n"
        f"GYM_PATH={shlex.quote(gym_path)}\n"
        # Where ng_run looks for per-component venvs (consumed by build_ng_run_invocation).
        # Defaults to GYM_PATH (== Gym's PARENT_DIR default); the CPU nemo-gym image
        # overrides it to /opt/gym-venvs to reuse the baked venvs.
        f"UV_VENV_DIR={shlex.quote(uv_venv_dir or gym_path)}\n"
        f"MODEL_PATH={shlex.quote(model_path)}\n"
        f"AGENT_NAME={shlex.quote(agent_name)}\n"
        f"INPUT_DATA={shlex.quote(input_data)}\n"
        f"OUTPUT_FILE={shlex.quote(output_file)}\n"
        f"DONE_FILE={shlex.quote(done_file)}\n"
        f"CONFIG_PATHS={shlex.quote(config_paths)}\n"
        f"NUM_PARALLEL={num_parallel}\n"
        f"JOB_LABEL={shlex.quote(job_label)}\n"
        f"VLLM_URL={_bash_dq(policy_vllm_url)}\n"
        f"JUDGE_URL={_bash_dq(judge_vllm_url)}\n"
        f"CHUNK_ID={chunk_id}\n"
        f"NUM_CHUNKS={num_chunks}\n"
    )


def _build_setup_segment(is_ray: bool = False, stop_sentinel: str = "") -> str:
    """Render mkdir, port-finder definition, cleanup trap, and wait_for_server.

    The cleanup trap is the load-bearing crash-safety hook; see the
    self-heal segment for the matching recovery side.
    """
    # On the Ray Jobs backend the policy/judge vLLM serve runs as a
    # SEPARATE Ray job that nothing reaps on a *successful* rollout (on Slurm the
    # het-group allocation teardown does it).  The client touches a shared-FS
    # stop sentinel UNCONDITIONALLY (success AND failure) so the paired serve's
    # watcher sees it, kills vLLM, and exits 0 — releasing GPUs and letting the
    # driver advance past the collect_rollouts gate.  Ray-only: on Slurm the
    # branch is never built, so the emitted command is byte-identical.
    stop_sentinel_touch = (
        f'    echo "[Cleanup] Touching server-stop sentinel {stop_sentinel}"\n'
        f'    touch "{stop_sentinel}" 2>&- || true\n'
        if is_ray and stop_sentinel
        else ""
    )
    if is_ray:
        failure_cleanup = (
            '        echo "[nvflow] Client exited with code $_nvflow_exit — terminating process group"\n'
            "        kill 0 2>&- || true\n"
        )
    else:
        failure_cleanup = (
            '        echo "[nvflow] Client exited with code $_nvflow_exit — cancelling job ${SLURM_JOB_ID}"\n'
            '        scancel "${SLURM_JOB_ID}" 2>&- || kill 0 2>&- || true\n'
        )
    return (
        "\n"
        'mkdir -p "$OUTPUT_DIR/logs"\n'
        "\n" + SHELL_FIND_FREE_PORT + "\n"
        'NG_RUN_PID=""\n'
        "\n"
        "cleanup() {\n"
        "    local _nvflow_exit=$?\n"
        '    echo ""\n'
        '    echo "[Cleanup] Shutting down NeMo-Gym servers ..."\n'
        "    # Suppress stderr via ``2>&-`` (close fd) instead of\n"
        "    # ``2>/dev/null`` so the cleanup trap stays quiet even if\n"
        "    # the container's /dev/null disappeared mid-script -- which\n"
        "    # is exactly what happened in the Nemotron-Nano smoke run\n"
        "    # where pyxis tore down the container while bash was still\n"
        "    # in the cleanup path, producing ``/dev/null: No such file\n"
        "    # or directory`` noise on top of the original failure.\n"
        '    [ -n "$NG_RUN_PID" ] && kill $NG_RUN_PID 2>&- && wait $NG_RUN_PID 2>&- || true\n'
        + stop_sentinel_touch
        + "    # Best-effort merge of .prev into -async.  On success, finalize already\n"
        "    # merged and removed .prev so this block is a no-op.  On failure/kill,\n"
        "    # this is a first attempt; the self-heal at next startup is the guarantee.\n"
        "    # Chain with && so .prev is NEVER deleted unless the merge succeeds.\n"
        '    if [ -n "$ASYNC_FILE" ] && [ -f "$ASYNC_FILE.prev" ] && [ "${PREV_MERGED:-0}" -eq 0 ]; then\n'
        '        echo "[Cleanup] Restoring previous results into -async for resume ..."\n'
        '        cat "$ASYNC_FILE.prev" > "$ASYNC_FILE.restored" \\\n'
        '            && { [ ! -f "$ASYNC_FILE" ] || cat "$ASYNC_FILE" >> "$ASYNC_FILE.restored"; } \\\n'
        '            && mv -f "$ASYNC_FILE.restored" "$ASYNC_FILE" \\\n'
        '            && rm -f "$ASYNC_FILE.prev" \\\n'
        '            || echo "[Cleanup] WARNING: merge failed — self-heal will recover on next start"\n'
        "    fi\n"
        "    if [ $_nvflow_exit -ne 0 ]; then\n" + failure_cleanup + "    fi\n"
        "}\n"
        "trap cleanup EXIT\n"
        "\n" + SHELL_WAIT_FOR_SERVER + "\n"
    )


def _build_banner_segment() -> str:
    """Render the human-readable header echoed at job start."""
    return (
        'echo "============================================================"\n'
        'echo "Rollout Collection  [$JOB_LABEL]"\n'
        'echo "============================================================"\n'
        'echo "Model:       $MODEL_PATH"\n'
        'echo "Agent:       $AGENT_NAME"\n'
        'echo "Input data:  $INPUT_DATA"\n'
        'echo "Output file: $OUTPUT_FILE"\n'
        'echo "Policy URL:  $VLLM_URL"\n'
        '[ -n "$JUDGE_URL" ] && echo "Judge URL:   $JUDGE_URL"\n'
        'echo "============================================================"\n'
    )


def _build_done_check_segment() -> str:
    """Render the early-exit guard for jobs whose chunk already completed.

    When ``dependent_jobs > 0`` Slurm pre-submits a chain; later runs in
    the chain hit this guard and exit 0 without redoing work.
    """
    return (
        'if [ -f "$DONE_FILE" ]; then\n'
        '    echo "Chunk already complete (.done exists) — skipping."\n'
        "    exit 0\n"
        "fi\n"
    )


def _build_step1_wait_segment(policy_vllm_url: str, judge_vllm_url: str) -> str:
    """Render the "[Step 1/3]" banner + ``wait_for_server`` invocations."""
    wait_for_vllm = _build_vllm_wait_snippet(policy_vllm_url, judge_vllm_url)
    return '\necho ""\necho "[Step 1/3] Waiting for vLLM servers ..."\n' + wait_for_vllm + "\n"


def _build_chunk_slice_segment(max_num_samples: int) -> str:
    """Render the runtime head|tail slice that gives this job its chunk.

    No physical pre-splitting on the launcher — keeps the launcher
    lightweight and filesystem-agnostic.  The ``if [ $NUM_CHUNKS -gt 1 ]``
    runtime guard suppresses slicing in the single-chunk case.
    """
    return (
        'CHUNK_INPUT=""\n'
        "if [ $NUM_CHUNKS -gt 1 ]; then\n"
        '    echo ""\n'
        '    echo "[Step 1a] Extracting chunk slice ..."\n'
        '    TOTAL_LINES=$(wc -l < "$INPUT_DATA")\n'
        "    EFFECTIVE=$TOTAL_LINES\n"
        f"    MAX_SAMPLES={max_num_samples}\n"
        "    if [ $MAX_SAMPLES -gt 0 ] && [ $MAX_SAMPLES -lt $TOTAL_LINES ]; then\n"
        "        EFFECTIVE=$MAX_SAMPLES\n"
        "    fi\n"
        "    CHUNK_SIZE=$(( (EFFECTIVE + NUM_CHUNKS - 1) / NUM_CHUNKS ))\n"
        "    START_LINE=$(( CHUNK_ID * CHUNK_SIZE + 1 ))\n"
        "    END_LINE=$(( (CHUNK_ID + 1) * CHUNK_SIZE ))\n"
        "    [ $END_LINE -gt $EFFECTIVE ] && END_LINE=$EFFECTIVE\n"
        '    CHUNK_INPUT="$OUTPUT_DIR/chunk_input_chunk$CHUNK_ID.jsonl"\n'
        '    head -n $END_LINE "$INPUT_DATA" | tail -n +$START_LINE > "$CHUNK_INPUT"\n'
        '    echo "  Chunk $CHUNK_ID/$NUM_CHUNKS: lines $START_LINE-$END_LINE ($((END_LINE - START_LINE + 1)) samples)"\n'
        '    INPUT_DATA="$CHUNK_INPUT"\n'
        "fi\n"
    )


def _build_selfheal_segment() -> str:
    """Render the three crash-recovery cases run before resume.

    Cases (see body for detail):
      A. Output file present but ``.done`` missing → restore to ``-async``.
      B. Orphaned ``.prev`` → merge into ``-async``.
      C. Orphaned ``.healed`` / ``.restored`` / ``.merged`` temp files → rm.

    The whole block is wrapped in a subshell so a recovery failure under
    ``set -e`` does not abort the job; the worst case is some redone work.
    """
    return (
        'ASYNC_FILE="$OUTPUT_FILE-async"\n'
        "(\n"
        "  # Case A: output exists without .done → restore to -async\n"
        '  if [ -f "$OUTPUT_FILE" ] && [ ! -f "$DONE_FILE" ]; then\n'
        '      echo "[Self-heal] Output file exists without .done — restoring to -async ..."\n'
        '      mv -f "$OUTPUT_FILE" "$ASYNC_FILE"\n'
        "  fi\n"
        "\n"
        "  # Case B: orphaned .prev → merge into -async\n"
        '  if [ -f "$ASYNC_FILE.prev" ]; then\n'
        '      echo "[Self-heal] Found orphaned .prev — merging into -async ..."\n'
        '      PREV_LINES=$(wc -l < "$ASYNC_FILE.prev")\n'
        "      ASYNC_LINES=0\n"
        '      [ -f "$ASYNC_FILE" ] && ASYNC_LINES=$(wc -l < "$ASYNC_FILE")\n'
        '      cat "$ASYNC_FILE.prev" > "$ASYNC_FILE.healed"\n'
        '      [ -f "$ASYNC_FILE" ] && cat "$ASYNC_FILE" >> "$ASYNC_FILE.healed"\n'
        '      mv -f "$ASYNC_FILE.healed" "$ASYNC_FILE" && rm -f "$ASYNC_FILE.prev"\n'
        '      MERGED_LINES=$(wc -l < "$ASYNC_FILE")\n'
        '      echo "  Recovered $PREV_LINES (prev) + $ASYNC_LINES (async) = $MERGED_LINES total rows"\n'
        "  fi\n"
        "\n"
        "  # Case C: clean up orphaned temp files from prior crash\n"
        '  rm -f "$ASYNC_FILE.healed" "$ASYNC_FILE.restored" "$ASYNC_FILE.merged"\n'
        ') || echo "[Self-heal] WARNING: recovery failed — continuing with available data"\n'
    )


def _build_resume_segment(num_chunks: int, max_num_samples: int, *, is_ray: bool = False) -> str:
    """Render the resume-filter step that skips already-completed rows.

    When chunked, the head|tail slice already truncated; pass 0 to
    resume_filter to avoid double-truncation.

    The early-exit branch (when all rows are already complete in
    ``-async``) explicitly cleans the orphan ``.prev`` and signals the
    cleanup trap via ``PREV_MERGED=1`` so the trap does NOT recreate
    ``-async`` from a stale ``.prev`` after the chunk is already
    finalized.  Without this hardening, a narrow race -- where
    self-heal Case B partially failed and left ``.prev`` on disk while
    ``-async`` still had all rows -- would have the trap re-materialise
    a stale ``-async`` orphan after ``.done`` was touched.  The
    ``.done`` marker still wins on the next run's done-check, so this
    was disk hygiene rather than a correctness bug, but defensive
    cleanup avoids the surprise.
    """
    resume_max = 0 if num_chunks > 1 else max_num_samples
    python_prefix = _python_module_prefix(is_ray)
    return (
        'REMAINING_INPUT="$OUTPUT_DIR/remaining_input_chunk$CHUNK_ID.jsonl"\n'
        "\n"
        f'if ! {python_prefix}python3 -m nvflow.lib.rl.resume_filter "$ASYNC_FILE" "$INPUT_DATA" "$REMAINING_INPUT" {resume_max}; then\n'
        '    echo "ERROR: resume_filter failed" >&2\n'
        "    exit 1\n"
        "fi\n"
        "\n"
        'if [ -f "$ASYNC_FILE" ] && [ ! -s "$REMAINING_INPUT" ]; then\n'
        '    echo "All rows already completed in -async -- finalizing."\n'
        '    cp -f "$ASYNC_FILE" "$OUTPUT_FILE"\n'
        '    touch "$DONE_FILE"\n'
        "    PREV_MERGED=1\n"
        '    rm -f "$ASYNC_FILE" "$ASYNC_FILE.prev"\n'
        '    echo "Done [$JOB_LABEL]."\n'
        "    exit 0\n"
        "fi\n"
    )


def _build_ng_run_segment(judge_ng_run_overrides: str, is_ray: bool = False) -> str:
    """Render "[Step 2/3]": background ng_run + wait_for_server probe.

    WORKAROUND(port-toctou): allocate the head-server port here (not in
    setup) to minimise the window between ``find_free_port()`` and ng_run
    binding to it (the retry loop in :func:`build_ng_run_invocation`
    re-allocates a fresh port per attempt).

    The shared invocation block is delegated to
    :func:`build_ng_run_invocation` so any future change to the ng_run
    arg shape applies to both ``collect_rollouts`` and
    ``compute_rewards`` atomically; *is_ray* adds the
    ``ray_head_node_address`` attach override there.
    """
    return '\nHEAD_SERVER_PORT=$(find_free_port)\n\ncd "$GYM_PATH"\n\n' + build_ng_run_invocation(
        step_label="[Step 2/3]",
        policy_base_url="$VLLM_URL",
        policy_model="$MODEL_PATH",
        judge_ng_run_overrides=judge_ng_run_overrides,
        include_port_range=True,
        is_ray=is_ray,
        # The retry loop includes random startup jitter and exists to avoid
        # collisions between colocated Ray Jobs.  A Slurm allocation has the
        # established one-shot launch and must remain byte-for-byte unchanged.
        retry=is_ray,
    )


def _build_collect_segment(responses_create_params: dict | None) -> str:
    """Render "[Step 3/3]": `gym eval run --no-serve` retry loop.

    Retries cushion the vLLM tokenizer race condition ("Already borrowed")
    that crashes the client on the initial request burst; ``resume_from_cache``
    ensures retries skip already-completed samples.
    """
    rcp_extras = "".join(
        f" \\\n        +responses_create_params.{k}={v}"
        for k, v in (responses_create_params or {}).items()
    )
    return (
        "\n"
        'echo ""\n'
        'echo "[Step 3/3] Collecting rollouts ..."\n'
        "# Back up previous partial results before `gym eval run` clears the file.\n"
        'ASYNC_BACKUP=""\n'
        "PREV_MERGED=0\n"
        'if [ -s "$ASYNC_FILE" ]; then\n'
        '    ASYNC_BACKUP="$ASYNC_FILE.prev"\n'
        '    cp "$ASYNC_FILE" "$ASYNC_BACKUP"\n'
        "fi\n"
        "# Ensure clean slate for first attempt.  Prior data is safe in .prev.\n"
        "# Stale materialized_inputs from a prior Slurm job would cause\n"
        "# resume_from_cache to load wrong task indexes.\n"
        'rm -f "$ASYNC_FILE"\n'
        'MATERIALIZED="$(dirname "$ASYNC_FILE")/$(basename "$ASYNC_FILE" .jsonl-async)_materialized_inputs.jsonl"\n'
        'rm -f "$MATERIALIZED"\n'
        "# Retry loop: the vLLM tokenizer race condition (RuntimeError: Already\n"
        "# borrowed) can crash the client on the initial request burst.  Retrying\n"
        "# after a short delay shifts the timing and almost always succeeds.\n"
        "# resume_from_cache=true ensures retries skip completed samples.\n"
        "_NVFLOW_MAX_RETRIES=3\n"
        "_NVFLOW_RETRY_DELAY=15\n"
        "_NVFLOW_EXIT=0\n"
        "for _attempt in $(seq 1 $_NVFLOW_MAX_RETRIES); do\n"
        "    # Guard: truncate corrupted last line from SIGKILL mid-write\n"
        '    if [ -f "$ASYNC_FILE" ] && [ -s "$ASYNC_FILE" ]; then\n'
        '        if [ "$(tail -c 1 "$ASYNC_FILE" | xxd -p)" != "0a" ]; then\n'
        '            head -n -1 "$ASYNC_FILE" > "$ASYNC_FILE.truncated" \\\n'
        '                && mv -f "$ASYNC_FILE.truncated" "$ASYNC_FILE" \\\n'
        '                || rm -f "$ASYNC_FILE.truncated"\n'
        '            echo "[nvflow] Truncated corrupted last line from $ASYNC_FILE"\n'
        "        fi\n"
        "    fi\n"
        "    set +e\n"
        "    gym eval run --no-serve \\\n"
        "        ${AGENT_NAME:++agent_name=$AGENT_NAME} \\\n"
        "        +input_jsonl_fpath=$REMAINING_INPUT \\\n"
        "        +output_jsonl_fpath=$ASYNC_FILE \\\n"
        "        +num_repeats=1 \\\n"
        "        +resume_from_cache=true \\\n"
        "        +num_samples_in_parallel=$NUM_PARALLEL \\\n"
        "        +head_server.host=127.0.0.1 \\\n"
        "        +head_server.port=$HEAD_SERVER_PORT" + rcp_extras + "\n"
        "    _NVFLOW_EXIT=$?\n"
        "    set -e\n"
        "    [ $_NVFLOW_EXIT -eq 0 ] && break\n"
        "    # F6: `gym eval run` can exit non-zero AFTER all rollouts\n"
        "    # have already been written to ASYNC_FILE -- the gym's\n"
        "    # post-collection aggregate_metrics call returned 500 in the\n"
        "    # Nemotron-Nano smoke run, killing the process at 99% even\n"
        "    # though all 1000 rollouts were on disk.  Retrying in that\n"
        "    # state is wasteful (re-runs everything) and dangerous if the\n"
        "    # container FS is being torn down (we hit\n"
        "    # ``/usr/bin/sleep: No such file or directory`` then).  If\n"
        "    # ASYNC_FILE has at least as many rows as REMAINING_INPUT, we\n"
        "    # already have what we need; declare success and let finalize\n"
        "    # do its job.\n"
        '    if [ -f "$ASYNC_FILE" ] && [ -s "$ASYNC_FILE" ]; then\n'
        '        _async_rows=$(wc -l < "$ASYNC_FILE" 2>&- || echo 0)\n'
        '        _input_rows=$(wc -l < "$REMAINING_INPUT" 2>&- || echo 0)\n'
        '        if [ "$_input_rows" -gt 0 ] && [ "$_async_rows" -ge "$_input_rows" ]; then\n'
        '            echo "[nvflow] gym eval run exited $_NVFLOW_EXIT but '
        "$ASYNC_FILE has $_async_rows/$_input_rows rows -- treating as complete "
        '(post-collection error in gym, rollouts intact)."\n'
        "            _NVFLOW_EXIT=0\n"
        "            break\n"
        "        fi\n"
        "    fi\n"
        "    if [ $_attempt -lt $_NVFLOW_MAX_RETRIES ]; then\n"
        '        echo "[nvflow] gym eval run exited $_NVFLOW_EXIT'
        " (attempt $_attempt/$_NVFLOW_MAX_RETRIES)."
        ' Retrying in ${_NVFLOW_RETRY_DELAY}s ..."\n'
        "        sleep $_NVFLOW_RETRY_DELAY\n"
        "        _NVFLOW_RETRY_DELAY=$((_NVFLOW_RETRY_DELAY * 2))\n"
        "    fi\n"
        "done\n"
        "if [ $_NVFLOW_EXIT -ne 0 ]; then\n"
        '    echo "[nvflow] gym eval run failed after'
        ' $_NVFLOW_MAX_RETRIES attempts."\n'
        "    exit $_NVFLOW_EXIT\n"
        "fi\n"
    )


def _build_finalize_segment() -> str:
    """Render the crash-safe finalize sequence.

    Order is load-bearing for crash safety:
      1. Merge ``.prev`` + ``-async`` into ``-async``  (all results in one file)
      2. ``cp -async → output``  (cp, not mv -- keeps ``-async`` as backup)
      3. ``touch .done``  (marks completion)
      4. Cleanup ``-async``, ``.prev``, temps  (safe -- ``.done`` exists)

    Reordering breaks self-heal Case A (output without .done) and the
    resume-filter precondition that ``-async`` is the source of truth.
    """
    return (
        "\n"
        "# Merge previous partial results with new results.\n"
        'if [ -n "$ASYNC_BACKUP" ] && [ -f "$ASYNC_BACKUP" ]; then\n'
        '    cat "$ASYNC_BACKUP" "$ASYNC_FILE" > "$ASYNC_FILE.merged"\n'
        '    mv -f "$ASYNC_FILE.merged" "$ASYNC_FILE"\n'
        "    PREV_MERGED=1\n"
        '    rm -f "$ASYNC_BACKUP"\n'
        "fi\n"
        'cp -f "$ASYNC_FILE" "$OUTPUT_FILE"\n'
        'touch "$DONE_FILE"\n'
        "# Safe to clean up — .done exists, chunk won't be rescheduled.\n"
        'rm -f "$ASYNC_FILE" "$REMAINING_INPUT"\n'
        '[ -n "$CHUNK_INPUT" ] && rm -f "$CHUNK_INPUT"\n'
        'echo "Done [$JOB_LABEL]. Cleanup via trap."\n'
    )


def _build_client_cmd(
    *,
    output_dir: str,
    gym_path: str,
    uv_venv_dir: str = "",
    model_path: str,
    agent_name: str,
    input_data: str,
    output_file: str,
    done_file: str,
    config_paths: str,
    num_parallel: int,
    job_label: str,
    policy_vllm_url: str,
    judge_vllm_url: str = "",
    judge_ng_run_overrides: str,
    max_num_samples: int = 0,
    chunk_id: int = 0,
    num_chunks: int = 1,
    responses_create_params: dict | None = None,
    is_ray: bool = False,
    stop_sentinel: str = "",
) -> str:
    """Build the rollout collection bash script.

    Generated script structure (concatenation order is load-bearing):
      1.  Variables, setup (cleanup trap), banner
      2.  Done-check early exit
      3.  Step 1: wait for vLLM servers
      4.  Step 1a: chunk slice (runtime head|tail)
      5.  Step 1b: self-heal (recover .prev / partial finalize from prior crash)
      6.  Step 1c: resume filter (skip already-completed rows)
      7.  Step 2: start NeMo-Gym servers via ``gym env start``
      8.  Step 3: collect rollouts via ``gym eval run --no-serve``
      9.  Finalize: merge partials, cp→output, touch .done, cleanup temps

    Each step is rendered by a dedicated ``_build_*_segment`` helper.
    Snapshot tests in ``tests/test_rollout.py`` pin the byte-exact output
    of this composition; do not reorder segments without refreshing them.
    """
    return (
        _build_variables_segment(
            output_dir=output_dir,
            gym_path=gym_path,
            uv_venv_dir=uv_venv_dir,
            model_path=model_path,
            agent_name=agent_name,
            input_data=input_data,
            output_file=output_file,
            done_file=done_file,
            config_paths=config_paths,
            num_parallel=num_parallel,
            job_label=job_label,
            policy_vllm_url=policy_vllm_url,
            judge_vllm_url=judge_vllm_url,
            chunk_id=chunk_id,
            num_chunks=num_chunks,
        )
        + _build_setup_segment(is_ray=is_ray, stop_sentinel=stop_sentinel)
        + _build_banner_segment()
        + _build_done_check_segment()
        + _build_step1_wait_segment(policy_vllm_url, judge_vllm_url)
        + _build_chunk_slice_segment(max_num_samples)
        + _build_selfheal_segment()
        + _build_resume_segment(num_chunks, max_num_samples, is_ray=is_ray)
        + _build_ng_run_segment(judge_ng_run_overrides, is_ray=is_ray)
        + _build_collect_segment(responses_create_params)
        + _build_finalize_segment()
    )


def _build_merge_cmd(
    *,
    gym_path: str,
    merged_file: str,
    analysis_dir: str,
    seed_label: str,
    num_chunks: int,
    chunk_file_pattern: str,
    merged_done_file: str,
    analyze_module: str,
    enrich_module: str,
    input_data: str,
    keep_chunk_files: bool = False,
    is_ray: bool = False,
) -> str:
    """Build the chunk-merge + enrich + analyze bash script.

    Generated script structure:
      1. Concatenate per-chunk rollout files into a single merged file
      2. Enrich merged rollouts with input metadata
      3. Analyze rollouts (accuracy, token stats, etc.)

    Every literal path or label is shlex-quoted so values containing
    whitespace, ``$``, or shell metacharacters cannot break the rendered
    script.  ``chunk_file_pattern`` is an exception: it carries a literal
    ``$i`` that bash must expand inside the for-loop, so quoting it
    would defeat the substitution.
    """
    # -- Shell variables --------------------------------------------------
    variables = (
        "set -e\n"
        "\n"
        f"MERGED_FILE={shlex.quote(merged_file)}\n"
        f"MERGED_DONE={shlex.quote(merged_done_file)}\n"
        f"ANALYSIS_DIR={shlex.quote(analysis_dir)}\n"
        f"SEED_LABEL={shlex.quote(seed_label)}\n"
        f"NUM_CHUNKS={num_chunks}\n"
        f"INPUT_DATA={shlex.quote(input_data)}\n"
    )

    # -- Idempotency: skip if this seed's merge already completed ----------
    # A prior successful merge consumes the chunks (deletes chunk_*.jsonl but
    # keeps their .done markers, which _get_remaining_jobs relies on) and writes
    # the merged output + its own .done marker.  On a resume into that dir the
    # Step-1 chunk-file check would abort ("chunk .done present but file gone");
    # detect the completed merge here and skip cleanly so collect_rollouts is
    # resumable after a crash/disconnect.
    idempotency = (
        "\n"
        'if [ -f "$MERGED_DONE" ] && [ -s "$MERGED_FILE" ]; then\n'
        '    echo "Merge already complete for $SEED_LABEL (output + .done present) — skipping."\n'
        "    exit 0\n"
        "fi\n"
    )

    # -- Step 1: Merge chunks (atomic — write to .tmp, then mv) ------------
    chunk_done_pattern = f"{chunk_file_pattern}.done"
    step1_merge = (
        "\n"
        'echo "============================================================"\n'
        'echo "Merge Rollout Chunks  [$SEED_LABEL]"\n'
        'echo "============================================================"\n'
        "\n"
        "# Clean up stale temp file from a prior crashed merge\n"
        'rm -f "$MERGED_FILE.tmp"\n'
        "\n"
        "# Precondition: ALL chunk .done markers must exist.  A missing\n"
        "# .done means upstream rollout never finalised (job failed before\n"
        "# the cp/touch finalize step, or output was deleted).  Fail loudly\n"
        "# so Slurm marks the merge job FAILED and an operator notices --\n"
        "# silently exit-0'ing here is what produced the silent-success\n"
        "# cascade where aggregate runs on N-1 seeds and the pipeline\n"
        "# reports COMPLETED 0:0 despite missing training data.\n"
        "for i in $(seq 0 $((NUM_CHUNKS - 1))); do\n"
        f'    CHUNK_DONE="{chunk_done_pattern}"\n'
        '    if [ ! -f "$CHUNK_DONE" ]; then\n'
        '        echo "ERROR: chunk $i .done missing for [$SEED_LABEL] -- upstream rollout failed before finalize. Aborting merge."\n'
        "        exit 1\n"
        "    fi\n"
        "done\n"
        "\n"
        'echo "[Step 1/3] Merging chunk files ..."\n'
        '> "$MERGED_FILE.tmp"\n'
        "for i in $(seq 0 $((NUM_CHUNKS - 1))); do\n"
        f'    CHUNK_FILE="{chunk_file_pattern}"\n'
        '    if [ ! -f "$CHUNK_FILE" ] || [ ! -s "$CHUNK_FILE" ]; then\n'
        '        echo "ERROR: chunk $i .done exists but file missing/empty — aborting."\n'
        '        rm -f "$MERGED_FILE.tmp"\n'
        "        exit 1\n"
        "    fi\n"
        '    LINES=$(wc -l < "$CHUNK_FILE")\n'
        '    echo "  Chunk $i: $LINES lines"\n'
        '    cat "$CHUNK_FILE" >> "$MERGED_FILE.tmp"\n'
        "done\n"
        "\n"
        'TOTAL=$(wc -l < "$MERGED_FILE.tmp")\n'
        'echo "  Merged total: $TOTAL lines"\n'
        'if [ "$TOTAL" -eq 0 ]; then\n'
        '    echo "ERROR: All chunks present but 0 lines merged."\n'
        '    rm -f "$MERGED_FILE.tmp"\n'
        "    exit 1\n"
        "fi\n"
        "\n"
        "# Atomic replace — old merged file untouched until this point\n"
        'mv -f "$MERGED_FILE.tmp" "$MERGED_FILE"\n'
    )

    python_prefix = _python_module_prefix(is_ray)

    # -- Step 2: Enrich with input metadata -------------------------------
    step2_enrich = (
        "\n"
        'echo ""\n'
        'echo "[Step 2/3] Enriching rollouts with input metadata ..."\n'
        f"{python_prefix}python3 -m {shlex.quote(enrich_module)} \\\n"
        '    "$INPUT_DATA" \\\n'
        '    "$MERGED_FILE"\n'
    )

    # -- Step 3: Analyze rollouts -----------------------------------------
    step3_analyze = (
        "\n"
        'echo ""\n'
        'echo "[Step 3/3] Analyzing rollouts ..."\n'
        f"{python_prefix}python3 -m {shlex.quote(analyze_module)} \\\n"
        '    "$MERGED_FILE" \\\n'
        f'    "$ANALYSIS_DIR"\n'
    )

    # -- Cleanup: mark done, then remove chunk output files -----------------
    # Order: touch .done FIRST, then delete chunk data.  This guarantees the
    # merged file is the verified complete copy before any source data is
    # removed.  Keep chunk .done markers — _get_remaining_jobs relies on them.
    chunk_cleanup = (
        "# Safe cleanup: delete chunk data files only (keep .done markers).\n"
        "for i in $(seq 0 $((NUM_CHUNKS - 1))); do\n"
        f'    CHUNK_FILE="{chunk_file_pattern}"\n'
        '    rm -f "$CHUNK_FILE"\n'
        "done\n"
        if not keep_chunk_files
        else (
            "# keep_chunk_files=true: retaining per-chunk raw files so a later\n"
            "# re-merge over a larger num_chunks (in-place cumulative growth)\n"
            "# can reconstruct the full merged output. Disk grows accordingly.\n"
            'echo "[merge] keep_chunk_files=true -- retaining per-chunk raw files."\n'
        )
    )
    cleanup = (
        "\n"
        f"touch {shlex.quote(merged_done_file)}\n"
        "\n" + chunk_cleanup + 'echo "Done [$SEED_LABEL]."\n'
        'echo ""\n'
        'echo "To browse rollouts interactively (in the nemo-gym container):"\n'
        'echo "  export PATH=/opt/gym-cli-venv/bin:$PATH && ng_viewer +jsonl_fpath=$MERGED_FILE"\n'
    )

    return variables + idempotency + step1_merge + step2_enrich + step3_analyze + cleanup


def build_aggregate_cmd(
    *,
    rollout_dir: str,
    aggregate_module: str,
    difficulty_filename: str = "difficulty.jsonl",
    expected_seeds: int | None = None,
    is_ray: bool = False,
) -> str:
    """Build the bash command for the cross-seed aggregation job.

    When ``expected_seeds`` is provided, the underlying CLI validates
    that exactly that many ``output-rs*.jsonl`` files are present and
    raises ``RuntimeError`` otherwise.  This is the second line of
    defense against the silent-success cascade: the cluster's default
    Slurm dep type is ``afterany`` (see
    ``cluster_configs/template-slurm.yaml``), so a FAILED upstream
    merge does NOT stop aggregate from running.  Pass
    ``num_random_seeds`` from the rollout caller to surface missing
    seeds as a loud failure instead of silently shrinking
    ``num_seeds`` in ``metrics.json``.
    """
    expected_arg = f" --expected-seeds {expected_seeds}" if expected_seeds is not None else ""
    python_prefix = _python_module_prefix(is_ray)
    return (
        "set -e\n"
        'echo "Cross-Seed Aggregation (pass@k)"\n'
        f"{python_prefix}python3 -m {aggregate_module} \\\n"
        f'    "{rollout_dir}" \\\n'
        f'    "{rollout_dir}/aggregate" \\\n'
        f'    --output_filename "{difficulty_filename}"{expected_arg}\n'
        f'echo "Done. Results in {rollout_dir}/aggregate/"\n'
    )


def build_filter_cmd(
    *,
    output_dir: str,
    difficulty_dir: str,
    filter_module: str,
    train_data: str,
    validation_data: str,
    min_reward_std: float = 1e-6,
    policy_model: str = "",
    judge_model: str = "",
    train_filename: str = "train.jsonl",
    val_filename: str = "validation.jsonl",
    difficulty_filename: str = "difficulty.jsonl",
    report_filename: str = "filter_report.json",
    is_ray: bool = False,
) -> str:
    python_prefix = _python_module_prefix(is_ray)
    cmd = (
        "set -e\n"
        'echo "Filter Training Data (reward-variance difficulty)"\n'
        f"{python_prefix}python3 -m {filter_module} \\\n"
        f'    "{train_data}" \\\n'
        f'    "{difficulty_dir}/aggregate/{difficulty_filename}" \\\n'
        f'    "{output_dir}" \\\n'
        f"    --min-reward-std {min_reward_std} \\\n"
        f'    --train-filename "{train_filename}" \\\n'
        f'    --val-filename "{val_filename}" \\\n'
        f'    --report-filename "{report_filename}"'
    )
    if validation_data:
        cmd += f' \\\n    --validation-data "{validation_data}"'
    if policy_model:
        cmd += f' \\\n    --policy-model "{policy_model}"'
    if judge_model:
        cmd += f' \\\n    --judge-model "{judge_model}"'
    cmd += "\n"
    cmd += f'echo "Done. Filtered data in {output_dir}/"\n'
    return cmd


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


@dataclass
class _RolloutParams:
    """Parsed rollout configuration."""

    output_dir: str
    gym_path: str
    uv_venv_dir: str
    client_container: str
    postprocess_container: str
    vllm_container: str
    installation_command: str | None
    input_data: str
    num_gpus: int
    num_parallel: int
    num_chunks: int
    num_random_seeds: int
    starting_seed: int
    rerun_done: bool
    dependent_jobs: int
    max_num_samples: int
    keep_chunk_files: bool
    responses_create_params: dict
    pcfg: dict
    jcfg: dict
    model_path: str
    rcfg: dict
    rcfg_with_env: dict
    agent_name: str
    config_paths_str: str
    judge_mode: str
    judge_ng_run_overrides: str
    need_policy_server: bool
    need_judge_server: bool
    filter_cfg: dict


def _parse_rollout_config(config: dict[str, Any]) -> _RolloutParams:
    """Unpack and validate rollout configuration."""
    output_dir = config["output_dir"]
    gym_path = config["gym_path"]
    # Per-component venv root passed to ng_run. Empty -> defaults to gym_path
    # (Gym's PARENT_DIR default); the CPU nemo-gym stages set /opt/gym-venvs.
    uv_venv_dir = config.get("gym_uv_venv_dir", "")
    client_container = config["container"]
    postprocess_container = config.get("postprocess_container", client_container)
    vllm_container = config.get("vllm_container", VLLM_CONTAINER)
    installation_command = config.get("installation_command")

    rcfg = config["rollout"]
    input_data = rcfg["input_data"]
    num_gpus = compute_num_gpus(rcfg, has_policy=True)
    num_parallel = rcfg.get("num_samples_in_parallel", 4)
    num_chunks = rcfg.get("num_chunks", 1)
    num_random_seeds = rcfg.get("num_random_seeds", 1)
    starting_seed = rcfg.get("starting_seed", 0)
    rerun_done = rcfg.get("rerun_done", False)
    dependent_jobs = rcfg.get("dependent_jobs", 0)
    max_num_samples = rcfg.get("max_num_samples") or 0
    # When true, the merge step keeps per-chunk raw files instead of deleting
    # them. Required for in-place cumulative growth (re-merge over a larger
    # num_chunks needs the original chunks). Default false preserves the
    # disk-saving delete behavior for recipes that don't grow in place.
    keep_chunk_files = rcfg.get("keep_chunk_files", False)
    responses_create_params = rcfg.get("responses_create_params") or {}

    pcfg = rcfg.get("policy_vllm") or {}
    jcfg = rcfg.get("judge_vllm") or {}
    model_path = pcfg["model_path"]

    rcfg_with_env = {**rcfg, "environments": config["environments"]}
    environment_name, env_inner_name, agent_name = get_env_from_environments(rcfg_with_env)
    rcfg_with_env["environment_name"] = environment_name
    rcfg_with_env["environment_inner_name"] = env_inner_name

    config_paths_str = build_config_paths_str(rcfg_with_env)
    judge_mode = determine_judge_mode(rcfg)
    judge_ng_run_overrides = build_judge_ng_run_overrides(rcfg_with_env, judge_mode)

    need_policy_server = not pcfg.get("base_url")
    need_judge_server = judge_mode == "local_vllm" and bool(jcfg.get("model_path"))

    filter_cfg = config.get("filter") or {}

    return _RolloutParams(
        output_dir=output_dir,
        gym_path=gym_path,
        uv_venv_dir=uv_venv_dir,
        client_container=client_container,
        postprocess_container=postprocess_container,
        vllm_container=vllm_container,
        installation_command=installation_command,
        input_data=input_data,
        num_gpus=num_gpus,
        num_parallel=num_parallel,
        num_chunks=num_chunks,
        num_random_seeds=num_random_seeds,
        starting_seed=starting_seed,
        rerun_done=rerun_done,
        dependent_jobs=dependent_jobs,
        max_num_samples=max_num_samples,
        keep_chunk_files=keep_chunk_files,
        responses_create_params=responses_create_params,
        pcfg=pcfg,
        jcfg=jcfg,
        model_path=model_path,
        rcfg=rcfg,
        rcfg_with_env=rcfg_with_env,
        agent_name=agent_name,
        config_paths_str=config_paths_str,
        judge_mode=judge_mode,
        judge_ng_run_overrides=judge_ng_run_overrides,
        need_policy_server=need_policy_server,
        need_judge_server=need_judge_server,
        filter_cfg=filter_cfg,
    )


# ---------------------------------------------------------------------------
# Progress estimation
# ---------------------------------------------------------------------------


def _estimate_progress(
    fs: LauncherFS,
    rollout_dir: str,
    seeds: list[int],
    chunk_ids: list[int],
    input_data: str,
    max_num_samples: int,
    num_chunks: int,
) -> str:
    """Return progress string like '~42% (15,000/35,000 rows)', or '' on failure.

    Best-effort and cosmetic: any exception is swallowed so a missing/unreadable
    file cannot break the launcher's status print.  Skipped entirely when
    launching off-cluster (``fs.remote``) -- counting rows in multi-GB -async
    files over an ssh tunnel is not worth the latency.
    """
    if fs.remote:
        return ""
    try:
        input_lines = fs.count_lines(input_data)
        effective = min(input_lines, max_num_samples) if max_num_samples else input_lines
        if effective <= 0:
            return ""
        chunk_size = (effective + num_chunks - 1) // num_chunks
        completed_rows = 0
        total_rows = 0
        for seed, chunk_id in [(s, c) for s in seeds for c in chunk_ids]:
            expected = max(0, min(chunk_size, effective - chunk_id * chunk_size))
            total_rows += expected
            fname = _output_filename(seed, chunk_id)
            if fs.exists(f"{rollout_dir}/{fname}.done"):
                completed_rows += expected
            else:
                for suffix in ["-async", "-async.prev"]:
                    async_path = f"{rollout_dir}/{fname}{suffix}"
                    if fs.exists(async_path):
                        completed_rows += fs.count_lines(async_path)
        if total_rows > 0:
            pct = completed_rows / total_rows * 100
            return f"~{pct:.0f}% ({completed_rows:,}/{total_rows:,} rows in -async files)"
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Job builders
# ---------------------------------------------------------------------------


def _build_collection_jobs(
    p: _RolloutParams,
    remaining: list[tuple[int, int]],
    cluster_config: dict,
    rollout_dir: str,
    expname: str,
    run_after: list[str] | None,
) -> tuple[list[dict], dict[int, list[dict]]]:
    """Build Slurm job specs for rollout collection.

    Returns ``(jobs, chunk_job_specs)`` where *chunk_job_specs* maps
    seed -> list of final job specs (for merge dependency wiring).
    """
    from nemo_skills.pipeline.utils.declarative import Command, CommandGroup, HardwareConfig

    jobs: list[dict] = []
    chunk_job_specs: dict[int, list[dict]] = {}
    job_log_dir = f"{p.output_dir}/logs"

    from nvflow.lib.sbatch import parse_extra_sbatch_args

    sbatch_kwargs = parse_extra_sbatch_args(cluster_config) or None
    use_ray_jobs = is_ray_backend(cluster_config)

    for seed, chunk_id in remaining:
        job_lbl = f"rs{seed}_chunk{chunk_id}"
        out_filename = _output_filename(seed, chunk_id)
        suffix_fmt = "-{dep_id}" if p.dependent_jobs > 0 else ""

        prev_job_spec = None
        for dep_id in range(p.dependent_jobs + 1):
            # Per-attempt nonce baked into BOTH the server (writer) and client
            # (reader) port-file paths.  Each distinct attempt (every dep_id in
            # the chain) gets its own id so a re-run never reads a stale port.
            # Critical under the Ray Jobs backend: the pre-provisioned head keeps one
            # constant $SLURM_JOB_ID for its whole life, so without this nonce
            # every rollout attempt would reuse the same port file and a client
            # could read a STALE port and hang on a dead server.  Must be a
            # Python-baked constant (not bash $$/$RANDOM): the writer and reader
            # are different processes/nodes and would otherwise never agree.
            is_ray = use_ray_jobs
            attempt_id = _new_ray_attempt_id() if is_ray else ""
            # Single per-attempt stop sentinel: the client touches it on EXIT and
            # both servers (policy + judge) poll the same path.  Ray-only — empty
            # on Slurm so the serve stays foreground and the emitted command is
            # byte-identical to the validated Slurm path.
            stop_sentinel = (
                _server_stop_sentinel(job_log_dir, job_lbl, attempt_id) if is_ray else ""
            )
            policy_script = (
                make_server_script(
                    p.pcfg,
                    cluster_config,
                    role="policy",
                    log_dir=job_log_dir,
                    job_label=job_lbl,
                    attempt_id=attempt_id,
                    is_ray=is_ray,
                    stop_sentinel=stop_sentinel,
                )
                if p.need_policy_server
                else None
            )
            judge_script = (
                make_server_script(
                    p.jcfg,
                    cluster_config,
                    role="judge",
                    log_dir=job_log_dir,
                    job_label=job_lbl,
                    attempt_id=attempt_id,
                    is_ray=is_ray,
                    stop_sentinel=stop_sentinel,
                )
                if p.need_judge_server
                else None
            )

            client_cmd = RolloutClientScript(
                policy_server=policy_script,
                judge_server=judge_script,
                policy_base_url=p.pcfg.get("base_url", ""),
                config=p.rcfg_with_env,
                judge_mode=p.judge_mode,
                judge_ng_run_overrides=p.judge_ng_run_overrides,
                output_dir=f"{rollout_dir}/rs{seed}",
                gym_path=p.gym_path,
                uv_venv_dir=p.uv_venv_dir,
                model_path=p.model_path,
                agent_name=p.agent_name,
                input_data=p.input_data,
                output_file=f"{rollout_dir}/{out_filename}",
                done_file=f"{rollout_dir}/{out_filename}.done",
                config_paths=p.config_paths_str,
                num_parallel=p.num_parallel,
                job_label=job_lbl,
                max_num_samples=p.max_num_samples,
                chunk_id=chunk_id,
                num_chunks=p.num_chunks,
                responses_create_params=p.responses_create_params,
                log_dir=job_log_dir,
                attempt_id=attempt_id,
                installation_command=p.installation_command,
                is_ray=is_ray,
                stop_sentinel=stop_sentinel,
            )

            job_deps: list = [prev_job_spec] if prev_job_spec is not None else (run_after or [])
            suffix = suffix_fmt.format(dep_id=dep_id)

            # Dual local servers -> het-group per server for dedicated GPUs.
            if judge_script is not None and policy_script is not None:
                policy_nodes = max(1, policy_script.num_nodes)
                judge_nodes = max(1, judge_script.num_nodes)
                primary_group = CommandGroup(
                    commands=[
                        Command(
                            script=policy_script,
                            container=p.vllm_container,
                            name=f"{job_lbl}_policy",
                        ),
                        Command(script=client_cmd, container=p.client_container, name=job_lbl),
                    ],
                    hardware=HardwareConfig(
                        num_gpus=p.pcfg.get("num_gpus", 0),
                        num_nodes=policy_nodes,
                        sbatch_kwargs=sbatch_kwargs,
                    ),
                    name=job_lbl,
                    log_dir=f"{p.output_dir}/logs",
                )
                judge_group = CommandGroup(
                    commands=[
                        Command(
                            script=judge_script,
                            container=p.vllm_container,
                            name=f"{job_lbl}_judge",
                        ),
                    ],
                    hardware=HardwareConfig(
                        num_gpus=p.jcfg.get("num_gpus", 0),
                        num_nodes=judge_nodes,
                        sbatch_kwargs=sbatch_kwargs,
                    ),
                    name=f"{job_lbl}_judge",
                    log_dir=f"{p.output_dir}/logs",
                )
                job_spec = {
                    "name": f"{expname}-rs{seed}-chunk{chunk_id}{suffix}",
                    "groups": [primary_group, judge_group],
                    "dependencies": job_deps,
                }
            else:
                components: list[Command] = []
                max_nodes = 1
                if policy_script is not None:
                    components.append(
                        Command(
                            script=policy_script,
                            container=p.vllm_container,
                            name=f"{job_lbl}_policy",
                        )
                    )
                    max_nodes = max(max_nodes, policy_script.num_nodes)
                if judge_script is not None:
                    components.append(
                        Command(
                            script=judge_script,
                            container=p.vllm_container,
                            name=f"{job_lbl}_judge",
                        )
                    )
                    max_nodes = max(max_nodes, judge_script.num_nodes)
                components.append(
                    Command(script=client_cmd, container=p.client_container, name=job_lbl)
                )
                cmd_group = CommandGroup(
                    commands=components,
                    hardware=HardwareConfig(
                        num_gpus=p.num_gpus,
                        num_nodes=max_nodes,
                        sbatch_kwargs=sbatch_kwargs,
                    ),
                    name=job_lbl,
                    log_dir=f"{p.output_dir}/logs",
                )
                job_spec = {
                    "name": f"{expname}-rs{seed}-chunk{chunk_id}{suffix}",
                    "group": cmd_group,
                    "dependencies": job_deps,
                }
            jobs.append(job_spec)
            prev_job_spec = job_spec

        # Merge depends on the LAST job in each chunk's chain.
        chunk_job_specs.setdefault(seed, []).append(prev_job_spec)

    return jobs, chunk_job_specs


def _build_merge_jobs(
    p: _RolloutParams,
    seeds: list[int],
    chunk_job_specs: dict[int, list[dict]],
    fs: LauncherFS,
    rollout_dir: str,
    expname: str,
    *,
    analyze_module: str,
    enrich_module: str,
    is_ray: bool = False,
) -> list[dict]:
    """Build one merge job per seed. Returns merge job specs."""
    from nemo_skills.pipeline.utils.declarative import Command, CommandGroup, HardwareConfig

    merge_job_specs: list[dict] = []
    for seed in seeds:
        seed_label = f"rs{seed}"
        merged_filename = _merged_filename(seed)

        if fs.exists(f"{rollout_dir}/{merged_filename}.done") and not p.rerun_done:
            continue

        chunk_pattern = f"{rollout_dir}/rs{seed}/chunk_$i.jsonl"

        merge_cmd_str = _build_merge_cmd(
            gym_path=p.gym_path,
            merged_file=f"{rollout_dir}/{merged_filename}",
            analysis_dir=f"{rollout_dir}/analysis_{seed_label}",
            seed_label=seed_label,
            num_chunks=p.num_chunks,
            chunk_file_pattern=chunk_pattern,
            merged_done_file=f"{rollout_dir}/{merged_filename}.done",
            analyze_module=analyze_module,
            enrich_module=enrich_module,
            input_data=p.input_data,
            keep_chunk_files=p.keep_chunk_files,
            is_ray=is_ray,
        )

        merge_cmd = Command(
            script=make_bash_script(
                merge_cmd_str,
                installation_command=_ray_postprocess_installation_command(
                    p.installation_command,
                    is_ray=is_ray,
                ),
            ),
            container=p.postprocess_container,
            name=f"merge-{seed_label}",
        )
        merge_group = CommandGroup(
            commands=[merge_cmd],
            hardware=HardwareConfig(num_gpus=0),
            name=f"merge-{seed_label}",
            log_dir=f"{p.output_dir}/logs",
        )

        seed_deps = chunk_job_specs.get(seed, [])
        merge_job_spec = {
            "name": f"{expname}-merge-{seed_label}",
            "group": merge_group,
            "dependencies": seed_deps if seed_deps else None,
        }
        merge_job_specs.append(merge_job_spec)

    return merge_job_specs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rollout(
    config: dict[str, Any],
    cluster: str,
    expname: str,
    run_after: list[str] | None = None,
    *,
    analyze_module: str = "",
    enrich_module: str = "",
    aggregate_module: str = "",
    filter_module: str = "",
) -> None:
    """Collect rollouts via NeMo-Gym, orchestrated through the nemo-skills Pipeline."""
    check_launcher_cwd()

    import nemo_skills.pipeline.utils as pipeline_utils
    from nemo_skills.pipeline.utils.declarative import (
        Command,
        CommandGroup,
        HardwareConfig,
        Pipeline,
    )

    p = _parse_rollout_config(config)

    cluster_config = pipeline_utils.get_cluster_config(cluster)
    use_ray_jobs = is_ray_backend(cluster_config)

    rollout_dir = f"{p.output_dir}/rollout"

    # Resume/status checks go through LauncherFS: local Path ops within the
    # cluster (unchanged behaviour), or over the ssh tunnel when launching
    # off-cluster.  No data I/O on the launcher — each Slurm job slices its
    # own chunk at runtime via head/tail (see _build_client_cmd chunk_slice).
    fs = LauncherFS(cluster_config)

    seeds = list(range(p.starting_seed, p.starting_seed + p.num_random_seeds))
    chunk_ids = list(range(p.num_chunks))

    if p.num_chunks > 1:
        console.detail("Input", f"{p.num_chunks} logical chunks (each job slices at runtime)")

    # -- Resume: find remaining (seed, chunk) pairs ---------------------
    if fs.exists(rollout_dir):
        remaining = _get_remaining_jobs(fs, rollout_dir, seeds, chunk_ids, p.rerun_done)
    else:
        remaining = [(s, c) for s in seeds for c in chunk_ids]
    skipped = len(seeds) * len(chunk_ids) - len(remaining)

    # Invalidate merge .done for any seed that has chunks to re-run,
    # so the merge step doesn't skip it with stale results.
    # Never delete the merged data file — it serves as backup until the
    # next merge atomically overwrites it (Fix 2: safe invalidation).
    rerun_seeds = {s for s, _ in remaining}
    fs.rm([f"{rollout_dir}/{_merged_filename(s)}.done" for s in rerun_seeds])

    # -- Progress estimate: count completed rows in partial -async files ---
    progress_msg = ""
    if remaining and fs.exists(rollout_dir):
        progress_msg = _estimate_progress(
            fs,
            rollout_dir,
            seeds,
            chunk_ids,
            p.input_data,
            p.max_num_samples,
            p.num_chunks,
        )

    console.status("Collecting rollouts (gym eval run --no-serve)")
    console.detail("Model", p.model_path)
    console.detail("Agent", p.agent_name)
    if p.pcfg.get("base_url"):
        console.detail("Policy vLLM", f"external ({p.pcfg['base_url']})")
    else:
        console.detail("Policy vLLM", f"local (GPUs={p.pcfg.get('num_gpus', 0)})")
    log_judge_details(console, p.rcfg, p.judge_mode)
    if p.need_policy_server and p.need_judge_server:
        console.detail(
            "Slurm GPUs/job",
            f"{p.num_gpus} (policy={p.pcfg.get('num_gpus', 0)} + judge={p.jcfg.get('num_gpus', 0)}, het-group)",
        )
    else:
        console.detail("Slurm GPUs/job", str(p.num_gpus))
    total_rollout_jobs = len(remaining) * (p.dependent_jobs + 1)
    chain_info = f" x {p.dependent_jobs + 1} chained" if p.dependent_jobs > 0 else ""
    console.detail(
        "Jobs",
        f"{total_rollout_jobs} to submit, {skipped} done | "
        f"{p.num_chunks} chunks x {p.num_random_seeds} seeds{chain_info}",
    )
    console.detail("Output", p.output_dir)
    if progress_msg:
        console.detail("Progress", progress_msg)
    console.blank()

    # A prior run may have completed every chunk (.done present, so `remaining`
    # is empty) yet never produced — or lost — a seed's merged output.  The
    # merged file `rollout/output-rs<seed>.jsonl` (synced up to
    # `<output_dir>/output-rs<seed>.jsonl` by analyze) is the sole producer of
    # the file downstream stages `cp`/parse.  If we return early here we skip
    # the merge job and leave that file missing, which is exactly what makes a
    # resumed genselect/evaluate fail with `cp: ... output-rs0.jsonl: No such
    # file`.  Detect seeds whose merge is incomplete so we still schedule it.
    if fs.exists(rollout_dir):
        merges_pending = any(
            not fs.exists(f"{rollout_dir}/{_merged_filename(s)}.done") for s in seeds
        )
    else:
        merges_pending = True

    if not remaining and not merges_pending and not p.filter_cfg:
        console.success("All rollout jobs already complete (use rerun_done to force).")
        return

    # -- Build Pipeline jobs ---------------------------------------------
    jobs, chunk_job_specs = _build_collection_jobs(
        p,
        remaining,
        cluster_config,
        rollout_dir,
        expname,
        run_after,
    )

    # -- Merge jobs (one per seed, depends on that seed's chunks) --------
    merge_job_specs = _build_merge_jobs(
        p,
        seeds,
        chunk_job_specs,
        fs,
        rollout_dir,
        expname,
        analyze_module=analyze_module,
        enrich_module=enrich_module,
        is_ray=use_ray_jobs,
    )
    jobs.extend(merge_job_specs)

    # -- Cross-seed aggregation job (pass@k) ----------------------------
    # Always run aggregate when filter is requested (needs difficulty.jsonl),
    # or when there are multiple seeds for cross-seed metrics.
    run_aggregate = aggregate_module and (p.num_random_seeds > 1 or filter_module)
    agg_job_spec: dict | None = None

    if run_aggregate:
        # Filename config is read from p.filter_cfg when a filter is
        # configured (since aggregate output becomes filter's input);
        # otherwise fall back to defaults so callers that only aggregate
        # (no filter) still produce the canonical difficulty.jsonl.
        difficulty_filename = (p.filter_cfg or {}).get("difficulty_filename", "difficulty.jsonl")
        agg_cmd_str = build_aggregate_cmd(
            rollout_dir=rollout_dir,
            aggregate_module=aggregate_module,
            difficulty_filename=difficulty_filename,
            expected_seeds=p.num_random_seeds,
            is_ray=use_ray_jobs,
        )

        agg_cmd = Command(
            script=make_bash_script(
                agg_cmd_str,
                installation_command=_ray_postprocess_installation_command(
                    p.installation_command,
                    is_ray=use_ray_jobs,
                ),
            ),
            container=p.postprocess_container,
            name="aggregate",
        )
        agg_group = CommandGroup(
            commands=[agg_cmd],
            hardware=HardwareConfig(num_gpus=0),
            name="aggregate",
            log_dir=f"{p.output_dir}/logs",
        )
        agg_job_spec = {
            "name": f"{expname}-aggregate",
            "group": agg_group,
            "dependencies": merge_job_specs or None,
        }
        jobs.append(agg_job_spec)

    # -- Filter job (CPU, depends on aggregate) --------------------------
    if filter_module and p.filter_cfg:
        filter_cmd_str = build_filter_cmd(
            output_dir=p.output_dir,
            difficulty_dir=rollout_dir,
            filter_module=filter_module,
            train_data=p.filter_cfg["input_data"],
            validation_data=p.filter_cfg.get("validation_data", ""),
            min_reward_std=p.filter_cfg.get("min_reward_std", 1e-6),
            policy_model=p.model_path,
            judge_model=p.jcfg.get("model_path", ""),
            train_filename=p.filter_cfg.get("train_filename", "train.jsonl"),
            val_filename=p.filter_cfg.get("val_filename", "validation.jsonl"),
            difficulty_filename=p.filter_cfg.get("difficulty_filename", "difficulty.jsonl"),
            report_filename=p.filter_cfg.get("report_filename", "filter_report.json"),
            is_ray=use_ray_jobs,
        )

        filter_cmd = Command(
            script=make_bash_script(
                filter_cmd_str,
                installation_command=_ray_postprocess_installation_command(
                    p.installation_command,
                    is_ray=use_ray_jobs,
                ),
            ),
            container=p.postprocess_container,
            name="filter",
        )
        filter_group = CommandGroup(
            commands=[filter_cmd],
            hardware=HardwareConfig(num_gpus=0),
            name="filter",
            log_dir=f"{p.output_dir}/logs",
        )
        filter_deps = [agg_job_spec] if agg_job_spec else (merge_job_specs or None)
        jobs.append(
            {
                "name": f"{expname}-filter",
                "group": filter_group,
                "dependencies": filter_deps,
            }
        )

    # -- Submit via Pipeline ---------------------------------------------
    if not jobs:
        console.success("All rollout jobs already complete (use rerun_done to force).")
        return

    pipeline = Pipeline(
        name=expname,
        cluster_config=cluster_config,
        jobs=jobs,
        with_ray=use_ray_jobs,
    )
    pipeline.run()

    console.success(f"{len(jobs)} job(s) submitted -> {p.output_dir}/")
