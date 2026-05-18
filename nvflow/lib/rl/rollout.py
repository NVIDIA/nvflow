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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemo_skills.pipeline.utils.scripts import BaseJobScript, ServerScript

from nvflow.core import console
from nvflow.lib.vllm_compat import get_server_entrypoint

from .helpers import (
    CONTAINER_CODE_DIR,
    NON_VLLM_KEYS,
    SERVER_CONTAINER,
    SHELL_FIND_FREE_PORT,
    SHELL_READ_PORT_FILE,
    SHELL_WAIT_FOR_SERVER,
    _build_overlay_setup_cmd,
    _overlay_path,
    build_config_paths_str,
    build_judge_ng_run_overrides,
    build_vllm_server_args,
    check_launcher_cwd,
    compute_num_gpus,
    determine_judge_mode,
    get_env_from_environments,
    log_judge_details,
    resolve_host_path,
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
            )

            preamble = _build_port_read_preamble(
                self.log_dir,
                self.job_label,
                has_policy=self.policy_server is not None,
                has_judge=self.judge_server is not None,
            )
            return preamble + cmd

        self.set_inline(build_cmd)
        super().__post_init__()


# ---------------------------------------------------------------------------
# Script helpers
# ---------------------------------------------------------------------------


def _vllm_port_file(log_dir: str, role: str, job_label: str = "") -> str:
    """Return the shared-filesystem path for the dynamic vLLM port file.

    Includes ``$SLURM_JOB_ID`` so each Slurm job gets a unique file.  This
    avoids a race condition where the client's ``rm -f`` of stale port files
    (after pip install) deletes the file the server already wrote.
    """
    suffix = f"_{job_label}" if job_label else ""
    return f"{log_dir}/.vllm_port_{role}{suffix}_${{SLURM_JOB_ID}}.txt"


def _wrap_server_with_dynamic_port(script: ServerScript, role: str, port_file: str) -> None:
    """Replace the hardcoded port in *script* with runtime-dynamic allocation.

    Wraps the server's inline command so that at runtime on the compute node:
      1. ``find_free_port()`` probes for an available port
      2. The port is written to *port_file* (shared filesystem)
      3. ``sed`` replaces the hardcoded port in the original command
    """
    hardcoded = str(script.port)
    original_inline = script.inline
    wrapped = (
        f"{SHELL_FIND_FREE_PORT}\n"
        f"VLLM_PORT=$(find_free_port)\n"
        f'echo "[dynamic-port] {role} vLLM using port $VLLM_PORT (node ${{SLURM_NODEID:-0}})"\n'
        f'if [ "${{SLURM_NODEID:-0}}" = "0" ]; then\n'
        f'    echo "$VLLM_PORT" > "{port_file}"\n'
        f"fi\n"
        f"ORIG_CMD=$(cat <<'__NVFLOW_VLLM_CMD__'\n"
        f"{original_inline}\n"
        f"__NVFLOW_VLLM_CMD__\n"
        f")\n"
        f'eval "$(echo "$ORIG_CMD" | sed "s/{hardcoded}/$VLLM_PORT/g")"\n'
    )
    script.set_inline(wrapped)


def _build_port_read_preamble(
    log_dir: str,
    job_label: str,
    *,
    has_policy: bool = False,
    has_judge: bool = False,
) -> str:
    """Build bash preamble that reads dynamic vLLM ports from port files.

    Returns empty string if neither server is present.
    """
    if not has_policy and not has_judge:
        return ""
    parts = [SHELL_READ_PORT_FILE]
    if has_policy:
        pf = _vllm_port_file(log_dir, "policy", job_label)
        parts.append(f'POLICY_PORT=$(read_port_file "{pf}" "Policy vLLM" 300)')
    if has_judge:
        jf = _vllm_port_file(log_dir, "judge", job_label)
        parts.append(f'JUDGE_PORT=$(read_port_file "{jf}" "Judge vLLM" 300)')
    return "\n".join(parts) + "\n"


def make_server_script(
    vllm_cfg: dict[str, Any],
    cluster_config: dict,
    *,
    role: str = "policy",
    log_dir: str = "",
    job_label: str = "",
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
        server_entrypoint=vllm_cfg.get("server_entrypoint", get_server_entrypoint()),
    )

    if overlay:
        setup_cmd = _build_overlay_setup_cmd(model_path, overlay, hf_overrides)
        script.set_inline(f"{setup_cmd} && {script.inline}")

    if log_dir:
        port_file = _vllm_port_file(log_dir, role, job_label)
        _wrap_server_with_dynamic_port(script, role, port_file)

    return script


def make_bash_script(
    bash_cmd: str,
    *,
    installation_command: str | None = None,
) -> BashScript:
    return BashScript(cmd=bash_cmd, installation_command=installation_command)


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
    host_dir: Path,
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
    """
    if rerun_done:
        for s in seeds:
            for c in chunk_ids:
                fname = _output_filename(s, c)
                (host_dir / f"{fname}.done").unlink(missing_ok=True)
                (host_dir / f"{fname}-async").unlink(missing_ok=True)
                (host_dir / f"{fname}-async.prev").unlink(missing_ok=True)
                (host_dir / fname).unlink(missing_ok=True)
        return [(s, c) for s in seeds for c in chunk_ids]

    remaining: list[tuple[int, int]] = []
    for s in seeds:
        mf = _merged_filename(s)
        merge_done = host_dir / f"{mf}.done"
        merge_file = host_dir / mf

        if merge_done.exists():
            if merge_file.exists():
                continue
            console.warning(f"Merged .done exists but {mf} is missing — resetting merge marker")
            merge_done.unlink()

        for c in chunk_ids:
            fname = _output_filename(s, c)
            done = host_dir / f"{fname}.done"
            output = host_dir / fname
            if done.exists() and not output.exists():
                console.warning(f"Stale .done for {fname} — re-scheduling")
                done.unlink()
            if not done.exists():
                remaining.append((s, c))

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


def _build_client_cmd(
    *,
    output_dir: str,
    gym_path: str,
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
) -> str:
    """Build the rollout collection bash script.

    Generated script structure:
      1.  Wait for vLLM servers (policy + optional judge)
      1a. (If chunked) Extract this job's slice via head/tail
      1b. Self-heal: recover orphaned .prev / partial finalize from prior crash
      1c. Resume-filter: skip already-completed rows from prior partial run
      2.  Start NeMo-Gym servers via ``ng_run`` (background)
      3.  Collect rollouts via ``ng_collect_rollouts``
      Finalize: merge partials, cp→output, touch .done, cleanup temps
    """
    # -- Shell variables & shared functions --------------------------------
    variables = (
        "set -e\n"
        "\n"
        f'OUTPUT_DIR="{output_dir}"\n'
        f'GYM_PATH="{gym_path}"\n'
        f'MODEL_PATH="{model_path}"\n'
        f'AGENT_NAME="{agent_name}"\n'
        f'INPUT_DATA="{input_data}"\n'
        f'OUTPUT_FILE="{output_file}"\n'
        f'DONE_FILE="{done_file}"\n'
        f'CONFIG_PATHS="{config_paths}"\n'
        f'NUM_PARALLEL="{num_parallel}"\n'
        f'JOB_LABEL="{job_label}"\n'
        f'VLLM_URL="{policy_vllm_url}"\n'
        f'JUDGE_URL="{judge_vllm_url}"\n'
        f"CHUNK_ID={chunk_id}\n"
        f"NUM_CHUNKS={num_chunks}\n"
    )

    setup = (
        "\n"
        'mkdir -p "$OUTPUT_DIR/logs"\n'
        "\n" + SHELL_FIND_FREE_PORT + "\n"
        'NG_RUN_PID=""\n'
        "\n"
        "cleanup() {\n"
        "    local _nvflow_exit=$?\n"
        '    echo ""\n'
        '    echo "[Cleanup] Shutting down NeMo-Gym servers ..."\n'
        '    [ -n "$NG_RUN_PID" ] && kill $NG_RUN_PID 2>/dev/null && wait $NG_RUN_PID 2>/dev/null || true\n'
        "    # Best-effort merge of .prev into -async.  On success, finalize already\n"
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
        "    if [ $_nvflow_exit -ne 0 ]; then\n"
        '        echo "[nvflow] Client exited with code $_nvflow_exit — cancelling job ${SLURM_JOB_ID}"\n'
        '        scancel "${SLURM_JOB_ID}" 2>/dev/null || kill 0 2>/dev/null || true\n'
        "    fi\n"
        "}\n"
        "trap cleanup EXIT\n"
        "\n" + SHELL_WAIT_FOR_SERVER + "\n"
    )

    banner = (
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

    # -- Step 1: Wait for vLLM servers ------------------------------------
    wait_for_vllm = _build_vllm_wait_snippet(policy_vllm_url, judge_vllm_url)
    step1_wait = (
        '\necho ""\necho "[Step 1/3] Waiting for vLLM servers ..."\n' + wait_for_vllm + "\n"
    )

    # -- Step 1a: Logical chunking (extract this job's slice) --------------
    # When num_chunks > 1, each Slurm job extracts its portion of the full
    # input at runtime via head|tail.  No physical pre-splitting on the
    # login node — keeps the launcher lightweight and filesystem-agnostic.
    chunk_slice = (
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

    # -- Early exit if already done ----------------------------------------
    # When dependent_jobs > 0, Slurm pre-submits a chain of jobs.  If an
    # earlier job in the chain already completed this chunk, the remaining
    # dependent jobs should exit immediately instead of re-doing the work.
    done_check = (
        'if [ -f "$DONE_FILE" ]; then\n'
        '    echo "Chunk already complete (.done exists) — skipping."\n'
        "    exit 0\n"
        "fi\n"
    )

    # -- Step 1b: Self-heal ------------------------------------------------
    # Recover from any interrupted prior run so the resume filter sees the
    # full set of completed samples.  Three recovery cases:
    #
    #   A. Partial finalize: output file exists but .done was never written
    #      (kill between mv -async→output and touch .done).
    #      Fix: move output back to -async.
    #
    #   B. Orphaned .prev: cleanup trap was killed (SIGKILL / OOM / node
    #      failure) before merging .prev back into -async.
    #      Fix: merge .prev into -async.
    #
    #   C. Orphaned temp files (.healed, .restored, .merged) from partial
    #      cleanup/finalize.  Harmless but noisy — clean them up.
    #
    # Wrapped in a subshell so failures don't abort the job under set -e.
    # If self-heal fails, the job continues (re-does some work, but runs).
    selfheal = (
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

    # -- Step 1c: Resume filter (skip completed rows) ---------------------
    # When chunked, truncation is handled by the slice above, so pass 0.
    resume_max = 0 if num_chunks > 1 else max_num_samples
    resume = (
        'REMAINING_INPUT="$OUTPUT_DIR/remaining_input_chunk$CHUNK_ID.jsonl"\n'
        "\n"
        f'if ! PYTHONPATH={CONTAINER_CODE_DIR} python3 -m nvflow.lib.rl.resume_filter "$ASYNC_FILE" "$INPUT_DATA" "$REMAINING_INPUT" {resume_max}; then\n'
        '    echo "ERROR: resume_filter failed" >&2\n'
        "    exit 1\n"
        "fi\n"
        "\n"
        'if [ -f "$ASYNC_FILE" ] && [ ! -s "$REMAINING_INPUT" ]; then\n'
        '    echo "All rows already completed in -async -- finalizing."\n'
        '    cp -f "$ASYNC_FILE" "$OUTPUT_FILE"\n'
        '    touch "$DONE_FILE"\n'
        '    rm -f "$ASYNC_FILE"\n'
        '    echo "Done [$JOB_LABEL]."\n'
        "    exit 0\n"
        "fi\n"
    )

    # -- Step 2: Start NeMo-Gym servers -----------------------------------
    # WORKAROUND(port-toctou): allocate port here (not in setup) to minimise
    # the window between find_free_port() and ng_run binding to it.
    # WORKAROUND(gym-port-range): keep NeMo-Gym internal ports in 1024-8999,
    # below the cluster ephemeral range (9000-65000 on ARM, 32768-60999 on x86).
    step2_ng_run = (
        "\n"
        "HEAD_SERVER_PORT=$(find_free_port)\n"
        "\n"
        'cd "$GYM_PATH"\n'
        "\n"
        'echo ""\n'
        'echo "[Step 2/3] Starting NeMo-Gym servers ..."\n'
        'ng_run "+config_paths=[$CONFIG_PATHS]" \\\n'
        '    "+policy_model.responses_api_models.vllm_model.base_url=$VLLM_URL" \\\n'
        '    "+policy_model.responses_api_models.vllm_model.api_key=EMPTY" \\\n'
        '    "+policy_model.responses_api_models.vllm_model.model=$MODEL_PATH" \\\n'
        '    "+head_server.host=127.0.0.1" \\\n'
        '    "+head_server.port=$HEAD_SERVER_PORT" \\\n'
        '    "+port_range_low=1024" \\\n'
        '    "+port_range_high=8999" \\\n'
        '    "+skip_venv_if_present=true" \\\n'
        f"{judge_ng_run_overrides}"
        '    > "$OUTPUT_DIR/logs/ng_run_$JOB_LABEL.log" 2>&1 &\n'
        "NG_RUN_PID=$!\n"
        "\n"
        'wait_for_server "http://127.0.0.1:$HEAD_SERVER_PORT/" "NeMo-Gym" $NG_RUN_PID 60 "$OUTPUT_DIR/logs/ng_run_$JOB_LABEL.log"\n'
    )

    # -- Step 3: Collect rollouts -----------------------------------------
    step3_collect = (
        "\n"
        'echo ""\n'
        'echo "[Step 3/3] Collecting rollouts ..."\n'
        "# Back up previous partial results before ng_collect_rollouts clears the file.\n"
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
        "    ng_collect_rollouts \\\n"
        "        ${AGENT_NAME:++agent_name=$AGENT_NAME} \\\n"
        "        +input_jsonl_fpath=$REMAINING_INPUT \\\n"
        "        +output_jsonl_fpath=$ASYNC_FILE \\\n"
        "        +num_repeats=1 \\\n"
        "        +resume_from_cache=true \\\n"
        "        +num_samples_in_parallel=$NUM_PARALLEL \\\n"
        "        +head_server.host=127.0.0.1 \\\n"
        "        +head_server.port=$HEAD_SERVER_PORT"
        + "".join(
            f" \\\n        +responses_create_params.{k}={v}"
            for k, v in (responses_create_params or {}).items()
        )
        + "\n"
        "    _NVFLOW_EXIT=$?\n"
        "    set -e\n"
        "    [ $_NVFLOW_EXIT -eq 0 ] && break\n"
        "    if [ $_attempt -lt $_NVFLOW_MAX_RETRIES ]; then\n"
        '        echo "[nvflow] ng_collect_rollouts exited $_NVFLOW_EXIT'
        " (attempt $_attempt/$_NVFLOW_MAX_RETRIES)."
        ' Retrying in ${_NVFLOW_RETRY_DELAY}s ..."\n'
        "        sleep $_NVFLOW_RETRY_DELAY\n"
        "        _NVFLOW_RETRY_DELAY=$((_NVFLOW_RETRY_DELAY * 2))\n"
        "    fi\n"
        "done\n"
        "if [ $_NVFLOW_EXIT -ne 0 ]; then\n"
        '    echo "[nvflow] ng_collect_rollouts failed after'
        ' $_NVFLOW_MAX_RETRIES attempts."\n'
        "    exit $_NVFLOW_EXIT\n"
        "fi\n"
    )

    # -- Finalize: merge partials, write output, mark done ----------------
    # Order matters for crash safety:
    #   1. Merge .prev + -async into -async  (all results in one file)
    #   2. Copy -async → output              (cp, not mv — keeps -async as backup)
    #   3. Touch .done                        (marks completion)
    #   4. Clean up -async, .prev, temps      (safe — .done exists)
    # If killed at any point, self-heal on next start recovers:
    #   after 1: -async has everything, resume finds all done
    #   after 2: output exists w/o .done → self-heal Case A restores to -async
    #   after 3: .done exists → _get_remaining_jobs skips this chunk entirely
    finalize = (
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

    return (
        variables
        + setup
        + banner
        + done_check
        + step1_wait
        + chunk_slice
        + selfheal
        + resume
        + step2_ng_run
        + step3_collect
        + finalize
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
) -> str:
    """Build the chunk-merge + enrich + analyze bash script.

    Generated script structure:
      1. Concatenate per-chunk rollout files into a single merged file
      2. Enrich merged rollouts with input metadata
      3. Analyze rollouts (accuracy, token stats, etc.)
    """
    # -- Shell variables --------------------------------------------------
    variables = (
        "set -e\n"
        "\n"
        f'MERGED_FILE="{merged_file}"\n'
        f'ANALYSIS_DIR="{analysis_dir}"\n'
        f'SEED_LABEL="{seed_label}"\n'
        f"NUM_CHUNKS={num_chunks}\n"
        f'INPUT_DATA="{input_data}"\n'
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
        "# Precondition: ALL chunk .done markers must exist\n"
        "for i in $(seq 0 $((NUM_CHUNKS - 1))); do\n"
        f'    CHUNK_DONE="{chunk_done_pattern}"\n'
        '    if [ ! -f "$CHUNK_DONE" ]; then\n'
        '        echo "Chunk $i not complete (.done missing) — skipping merge."\n'
        "        exit 0\n"
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

    # -- Step 2: Enrich with input metadata -------------------------------
    step2_enrich = (
        "\n"
        'echo ""\n'
        'echo "[Step 2/3] Enriching rollouts with input metadata ..."\n'
        f"PYTHONPATH={CONTAINER_CODE_DIR} python3 -m {enrich_module} \\\n"
        '    "$INPUT_DATA" \\\n'
        '    "$MERGED_FILE"\n'
    )

    # -- Step 3: Analyze rollouts -----------------------------------------
    step3_analyze = (
        "\n"
        'echo ""\n'
        'echo "[Step 3/3] Analyzing rollouts ..."\n'
        f"PYTHONPATH={CONTAINER_CODE_DIR} python3 -m {analyze_module} \\\n"
        '    "$MERGED_FILE" \\\n'
        f'    "$ANALYSIS_DIR"\n'
    )

    # -- Cleanup: mark done, then remove chunk output files -----------------
    # Order: touch .done FIRST, then delete chunk data.  This guarantees the
    # merged file is the verified complete copy before any source data is
    # removed.  Keep chunk .done markers — _get_remaining_jobs relies on them.
    cleanup = (
        "\n"
        f'touch "{merged_done_file}"\n'
        "\n"
        "# Safe cleanup: delete chunk data files only (keep .done markers).\n"
        "for i in $(seq 0 $((NUM_CHUNKS - 1))); do\n"
        f'    CHUNK_FILE="{chunk_file_pattern}"\n'
        '    rm -f "$CHUNK_FILE"\n'
        "done\n"
        'echo "Done [$SEED_LABEL]."\n'
        'echo ""\n'
        'echo "To browse rollouts interactively (requires Gym venv):"\n'
        f'echo "  cd {gym_path} && source .venv/bin/activate"\n'
        'echo "  ng_viewer +jsonl_fpath=$MERGED_FILE"\n'
    )

    return variables + step1_merge + step2_enrich + step3_analyze + cleanup


def build_aggregate_cmd(
    *,
    rollout_dir: str,
    aggregate_module: str,
    difficulty_filename: str = "difficulty.jsonl",
) -> str:
    return (
        "set -e\n"
        'echo "Cross-Seed Aggregation (pass@k)"\n'
        f"PYTHONPATH={CONTAINER_CODE_DIR} python3 -m {aggregate_module} \\\n"
        f'    "{rollout_dir}" \\\n'
        f'    "{rollout_dir}/aggregate" \\\n'
        f'    --output_filename "{difficulty_filename}"\n'
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
) -> str:
    cmd = (
        "set -e\n"
        'echo "Filter Training Data (reward-variance difficulty)"\n'
        f"PYTHONPATH={CONTAINER_CODE_DIR} python3 -m {filter_module} \\\n"
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
    client_container: str
    post_container: str
    server_container: str
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
    client_container = config["container"]
    post_container = config.get("post_container", client_container)
    server_container = SERVER_CONTAINER
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
        client_container=client_container,
        post_container=post_container,
        server_container=server_container,
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
    host_rollout_dir: Path,
    seeds: list[int],
    chunk_ids: list[int],
    input_data: str,
    max_num_samples: int,
    num_chunks: int,
) -> str:
    """Return progress string like '~42% (15,000/35,000 rows)', or '' on failure."""
    try:
        host_input = resolve_host_path(input_data)
        input_lines = sum(1 for _ in open(host_input)) if host_input.exists() else 0
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
            if (host_rollout_dir / f"{fname}.done").exists():
                completed_rows += expected
            else:
                for suffix in ["-async", "-async.prev"]:
                    async_path = host_rollout_dir / f"{fname}{suffix}"
                    if async_path.exists():
                        completed_rows += sum(1 for _ in open(async_path))
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

    for seed, chunk_id in remaining:
        job_lbl = f"rs{seed}_chunk{chunk_id}"
        out_filename = _output_filename(seed, chunk_id)
        suffix_fmt = "-{dep_id}" if p.dependent_jobs > 0 else ""

        prev_job_spec = None
        for dep_id in range(p.dependent_jobs + 1):
            policy_script = (
                make_server_script(
                    p.pcfg, cluster_config, role="policy", log_dir=job_log_dir, job_label=job_lbl
                )
                if p.need_policy_server
                else None
            )
            judge_script = (
                make_server_script(
                    p.jcfg, cluster_config, role="judge", log_dir=job_log_dir, job_label=job_lbl
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
                installation_command=p.installation_command,
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
                            container=p.server_container,
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
                            container=p.server_container,
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
                            container=p.server_container,
                            name=f"{job_lbl}_policy",
                        )
                    )
                    max_nodes = max(max_nodes, policy_script.num_nodes)
                if judge_script is not None:
                    components.append(
                        Command(
                            script=judge_script,
                            container=p.server_container,
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
    host_rollout_dir: Path,
    rollout_dir: str,
    expname: str,
    *,
    analyze_module: str,
    enrich_module: str,
) -> list[dict]:
    """Build one merge job per seed. Returns merge job specs."""
    from nemo_skills.pipeline.utils.declarative import Command, CommandGroup, HardwareConfig

    merge_job_specs: list[dict] = []
    for seed in seeds:
        seed_label = f"rs{seed}"
        merged_filename = _merged_filename(seed)

        if (host_rollout_dir / f"{merged_filename}.done").exists() and not p.rerun_done:
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
        )

        merge_cmd = Command(
            script=make_bash_script(merge_cmd_str),
            container=p.post_container,
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

    rollout_dir = f"{p.output_dir}/rollout"

    # Resolve host paths for resume checks only.
    # No file I/O on the login node — each Slurm job slices its own chunk
    # at runtime via head/tail (see _build_client_cmd chunk_slice step).
    host_dir = resolve_host_path(p.output_dir)
    host_rollout_dir = host_dir / "rollout"

    seeds = list(range(p.starting_seed, p.starting_seed + p.num_random_seeds))
    chunk_ids = list(range(p.num_chunks))

    if p.num_chunks > 1:
        console.detail("Input", f"{p.num_chunks} logical chunks (each job slices at runtime)")

    # -- Resume: find remaining (seed, chunk) pairs ---------------------
    if host_rollout_dir.exists():
        remaining = _get_remaining_jobs(host_rollout_dir, seeds, chunk_ids, p.rerun_done)
    else:
        remaining = [(s, c) for s in seeds for c in chunk_ids]
    skipped = len(seeds) * len(chunk_ids) - len(remaining)

    # Invalidate merge .done for any seed that has chunks to re-run,
    # so the merge step doesn't skip it with stale results.
    # Never delete the merged data file — it serves as backup until the
    # next merge atomically overwrites it (Fix 2: safe invalidation).
    rerun_seeds = {s for s, _ in remaining}
    for s in rerun_seeds:
        mf = _merged_filename(s)
        (host_rollout_dir / f"{mf}.done").unlink(missing_ok=True)

    # -- Progress estimate: count completed rows in partial -async files ---
    progress_msg = ""
    if remaining and host_rollout_dir.exists():
        progress_msg = _estimate_progress(
            host_rollout_dir,
            seeds,
            chunk_ids,
            p.input_data,
            p.max_num_samples,
            p.num_chunks,
        )

    console.status("Collecting rollouts (ng_collect_rollouts)")
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

    if not remaining and not p.filter_cfg:
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
        host_rollout_dir,
        rollout_dir,
        expname,
        analyze_module=analyze_module,
        enrich_module=enrich_module,
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
        )

        agg_cmd = Command(
            script=make_bash_script(agg_cmd_str),
            container=p.post_container,
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
        )

        filter_cmd = Command(
            script=make_bash_script(filter_cmd_str),
            container=p.post_container,
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
    )
    pipeline.run()

    console.success(f"{len(jobs)} job(s) submitted -> {p.output_dir}/")
