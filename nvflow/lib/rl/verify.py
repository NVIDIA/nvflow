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
"""Re-compute rewards on existing rollouts using a different judge.

Provides :func:`verify` -- reads rollout JSONL files produced by
:func:`~nvflow.lib.rl.rollout.rollout` and re-evaluates them by calling
the NeMo-Gym ``/verify`` endpoint with a (potentially different) judge.

Architecture (follows nemo-skills ``generate()`` CommandGroup pattern):

  Each verify job is a ``CommandGroup`` containing:
    - Judge vLLM server ``Command`` (GPU, if local_vllm mode)
    - Client ``Command`` (CPU: ng_run + verify_worker.py)

  vLLM servers are managed by Slurm -- they start automatically and are
  killed when the client command finishes (overlap mode).

Judge modes:

  - **Local vLLM judge** (``judge_vllm.model_path``): starts a vLLM server.
  - **External vLLM judge** (``judge_vllm.base_url``): pre-launched server.
  - **OpenAI API judge** (``judge_vllm.openai_base_url``): external API.
  - Policy-as-judge is NOT supported (no policy vLLM to reuse).
"""

from pathlib import Path
from typing import Any

from nvflow.core import console
from nvflow.lib.executor import is_ray_backend

from .helpers import (
    CONTAINER_CODE_DIR,
    SHELL_FIND_FREE_PORT,
    SHELL_WAIT_FOR_SERVER,
    VLLM_CONTAINER,
    LauncherFS,
    build_config_paths_str,
    build_judge_ng_run_overrides,
    build_ng_run_invocation,
    check_launcher_cwd,
    compute_num_gpus,
    determine_judge_mode,
    get_env_from_environments,
    log_judge_details,
)
from .rollout import (
    _build_port_read_preamble,
    _new_ray_attempt_id,
    _python_module_prefix,
    _ray_postprocess_installation_command,
    _server_stop_sentinel,
    build_aggregate_cmd,
    build_filter_cmd,
    make_bash_script,
    make_server_script,
)

# ---------------------------------------------------------------------------
# Inline command builders
# ---------------------------------------------------------------------------


def _build_verify_cmd(
    *,
    output_dir: str,
    gym_path: str,
    uv_venv_dir: str = "",
    input_file: str,
    output_file: str,
    done_file: str,
    config_paths: str,
    num_parallel: int,
    job_label: str,
    judge_mode: str,
    environment_name: str,
    judge_ng_run_overrides: str,
    is_ray: bool = False,
    stop_sentinel: str = "",
) -> str:
    """Build the re-judge (verify) bash script.

    Generated script structure:
      1. Start NeMo-Gym servers via ``gym env start`` (judge only, no policy)
      2. Re-judge rollouts via ``verify_worker``
    """
    # -- Shell variables & shared functions --------------------------------
    variables = (
        "set -e\n"
        "\n"
        f'OUTPUT_DIR="{output_dir}"\n'
        f'GYM_PATH="{gym_path}"\n'
        # ng_run per-component venv root; defaults to GYM_PATH (Gym's PARENT_DIR
        # default), overridden to /opt/gym-venvs on the CPU nemo-gym image.
        f'UV_VENV_DIR="{uv_venv_dir or gym_path}"\n'
        f'INPUT_FILE="{input_file}"\n'
        f'OUTPUT_FILE="{output_file}"\n'
        f'DONE_FILE="{done_file}"\n'
        f'CONFIG_PATHS="{config_paths}"\n'
        f'NUM_PARALLEL="{num_parallel}"\n'
        f'JOB_LABEL="{job_label}"\n'
        f'ENVIRONMENT_NAME="{environment_name}"\n'
    )

    setup = (
        "\n"
        'mkdir -p "$OUTPUT_DIR/logs" "$OUTPUT_DIR/rejudge"\n'
        "\n" + SHELL_FIND_FREE_PORT + "\n"
        "HEAD_SERVER_PORT=$(find_free_port)\n"
        "\n"
        'NG_RUN_PID=""\n'
        "\n"
        "cleanup() {\n"
        '    echo ""\n'
        '    echo "[Cleanup] Shutting down NeMo-Gym servers ..."\n'
        '    [ -n "$NG_RUN_PID" ] && kill $NG_RUN_PID 2>/dev/null && wait $NG_RUN_PID 2>/dev/null || true\n'
        # Ray-only (see WORKAROUND in rollout.py): touch the server-stop sentinel
        # UNCONDITIONALLY so the paired judge serve's watcher stops vLLM + exits 0
        # on a SUCCESSFUL rejudge too (Slurm's allocation teardown does this, but
        # nothing reaps the separate Ray serve job on success).  Empty on Slurm,
        # so the emitted command stays byte-identical.
        + (
            f'    echo "[Cleanup] Touching server-stop sentinel {stop_sentinel}"\n'
            f'    touch "{stop_sentinel}" 2>/dev/null || true\n'
            if is_ray and stop_sentinel
            else ""
        )
        + "}\n"
        "trap cleanup EXIT\n"
        "\n" + SHELL_WAIT_FOR_SERVER + "\n"
    )

    banner = (
        'echo "============================================================"\n'
        'echo "Compute Rewards (re-judge)  [$JOB_LABEL]"\n'
        'echo "============================================================"\n'
        'echo "Input file:   $INPUT_FILE"\n'
        'echo "Output file:  $OUTPUT_FILE"\n'
        f'echo "Judge mode:   {judge_mode}"\n'
        'echo "Environment:  $ENVIRONMENT_NAME"\n'
        'echo "============================================================"\n'
    )

    # -- Step 1: Start NeMo-Gym servers (judge only) ----------------------
    # Re-judge has no policy vLLM, so policy_base_url and policy_model are
    # stub values; the head-server port-range workaround is also unneeded
    # since no policy server collides with the gym ephemeral range.
    # No `source .venv/bin/activate`: the CLI is provided by the stage's
    # installation_command (baked .venv on nemo-rl, or /opt/gym-cli-venv on PATH
    # for nemo-gym). ng_run resolves component venvs via +uv_venv_dir=$UV_VENV_DIR.
    step1_ng_run = '\ncd "$GYM_PATH"\n\n' + build_ng_run_invocation(
        step_label="[Step 1/2]",
        policy_base_url="http://localhost:0/v1",
        policy_model="unused",
        judge_ng_run_overrides=judge_ng_run_overrides,
        include_port_range=False,
        is_ray=is_ray,
    )

    # -- Step 2: Re-judge rollouts ----------------------------------------
    if is_ray:
        # Ray Jobs can transiently lose a colocated service while the cluster
        # settles.  Keep that resilience entirely inside the Ray rendering;
        # the ordinary Slurm command below remains byte-for-byte unchanged.
        step2_rejudge = (
            "\n"
            'echo ""\n'
            'echo "[Step 2/2] Re-judging rollouts ..."\n'
            "VERIFY_MAX_ATTEMPTS=3\n"
            "VERIFY_RETRY_DELAY=10\n"
            "verify_attempt=1\n"
            "while true; do\n"
            "    if python3 -m nvflow.lib.rl.verify_worker \\\n"
            '        "$INPUT_FILE" \\\n'
            '        "$OUTPUT_FILE-async" \\\n'
            '        "127.0.0.1" \\\n'
            '        "$HEAD_SERVER_PORT" \\\n'
            '        "$ENVIRONMENT_NAME" \\\n'
            '        "$NUM_PARALLEL"; then\n'
            "        break\n"
            "    fi\n"
            '    if [ "$verify_attempt" -ge "$VERIFY_MAX_ATTEMPTS" ]; then\n'
            '        echo "[nvflow] ERROR: verify_worker failed after $VERIFY_MAX_ATTEMPTS attempt(s) for [$JOB_LABEL]" >&2\n'
            "        exit 1\n"
            "    fi\n"
            '    echo "[nvflow] verify_worker attempt $verify_attempt/$VERIFY_MAX_ATTEMPTS failed for [$JOB_LABEL]; retrying in ${VERIFY_RETRY_DELAY}s ..." >&2\n'
            '    sleep "$VERIFY_RETRY_DELAY"\n'
            "    verify_attempt=$((verify_attempt + 1))\n"
            "    VERIFY_RETRY_DELAY=$((VERIFY_RETRY_DELAY * 2))\n"
            "done\n"
        )
    else:
        step2_rejudge = (
            "\n"
            'echo ""\n'
            'echo "[Step 2/2] Re-judging rollouts ..."\n'
            f"PYTHONPATH={CONTAINER_CODE_DIR} python3 -m nvflow.lib.rl.verify_worker \\\n"
            '    "$INPUT_FILE" \\\n'
            '    "$OUTPUT_FILE-async" \\\n'
            '    "127.0.0.1" \\\n'
            '    "$HEAD_SERVER_PORT" \\\n'
            '    "$ENVIRONMENT_NAME" \\\n'
            '    "$NUM_PARALLEL"\n'
        )

    # -- Finalize: rename output, mark done -------------------------------
    finalize = (
        "\n"
        'mv "$OUTPUT_FILE-async" "$OUTPUT_FILE"\n'
        'touch "$DONE_FILE"\n'
        'echo "Done [$JOB_LABEL]. Cleanup via trap."\n'
    )

    return variables + setup + banner + step1_ng_run + step2_rejudge + finalize


def _build_seeds_present_check(*, rejudge_dir: str, expected_seed_files: list[str]) -> str:
    """Bash preflight asserting every expected seed produced a re-judged file.

    Runs at the head of the aggregate job (after its seed deps).  Fails loudly
    with a non-zero exit if any expected ``output-rs{seed}.jsonl`` / its
    ``.done`` marker is missing, so cross-seed pass@k is never silently
    computed over a subset when a seed died mid-run.
    """
    files_bash = " ".join(f'"{rejudge_dir}/{name}"' for name in expected_seed_files)
    return (
        "set -e\n"
        f"EXPECTED_SEED_FILES=({files_bash})\n"
        "_missing_seeds=0\n"
        'for _seed_file in "${EXPECTED_SEED_FILES[@]}"; do\n'
        '    if [ ! -s "$_seed_file" ] || [ ! -f "$_seed_file.done" ]; then\n'
        '        echo "[nvflow] ERROR: missing/incomplete re-judged seed: $_seed_file (.done present? $([ -f "$_seed_file.done" ] && echo yes || echo no))" >&2\n'
        "        _missing_seeds=$((_missing_seeds + 1))\n"
        "    fi\n"
        "done\n"
        'if [ "$_missing_seeds" -ne 0 ]; then\n'
        '    echo "[nvflow] ERROR: $_missing_seeds of ${#EXPECTED_SEED_FILES[@]} expected seed(s) missing; refusing to aggregate pass@k over a subset." >&2\n'
        "    exit 1\n"
        "fi\n"
        'echo "[nvflow] All ${#EXPECTED_SEED_FILES[@]} expected seed(s) present; aggregating."\n'
    )


def _build_analysis_cmd(
    *,
    rejudge_dir: str,
    gym_path: str,
    analyze_module: str,
    analysis_entries: list[tuple[str, str]],
    is_ray: bool = False,
) -> str:
    """Build the inline bash command for the reward analysis job.

    Args:
        analysis_entries: List of ``(seed_label, rewards_file)`` tuples.
    """
    python_prefix = _python_module_prefix(is_ray)
    parts = [
        "set -e\n",
        'echo "Reward Analysis"\n',
    ]
    for seed_label, rewards_file in analysis_entries:
        parts.append(
            f'echo "Analyzing {seed_label} ..."\n'
            f"{python_prefix}python3 -m {analyze_module} \\\n"
            f'    "{rewards_file}" \\\n'
            f'    "{rejudge_dir}/analysis_{seed_label}" \\\n'
            '    "REWARD RE-COMPUTATION ANALYSIS"\n'
        )
    first_file = analysis_entries[0][1] if analysis_entries else f"{rejudge_dir}/output-rs0.jsonl"
    parts.append(
        'echo "Done. Analysis complete."\n'
        'echo ""\n'
        'echo "To browse re-judged rollouts interactively (in the nemo-gym container):"\n'
        f'echo "  export PATH=/opt/gym-cli-venv/bin:$PATH && ng_viewer +jsonl_fpath={first_file}"\n'
    )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def verify(
    config: dict[str, Any],
    cluster: str,
    expname: str,
    run_after: list[str] | None = None,
    *,
    analyze_module: str = "",
    aggregate_module: str = "",
    filter_module: str = "",
) -> None:
    """Re-compute rewards on existing rollouts using a different judge.

    Uses the nemo-skills ``Pipeline`` + ``CommandGroup`` declarative API
    to orchestrate the judge vLLM server alongside the verify client
    within a single Slurm job.

    Args:
        config: Stage configuration dict (from workflow YAML).
            Caller is responsible for validation before calling.
        cluster: Cluster name for nemo-skills.
        expname: Base experiment name for Slurm jobs.
        run_after: Slurm job dependencies.
        analyze_module: Python module for per-seed analysis
            (invoked as ``python3 -m <module>``).
        aggregate_module: Python module for cross-seed aggregation
            (invoked as ``python3 -m <module>``).
        filter_module: Python module for training data filtering
            (invoked as ``python3 -m <module>``).
    """
    check_launcher_cwd()

    import nemo_skills.pipeline.utils as pipeline_utils
    from nemo_skills.pipeline.utils.declarative import (
        Command,
        CommandGroup,
        HardwareConfig,
        Pipeline,
    )

    output_dir = config["output_dir"]
    gym_path = config["gym_path"]
    # Empty -> defaults to gym_path; CPU nemo-gym sets /opt/gym-venvs to reuse baked venvs.
    uv_venv_dir = config.get("gym_uv_venv_dir", "")
    client_container = config["container"]
    postprocess_container = config.get("postprocess_container", client_container)
    vllm_container = config.get("vllm_container", VLLM_CONTAINER)
    installation_command = config.get("installation_command")

    rcfg = config["rejudge"]
    input_dir = rcfg["input_dir"]
    num_parallel = rcfg.get("num_samples_in_parallel", 8)
    num_gpus = compute_num_gpus(rcfg, has_policy=False)
    rerun_done = rcfg.get("rerun_done", False)

    rcfg_with_env = {**rcfg, "environments": config["environments"]}
    environment_name, env_inner_name, _ = get_env_from_environments(rcfg_with_env)
    rcfg_with_env["environment_name"] = environment_name
    rcfg_with_env["environment_inner_name"] = env_inner_name

    judge_mode = determine_judge_mode(rcfg, allow_policy_as_judge=False)
    config_paths_str = build_config_paths_str(rcfg_with_env)
    judge_ng_run_overrides = build_judge_ng_run_overrides(rcfg_with_env, judge_mode)

    cluster_config = pipeline_utils.get_cluster_config(cluster)
    use_ray_jobs = is_ray_backend(cluster_config)

    fs = LauncherFS(cluster_config)
    rejudge_dir = f"{output_dir}/rejudge"

    # -- Discover rollout files from input_dir --------------------------
    # LauncherFS.ls works within the cluster (local glob) or off-cluster
    # (remote `ls` over the ssh tunnel).
    names = [
        n
        for n in fs.ls(input_dir, "output-rs*.jsonl")
        if "_chunk_" not in n and not n.endswith("-async")
    ]

    if not names:
        console.warning(f"No rollout files found in {input_dir}")
        return

    # Path() here is used only for .name/.stem downstream (no filesystem access).
    rollout_files = [Path(n) for n in names]

    # -- Resume: find remaining files -----------------------------------
    done_exist = fs.batch_exists([f"{rejudge_dir}/{n}.done" for n in names])
    remaining: list[Path] = [
        rf
        for rf in rollout_files
        if rerun_done or not done_exist.get(f"{rejudge_dir}/{rf.name}.done")
    ]

    skipped = len(rollout_files) - len(remaining)

    console.status("Computing rewards (re-judge via /verify)")
    log_judge_details(console, rcfg, judge_mode)
    console.detail("Slurm GPUs/job", str(num_gpus))
    console.detail("Environment", environment_name)
    console.detail("Jobs", f"{len(remaining)} to submit, {skipped} done")
    console.detail("Output", output_dir)
    console.blank()

    filter_cfg = config.get("filter") or {}

    if not remaining and not filter_cfg:
        console.success("All reward jobs already complete (use rerun_done to force).")
        return

    # -- Build Pipeline jobs ---------------------------------------------
    jcfg = rcfg.get("judge_vllm") or {}
    need_judge_server = judge_mode == "local_vllm" and jcfg.get("model_path")
    job_log_dir = f"{output_dir}/logs"

    jobs: list[dict] = []
    verify_job_specs: list[dict] = []

    for rollout_file in remaining:
        seed_label = rollout_file.stem.replace("output-", "")
        job_label = f"rejudge_{seed_label}"

        # Per-attempt nonce baked into BOTH the judge server (writer) and the
        # client port-read preamble (reader) so a re-run never reads a stale
        # port.  Under the Ray Jobs backend $SLURM_JOB_ID is constant for the head's whole
        # life, so without this every attempt would reuse the same port file.
        # Must be a Python-baked constant (writer and reader are different
        # processes/nodes), never a bash $$/$RANDOM value.
        is_ray = use_ray_jobs
        attempt_id = _new_ray_attempt_id() if is_ray else ""
        # Ray-only stop sentinel: the verify client touches it on EXIT and the
        # judge serve polls it.  Empty on Slurm so the serve stays foreground and
        # the emitted command is byte-identical to the validated Slurm path.
        stop_sentinel = _server_stop_sentinel(job_log_dir, job_label, attempt_id) if is_ray else ""
        judge_script = (
            make_server_script(
                jcfg,
                cluster_config,
                role="judge",
                log_dir=job_log_dir,
                job_label=job_label,
                attempt_id=attempt_id,
                is_ray=is_ray,
                stop_sentinel=stop_sentinel,
            )
            if need_judge_server
            else None
        )

        if judge_script is not None:
            judge_vllm_url = "http://127.0.0.1:$JUDGE_PORT/v1"
            job_judge_overrides = build_judge_ng_run_overrides(
                rcfg_with_env, judge_mode, judge_url_var=judge_vllm_url
            )
        else:
            job_judge_overrides = judge_ng_run_overrides

        port_preamble = _build_port_read_preamble(
            job_log_dir, job_label, has_judge=judge_script is not None, attempt_id=attempt_id
        )

        client_cmd_str = _build_verify_cmd(
            output_dir=output_dir,
            gym_path=gym_path,
            uv_venv_dir=uv_venv_dir,
            input_file=f"{input_dir}/{rollout_file.name}",
            output_file=f"{rejudge_dir}/{rollout_file.name}",
            done_file=f"{rejudge_dir}/{rollout_file.name}.done",
            config_paths=config_paths_str,
            num_parallel=num_parallel,
            job_label=job_label,
            judge_mode=judge_mode,
            environment_name=environment_name,
            judge_ng_run_overrides=job_judge_overrides,
            is_ray=is_ray,
            stop_sentinel=stop_sentinel,
        )
        if port_preamble:
            client_cmd_str = port_preamble + client_cmd_str

        components: list[Command] = []
        max_nodes = 1

        if judge_script is not None:
            components.append(
                Command(script=judge_script, container=vllm_container, name=f"{job_label}_judge")
            )
            max_nodes = max(max_nodes, judge_script.num_nodes)

        client_script = make_bash_script(
            client_cmd_str,
            installation_command=installation_command,
        )
        components.append(Command(script=client_script, container=client_container, name=job_label))

        cmd_group = CommandGroup(
            commands=components,
            hardware=HardwareConfig(
                num_gpus=num_gpus,
                num_nodes=max_nodes,
            ),
            name=job_label,
            log_dir=f"{output_dir}/logs",
        )

        job_spec = {
            "name": f"{expname}-{seed_label}",
            "group": cmd_group,
            "dependencies": run_after or None,
        }
        jobs.append(job_spec)
        verify_job_specs.append(job_spec)

    # -- Analysis job (CPU, depends on all verify jobs) -----------------
    if analyze_module:
        analysis_entries = [
            (rf.stem.replace("output-", ""), f"{rejudge_dir}/{rf.name}") for rf in remaining
        ]

        analysis_cmd_str = _build_analysis_cmd(
            rejudge_dir=rejudge_dir,
            gym_path=gym_path,
            analyze_module=analyze_module,
            analysis_entries=analysis_entries,
            is_ray=use_ray_jobs,
        )

        analysis_cmd = Command(
            script=make_bash_script(
                analysis_cmd_str,
                installation_command=_ray_postprocess_installation_command(
                    installation_command,
                    is_ray=use_ray_jobs,
                ),
            ),
            container=postprocess_container,
            name="analysis",
        )
        analysis_group = CommandGroup(
            commands=[analysis_cmd],
            hardware=HardwareConfig(num_gpus=0),
            name="analysis",
            log_dir=f"{output_dir}/logs",
        )
        jobs.append(
            {
                "name": f"{expname}-analysis",
                "group": analysis_group,
                "dependencies": verify_job_specs,
            }
        )

    # -- Cross-seed aggregation job (pass@k) ----------------------------
    run_aggregate = aggregate_module and (len(rollout_files) > 1 or filter_module)
    agg_job_spec: dict | None = None

    if run_aggregate:
        # Assert every expected seed (the full set discovered in input_dir)
        # produced a re-judged file + .done before pass@k is computed, so a
        # dead seed fails loudly instead of silently under-counting.
        expected_seed_files = [rf.name for rf in rollout_files]
        agg_cmd_str = _build_seeds_present_check(
            rejudge_dir=rejudge_dir,
            expected_seed_files=expected_seed_files,
        ) + build_aggregate_cmd(
            rollout_dir=rejudge_dir,
            aggregate_module=aggregate_module,
            is_ray=use_ray_jobs,
        )

        agg_cmd = Command(
            script=make_bash_script(
                agg_cmd_str,
                installation_command=_ray_postprocess_installation_command(
                    installation_command,
                    is_ray=use_ray_jobs,
                ),
            ),
            container=postprocess_container,
            name="aggregate",
        )
        agg_group = CommandGroup(
            commands=[agg_cmd],
            hardware=HardwareConfig(num_gpus=0),
            name="aggregate",
            log_dir=f"{output_dir}/logs",
        )
        agg_job_spec = {
            "name": f"{expname}-aggregate",
            "group": agg_group,
            "dependencies": verify_job_specs or None,
        }
        jobs.append(agg_job_spec)

    # -- Filter job (CPU, depends on aggregate) --------------------------
    if filter_module and filter_cfg:
        filter_cmd_str = build_filter_cmd(
            output_dir=output_dir,
            difficulty_dir=rejudge_dir,
            filter_module=filter_module,
            train_data=filter_cfg["input_data"],
            validation_data=filter_cfg.get("validation_data", ""),
            min_reward_std=filter_cfg.get("min_reward_std", 1e-6),
            is_ray=use_ray_jobs,
        )

        filter_cmd = Command(
            script=make_bash_script(
                filter_cmd_str,
                installation_command=_ray_postprocess_installation_command(
                    installation_command,
                    is_ray=use_ray_jobs,
                ),
            ),
            container=postprocess_container,
            name="filter",
        )
        filter_group = CommandGroup(
            commands=[filter_cmd],
            hardware=HardwareConfig(num_gpus=0),
            name="filter",
            log_dir=f"{output_dir}/logs",
        )
        filter_deps = [agg_job_spec] if agg_job_spec else (verify_job_specs or None)
        jobs.append(
            {
                "name": f"{expname}-filter",
                "group": filter_group,
                "dependencies": filter_deps,
            }
        )

    # -- Submit via Pipeline ---------------------------------------------
    if not jobs:
        console.success("All reward jobs already complete (use rerun_done to force).")
        return

    pipeline = Pipeline(
        name=expname,
        cluster_config=cluster_config,
        jobs=jobs,
        with_ray=use_ray_jobs,
    )
    pipeline.run()

    console.success(f"{len(jobs)} job(s) submitted -> {output_dir}/")
