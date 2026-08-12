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
"""Tests for nvflow.lib.rl.rollout — collect_rollouts orchestrator.

Pins the pure-function rendering of bash command builders + behavioral
contracts of the resume / progress logic so future refactors of the
~1500-line ``rollout.py`` can detect:

  - **Byte-level regressions** in rendered shell scripts via fixture
    snapshots under ``tests/fixtures/rollout/``.  These scripts run
    inside Slurm jobs; even a whitespace change can shift exit-code
    semantics, so bytes-exact pinning is the right guard.

  - **Behavioral regressions** in the resume / two-level integrity
    check via ``_get_remaining_jobs`` cases.

Refresh fixtures via ``uv run python3 scripts/_dump_rollout_fixtures.py``
and inspect ``git diff tests/fixtures/rollout/`` carefully before
committing.  Any unintended diff is a regression.

Why prefer fixture files over inline strings: each snapshot is ~300
lines of bash; inlining would make this test file unreadable and
``git blame`` of the actual change-points useless.  Files keep the
test logic compact while preserving full byte fidelity.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("nemo_skills")  # heavy core dep, absent in the lightweight CI

from nvflow.lib.rl.helpers import LauncherFS, _build_overlay_setup_cmd
from nvflow.lib.rl.rollout import (
    _build_client_cmd,
    _build_merge_cmd,
    _build_port_read_preamble,
    _build_vllm_wait_snippet,
    _get_remaining_jobs,
    _merged_filename,
    _output_filename,
    _ray_postprocess_installation_command,
    _vllm_port_file,
    build_aggregate_cmd,
    build_filter_cmd,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures" / "rollout"


def _load_fixture(name: str) -> str:
    """Read a snapshot fixture file as text.

    Fixtures are produced by ``scripts/_dump_rollout_fixtures.py``; this
    helper centralises the path so tests just name the snapshot.
    """
    return (FIXTURES / name).read_text()


def _client_cmd_kwargs(**overrides: object) -> dict[str, object]:
    """Return a known-good baseline kwargs dict for ``_build_client_cmd``.

    Mirrors the inputs used by ``scripts/_dump_rollout_fixtures.py``
    so the snapshot files stay in lockstep with the test inputs.  Any
    drift between this baseline and the fixture-dump script will surface
    as a snapshot mismatch on the very first run, catching it loudly.
    """
    base: dict[str, object] = {
        "output_dir": "/out/rollout",
        "gym_path": "/opt/Gym",
        "model_path": "/hf_models/Qwen/Qwen3-30B-A3B",
        "agent_name": "finance_agent",
        "input_data": "/data/train.jsonl",
        "output_file": "/out/rollout/rs0/chunk_0.jsonl",
        "done_file": "/out/rollout/rs0/chunk_0.jsonl.done",
        "config_paths": "vllm.yaml,env.yaml,overlay.yaml",
        "num_parallel": 512,
        "job_label": "rs0_chunk0",
        "policy_vllm_url": "http://policy:8000/v1",
        "judge_vllm_url": "http://judge:8001/v1",
        "judge_ng_run_overrides": (
            '    "+judge_model.responses_api_models.vllm_model.entrypoint=app.py" \\\n'
            '    "+judge_model.responses_api_models.vllm_model.base_url=http://judge:8001/v1" \\\n'
        ),
        "max_num_samples": 0,
        "chunk_id": 0,
        "num_chunks": 8,
        "responses_create_params": {"max_output_tokens": 32768, "temperature": 1.0},
    }
    base.update(overrides)
    return base


def _merge_cmd_kwargs(**overrides: object) -> dict[str, object]:
    """Baseline kwargs for ``_build_merge_cmd`` (mirrors fixture-dump script)."""
    base: dict[str, object] = {
        "gym_path": "/opt/Gym",
        "merged_file": "/out/rollout/output-rs0.jsonl",
        "analysis_dir": "/out/rollout/analysis_rs0",
        "seed_label": "rs0",
        "num_chunks": 8,
        "chunk_file_pattern": "/out/rollout/rs0/chunk_$i.jsonl",
        "merged_done_file": "/out/rollout/output-rs0.jsonl.done",
        "analyze_module": "nvflow.recipes.finance.utils.rl.analyze_rollouts",
        "enrich_module": "nvflow.recipes.finance.utils.rl.enrich_rollouts",
        "input_data": "/data/train.jsonl",
    }
    base.update(overrides)
    return base


# ===========================================================================
# Filename helpers — load-bearing string formats consumed by resume logic
# ===========================================================================


def test_output_filename_format() -> None:
    """The ``rs{seed}/chunk_{chunk_id}.jsonl`` layout is consumed by
    ``_get_remaining_jobs`` to compute ``.done`` marker paths.  A change
    here would silently break resume semantics across the whole stage.
    """
    assert _output_filename(0, 0) == "rs0/chunk_0.jsonl"
    assert _output_filename(7, 3) == "rs7/chunk_3.jsonl"
    assert _output_filename(42, 100) == "rs42/chunk_100.jsonl"


def test_merged_filename_format() -> None:
    """The ``output-rs{seed}.jsonl`` layout is consumed by both
    ``_get_remaining_jobs`` (merge-level integrity check) and
    ``_build_merge_cmd``'s atomic ``mv -f .tmp $MERGED_FILE`` step.
    Pinning the format protects both call sites simultaneously.
    """
    assert _merged_filename(0) == "output-rs0.jsonl"
    assert _merged_filename(7) == "output-rs7.jsonl"


# ===========================================================================
# vLLM port helpers
# ===========================================================================


def test_vllm_port_file_includes_slurm_job_id() -> None:
    """Port file path MUST include ``$SLURM_JOB_ID`` so concurrent jobs
    do not stomp each other's port files.  Removing the suffix would
    re-introduce the race documented in ``_vllm_port_file``'s docstring
    (client ``rm -f`` deleting the file the server just wrote).
    """
    path = _vllm_port_file("/log", "policy", "rs0_chunk0")
    assert path == "/log/.vllm_port_policy_rs0_chunk0_${SLURM_JOB_ID}.txt"
    # The ``SLURM_JOB_ID`` shell variable --
    # expanded by bash at runtime, not by Python at render time.  This is
    # intentional and load-bearing.
    assert "SLURM_JOB_ID" in path


def test_vllm_port_file_no_label_omits_underscore() -> None:
    """Empty job_label must not leave a stray ``_`` in the filename."""
    path = _vllm_port_file("/log", "judge", "")
    assert path == "/log/.vllm_port_judge_${SLURM_JOB_ID}.txt"


def test_build_port_read_preamble_returns_empty_when_no_servers() -> None:
    """No policy *and* no judge → empty preamble (no shell snippet at
    all, not just an empty assignment).  Important because the caller
    concatenates this preamble with the rest of the client cmd; a stray
    ``read_port_file`` function definition with no callers would just
    bloat the rendered script with no effect, but pinning empty makes
    the contract explicit and lets us spot accidental fall-throughs.
    """
    out = _build_port_read_preamble("/log", "rs0", has_policy=False, has_judge=False)
    assert out == ""


def test_build_port_read_preamble_policy_only() -> None:
    out = _build_port_read_preamble("/log", "rs0", has_policy=True, has_judge=False)
    assert "POLICY_PORT=$(read_port_file " in out
    assert "JUDGE_PORT=" not in out
    # Function definition must be emitted before any callers.
    assert out.index("read_port_file()") < out.index("POLICY_PORT=")


def test_build_port_read_preamble_dual() -> None:
    out = _build_port_read_preamble("/log", "rs0", has_policy=True, has_judge=True)
    assert "POLICY_PORT=$(read_port_file " in out
    assert "JUDGE_PORT=$(read_port_file " in out
    assert out.index("POLICY_PORT=") < out.index("JUDGE_PORT=")


# ===========================================================================
# vLLM wait snippet
# ===========================================================================


def test_build_vllm_wait_snippet_policy_only() -> None:
    out = _build_vllm_wait_snippet("http://h:1/v1")
    assert 'wait_for_server "http://h:1/v1/models" "Policy vLLM"' in out
    assert "Judge vLLM" not in out


def test_build_vllm_wait_snippet_with_judge() -> None:
    out = _build_vllm_wait_snippet("http://h:1/v1", "http://h:2/v1")
    assert 'wait_for_server "http://h:1/v1/models" "Policy vLLM"' in out
    assert 'wait_for_server "http://h:2/v1/models" "Judge vLLM"' in out
    # Policy must be probed before judge -- the policy is the workload's
    # critical path; if it dies while we're still waiting on the judge
    # we want to fail fast.
    assert out.index("Policy vLLM") < out.index("Judge vLLM")


def test_build_vllm_wait_snippet_uses_dummy_pid() -> None:
    """``$$`` (current shell PID) is the documented dummy -- vLLM runs
    in a separate het-group so we can't check its real PID.  Pinning
    this prevents accidental switches to ``$NG_RUN_PID`` (which is
    undefined at this point) or ``$!`` (no recent background job yet).
    """
    out = _build_vllm_wait_snippet("http://h/v1", "http://h2/v1")
    # Wait function signature: wait_for_server URL NAME PID ATTEMPTS LOG
    # so the third whitespace-separated token after the URL is the PID arg.
    for line in out.strip().split("\n"):
        if "wait_for_server" in line:
            tokens = line.split()
            # tokens[0]=wait_for_server, [1]="URL", [2]="Name", [3]=vLLM"
            # then [4]=$$, [5]=400, [6]=/dev/null
            assert "$$" in tokens, f"expected $$ as PID arg in: {line!r}"


# ===========================================================================
# _build_client_cmd snapshots
# ===========================================================================


def test_client_cmd_dual_server_snapshot() -> None:
    """Most-exercised path: policy + judge URLs, multi-chunk, with
    responses_create_params.  Pins ~9.5 KB of bash that drives every
    GRPO rollout job in production.
    """
    rendered = _build_client_cmd(**_client_cmd_kwargs())  # type: ignore[arg-type]
    assert rendered == _load_fixture("client_cmd_dual_server.txt")


def test_client_cmd_policy_only_snapshot() -> None:
    """policy_as_judge or no-judge mode: empty judge URL + empty
    overrides string.  Distinct snapshot because an empty judge_url
    suppresses the ``[ -n "$JUDGE_URL" ] && echo`` line at runtime
    but the rendered script still contains the conditional -- this
    snapshot pins that "render still emits the conditional even when
    the value is empty" contract.
    """
    rendered = _build_client_cmd(
        **_client_cmd_kwargs(judge_vllm_url="", judge_ng_run_overrides="")  # type: ignore[arg-type]
    )
    assert rendered == _load_fixture("client_cmd_policy_only.txt")


def test_client_cmd_single_chunk_snapshot() -> None:
    """``num_chunks=1`` does NOT skip the chunk-slice block at render
    time -- the ``if [ $NUM_CHUNKS -gt 1 ]`` runtime guard is what
    suppresses the slicing.  Pinning this prevents a "helpful"
    refactor from removing the runtime guard and breaking the single-
    chunk path (which would then try to slice with NUM_CHUNKS=1 and
    silently produce zero rows from the head|tail computation).
    """
    rendered = _build_client_cmd(**_client_cmd_kwargs(num_chunks=1))  # type: ignore[arg-type]
    assert rendered == _load_fixture("client_cmd_no_chunk.txt")


def test_client_cmd_max_samples_snapshot() -> None:
    """``max_num_samples=10000`` must emit ``MAX_SAMPLES=10000`` in the
    chunk-slice computation so the per-chunk size shrinks to fit the
    cap.  A regression that hardcodes 0 (the unlimited sentinel) would
    silently make ``max_num_samples`` a no-op for chunked runs.
    """
    rendered = _build_client_cmd(**_client_cmd_kwargs(max_num_samples=10000))  # type: ignore[arg-type]
    assert rendered == _load_fixture("client_cmd_max_samples.txt")
    assert "MAX_SAMPLES=10000" in rendered


def test_client_cmd_no_rcp_snapshot() -> None:
    """Empty ``responses_create_params`` must NOT emit any
    ``+responses_create_params.K=V`` continuation lines on the
    ng_collect_rollouts invocation.  Pinning this guards against an
    accidental ``+responses_create_params.={}`` leak that would
    confuse Hydra's CLI parser.
    """
    rendered = _build_client_cmd(**_client_cmd_kwargs(responses_create_params={}))  # type: ignore[arg-type]
    assert rendered == _load_fixture("client_cmd_no_rcp.txt")
    assert "+responses_create_params." not in rendered


def test_client_ray_retry_uses_ray_managed_pythonpath() -> None:
    slurm = _build_client_cmd(**_client_cmd_kwargs())  # type: ignore[arg-type]
    ray = _build_client_cmd(**_client_cmd_kwargs(is_ray=True))  # type: ignore[arg-type]

    assert "_NG_RUN_MAX_RETRIES" not in slurm
    assert "PYTHONPATH=/nemo_run/code" in slurm
    assert "_NG_RUN_MAX_RETRIES=5" in ray
    assert "PYTHONPATH=" not in ray


def test_client_cmd_uses_python3_not_python() -> None:
    """All ``python3 -m <module>`` invocations -- no bare ``python``.
    Same standardisation we pinned in test_cli_cmd: the unversioned
    ``python`` symlink is absent on some minimal images.
    """
    rendered = _build_client_cmd(**_client_cmd_kwargs())  # type: ignore[arg-type]
    # Find every ``python`` token and assert it's actually ``python3``.
    # ``find_free_port()``'s body uses ``python3 -c`` which is fine.
    for line in rendered.split("\n"):
        # Strip leading ``python``-prefixed words that are part of an
        # identifier (e.g. ``$PYTHON_PATH``).  We only care about
        # invocation tokens.
        tokens = line.split()
        for tok in tokens:
            if tok == "python":
                pytest.fail(f"bare 'python' found in rendered cmd (use python3): {line!r}")


def test_client_cmd_emits_atomic_finalize_pattern() -> None:
    """The ``cp -f $ASYNC -> $OUTPUT; touch $DONE; rm $ASYNC`` order
    is load-bearing for crash safety: if killed between the cp and
    touch, self-heal Case A restores via ``mv $OUTPUT $ASYNC``; if
    killed between touch and rm, the chunk is treated as complete and
    skipped on resume.  Reordering breaks the recovery contract.
    """
    rendered = _build_client_cmd(**_client_cmd_kwargs())  # type: ignore[arg-type]
    cp_idx = rendered.rindex('cp -f "$ASYNC_FILE" "$OUTPUT_FILE"')
    touch_idx = rendered.rindex('touch "$DONE_FILE"')
    rm_idx = rendered.rindex('rm -f "$ASYNC_FILE"')
    assert cp_idx < touch_idx < rm_idx, (
        "finalize order must be cp -> touch -> rm; reordering breaks crash recovery"
    )


def test_client_cmd_resume_early_exit_cleans_prev_orphan() -> None:
    """The resume early-exit branch (all rows already complete in
    ``-async``) must rm BOTH ``$ASYNC_FILE`` *and* ``$ASYNC_FILE.prev``.
    Without the ``.prev`` cleanup, a narrow race -- where self-heal
    Case B partially failed and left ``.prev`` on disk while ``-async``
    still had all rows -- would have the cleanup trap recreate
    ``-async`` from the stale ``.prev`` after ``.done`` was already
    touched, leaving an orphan ``-async`` on disk.  The chunk's
    ``.done`` marker still wins on the next run's done-check, so this
    is disk hygiene rather than a correctness bug, but the defensive
    cleanup avoids a confusing artifact.
    """
    rendered = _build_client_cmd(**_client_cmd_kwargs())  # type: ignore[arg-type]
    # The exact line must clean both files in a single rm -- not two
    # separate rm calls (which would be more fragile under set -e).
    assert 'rm -f "$ASYNC_FILE" "$ASYNC_FILE.prev"' in rendered


def test_client_cmd_resume_early_exit_signals_prev_merged() -> None:
    """The resume early-exit branch must set ``PREV_MERGED=1`` before
    ``exit 0`` so the cleanup trap's
    ``[ "${PREV_MERGED:-0}" -eq 0 ]`` guard short-circuits and the
    trap does NOT enter the merge-recovery branch.  Belt-and-suspenders
    with the explicit ``rm .prev`` above: either alone suppresses the
    recovery branch, but both together make the contract explicit and
    survive future trap-logic refactors.
    """
    rendered = _build_client_cmd(**_client_cmd_kwargs())  # type: ignore[arg-type]
    # Locate the early-exit block by its banner echo, then verify that
    # PREV_MERGED=1 appears before the corresponding ``exit 0``.
    early_exit_marker = "All rows already completed in -async"
    early_exit_idx = rendered.index(early_exit_marker)
    # Search forward from the marker to the next ``exit 0``; the
    # PREV_MERGED=1 must lie between them.
    exit_idx = rendered.index("exit 0", early_exit_idx)
    block = rendered[early_exit_idx:exit_idx]
    assert "PREV_MERGED=1" in block, (
        "PREV_MERGED=1 must be set in the early-exit branch before exit 0"
    )


def test_client_cmd_url_assignment_preserves_bash_expansion() -> None:
    """``VLLM_URL`` and ``JUDGE_URL`` are assembled from
    :meth:`ServerScript.hostname_ref` (which returns shell parameter
    expansions like ``${SLURM_MASTER_NODE_HET_GROUP_0:-localhost}``)
    plus dynamic port variables (``$POLICY_PORT``, ``$JUDGE_PORT``).
    Bash MUST expand those at variable assignment time so the
    resolved hostname/port flow correctly into ``ng_run
    +policy_model.responses_api_models.vllm_model.base_url=$VLLM_URL``.

    Regression guard for a real outage: a prior over-correction passed
    these URLs through :func:`shlex.quote`, which single-quoted the
    string and froze the bash variables as literal characters.
    OmegaConf inside ng_run then parsed the leaked
    ``${SLURM_MASTER_NODE_HET_GROUP_0:-localhost}`` as a config
    interpolation and crashed with ``UnsupportedInterpolationType``,
    killing every rollout job ~6 min in (after vLLM was fully loaded).

    Pin the contract: URLs must be emitted as bash double-quoted
    literals, not single-quoted, so ``$VAR`` and ``${VAR:-default}``
    expand on assignment.
    """
    het_url_policy = "http://${SLURM_MASTER_NODE_HET_GROUP_0:-localhost}:$POLICY_PORT/v1"
    het_url_judge = "http://${SLURM_MASTER_NODE_HET_GROUP_1:-localhost}:$JUDGE_PORT/v1"
    rendered = _build_client_cmd(
        **_client_cmd_kwargs(
            policy_vllm_url=het_url_policy,
            judge_vllm_url=het_url_judge,
        )  # type: ignore[arg-type]
    )

    # 1. Direct double-quoted assignments preserve $-expansion at assignment.
    assert f'VLLM_URL="{het_url_policy}"\n' in rendered
    assert f'JUDGE_URL="{het_url_judge}"\n' in rendered

    # 2. Single-quoted form would suppress expansion -- must NOT appear.
    assert f"VLLM_URL='{het_url_policy}'" not in rendered
    assert f"JUDGE_URL='{het_url_judge}'" not in rendered


def test_collect_segment_short_circuits_when_async_already_complete() -> None:
    """F6 contract: when ng_collect_rollouts exits non-zero but
    ``$ASYNC_FILE`` already contains every row of ``$REMAINING_INPUT``,
    the bash MUST treat the attempt as successful (set _NVFLOW_EXIT=0
    and break) instead of sleeping into a retry.

    Production failure that motivated this: the Nemotron-Nano smoke run
    completed all 1000 rollouts then crashed during the gym's
    post-collection ``aggregate_metrics`` HTTP call (500 Internal
    Server Error).  The bash retry loop slept 15s and tried again --
    during which pyxis tore down the container, leaving
    ``/usr/bin/sleep: No such file or directory`` and exit 127.

    With this short-circuit, an exit-on-aggregate_metrics failure with
    intact -async data finalizes correctly on the first attempt; the
    retry loop is only entered when rollouts are genuinely incomplete.
    """
    rendered = _build_client_cmd(**_client_cmd_kwargs())  # type: ignore[arg-type]

    # Locate the retry loop body.
    loop_start = rendered.index("for _attempt in $(seq 1 $_NVFLOW_MAX_RETRIES)")
    loop_end = rendered.index("done", loop_start)
    loop_body = rendered[loop_start:loop_end]

    # Three load-bearing structural elements of the F6 branch:
    # 1. The branch reads BOTH ASYNC_FILE and REMAINING_INPUT row counts.
    assert 'wc -l < "$ASYNC_FILE"' in loop_body
    assert 'wc -l < "$REMAINING_INPUT"' in loop_body
    # 2. It compares >= (so a partial -async still falls through to retry).
    assert '"$_async_rows" -ge "$_input_rows"' in loop_body
    # 3. On match it must reset _NVFLOW_EXIT to 0 AND break out of the
    #    loop -- otherwise the retry-on-non-zero logic would still fire.
    short_circuit_idx = loop_body.index('"$_async_rows" -ge "$_input_rows"')
    short_circuit_block = loop_body[short_circuit_idx : short_circuit_idx + 400]
    assert "_NVFLOW_EXIT=0" in short_circuit_block
    assert "break" in short_circuit_block
    # 4. The short-circuit must come BEFORE the retry-with-sleep block,
    #    otherwise we'd sleep first (the failure mode we're avoiding).
    sleep_idx = loop_body.index("sleep $_NVFLOW_RETRY_DELAY")
    assert short_circuit_idx < sleep_idx

    # Defensive: stderr is closed (2>&-) on the wc invocations so a
    # missing /dev/null mid-cleanup-tear-down can't break the count.
    assert 'wc -l < "$ASYNC_FILE" 2>&-' in loop_body
    assert 'wc -l < "$REMAINING_INPUT" 2>&-' in loop_body


def test_cleanup_trap_uses_close_fd_not_dev_null() -> None:
    """F5 contract: the cleanup trap MUST suppress stderr by closing
    fd 2 (``2>&-``) rather than redirecting to ``/dev/null``.

    Production failure that motivated this change: the Nemotron-Nano
    smoke run hit a gym ``aggregate_metrics`` 500 after rollouts
    completed; pyxis then tore down the container while bash was
    still in the cleanup path, causing ``2>/dev/null`` to fail with
    ``/dev/null: No such file or directory`` -- adding noise on top
    of the original failure.  ``2>&-`` doesn't depend on any path on
    the (potentially gone) container filesystem.

    Regressing this would re-introduce the same noise pattern on any
    transient container-FS issue during cleanup.
    """
    rendered = _build_client_cmd(**_client_cmd_kwargs())  # type: ignore[arg-type]
    cleanup_start = rendered.index("cleanup() {")
    cleanup_end = rendered.index("}\ntrap cleanup EXIT", cleanup_start)
    cleanup_body = rendered[cleanup_start:cleanup_end]

    # Strip shell comment lines (the explanatory comment intentionally
    # mentions the OLD ``2>/dev/null`` pattern; we only care about the
    # actual executable bash redirects).
    code_lines = [line for line in cleanup_body.split("\n") if not line.lstrip().startswith("#")]
    code = "\n".join(code_lines)

    assert "2>/dev/null" not in code, (
        "cleanup trap must use 2>&- (close fd) not 2>/dev/null; the latter "
        "fails noisily when the container's /dev/null has been unmounted."
    )
    # And there ARE 2>&- redirects (the kill/wait/scancel/kill-0 ones).
    assert code.count("2>&-") >= 4


# ===========================================================================
# _build_merge_cmd snapshots
# ===========================================================================


def test_merge_cmd_8chunks_snapshot() -> None:
    rendered = _build_merge_cmd(**_merge_cmd_kwargs())  # type: ignore[arg-type]
    assert rendered == _load_fixture("merge_cmd_8chunks.txt")


def test_merge_cmd_1chunk_snapshot() -> None:
    """Single-chunk merge differs from 8-chunk only by the ``NUM_CHUNKS=``
    inline value -- the loop logic is dynamic in shell.  Pinning both
    catches accidental Python-side branching that would render
    different shell text for the trivial case.
    """
    rendered = _build_merge_cmd(**_merge_cmd_kwargs(num_chunks=1))  # type: ignore[arg-type]
    assert rendered == _load_fixture("merge_cmd_1chunk.txt")


def test_merge_cmd_uses_atomic_replace() -> None:
    """Pin the load-bearing ``mv -f $MERGED.tmp $MERGED`` step.  The
    accompanying ``rm -f $MERGED.tmp`` cleanup happens only on the
    failure path; success path replaces atomically, so a partial run
    can never leave an in-flight merged file behind.
    """
    rendered = _build_merge_cmd(**_merge_cmd_kwargs())  # type: ignore[arg-type]
    assert '> "$MERGED_FILE.tmp"' in rendered
    assert 'mv -f "$MERGED_FILE.tmp" "$MERGED_FILE"' in rendered
    # Cleanup of stale .tmp must run *before* any writes to it.
    cleanup_idx = rendered.index('rm -f "$MERGED_FILE.tmp"')
    write_idx = rendered.index('> "$MERGED_FILE.tmp"')
    assert cleanup_idx < write_idx


def test_merge_cmd_aborts_on_missing_chunk_done() -> None:
    """Pin the load-bearing F1 fix: when a chunk ``.done`` marker is
    missing the merge bash MUST exit 1 (loud failure visible to Slurm),
    not exit 0 (silent skip).

    Silent skip was the primary origin of the silent-success cascade --
    aggregate would then glob whatever per-seed outputs existed and
    report COMPLETED 0:0 on N-1 seeds.  Re-introducing exit 0 here
    would re-open that cascade.
    """
    rendered = _build_merge_cmd(**_merge_cmd_kwargs())  # type: ignore[arg-type]
    # The precondition block must be present.
    assert '"$CHUNK_DONE"' in rendered
    # Fail-loud contract: branch contains exit 1, not exit 0.
    precondition_idx = rendered.index("Precondition: ALL chunk .done markers must exist")
    step1_idx = rendered.index("[Step 1/3] Merging chunk files")
    precondition_block = rendered[precondition_idx:step1_idx]
    assert "exit 1" in precondition_block
    assert "exit 0" not in precondition_block
    # Error message must surface in stdout for log triage.
    assert "ERROR:" in precondition_block
    assert ".done missing" in precondition_block


def test_merge_cmd_deletes_chunks_by_default() -> None:
    """Default (``keep_chunk_files`` unset/False) preserves the disk-saving
    behaviour: the merge cleanup deletes per-chunk raw files after the merged
    file is finalised.  This is the behaviour the snapshot fixtures pin; the
    explicit assert documents the contract so a refactor cannot silently flip
    it (and so the keep-flag test below has a clear counterpart).
    """
    rendered = _build_merge_cmd(**_merge_cmd_kwargs())  # type: ignore[arg-type]
    assert 'rm -f "$CHUNK_FILE"' in rendered
    assert "delete chunk data files only" in rendered
    # Finalisation (touch .done) still happens before any deletion.
    assert "touch " in rendered


def test_merge_cmd_keeps_chunks_when_flag_set() -> None:
    """``keep_chunk_files=True`` retains the per-chunk raw files so a later
    re-merge over a *larger* ``num_chunks`` (in-place cumulative growth:
    36K -> 72K -> 108K ...) can reconstruct the full merged output by
    cat-ing chunks 0..N-1.

    Without retention the merge deletes its source chunks, and a subsequent
    in-place grow's re-merge aborts with "chunk i .done exists but file
    missing/empty" (this is exactly the wall that forced the disjoint-slice
    workaround).  This test pins the flag so that future-growth capability
    cannot regress.
    """
    rendered = _build_merge_cmd(**_merge_cmd_kwargs(keep_chunk_files=True))  # type: ignore[arg-type]
    # No deletion of chunk DATA files when retention is requested.
    assert 'rm -f "$CHUNK_FILE"' not in rendered
    # Retention is announced for log triage.
    assert "keep_chunk_files=true" in rendered
    # The merge itself is unchanged: chunks are still cat-ed and the merged
    # .done marker is still touched (finalisation is independent of retention).
    assert 'cat "$CHUNK_FILE" >> "$MERGED_FILE.tmp"' in rendered
    assert "touch " in rendered


def test_merge_cmd_keep_flag_default_matches_snapshot() -> None:
    """The keep-flag default (False) must render byte-identically to the
    pinned 8-chunk snapshot — i.e. adding the flag did NOT change existing
    (delete-by-default) behaviour.  Guards against the gating refactor
    accidentally perturbing the default output that prod relies on.
    """
    explicit_false = _build_merge_cmd(**_merge_cmd_kwargs(keep_chunk_files=False))  # type: ignore[arg-type]
    default = _build_merge_cmd(**_merge_cmd_kwargs())  # type: ignore[arg-type]
    assert explicit_false == default == _load_fixture("merge_cmd_8chunks.txt")


# ===========================================================================
# build_aggregate_cmd snapshots
# ===========================================================================


def test_aggregate_cmd_default_snapshot() -> None:
    rendered = build_aggregate_cmd(
        rollout_dir="/out/rollout",
        aggregate_module="nvflow.recipes.finance.utils.rl.aggregate_seeds",
    )
    assert rendered == _load_fixture("aggregate_cmd_default.txt")


def test_aggregate_cmd_custom_filename_snapshot() -> None:
    """Custom ``difficulty_filename`` flows through to the
    ``--output_filename`` arg verbatim.  Pinning catches any
    accidental sanitisation / quoting that might mangle a non-default
    filename.
    """
    rendered = build_aggregate_cmd(
        rollout_dir="/out/rollout",
        aggregate_module="nvflow.recipes.finance.utils.rl.aggregate_seeds",
        difficulty_filename="custom_difficulty.jsonl",
    )
    assert rendered == _load_fixture("aggregate_cmd_custom_filename.txt")


def test_aggregate_cmd_emits_expected_seeds_when_passed() -> None:
    """F2 contract: when ``expected_seeds`` is plumbed through, the
    rendered bash MUST include ``--expected-seeds N`` so the underlying
    aggregate_seeds.py CLI can validate against the configured seed
    count.  Without this flag the cluster-default ``afterany`` Slurm
    dep would let aggregate run and silently shrink ``num_seeds`` in
    metrics.json after a FAILED upstream merge.
    """
    rendered = build_aggregate_cmd(
        rollout_dir="/out/rollout",
        aggregate_module="nvflow.recipes.finance.utils.rl.aggregate_seeds",
        expected_seeds=3,
    )
    assert "--expected-seeds 3" in rendered
    # Output filename remains the canonical default.
    assert '--output_filename "difficulty.jsonl"' in rendered


def test_aggregate_cmd_omits_expected_seeds_when_none() -> None:
    """Back-compat: ``expected_seeds=None`` (the default) MUST NOT
    emit the flag.  External callers (including verify.py's rejudge
    aggregate path) that don't enforce a seed count rely on this.
    """
    rendered = build_aggregate_cmd(
        rollout_dir="/out/rollout",
        aggregate_module="nvflow.recipes.finance.utils.rl.aggregate_seeds",
    )
    assert "--expected-seeds" not in rendered


# ===========================================================================
# build_filter_cmd snapshots
# ===========================================================================


def test_filter_cmd_minimal_snapshot() -> None:
    """Minimal call -- only required positional + flags.  All optional
    fields (``--validation-data``, ``--policy-model``, ``--judge-model``)
    are absent.  Pinning prevents accidental "always emit the flag with
    empty value" regressions.
    """
    rendered = build_filter_cmd(
        output_dir="/out",
        difficulty_dir="/out/rollout",
        filter_module="nvflow.recipes.finance.utils.rl.filter_training_data",
        train_data="/in/train.jsonl",
        validation_data="",
    )
    assert rendered == _load_fixture("filter_cmd_minimal.txt")
    # Negative assertions -- absent flags must NOT appear at all.
    assert "--validation-data" not in rendered
    assert "--policy-model" not in rendered
    assert "--judge-model" not in rendered


def test_filter_cmd_full_snapshot() -> None:
    """All optional fields populated.  Pins the exact ordering of the
    optional ``\\\\\\n`` continuations -- changing the order would
    produce a syntactically valid but unstable diff that obscures
    real changes.
    """
    rendered = build_filter_cmd(
        output_dir="/out",
        difficulty_dir="/out/rollout",
        filter_module="nvflow.recipes.finance.utils.rl.filter_training_data",
        train_data="/in/train.jsonl",
        validation_data="/in/val.jsonl",
        min_reward_std=1e-6,
        policy_model="/hf_models/Qwen/Qwen3-30B-A3B",
        judge_model="/hf_models/openai/gpt-oss-120b",
        train_filename="train.jsonl",
        val_filename="validation.jsonl",
        difficulty_filename="difficulty.jsonl",
        report_filename="filter_report.json",
    )
    assert rendered == _load_fixture("filter_cmd_full.txt")


def test_postprocess_pythonpath_is_backend_specific() -> None:
    slurm_merge = _build_merge_cmd(**_merge_cmd_kwargs())  # type: ignore[arg-type]
    ray_merge = _build_merge_cmd(**_merge_cmd_kwargs(is_ray=True))  # type: ignore[arg-type]
    slurm_aggregate = build_aggregate_cmd(rollout_dir="/out", aggregate_module="pkg.aggregate")
    ray_aggregate = build_aggregate_cmd(
        rollout_dir="/out", aggregate_module="pkg.aggregate", is_ray=True
    )
    slurm_filter = build_filter_cmd(
        output_dir="/out",
        difficulty_dir="/diff",
        filter_module="pkg.filter",
        train_data="/train",
        validation_data="",
    )
    ray_filter = build_filter_cmd(
        output_dir="/out",
        difficulty_dir="/diff",
        filter_module="pkg.filter",
        train_data="/train",
        validation_data="",
        is_ray=True,
    )

    for command in (slurm_merge, slurm_aggregate, slurm_filter):
        assert "PYTHONPATH=/nemo_run/code" in command
        assert "${PYTHONPATH:+$PYTHONPATH:}" not in command
    for command in (ray_merge, ray_aggregate, ray_filter):
        assert "PYTHONPATH=" not in command

    slurm_overlay = _build_overlay_setup_cmd("/model", "/overlay", {"rope": "yarn"})
    ray_overlay = _build_overlay_setup_cmd("/model", "/overlay", {"rope": "yarn"}, is_ray=True)
    assert slurm_overlay.startswith("PYTHONPATH=/nemo_run/code ")
    assert ray_overlay.startswith("python3 -m nvflow.lib.rl.create_overlay ")
    assert "PYTHONPATH=" not in ray_overlay


def test_postprocess_installation_command_is_ray_only() -> None:
    installation_command = "export PATH=/opt/gym-cli-venv/bin:$PATH"

    assert (
        _ray_postprocess_installation_command(installation_command, is_ray=True)
        == installation_command
    )
    assert _ray_postprocess_installation_command(installation_command, is_ray=False) is None
    assert _ray_postprocess_installation_command(None, is_ray=True) is None


# ===========================================================================
# _get_remaining_jobs — resume logic (file-system side effects)
# ===========================================================================


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_get_remaining_jobs_rerun_done_clears_everything(tmp_path: Path) -> None:
    """``rerun_done=True`` must (a) delete every chunk's ``.done``,
    ``-async``, ``-async.prev``, and chunk file, then (b) return all
    (seed, chunk) pairs.  Pins this contract so a refactor cannot
    silently leave stale ``.done`` markers behind that would still
    cause the next run to skip the chunk.
    """
    seeds = [0, 1]
    chunks = [0, 1]
    for s in seeds:
        for c in chunks:
            _touch(tmp_path / f"rs{s}/chunk_{c}.jsonl.done")
            _touch(tmp_path / f"rs{s}/chunk_{c}.jsonl")
            _touch(tmp_path / f"rs{s}/chunk_{c}.jsonl-async")
            _touch(tmp_path / f"rs{s}/chunk_{c}.jsonl-async.prev")

    remaining = _get_remaining_jobs(LauncherFS({}), str(tmp_path), seeds, chunks, rerun_done=True)
    assert sorted(remaining) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    for s in seeds:
        for c in chunks:
            assert not (tmp_path / f"rs{s}/chunk_{c}.jsonl.done").exists()
            assert not (tmp_path / f"rs{s}/chunk_{c}.jsonl").exists()
            assert not (tmp_path / f"rs{s}/chunk_{c}.jsonl-async").exists()
            assert not (tmp_path / f"rs{s}/chunk_{c}.jsonl-async.prev").exists()


def test_get_remaining_jobs_skips_merged_complete(tmp_path: Path) -> None:
    """Two-level integrity check: when both the merged ``.done`` AND
    the merged data file exist, the seed is skipped *even if* chunk
    output files are missing (post-merge cleanup deletes them by
    design).  Pins the "merge cleanup leaves chunk files deleted but
    keeps chunk .done markers" contract.
    """
    seeds = [0]
    chunks = [0, 1]
    # Merged complete state: data + .done both present
    _touch(tmp_path / "output-rs0.jsonl")
    _touch(tmp_path / "output-rs0.jsonl.done")
    # Chunk files deleted by merge cleanup, but .done markers preserved
    for c in chunks:
        _touch(tmp_path / f"rs0/chunk_{c}.jsonl.done")

    remaining = _get_remaining_jobs(LauncherFS({}), str(tmp_path), seeds, chunks, rerun_done=False)
    assert remaining == []


def test_get_remaining_jobs_resets_stale_merged_done(tmp_path: Path) -> None:
    """Merged ``.done`` exists but data file is missing → marker is
    stale and must be deleted, then chunk-level checks run.  This is
    the "fix 2: safe invalidation" path documented in the source.
    """
    seeds = [0]
    chunks = [0, 1]
    # Stale merged .done (no data file)
    _touch(tmp_path / "output-rs0.jsonl.done")
    # Chunks all complete
    for c in chunks:
        _touch(tmp_path / f"rs0/chunk_{c}.jsonl.done")
        _touch(tmp_path / f"rs0/chunk_{c}.jsonl")

    remaining = _get_remaining_jobs(LauncherFS({}), str(tmp_path), seeds, chunks, rerun_done=False)
    # Stale .done was unlinked
    assert not (tmp_path / "output-rs0.jsonl.done").exists()
    # All chunks have valid .done + data, so nothing left to run
    assert remaining == []


def test_get_remaining_jobs_resets_stale_chunk_done(tmp_path: Path) -> None:
    """Chunk ``.done`` exists but the chunk output file is missing
    (and no merge happened) → stale marker, must be deleted and the
    chunk re-scheduled.  This catches the "killed between writing
    output and ng_collect_rollouts moving it to final position" race.
    """
    seeds = [0]
    chunks = [0, 1]
    # Chunk 0: stale .done (no data)
    _touch(tmp_path / "rs0/chunk_0.jsonl.done")
    # Chunk 1: complete
    _touch(tmp_path / "rs0/chunk_1.jsonl.done")
    _touch(tmp_path / "rs0/chunk_1.jsonl")

    remaining = _get_remaining_jobs(LauncherFS({}), str(tmp_path), seeds, chunks, rerun_done=False)
    assert remaining == [(0, 0)]
    # Stale .done for chunk 0 was unlinked, chunk 1 .done preserved
    assert not (tmp_path / "rs0/chunk_0.jsonl.done").exists()
    assert (tmp_path / "rs0/chunk_1.jsonl.done").exists()


def test_get_remaining_jobs_mixed_state(tmp_path: Path) -> None:
    """End-to-end realistic case across two seeds:

      - rs0: merged complete (skip)
      - rs1: chunk 0 done, chunk 1 missing → only (1, 1) remaining

    The function must handle both seeds correctly in a single call --
    a refactor that processes seeds independently and returns early
    on first match would miss the rs1 chunk.
    """
    seeds = [0, 1]
    chunks = [0, 1]
    # rs0 merged complete
    _touch(tmp_path / "output-rs0.jsonl")
    _touch(tmp_path / "output-rs0.jsonl.done")
    for c in chunks:
        _touch(tmp_path / f"rs0/chunk_{c}.jsonl.done")
    # rs1: chunk 0 done, chunk 1 fresh
    _touch(tmp_path / "rs1/chunk_0.jsonl.done")
    _touch(tmp_path / "rs1/chunk_0.jsonl")

    remaining = _get_remaining_jobs(LauncherFS({}), str(tmp_path), seeds, chunks, rerun_done=False)
    assert remaining == [(1, 1)]
