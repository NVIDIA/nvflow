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
"""Tests for nvflow.lib.rl.verify — re-judge (compute_rewards) bash builders.

Pins the pure-function rendering of ``_build_verify_cmd`` and
``_build_analysis_cmd`` so future refactors of ``verify.py`` (and any
shared-helper extraction with ``rollout.py``) can detect:

  - **Byte-level regressions** in rendered shell scripts via fixture
    snapshots under ``tests/fixtures/verify/``.  These scripts run
    inside Slurm jobs; even a whitespace shift can change exit-code
    semantics, so byte-exact pinning is the right guard.

  - **Behavioral regressions** in the verify-cmd structure (banner,
    step ordering, finalize sequence, atomic rename) and the
    analysis-cmd entry handling (per-seed block + default first_file
    fallback for the empty-entries edge case).

Refresh fixtures via ``uv run python3 scripts/_dump_verify_fixtures.py``
and inspect ``git diff tests/fixtures/verify/`` carefully before
committing.  Any unintended diff is a regression.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("nemo_skills")  # heavy core dep, absent in the lightweight CI

from nvflow.lib.rl.verify import _build_analysis_cmd, _build_verify_cmd

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures" / "verify"


def _load_fixture(name: str) -> str:
    """Read a snapshot fixture file as text."""
    return (FIXTURES / name).read_text()


_LOCAL_VLLM_OVERRIDES = (
    '    "+judge_model.responses_api_models.vllm_model.entrypoint=app.py" \\\n'
    '    "+judge_model.responses_api_models.vllm_model.base_url=http://127.0.0.1:$JUDGE_PORT/v1" \\\n'
    '    "+judge_model.responses_api_models.vllm_model.api_key=EMPTY" \\\n'
    '    "+judge_model.responses_api_models.vllm_model.model=/hf_models/openai/gpt-oss-120b" \\\n'
    '    "+judge_model.responses_api_models.vllm_model.return_token_id_information=false" \\\n'
    '    "+judge_model.responses_api_models.vllm_model.uses_reasoning_parser=true" \\\n'
    '    "+finance_env.resources_servers.finance_env.judge_model_server.name=judge_model" \\\n'
)

_OPENAI_OVERRIDES = (
    '    "+judge_model.responses_api_models.openai_model.base_url=https://api.openai.com/v1" \\\n'
    '    "+judge_model.responses_api_models.openai_model.api_key_env_var=OPENAI_API_KEY" \\\n'
    '    "+judge_model.responses_api_models.openai_model.model=gpt-4o-mini" \\\n'
    '    "+finance_env.resources_servers.finance_env.judge_model_server.name=judge_model" \\\n'
)


def _verify_cmd_kwargs(**overrides: object) -> dict[str, object]:
    """Baseline kwargs for ``_build_verify_cmd`` (mirrors fixture-dump script)."""
    base: dict[str, object] = {
        "output_dir": "/out/verify",
        "gym_path": "/opt/Gym",
        "input_file": "/in/rollouts/output-rs0.jsonl",
        "output_file": "/out/verify/rejudge/output-rs0.jsonl",
        "done_file": "/out/verify/rejudge/output-rs0.jsonl.done",
        "config_paths": "vllm.yaml,env.yaml,overlay.yaml",
        "num_parallel": 8,
        "job_label": "rejudge_rs0",
        "judge_mode": "local_vllm",
        "environment_name": "finance_env",
        "judge_ng_run_overrides": _LOCAL_VLLM_OVERRIDES,
    }
    base.update(overrides)
    return base


def _analysis_cmd_kwargs(**overrides: object) -> dict[str, object]:
    """Baseline kwargs for ``_build_analysis_cmd`` (mirrors fixture-dump script)."""
    base: dict[str, object] = {
        "rejudge_dir": "/out/verify/rejudge",
        "gym_path": "/opt/Gym",
        "analyze_module": "nvflow.recipes.finance.utils.rl.analyze_rollouts",
        "analysis_entries": [
            ("rs0", "/out/verify/rejudge/output-rs0.jsonl"),
            ("rs1", "/out/verify/rejudge/output-rs1.jsonl"),
        ],
    }
    base.update(overrides)
    return base


# ===========================================================================
# _build_verify_cmd snapshots
# ===========================================================================


def test_verify_cmd_local_judge_snapshot() -> None:
    """Most-exercised path: local-vLLM judge with full ng_run override
    block (entrypoint + base_url + api_key + model + reasoning_parser
    + judge_model_server.name).  Pins ~3.3 KB of bash that drives every
    re-judge job in production for local-vLLM judges.
    """
    rendered = _build_verify_cmd(**_verify_cmd_kwargs())  # type: ignore[arg-type]
    assert rendered == _load_fixture("verify_cmd_local_judge.txt")


def test_verify_cmd_openai_judge_snapshot() -> None:
    """OpenAI-API judge: override block uses ``openai_model.*`` keys
    instead of ``vllm_model.*``.  Distinct snapshot because the verify
    script's ``base_url=http://localhost:0/v1`` (policy stub) is
    unchanged across modes -- only the ``judge_ng_run_overrides``
    block differs.  Pinning protects against an accidental "smarten
    up the policy stub" refactor that would break openai-judge mode.
    """
    rendered = _build_verify_cmd(
        **_verify_cmd_kwargs(  # type: ignore[arg-type]
            judge_mode="openai", judge_ng_run_overrides=_OPENAI_OVERRIDES
        )
    )
    assert rendered == _load_fixture("verify_cmd_openai_judge.txt")


# ===========================================================================
# _build_verify_cmd structural / behavioral checks
# ===========================================================================


def test_verify_cmd_has_section_ordering() -> None:
    """Sections must appear in this order: variables -> setup ->
    banner -> step1 (ng_run) -> step2 (verify_worker) -> finalize.
    The trap-on-cleanup pattern depends on this order: ``trap cleanup
    EXIT`` is set in setup, then any later failure (server start,
    rejudge) triggers it.  Reordering would break crash-cleanup.
    """
    rendered = _build_verify_cmd(**_verify_cmd_kwargs())  # type: ignore[arg-type]
    markers = [
        ("variables", 'OUTPUT_DIR="/out/verify"'),
        ("setup", "trap cleanup EXIT"),
        ("banner", "Compute Rewards (re-judge)"),
        ("step1_ng_run", "[Step 1/2] Starting NeMo-Gym servers"),
        ("step2_rejudge", "[Step 2/2] Re-judging rollouts"),
        ("finalize", 'mv "$OUTPUT_FILE-async" "$OUTPUT_FILE"'),
    ]
    indices = [(name, rendered.find(marker)) for name, marker in markers]
    for name, idx in indices:
        assert idx != -1, f"missing section {name!r}"
    positions = [idx for _, idx in indices]
    assert positions == sorted(positions), f"sections out of order: {indices}"


def test_verify_cmd_uses_verify_worker_not_collect() -> None:
    """The re-judge path MUST invoke ``verify_worker`` (not the
    rollout-side ``collect_rollouts.py`` worker).  A copy-paste from
    rollout.py would silently call the wrong worker and re-roll
    instead of re-judging.  Pin the module name explicitly.
    """
    rendered = _build_verify_cmd(**_verify_cmd_kwargs())  # type: ignore[arg-type]
    assert "nvflow.lib.rl.verify_worker" in rendered
    assert "collect_rollouts" not in rendered


def test_verify_cmd_finalize_is_atomic_rename_then_done() -> None:
    """Output write contract: rejudge writes to ``$OUTPUT_FILE-async``,
    then ``mv`` to the canonical ``$OUTPUT_FILE``, then ``touch``
    ``$DONE_FILE``.  The ``-async`` suffix protects against partial
    writes being mistaken for complete output by a resume scan, and
    the ``mv`` precedes the ``.done`` touch so any reader observing
    ``.done`` is guaranteed to see a complete file.  Pinning this
    sequence prevents an accidental flip of the order.
    """
    rendered = _build_verify_cmd(**_verify_cmd_kwargs())  # type: ignore[arg-type]
    mv_idx = rendered.index('mv "$OUTPUT_FILE-async" "$OUTPUT_FILE"')
    touch_idx = rendered.index('touch "$DONE_FILE"')
    assert mv_idx < touch_idx, "mv must precede touch .done for resume-safety"


def test_verify_cmd_banner_contains_judge_mode_literal() -> None:
    """The ``judge_mode`` argument is interpolated *literally* into
    the banner echo (``echo "Judge mode:   {judge_mode}"``).  This
    test pins the literal-vs-shell-variable choice -- using a shell
    var here would require an ``ENVIRONMENT_NAME``-style declaration
    earlier in the script, which is currently not emitted.
    """
    rendered = _build_verify_cmd(**_verify_cmd_kwargs(judge_mode="external_vllm"))  # type: ignore[arg-type]
    assert 'echo "Judge mode:   external_vllm"' in rendered


def test_verify_cmd_step1_starts_servers_in_background() -> None:
    """``ng_run ... &`` must run in the background so the script can
    proceed to ``wait_for_server`` and then run ``verify_worker``.
    Removing the ``&`` would block the script forever waiting on the
    head server's stdout.  ``NG_RUN_PID=$!`` captures the PID for
    cleanup-on-trap.
    """
    rendered = _build_verify_cmd(**_verify_cmd_kwargs())  # type: ignore[arg-type]
    assert " 2>&1 &\nNG_RUN_PID=$!" in rendered


def test_verify_cmd_does_not_hardcode_venv_activation() -> None:
    """The verify script must NOT hardcode ``source .venv/bin/activate``.

    The Gym CLI on PATH is provided by the stage's ``installation_command``
    (baked ``.venv`` on nemo-rl, or ``/opt/gym-cli-venv`` on PATH for the CPU
    nemo-gym image). A hardcoded single-``.venv`` activation fails on nemo-gym
    (there is no ``/opt/Gym/.venv`` -- venvs are per-component under
    ``/opt/gym-venvs``), so it was removed.
    """
    rendered = _build_verify_cmd(**_verify_cmd_kwargs())  # type: ignore[arg-type]
    assert "source .venv/bin/activate" not in rendered


def test_verify_cmd_points_ng_run_at_uv_venv_dir() -> None:
    """``ng_run`` must receive ``+uv_venv_dir=$UV_VENV_DIR`` so it reuses the
    baked per-component venvs; ``UV_VENV_DIR`` is set in the variables segment
    (defaults to ``$GYM_PATH``; the recipe overrides it to ``/opt/gym-venvs``)."""
    rendered = _build_verify_cmd(**_verify_cmd_kwargs())  # type: ignore[arg-type]
    assert "UV_VENV_DIR=" in rendered
    assert '"+uv_venv_dir=$UV_VENV_DIR"' in rendered


# ===========================================================================
# _build_analysis_cmd snapshots + structural
# ===========================================================================


def test_analysis_cmd_multi_seed_snapshot() -> None:
    """Two-seed common case: each entry emits its own
    ``echo "Analyzing rsN ..." ; python3 -m {analyze_module} ...``
    block, then a shared epilogue prints the ng_viewer hint pointing
    at the *first* entry.  Pins the per-seed loop body + epilogue
    composition.
    """
    rendered = _build_analysis_cmd(**_analysis_cmd_kwargs())  # type: ignore[arg-type]
    assert rendered == _load_fixture("analysis_cmd_multi_seed.txt")


def test_analysis_cmd_single_seed_snapshot() -> None:
    """Single-seed: one block + epilogue.  Distinct snapshot to pin
    that the seed-loop emits exactly one block (not zero, not two)
    when given a one-element list.
    """
    rendered = _build_analysis_cmd(
        **_analysis_cmd_kwargs(  # type: ignore[arg-type]
            analysis_entries=[("rs0", "/out/verify/rejudge/output-rs0.jsonl")]
        )
    )
    assert rendered == _load_fixture("analysis_cmd_single_seed.txt")


def test_analysis_cmd_empty_entries_uses_default_first_file() -> None:
    """Empty ``analysis_entries`` (no remaining seeds) must still
    render a valid script -- the epilogue's ng_viewer hint falls back
    to ``{rejudge_dir}/output-rs0.jsonl`` so the printed command is
    still copy-pasteable.  Pinning the fallback prevents an
    ``IndexError`` regression on the empty-list path.
    """
    rendered = _build_analysis_cmd(**_analysis_cmd_kwargs(analysis_entries=[]))  # type: ignore[arg-type]
    assert rendered == _load_fixture("analysis_cmd_empty_entries.txt")
    assert "Analyzing" not in rendered, "empty-entries must not emit per-seed blocks"
    assert "/out/verify/rejudge/output-rs0.jsonl" in rendered, (
        "fallback first_file must be present in epilogue"
    )


def test_analysis_cmd_emits_one_block_per_entry() -> None:
    """Each entry in ``analysis_entries`` must emit exactly one
    ``Analyzing {seed_label}`` block.  A regression that loses the
    per-entry append (e.g., overwriting ``parts`` instead of
    appending) would silently skip seeds without erroring.
    """
    entries = [
        ("rs0", "/out/r/output-rs0.jsonl"),
        ("rs1", "/out/r/output-rs1.jsonl"),
        ("rs2", "/out/r/output-rs2.jsonl"),
    ]
    rendered = _build_analysis_cmd(**_analysis_cmd_kwargs(analysis_entries=entries))  # type: ignore[arg-type]
    for label, _ in entries:
        assert f'echo "Analyzing {label} ..."' in rendered
    assert rendered.count('echo "Analyzing ') == len(entries)


def test_analysis_cmd_uses_provided_analyze_module() -> None:
    """The ``analyze_module`` argument must be interpolated literally
    into ``python3 -m {analyze_module}``.  A regression that hardcodes
    a module name would silently invoke the wrong analysis script.
    """
    rendered = _build_analysis_cmd(
        **_analysis_cmd_kwargs(analyze_module="my.custom.analyzer")  # type: ignore[arg-type]
    )
    assert "python3 -m my.custom.analyzer" in rendered
    assert "analyze_rollouts" not in rendered  # default must NOT leak in
