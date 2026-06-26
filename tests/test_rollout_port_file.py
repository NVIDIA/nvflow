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
"""Regression tests for the Slurm-vs-Ray safety of the rollout bash builders.

These cover the port-file naming and cleanup trap so the generated rollout
script stays correct when ``$SLURM_JOB_ID`` is unset (e.g. under the Ray Jobs
backend): the port file must fall back to a per-job id and the cleanup trap
must guard ``scancel`` with ``command -v``.
"""

import importlib.util
import sys
import types

# ``nvflow.lib.rl.rollout`` imports two base classes from nemo-skills at module
# load.  The unit-test CI installs the project without that heavy package, so
# stub just the one submodule (a no-op when nemo-skills is actually present).
# Both functions under test are pure string builders that never touch these
# classes at call time, so the stub does not weaken the assertions.
if importlib.util.find_spec("nemo_skills") is None:

    class _StubBaseJobScript:
        def set_inline(self, *args, **kwargs):
            pass

        def __post_init__(self):
            pass

    class _StubServerScript:
        pass

    _pkg = types.ModuleType("nemo_skills")
    _pkg.__path__ = []  # mark as package
    _pipeline = types.ModuleType("nemo_skills.pipeline")
    _pipeline.__path__ = []
    _utils = types.ModuleType("nemo_skills.pipeline.utils")
    _utils.__path__ = []
    _scripts = types.ModuleType("nemo_skills.pipeline.utils.scripts")
    _scripts.BaseJobScript = _StubBaseJobScript  # type: ignore[attr-defined]
    _scripts.ServerScript = _StubServerScript  # type: ignore[attr-defined]
    sys.modules.setdefault("nemo_skills", _pkg)
    sys.modules.setdefault("nemo_skills.pipeline", _pipeline)
    sys.modules.setdefault("nemo_skills.pipeline.utils", _utils)
    sys.modules.setdefault("nemo_skills.pipeline.utils.scripts", _scripts)

from nvflow.lib.rl.rollout import (  # noqa: E402
    _build_client_cmd,
    _build_port_read_preamble,
    _server_stop_sentinel,
    _vllm_port_file,
    _wrap_server_with_dynamic_port,
)

# The literal bash fallback chain the rollout builder must emit so the port
# file is unique even when $SLURM_JOB_ID is empty (Ray Jobs backend).
_JOB_ID_FALLBACK = "${SLURM_JOB_ID:-${RAY_JOB_ID:-$$}}"


class TestVllmPortFile:
    """``_vllm_port_file`` must stay unique outside Slurm and per seed/chunk."""

    def test_contains_slurm_ray_pid_fallback(self):
        """The name carries the $SLURM_JOB_ID -> $RAY_JOB_ID -> $$ fallback."""
        path = _vllm_port_file("/logs", "policy", "rs0_chunk0")
        assert _JOB_ID_FALLBACK in path

    def test_contains_per_seed_suffix(self):
        """The per-(seed, chunk) label is embedded so seeds never collide."""
        path = _vllm_port_file("/logs", "policy", "rs3_chunk1")
        assert "rs3_chunk1" in path
        # Full expected shape: dir / role / suffix / job-id fallback.
        assert path == f"/logs/.vllm_port_policy_rs3_chunk1_{_JOB_ID_FALLBACK}.txt"

    def test_attempt_id_makes_path_unique_per_attempt(self):
        """A per-attempt id is baked into the path so re-runs never collide.

        Under Ray Mode-3 ``$SLURM_JOB_ID`` is constant for the head's whole
        life, so the job-id fallback alone cannot distinguish attempts; the
        Python-baked ``attempt_id`` must.
        """
        a = _vllm_port_file("/logs", "policy", "rs0_chunk0", "abc123")
        b = _vllm_port_file("/logs", "policy", "rs0_chunk0", "def456")
        assert "abc123" in a
        assert "def456" in b
        assert a != b
        # Same seed/chunk/role, differing only by attempt id.
        assert a == f"/logs/.vllm_port_policy_rs0_chunk0_abc123_{_JOB_ID_FALLBACK}.txt"

    def test_writer_and_reader_paths_match_within_an_attempt(self):
        """Same (role, label, attempt_id) → identical writer & reader path.

        The server (writer) and client (reader) build their paths independently;
        they must resolve to the SAME file for a given attempt or rendezvous
        fails.  Identical inputs must yield identical output.
        """
        attempt = "deadbeef0001"
        writer = _vllm_port_file("/logs", "policy", "rs0_chunk0", attempt)
        reader = _vllm_port_file("/logs", "policy", "rs0_chunk0", attempt)
        assert writer == reader

    def test_omitted_attempt_id_preserves_legacy_shape(self):
        """Callers that pass no attempt id keep the original path shape."""
        path = _vllm_port_file("/logs", "policy", "rs0_chunk0")
        assert path == f"/logs/.vllm_port_policy_rs0_chunk0_{_JOB_ID_FALLBACK}.txt"

    def test_role_is_reflected(self):
        """Policy and judge port files differ by role, not just by job id."""
        policy = _vllm_port_file("/logs", "policy", "rs0_chunk0")
        judge = _vllm_port_file("/logs", "judge", "rs0_chunk0")
        assert "policy" in policy
        assert "judge" in judge
        assert policy != judge

    def test_no_job_label_omits_suffix(self):
        """An empty label drops the suffix but keeps the job-id fallback."""
        path = _vllm_port_file("/logs", "policy", "")
        assert path == f"/logs/.vllm_port_policy_{_JOB_ID_FALLBACK}.txt"


class _FakeServerScript:
    """Minimal ServerScript stand-in exercising ``_wrap_server_with_dynamic_port``.

    The wrapper only reads ``.port`` / ``.inline`` and calls ``.set_inline``.
    """

    def __init__(self, port: int = 5000, inline: str = "vllm serve --port 5000"):
        self.port = port
        self.inline = inline

    def set_inline(self, value):
        self.inline = value


class TestServerReaderPairing:
    """The server (writer) and client (reader) must agree on the port-file path.

    They build the path independently, so for one attempt the literal path must
    be byte-identical on both sides — and it must differ across attempts so a
    Ray Mode-3 re-run (constant ``$SLURM_JOB_ID``) cannot read a stale port.
    """

    def test_writer_and_reader_use_identical_path_for_an_attempt(self):
        attempt = "cafef00d1234"
        # Reader side: the client preamble embeds the port-file path verbatim.
        preamble = _build_port_read_preamble(
            "/logs", "rs0_chunk0", has_policy=True, attempt_id=attempt
        )
        # Writer side: the server wrapper echoes the same path it writes to.
        script = _FakeServerScript()
        _wrap_server_with_dynamic_port(
            script, "policy", _vllm_port_file("/logs", "policy", "rs0_chunk0", attempt)
        )
        expected = f"/logs/.vllm_port_policy_rs0_chunk0_{attempt}_{_JOB_ID_FALLBACK}.txt"
        assert expected in preamble
        assert expected in script.inline

    def test_two_attempts_yield_different_paths(self):
        a = _build_port_read_preamble("/logs", "rs0_chunk0", has_policy=True, attempt_id="aaaa1111")
        b = _build_port_read_preamble("/logs", "rs0_chunk0", has_policy=True, attempt_id="bbbb2222")
        assert a != b


class TestClientCleanupScancelGuard:
    """The cleanup trap must not call ``scancel`` blindly outside Slurm."""

    def _build(self):
        return _build_client_cmd(
            output_dir="/out",
            gym_path="/gym",
            model_path="/model",
            agent_name="agent",
            input_data="/in.jsonl",
            output_file="/out.jsonl",
            done_file="/out.done",
            config_paths="cfg",
            num_parallel=4,
            job_label="rs0_chunk0",
            policy_vllm_url="http://host:1234/v1",
            judge_ng_run_overrides="",
        )

    def test_scancel_is_guarded_by_command_v(self):
        """``scancel`` is gated on ``command -v scancel`` + a non-empty job id."""
        script = self._build()
        assert "command -v scancel >/dev/null 2>&1" in script
        assert '[ -n "${SLURM_JOB_ID:-}" ]' in script

    def test_falls_back_to_process_group_kill(self):
        """Without Slurm the trap terminates the process group instead."""
        script = self._build()
        assert "kill 0" in script


class TestGymRayHeadAttach:
    """The gym must attach to the existing Ray cluster on the Mode-3 backend.

    NeMo-Gym calls ``ray.init()`` unconditionally; without
    ``ray_head_node_address`` it forks a SECOND Ray cluster on the node, which
    collides with the pre-provisioned one and 500s the gym head.  On Slurm there
    is no pre-provisioned cluster, so the override must be absent (the generated
    command stays byte-identical to the validated Slurm path).
    """

    def _build(self, *, is_ray: bool):
        return _build_client_cmd(
            output_dir="/out",
            gym_path="/gym",
            model_path="/model",
            agent_name="agent",
            input_data="/in.jsonl",
            output_file="/out.jsonl",
            done_file="/out.done",
            config_paths="cfg",
            num_parallel=4,
            job_label="rs0_chunk0",
            policy_vllm_url="http://host:1234/v1",
            judge_ng_run_overrides="",
            is_ray=is_ray,
        )

    def test_ray_backend_attaches_to_existing_cluster(self):
        """Ray backend injects ``+ray_head_node_address=auto`` into ng_run."""
        assert '"+ray_head_node_address=auto"' in self._build(is_ray=True)

    def test_slurm_omits_the_override(self):
        """Default/Slurm path never emits the override."""
        assert "ray_head_node_address" not in self._build(is_ray=False)

    def test_default_is_slurm_safe(self):
        """``is_ray`` defaults False so callers that don't set it stay on Slurm."""
        script = _build_client_cmd(
            output_dir="/out",
            gym_path="/gym",
            model_path="/model",
            agent_name="agent",
            input_data="/in.jsonl",
            output_file="/out.jsonl",
            done_file="/out.done",
            config_paths="cfg",
            num_parallel=4,
            job_label="rs0_chunk0",
            policy_vllm_url="http://host:1234/v1",
            judge_ng_run_overrides="",
        )
        assert "ray_head_node_address" not in script


# Per-attempt sentinel path used by the server-stop tests below.
_SENTINEL = "/logs/.server_stop_rs0_chunk0_cafef00d1234.sentinel"


class TestServerStopSentinel:
    """``_server_stop_sentinel`` rendezvous path: per-attempt, no job-id suffix.

    The rollout client (writer) and the policy/judge serve (reader) build this
    path independently, so identical inputs must yield an identical path — and
    differing attempts must yield different paths so a Ray Mode-3 re-run never
    matches a stale sentinel.  Unlike the port file it must NOT embed
    ``$SLURM_JOB_ID``/``$RAY_JOB_ID``: client and serve are SEPARATE Ray jobs.
    """

    def test_path_shape(self):
        path = _server_stop_sentinel("/logs", "rs0_chunk0", "cafef00d1234")
        assert path == _SENTINEL

    def test_no_job_id_suffix(self):
        """Client and serve are different Ray jobs — a per-job-id would diverge."""
        path = _server_stop_sentinel("/logs", "rs0_chunk0", "cafef00d1234")
        assert "SLURM_JOB_ID" not in path
        assert "RAY_JOB_ID" not in path

    def test_writer_and_reader_match_within_attempt(self):
        a = _server_stop_sentinel("/logs", "rs0_chunk0", "abc123")
        b = _server_stop_sentinel("/logs", "rs0_chunk0", "abc123")
        assert a == b

    def test_two_attempts_differ(self):
        a = _server_stop_sentinel("/logs", "rs0_chunk0", "aaaa1111")
        b = _server_stop_sentinel("/logs", "rs0_chunk0", "bbbb2222")
        assert a != b


def _client(*, is_ray: bool, stop_sentinel: str = ""):
    return _build_client_cmd(
        output_dir="/out",
        gym_path="/gym",
        model_path="/model",
        agent_name="agent",
        input_data="/in.jsonl",
        output_file="/out.jsonl",
        done_file="/out.done",
        config_paths="cfg",
        num_parallel=4,
        job_label="rs0_chunk0",
        policy_vllm_url="http://host:1234/v1",
        judge_ng_run_overrides="",
        is_ray=is_ray,
        stop_sentinel=stop_sentinel,
    )


class TestServerStopRayClientTrap:
    """(a) Ray client touches the sentinel UNCONDITIONALLY in its cleanup trap.

    On a successful rollout nothing reaps the separate Ray serve job, so the
    client must signal it to stop on EVERY exit path — both success and failure.
    """

    def test_ray_client_touches_sentinel(self):
        script = _client(is_ray=True, stop_sentinel=_SENTINEL)
        assert f'touch "{_SENTINEL}"' in script

    def test_touch_lives_inside_the_exit_trap(self):
        """The touch must be in ``cleanup()`` (runs on EXIT, success OR failure),
        not gated behind the non-zero-exit branch."""
        script = _client(is_ray=True, stop_sentinel=_SENTINEL)
        cleanup_body = script.split("cleanup() {", 1)[1].split("trap cleanup EXIT", 1)[0]
        assert f'touch "{_SENTINEL}"' in cleanup_body
        # And it must precede the non-zero-exit handler so success exits touch it.
        touch_pos = cleanup_body.index(f'touch "{_SENTINEL}"')
        nonzero_pos = cleanup_body.index("if [ $_nvflow_exit -ne 0 ]")
        assert touch_pos < nonzero_pos


class TestServerStopRayServeWatcher:
    """(a) Ray serve backgrounds vLLM and polls for the stop sentinel."""

    def _wrap(self, *, is_ray: bool, stop_sentinel: str = ""):
        script = _FakeServerScript()
        _wrap_server_with_dynamic_port(
            script,
            "policy",
            _vllm_port_file("/logs", "policy", "rs0_chunk0", "cafef00d1234"),
            is_ray=is_ray,
            stop_sentinel=stop_sentinel,
        )
        return script.inline

    def test_ray_serve_backgrounds_and_polls(self):
        inline = self._wrap(is_ray=True, stop_sentinel=_SENTINEL)
        assert " &\n" in inline  # vLLM launched in background
        assert f'if [ -f "{_SENTINEL}" ]' in inline  # polls for sentinel
        assert "exit 0" in inline  # exits SUCCEEDED when sentinel appears

    def test_ray_serve_launches_under_setsid(self):
        """vLLM is launched as its own session/process-group leader via setsid.

        ``setsid`` makes the child's PGID == its PID so the watcher can signal
        the whole vLLM tree (EngineCore + Worker_TP* children) with a negative
        PID without killing itself.
        """
        inline = self._wrap(is_ray=True, stop_sentinel=_SENTINEL)
        assert "command -v setsid" in inline  # availability check
        assert 'setsid bash -c "$VLLM_RUN_CMD" &' in inline

    def test_ray_serve_group_kills_on_sentinel(self):
        """On the sentinel the watcher signals the whole PROCESS GROUP.

        Negative-PID kills (``kill -TERM -<pgid>`` / ``kill -KILL -<pgid>``)
        target every member of vLLM's process group so orphaned GPU-holding
        workers die before this serve job exits — fixing the downstream OOM.
        """
        inline = self._wrap(is_ray=True, stop_sentinel=_SENTINEL)
        assert 'kill -TERM -"$_NVFLOW_VLLM_PID"' in inline  # SIGTERM the group
        assert 'kill -KILL -"$_NVFLOW_VLLM_PID"' in inline  # escalate to SIGKILL

    def test_ray_serve_term_then_kill_escalation_order(self):
        """SIGTERM precedes a grace sleep which precedes SIGKILL (graceful first)."""
        inline = self._wrap(is_ray=True, stop_sentinel=_SENTINEL)
        body = inline.split("_nvflow_kill_vllm()", 1)[1]
        term_pos = body.index('kill -TERM -"$_NVFLOW_VLLM_PID"')
        kill_pos = body.index('kill -KILL -"$_NVFLOW_VLLM_PID"')
        sleep_pos = body.index("sleep 5", term_pos)
        assert term_pos < sleep_pos < kill_pos

    def test_ray_serve_settles_after_kill_for_gpu_free(self):
        """A settle ``sleep`` follows the kill so CUDA frees memory before exit.

        The next colocated stage (GRPO training) must not start engine init
        while the just-killed workers are still releasing ~74 GiB of GPU memory.
        """
        inline = self._wrap(is_ray=True, stop_sentinel=_SENTINEL)
        kill_fn = inline.split("_nvflow_kill_vllm() {", 1)[1].split("}\n", 1)[0]
        # The final statement of the kill helper is a settle sleep, after wait.
        assert 'wait "$_NVFLOW_VLLM_PID"' in kill_fn
        wait_pos = kill_fn.index('wait "$_NVFLOW_VLLM_PID"')
        assert "sleep 5" in kill_fn[wait_pos:]

    def test_ray_serve_does_not_mask_real_crash(self):
        """A vLLM that dies on its own propagates its non-zero exit code.

        The self-crash branch may best-effort reap surviving group members, but
        must still ``exit`` with vLLM's real (possibly non-zero) code.
        """
        inline = self._wrap(is_ray=True, stop_sentinel=_SENTINEL)
        assert "exit $_NVFLOW_VLLM_RC" in inline
        # The self-crash branch must NOT clobber the captured real code with 0.
        crash_branch = inline.split('if ! kill -0 "$_NVFLOW_VLLM_PID"', 1)[1]
        assert "exit $_NVFLOW_VLLM_RC" in crash_branch
        assert "exit 0" not in crash_branch

    def test_ray_serve_falls_back_when_setsid_absent(self):
        """Defensive fallback: plain background launch + single-PID kill."""
        inline = self._wrap(is_ray=True, stop_sentinel=_SENTINEL)
        # Fallback branch keeps the legacy eval-background launch...
        assert 'eval "$VLLM_RUN_CMD" &' in inline
        # ...and single-process (non-negative) kills in the GROUP=0 path.
        assert 'kill -TERM "$_NVFLOW_VLLM_PID"' in inline
        assert 'kill -KILL "$_NVFLOW_VLLM_PID"' in inline


class TestServerStopSlurmByteIdentical:
    """(b) Slurm backend emits NEITHER the sentinel touch nor the watcher.

    The generated command must be byte-identical to the pre-fix Slurm path, so
    the validated Slurm behaviour (allocation teardown reaps the serve) is
    untouched.  The build-time gate is ``is_ray and stop_sentinel``.
    """

    def test_slurm_client_has_no_sentinel(self):
        script = _client(is_ray=False, stop_sentinel=_SENTINEL)
        assert "server_stop" not in script
        assert "server-stop sentinel" not in script

    def test_slurm_client_byte_identical_to_default(self):
        """is_ray=False (even with a sentinel passed) == the legacy default call."""
        default = _build_client_cmd(
            output_dir="/out",
            gym_path="/gym",
            model_path="/model",
            agent_name="agent",
            input_data="/in.jsonl",
            output_file="/out.jsonl",
            done_file="/out.done",
            config_paths="cfg",
            num_parallel=4,
            job_label="rs0_chunk0",
            policy_vllm_url="http://host:1234/v1",
            judge_ng_run_overrides="",
        )
        assert _client(is_ray=False, stop_sentinel=_SENTINEL) == default

    def test_slurm_serve_foreground_byte_identical(self):
        """The serve wrapper on Slurm equals the legacy (no-kwargs) wrapper.

        Also asserts the Slurm branch emits NEITHER setsid NOR any group-kill,
        so the validated foreground Slurm path is untouched by the refinement.
        """
        legacy = _FakeServerScript()
        _wrap_server_with_dynamic_port(legacy, "policy", "/logs/pf.txt")
        slurm = _FakeServerScript()
        _wrap_server_with_dynamic_port(
            slurm, "policy", "/logs/pf.txt", is_ray=False, stop_sentinel=_SENTINEL
        )
        assert slurm.inline == legacy.inline
        assert "server_stop" not in slurm.inline
        assert "setsid" not in slurm.inline
        assert "kill -TERM" not in slurm.inline
        assert "kill -KILL" not in slurm.inline
        assert "_NVFLOW_VLLM_PID" not in slurm.inline

    def test_slurm_serve_byte_identical_to_pre_refinement_baseline(self):
        """The Slurm serve bash matches the exact pre-refinement literal.

        Pinning the literal guards against any accidental drift of the
        foreground Slurm path while the Ray watcher evolves.
        """
        baseline = (
            "find_free_port() {\n"
            '    python3 -c "import socket; s=socket.socket(); '
            "s.bind(('',0)); print(s.getsockname()[1]); s.close()\"\n"
            "}\n"
            "\n"
            "VLLM_PORT=$(find_free_port)\n"
            'echo "[dynamic-port] policy vLLM using port $VLLM_PORT '
            '(node ${SLURM_NODEID:-0})"\n'
            'if [ "${SLURM_NODEID:-0}" = "0" ]; then\n'
            '    echo "$VLLM_PORT" > "/logs/pf.txt"\n'
            "fi\n"
            "ORIG_CMD=$(cat <<'__NVFLOW_VLLM_CMD__'\n"
            "vllm serve --port 5000\n"
            "__NVFLOW_VLLM_CMD__\n"
            ")\n"
            'eval "$(echo "$ORIG_CMD" | sed "s/5000/$VLLM_PORT/g")"\n'
        )
        slurm = _FakeServerScript()
        _wrap_server_with_dynamic_port(
            slurm, "policy", "/logs/pf.txt", is_ray=False, stop_sentinel=_SENTINEL
        )
        assert slurm.inline == baseline


class TestServerStopPathRendezvous:
    """(c) The sentinel path matches between serve and client for an attempt."""

    def test_client_touch_and_serve_poll_use_identical_path(self):
        attempt = "cafef00d1234"
        sentinel = _server_stop_sentinel("/logs", "rs0_chunk0", attempt)
        # Client (writer) side.
        client = _client(is_ray=True, stop_sentinel=sentinel)
        # Serve (reader) side.
        script = _FakeServerScript()
        _wrap_server_with_dynamic_port(
            script,
            "policy",
            _vllm_port_file("/logs", "policy", "rs0_chunk0", attempt),
            is_ray=True,
            stop_sentinel=sentinel,
        )
        assert f'touch "{sentinel}"' in client
        assert f'if [ -f "{sentinel}" ]' in script.inline
