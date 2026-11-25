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
"""Independent reward recomputation over canonical native Gym rollouts."""

from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from nvflow.core import console
from nvflow.execution.filesystem import TargetFilesystem
from nvflow.integrations.nemo_skills.filesystem import CONTAINER_CODE_DIR
from nvflow.integrations.nemo_skills.slurm import (
    dependency_sbatch_kwargs,
    parse_extra_sbatch_args,
)

from .compare import build_compare_command
from .contracts import RewardConfig, result_gym_overrides
from .job_scripts import _build_port_read_preamble, make_bash_script, make_server_script
from .native_commands import OverrideValue, render_gym_command, shell_variable
from .native_contract import rollout_attempt_limit
from .native_rollout import (
    _api_key,
    _flatten,
    _gym_model_config,
    _reject_legacy_layout,
    _report_options_flag,
    _validate,
)
from .operation_config import parse_reward_config
from .postprocess import (
    build_blade_script,
    build_dataset_selection_cleanup,
    build_dataset_selection_cmd,
)
from .runtime_binding import render_runtime_binding_command
from .shell_snippets import (
    SHELL_FIND_FREE_PORT,
    SHELL_STOP_BACKGROUND_PROCESS,
    SHELL_WAIT_FOR_SERVER,
)

_FALLBACK_WORKER = "nvflow.integrations.nemo_gym.workers.verify"


def _fleet_wait(log_path: str) -> str:
    probe = (
        "import json,socket,sys,urllib.parse as u;"
        "d=json.load(sys.stdin);assert d;"
        "[(lambda p: socket.create_connection((p.hostname,p.port or 80),2).close())"
        "(u.urlsplit(x['url'])) for x in d if x.get('url')];"
        "assert all(x.get('url') for x in d)"
    )
    return f"""NVFLOW_GYM_FLEET_READY=0
for _attempt in $(seq 1 720); do
  if curl -fsS "http://127.0.0.1:$NVFLOW_REJUDGE_HEAD_PORT/server_instances" 2>/dev/null \
      | python3 -c {shlex.quote(probe)}; then
    NVFLOW_GYM_FLEET_READY=1
    break
  fi
  if ! kill -0 "$NVFLOW_GYM_PID" 2>&-; then
    echo "ERROR: NeMo Gym fleet exited during startup; see {shlex.quote(log_path)}" >&2
    exit 1
  fi
  sleep 5
done
[ "$NVFLOW_GYM_FLEET_READY" -eq 1 ] || {{
  echo "ERROR: NeMo Gym fleet did not become ready" >&2
  exit 1
}}
"""


def _judge_overrides(
    params: RewardConfig,
    endpoint_url: OverrideValue | None,
) -> dict[str, OverrideValue]:
    judge = params.judge
    # `_policy_args` owns policy_model_name/base_url/api_key through Gym's
    # first-class flags. Repeating those keys here produces duplicate Hydra
    # overrides and prevents the environment from starting.
    overrides: dict[str, OverrideValue] = {
        "head_server.host": "127.0.0.1",
        "head_server.port": shell_variable("NVFLOW_REJUDGE_HEAD_PORT"),
        "skip_venv_if_present": True,
        "upload_rollouts": False,
    }
    prefix = f"policy_model.responses_api_models.{judge.model_type}"
    overrides.update(_flatten(prefix, _gym_model_config(judge)))
    overrides.update(_flatten("", params.gym_overrides))
    return overrides


def _rejudge_payload(params: RewardConfig) -> dict[str, Any]:
    return {
        "source_rollouts": params.source.rollouts.as_posix(),
        "container": params.client_container,
        "vllm_container": (
            params.vllm_container if params.judge.uses_workflow_vllm_container else None
        ),
        "installation_command": params.installation_command,
        "config_paths": list(params.config_paths),
        "agent_name": params.agent_name,
        "mode": params.mode,
        "source_responses_create_params": params.source_responses_create_params,
    }


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rejudge_config_sha256(params: RewardConfig) -> str:
    """Fingerprint every setting that can change rejudged rollout semantics.

    How many rollouts are graded at once, follow-up jobs, how the judge's server
    is sized and placed, and how long Gym waits for it are left out, so a resumed
    rejudge may change them.
    """
    return _sha256_json(
        {
            **_rejudge_payload(params),
            "schema_version": 3,
            "judge": params.judge.behavior(),
            "gym_overrides": result_gym_overrides(params.gym_overrides),
            **({"unique_prompts": True} if params.unique_prompts else {}),
        }
    )


def _legacy_rejudge_config_sha256(params: RewardConfig) -> str | None:
    """Fingerprint the settings the way rejudges started before resizing on resume recorded them.

    None with ``unique_prompts``, which no such rejudge had.
    """
    if params.unique_prompts:
        return None
    return _sha256_json(
        {
            **_rejudge_payload(params),
            "schema_version": 2,
            "concurrency": params.concurrency,
            "dependent_jobs": params.dependent_jobs,
            "judge": params.judge.legacy_behavior(),
            "gym_overrides": params.gym_overrides,
        }
    )


def _rejudge_analysis_sha256(params: RewardConfig, rejudge_sha256: str | None = None) -> str:
    payload = {
        "kind": "rejudge_analysis",
        "schema_version": 1,
        "rejudge_sha256": rejudge_sha256 or _rejudge_config_sha256(params),
        "report_module": params.report_module,
        "blade_command": params.blade_command,
    }
    if params.report_options:
        payload["report_options"] = params.report_options
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _rejudge_is_complete(
    params: RewardConfig,
    filesystem: TargetFilesystem,
) -> bool:
    """Validate judge output and source-bound behavior before pruning GPU work."""
    validate = getattr(filesystem, "batch_completion_matches", None)
    if not callable(validate):
        return False
    legacy = _legacy_rejudge_config_sha256(params)
    accepted = [legacy] if legacy else []
    common = {
        "materialized_inputs": params.artifacts.materialized_inputs.as_posix(),
        "successes": params.artifacts.rollouts.as_posix(),
        "behavior_sha256": _rejudge_config_sha256(params),
        "accepted_behavior_sha256": accepted,
        "manifest_behavior_sha256": _rejudge_config_sha256(params),
        "accepted_manifest_behavior_sha256": accepted,
        "behavior_manifest": params.behavior_manifest.as_posix(),
        "reference_completion": params.source.completion.as_posix(),
    }
    matches = validate(
        [
            {
                "materialized_inputs": params.source.materialized_inputs.as_posix(),
                "successes": params.source.rollouts.as_posix(),
                "marker": params.source.completion.as_posix(),
                "behavior_sha256": None,
                "behavior_manifest": None,
                "reference_completion": None,
            },
            {
                **common,
                "behavior_sha256": _rejudge_analysis_sha256(params),
                "accepted_behavior_sha256": [
                    _rejudge_analysis_sha256(params, sha256) for sha256 in accepted
                ],
                "materialized_inputs": (params.analyzed_artifacts.materialized_inputs.as_posix()),
                "successes": params.analyzed_artifacts.rollouts.as_posix(),
                "marker": params.analyzed_artifacts.completion.as_posix(),
                "artifacts": {
                    name: path.as_posix()
                    for name, path in params.layout.analyzed_completion_artifacts(
                        include_aggregate=False,
                        include_profiles=bool(params.dataset_selection),
                        include_evaluation=params.report_module is not None,
                        include_blade=params.blade_command is not None,
                        include_compare=True,
                    ).items()
                },
            },
            {**common, "marker": params.rejudge_marker.as_posix()},
        ]
    )
    return matches[0] and any(matches[1:])


def _policy_args(params: RewardConfig, endpoint_url: OverrideValue | None) -> list[OverrideValue]:
    args: list[OverrideValue] = ["--model", params.judge.model]
    if endpoint_url is not None:
        args.extend(("--model-url", endpoint_url, "--model-api-key", _api_key(params.judge)))
    return args


def _policy_overrides(
    params: RewardConfig,
    endpoint_url: OverrideValue | None,
) -> dict[str, OverrideValue]:
    """Supply policy identity to Gym commands that do not expose model flags."""
    overrides: dict[str, OverrideValue] = {"policy_model_name": params.judge.model}
    if endpoint_url is not None:
        overrides["policy_base_url"] = endpoint_url
        overrides["policy_api_key"] = _api_key(params.judge)
    return overrides


def _runtime_binding_paths(
    params: RewardConfig,
) -> tuple[PurePosixPath, PurePosixPath, PurePosixPath]:
    state = params.layout.initial_job_dir
    return (
        state / "gym_binding_overlay.yaml",
        state / "gym_binding.json",
        state / "gym_binding.env",
    )


def _effective_config_paths(params: RewardConfig) -> tuple[str | PurePosixPath, ...]:
    overlay, _, _ = _runtime_binding_paths(params)
    return (*params.config_paths, overlay)


def _runtime_binding_preamble(params: RewardConfig) -> str:
    overlay, binding, environment = _runtime_binding_paths(params)
    command = render_runtime_binding_command(
        config_paths=params.config_paths,
        agent_name=params.agent_name,
        output_config=overlay,
        output_binding=binding,
        output_env=environment,
        judge_model_name="policy_model",
        require_resources_server=True,
    )
    return f"{command}\n. {shlex.quote(environment.as_posix())}\n"


def _native_command(params: RewardConfig, endpoint_url: OverrideValue | None) -> str:
    return render_gym_command(
        "gym",
        "eval",
        "reverify",
        config_paths=_effective_config_paths(params),
        args=(
            "--inputs",
            params.artifacts.materialized_inputs,
            "--rollouts",
            params.source.rollouts,
            "--output",
            params.artifacts.rollouts,
            "--concurrency",
            params.concurrency,
            "--resume",
            *_policy_args(params, endpoint_url),
        ),
        overrides=_judge_overrides(params, endpoint_url),
    )


def _profile_health_complete(params: RewardConfig) -> str:
    artifacts = params.analyzed_artifacts
    profile = render_gym_command(
        "gym",
        "eval",
        "profile",
        args=(
            "--inputs",
            artifacts.materialized_inputs,
            "--rollouts",
            artifacts.rollouts,
        ),
    )
    health = render_gym_command(
        "gym",
        "eval",
        "health-check",
        args=(artifacts.rollouts.parent, "--rollouts-file", artifacts.rollouts.name),
    )
    report = (
        ""
        if params.report_module is None
        else (
            f'PYTHONPATH="{CONTAINER_CODE_DIR}${{PYTHONPATH:+:$PYTHONPATH}}" '
            f"python3 -m {shlex.quote(params.report_module)} "
            f"{shlex.quote(artifacts.rollouts.as_posix())} "
            f"{shlex.quote(artifacts.materialized_inputs.as_posix())} "
            f"{shlex.quote(params.layout.evaluation_analysis_dir.as_posix())}"
            f"{_report_options_flag(params.report_options)}\n"
        )
    )
    repeat_probe = (
        "import json,sys;"
        "indices={json.loads(line).get('_ng_rollout_index') "
        "for line in open(sys.argv[1]) if line.strip()};"
        "raise SystemExit(len(indices) < 2)"
    )
    no_profile = (
        'echo "ERROR: dataset selection requires at least two Gym repeats" >&2\nexit 1\n'
        if params.dataset_selection
        else ":\n"
    )
    profile_step = (
        f"if python3 -c {shlex.quote(repeat_probe)} "
        f"{shlex.quote(artifacts.materialized_inputs.as_posix())}; then\n"
        f"{profile}\n"
        f"else\n{no_profile}fi\n"
    )
    return f"{profile_step}{health}\n{report}"


def _completion_command(
    params: RewardConfig,
    artifacts: Any,
    action: str,
    marker: PurePosixPath,
) -> str:
    """Build completion validation for the output-local canonical bundle."""
    legacy = _legacy_rejudge_config_sha256(params)
    accepted_sha256: str | None
    if artifacts is params.analyzed_artifacts:
        behavior_sha256 = _rejudge_analysis_sha256(params)
        accepted_sha256 = _rejudge_analysis_sha256(params, legacy) if legacy else None
    else:
        behavior_sha256 = _rejudge_config_sha256(params)
        accepted_sha256 = legacy
    completion_artifacts: dict[str, PurePosixPath] = {}
    if artifacts is params.analyzed_artifacts:
        completion_artifacts = params.layout.analyzed_completion_artifacts(
            include_aggregate=False,
            include_profiles=bool(params.dataset_selection),
            include_evaluation=params.report_module is not None,
            include_blade=params.blade_command is not None,
            include_compare=True,
        )
    artifact_arguments = " ".join(
        f"--artifact {shlex.quote(f'{name}={path}')}"
        for name, path in sorted(completion_artifacts.items())
    )
    accepted_argument = (
        f"--accept-behavior-sha256 {accepted_sha256} "
        if action == "validate" and accepted_sha256
        else ""
    )
    return (
        f'PYTHONPATH="{CONTAINER_CODE_DIR}${{PYTHONPATH:+:$PYTHONPATH}}" '
        f"python3 -m nvflow.integrations.nemo_gym.native_sharding {action} "
        f"--materialized-inputs {shlex.quote(artifacts.materialized_inputs.as_posix())} "
        f"--successes {shlex.quote(artifacts.rollouts.as_posix())} "
        f"--completion {shlex.quote(marker.as_posix())} "
        f"--behavior-sha256 {behavior_sha256} "
        f"{accepted_argument}"
        f"--reference-completion {shlex.quote(params.source.completion.as_posix())} "
        f"{artifact_arguments}"
    )


def _mark_rejudge_complete(params: RewardConfig) -> str:
    return _completion_command(
        params,
        params.artifacts,
        "complete",
        params.rejudge_marker,
    )


def _validate_rejudge(params: RewardConfig) -> str:
    return _completion_command(
        params,
        params.artifacts,
        "validate",
        params.rejudge_marker,
    )


def _complete_output(params: RewardConfig) -> str:
    return _completion_command(
        params,
        params.analyzed_artifacts,
        "complete",
        params.analyzed_artifacts.completion,
    )


def _validate_done(params: RewardConfig) -> str:
    return _completion_command(
        params,
        params.analyzed_artifacts,
        "validate",
        params.analyzed_artifacts.completion,
    )


def _bind_rejudge_behavior(params: RewardConfig) -> str:
    legacy = _legacy_rejudge_config_sha256(params)
    return (
        f'PYTHONPATH="{CONTAINER_CODE_DIR}${{PYTHONPATH:+:$PYTHONPATH}}" '
        "python3 -m nvflow.integrations.nemo_gym.native_sharding bind-behavior "
        f"--manifest {shlex.quote(params.behavior_manifest.as_posix())} "
        f"--behavior-sha256 {_rejudge_config_sha256(params)} "
        f"{f'--accept-behavior-sha256 {legacy} ' if legacy else ''}"
        f"--source-completion {shlex.quote(params.source.completion.as_posix())}"
    )


def _materialize_inputs(params: RewardConfig) -> str:
    """Publish an output-local canonical copy of the source materialized inputs."""
    source = shlex.quote(params.source.materialized_inputs.as_posix())
    destination = shlex.quote(params.artifacts.materialized_inputs.as_posix())
    parent = shlex.quote(params.artifacts.materialized_inputs.parent.as_posix())
    return f"""mkdir -p {parent}
if [ ! -f {destination} ] || ! cmp -s {source} {destination}; then
  _inputs_tmp={destination}.tmp.$$
  trap 'rm -f "$_inputs_tmp"' EXIT
  cp {source} "$_inputs_tmp"
  mv -f "$_inputs_tmp" {destination}
  trap - EXIT
fi"""


def _publish_analyzed_inputs(params: RewardConfig) -> str:
    """Publish immutable job-0 outputs into the job-1 Gym bundle."""
    commands = [f"mkdir -p {shlex.quote(params.layout.analyze_outputs_dir.as_posix())}"]
    for source, destination in (
        (params.artifacts.rollouts, params.analyzed_artifacts.rollouts),
        (
            params.artifacts.materialized_inputs,
            params.analyzed_artifacts.materialized_inputs,
        ),
        (params.artifacts.failures, params.analyzed_artifacts.failures),
        (
            params.artifacts.aggregate_metrics,
            params.analyzed_artifacts.aggregate_metrics,
        ),
    ):
        quoted_source = shlex.quote(source.as_posix())
        quoted_destination = shlex.quote(destination.as_posix())
        commands.append(
            f"if [ -e {quoted_source} ] && "
            f"{{ [ ! -e {quoted_destination} ] || "
            f"! cmp -s {quoted_source} {quoted_destination}; }}; then\n"
            f"  _tmp={quoted_destination}.tmp.$$\n"
            '  rm -f "$_tmp"\n'
            f'  ln {quoted_source} "$_tmp" 2>/dev/null || '
            f'cp {quoted_source} "$_tmp"\n'
            f'  mv -f "$_tmp" {quoted_destination}\n'
            "fi"
        )
    return "\n".join(commands)


def _fallback_command(params: RewardConfig, endpoint_url: OverrideValue | None) -> str:
    overrides = _judge_overrides(params, endpoint_url)
    env_start = render_gym_command(
        "gym",
        "env",
        "start",
        config_paths=_effective_config_paths(params),
        args=_policy_args(params, endpoint_url),
        overrides=overrides,
    )
    aggregate_overrides = {
        **overrides,
        **_policy_overrides(params, endpoint_url),
        # The rejudged rollouts are both input and output; Gym won't merge a
        # file into itself, so it recomputes their metrics in place.
        "merge_shards": False,
    }
    aggregate = render_gym_command(
        "gym",
        "eval",
        "aggregate",
        config_paths=_effective_config_paths(params),
        args=(
            "--input-glob",
            params.artifacts.rollouts,
            "--output",
            params.artifacts.rollouts,
            "--no-health-check",
        ),
        overrides=aggregate_overrides,
    )
    worker = shlex.join(
        [
            "python3",
            "-m",
            _FALLBACK_WORKER,
            params.artifacts.materialized_inputs.as_posix(),
            params.source.rollouts.as_posix(),
            params.artifacts.rollouts.as_posix(),
            params.artifacts.failures.as_posix(),
            "127.0.0.1",
            "$NVFLOW_REJUDGE_HEAD_PORT",
            "$NVFLOW_GYM_RESOURCES_SERVER_INSTANCE",
            str(params.concurrency),
            *(["--unique-prompts"] if params.unique_prompts else []),
        ]
    )
    worker = worker.replace(
        "'$NVFLOW_REJUDGE_HEAD_PORT'",
        '"$NVFLOW_REJUDGE_HEAD_PORT"',
    ).replace(
        "'$NVFLOW_GYM_RESOURCES_SERVER_INSTANCE'",
        '"$NVFLOW_GYM_RESOURCES_SERVER_INSTANCE"',
    )
    log = (params.layout.initial_logs_dir / "rejudge-gym.log").as_posix()
    return f"""(
  trap - INT TERM
  exec {env_start}
) > {shlex.quote(log)} 2>&1 &
NVFLOW_GYM_PID=$!
{SHELL_STOP_BACKGROUND_PROCESS}
cleanup() {{
  stop_background_process "$NVFLOW_GYM_PID" "Gym reward environment"
}}
trap cleanup EXIT
{SHELL_WAIT_FOR_SERVER}
wait_for_server "http://127.0.0.1:$NVFLOW_REJUDGE_HEAD_PORT/" "NeMo Gym" "$NVFLOW_GYM_PID" 720 {shlex.quote(log)}
{_fleet_wait(log)}
PYTHONPATH="{CONTAINER_CODE_DIR}${{PYTHONPATH:+:$PYTHONPATH}}" {worker}
{aggregate}
cleanup
trap - EXIT
"""


def _attempt_script(
    params: RewardConfig,
    endpoint_url: OverrideValue | None,
    *,
    port_preamble: str,
) -> str:
    repairs = "\n".join(
        (
            f"[ ! -e {shlex.quote(path.as_posix())} ] || "
            f'PYTHONPATH="{CONTAINER_CODE_DIR}${{PYTHONPATH:+:$PYTHONPATH}}" '
            "python3 -m nvflow.integrations.nemo_gym.native_sharding repair "
            f"--jsonl {shlex.quote(path.as_posix())}"
        )
        for path in (params.artifacts.rollouts, params.artifacts.failures)
    )
    operation = (
        _native_command(params, endpoint_url)
        if params.mode == "native"
        else _fallback_command(params, endpoint_url)
    )
    reset = (
        "rm -rf "
        + " ".join(
            shlex.quote(path.as_posix())
            for path in (
                params.layout.initial_outputs_dir,
                params.layout.analyze_job_dir,
                params.layout.dataset_job_dir,
            )
        )
        + "\nrm -f "
        + " ".join(
            shlex.quote(path.as_posix())
            for path in (
                params.behavior_manifest,
                params.rejudge_marker,
                params.layout.profiled_dataset,
                params.layout.training_dataset,
            )
        )
    )
    orphan_guard = (
        f"if [ ! -f {shlex.quote(params.behavior_manifest.as_posix())} ] && "
        f"{{ [ -e {shlex.quote(params.artifacts.rollouts.as_posix())} ] || "
        f"[ -e {shlex.quote(params.artifacts.materialized_inputs.as_posix())} ]; }}; then\n"
        '  echo "WARNING: clearing rejudge outputs without behavior identity" >&2\n'
        f"  {reset}\n"
        "fi"
    )
    return f"""set -euo pipefail
{_reject_legacy_layout(params)}
{_validate(params.source)}
{orphan_guard}
if ! {_bind_rejudge_behavior(params)}; then
  echo "WARNING: clearing rejudge outputs with incompatible behavior identity" >&2
  {reset}
  {_bind_rejudge_behavior(params)}
fi
if [ -f {shlex.quote(params.analyzed_artifacts.completion.as_posix())} ]; then
  if {_validate_done(params)}; then
    exit 0
  fi
  echo "WARNING: removing invalid completion marker {params.analyzed_artifacts.completion}" >&2
  rm -f {shlex.quote(params.analyzed_artifacts.completion.as_posix())} \
    {shlex.quote(params.layout.dataset_completion.as_posix())}
fi
if [ -f {shlex.quote(params.rejudge_marker.as_posix())} ]; then
  if {_validate_rejudge(params)}; then
    exit 0
  fi
  echo "WARNING: removing invalid rejudge marker {params.rejudge_marker}" >&2
  rm -f {shlex.quote(params.rejudge_marker.as_posix())}
fi
{_materialize_inputs(params)}
{_validate(params.source)}
{_bind_rejudge_behavior(params)}
mkdir -p {shlex.quote(params.layout.initial_outputs_dir.as_posix())} \
  {shlex.quote(params.layout.initial_logs_dir.as_posix())}
{repairs}
{SHELL_FIND_FREE_PORT}
{port_preamble}export NEMO_GYM_MAX_ROLLOUT_ATTEMPTS={rollout_attempt_limit(params.dependent_jobs)}
NVFLOW_REJUDGE_HEAD_PORT=$(find_free_port)
cd /opt/nemo-gym
{_runtime_binding_preamble(params)}
{operation}
{_mark_rejudge_complete(params)}
"""


def _rejudge_done_guard(params: RewardConfig) -> str:
    return f"""if [ -f {shlex.quote(params.analyzed_artifacts.completion.as_posix())} ] && {_validate_done(params)}; then
  exit 0
fi
if [ -f {shlex.quote(params.rejudge_marker.as_posix())} ] && {_validate_rejudge(params)}; then
  exit 0
fi"""


def _build_attempt(
    params: RewardConfig,
    cluster_config: dict[str, Any],
    *,
    attempt: int,
    expname: str,
    dependencies: list[Any] | None,
) -> dict[str, Any]:
    from nemo_skills.pipeline.utils.declarative import Command, CommandGroup, HardwareConfig

    label = f"job-0-rejudge-rollouts-attempt-{attempt}"
    log_dir = params.layout.rejudge_attempt_logs_dir(attempt).as_posix()
    judge_server = None
    if params.judge.nvflow_deployed:
        server_config = {
            "model_path": params.judge.model,
            "num_gpus": params.judge.num_gpus,
            "server_nodes": params.judge.server_nodes,
            **params.judge.gym_model.get("vllm_serve_kwargs", {}),
        }
        judge_server = make_server_script(
            server_config,
            cluster_config,
            role="judge",
            server_type=params.judge.server_type or "vllm",
            log_dir=log_dir,
            job_label=label,
            environment=params.judge.gym_model.get("vllm_serve_env_vars"),
        )
        judge_server.set_inline(f"{_rejudge_done_guard(params)}\n{judge_server.inline}")
    endpoint_url: OverrideValue | None = params.judge.base_url
    port_preamble = _build_port_read_preamble(
        log_dir,
        label,
        judge_host=None if judge_server is None else judge_server.hostname_ref(),
    )
    if judge_server is not None:
        endpoint_url = shell_variable("NVFLOW_JUDGE_URL")
        port_preamble += 'NVFLOW_JUDGE_URL="http://$JUDGE_HOST:$JUDGE_PORT/v1"\n'
    client = make_bash_script(
        _attempt_script(params, endpoint_url, port_preamble=port_preamble),
        installation_command=params.installation_command,
    )
    commands = []
    if judge_server is not None:
        commands.append(
            Command(
                script=judge_server,
                container=(
                    params.judge.container
                    or (
                        params.vllm_container
                        if params.judge.server_type == "vllm"
                        else params.judge.server_type
                    )
                ),
                name=f"{label}-judge",
            )
        )
    commands.append(Command(script=client, container=params.client_container, name=label))
    group = CommandGroup(
        commands=commands,
        hardware=HardwareConfig(
            num_gpus=params.judge.num_gpus,
            num_nodes=params.judge.server_nodes,
            sbatch_kwargs=dependency_sbatch_kwargs(
                parse_extra_sbatch_args(cluster_config),
                continue_after_failure=attempt > 0,
            ),
        ),
        name=label,
        log_dir=log_dir,
    )
    return {
        "name": f"{expname}-{label}",
        "role": "rejudge-rollouts",
        "group": group,
        "dependencies": dependencies,
    }


def _build_analyze(
    params: RewardConfig,
    terminal: dict[str, Any] | None,
    *,
    expname: str,
    run_after: list[str] | None = None,
) -> dict[str, Any]:
    from nemo_skills.pipeline.utils.declarative import Command, CommandGroup, HardwareConfig

    blade = build_blade_script(
        command=params.blade_command,
        output_dir=params.output_dir,
        blade_output_dir=params.layout.blade_dir.as_posix(),
        rollouts=params.analyzed_artifacts.rollouts.as_posix(),
        materialized_inputs=params.analyzed_artifacts.materialized_inputs.as_posix(),
        reward_profile=params.analyzed_artifacts.reward_profiling.as_posix(),
    )
    compare = build_compare_command(
        {
            "baseline_rollouts": params.source.rollouts.as_posix(),
            "baseline_completion": params.source.completion.as_posix(),
            "candidate_rollouts": params.analyzed_artifacts.rollouts.as_posix(),
            "candidate_completion": params.rejudge_marker.as_posix(),
            "output_dir": params.layout.compare_analysis_dir.as_posix(),
            "report_format": "both",
        }
    )
    selection_evidence_cleanup = (
        f"rm -f {shlex.quote(params.layout.dataset_completion.as_posix())}\n"
        if params.dataset_selection
        else build_dataset_selection_cleanup(
            params.layout.profiled_dataset.as_posix(),
            params.layout.training_dataset.as_posix(),
            params.layout.dataset_selection_report.as_posix(),
            params.layout.dataset_completion.as_posix(),
            params.layout.unprofiled_samples.as_posix(),
        )
    )
    script = f"""set -euo pipefail
{_reject_legacy_layout(params)}
{selection_evidence_cleanup}
{_validate(params.source)}
{_bind_rejudge_behavior(params)}
if [ -f {shlex.quote(params.analyzed_artifacts.completion.as_posix())} ]; then
  if {_validate_done(params)}; then
    exit 0
  else
    echo "WARNING: rebuilding invalid rejudge completion evidence" >&2
    rm -f {shlex.quote(params.analyzed_artifacts.completion.as_posix())}
    {_validate_rejudge(params)}
  fi
else
  {_validate_rejudge(params)}
fi
{_publish_analyzed_inputs(params)}
{_profile_health_complete(params)}
{blade}
{compare}
{_complete_output(params)}
"""
    group = CommandGroup(
        commands=[
            Command(
                script=make_bash_script(
                    script,
                    installation_command=params.installation_command,
                ),
                container=params.client_container,
                name="job-1-analyze-rollouts",
            )
        ],
        hardware=HardwareConfig(
            num_gpus=0,
            sbatch_kwargs=dependency_sbatch_kwargs(
                None,
                continue_after_failure=params.dependent_jobs > 0,
            ),
        ),
        name="job-1-analyze-rollouts",
        log_dir=params.layout.analyze_logs_dir.as_posix(),
    )
    return {
        "name": f"{expname}-job-1-analyze-rollouts",
        "role": "analyze-rollouts",
        "group": group,
        "dependencies": [terminal] if terminal is not None else list(run_after or []),
    }


def _build_dataset_selection(
    params: RewardConfig,
    terminal: dict[str, Any],
    *,
    expname: str,
) -> dict[str, Any] | None:
    if not params.dataset_selection:
        return None
    from nemo_skills.pipeline.utils.declarative import Command, CommandGroup, HardwareConfig

    assert params.dataset_source is not None
    selection = dict(params.dataset_selection)
    selection["provenance"] = {
        **dict(selection.get("provenance") or {}),
        "judge_model": params.judge.model,
        "agent_name": params.agent_name,
        "source_rollouts": params.source.rollouts.as_posix(),
    }
    command = build_dataset_selection_cmd(
        source=params.dataset_source,
        reward_profile=params.analyzed_artifacts.reward_profiling.as_posix(),
        materialized_inputs=params.analyzed_artifacts.materialized_inputs.as_posix(),
        rollouts=params.analyzed_artifacts.rollouts.as_posix(),
        rollout_completion=params.analyzed_artifacts.completion.as_posix(),
        profiled_output=params.layout.profiled_dataset.as_posix(),
        training_output=params.layout.training_dataset.as_posix(),
        report_output=params.layout.dataset_selection_report.as_posix(),
        completion_output=params.layout.dataset_completion.as_posix(),
        diagnostics_output=params.layout.unprofiled_samples.as_posix(),
        selection=selection,
        responses_create_params_overrides=params.source_responses_create_params,
    )
    group = CommandGroup(
        commands=[
            Command(
                script=make_bash_script(command),
                container=params.dataset_selection_container,
                name="job-2-build-training-dataset",
            )
        ],
        hardware=HardwareConfig(num_gpus=0),
        name="job-2-build-training-dataset",
        log_dir=params.layout.dataset_logs_dir.as_posix(),
    )
    return {
        "name": f"{expname}-job-2-build-training-dataset",
        "role": "build-training-dataset",
        "group": group,
        "dependencies": [terminal],
    }


def build_reward_jobs(
    params: RewardConfig,
    cluster_config: dict[str, Any],
    expname: str,
    run_after: list[str] | None = None,
    *,
    rejudge_complete: bool = False,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    previous = None
    if not rejudge_complete:
        for attempt in range(params.dependent_jobs + 1):
            dependencies: list[Any] | None = [previous] if previous else list(run_after or [])
            previous = _build_attempt(
                params,
                cluster_config,
                attempt=attempt,
                expname=expname,
                dependencies=dependencies,
            )
            jobs.append(previous)
        assert previous is not None
    analysis = _build_analyze(
        params,
        previous,
        expname=expname,
        run_after=run_after,
    )
    jobs.append(analysis)
    selection_job = _build_dataset_selection(params, analysis, expname=expname)
    if selection_job is not None:
        jobs.append(selection_job)
    return jobs


def rejudge_operation(
    params: RewardConfig,
    cluster: str,
    expname: str,
    run_after: list[str] | None = None,
    *,
    inspect_live_state: bool = False,
    filesystem: TargetFilesystem | None = None,
) -> list[dict[str, Any]]:
    """Plan one normalized reward-recomputation operation."""
    from nvflow.integrations.nemo_skills import portable_cluster_config

    cluster_config = portable_cluster_config()
    if inspect_live_state and filesystem is None:
        from nvflow.integrations.nemo_skills import load_target_config
        from nvflow.integrations.nemo_skills.filesystem import (
            NemoSkillsTargetFilesystem,
        )

        filesystem = NemoSkillsTargetFilesystem(load_target_config(cluster))
    rejudge_complete = (
        _rejudge_is_complete(params, filesystem)
        if inspect_live_state and filesystem is not None
        else False
    )
    jobs = build_reward_jobs(
        params,
        cluster_config,
        expname,
        run_after,
        rejudge_complete=rejudge_complete,
    )
    model_paths = (
        (params.judge.model,)
        if params.judge.deployment in {"gym", "nvflow"}
        and params.judge.model is not None
        and PurePosixPath(params.judge.model).is_absolute()
        else ()
    )
    for job in jobs:
        if job.get("role") == "rejudge-rollouts":
            job["_nvflow_managed_model_paths"] = model_paths
    console.status(f"Recomputing rewards ({params.mode})")
    console.detail("Source", params.source.rollouts.as_posix())
    console.detail("Output", params.output_dir)
    if rejudge_complete:
        console.detail("Resume", "validated judge output skipped")
    console.blank()
    return jobs


def rejudge(
    config: dict[str, Any],
    cluster: str,
    expname: str,
    run_after: list[str] | None = None,
    *,
    report_module: str | None = None,
    inspect_live_state: bool = False,
    filesystem: TargetFilesystem | None = None,
) -> list[dict[str, Any]]:
    """Parse and plan one native reward-recomputation operation."""
    params = parse_reward_config(
        config,
        report_module=report_module,
    )
    return rejudge_operation(
        params,
        cluster,
        expname,
        run_after,
        inspect_live_state=inspect_live_state,
        filesystem=filesystem,
    )


__all__ = [
    "RewardConfig",
    "build_reward_jobs",
    "parse_reward_config",
    "rejudge",
    "rejudge_operation",
]
