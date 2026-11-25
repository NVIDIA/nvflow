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
"""Thin NVFlow control plane for native NeMo Gym rollout collection."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from nemo_skills.pipeline.utils.scripts import BaseJobScript, ServerScript

from nvflow.core import console
from nvflow.execution.filesystem import TargetFilesystem
from nvflow.integrations.nemo_skills.filesystem import CONTAINER_CODE_DIR
from nvflow.integrations.nemo_skills.slurm import (
    dependency_sbatch_kwargs,
    parse_extra_sbatch_args,
)

from .contracts import (
    ModelEndpoint,
    NativeRolloutConfigError,
    RewardConfig,
    RolloutConfig,
    result_gym_overrides,
)
from .job_scripts import _build_port_read_preamble, make_bash_script, make_server_script
from .native_commands import (
    ArgumentValue,
    OverrideValue,
    hydra_expression,
    render_gym_command,
    shell_variable,
)
from .native_contract import (
    NATIVE_GYM_SOURCE_ROOT,
    NativeGymArtifactPaths,
    rollout_attempt_limit,
)
from .operation_config import parse_model_endpoint, parse_rollout_config
from .postprocess import (
    build_blade_script,
    build_dataset_selection_cleanup,
    build_dataset_selection_cmd,
)
from .runtime_binding import render_runtime_binding_command
from .shell_snippets import SHELL_FIND_FREE_PORT, SHELL_STOP_BACKGROUND_PROCESS

_SHARD_MODULE = "nvflow.integrations.nemo_gym.native_sharding"


def _hydra_collection(value: object, path: str) -> str:
    if isinstance(value, Mapping):
        entries = []
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise NativeRolloutConfigError(f"{path!r} contains an invalid key")
            entries.append(f"{key}:{_hydra_collection(item, f'{path}.{key}')}")
        return "{" + ",".join(entries) + "}"
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return "[" + ",".join(_hydra_collection(item, path) for item in value) + "]"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, int | float):
        return str(value)
    raise NativeRolloutConfigError(
        f"Gym override {path!r} has unsupported value {type(value).__name__}"
    )


def _flatten(prefix: str, value: Mapping[str, Any]) -> dict[str, OverrideValue]:
    flattened: dict[str, OverrideValue] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise NativeRolloutConfigError(f"{prefix!r} contains an invalid key")
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping):
            if item:
                flattened.update(_flatten(path, item))
            else:
                flattened[path] = hydra_expression("{}")
        elif isinstance(item, Sequence) and not isinstance(item, str | bytes):
            flattened[path] = hydra_expression(_hydra_collection(item, path))
        elif item is None or isinstance(item, str | int | float | bool):
            flattened[path] = item
        else:
            raise NativeRolloutConfigError(
                f"Gym override {path!r} has unsupported value {type(item).__name__}"
            )
    return flattened


def _api_key(endpoint: ModelEndpoint) -> OverrideValue:
    return shell_variable(endpoint.api_key_env) if endpoint.api_key_env else "EMPTY"


def _gym_model_config(endpoint: ModelEndpoint) -> dict[str, Any]:
    """Return Gym-native fields, excluding NVFlow server-launch settings."""
    config = dict(endpoint.gym_model)
    if not endpoint.gym_deployed:
        config.pop("vllm_serve_kwargs", None)
        config.pop("vllm_serve_env_vars", None)
        config.pop("show_vllm_engine_stats", None)
        config.pop("debug", None)
    return config


def _model_overrides(
    params: RolloutConfig,
    policy_url: OverrideValue | None,
    judge_url: OverrideValue | None,
) -> dict[str, OverrideValue]:
    policy = params.policy
    overrides: dict[str, OverrideValue] = {
        "head_server.host": "127.0.0.1",
        "head_server.port": shell_variable("NVFLOW_ROLLOUT_HEAD_PORT"),
        "policy_model_name": policy.model,
        "skip_venv_if_present": True,
        "reuse_existing_data_preparation": True,
        "num_repeats_add_seed": True,
        "route_failures_to_sidecar": True,
        "upload_rollouts": False,
    }
    if policy_url is not None:
        overrides["policy_base_url"] = policy_url
        overrides["policy_api_key"] = _api_key(policy)
    prefix = f"policy_model.responses_api_models.{policy.model_type}"
    overrides.update(_flatten(prefix, _gym_model_config(policy)))
    overrides.update(_flatten("", params.gym_overrides))

    judge = params.judge
    if judge is None:
        return overrides
    judge_type = judge.model_type
    assert judge_type is not None
    prefix = f"judge_model.responses_api_models.{judge_type}"
    overrides[f"{prefix}.entrypoint"] = "app.py"
    if judge_type == "openai_model":
        overrides[f"{prefix}.openai_model"] = judge.model
        overrides[f"{prefix}.openai_base_url"] = judge_url
        overrides[f"{prefix}.openai_api_key"] = _api_key(judge)
    else:
        overrides[f"{prefix}.model"] = judge.model
        if judge_type != "local_vllm_model":
            overrides[f"{prefix}.base_url"] = judge_url
            overrides[f"{prefix}.api_key"] = _api_key(judge)
    overrides.update(_flatten(prefix, _gym_model_config(judge)))
    return overrides


def _shard_artifacts(params: RolloutConfig, chunk_index: int) -> NativeGymArtifactPaths:
    return params.layout.chunk_artifacts(chunk_index)


def _shard_manifest(params: RolloutConfig, chunk_index: int) -> PurePosixPath:
    return params.layout.chunk_manifest(chunk_index)


def _root_artifacts(params: RolloutConfig) -> NativeGymArtifactPaths:
    return params.layout.analyzed_artifacts


def _analyzed_completion_artifacts(
    params: RolloutConfig,
    artifacts: NativeGymArtifactPaths,
) -> dict[str, PurePosixPath]:
    assert artifacts == params.layout.analyzed_artifacts
    return params.layout.analyzed_completion_artifacts(
        include_profiles=params.num_repeats >= 2,
        include_evaluation=params.report_module is not None,
        include_blade=params.blade_command is not None,
    )


def _rollout_payload(params: RolloutConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input_data": params.input_data,
        "client_container": params.client_container,
        "vllm_container": (
            params.vllm_container
            if params.policy.uses_workflow_vllm_container
            or (params.judge is not None and params.judge.uses_workflow_vllm_container)
            else None
        ),
        "installation_command": params.installation_command,
        "config_paths": list(params.config_paths),
        "agent_name": params.agent_name,
        "num_repeats": params.num_repeats,
        "num_chunks": params.num_chunks,
        "max_num_samples": params.max_num_samples,
        "responses_create_params": params.responses_create_params,
    }
    if params.allow_empty_input:
        payload["allow_empty_input"] = True
    return payload


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rollout_config_sha256(params: RolloutConfig) -> str:
    """Fingerprint every setting that can change collected rollout semantics.

    How many rollouts run at once, follow-up jobs, how the servers are sized and
    placed, and how long Gym waits for them are left out, so a resumed run may
    change them.
    """
    return _sha256_json(
        {
            **_rollout_payload(params),
            "schema_version": 2,
            "gym_overrides": result_gym_overrides(params.gym_overrides),
            "policy": params.policy.behavior(),
            "judge": params.judge.behavior() if params.judge is not None else None,
        }
    )


def _legacy_rollout_config_sha256(params: RolloutConfig) -> str:
    """Fingerprint the settings the way runs started before resizing on resume recorded them."""
    return _sha256_json(
        {
            **_rollout_payload(params),
            "schema_version": 1,
            "dependent_jobs": params.dependent_jobs,
            "concurrency": params.concurrency,
            "gym_overrides": params.gym_overrides,
            "policy": params.policy.legacy_behavior(),
            "judge": params.judge.legacy_behavior() if params.judge is not None else None,
        }
    )


def _rollout_checks(params: RolloutConfig) -> dict[str, Any]:
    """Return the fingerprints a chunk's evidence must carry.

    Evidence from a run started before resizing on resume carries the legacy
    fingerprint, which stays accepted while that run's settings are unchanged.
    """
    current = _rollout_config_sha256(params)
    legacy = (_legacy_rollout_config_sha256(params),)
    return {
        "behavior_sha256": current,
        "accepted_behavior_sha256": legacy,
        "manifest_behavior_sha256": current,
        "accepted_manifest_behavior_sha256": legacy,
    }


def _analysis_checks(params: RolloutConfig) -> dict[str, Any]:
    """Return the fingerprints the analyzed root's evidence must carry."""
    legacy_rollout = _legacy_rollout_config_sha256(params)
    return {
        "behavior_sha256": _analysis_config_sha256(params),
        "accepted_behavior_sha256": (_analysis_config_sha256(params, legacy_rollout),),
        "manifest_behavior_sha256": _rollout_config_sha256(params),
        "accepted_manifest_behavior_sha256": (legacy_rollout,),
    }


def _analysis_config_sha256(params: RolloutConfig, rollout_sha256: str | None = None) -> str:
    payload = {
        "kind": "rollout_analysis",
        "schema_version": 1,
        "rollout_sha256": rollout_sha256 or _rollout_config_sha256(params),
        "enrich_module": params.enrich_module,
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


def _managed_model_paths(params: RolloutConfig) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            endpoint.model
            for endpoint in (params.policy, params.judge)
            if endpoint is not None
            and endpoint.deployment in {"gym", "nvflow"}
            and endpoint.model is not None
            and PurePosixPath(endpoint.model).is_absolute()
        )
    )


def _completed_chunks(
    params: RolloutConfig,
    filesystem: TargetFilesystem,
) -> frozenset[int]:
    """Return only chunks whose content-hashed completion evidence validates."""
    validate = getattr(filesystem, "batch_completion_matches", None)
    if not callable(validate):
        return frozenset()
    artifacts = [_shard_artifacts(params, index) for index in range(params.num_chunks)]
    checks = _rollout_checks(params)
    matches = validate(
        [
            {
                "materialized_inputs": artifact.materialized_inputs.as_posix(),
                "successes": artifact.rollouts.as_posix(),
                "marker": artifact.completion.as_posix(),
                "behavior_sha256": checks["behavior_sha256"],
                "accepted_behavior_sha256": list(checks["accepted_behavior_sha256"]),
                "behavior_manifest": None,
                "reference_completion": None,
                "source": params.input_data,
                "shard_manifest": _shard_manifest(params, index).as_posix(),
                "manifest_behavior_sha256": checks["manifest_behavior_sha256"],
                "accepted_manifest_behavior_sha256": list(
                    checks["accepted_manifest_behavior_sha256"]
                ),
            }
            for index, artifact in enumerate(artifacts)
        ]
    )
    return frozenset(index for index, matched in enumerate(matches) if matched)


def _worker(*args: object) -> str:
    argv = [
        "python3",
        "-m",
        _SHARD_MODULE,
        *(os.fspath(arg) if isinstance(arg, os.PathLike) else str(arg) for arg in args),
    ]
    return f'PYTHONPATH="{CONTAINER_CODE_DIR}${{PYTHONPATH:+:$PYTHONPATH}}" ' + shlex.join(argv)


def _reject_legacy_layout(params: RolloutConfig | RewardConfig) -> str:
    checks = " || ".join(
        f"[ -e {shlex.quote(path.as_posix())} ]" for path in params.layout.legacy_paths
    )
    return (
        f"if {checks}; then\n"
        f'  echo "ERROR: legacy flat Gym artifacts exist under {params.output_dir}; '
        'use a fresh output root for the job-oriented layout" >&2\n'
        "  exit 1\n"
        "fi"
    )


def _completion_arguments(
    action: str,
    artifacts: NativeGymArtifactPaths,
    behavior_sha256: str | None,
    accepted_behavior_sha256: Sequence[str],
    shard_manifests: Sequence[PurePosixPath],
    completion_artifacts: Mapping[str, PurePosixPath] | None,
    manifest_source: str | None,
    manifest_behavior_sha256: str | None,
    accepted_manifest_behavior_sha256: Sequence[str],
) -> list[object]:
    arguments: list[object] = [
        action,
        "--materialized-inputs",
        artifacts.materialized_inputs,
        "--successes",
        artifacts.rollouts,
        "--completion",
        artifacts.completion,
    ]
    if behavior_sha256 is not None:
        arguments.extend(("--behavior-sha256", behavior_sha256))
        if action == "validate":
            for accepted in accepted_behavior_sha256:
                arguments.extend(("--accept-behavior-sha256", accepted))
    for manifest in shard_manifests:
        arguments.extend(("--shard-manifest", manifest))
    if shard_manifests and manifest_source is not None:
        arguments.extend(("--source", manifest_source))
    if shard_manifests and manifest_behavior_sha256 is not None:
        arguments.extend(("--manifest-behavior-sha256", manifest_behavior_sha256))
        for accepted in accepted_manifest_behavior_sha256:
            arguments.extend(("--accept-manifest-behavior-sha256", accepted))
    for name, path in sorted((completion_artifacts or {}).items()):
        arguments.extend(("--artifact", f"{name}={path}"))
    return arguments


def _validate(
    artifacts: NativeGymArtifactPaths,
    behavior_sha256: str | None = None,
    shard_manifests: Sequence[PurePosixPath] = (),
    completion_artifacts: Mapping[str, PurePosixPath] | None = None,
    manifest_source: str | None = None,
    manifest_behavior_sha256: str | None = None,
    accepted_behavior_sha256: Sequence[str] = (),
    accepted_manifest_behavior_sha256: Sequence[str] = (),
) -> str:
    return _worker(
        *_completion_arguments(
            "validate",
            artifacts,
            behavior_sha256,
            accepted_behavior_sha256,
            shard_manifests,
            completion_artifacts,
            manifest_source,
            manifest_behavior_sha256,
            accepted_manifest_behavior_sha256,
        )
    )


def _complete(
    artifacts: NativeGymArtifactPaths,
    behavior_sha256: str | None = None,
    shard_manifests: Sequence[PurePosixPath] = (),
    completion_artifacts: Mapping[str, PurePosixPath] | None = None,
    manifest_source: str | None = None,
    manifest_behavior_sha256: str | None = None,
    accepted_behavior_sha256: Sequence[str] = (),
    accepted_manifest_behavior_sha256: Sequence[str] = (),
) -> str:
    return _worker(
        *_completion_arguments(
            "complete",
            artifacts,
            behavior_sha256,
            accepted_behavior_sha256,
            shard_manifests,
            completion_artifacts,
            manifest_source,
            manifest_behavior_sha256,
            accepted_manifest_behavior_sha256,
        )
    )


def _prepare(params: RolloutConfig, chunk_index: int) -> str:
    artifacts = _shard_artifacts(params, chunk_index)
    prepared = artifacts.rollouts.with_suffix("") / "preprocessed_datasets" / "train.jsonl"
    manifest = _shard_manifest(params, chunk_index)
    arguments: list[object] = [
        "prepare",
        "--source",
        params.input_data,
        "--output",
        prepared,
        "--manifest",
        manifest,
        "--chunk-index",
        chunk_index,
        "--num-chunks",
        params.num_chunks,
        "--num-repeats",
        params.num_repeats,
        "--rollout-config-sha256",
        _rollout_config_sha256(params),
        "--accept-rollout-config-sha256",
        _legacy_rollout_config_sha256(params),
    ]
    if params.max_num_samples is not None:
        arguments.extend(("--max-num-samples", params.max_num_samples))
    if params.allow_empty_input:
        arguments.append("--allow-empty")
    orphan_checks = " || ".join(
        f"[ -e {shlex.quote(path.as_posix())} ]"
        for path in (
            artifacts.rollouts,
            artifacts.materialized_inputs,
            artifacts.failures,
            artifacts.completion,
        )
    )
    orphan_guard = (
        f"if [ ! -f {shlex.quote(manifest.as_posix())} ] && "
        f"{{ {orphan_checks}; }}; then\n"
        f'  echo "ERROR: rollout artifacts exist without identity manifest {manifest}; '
        'use a new output directory or remove the orphan artifacts" >&2\n'
        "  exit 1\n"
        "fi\n"
    )
    return f"{orphan_guard}{_worker(*arguments)}"


def _write_dataset_config(params: RolloutConfig, chunk_index: int) -> str:
    artifacts = _shard_artifacts(params, chunk_index)
    prepared = artifacts.rollouts.with_suffix("") / "preprocessed_datasets" / "train.jsonl"
    dataset_config = artifacts.rollouts.parent / "dataset_config.yaml"
    return render_runtime_binding_command(
        config_paths=params.config_paths,
        agent_name=params.agent_name,
        output_config=dataset_config,
        output_binding=artifacts.rollouts.parent / "gym_binding.json",
        datasets=(
            {
                "name": "nvflow",
                "type": "train",
                "license": "TBD",
                "jsonl_fpath": prepared.as_posix(),
            },
        ),
        judge_model_name="judge_model" if params.judge is not None else "policy_model",
    )


def _rollout_command(
    params: RolloutConfig,
    chunk_index: int,
    policy_url: OverrideValue | None,
    judge_url: OverrideValue | None,
) -> str:
    artifacts = _shard_artifacts(params, chunk_index)
    arguments: list[ArgumentValue] = [
        "--agent",
        params.agent_name,
        "--split",
        "train",
        "--output",
        artifacts.rollouts,
        "--num-repeats",
        params.num_repeats,
        "--resume",
        "--no-health-check",
    ]
    if params.num_chunks > 1:
        arguments.append("--disable-aggregation")
    if params.concurrency is not None:
        arguments.extend(("--concurrency", params.concurrency))
    overrides = _model_overrides(params, policy_url, judge_url)
    overrides.update(_flatten("responses_create_params", params.responses_create_params))
    return render_gym_command(
        "gym",
        "eval",
        "run",
        config_paths=(
            *params.config_paths,
            artifacts.rollouts.parent / "dataset_config.yaml",
        ),
        args=arguments,
        overrides=overrides,
    )


GYM_STARTUP_ATTEMPTS = 3


def _with_startup_retry(command: str, rollouts: PurePosixPath) -> str:
    """Run a Gym collection, starting it again when it exits before collecting a rollout.

    Gym picks its servers' ports by probing and binds them a moment later, so a
    process on the same node can take a port in between. Each attempt picks a
    new head port; ``NVFLOW_GYM_RETRY_DELAY`` overrides the random pause.
    """
    path = shlex.quote(rollouts.as_posix())
    return f"""nvflow_rows() {{ if [ -f "$1" ]; then wc -l < "$1" | tr -d ' '; else echo 0; fi; }}
NVFLOW_GYM_ATTEMPT=1
while :; do
  NVFLOW_ROLLOUT_HEAD_PORT=$(find_free_port)
  NVFLOW_ROWS_BEFORE=$(nvflow_rows {path})
  set +e
  {command}
  NVFLOW_GYM_STATUS=$?
  set -e
  if [ "$NVFLOW_GYM_STATUS" -eq 0 ]; then
    break
  fi
  if [ "$(nvflow_rows {path})" -ne "$NVFLOW_ROWS_BEFORE" ] || [ "$NVFLOW_GYM_ATTEMPT" -ge {GYM_STARTUP_ATTEMPTS} ]; then
    exit "$NVFLOW_GYM_STATUS"
  fi
  NVFLOW_GYM_ATTEMPT=$((NVFLOW_GYM_ATTEMPT + 1))
  echo "WARNING: NeMo Gym exited with status $NVFLOW_GYM_STATUS before collecting a rollout; starting it again (attempt $NVFLOW_GYM_ATTEMPT of {GYM_STARTUP_ATTEMPTS})" >&2
  sleep "${{NVFLOW_GYM_RETRY_DELAY:-$((RANDOM % 20 + 5))}}"
done"""


def _collection_script(
    params: RolloutConfig,
    chunk_index: int,
    policy_url: OverrideValue | None,
    judge_url: OverrideValue | None,
    *,
    port_preamble: str,
) -> str:
    artifacts = _shard_artifacts(params, chunk_index)
    manifest = _shard_manifest(params, chunk_index)
    empty_probe = (
        "import json,sys;"
        "data=json.load(open(sys.argv[1]));"
        "raise SystemExit(data['selected_row_count'] != 0)"
    )
    empty_completion = "\n".join(
        f": > {shlex.quote(path.as_posix())}"
        for path in (artifacts.materialized_inputs, artifacts.rollouts, artifacts.failures)
    )
    repairs = "\n".join(
        f"[ ! -e {shlex.quote(path.as_posix())} ] || {_worker('repair', '--jsonl', path)}"
        for path in (artifacts.rollouts, artifacts.failures)
    )
    selection_cleanup = (
        f"rm -f {shlex.quote(params.layout.dataset_completion.as_posix())}"
        if params.dataset_selection
        else ":"
    )
    return f"""set -euo pipefail
{_reject_legacy_layout(params)}
{selection_cleanup}
{_prepare(params, chunk_index)}
if python3 -c {shlex.quote(empty_probe)} {shlex.quote(manifest.as_posix())}; then
  mkdir -p {shlex.quote(artifacts.rollouts.parent.as_posix())}
  {empty_completion}
  {_complete(artifacts, shard_manifests=(manifest,), manifest_source=params.input_data, **_rollout_checks(params))}
  exit 0
fi
if [ -f {shlex.quote(artifacts.completion.as_posix())} ]; then
  if {_validate(artifacts, shard_manifests=(manifest,), manifest_source=params.input_data, **_rollout_checks(params))}; then
    exit 0
  fi
  echo "WARNING: removing invalid completion marker {artifacts.completion}" >&2
  rm -f {shlex.quote(artifacts.completion.as_posix())}
fi
{params.installation_command or "true"}
cd {shlex.quote(NATIVE_GYM_SOURCE_ROOT)}
{_write_dataset_config(params, chunk_index)}
{repairs}
{SHELL_FIND_FREE_PORT}
{port_preamble}export NEMO_GYM_MAX_ROLLOUT_ATTEMPTS={rollout_attempt_limit(params.dependent_jobs)}
cd {shlex.quote(NATIVE_GYM_SOURCE_ROOT)}
{_with_startup_retry(_rollout_command(params, chunk_index, policy_url, judge_url), artifacts.rollouts)}
{_complete(artifacts, shard_manifests=(manifest,), manifest_source=params.input_data, **_rollout_checks(params))}
"""


@dataclass(kw_only=True)
class NativeCollectionScript(BaseJobScript):
    """Lazily bind managed endpoint hostnames after heterogeneous planning."""

    params: RolloutConfig
    chunk_index: int
    job_label: str
    log_dir: str
    policy_server: ServerScript | None = None
    judge_server: ServerScript | None = None

    def __post_init__(self) -> None:
        def build() -> str:
            policy_url: OverrideValue | None
            judge_url: OverrideValue | None
            if self.policy_server is None:
                policy_url = self.params.policy.base_url
            else:
                policy_url = shell_variable("NVFLOW_POLICY_URL")
            if self.judge_server is None:
                judge_url = self.params.judge.base_url if self.params.judge is not None else None
            else:
                judge_url = shell_variable("NVFLOW_JUDGE_URL")
            preamble = _build_port_read_preamble(
                self.log_dir,
                self.job_label,
                policy_host=(
                    None if self.policy_server is None else self.policy_server.hostname_ref()
                ),
                judge_host=(
                    None if self.judge_server is None else self.judge_server.hostname_ref()
                ),
            )
            if self.policy_server is not None:
                preamble += 'NVFLOW_POLICY_URL="http://$POLICY_HOST:$POLICY_PORT/v1"\n'
            if self.judge_server is not None:
                preamble += 'NVFLOW_JUDGE_URL="http://$JUDGE_HOST:$JUDGE_PORT/v1"\n'
            return _collection_script(
                self.params,
                self.chunk_index,
                policy_url,
                judge_url,
                port_preamble=preamble,
            )

        self.set_inline(build)
        super().__post_init__()


def _server_script(
    endpoint: ModelEndpoint | None,
    cluster_config: dict[str, Any],
    *,
    log_dir: str,
    job_label: str,
    done_guard: str,
) -> ServerScript | None:
    if endpoint is None or not endpoint.nvflow_deployed:
        return None
    config = {
        "model_path": endpoint.model,
        "num_gpus": endpoint.num_gpus,
        "server_nodes": endpoint.server_nodes,
        **endpoint.gym_model.get("vllm_serve_kwargs", {}),
    }
    script = make_server_script(
        config,
        cluster_config,
        role=endpoint.role,
        server_type=endpoint.server_type or "vllm",
        log_dir=log_dir,
        job_label=job_label,
        environment=endpoint.gym_model.get("vllm_serve_env_vars"),
    )
    script.set_inline(f"{done_guard}\n{script.inline}")
    return script


def _done_guard(params: RolloutConfig, artifacts: NativeGymArtifactPaths) -> str:
    manifest = artifacts.rollouts.parent.parent / "manifest.json"
    return (
        f"if [ -f {shlex.quote(artifacts.completion.as_posix())} ] && "
        f"{_validate(artifacts, shard_manifests=(manifest,), manifest_source=params.input_data, **_rollout_checks(params))}; then\n"
        "  exit 0\n"
        "fi"
    )


def _groups(
    params: RolloutConfig,
    client_script: BaseJobScript,
    policy_server: ServerScript | None,
    judge_server: ServerScript | None,
    *,
    group_name: str,
    log_dir: str,
    cluster_config: dict[str, Any],
    retry_attempt: bool,
) -> list[Any]:
    from nemo_skills.pipeline.utils.declarative import Command, CommandGroup, HardwareConfig

    hardware_sbatch_kwargs = dependency_sbatch_kwargs(
        parse_extra_sbatch_args(cluster_config),
        continue_after_failure=retry_attempt,
    )
    gym_gpus = sum(
        endpoint.num_gpus
        for endpoint in (params.policy, params.judge)
        if endpoint is not None and endpoint.gym_deployed
    )
    managed = [
        (params.policy, policy_server),
        (params.judge, judge_server),
    ]
    managed = [
        (endpoint, script)
        for endpoint, script in managed
        if endpoint is not None and script is not None
    ]
    groups: list[CommandGroup] = []

    client = Command(
        script=client_script,
        container=params.client_container,
        name=f"{group_name}-client",
    )
    if gym_gpus or not managed:
        groups.append(
            CommandGroup(
                commands=[client],
                hardware=HardwareConfig(
                    num_gpus=gym_gpus,
                    num_nodes=1,
                    sbatch_kwargs=hardware_sbatch_kwargs,
                ),
                name=group_name,
                log_dir=log_dir,
            )
        )
    for index, (endpoint, script) in enumerate(managed):
        commands = [
            Command(
                script=script,
                container=(
                    endpoint.container
                    or (
                        params.vllm_container
                        if endpoint.server_type == "vllm"
                        else endpoint.server_type
                    )
                ),
                name=f"{group_name}-{endpoint.role}",
            )
        ]
        if not gym_gpus and index == 0:
            commands.append(client)
        groups.append(
            CommandGroup(
                commands=commands,
                hardware=HardwareConfig(
                    num_gpus=endpoint.num_gpus,
                    num_nodes=endpoint.server_nodes,
                    sbatch_kwargs=hardware_sbatch_kwargs,
                ),
                name=f"{group_name}-{endpoint.role}",
                log_dir=log_dir,
            )
        )
    return groups


def _job(
    groups: list[Any],
    *,
    name: str,
    role: str,
    dependencies: list[Any] | None,
) -> dict[str, Any]:
    key = "group" if len(groups) == 1 else "groups"
    return {
        "name": name,
        "role": role,
        key: groups[0] if len(groups) == 1 else groups,
        "dependencies": dependencies,
    }


def _build_attempt(
    params: RolloutConfig,
    cluster_config: dict[str, Any],
    *,
    chunk_index: int,
    attempt: int,
    expname: str,
    dependencies: list[Any] | None,
) -> dict[str, Any]:
    artifacts = _shard_artifacts(params, chunk_index)
    label = f"job-0-collect-rollouts-chunk-{chunk_index}-attempt-{attempt}"
    log_dir = params.layout.attempt_logs_dir(chunk_index, attempt).as_posix()
    guard = _done_guard(params, artifacts)
    policy_server = _server_script(
        params.policy,
        cluster_config,
        log_dir=log_dir,
        job_label=label,
        done_guard=guard,
    )
    judge_server = _server_script(
        params.judge,
        cluster_config,
        log_dir=log_dir,
        job_label=label,
        done_guard=guard,
    )
    client = NativeCollectionScript(
        params=params,
        chunk_index=chunk_index,
        job_label=label,
        log_dir=log_dir,
        policy_server=policy_server,
        judge_server=judge_server,
    )
    groups = _groups(
        params,
        client,
        policy_server,
        judge_server,
        group_name=label,
        log_dir=log_dir,
        cluster_config=cluster_config,
        retry_attempt=attempt > 0,
    )
    return _job(
        groups,
        name=f"{expname}-{label}",
        role="collect-rollouts",
        dependencies=dependencies,
    )


def _atomic_cat(paths: Sequence[PurePosixPath], destination: PurePosixPath) -> str:
    quoted_destination = shlex.quote(destination.as_posix())
    sources = " ".join(shlex.quote(path.as_posix()) for path in paths)
    return (
        f"_tmp={quoted_destination}.tmp.$$\n"
        'rm -f "$_tmp"\n'
        "trap 'rm -f \"$_tmp\"' EXIT\n"
        f'cat {sources} > "$_tmp"\n'
        f'mv -f "$_tmp" {quoted_destination}\n'
        "trap - EXIT"
    )


def _aggregation_overrides(
    params: RolloutConfig,
) -> dict[str, OverrideValue]:
    """Configure CPU-only model proxies for environment metric aggregation."""
    unused_model_url = "http://127.0.0.1:1/v1"
    overrides: dict[str, OverrideValue] = {
        "head_server.host": "127.0.0.1",
        "head_server.port": shell_variable("NVFLOW_AGGREGATE_HEAD_PORT"),
        "model_endpoint_readiness_timeout_seconds": 0,
        "policy_base_url": unused_model_url,
        "policy_api_key": "EMPTY",
        "policy_model_name": params.policy.model,
        "skip_venv_if_present": True,
    }
    overrides.update(_flatten("", params.gym_overrides))
    # The proxies above never answer, so no Gym override may make Gym wait for them.
    overrides["model_endpoint_readiness_timeout_seconds"] = 0
    if params.judge is not None:
        prefix = "judge_model.responses_api_models.vllm_model"
        overrides.update(
            {
                f"{prefix}.entrypoint": "app.py",
                f"{prefix}.base_url": unused_model_url,
                f"{prefix}.api_key": "EMPTY",
                f"{prefix}.model": params.judge.model,
                f"{prefix}.return_token_id_information": False,
                f"{prefix}.uses_reasoning_parser": False,
            }
        )
    return overrides


def _server_backed_aggregate(
    params: RolloutConfig,
    shards: Sequence[NativeGymArtifactPaths],
    root: NativeGymArtifactPaths,
) -> str:
    """Run Gym aggregation with only its CPU head, agent, and resource servers."""
    config_paths = (
        *params.aggregation_config_paths,
        shards[0].rollouts.parent / "dataset_config.yaml",
    )
    overrides = _aggregation_overrides(params)
    environment_command = render_gym_command(
        "gym",
        "env",
        "start",
        config_paths=config_paths,
        overrides=overrides,
    )
    # Gym won't merge into an output whose inputs or failures already sit
    # beside it, so the shards are joined here and Gym only computes metrics.
    aggregate_command = render_gym_command(
        "gym",
        "eval",
        "aggregate",
        args=(
            "--input-glob",
            root.rollouts,
            "--output",
            root.rollouts,
            "--no-health-check",
        ),
        config_paths=config_paths,
        overrides={**overrides, "merge_shards": False},
    )
    environment_log = params.layout.analyze_logs_dir / "aggregate-environment.log"
    return f"""{_atomic_cat([shard.rollouts for shard in shards], root.rollouts)}
{SHELL_FIND_FREE_PORT}
{SHELL_STOP_BACKGROUND_PROCESS}
{_start_aggregate_environment(environment_command, environment_log)}
{aggregate_command}
cleanup_aggregate_env
trap - EXIT"""


def _start_aggregate_environment(command: str, log: PurePosixPath) -> str:
    """Start Gym's aggregation environment and wait until its servers are ready.

    Like a collection (see ``_with_startup_retry``), an environment that exits
    before it is ready is started again on a new head port.
    """
    quoted_log = shlex.quote(log.as_posix())
    return f"""aggregate_env_pid=""
cleanup_aggregate_env() {{
  if [ -n "$aggregate_env_pid" ]; then
    stop_background_process "$aggregate_env_pid" "Gym aggregation environment"
  fi
}}
trap cleanup_aggregate_env EXIT
aggregate_env_ready=false
for aggregate_env_attempt in $(seq 1 {GYM_STARTUP_ATTEMPTS}); do
  NVFLOW_AGGREGATE_HEAD_PORT=$(find_free_port)
  export NVFLOW_AGGREGATE_HEAD_PORT
  (
    trap - INT TERM
    exec {command}
  ) > {quoted_log} 2>&1 &
  aggregate_env_pid=$!
  for _poll in $(seq 1 600); do
    if grep -Eq 'All [0-9]+ / [0-9]+ servers ready!' {quoted_log}; then
      aggregate_env_ready=true
      break
    fi
    if ! kill -0 "$aggregate_env_pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if [ "$aggregate_env_ready" = true ]; then
    break
  fi
  if kill -0 "$aggregate_env_pid" 2>/dev/null; then
    echo "ERROR: Gym aggregation environment was not ready after 600s; see {log}" >&2
    exit 1
  fi
  if [ "$aggregate_env_attempt" -ge {GYM_STARTUP_ATTEMPTS} ]; then
    echo "ERROR: Gym aggregation environment exited during startup; see {log}" >&2
    exit 1
  fi
  echo "WARNING: Gym aggregation environment exited during startup; starting it again (attempt $((aggregate_env_attempt + 1)) of {GYM_STARTUP_ATTEMPTS})" >&2
  sleep "${{NVFLOW_GYM_RETRY_DELAY:-$((RANDOM % 20 + 5))}}"
done"""


def _aggregate_shards(
    params: RolloutConfig,
    shards: Sequence[NativeGymArtifactPaths],
    root: NativeGymArtifactPaths,
) -> str:
    """Publish one shard directly or ask Gym to aggregate multiple shards."""
    if len(shards) != 1:
        return _server_backed_aggregate(params, shards, root)
    shard = shards[0]
    fallback = _server_backed_aggregate(params, shards, root)
    return f"""if [ -s {shlex.quote(shard.aggregate_metrics.as_posix())} ]; then
  {_atomic_cat([shard.rollouts], root.rollouts)}
  {_atomic_cat([shard.aggregate_metrics], root.aggregate_metrics)}
else
  {fallback}
fi"""


def _offline_commands(params: RolloutConfig, root: NativeGymArtifactPaths) -> str:
    enrich = (
        ""
        if params.enrich_module is None
        else (
            f'PYTHONPATH="{CONTAINER_CODE_DIR}${{PYTHONPATH:+:$PYTHONPATH}}" '
            f"python3 -m {shlex.quote(params.enrich_module)} "
            f"{shlex.quote(params.input_data)} {shlex.quote(root.rollouts.as_posix())} "
            f"--expected-repeats {params.num_repeats}\n"
        )
    )
    profile = (
        ""
        if params.num_repeats < 2
        else render_gym_command(
            "gym",
            "eval",
            "profile",
            args=("--inputs", root.materialized_inputs, "--rollouts", root.rollouts),
        )
    )
    health = render_gym_command(
        "gym",
        "eval",
        "health-check",
        args=(root.rollouts.parent, "--rollouts-file", root.rollouts.name),
    )
    profile_step = f"{profile}\n" if profile else ""
    return f"{enrich}{profile_step}{health}\n{_report_command(params, root)}"


def _report_command(params: RolloutConfig, root: NativeGymArtifactPaths) -> str:
    if params.report_module is None:
        return ""
    return (
        f'PYTHONPATH="{CONTAINER_CODE_DIR}${{PYTHONPATH:+:$PYTHONPATH}}" '
        f"python3 -m {shlex.quote(params.report_module)} "
        f"{shlex.quote(root.rollouts.as_posix())} "
        f"{shlex.quote(root.materialized_inputs.as_posix())} "
        f"{shlex.quote(params.layout.evaluation_analysis_dir.as_posix())} "
        f"--expected-repeats {params.num_repeats}{_report_options_flag(params.report_options)}\n"
    )


def _report_options_flag(options: Mapping[str, Any]) -> str:
    return f" --options {shlex.quote(json.dumps(options, sort_keys=True))}" if options else ""


def _unless_empty(params: RolloutConfig, root: NativeGymArtifactPaths, analysis: str) -> str:
    """Run *analysis* when rows were collected; otherwise publish empty analysis files.

    Gym's aggregation and health check need rollouts, so an allowed empty input
    skips them and still runs the report.
    """
    placeholders = [
        path
        for name, path in _analyzed_completion_artifacts(params, root).items()
        if name not in {"failures", "evaluation_analysis"}
    ]
    parents = " ".join(dict.fromkeys(shlex.quote(path.parent.as_posix()) for path in placeholders))
    writes = "\n".join(
        f"  : > {shlex.quote(path.as_posix())}"
        if path.suffix == ".jsonl"
        else f"  printf '{{}}\\n' > {shlex.quote(path.as_posix())}"
        for path in placeholders
    )
    return f"""if [ -s {shlex.quote(root.materialized_inputs.as_posix())} ]; then
{analysis}
else
  echo "No input rows: publishing empty analysis files" >&2
  mkdir -p {parents}
  : > {shlex.quote(root.rollouts.as_posix())}
{writes}
{_report_command(params, root)}fi"""


def _analyze_script(
    params: RolloutConfig,
) -> str:
    root = _root_artifacts(params)
    shards = [_shard_artifacts(params, index) for index in range(params.num_chunks)]
    validations = "\n".join(
        f"{_prepare(params, index)}\n"
        f"{_validate(shard, shard_manifests=(_shard_manifest(params, index),), manifest_source=params.input_data, **_rollout_checks(params))}"
        for index, shard in enumerate(shards)
    )
    materialized_merge = _atomic_cat(
        [shard.materialized_inputs for shard in shards],
        root.materialized_inputs,
    )
    failures_merge = _atomic_cat([shard.failures for shard in shards], root.failures)
    analysis = f"{_aggregate_shards(params, shards, root)}\n{_offline_commands(params, root)}"
    if params.allow_empty_input:
        analysis = _unless_empty(params, root, analysis)
    blade = build_blade_script(
        command=params.blade_command,
        output_dir=params.output_dir,
        blade_output_dir=params.layout.blade_dir.as_posix(),
        rollouts=root.rollouts.as_posix(),
        materialized_inputs=root.materialized_inputs.as_posix(),
        reward_profile=root.reward_profiling.as_posix(),
    )
    manifests = tuple(_shard_manifest(params, index) for index in range(params.num_chunks))
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
    return f"""set -euo pipefail
{_reject_legacy_layout(params)}
{selection_evidence_cleanup}
if [ -f {shlex.quote(root.completion.as_posix())} ]; then
  if {_validate(root, shard_manifests=manifests, completion_artifacts=_analyzed_completion_artifacts(params, root), manifest_source=params.input_data, **_analysis_checks(params))}; then
    exit 0
  else
    echo "WARNING: rebuilding invalid root rollout completion evidence" >&2
    rm -f {shlex.quote(root.completion.as_posix())}
  fi
fi
{validations}
mkdir -p {shlex.quote(params.layout.analyze_outputs_dir.as_posix())} \
  {shlex.quote(params.layout.analyze_logs_dir.as_posix())}
{materialized_merge}
{failures_merge}
{analysis}
{blade}
{_complete(root, shard_manifests=manifests, completion_artifacts=_analyzed_completion_artifacts(params, root), manifest_source=params.input_data, **_analysis_checks(params))}
"""


def _build_analyze(
    params: RolloutConfig,
    *,
    expname: str,
    dependencies: list[Any],
) -> dict[str, Any]:
    from nemo_skills.pipeline.utils.declarative import Command, CommandGroup, HardwareConfig

    group = CommandGroup(
        commands=[
            Command(
                script=make_bash_script(
                    _analyze_script(params),
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
    return _job(
        [group],
        name=f"{expname}-job-1-analyze-rollouts",
        role="analyze-rollouts",
        dependencies=dependencies,
    )


def _build_dataset_selection(
    params: RolloutConfig,
    predecessor: dict[str, Any],
    *,
    expname: str,
) -> dict[str, Any] | None:
    if not params.dataset_selection:
        return None
    from nemo_skills.pipeline.utils.declarative import Command, CommandGroup, HardwareConfig

    selection = dict(params.dataset_selection)
    selection["provenance"] = {
        **dict(selection.get("provenance") or {}),
        "policy_model": params.policy.model,
        "judge_model": (params.judge.model if params.judge else params.policy.model),
        "agent_name": params.agent_name,
    }
    command = build_dataset_selection_cmd(
        source=params.input_data,
        reward_profile=_root_artifacts(params).reward_profiling.as_posix(),
        materialized_inputs=_root_artifacts(params).materialized_inputs.as_posix(),
        rollouts=_root_artifacts(params).rollouts.as_posix(),
        rollout_completion=_root_artifacts(params).completion.as_posix(),
        profiled_output=params.layout.profiled_dataset.as_posix(),
        training_output=params.layout.training_dataset.as_posix(),
        report_output=params.layout.dataset_selection_report.as_posix(),
        completion_output=params.layout.dataset_completion.as_posix(),
        diagnostics_output=params.layout.unprofiled_samples.as_posix(),
        selection=selection,
        responses_create_params_overrides=params.responses_create_params,
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
    return _job(
        [group],
        name=f"{expname}-job-2-build-training-dataset",
        role="build-training-dataset",
        dependencies=[predecessor],
    )


def build_rollout_jobs(
    params: RolloutConfig,
    cluster_config: dict[str, Any],
    expname: str,
    run_after: list[str] | None = None,
    *,
    completed_chunks: frozenset[int] = frozenset(),
) -> list[dict[str, Any]]:
    """Build rollout attempts plus analyzed and optional dataset outputs."""
    jobs: list[dict[str, Any]] = []
    terminal_attempts: list[dict[str, Any]] = []
    for chunk_index in range(params.num_chunks):
        if chunk_index in completed_chunks:
            continue
        dependencies: list[Any] | None = list(run_after or [])
        previous: dict[str, Any] | None = None
        for attempt in range(params.dependent_jobs + 1):
            if previous is not None:
                dependencies = [previous]
            current = _build_attempt(
                params,
                cluster_config,
                chunk_index=chunk_index,
                attempt=attempt,
                expname=expname,
                dependencies=dependencies,
            )
            jobs.append(current)
            previous = current
        assert previous is not None
        terminal_attempts.append(previous)
    analysis = _build_analyze(
        params,
        expname=expname,
        dependencies=terminal_attempts or list(run_after or []),
    )
    jobs.append(analysis)
    selection_job = _build_dataset_selection(params, analysis, expname=expname)
    if selection_job is not None:
        jobs.append(selection_job)
    return jobs


def rollout_operation(
    params: RolloutConfig,
    cluster: str,
    expname: str,
    run_after: list[str] | None = None,
    *,
    inspect_live_state: bool = False,
    filesystem: TargetFilesystem | None = None,
) -> list[dict[str, Any]]:
    """Plan one normalized native Gym rollout operation.

    Static plans retain every chunk. Live execution prunes chunks with complete
    artifacts; the analyze job revalidates their identity and content.
    """
    from nvflow.integrations.nemo_skills import portable_cluster_config

    cluster_config = portable_cluster_config()
    if inspect_live_state and filesystem is None:
        from nvflow.integrations.nemo_skills import load_target_config
        from nvflow.integrations.nemo_skills.filesystem import (
            NemoSkillsTargetFilesystem,
        )

        filesystem = NemoSkillsTargetFilesystem(load_target_config(cluster))
    completed_chunks = (
        _completed_chunks(params, filesystem)
        if inspect_live_state and filesystem is not None
        else frozenset()
    )
    jobs = build_rollout_jobs(
        params,
        cluster_config,
        expname,
        run_after,
        completed_chunks=completed_chunks,
    )
    model_paths = _managed_model_paths(params)
    for job in jobs:
        if job.get("role") == "collect-rollouts":
            job["_nvflow_managed_model_paths"] = model_paths

    console.status("Collecting rollouts with native NeMo Gym")
    console.detail(
        "Work",
        f"{params.num_chunks} chunks × {params.num_repeats} repeats; "
        f"{params.dependent_jobs + 1} attempt(s)/chunk",
    )
    if completed_chunks:
        console.detail(
            "Resume",
            f"{len(completed_chunks)}/{params.num_chunks} validated chunk(s) skipped",
        )
    console.detail("Output", params.output_dir)
    console.blank()
    return jobs


def rollout(
    config: dict[str, Any],
    cluster: str,
    expname: str,
    run_after: list[str] | None = None,
    *,
    enrich_module: str | None = None,
    report_module: str | None = None,
    inspect_live_state: bool = False,
    filesystem: TargetFilesystem | None = None,
) -> list[dict[str, Any]]:
    """Parse and plan one native rollout operation."""
    params = parse_rollout_config(
        config,
        enrich_module=enrich_module,
        report_module=report_module,
    )
    return rollout_operation(
        params,
        cluster,
        expname,
        run_after,
        inspect_live_state=inspect_live_state,
        filesystem=filesystem,
    )


__all__ = [
    "ModelEndpoint",
    "NativeRolloutConfigError",
    "RolloutConfig",
    "build_rollout_jobs",
    "parse_model_endpoint",
    "parse_rollout_config",
    "rollout",
    "rollout_operation",
]
