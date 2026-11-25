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
"""GRPO reinforcement learning with NeMo-RL and NeMo Gym environments.

Plans GRPO training that runs NeMo-RL's ``run_grpo_nemo_gym.py`` directly,
bypassing ``nemo-skills`` ``grpo_nemo_rl()`` to avoid unwanted data-key
injections and the ``cp`` entry-point hack.

The full config (preset + overrides) is base64-encoded and decoded
inside the job, then passed to ``run_grpo_nemo_gym.py`` via
``--config``.  Runtime overrides (model_name, cluster, checkpoint_dir,
data paths, etc.) are applied on top via ``++key=value`` CLI args.
"""

from __future__ import annotations

import shlex
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml
from omegaconf import OmegaConf

from nvflow.core import console
from nvflow.core.node_address import NODE_ADDRESS_COMMAND
from nvflow.core.vllm_multi_node import MP_NODE_ARGS, VLLM_LAUNCHER
from nvflow.execution import PlannedExecutionStage
from nvflow.execution.resources import resolve_gpu_layout
from nvflow.integrations.model_servers.serve import server_environment
from nvflow.integrations.nemo_gym.environments import VLLM_MODEL_FOR_TRAINING
from nvflow.integrations.nemo_gym.operation_config import parse_server_environment
from nvflow.integrations.nemo_gym.runtime_binding import render_runtime_binding_command
from nvflow.integrations.nemo_gym.vllm_judge import (
    build_vllm_server_args,
    vllm_serve_overrides,
)

if TYPE_CHECKING:
    from nvflow.execution import PlanningContext, StagePlan


def _training_judge_mode(endpoint: dict[str, Any]) -> str:
    """Map the canonical endpoint owner/type to NeMo-RL's judge topology."""
    if not endpoint:
        return "policy_as_judge"
    deployment = endpoint.get("deployment")
    model_type = endpoint.get("model_type", "vllm_model")
    if deployment in {"gym", "nvflow"}:
        if endpoint.get("server_type", "vllm") != "vllm":
            raise ValueError("GRPO training supports only vLLM for a managed judge endpoint")
        if model_type not in {"vllm_model", "local_vllm_model"}:
            raise ValueError("a managed GRPO judge requires a vLLM Gym model type")
        return "local_vllm"
    if deployment == "external" and model_type == "vllm_model":
        return "external_vllm"
    if deployment == "external" and model_type == "openai_model":
        return "openai"
    if deployment == "external" and model_type == "inference_provider":
        return "inference_provider"
    raise ValueError(
        "models.judge must use deployment gym/nvflow with vLLM or "
        "deployment external with a supported Gym model type"
    )


def _training_judge_container(
    config: dict[str, Any],
    endpoint: dict[str, Any],
) -> str:
    """Resolve an explicitly configured container for a managed judge."""
    container = endpoint.get("container") or config.get("vllm_container")
    if not isinstance(container, str) or not container:
        raise ValueError(
            "managed training judge requires 'vllm_container' or models.judge.container"
        )
    return container


def _on_first_node(command: str) -> str:
    """Run *command* on a multi-node judge's first node only, the one that serves HTTP."""
    return f'if [ "${{SLURM_PROCID:-0}}" = 0 ]; then {command}; fi'


def _judge_model_fragment(endpoint: dict[str, Any], judge_mode: str) -> dict[str, Any]:
    """Build Gym's judge model adapter directly from the canonical endpoint."""
    if judge_mode == "policy_as_judge":
        return {}
    api_key = f"${endpoint['api_key_env']}" if endpoint.get("api_key_env") else ""
    if judge_mode in {"local_vllm", "external_vllm"}:
        base_url = endpoint.get("base_url", "") if judge_mode == "external_vllm" else ""
        gym_model = endpoint.get("gym_model") or {}
        client_config = {
            key: value
            for key, value in gym_model.items()
            if key not in {"vllm_serve_kwargs", "vllm_serve_env_vars"}
        }
        model_config = {
            **client_config,
            "entrypoint": "app.py",
            "base_url": base_url,
            "api_key": api_key or "EMPTY",
            "model": endpoint["model"],
        }
        model_config.setdefault("return_token_id_information", False)
        model_config.setdefault("uses_reasoning_parser", False)
        return {"judge_model": {"responses_api_models": {"vllm_model": model_config}}}
    if judge_mode == "openai":
        model_config = {
            "entrypoint": "app.py",
            "openai_base_url": endpoint["base_url"],
            "openai_api_key": api_key,
            "openai_model": endpoint["model"],
        }
    else:
        model_config = {
            "entrypoint": "app.py",
            "base_url": endpoint["base_url"],
            "api_key": api_key,
            "model": endpoint["model"],
        }
    adapter = "openai_model" if judge_mode == "openai" else judge_mode
    return {"judge_model": {"responses_api_models": {adapter: model_config}}}


@dataclass
class PreparedGRPOConfig:
    """Prepared training configuration for GRPO NeMo-RL format presets."""

    nemo_rl_config: dict
    run_name: str
    output_dir: str
    expname: str
    hf_model_name: str
    num_nodes: int
    num_gpus: int
    backend: str
    judge_mode: str = "policy_as_judge"
    judge_job_info: dict | None = None
    gym_binding_commands: tuple[str, ...] = ()
    run_after: list[str] | None = None


class GRPOStage(PlannedExecutionStage):
    """GRPO reinforcement learning with NeMo-RL.

    Merges a preset from the project's ``presets_file`` with workflow
    overrides, validates parallelism, builds ``env.nemo_gym.config_paths``
    from the ``environments`` dict, and plans the training, judge, and
    judge-cleanup tasks.
    """

    @staticmethod
    def _load_presets(config: Mapping[str, Any]) -> dict[str, Any]:
        """Load presets from the launcher-resolved ``presets_file``."""
        presets_file = config.get("presets_file")
        if not presets_file:
            return {}
        path = Path(presets_file)
        if not path.is_file():
            raise ValueError(f"'presets_file' does not exist: {presets_file}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return dict(data.get("presets") or {})

    def _get_parallelism_config(self, policy: dict, backend: str) -> dict[str, int]:
        """Extract parallelism configuration from policy based on backend.

        Returns:
            dict with keys: tp, pp, cp, ep, etp
        """
        if backend == "fsdp":
            automodel = policy.get("automodel_cfg", {})
            return {
                "tp": automodel.get("tensor_parallel_size", 1),
                "pp": 1,
                "cp": automodel.get("context_parallel_size", 1),
                "ep": automodel.get("expert_parallel_size", 1),
                "etp": 1,
            }
        else:  # megatron
            megatron = policy.get("megatron_cfg", {})
            return {
                "tp": megatron.get("tensor_model_parallel_size", 1),
                "pp": megatron.get("pipeline_model_parallel_size", 1),
                "cp": megatron.get("context_parallel_size", 1),
                "ep": megatron.get("expert_model_parallel_size", 1),
                "etp": megatron.get("expert_tensor_parallel_size", 1),
            }

    def _resolve_nemo_rl_config(self, config: dict, presets: dict[str, Any]) -> dict:
        """Resolve NeMo-RL format preset with overrides.

        Dynamically builds ``env.nemo_gym.config_paths`` from the
        ``environments`` dict.  When ``training_datasets`` is present
        (multi-environment combined training), injects ``data.train``
        and ``data.validation`` as lists so NeMo-RL uses its native
        multi-dataset support instead of a single merged file.

        For single-environment training, merges the environment's
        ``training_policy`` onto the resolved policy config (e.g. to
        override context length).  Combined training uses the
        model-level default and ignores per-environment overrides.
        """
        preset_name = config.get("preset")
        if not preset_name or preset_name not in presets:
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {', '.join(presets)}")

        console.detail("Using GRPO preset", preset_name)

        preset = OmegaConf.create(presets[preset_name])
        overrides = OmegaConf.create(config.get("overrides", {}))
        merged = cast(dict[Any, Any], OmegaConf.to_container(OmegaConf.merge(preset, overrides)))

        environments = config["environments"]
        config_paths = [VLLM_MODEL_FOR_TRAINING]
        for env_cfg in environments.values():
            config_paths.extend(env_cfg.get("config_paths", []))
        nemo_gym = merged.setdefault("env", {}).setdefault("nemo_gym", {})
        nemo_gym["config_paths"] = config_paths
        # Reuse baked Gym venvs if present, else build once (skip_venv_if_present).
        nemo_gym["skip_venv_if_present"] = True
        # Keep project-owned Gym values in the workflow instead of separate
        # overlay files, then compose them into Gym's runtime config.
        for env_cfg in environments.values():
            if gym_overrides := env_cfg.get("gym"):
                merged["env"]["nemo_gym"] = cast(
                    dict[Any, Any],
                    OmegaConf.to_container(
                        OmegaConf.merge(merged["env"]["nemo_gym"], gym_overrides)
                    ),
                )

        if config.get("training_datasets"):
            merged["data"]["train"] = config["training_datasets"]
        if config.get("validation_datasets"):
            merged["data"]["validation"] = config["validation_datasets"]

        if len(environments) == 1:
            env_name = next(iter(environments))
            env_cfg = environments[env_name]
            tp = env_cfg.get("training_policy")
            if tp:
                merged["policy"] = OmegaConf.to_container(
                    OmegaConf.merge(
                        OmegaConf.create(merged.get("policy", {})),
                        OmegaConf.create(tp),
                    )
                )
                console.detail("Training policy", f"Applied overrides from {env_name}")

        return merged

    def _auto_correct_sequence_parallel(self, nemo_rl_config: dict, backend: str) -> None:
        """Auto-correct sequence_parallel if TP=1 (requires TP>1)."""
        policy = nemo_rl_config.get("policy", {})
        parallel = self._get_parallelism_config(policy, backend)

        if backend == "fsdp":
            cfg = policy.get("automodel_cfg", {})
        else:
            cfg = policy.get("megatron_cfg", {})

        if parallel["tp"] == 1 and cfg.get("sequence_parallel", False):
            console.warning(
                f"Sequence parallelism requires TP > 1, but TP={parallel['tp']}. "
                f"Auto-disabling for {backend.upper()} backend."
            )
            cfg["sequence_parallel"] = False

    def _validate_parallelism_config(
        self, nemo_rl_config: dict, backend: str, num_nodes: int, num_gpus: int
    ) -> None:
        """Validate parallelism configuration before job submission.

        Validates that world_size is divisible by the parallelism product.
        Handles both dense and MoE (Mixture of Experts) models.

        Parallelism rules:
        - Dense models: world_size % (TP x PP x CP) == 0
        - MoE models:
          - FSDP: world_size % (TP x CP x EP) == 0
          - Megatron: world_size % (TP x PP x CP x EP) == 0
          - ETP (expert_tensor_parallel): Must satisfy TP % ETP == 0,
            but does NOT multiply into world_size.
        """
        world_size = num_nodes * num_gpus
        policy = nemo_rl_config.get("policy", {})
        parallel = self._get_parallelism_config(policy, backend)

        tp, pp, cp, ep = parallel["tp"], parallel["pp"], parallel["cp"], parallel["ep"]
        etp = parallel["etp"]

        is_moe = ep > 1

        if is_moe:
            if backend == "fsdp":
                parallelism_product = tp * cp * ep
                formula = f"TP×CP×EP = {tp}×{cp}×{ep}"
            else:  # megatron
                regular_model_size = tp * pp * cp
                expert_model_size = etp * ep * pp

                if world_size % regular_model_size != 0:
                    raise ValueError(
                        f"Parallelism validation failed for MoE model on MEGATRON backend:\n"
                        f"  world_size ({world_size}) must be divisible by TP×PP×CP = {tp}×{pp}×{cp} = {regular_model_size}\n"
                        f"  Regular layer DP would be: {world_size}/{regular_model_size} = "
                        f"{world_size / regular_model_size:.2f} (must be integer)"
                    )

                if world_size % expert_model_size != 0:
                    raise ValueError(
                        f"Parallelism validation failed for MoE model on MEGATRON backend:\n"
                        f"  world_size ({world_size}) must be divisible by ETP×EP×PP = {etp}×{ep}×{pp} = {expert_model_size}\n"
                        f"  Expert layer DP would be: {world_size}/{expert_model_size} = "
                        f"{world_size / expert_model_size:.2f} (must be integer)"
                    )

                regular_dp = world_size // regular_model_size
                expert_dp = world_size // expert_model_size

                if regular_dp != expert_dp:
                    warnings.warn(
                        f"MoE DP mismatch detected: Regular DP={regular_dp}, Expert DP={expert_dp}. "
                        f"This may cause distributed optimizer gradient buffer allocation issues. "
                        f"For best results, set CP=EP and ETP=TP to match both DPs.",
                        UserWarning,
                        stacklevel=2,
                    )

                if etp > 1 and tp % etp != 0:
                    raise ValueError(
                        f"Parallelism validation failed for MoE model on MEGATRON backend:\n"
                        f"  expert_tensor_parallel_size (ETP={etp}) must divide "
                        f"tensor_model_parallel_size (TP={tp}) evenly.\n"
                        f"  Currently: TP % ETP = {tp} % {etp} = {tp % etp} (must be 0)"
                    )

                parallelism_product = regular_model_size
                formula = f"TP×PP×CP = {tp}×{pp}×{cp}"
        else:
            if backend == "fsdp":
                parallelism_product = tp * cp
                formula = f"TP×CP = {tp}×{cp}"
            else:  # megatron
                parallelism_product = tp * pp * cp
                formula = f"TP×PP×CP = {tp}×{pp}×{cp}"

            if world_size % parallelism_product != 0:
                raise ValueError(
                    f"Parallelism validation failed for Dense model on {backend.upper()} backend:\n"
                    f"  world_size ({world_size}) must be divisible by {formula} = {parallelism_product}\n"
                    f"  Data parallel size would be: {world_size}/{parallelism_product} = "
                    f"{world_size / parallelism_product:.2f} (must be integer)"
                )

    def _validate_sequence_packing_for_cp(self, nemo_rl_config: dict, backend: str) -> None:
        """Validate sequence packing is enabled when using CP > 1 with Megatron."""
        if backend != "megatron":
            return

        policy = nemo_rl_config.get("policy", {})
        parallel = self._get_parallelism_config(policy, backend)

        if parallel["cp"] > 1:
            seq_packing = policy.get("sequence_packing", {})
            if not seq_packing.get("enabled", False):
                raise ValueError(
                    f"Sequence packing validation failed for MEGATRON backend:\n"
                    f"  context_parallel_size (CP={parallel['cp']}) > 1 requires "
                    f"sequence_packing.enabled=true\n"
                    f"  This is a Megatron Core requirement (FSDP does not need this).\n"
                    f"  Add to your config overrides:\n"
                    f"    policy:\n"
                    f"      sequence_packing:\n"
                    f"        enabled: true\n"
                    f"        train_mb_tokens: {policy.get('max_total_sequence_length', 4096)}"
                )

    def _auto_correct_sequence_length_divisibility(
        self, nemo_rl_config: dict, backend: str
    ) -> None:
        """Auto-compute policy.make_sequence_length_divisible_by from parallelism.

        Megatron splits individual sequences across CP and TP (when SP=true)
        ranks, requiring sequence lengths to be divisible by a minimum factor:
          - CP > 1 contributes cp_size * 2  (send/receive pattern)
          - TP > 1 + SP=true contributes tp_size

        If the user explicitly set a higher value (e.g., for FP8 alignment),
        it is preserved.  Mirrors the SFT stage's identical method.
        """
        policy = nemo_rl_config.get("policy", {})
        parallel = self._get_parallelism_config(policy, backend)

        if backend == "fsdp":
            cfg = policy.get("automodel_cfg", {})
        else:
            cfg = policy.get("megatron_cfg", {})

        tp = parallel["tp"]
        cp = parallel["cp"]
        sp = cfg.get("sequence_parallel", False)

        minimum = 1
        if cp > 1:
            minimum *= cp * 2
        if tp > 1 and sp:
            minimum *= tp

        current = policy.get("make_sequence_length_divisible_by", 1)
        corrected = max(current, minimum)

        if corrected != current:
            console.detail(
                "Auto-corrected make_sequence_length_divisible_by",
                f"{current} → {corrected} (CP={cp}, TP={tp}, SP={sp})",
            )
            policy["make_sequence_length_divisible_by"] = corrected

    def _inject_judge_config(
        self,
        config: dict[str, Any],
        nemo_rl_config: dict,
        output_dir: str,
        cluster_config: dict,
        judge_port: int | str | None = None,
    ) -> tuple[str, dict | None]:
        """Inject dedicated judge model config into NeMo-Gym and optionally
        build a judge job info dict for local_vllm mode.

        Judge configuration is read from the shared per-environment
        ``models.judge`` endpoint contract.

        Returns:
            ``(judge_mode, judge_job_info)`` where judge job information is set
            only in local-vLLM mode.
        """
        environments = config["environments"]
        judge_environments: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        for name, ecfg in environments.items():
            endpoint = (ecfg.get("models") or {}).get("judge") or {}
            if endpoint.get("deployment"):
                judge_environments.append((name, ecfg, endpoint))

        judge_endpoint = judge_environments[0][2] if judge_environments else {}
        conflicting = [
            name
            for name, _environment, endpoint in judge_environments[1:]
            if endpoint != judge_endpoint
        ]
        if conflicting:
            names = ", ".join([judge_environments[0][0], *conflicting])
            raise ValueError(
                "combined training requires one shared models.judge configuration; "
                f"judge settings differ across environments: {names}"
            )

        gym_model = judge_endpoint.get("gym_model") or {}
        serve_kwargs = gym_model.get("vllm_serve_kwargs") or {}
        judge_mode = _training_judge_mode(judge_endpoint)
        nemo_gym_cfg = nemo_rl_config.setdefault("env", {}).setdefault("nemo_gym", {})
        judge_fragment = _judge_model_fragment(judge_endpoint, judge_mode)
        if judge_fragment:
            merged = OmegaConf.to_container(
                OmegaConf.merge(
                    OmegaConf.create(nemo_gym_cfg),
                    OmegaConf.create(judge_fragment),
                )
            )
            nemo_rl_config["env"]["nemo_gym"] = merged
            nemo_gym_cfg = merged

        judge_job_info = None
        if judge_mode == "local_vllm":
            num_gpus = judge_endpoint.get("num_gpus", 4)
            num_nodes = judge_endpoint.get("server_nodes", 1)
            overrides = vllm_serve_overrides(serve_kwargs)
            server_args = build_vllm_server_args(overrides)
            if num_nodes > 1 and overrides.get("distributed_executor_backend") == "mp":
                server_args = f"{server_args} {MP_NODE_ARGS}"

            host_file = f"{output_dir}/judge_host.txt"
            serve_cmd = VLLM_LAUNCHER
            serve_environment = server_environment(
                parse_server_environment(
                    gym_model.get("vllm_serve_env_vars") or {},
                    "models.judge.gym_model.vllm_serve_env_vars",
                )
            )
            if serve_environment:
                assignments = shlex.join(
                    f"{key}={value}" for key, value in sorted(serve_environment.items())
                )
                serve_cmd = f"env {assignments} {serve_cmd}"
            vllm_cmd = (
                f"{serve_cmd}"
                f"    --model {judge_endpoint['model']}"
                f"    --num_gpus {num_gpus}"
                f"    --num_nodes {num_nodes}"
                '    --port "$JUDGE_PORT"'
                f"    {server_args}"
            )
            port_setup = (
                f"JUDGE_PORT={int(judge_port)}; "
                if judge_port is not None
                else (
                    "JUDGE_PORT=$(python3 - <<'PY'\n"
                    "import socket\n"
                    "with socket.socket() as sock:\n"
                    "    sock.bind(('', 0))\n"
                    "    print(sock.getsockname()[1])\n"
                    "PY\n"
                    "); "
                )
            )
            publish_host = (
                f'printf "%s:%s\\n" "$SERVER_HOST" "$JUDGE_PORT" > {shlex.quote(host_file)}'
            )
            if num_nodes > 1:
                publish_host = _on_first_node(publish_host)
            wrapped_cmd = (
                f"{port_setup}"
                "SERVER_HOST=; "
                'if [ "${NVFLOW_ENGINE:-}" = "ray" ]; then '
                f"SERVER_HOST=$({NODE_ADDRESS_COMMAND}); fi; "
                '[ -n "$SERVER_HOST" ] || SERVER_HOST=$(hostname); '
                f"{publish_host} && "
                'export NVFLOW_CODE_ROOT="${NVFLOW_CODE_ROOT:-/nemo_run/code}" && '
                f'nvidia-smi && cd "$NVFLOW_CODE_ROOT" && '
                f"export PYTHONPATH=$PYTHONPATH:$NVFLOW_CODE_ROOT && "
                f"{vllm_cmd}"
            )

            container_reference = _training_judge_container(config, judge_endpoint)
            judge_job_info = {
                "server_cmd": wrapped_cmd,
                "port": judge_port,
                "num_gpus": num_gpus,
                "num_nodes": num_nodes,
                "container": cluster_config["containers"].get(
                    container_reference,
                    container_reference,
                ),
                "container_role": container_reference,
                "host_file": host_file,
            }

            nemo_rl_config["env"]["nemo_gym"]["judge_model"]["responses_api_models"]["vllm_model"][
                "base_url"
            ] = "__JUDGE_URL_PLACEHOLDER__"

        return judge_mode, judge_job_info

    @staticmethod
    def _configure_runtime_bindings(
        config: dict[str, Any],
        nemo_rl_config: dict,
        output_dir: str,
        judge_mode: str,
    ) -> tuple[str, ...]:
        """Add worker-generated Gym overlays without exposing provider topology."""
        nemo_gym_cfg = nemo_rl_config.setdefault("env", {}).setdefault("nemo_gym", {})
        config_paths = nemo_gym_cfg.setdefault("config_paths", [])
        commands: list[str] = []
        for name, environment in config["environments"].items():
            environment_paths = environment.get("config_paths")
            agent_name = environment.get("agent_name")
            if (
                not isinstance(environment_paths, list)
                or not environment_paths
                or not all(isinstance(path, str) and path for path in environment_paths)
            ):
                raise ValueError(f"environment {name!r} requires config_paths")
            if not isinstance(agent_name, str) or not agent_name:
                raise ValueError(f"environment {name!r} requires agent_name")
            binding_root = f"{output_dir}/gym-bindings/{name}"
            commands.append(
                render_runtime_binding_command(
                    config_paths=environment_paths,
                    agent_name=agent_name,
                    output_config=f"{binding_root}.yaml",
                    output_binding=f"{binding_root}.json",
                    judge_model_name=(
                        "policy_model" if judge_mode == "policy_as_judge" else "judge_model"
                    ),
                    # Provider Gym configs can contain interpolations whose
                    # values are supplied by the recipe's environment overlay.
                    # Binding discovery must compose the same overlay even
                    # though it only inspects topology offline.
                    config_overrides=environment.get("gym"),
                )
            )
            config_paths.append(f"{binding_root}.yaml")
        return tuple(commands)

    def _prepare_grpo_config(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        presets: dict[str, Any],
        run_after: list[str] | None = None,
        cluster_config: dict | None = None,
        judge_port: int | str | None = None,
    ) -> PreparedGRPOConfig:
        """Merge preset + overrides, validate, and build PreparedGRPOConfig."""
        if cluster_config is None:
            from nvflow.integrations.nemo_skills import load_target_config

            cluster_config = load_target_config(cluster)

        hf_model_name = config["model_name"]
        backend = config.get("backend", "fsdp")

        # Resolve GPU layout: total_gpus (portable) or legacy num_nodes+num_gpus
        layout = resolve_gpu_layout(config, cluster_config)
        num_nodes = layout.num_nodes
        num_gpus = layout.gpus_per_node
        console.detail(
            "GPU layout", f"{num_nodes} node(s) x {num_gpus} GPUs = {layout.total_gpus} total"
        )

        nemo_rl_config = self._resolve_nemo_rl_config(config, presets)
        self._auto_correct_sequence_parallel(nemo_rl_config, backend)
        self._auto_correct_sequence_length_divisibility(nemo_rl_config, backend)
        self._validate_parallelism_config(nemo_rl_config, backend, num_nodes, num_gpus)
        self._validate_sequence_packing_for_cp(nemo_rl_config, backend)

        policy = nemo_rl_config.get("policy", {})
        parallel = self._get_parallelism_config(policy, backend)
        seq_k = policy.get("max_total_sequence_length", 32768) // 1024
        model_short = Path(hf_model_name).name.lower().replace("_", "-")
        run_name = f"grpo-{model_short}-{layout.total_gpus}g-tp{parallel['tp']}-cp{parallel['cp']}-seq{seq_k}k"
        output_dir = str(Path(config["output_dir"]) / run_name)

        judge_mode, judge_job_info = self._inject_judge_config(
            config,
            nemo_rl_config,
            output_dir,
            cluster_config,
            judge_port=judge_port,
        )
        gym_binding_commands = self._configure_runtime_bindings(
            config,
            nemo_rl_config,
            output_dir,
            judge_mode,
        )

        return PreparedGRPOConfig(
            nemo_rl_config=nemo_rl_config,
            run_name=run_name,
            output_dir=output_dir,
            expname=expname,
            hf_model_name=hf_model_name,
            num_nodes=num_nodes,
            num_gpus=num_gpus,
            backend=backend,
            judge_mode=judge_mode,
            judge_job_info=judge_job_info,
            gym_binding_commands=gym_binding_commands,
            run_after=run_after,
        )

    def _config_shell_snippet(self, prepared: PreparedGRPOConfig) -> tuple[str, str]:
        """Return a shell snippet that writes the NeMo-RL config YAML at job runtime.

        The config is base64-encoded and decoded inside the Slurm job,
        avoiding any host-side filesystem writes.
        """
        from nvflow.execution.runtime import write_payload_command

        content = yaml.dump(
            prepared.nemo_rl_config,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
        config_path = f"{prepared.output_dir}/grpo_config.yaml"
        snippet = f"mkdir -p {prepared.output_dir} && {write_payload_command(content, config_path)}"
        return snippet, config_path

    @staticmethod
    def _training_split_inputs(
        config: dict[str, Any],
    ) -> list[tuple[str, str]]:
        environments = list(config.get("environments", {}))
        if config.get("training_data"):
            environment = environments[0] if environments else "training"
            return [(environment, str(Path(str(config["training_data"])).parent))]
        datasets = config.get("training_datasets", [])
        return [
            (
                environments[index] if index < len(environments) else str(index),
                str(Path(str(dataset["data_path"])).parent),
            )
            for index, dataset in enumerate(datasets)
        ]

    def _build_train_cmd(
        self,
        prepared: PreparedGRPOConfig,
        config: dict[str, Any],
        config_snippet: str,
        config_path: str,
    ) -> str:
        """Build the training command string for run_grpo_nemo_gym.py."""
        from nvflow.execution.runtime import (
            CHECKPOINT_SAVE_BY_OVERRIDE,
            NRL_GYM_BIN,
            NRL_GYM_ROOT,
            NRL_PYTHON_PREAMBLE,
        )

        stage_kwargs = config.get("stage_kwargs", {})
        timeout = (
            config.get("overrides", {}).get("checkpointing", {}).get("checkpoint_must_save_by")
        )
        save_by = (
            f"++checkpointing.checkpoint_must_save_by={timeout}"
            if timeout
            else CHECKPOINT_SAVE_BY_OVERRIDE
        )
        hf_model = config.get("hf_checkpoint_path", config["model_name"])

        split_validation = ""
        if config.get("input_contract") == "train_validation_split":
            split_dirs = tuple(
                dict.fromkeys(
                    directory for _environment, directory in self._training_split_inputs(config)
                )
            )
            if not split_dirs:
                raise ValueError("train_validation_split input contract requires training data")
            split_validation = (
                " && ".join(
                    "python -m nvflow.stages.data.split_contract "
                    f"--output-dir {shlex.quote(directory)}"
                    for directory in split_dirs
                )
                + " && "
            )
        binding_snippet = ""
        if prepared.gym_binding_commands:
            # The subshell keeps Gym's venv and working directory out of training.
            binding_snippet = (
                f"(export PATH={NRL_GYM_BIN}:$PATH && cd {NRL_GYM_ROOT} && "
                + " && ".join(prepared.gym_binding_commands)
                + ") && "
            )
        cmd = (
            f"{NRL_PYTHON_PREAMBLE} && "
            f"{split_validation}"
            f"{config_snippet} && "
            'export NVFLOW_CODE_ROOT="${NVFLOW_CODE_ROOT:-/nemo_run/code}" && '
            f"export PYTHONPATH=$PYTHONPATH:$NVFLOW_CODE_ROOT:/opt/nemo-rl && "
            f"{binding_snippet}"
            f"echo 'Starting training' && "
            # NeMo-RL shells out to `uv`, which reads config by walking up from
            # the cwd: outside its own project dir it picks up this project's
            # [tool.uv] required-version and aborts.
            f"cd /opt/nemo-rl && "
            f"$NRL_PYTHON /opt/nemo-rl/examples/nemo_gym/run_grpo_nemo_gym.py "
            f"  --config {config_path}"
            f"  ++policy.model_name={hf_model}"
            f"  ++cluster.gpus_per_node={prepared.num_gpus}"
            f"  ++cluster.num_nodes={prepared.num_nodes}"
            f"  {save_by}"
            f"  ++logger.log_dir={prepared.output_dir}/training-logs"
            f"  ++checkpointing.checkpoint_dir={prepared.output_dir}/checkpoints"
        )

        if prepared.backend == "megatron":
            cmd += " ++policy.automodel_cfg.enabled=false ++policy.megatron_cfg.enabled=true"
            cmd += " ++policy.optimizer=null ++policy.dynamic_batching.enabled=false"
        else:
            cmd += " ++policy.automodel_cfg.enabled=true ++policy.megatron_cfg.enabled=false"

        if config.get("training_data"):
            cmd += f" ++data.train.data_path={config['training_data']}"
        if config.get("validation_data"):
            cmd += f" ++data.validation.data_path={config['validation_data']}"
        wandb_mode = config.get("wandb_mode", "disabled")
        if wandb_mode == "disabled":
            cmd += " ++logger.wandb_enabled=false"
        elif wandb_mode == "offline":
            cmd += " ++logger.wandb_enabled=true ++logger.wandb_mode=offline"
        elif wandb_mode == "online":
            wandb_project = config.get("wandb_project", "nvflow-grpo")
            cmd += (
                f" ++logger.wandb_enabled=true"
                f" ++logger.wandb.project={wandb_project}"
                f" ++logger.wandb.name={prepared.expname}"
                f" ++logger.wandb.group={prepared.expname}"
            )

        extra_parts = []
        if base_args := config.get("extra_arguments"):
            extra_parts.append(base_args)
        if stage_args := stage_kwargs.get("extra_arguments"):
            extra_parts.append(stage_args)
        if extra_parts:
            cmd += " " + " ".join(extra_parts)

        return cmd

    @staticmethod
    def _judge_pair_commands(
        prepared: PreparedGRPOConfig,
        train_cmd: str,
        job_id: int,
        dependent_jobs: int,
    ) -> tuple[str, str, str]:
        """Build the paired judge and training commands for one chain step."""
        judge = prepared.judge_job_info
        base_host_file = judge["host_file"]
        log_dir = f"{prepared.output_dir}/training-logs"
        host_file = (
            base_host_file.replace(".txt", f"_{job_id}.txt")
            if dependent_jobs > 0
            else base_host_file
        )
        raw_judge_cmd = judge["server_cmd"].replace(base_host_file, host_file)
        remove_host = f"rm -f {host_file}"
        if judge["num_nodes"] > 1:
            remove_host = _on_first_node(remove_host)
        judge_cmd = f"{remove_host} && {raw_judge_cmd}"
        wait_and_cat = (
            f"n=0; while [ ! -f {host_file} ] && [ $n -lt 300 ]; do"
            f" sleep 2; n=$((n+1)); done; cat {host_file}"
        )
        judge_url_override = (
            "++env.nemo_gym.judge_model.responses_api_models.vllm_model.base_url="
            f"http://$({wait_and_cat})/v1"
        )
        judge_health_check = (
            "{ _nvflow_jhc() { "
            f"while [ ! -f {host_file} ]; do sleep 10; done; "
            f'JH=$(cat {host_file} 2>/dev/null || echo ""); '
            '[ -z "$JH" ] && return; '
            'echo "[nvflow] Judge host=$JH, waiting for /health..."; '
            'while ! curl -sf "http://$JH/health" >/dev/null 2>&1; do sleep 15; done; '
            'echo "[nvflow] Judge healthy, monitoring started"; '
            "F=0; "
            "while true; do "
            "  sleep 120; "
            '  if ! curl -sf "http://$JH/health" >/dev/null 2>&1; then '
            "    F=$((F+1)); "
            '    echo "[nvflow] Judge health check failed ($F/30)"; '
            "    [ $F -ge 30 ] && { "
            '      echo "[nvflow] Judge unreachable, triggering shutdown..."; '
            f"      touch {log_dir}/ENDED; "
            "      return; }; "
            "  else F=0; fi; "
            "done; "
            "}; _nvflow_jhc & } && "
        )
        return (
            f"{judge_health_check}{train_cmd} {judge_url_override}",
            judge_cmd,
            host_file,
        )

    @staticmethod
    def _resolve_run_configs(
        config: dict[str, Any],
        expname: str,
    ) -> list[tuple[str, dict[str, Any], str]]:
        """Resolve single- or multi-environment training inputs once.

        An environment that trains alone applies its ``training`` mapping over the
        stage settings, for example its own ``backend`` and ``total_gpus``.
        """
        from nvflow.integrations.nemo_gym.environments import deep_merge, training_environments

        environments = training_environments(config)
        if not environments:
            raise ValueError("no selected, enabled environment has raw_train_data configured")
        data_source_dir = config["data_source_dir"]
        train_filename = config.get("train_filename", "train.jsonl")
        val_filename = config.get("val_filename", "val.jsonl")
        if len(environments) == 1:
            env_name = next(iter(environments))
            env_cfg = environments[env_name]
            return [
                (
                    f"Training for environment: {env_name}",
                    {
                        **deep_merge(config, env_cfg.get("training") or {}),
                        "output_dir": f"{config['output_dir']}/{env_name}",
                        "training_data": f"{data_source_dir}/{env_name}/{train_filename}",
                        "validation_data": f"{data_source_dir}/{env_name}/{val_filename}",
                        "environments": {env_name: env_cfg},
                    },
                    f"{expname}-{env_name}",
                )
            ]

        env_names = list(environments)
        combined_label = "+".join(env_names)
        train_datasets = []
        val_datasets = []
        for env_name, env_cfg in environments.items():
            train_entry: dict[str, Any] = {
                "data_path": f"{data_source_dir}/{env_name}/{train_filename}",
            }
            repeat = env_cfg.get("training_repeat", 1)
            if repeat > 1:
                train_entry["repeat"] = repeat
            train_datasets.append(train_entry)
            val_datasets.append(
                {
                    "data_path": f"{data_source_dir}/{env_name}/{val_filename}",
                }
            )

        return [
            (
                f"Training on combined environments: {combined_label}",
                {
                    **config,
                    "output_dir": f"{config['output_dir']}/{combined_label}",
                    "training_datasets": train_datasets,
                    "validation_datasets": val_datasets,
                    "environments": dict(environments),
                },
                f"{expname}-{combined_label}",
            )
        ]

    def plan(self, context: PlanningContext) -> StagePlan:
        """Build the GRPO service and training topology without submission."""
        from nvflow.execution import (
            ArtifactDeclaration,
            CommandPlan,
            DependencyPlan,
            ResourceRequest,
            ServicePlan,
            StagePlan,
            TaskPlan,
            opaque_payload_marker,
        )
        from nvflow.integrations.nemo_skills import portable_cluster_config

        presets = self._load_presets(context.launcher_config)
        tasks = []
        for _, run_config, run_expname in self._resolve_run_configs(
            dict(context.config),
            context.expname,
        ):
            gpus_per_node = context.target_gpus_per_node("nemo-rl")
            if (
                context.engine == "ray"
                and gpus_per_node is None
                and ("total_gpus" in run_config or "num_gpus" not in run_config)
            ):
                raise ValueError(
                    "Ray target route 'nemo-rl' must declare gpus_per_node "
                    "for target-resolved GRPO layout"
                )
            if (
                context.live
                and not context.run_after
                and run_config.get("input_contract") == "train_validation_split"
            ):
                assert context.filesystem is not None
                markers = [
                    f"{directory}/split.done"
                    for _environment, directory in self._training_split_inputs(run_config)
                ]
                present = context.filesystem.batch_exists(markers)
                missing = [path for path in markers if not present.get(path, False)]
                if missing:
                    raise ValueError(
                        "live GRPO training plan requires completed split marker(s): "
                        + ", ".join(missing)
                    )
            prepared = self._prepare_grpo_config(
                run_config,
                context.target.cluster,
                run_expname,
                presets,
                run_after=list(context.run_after) or None,
                cluster_config=portable_cluster_config(
                    gpus_per_node=gpus_per_node or 8,
                ),
                judge_port=None,
            )
            config_snippet, config_path = self._config_shell_snippet(prepared)
            train_cmd = self._build_train_cmd(
                prepared,
                run_config,
                config_snippet,
                config_path,
            )
            stage_kwargs = run_config.get("stage_kwargs", {})
            installation_command = run_config.get("installation_command") or stage_kwargs.get(
                "installation_command"
            )
            if installation_command:
                train_cmd = f"set -e\n{installation_command}\n{train_cmd}"
            payload_marker = opaque_payload_marker(
                config_snippet,
                label="nemo-rl-config",
            )
            display_cmd = train_cmd.replace(
                config_snippet,
                f"mkdir -p {prepared.output_dir} && write {payload_marker} to {config_path}",
            )
            dependent_jobs = run_config.get("dependent_jobs", 0)
            previous_train_id = None
            for job_id in range(dependent_jobs + 1):
                train_id = f"{context.stage}.{run_expname}.policy.{job_id}"
                dependencies: list[DependencyPlan] = []
                if previous_train_id:
                    dependencies.append(
                        DependencyPlan(
                            previous_train_id,
                            condition="after_any",
                        )
                    )
                else:
                    dependencies.extend(
                        DependencyPlan(name, external=True) for name in context.run_after
                    )

                if prepared.judge_job_info is None:
                    train_command = train_cmd
                    display_train_command = display_cmd
                else:
                    train_command, judge_command, judge_host_file = self._judge_pair_commands(
                        prepared,
                        train_cmd,
                        job_id,
                        dependent_jobs,
                    )
                    display_train_command, _, _ = self._judge_pair_commands(
                        prepared,
                        display_cmd,
                        job_id,
                        dependent_jobs,
                    )
                    judge_id = f"{context.stage}.{run_expname}.judge.{job_id}"
                    tasks.append(
                        TaskPlan(
                            task_id=judge_id,
                            stage=context.stage,
                            name=f"{prepared.expname}-judge-{job_id}",
                            operation="service",
                            commands=(
                                CommandPlan(
                                    command=judge_command,
                                    container=prepared.judge_job_info["container_role"],
                                    runtime_role=prepared.judge_job_info["container_role"],
                                    role="judge",
                                    kind="service",
                                    readiness=f"files:{judge_host_file}",
                                ),
                            ),
                            resources=(
                                ResourceRequest(
                                    role="judge",
                                    nodes=prepared.judge_job_info["num_nodes"],
                                    gpus_per_node=prepared.judge_job_info["num_gpus"],
                                    partition=stage_kwargs.get("partition"),
                                ),
                            ),
                            dependencies=(DependencyPlan(train_id, condition="after_start"),),
                            services=(
                                ServicePlan(
                                    name=f"judge-{job_id}",
                                    task_id=judge_id,
                                    readiness=f"files:{judge_host_file}",
                                    consumer_task_id=train_id,
                                ),
                            ),
                            parameters={
                                "experiment_name": prepared.expname,
                                "nemo_rl_group": True,
                                "slurm": {"log_dir": f"{prepared.output_dir}/training-logs"},
                            },
                        )
                    )

                tasks.append(
                    TaskPlan(
                        task_id=train_id,
                        stage=context.stage,
                        name=f"{prepared.expname}-grpo-{job_id}",
                        operation="nemo_rl.grpo",
                        commands=(
                            CommandPlan(
                                command=train_command,
                                container="nemo-rl",
                                runtime_role="nemo-rl",
                                display_command=display_train_command,
                            ),
                        ),
                        resources=(
                            ResourceRequest(
                                role="policy",
                                nodes=prepared.num_nodes,
                                gpus_per_node=prepared.num_gpus,
                                partition=stage_kwargs.get("partition"),
                            ),
                        ),
                        dependencies=tuple(dependencies),
                        outputs=(
                            ArtifactDeclaration(
                                name="checkpoint",
                                path=f"{prepared.output_dir}/checkpoints",
                                kind="checkpoint",
                            ),
                        ),
                        parameters={
                            "experiment_name": prepared.expname,
                            "ray_application": True,
                            "slurm": {
                                "sbatch_kwargs": stage_kwargs.get(
                                    "sbatch_kwargs",
                                    "",
                                ),
                                "log_dir": f"{prepared.output_dir}/training-logs",
                            },
                        },
                    )
                )
                if prepared.judge_job_info is not None:
                    tasks.append(
                        TaskPlan(
                            task_id=f"{context.stage}.{run_expname}.cleanup.{job_id}",
                            stage=context.stage,
                            name=f"{prepared.expname}-judge-cleanup-{job_id}",
                            operation="cleanup",
                            commands=(
                                CommandPlan(
                                    command=f"scancel --name={prepared.expname}-judge-{job_id}",
                                    container="host",
                                    runtime_role="host",
                                ),
                            ),
                            resources=(ResourceRequest(role="cleanup", gpus_per_node=0),),
                            dependencies=(DependencyPlan(train_id, condition="after_any"),),
                            parameters={
                                "experiment_name": prepared.expname,
                                "nemo_rl_group": True,
                                "slurm_cleanup_target": (f"{prepared.expname}-judge-{job_id}"),
                                "slurm_cleanup_target_task": judge_id,
                                "slurm": {"log_dir": (f"{prepared.output_dir}/training-logs")},
                            },
                        )
                    )
                previous_train_id = train_id
        return StagePlan(stage=context.stage, tasks=tuple(tasks))

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate configuration."""
        required = ["output_dir", "model_name", "data_source_dir"]
        for required_field in required:
            if required_field not in config:
                raise ValueError(f"'{required_field}' is required in GRPO config")

        environments = config.get("environments")
        if not isinstance(environments, dict) or not environments:
            raise ValueError("'environments' dict is required in GRPO training config")

        for environment in environments.values():
            if not isinstance(environment, dict):
                continue
            models = environment.get("models")
            endpoint = models.get("judge") if isinstance(models, dict) else None
            if isinstance(endpoint, dict) and endpoint.get("deployment") in {"gym", "nvflow"}:
                _training_judge_container(config, endpoint)

        preset_name = config.get("preset")
        if not preset_name or not config.get("presets_file"):
            raise ValueError(
                "'preset' and 'presets_file' (a project:// URI to the presets YAML) are required"
            )
        presets = self._load_presets(config)
        if preset_name not in presets:
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {', '.join(presets)}")
