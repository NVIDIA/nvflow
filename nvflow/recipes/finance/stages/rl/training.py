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
"""GRPO Reinforcement Learning Training for financial reasoning models.

Submits GRPO training via direct ``add_task()`` calls to nemo-run,
bypassing ``nemo-skills`` ``grpo_nemo_rl()`` to avoid unwanted data-key
injections and the ``cp`` entry-point hack.

The full config (preset + overrides) is base64-encoded and decoded
inside the Slurm job, then passed to ``run_grpo_nemo_gym.py`` via
``--config``.  Runtime overrides (model_name, cluster, checkpoint_dir,
data paths, etc.) are applied on top via ``++key=value`` CLI args.
"""

import base64
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf

from nvflow.core import BaseStage, StageRegistry, console
from nvflow.lib.gpu_layout import resolve_gpu_layout
from nvflow.lib.rl.helpers import (
    NON_VLLM_KEYS,
    VLLM_MODEL_FOR_TRAINING,
    build_judge_nemo_gym_config,
    build_vllm_server_args,
    determine_judge_mode,
    launcher_is_remote,
    log_judge_details,
    resolve_host_path,
    validate_judge_config,
)


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
    run_after: list[str] | None = None


@StageRegistry.register(recipe="finance", workflow="grpo", stage="training")
class GRPOStage(BaseStage):
    """GRPO reinforcement learning training for financial reasoning.

    Merges a preset (grpo_presets.yaml) with workflow overrides, validates
    parallelism, builds ``env.nemo_gym.config_paths`` from the
    ``environments`` dict, and submits via direct ``add_task()`` calls.
    """

    def __init__(self):
        super().__init__()
        self._presets = None

    @property
    def presets(self):
        """Lazy-load presets from grpo_presets.yaml."""
        if self._presets is None:
            self._presets = {}
            recipe_dir = Path(__file__).parent.parent.parent
            presets_path = recipe_dir / "workflows" / "grpo" / "grpo_presets.yaml"

            if presets_path.exists():
                with open(presets_path) as f:
                    data = yaml.safe_load(f)
                    self._presets.update(data.get("presets", {}))
            else:
                console.warning(f"Presets file not found: {presets_path}")

        return self._presets

    def _get_parallelism_config(self, policy: dict, backend: str) -> dict[str, int]:
        """Extract parallelism configuration from policy based on backend.

        Returns:
            dict with keys: tp, pp, cp, ep, etp
        """
        if backend == "fsdp":
            dtensor = policy.get("dtensor_cfg", {})
            return {
                "tp": dtensor.get("tensor_parallel_size", 1),
                "pp": 1,
                "cp": dtensor.get("context_parallel_size", 1),
                "ep": dtensor.get("expert_parallel_size", 1),
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

    def _resolve_nemo_rl_config(self, config: dict) -> dict:
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
        if not preset_name or preset_name not in self.presets:
            raise ValueError(
                f"Unknown preset '{preset_name}'. Available: {', '.join(self.presets.keys())}"
            )

        console.detail("Using GRPO preset", preset_name)

        preset = OmegaConf.create(self.presets[preset_name])
        overrides = OmegaConf.create(config.get("overrides", {}))
        merged = OmegaConf.to_container(OmegaConf.merge(preset, overrides))

        environments = config["environments"]
        config_paths = [VLLM_MODEL_FOR_TRAINING]
        for env_cfg in environments.values():
            config_paths.extend(env_cfg.get("config_paths", []))
        nemo_gym = merged.setdefault("env", {}).setdefault("nemo_gym", {})
        nemo_gym["config_paths"] = config_paths
        # Reuse baked Gym venvs if present, else build once (skip_venv_if_present).
        nemo_gym["skip_venv_if_present"] = True

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
            cfg = policy.get("dtensor_cfg", {})
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
            cfg = policy.get("dtensor_cfg", {})
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
    ) -> tuple[str, dict | None]:
        """Inject dedicated judge model config into NeMo-Gym and optionally
        build a judge job info dict for local_vllm mode.

        Judge configuration is read from the per-environment ``judge_vllm``
        block.  For single-environment training, the judge comes from that
        environment.  For combined training, the first environment with a
        non-trivial judge (``num_gpus > 0`` or ``model_path`` set) is used.

        Returns:
            (judge_mode, judge_job_info) where judge_job_info is set only in
            local_vllm mode.
        """
        environments = config["environments"]

        judge_env_name = None
        judge_env_cfg: dict[str, Any] = {}
        judge_vllm_cfg: dict[str, Any] = {}
        for name, ecfg in environments.items():
            jv = ecfg.get("judge_vllm") or {}
            if jv.get("model_path") or jv.get("base_url") or jv.get("openai_base_url"):
                judge_env_name = name
                judge_env_cfg = ecfg
                judge_vllm_cfg = jv
                break

        if not judge_env_name:
            judge_env_name = next(iter(environments))
            judge_env_cfg = environments[judge_env_name]

        rs_name = judge_env_cfg.get("resources_server_name", judge_env_name)

        config_with_judge = {**config, "judge_vllm": judge_vllm_cfg}
        judge_mode = determine_judge_mode(config_with_judge)
        if judge_mode != "policy_as_judge":
            validate_judge_config(config_with_judge)

        nemo_gym_cfg = nemo_rl_config.setdefault("env", {}).setdefault("nemo_gym", {})

        judge_cfg_fragment = build_judge_nemo_gym_config(
            config_with_judge,
            judge_mode,
            environment_name=rs_name,
            environment_inner_name=judge_env_name,
        )

        if judge_cfg_fragment:
            merged = OmegaConf.to_container(
                OmegaConf.merge(
                    OmegaConf.create(nemo_gym_cfg),
                    OmegaConf.create(judge_cfg_fragment),
                )
            )
            nemo_rl_config["env"]["nemo_gym"] = merged

        judge_job_info = None
        if judge_mode == "local_vllm":
            vllm_overrides = {k: v for k, v in judge_vllm_cfg.items() if k not in NON_VLLM_KEYS}
            from nemo_skills.pipeline.utils.server import get_free_port

            judge_port = get_free_port(strategy="random")
            num_gpus = judge_vllm_cfg.get("num_gpus", 4)
            num_nodes = judge_vllm_cfg.get("server_nodes", 1)
            server_args = build_vllm_server_args(vllm_overrides)

            host_file = f"{output_dir}/judge_host.txt"
            serve_cmd = "python3 -m nemo_skills.inference.server.serve_vllm"
            vllm_cmd = (
                f"{serve_cmd}"
                f"    --model {judge_vllm_cfg['model_path']}"
                f"    --num_gpus {num_gpus}"
                f"    --num_nodes {num_nodes}"
                f"    --port {judge_port}"
                f"    {server_args}"
            )
            wrapped_cmd = (
                f'echo "$(hostname):{judge_port}" > {host_file} && '
                f"nvidia-smi && cd /nemo_run/code && "
                f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
                f"{vllm_cmd}"
            )

            judge_job_info = {
                "server_cmd": wrapped_cmd,
                "port": judge_port,
                "num_gpus": num_gpus,
                "num_nodes": num_nodes,
                "container": cluster_config["containers"]["vllm"],
                "host_file": host_file,
            }

            nemo_rl_config["env"]["nemo_gym"]["judge_model"]["responses_api_models"]["vllm_model"][
                "base_url"
            ] = "__JUDGE_URL_PLACEHOLDER__"

        return judge_mode, judge_job_info

    def _prepare_grpo_config(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
        cluster_config: dict | None = None,
    ) -> PreparedGRPOConfig:
        """Merge preset + overrides, validate, and build PreparedGRPOConfig."""
        if cluster_config is None:
            from nemo_skills.pipeline.utils.cluster import get_cluster_config

            cluster_config = get_cluster_config(cluster)

        hf_model_name = config["model_name"]
        backend = config.get("backend", "fsdp")

        # Resolve GPU layout: total_gpus (portable) or legacy num_nodes+num_gpus
        layout = resolve_gpu_layout(config, cluster_config)
        num_nodes = layout.num_nodes
        num_gpus = layout.gpus_per_node
        console.detail(
            "GPU layout", f"{num_nodes} node(s) x {num_gpus} GPUs = {layout.total_gpus} total"
        )

        nemo_rl_config = self._resolve_nemo_rl_config(config)
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
            config, nemo_rl_config, output_dir, cluster_config
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
            run_after=run_after,
        )

    def _display_grpo_summary(self, prepared: PreparedGRPOConfig, config: dict[str, Any]) -> None:
        """Display training job configuration summary."""
        world_size = prepared.num_nodes * prepared.num_gpus
        policy = prepared.nemo_rl_config.get("policy", {})
        parallel = self._get_parallelism_config(policy, prepared.backend)

        console.status("Preparing GRPO training job (NeMo-RL + NeMo-Gym)")
        console.detail("Model", prepared.hf_model_name)
        if config.get("training_datasets"):
            datasets = config["training_datasets"]
            console.detail("Training data", f"{len(datasets)} datasets (multi-environment)")
            for ds in datasets:
                repeat_str = f" (repeat={ds['repeat']})" if ds.get("repeat", 1) > 1 else ""
                console.detail("  Dataset", f"{ds['data_path']}{repeat_str}")
            val_datasets = config.get("validation_datasets", [])
            console.detail("Validation data", f"{len(val_datasets)} datasets")
        else:
            console.detail("Training data", config.get("training_data", "(from config)"))
            console.detail("Validation data", config.get("validation_data", "(from config)"))
        console.detail("Cluster", f"{prepared.num_nodes}×{prepared.num_gpus} = {world_size} GPUs")

        if prepared.backend == "fsdp":
            parallel_str = f"TP={parallel['tp']}, CP={parallel['cp']}"
        else:
            parallel_str = f"TP={parallel['tp']}, PP={parallel['pp']}, CP={parallel['cp']}"

        console.detail(f"Parallelism ({prepared.backend.upper()})", parallel_str)

        grpo = prepared.nemo_rl_config.get("grpo", {})
        console.detail(
            "GRPO",
            f"prompts/step={grpo.get('num_prompts_per_step', '?')}, "
            f"generations/prompt={grpo.get('num_generations_per_prompt', '?')}",
        )

        console.detail("NeMo-Gym environments", ", ".join(config["environments"].keys()))

        log_judge_details(console, config, prepared.judge_mode)
        if prepared.judge_job_info:
            console.detail(
                "Judge Slurm GPUs",
                f"{prepared.judge_job_info['num_gpus']} (separate job)",
            )
        console.blank()

    def _config_shell_snippet(self, prepared: PreparedGRPOConfig) -> tuple[str, str]:
        """Return a shell snippet that writes the NeMo-RL config YAML at job runtime.

        The config is base64-encoded and decoded inside the Slurm job,
        avoiding any host-side filesystem writes.  Same pattern used by
        ``PrepareDataForGRPOStage._overlay_shell_snippet``.
        """
        content = yaml.dump(
            prepared.nemo_rl_config,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
        encoded = base64.b64encode(content.encode()).decode()
        config_path = f"{prepared.output_dir}/grpo_config.yaml"
        snippet = f"mkdir -p {prepared.output_dir} && echo {encoded} | base64 -d > {config_path}"
        return snippet, config_path

    def _build_train_cmd(
        self,
        prepared: PreparedGRPOConfig,
        config: dict[str, Any],
        config_snippet: str,
        config_path: str,
        cluster_config: dict,
    ) -> str:
        """Build the training command string for run_grpo_nemo_gym.py."""
        from nemo_skills.pipeline.nemo_rl.grpo import get_timeout_str

        stage_kwargs = config.get("stage_kwargs", {})
        partition = stage_kwargs.get("partition")
        timeout = config.get("overrides", {}).get("checkpointing", {}).get(
            "checkpoint_must_save_by"
        ) or get_timeout_str(cluster_config, partition)
        hf_model = config.get("hf_checkpoint_path", config["model_name"])

        from nvflow.lib.runtime import NRL_PYTHON_PREAMBLE

        cmd = (
            f"{NRL_PYTHON_PREAMBLE} && "
            f"{config_snippet} && "
            f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code:/opt/nemo-rl && "
            f"echo 'Starting training' && "
            f"$NRL_PYTHON /opt/nemo-rl/examples/nemo_gym/run_grpo_nemo_gym.py "
            f"  --config {config_path}"
            f"  ++policy.model_name={hf_model}"
            f"  ++cluster.gpus_per_node={prepared.num_gpus}"
            f"  ++cluster.num_nodes={prepared.num_nodes}"
            f"  ++checkpointing.checkpoint_must_save_by={timeout}"
            f"  ++logger.log_dir={prepared.output_dir}/training-logs"
            f"  ++checkpointing.checkpoint_dir={prepared.output_dir}/checkpoints"
        )

        if prepared.backend == "megatron":
            cmd += " ++policy.dtensor_cfg.enabled=false ++policy.megatron_cfg.enabled=true"
            cmd += " ++policy.optimizer=null ++policy.dynamic_batching.enabled=false"
        else:
            cmd += " ++policy.dtensor_cfg.enabled=true ++policy.megatron_cfg.enabled=false"

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
            wandb_project = config.get("wandb_project", "finance-grpo")
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

    def _submit_grpo_job(
        self, prepared: PreparedGRPOConfig, cluster_config: dict, config: dict[str, Any]
    ) -> None:
        """Submit GRPO training job via direct add_task() + run_exp()."""
        from nemo_skills.pipeline.nemo_rl.grpo import parse_kwargs
        from nemo_skills.pipeline.utils.exp import add_task, get_exp, run_exp

        config_snippet, config_path = self._config_shell_snippet(prepared)

        train_cmd = self._build_train_cmd(
            prepared, config, config_snippet, config_path, cluster_config
        )
        stage_kwargs = config.get("stage_kwargs", {})
        partition = stage_kwargs.get("partition")
        sbatch_kwargs = parse_kwargs(stage_kwargs.get("sbatch_kwargs", ""))

        dependent_jobs = config.get("dependent_jobs", 0)

        if prepared.judge_job_info is not None:
            self._submit_judge_and_training(
                prepared,
                train_cmd,
                cluster_config,
                config,
                partition,
                sbatch_kwargs,
                dependent_jobs,
            )
        else:
            with get_exp(prepared.expname, cluster_config) as exp:
                prev_task = None
                for job_id in range(dependent_jobs + 1):
                    prev_task = add_task(
                        exp,
                        cmd=train_cmd,
                        task_name=f"{prepared.expname}-grpo-{job_id}",
                        log_dir=f"{prepared.output_dir}/training-logs",
                        container=cluster_config["containers"]["nemo-rl"],
                        num_gpus=prepared.num_gpus,
                        num_nodes=prepared.num_nodes,
                        cluster_config=cluster_config,
                        with_ray=True,
                        # Forward the cluster's Ray template (e.g. ray_enroot.sub.j2)
                        # so SLURM 25.x uses the enroot-compatible head/worker
                        # launch; without this nemo-run defaults to ray.sub.j2 and
                        # the Ray head container never becomes ready.
                        ray_template=cluster_config.get("ray_template"),
                        sbatch_kwargs=sbatch_kwargs,
                        installation_command=config.get("installation_command"),
                        partition=partition,
                        run_after=prepared.run_after,
                        task_dependencies=[prev_task] if prev_task else None,
                    )
                run_exp(exp, cluster_config, sequential=False)

        console.success("GRPO training job submitted")

    def _submit_judge_and_training(
        self,
        prepared: PreparedGRPOConfig,
        train_cmd: str,
        cluster_config: dict,
        config: dict[str, Any],
        partition: str | None,
        sbatch_kwargs: dict | None,
        dependent_jobs: int = 0,
    ) -> None:
        """Submit paired (judge + training) Slurm jobs.

        Each training job gets its own judge vLLM server so the judge
        doesn't time out when ``dependent_jobs > 0``.  For the default
        case (``dependent_jobs = 0``), this produces one judge + one
        training job, same as before.

        The training job is submitted first so the judge can use
        ``--dependency=after:<training_job_id>`` to avoid allocating
        GPUs before training is actually running.  A background
        health-check inside the training command detects judge
        failures and triggers graceful Ray shutdown via the ENDED
        file mechanism.
        """
        from nemo_skills.pipeline.utils.cluster import get_slurm_timeout_str
        from nemo_skills.pipeline.utils.exp import add_task, get_exp, run_exp

        judge = prepared.judge_job_info
        base_host_file = judge["host_file"]
        log_dir = f"{prepared.output_dir}/training-logs"
        training_timeout = get_slurm_timeout_str(cluster_config, partition, with_save_delay=False)

        with get_exp(prepared.expname, cluster_config) as exp:
            prev_train_task = None

            for job_id in range(dependent_jobs + 1):
                if dependent_jobs > 0:
                    host_file_i = base_host_file.replace(".txt", f"_{job_id}.txt")
                else:
                    host_file_i = base_host_file

                raw_judge_cmd = judge["server_cmd"].replace(base_host_file, host_file_i)
                judge_cmd_i = f"rm -f {host_file_i} && {raw_judge_cmd}"

                wait_and_cat = (
                    f"n=0; while [ ! -f {host_file_i} ] && [ $n -lt 300 ]; do"
                    f" sleep 2; n=$((n+1)); done; cat {host_file_i}"
                )
                judge_url_override = (
                    "++env.nemo_gym.judge_model.responses_api_models.vllm_model.base_url="
                    f"http://$({wait_and_cat})/v1"
                )

                judge_health_check = (
                    "{ _nvflow_jhc() { "
                    f"while [ ! -f {host_file_i} ]; do sleep 10; done; "
                    f'JH=$(cat {host_file_i} 2>/dev/null || echo ""); '
                    '[ -z "$JH" ] && return; '
                    'echo "[nvflow] Judge host=$JH, waiting for /health..."; '
                    'while ! curl -sf "http://$JH/health" >/dev/null 2>&1; do sleep 15; done; '
                    'echo "[nvflow] Judge healthy, monitoring started"; '
                    # Tolerance raised from 3x60s (~3min) to 30x120s (~60min): the judge is only
                    # needed during rollout collection, NOT during the (long) logprob/train phase.
                    # A transient blip or a scheduler preemption of the idle judge during that phase
                    # must NOT tear training down before it completes the step and checkpoints. If the
                    # judge is genuinely gone for ~60min, then shut down (step-2 rollouts can't score).
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

                train_cmd_i = f"{judge_health_check}{train_cmd} {judge_url_override}"

                prev_train_task = add_task(
                    exp,
                    cmd=train_cmd_i,
                    task_name=f"{prepared.expname}-grpo-{job_id}",
                    log_dir=log_dir,
                    container=cluster_config["containers"]["nemo-rl"],
                    num_gpus=prepared.num_gpus,
                    num_nodes=prepared.num_nodes,
                    cluster_config=cluster_config,
                    with_ray=True,
                    # Forward the cluster's Ray template (e.g. ray_enroot.sub.j2)
                    # so SLURM 25.x uses the enroot-compatible head/worker launch;
                    # without this nemo-run defaults to ray.sub.j2 and the Ray head
                    # container never becomes ready.
                    ray_template=cluster_config.get("ray_template"),
                    sbatch_kwargs=sbatch_kwargs,
                    installation_command=config.get("installation_command"),
                    partition=partition,
                    run_after=prepared.run_after,
                    task_dependencies=[prev_train_task] if prev_train_task else None,
                )

                judge_sbatch = {"dependency_type": "after", "time": training_timeout}
                add_task(
                    exp,
                    cmd=judge_cmd_i,
                    task_name=f"{prepared.expname}-judge-{job_id}",
                    log_dir=log_dir,
                    container=judge["container"],
                    num_gpus=judge["num_gpus"],
                    num_nodes=judge["num_nodes"],
                    cluster_config=cluster_config,
                    task_dependencies=[prev_train_task],
                    sbatch_kwargs=judge_sbatch,
                    partition=partition,
                )

            num_pairs = dependent_jobs + 1
            console.detail(
                "Judge + training jobs",
                f"{num_pairs} pair(s) submitted",
            )
            run_exp(exp, cluster_config, sequential=False)
            self._submit_judge_cleanup(exp, prepared, cluster_config, num_pairs=num_pairs)

    def _submit_judge_cleanup(
        self,
        exp,
        prepared: PreparedGRPOConfig,
        cluster_config: dict,
        num_pairs: int = 1,
    ) -> None:
        """Submit per-pair cleanup jobs that cancel each judge after its training job.

        ``exp.jobs`` is ordered ``[grpo-0, judge-0, grpo-1, judge-1, ...]``.
        For each pair *i*, a lightweight CPU job is submitted with
        ``--dependency=afterany:<grpo-i>`` that runs
        ``scancel --name=<judge-i>``.
        """
        if not exp.jobs:
            return

        prefix = cluster_config.get("job_name_prefix", "")
        account = cluster_config.get("account", "")
        partition = cluster_config.get("cpu_partition") or cluster_config.get("partition", "batch")
        remote = launcher_is_remote(cluster_config)
        if remote:
            from nemo_skills.pipeline.utils import get_unmounted_path

            host_log_dir = get_unmounted_path(
                cluster_config, f"{prepared.output_dir}/training-logs"
            )
        else:
            host_log_dir = resolve_host_path(f"{prepared.output_dir}/training-logs")
        log_file = f"{host_log_dir}/judge-cleanup-%j.log"

        for pair_idx in range(num_pairs):
            train_job_idx = pair_idx * 2  # grpo-0 at 0, grpo-1 at 2, ...

            try:
                handle = exp.jobs[train_job_idx].handle
            except IndexError:
                continue
            if not handle:
                continue

            # Handle format: "<scheme>://<empty>/<job_id>/master/0"
            try:
                _, _, path_str = handle.partition("://")
                slurm_job_id = path_str.split("/")[1]
                int(slurm_job_id)
            except (ValueError, IndexError):
                console.warning(
                    f"Could not extract Slurm job ID from handle '{handle}', "
                    f"skipping cleanup for pair {pair_idx}"
                )
                continue

            judge_name = f"{prefix}{prepared.expname}-judge-{pair_idx}"
            sbatch_script = (
                "#!/bin/bash\n"
                f"#SBATCH --job-name={prefix}{prepared.expname}-judge-cleanup-{pair_idx}\n"
                f"#SBATCH --account={account}\n"
                f"#SBATCH --partition={partition}\n"
                "#SBATCH --nodes=1\n"
                "#SBATCH --ntasks=1\n"
                "#SBATCH --gpus-per-node=0\n"
                "#SBATCH --time=00:05:00\n"
                f"#SBATCH --output={log_file}\n"
                f"#SBATCH --error={log_file}\n"
                f"#SBATCH --dependency=afterany:{slurm_job_id}\n"
                f"scancel --name={judge_name} --user=$USER 2>/dev/null || true\n"
            )

            try:
                if remote:
                    # Off-cluster: submit over the ssh tunnel (the launch host has
                    # no local sbatch).  Feed the script via heredoc on stdin --
                    # the same way `sbatch` reads a script piped to it.
                    from nemo_skills.pipeline.utils.cluster import get_tunnel

                    heredoc = f"sbatch <<'NVFLOW_SBATCH_EOF'\n{sbatch_script}\nNVFLOW_SBATCH_EOF\n"
                    res = get_tunnel(cluster_config).run(heredoc, hide=True, warn=True)
                    rc = getattr(res, "exited", getattr(res, "return_code", 1))
                    out, err = res.stdout, res.stderr
                else:
                    result = subprocess.run(
                        ["sbatch"],
                        input=sbatch_script,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    rc, out, err = result.returncode, result.stdout, result.stderr
                if rc == 0:
                    console.detail(
                        f"Judge cleanup (pair {pair_idx})",
                        f"submitted ({out.strip()}), depends on grpo job {slurm_job_id}",
                    )
                else:
                    console.warning(
                        f"Failed to submit judge cleanup for pair {pair_idx}: {err.strip()}"
                    )
            except Exception as e:
                console.warning(f"Could not submit judge cleanup for pair {pair_idx}: {e}")

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Prepare config, display summary, and submit GRPO training job.

        Single-environment mode (one env selected or only one defined):
            Trains on that environment's data with a single ``data.train``
            entry.  Output goes to ``{output_dir}/{env_name}/``.

        Multi-environment mode (multiple envs, via ``-e env1 env2`` or all):
            Builds ``data.train`` as a list of per-environment dataset
            entries (leveraging NeMo-RL's native multi-dataset support).
            NeMo-Gym routes each sample to the correct agent via
            ``agent_ref``.  Output goes to ``{output_dir}/{env1+env2+...}/``.
        """
        from nemo_skills.pipeline.utils.cluster import get_cluster_config

        from nvflow.lib.rl.helpers import resolve_environments

        environments = resolve_environments(config)
        data_source_dir = config["data_source_dir"]
        train_filename = config.get("train_filename", "train.jsonl")
        val_filename = config.get("val_filename", "validation.jsonl")
        cluster_config = get_cluster_config(cluster)

        if len(environments) == 1:
            env_name = next(iter(environments))
            env_cfg = environments[env_name]
            env_config = {
                **config,
                "output_dir": f"{config['output_dir']}/{env_name}",
                "training_data": f"{data_source_dir}/{env_name}/{train_filename}",
                "validation_data": f"{data_source_dir}/{env_name}/{val_filename}",
                "environments": {env_name: env_cfg},
                "judge_vllm": env_cfg.get("judge_vllm") or {},
            }
            console.status(f"Training for environment: {env_name}")
            prepared = self._prepare_grpo_config(
                env_config,
                cluster,
                f"{expname}-{env_name}",
                run_after=run_after,
                cluster_config=cluster_config,
            )
            self._display_grpo_summary(prepared, env_config)
            self._submit_grpo_job(prepared, cluster_config, env_config)
        else:
            env_names = list(environments.keys())
            combined_label = "+".join(env_names)
            combined_dir = f"{config['output_dir']}/{combined_label}"

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

            combined_judge_vllm: dict[str, Any] = {}
            for ecfg in environments.values():
                jv = ecfg.get("judge_vllm") or {}
                if jv.get("model_path") or jv.get("base_url") or jv.get("openai_base_url"):
                    combined_judge_vllm = jv
                    break

            combined_config = {
                **config,
                "output_dir": combined_dir,
                "training_datasets": train_datasets,
                "validation_datasets": val_datasets,
                "environments": dict(environments),
                "judge_vllm": combined_judge_vllm,
            }
            console.status(f"Training on combined environments: {combined_label}")
            prepared = self._prepare_grpo_config(
                combined_config,
                cluster,
                f"{expname}-{combined_label}",
                run_after=run_after,
                cluster_config=cluster_config,
            )
            self._display_grpo_summary(prepared, combined_config)
            self._submit_grpo_job(prepared, cluster_config, combined_config)

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate configuration."""
        required = ["output_dir", "model_name", "data_source_dir"]
        for required_field in required:
            if required_field not in config:
                raise ValueError(f"'{required_field}' is required in GRPO config")

        if not config.get("environments"):
            raise ValueError("'environments' dict is required in GRPO training config")

        preset_name = config.get("preset")
        if preset_name and preset_name not in self.presets:
            raise ValueError(
                f"Unknown preset '{preset_name}'. Available: {', '.join(self.presets.keys())}"
            )
