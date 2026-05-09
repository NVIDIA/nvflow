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
"""Supervised Fine-Tuning for financial reasoning models.

This module handles SFT training by calling the RL repo's run_sft.py directly
(same execution pattern as GRPO), bypassing nemo-skills' start_sft.py wrapper.

The full NeMo-RL config is base64-encoded into the Slurm command and decoded
at job runtime.  Runtime overrides (model_name, cluster, data paths, etc.)
are applied via ++key=value CLI args.

File organization:
1. Configuration Schemas - Data classes for prepared configs
2. SFTStage Class:
   a. Preset Loading - Load sft_presets.yaml
   b. Parallelism Helpers - Unified extraction of TP/PP/CP/EP
   c. Validation & Auto-Correction - Check configs before submission
   d. Main Config Preparation - Merge preset + overrides, validate
   e. Job Submission & Display - Submit via add_task/get_exp/run_exp
"""

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf

from nvflow.core import BaseStage, StageRegistry, console
from nvflow.lib.gpu_layout import resolve_gpu_layout

# ============================================================================
# Configuration Schemas
# ============================================================================


@dataclass
class PreparedNemoRLConfig:
    """Prepared training configuration for NeMo-RL format presets."""

    nemo_rl_config: dict  # The merged NeMo-RL config (policy, sft, checkpointing, data)
    run_name: str
    output_dir: str
    expname: str
    training_data: str
    validation_data: str | None
    hf_model_name: str
    hf_checkpoint_path: str
    num_nodes: int
    num_gpus: int
    dependent_jobs: int
    backend: str
    wandb_project: str | None
    wandb_mode: str
    preset: str | None = None
    extra_arguments: str | None = None


@StageRegistry.register(recipe="finance", workflow="sft", stage="training")
class SFTStage(BaseStage):
    """Supervised fine-tuning on financial reasoning data.

    All workflows use NeMo-RL's native config format with direct pass-through
    to nemo_rl.algorithms.sft.SFTTrainer. Config uses exact NeMo-RL paths:
      - policy.*: Model, parallelism, optimizer config (dtensor_cfg, megatron_cfg)
      - sft.*: Training loop config (epochs, steps, val_period)
      - checkpointing.*: Checkpoint save/resume config
      - data.*: Data processing config

    Architecture: sft-base preset (model-independent defaults) + workflow overrides

    Example workflow config:
        training:
          preset: "sft-base"
          backend: megatron  # or fsdp
          overrides:
            sft:
              max_num_epochs: 3
              val_period: 50
            policy:
              train_global_batch_size: 128
              max_total_sequence_length: 49152
              megatron_cfg:
                tensor_model_parallel_size: 4
                context_parallel_size: 8
                optimizer:
                  lr: 5e-6
    """

    # NeMo-RL format presets are detected by having 'policy' or 'sft' keys
    NEMO_RL_PRESET_KEYS = {"policy", "sft", "data", "logger"}

    def __init__(self):
        super().__init__()
        self._presets = None

    # ========================================================================
    # Preset Loading
    # ========================================================================

    @property
    def presets(self):
        """Lazy-load presets from sft_presets.yaml."""
        if self._presets is None:
            self._presets = {}
            # Presets are co-located with SFT workflow configs
            recipe_dir = Path(__file__).parent.parent.parent
            presets_path = recipe_dir / "workflows" / "sft" / "sft_presets.yaml"

            if presets_path.exists():
                with open(presets_path) as f:
                    data = yaml.safe_load(f)
                    self._presets.update(data.get("presets", {}))
            else:
                console.warning(f"Presets file not found: {presets_path}")

        return self._presets

    # ========================================================================
    # Helper Methods - Parallelism Configuration
    # ========================================================================

    def _get_parallelism_config(self, policy: dict, backend: str) -> dict[str, int]:
        """Extract parallelism configuration from policy based on backend.

        Returns a unified dict with parallelism values for both dense and MoE models.

        Returns:
            dict with keys: tp, pp, cp, ep, etp (expert parallelism if MoE)
        """
        if backend == "fsdp":
            # FSDP backend uses dtensor_cfg
            dtensor = policy.get("dtensor_cfg", {})
            return {
                "tp": dtensor.get("tensor_parallel_size", 1),
                "pp": 1,  # FSDP doesn't have pipeline parallelism
                "cp": dtensor.get("context_parallel_size", 1),
                "ep": dtensor.get("expert_parallel_size", 1),  # MoE: expert parallelism
                "etp": 1,  # FSDP doesn't have expert_tensor_parallel
            }
        else:  # megatron
            # Megatron backend uses megatron_cfg
            megatron = policy.get("megatron_cfg", {})
            return {
                "tp": megatron.get("tensor_model_parallel_size", 1),
                "pp": megatron.get("pipeline_model_parallel_size", 1),
                "cp": megatron.get("context_parallel_size", 1),
                "ep": megatron.get("expert_model_parallel_size", 1),  # MoE: expert parallelism
                "etp": megatron.get(
                    "expert_tensor_parallel_size", 1
                ),  # MoE: expert tensor parallel
            }

    def _is_nemo_rl_preset(self, preset: dict) -> bool:
        """Check if preset uses NeMo-RL format (has policy/sft keys)."""
        return bool(set(preset.keys()) & self.NEMO_RL_PRESET_KEYS)

    # ========================================================================
    # Config Resolution
    # ========================================================================

    def _resolve_nemo_rl_config(self, config: dict) -> dict:
        """Resolve NeMo-RL format preset with overrides."""
        preset_name = config.get("preset")
        if not preset_name or preset_name not in self.presets:
            raise ValueError(
                f"Unknown preset '{preset_name}'. Available: {', '.join(self.presets.keys())}"
            )

        console.detail("Using NeMo-RL preset", preset_name)

        # Deep merge preset with overrides
        preset = OmegaConf.create(self.presets[preset_name])
        overrides = OmegaConf.create(config.get("overrides", {}))
        merged = OmegaConf.to_container(OmegaConf.merge(preset, overrides))

        return merged

    # ========================================================================
    # Config Validation & Auto-Correction
    # ========================================================================

    def _auto_correct_sequence_parallel(self, nemo_rl_config: dict, backend: str) -> None:
        """Auto-correct sequence_parallel if TP=1 (requires TP>1).

        Sequence parallelism splits sequences across tensor parallel ranks,
        so it requires TP > 1 to function.
        """
        policy = nemo_rl_config.get("policy", {})
        parallel = self._get_parallelism_config(policy, backend)

        # Get the config dict for the specific backend to modify
        if backend == "fsdp":
            cfg = policy.get("dtensor_cfg", {})
        else:  # megatron
            cfg = policy.get("megatron_cfg", {})

        # Auto-disable if TP=1 but sequence_parallel=True
        if parallel["tp"] == 1 and cfg.get("sequence_parallel", False):
            console.warning(
                f"Sequence parallelism requires TP > 1, but TP={parallel['tp']}. "
                f"Auto-disabling for {backend.upper()} backend."
            )
            cfg["sequence_parallel"] = False

    def _auto_correct_sequence_length_divisibility(
        self, nemo_rl_config: dict, backend: str
    ) -> None:
        """Auto-compute policy.make_sequence_length_divisible_by from parallelism settings.

        Megatron splits individual sequences across CP and TP (when SP=true) ranks,
        requiring sequence lengths to be divisible by a minimum pad factor:
          - CP > 1 contributes cp_size * 2  (send/receive pattern)
          - TP > 1 + SP=true contributes tp_size

        Before NeMo-RL PR #2053, this was auto-computed internally.  That PR
        changed it to a user-provided validated parameter (for GRPO top-p/top-k
        sampling which needs higher alignment).  For SFT (no sampling), the
        minimum is always sufficient, so we restore auto-computation here.

        If the user explicitly set a higher value (e.g., for FP8 alignment),
        it is preserved.
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

    def _validate_parallelism_config(
        self, nemo_rl_config: dict, backend: str, num_nodes: int, num_gpus: int
    ) -> None:
        """Validate parallelism configuration before job submission.

        Validates that world_size is divisible by the parallelism product.
        Handles both dense and MoE (Mixture of Experts) models.

        Parallelism rules:
        - Dense models: world_size % (TP × PP × CP) == 0
        - MoE models:
          - FSDP: world_size % (TP × CP × EP) == 0
          - Megatron: world_size % (TP × PP × CP × EP) == 0
          - ETP (expert_tensor_parallel): Subdivides experts within TP ranks.
            Must satisfy TP % ETP == 0, but does NOT multiply into world_size.
            Example: TP=4, ETP=2 means each TP rank handles experts with ETP=2.

        This catches configuration errors before expensive SLURM job submission.
        """
        world_size = num_nodes * num_gpus
        policy = nemo_rl_config.get("policy", {})
        parallel = self._get_parallelism_config(policy, backend)

        # Extract parallelism values
        tp, pp, cp, ep = parallel["tp"], parallel["pp"], parallel["cp"], parallel["ep"]
        etp = parallel["etp"]

        # Determine if this is a MoE model (EP > 1 indicates MoE)
        is_moe = ep > 1

        # Calculate parallelism product based on model type
        if is_moe:
            # MoE model: Megatron uses TWO separate DP groups (regular and expert)
            # Reference: megatron/core/parallel_state.py lines 695-748
            if backend == "fsdp":
                parallelism_product = tp * cp * ep
                formula = f"TP×CP×EP = {tp}×{cp}×{ep}"
            else:  # megatron
                # Megatron MoE has separate parallelism for regular vs expert layers
                # Regular layers: DP_regular = world_size / (TP × PP × CP)
                # Expert layers:  DP_expert  = world_size / (ETP × EP × PP)
                # Both must be integers and ideally equal for distributed optimizer

                regular_model_size = tp * pp * cp
                expert_model_size = etp * ep * pp

                # Validate both divisions result in integer DP
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

                # Calculate actual DP sizes
                regular_dp = world_size // regular_model_size
                expert_dp = world_size // expert_model_size

                # Warn if DPs don't match (can cause distributed optimizer issues)
                if regular_dp != expert_dp:
                    import warnings

                    warnings.warn(
                        f"MoE DP mismatch detected: Regular DP={regular_dp}, Expert DP={expert_dp}. "
                        f"This may cause distributed optimizer gradient buffer allocation issues. "
                        f"For best results, set CP=EP and ETP=TP to match both DPs.",
                        UserWarning,
                        stacklevel=2,
                    )

                # Megatron MoE: Validate ETP subdivides TP evenly (if ETP specified)
                if etp > 1 and tp % etp != 0:
                    raise ValueError(
                        f"Parallelism validation failed for MoE model on MEGATRON backend:\n"
                        f"  expert_tensor_parallel_size (ETP={etp}) must divide "
                        f"tensor_model_parallel_size (TP={tp}) evenly.\n"
                        f"  Currently: TP % ETP = {tp} % {etp} = {tp % etp} (must be 0)"
                    )

                # Set for single validation message if needed (won't reach here)
                parallelism_product = regular_model_size
                formula = f"TP×PP×CP = {tp}×{pp}×{cp}"
        else:
            # Dense model: standard parallelism
            if backend == "fsdp":
                parallelism_product = tp * cp
                formula = f"TP×CP = {tp}×{cp}"
            else:  # megatron
                parallelism_product = tp * pp * cp
                formula = f"TP×PP×CP = {tp}×{pp}×{cp}"

            # Validate world_size divisibility for dense models
            if world_size % parallelism_product != 0:
                raise ValueError(
                    f"Parallelism validation failed for Dense model on {backend.upper()} backend:\n"
                    f"  world_size ({world_size}) must be divisible by {formula} = {parallelism_product}\n"
                    f"  Data parallel size would be: {world_size}/{parallelism_product} = "
                    f"{world_size / parallelism_product:.2f} (must be integer)"
                )

    def _validate_sequence_packing_for_cp(self, nemo_rl_config: dict, backend: str) -> None:
        """Validate sequence packing is enabled when using CP > 1 with Megatron.

        This is a Megatron Core specific requirement. FSDP does not require this.

        Reference: nemo_rl/models/policy/workers/megatron_policy_worker.py:634-637
        """
        if backend != "megatron":
            return  # FSDP does not have this requirement

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

    # ========================================================================
    # Main Configuration Preparation
    # ========================================================================

    def _prepare_nemo_rl_config(
        self, config: dict[str, Any], expname: str, cluster_config: dict | None = None
    ) -> PreparedNemoRLConfig:
        """Prepare training configuration for NeMo-RL format presets.

        Steps:
        1. Resolve NeMo-RL config (merge preset + overrides)
        2. Auto-correct invalid settings (e.g., sequence_parallel with TP=1)
        3. Validate parallelism configuration
        4. Generate run name and prepare job submission parameters
        """
        hf_model_name = config["model_name"]
        layout = resolve_gpu_layout(config, cluster_config)
        num_nodes = layout.num_nodes
        num_gpus = layout.gpus_per_node
        console.detail(
            "GPU layout", f"{num_nodes} node(s) x {num_gpus} GPUs = {layout.total_gpus} total"
        )
        # Default to megatron backend (more scalable for large models)
        # Switch to fsdp only when megatron is not supported or has issues
        backend = config.get("backend", "megatron")

        # Step 1: Resolve NeMo-RL config (preset + overrides)
        nemo_rl_config = self._resolve_nemo_rl_config(config)

        # Step 2: Auto-correct invalid settings
        self._auto_correct_sequence_parallel(nemo_rl_config, backend)
        self._auto_correct_sequence_length_divisibility(nemo_rl_config, backend)

        # Step 3: Validate parallelism configuration
        self._validate_parallelism_config(nemo_rl_config, backend, num_nodes, num_gpus)
        self._validate_sequence_packing_for_cp(nemo_rl_config, backend)

        # Step 4: Generate run name and prepare job parameters
        policy = nemo_rl_config.get("policy", {})
        parallel = self._get_parallelism_config(policy, backend)

        # Extract sequence length for run name
        seq_len = policy.get("max_total_sequence_length", 131072)
        seq_k = seq_len // 1024

        # Generate run name: model-{name}-{total_gpus}g-tp{tp}-pp{pp}-cp{cp}-seq{seq}k
        model_short = Path(hf_model_name).name.lower().replace("_", "-")
        run_name = (
            f"model-{model_short}-{layout.total_gpus}g-"
            f"tp{parallel['tp']}-pp{parallel['pp']}-cp{parallel['cp']}-seq{seq_k}k"
        )

        output_dir = str(Path(config["output_dir"]) / run_name)

        # Display configuration summary
        console.detail("Run name", run_name)
        console.detail("Output directory", output_dir)
        console.detail("Backend", backend.upper())

        dependent_jobs = config.get("dependent_jobs", 3)
        if dependent_jobs > 0:
            console.detail("Checkpoint resumption", f"enabled ({dependent_jobs} dependent jobs)")

        stage_kwargs = config.get("stage_kwargs", {})
        return PreparedNemoRLConfig(
            nemo_rl_config=nemo_rl_config,
            run_name=run_name,
            output_dir=output_dir,
            expname=expname,
            training_data=config["training_data"],
            validation_data=config.get("validation_data"),
            hf_model_name=hf_model_name,
            hf_checkpoint_path=config.get("hf_checkpoint_path", hf_model_name),
            num_nodes=num_nodes,
            num_gpus=num_gpus,
            dependent_jobs=dependent_jobs,
            backend=backend,
            wandb_project=config.get("wandb_project"),
            wandb_mode=config.get("wandb_mode", "online"),
            preset=config.get("preset"),
            extra_arguments=stage_kwargs.get("extra_arguments"),
        )

    # ========================================================================
    # Job Submission & Display
    # ========================================================================

    def _display_nemo_rl_summary(self, prepared: PreparedNemoRLConfig) -> None:
        """Display training job configuration summary for NeMo-RL format."""
        world_size = prepared.num_nodes * prepared.num_gpus
        policy = prepared.nemo_rl_config.get("policy", {})
        parallel = self._get_parallelism_config(policy, prepared.backend)

        console.status("Preparing SFT training job (NeMo-RL format)")
        console.detail("Model", prepared.hf_model_name)
        console.detail("Training data", prepared.training_data)
        if prepared.validation_data:
            console.detail("Validation data", prepared.validation_data)
        console.detail("Cluster", f"{prepared.num_nodes}×{prepared.num_gpus} = {world_size} GPUs")

        # Build parallelism display string
        # Show EP (expert parallelism) only for MoE models (EP > 1)
        is_moe = parallel["ep"] > 1
        if prepared.backend == "fsdp":
            if is_moe:
                parallel_str = (
                    f"TP={parallel['tp']}, CP={parallel['cp']}, EP={parallel['ep']} (MoE)"
                )
            else:
                parallel_str = f"TP={parallel['tp']}, CP={parallel['cp']}"
        else:  # megatron
            if is_moe:
                parallel_str = (
                    f"TP={parallel['tp']}, PP={parallel['pp']}, "
                    f"CP={parallel['cp']}, EP={parallel['ep']} (MoE)"
                )
            else:
                parallel_str = f"TP={parallel['tp']}, PP={parallel['pp']}, CP={parallel['cp']}"

        console.detail(f"Parallelism ({prepared.backend.upper()})", parallel_str)

        # Display batch configuration
        console.detail(
            "Batch",
            f"global={policy.get('train_global_batch_size', 128)}, "
            f"micro={policy.get('train_micro_batch_size', 1)}",
        )
        console.blank()

    def _config_shell_snippet(self, prepared: PreparedNemoRLConfig) -> tuple[str, str]:
        """Return a shell snippet that writes the NeMo-RL config YAML at job runtime.

        The config is base64-encoded and decoded inside the Slurm job,
        avoiding any host-side filesystem writes.
        """
        content = yaml.dump(
            prepared.nemo_rl_config,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
        encoded = base64.b64encode(content.encode()).decode()
        config_path = f"{prepared.output_dir}/sft_config.yaml"
        snippet = f"mkdir -p {prepared.output_dir} && echo {encoded} | base64 -d > {config_path}"
        return snippet, config_path

    def _build_train_cmd(
        self,
        prepared: PreparedNemoRLConfig,
        config: dict[str, Any],
        config_snippet: str,
        config_path: str,
        cluster_config: dict,
    ) -> str:
        """Build the training command string for run_sft.py."""
        from nemo_skills.pipeline.nemo_rl.grpo import get_timeout_str

        stage_kwargs = config.get("stage_kwargs", {})
        partition = stage_kwargs.get("partition")
        timeout = get_timeout_str(cluster_config, partition)
        hf_model = config.get("hf_checkpoint_path", config["model_name"])

        cmd = (
            f"{config_snippet} && "
            f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code:/opt/NeMo-RL && "
            f"export UV_PROJECT=/opt/NeMo-RL && "
            f"echo 'Starting training' && "
            f"uv run --active python /opt/NeMo-RL/examples/run_sft.py "
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
            cmd += " ++policy.optimizer=None ++policy.dynamic_batching.enabled=false"
        else:
            cmd += " ++policy.dtensor_cfg.enabled=true ++policy.megatron_cfg.enabled=false"

        if config.get("training_data"):
            cmd += f" ++data.train.data_path={config['training_data']}"
        if config.get("validation_data"):
            cmd += f" ++data.validation.data_path={config['validation_data']}"
        else:
            cmd += " ++data.validation=null"

        wandb_mode = config.get("wandb_mode", "disabled")
        if wandb_mode == "disabled":
            cmd += " ++logger.wandb_enabled=false"
        elif wandb_mode == "offline":
            cmd += " ++logger.wandb_enabled=true ++logger.wandb_mode=offline"
        elif wandb_mode == "online":
            wandb_project = config.get("wandb_project", "finance-sft")
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

    def _submit_sft_job(
        self,
        prepared: PreparedNemoRLConfig,
        cluster_config: dict,
        config: dict[str, Any],
        run_after: list[str] | None = None,
    ) -> None:
        """Submit SFT training job via direct add_task() + run_exp()."""
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

        with get_exp(prepared.expname, cluster_config) as exp:
            prev_task = None
            for job_id in range(dependent_jobs + 1):
                prev_task = add_task(
                    exp,
                    cmd=train_cmd,
                    task_name=f"{prepared.expname}-sft-{job_id}",
                    log_dir=f"{prepared.output_dir}/training-logs",
                    container=cluster_config["containers"]["nemo-rl"],
                    num_gpus=prepared.num_gpus,
                    num_nodes=prepared.num_nodes,
                    cluster_config=cluster_config,
                    with_ray=True,
                    sbatch_kwargs=sbatch_kwargs,
                    installation_command=stage_kwargs.get("installation_command"),
                    partition=partition,
                    run_after=run_after,
                    task_dependencies=[prev_task] if prev_task else None,
                )
            run_exp(exp, cluster_config, sequential=False)

        console.success("SFT training job submitted")

    # ========================================================================
    # Main Entry Points
    # ========================================================================

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute SFT training.

        Resolves cluster_config once and passes the dict to all downstream
        methods (same pattern as GRPO).
        """
        from nemo_skills.pipeline.utils import get_cluster_config

        cluster_config = get_cluster_config(cluster)
        prepared = self._prepare_nemo_rl_config(config, expname, cluster_config=cluster_config)
        self._display_nemo_rl_summary(prepared)
        self._submit_sft_job(prepared, cluster_config, config, run_after=run_after)

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate configuration.

        Only training_data, output_dir, and model_name are required.
        validation_data is optional. All other parameters have sensible defaults.
        """
        # Required fields
        required = ["training_data", "output_dir", "model_name"]
        for required_field in required:
            if required_field not in config:
                raise ValueError(f"'{required_field}' is required in config")
