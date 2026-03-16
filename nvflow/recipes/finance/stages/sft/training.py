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

This module handles SFT training using NeMo-RL's native config format.
Config is passed directly to nemo_rl.algorithms.sft.SFTTrainer without translation.

File organization:
1. Configuration Schemas - Data classes for prepared configs
2. SFTStage Class:
   a. Preset Loading - Load sft_presets.yaml
   b. Config Flattening - Convert dicts to Hydra overrides
   c. Parallelism Helpers - Unified extraction of TP/PP/CP/EP
   d. Validation & Auto-Correction - Check configs before submission
   e. Main Config Preparation - Merge preset + overrides, validate
   f. Job Submission & Display - Submit to nemo-skills
   g. Metadata Saving - Save run metadata for reproducibility
"""

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf

from nvflow.core import BaseStage, StageRegistry, console

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
    # Config Flattening & Resolution
    # ========================================================================

    def _flatten_config_to_args(self, config: dict, prefix: str = "") -> list[str]:
        """Recursively flatten config dict to Hydra override args.

        Converts nested dict like:
            {"policy": {"dtensor_cfg": {"tensor_parallel_size": 2}}}
        To:
            ["++policy.dtensor_cfg.tensor_parallel_size=2"]

        Skips complex nested structures (list of dicts) which should be in YAML presets.
        Skips env_vars dict to avoid Hydra type conversion issues (stays in merged YAML).
        """
        args = []
        for key, value in config.items():
            # Skip description field (not a NeMo-RL config)
            if key == "description":
                continue

            # Skip env_vars dict - it should stay in the merged config YAML
            # Flattening to CLI args causes Hydra to convert string values like "720" to int
            if key == "env_vars" and isinstance(value, dict):
                continue

            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                # Recurse into nested dicts
                args.extend(self._flatten_config_to_args(value, full_key))
            elif isinstance(value, bool):
                args.append(f"++{full_key}={str(value).lower()}")
            elif isinstance(value, list):
                # Skip complex nested structures (e.g., scheduler with list of dicts)
                # These should be defined in sft_presets.yaml, not flattened to CLI args
                if value and isinstance(value[0], dict):
                    continue
                # Simple lists like betas: [0.9, 0.98] → "[0.9,0.98]"
                list_str = "[" + ",".join(str(v) for v in value) + "]"
                args.append(f"++{full_key}={list_str}")
            elif value is not None:
                args.append(f"++{full_key}={value}")
        return args

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

    def _build_nemo_rl_training_args(self, nemo_rl_config: dict) -> str:
        """Build training arguments from NeMo-RL format config.

        Simply flattens the config dict to Hydra override args.

        Args:
            nemo_rl_config: NeMo-RL format config dict

        Returns:
            Formatted arguments string
        """
        args = self._flatten_config_to_args(nemo_rl_config)
        return " ".join(args)

    # ========================================================================
    # Main Configuration Preparation
    # ========================================================================

    def _prepare_nemo_rl_config(self, config: dict[str, Any], expname: str) -> PreparedNemoRLConfig:
        """Prepare training configuration for NeMo-RL format presets.

        Steps:
        1. Resolve NeMo-RL config (merge preset + overrides)
        2. Auto-correct invalid settings (e.g., sequence_parallel with TP=1)
        3. Validate parallelism configuration
        4. Generate run name and prepare job submission parameters
        """
        # Extract basic config
        hf_model_name = config["model_name"]
        num_nodes = config.get("num_nodes", 1)
        num_gpus = config.get("num_gpus", 8)
        # Default to megatron backend (more scalable for large models)
        # Switch to fsdp only when megatron is not supported or has issues
        backend = config.get("backend", "megatron")

        # Step 1: Resolve NeMo-RL config (preset + overrides)
        nemo_rl_config = self._resolve_nemo_rl_config(config)

        # Step 2: Auto-correct invalid settings
        self._auto_correct_sequence_parallel(nemo_rl_config, backend)

        # Step 3: Validate parallelism configuration
        self._validate_parallelism_config(nemo_rl_config, backend, num_nodes, num_gpus)
        self._validate_sequence_packing_for_cp(nemo_rl_config, backend)

        # Step 4: Generate run name and prepare job parameters
        policy = nemo_rl_config.get("policy", {})
        parallel = self._get_parallelism_config(policy, backend)

        # Extract sequence length for run name
        seq_len = policy.get("max_total_sequence_length", 131072)
        seq_k = seq_len // 1024

        # Generate run name: model-{name}-{nodes}n-tp{tp}-pp{pp}-cp{cp}-seq{seq}k
        model_short = Path(hf_model_name).name.lower().replace("_", "-")
        run_name = (
            f"model-{model_short}-{num_nodes}n-"
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

    def _submit_nemo_rl_job(
        self,
        prepared: PreparedNemoRLConfig,
        cluster: str,
        config: dict[str, Any],
        run_after: list[str] | None = None,
    ) -> None:
        """Submit SFT training job using NeMo-RL format config."""
        from nemo_skills.pipeline.cli import sft_nemo_rl, wrap_arguments

        self._save_nemo_rl_metadata(prepared)

        # Build training arguments by flattening the NeMo-RL config
        args = self._build_nemo_rl_training_args(prepared.nemo_rl_config)

        # Debug: Display generated training arguments
        console.blank()
        console.info("=" * 80)
        console.info("GENERATED TRAINING ARGUMENTS (Hydra overrides):")
        console.info("-" * 80)
        # Split args for readability (show each ++key=value on separate line)
        for arg in args.split(" ++"):
            if arg.startswith("++"):
                console.info(f"  {arg}")
            elif arg:
                console.info(f"  ++{arg}")
        console.info("=" * 80)
        console.blank()

        # Add W&B configuration to training args
        if prepared.wandb_mode == "disabled":
            args = f"{args} ++logger.wandb_enabled=false"
        elif prepared.wandb_mode == "offline":
            args = f"{args} ++logger.wandb_enabled=true ++logger.wandb_mode=offline"

        # Append extra_arguments from stage_kwargs
        stage_kwargs = config.get("stage_kwargs", {})
        if extra_args := stage_kwargs.get("extra_arguments"):
            args = f"{args} {extra_args}"
            console.detail("Extra arguments", extra_args)

        # Build job submission kwargs
        sft_kwargs: dict[str, Any] = {
            "ctx": wrap_arguments(args),
            "cluster": cluster,
            "expname": prepared.expname,
            "backend": prepared.backend,
            "output_dir": prepared.output_dir,
            "hf_model": prepared.hf_checkpoint_path,
            "training_data": prepared.training_data,
            "num_gpus": prepared.num_gpus,
            "num_nodes": prepared.num_nodes,
            "dependent_jobs": prepared.dependent_jobs,
        }

        if run_after:
            sft_kwargs["run_after"] = run_after

        # Optional kwargs from stage_kwargs
        for key in ("partition", "installation_command"):
            if key in stage_kwargs:
                sft_kwargs[key] = stage_kwargs[key]

        if prepared.validation_data:
            sft_kwargs["validation_data"] = prepared.validation_data

        if prepared.wandb_mode == "online" and prepared.wandb_project:
            sft_kwargs["wandb_project"] = prepared.wandb_project

        sft_nemo_rl(**sft_kwargs)
        console.success("SFT training job submitted")

    # ========================================================================
    # Metadata & Utilities
    # ========================================================================

    def _save_nemo_rl_metadata(self, prepared: PreparedNemoRLConfig) -> None:
        """Save run metadata YAML for NeMo-RL format config."""
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        run_id = f"job_{slurm_job_id}" if slurm_job_id else datetime.now().strftime("%Y%m%d_%H%M%S")

        metadata = {
            "start_time": datetime.now().isoformat(),
            "slurm_job_id": slurm_job_id,
            "status": "submitted",
            "format": "nemo_rl",
            "preset": prepared.preset,
            "backend": prepared.backend,
            "run_name": prepared.run_name,
            "output_dir": prepared.output_dir,
            "hf_model_name": prepared.hf_model_name,
            "num_nodes": prepared.num_nodes,
            "num_gpus": prepared.num_gpus,
            "nemo_rl_config": prepared.nemo_rl_config,
        }

        output_path = self._resolve_output_path(prepared.output_dir)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            metadata_file = output_path / f"run_metadata_{run_id}.yaml"

            with open(metadata_file, "w") as f:
                f.write(f"# Run Metadata - {run_id}\n")
                f.write("# Auto-generated for reproducibility (NeMo-RL format)\n\n")
                yaml.dump(
                    metadata, f, default_flow_style=False, sort_keys=False, allow_unicode=True
                )

            console.detail("Run metadata saved", str(metadata_file))
        except (OSError, PermissionError) as e:
            console.warning(f"Could not save metadata: {e}")

    def _resolve_output_path(self, output_dir: str) -> Path:
        """Resolve output path, handling /workspace vs local submission node."""
        if output_dir.startswith("/workspace/") and not Path("/workspace").exists():
            return Path(output_dir.replace("/workspace/", "./"))
        return Path(output_dir)

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

        This is the main entry point for SFT training. It:
        1. Prepares training configuration (NeMo-RL format)
        2. Displays training job summary
        3. Submits training job to cluster

        All presets now use NeMo-RL native format (policy.*, sft.*, etc.)
        """
        # Prepare and submit NeMo-RL format job
        prepared = self._prepare_nemo_rl_config(config, expname)
        self._display_nemo_rl_summary(prepared)
        self._submit_nemo_rl_job(prepared, cluster, config, run_after=run_after)

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
