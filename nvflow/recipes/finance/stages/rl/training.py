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

Uses NeMo-RL + NeMo-Gym via the nemo-skills grpo_nemo_rl() orchestrator.
The NeMo-Gym entry point swap and dependencies are configured in the workflow
YAML (installation_command) for full transparency.

The full config (preset + overrides) is written as a YAML file and passed
to run_grpo_nemo_gym.py via ``--config``.  nemo-skills' runtime overrides
(model_name, cluster, checkpoint_dir, etc.) are applied on top via
``++key=value`` CLI args.  This "pass everything from scratch" approach
matches SFT and avoids any dependency on upstream default config files.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf

from nvflow.core import BaseStage, StageRegistry, console
from nvflow.lib.rl.helpers import (
    JUDGE_NON_VLLM_KEYS,
    build_judge_nemo_gym_config,
    build_vllm_server_args,
    determine_judge_mode,
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
    parallelism, and submits via nemo-skills grpo_nemo_rl().

    Example workflow config::

        training:
          preset: "grpo-base"
          backend: fsdp
          overrides:
            grpo:
              num_prompts_per_step: 64
            policy:
              dtensor_cfg:
                tensor_parallel_size: 2
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
        """Resolve NeMo-RL format preset with overrides."""
        preset_name = config.get("preset")
        if not preset_name or preset_name not in self.presets:
            raise ValueError(
                f"Unknown preset '{preset_name}'. Available: {', '.join(self.presets.keys())}"
            )

        console.detail("Using GRPO preset", preset_name)

        preset = OmegaConf.create(self.presets[preset_name])
        overrides = OmegaConf.create(config.get("overrides", {}))
        merged = OmegaConf.to_container(OmegaConf.merge(preset, overrides))

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
                    import warnings

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

    def _inject_judge_config(
        self,
        config: dict[str, Any],
        nemo_rl_config: dict,
        output_dir: str,
        cluster: str,
    ) -> tuple[str, dict | None]:
        """Inject dedicated judge model config into NeMo-Gym and optionally
        build a judge job info dict for local_vllm mode.

        Returns:
            (judge_mode, judge_job_info) where judge_job_info is set only in
            local_vllm mode.
        """
        judge_mode = determine_judge_mode(config)
        if judge_mode != "policy_as_judge":
            validate_judge_config(config)

        config_paths = nemo_rl_config.get("env", {}).get("nemo_gym", {}).get("config_paths", [])
        env_name = next(
            (Path(p).stem for p in config_paths if "resources_servers" in p),
            "equivalence_llm_judge",
        )
        nemo_gym_cfg = nemo_rl_config.setdefault("env", {}).setdefault("nemo_gym", {})

        judge_cfg_fragment = build_judge_nemo_gym_config(
            config,
            judge_mode,
            environment_name=env_name,
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
            jcfg = config.get("judge_vllm") or {}
            vllm_overrides = {k: v for k, v in jcfg.items() if k not in JUDGE_NON_VLLM_KEYS}
            from nemo_skills.pipeline.utils.server import get_free_port

            judge_port = get_free_port(strategy="random")
            num_gpus = jcfg.get("num_gpus", 4)
            num_nodes = jcfg.get("server_nodes", 1)
            server_args = build_vllm_server_args(vllm_overrides)

            host_file = f"{output_dir}/judge_host.txt"
            vllm_cmd = (
                f"python3 -m nemo_skills.inference.server.serve_vllm"
                f"    --model {jcfg['model_path']}"
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

            from nemo_skills.pipeline.utils.cluster import get_cluster_config

            cluster_config = get_cluster_config(cluster)
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
    ) -> PreparedGRPOConfig:
        """Merge preset + overrides, validate, and build PreparedGRPOConfig."""
        hf_model_name = config["model_name"]
        num_nodes = config.get("num_nodes", 1)
        num_gpus = config.get("num_gpus", 8)
        backend = config.get("backend", "fsdp")

        nemo_rl_config = self._resolve_nemo_rl_config(config)
        self._auto_correct_sequence_parallel(nemo_rl_config, backend)
        self._validate_parallelism_config(nemo_rl_config, backend, num_nodes, num_gpus)
        self._validate_sequence_packing_for_cp(nemo_rl_config, backend)

        policy = nemo_rl_config.get("policy", {})
        parallel = self._get_parallelism_config(policy, backend)
        seq_k = policy.get("max_total_sequence_length", 32768) // 1024
        model_short = Path(hf_model_name).name.lower().replace("_", "-")
        run_name = (
            f"grpo-{model_short}-{num_nodes}n-tp{parallel['tp']}-cp{parallel['cp']}-seq{seq_k}k"
        )
        output_dir = str(Path(config["output_dir"]) / run_name)

        judge_mode, judge_job_info = self._inject_judge_config(
            config, nemo_rl_config, output_dir, cluster
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
        if config.get("training_data"):
            console.detail("Training data", config["training_data"])
        if config.get("validation_data"):
            console.detail("Validation data", config["validation_data"])
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

        env = prepared.nemo_rl_config.get("env", {})
        config_paths = env.get("nemo_gym", {}).get("config_paths", [])
        env_names = [Path(p).stem for p in config_paths if "resources_servers" in p]
        if env_names:
            console.detail("NeMo-Gym environment", ", ".join(env_names))

        log_judge_details(console, config, prepared.judge_mode)
        if prepared.judge_job_info:
            console.detail(
                "Judge Slurm GPUs",
                f"{prepared.judge_job_info['num_gpus']} (separate job)",
            )
        console.blank()

    def _write_config_yaml(self, prepared: PreparedGRPOConfig) -> str:
        """Write the full NeMo-RL config as a YAML file and return its container path."""
        output_path = resolve_host_path(prepared.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        config_file = output_path / "grpo_config.yaml"

        with open(config_file, "w") as f:
            f.write("# Auto-generated GRPO config (preset + overrides)\n")
            f.write("# Passed to run_grpo_nemo_gym.py via --config\n\n")
            yaml.dump(
                prepared.nemo_rl_config,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

        console.detail("Config YAML written to", str(config_file))
        return f"{prepared.output_dir}/grpo_config.yaml"

    def _submit_grpo_job(
        self, prepared: PreparedGRPOConfig, cluster: str, config: dict[str, Any]
    ) -> None:
        """Submit GRPO training job via nemo-skills grpo_nemo_rl()."""
        from nemo_skills.pipeline.cli import grpo_nemo_rl, wrap_arguments

        self._save_grpo_metadata(prepared, config)
        config_path = self._write_config_yaml(prepared)

        wandb_mode = config.get("wandb_mode", "disabled")
        extra_parts = []
        if base_args := config.get("extra_arguments"):
            extra_parts.append(base_args)
        if stage_args := config.get("stage_kwargs", {}).get("extra_arguments"):
            extra_parts.append(stage_args)
        extra_arguments = " ".join(extra_parts) if extra_parts else None

        args = f"--config {config_path}"
        if config.get("training_data"):
            args = f"{args} ++data.train_jsonl_fpath={config['training_data']}"
        if config.get("validation_data"):
            args = f"{args} ++data.validation_jsonl_fpath={config['validation_data']}"
        if wandb_mode == "disabled":
            args = f"{args} ++logger.wandb_enabled=false"
        elif wandb_mode == "offline":
            args = f"{args} ++logger.wandb_enabled=true ++logger.wandb_mode=offline"
        if extra_arguments:
            args = f"{args} {extra_arguments}"

        grpo_kwargs: dict[str, Any] = {
            "ctx": wrap_arguments(args),
            "cluster": cluster,
            "expname": prepared.expname,
            "backend": prepared.backend,
            "output_dir": prepared.output_dir,
            "hf_model": config.get("hf_checkpoint_path", config["model_name"]),
            "num_gpus": prepared.num_gpus,
            "num_nodes": prepared.num_nodes,
            "dependent_jobs": config.get("dependent_jobs", 0),
            "installation_command": config.get("installation_command"),
        }

        if prepared.run_after:
            grpo_kwargs["run_after"] = prepared.run_after
        stage_kwargs = config.get("stage_kwargs", {})
        if "partition" in stage_kwargs:
            grpo_kwargs["partition"] = stage_kwargs["partition"]
        if wandb_mode == "online" and config.get("wandb_project"):
            grpo_kwargs["wandb_project"] = config["wandb_project"]

        if prepared.judge_job_info is not None:
            self._submit_judge_and_training(prepared, grpo_kwargs, cluster)
        else:
            grpo_nemo_rl(**grpo_kwargs)

        console.success("GRPO training job submitted")

    def _submit_judge_and_training(
        self,
        prepared: PreparedGRPOConfig,
        grpo_kwargs: dict[str, Any],
        cluster: str,
    ) -> None:
        """Submit judge vLLM and training as two separate Slurm jobs."""
        from nemo_skills.pipeline.cli import grpo_nemo_rl, wrap_arguments
        from nemo_skills.pipeline.utils.cluster import get_cluster_config
        from nemo_skills.pipeline.utils.exp import add_task, get_exp

        judge = prepared.judge_job_info
        cluster_config = get_cluster_config(cluster)

        # Poll for the judge hostname file (300 × 2s = 10 min timeout),
        # then inject the URL as a CLI override for the training job.
        host_file = judge["host_file"]
        wait_and_cat = (
            f"n=0; while [ ! -f {host_file} ] && [ $n -lt 300 ]; do"
            f" sleep 2; n=$((n+1)); done; cat {host_file}"
        )
        judge_url_override = (
            "++env.nemo_gym.judge_model.responses_api_models.vllm_model.base_url="
            f"http://$({wait_and_cat})/v1"
        )

        original_args = " ".join(grpo_kwargs["ctx"].args)
        grpo_kwargs["ctx"] = wrap_arguments(f"{original_args} {judge_url_override}")

        resolve_host_path(host_file).unlink(missing_ok=True)

        with get_exp(prepared.expname, cluster_config) as exp:
            add_task(
                exp,
                cmd=judge["server_cmd"],
                task_name=f"{prepared.expname}-judge",
                log_dir=f"{prepared.output_dir}/training-logs",
                container=judge["container"],
                num_gpus=judge["num_gpus"],
                num_nodes=judge["num_nodes"],
                cluster_config=cluster_config,
                run_after=prepared.run_after,
            )
            console.detail("Judge vLLM job", "submitted (separate Slurm job)")

            grpo_kwargs["_reuse_exp"] = exp
            grpo_nemo_rl(**grpo_kwargs)

            self._submit_judge_cleanup(exp, prepared, cluster_config)

    def _submit_judge_cleanup(
        self, exp, prepared: PreparedGRPOConfig, cluster_config: dict
    ) -> None:
        """Submit a bare sbatch job that cancels the judge after training."""
        import subprocess

        if not exp.jobs:
            return

        last_handle = exp.jobs[-1].handle
        if not last_handle:
            return

        # Handle format: "<scheme>://<empty>/<job_id>/master/0"
        try:
            _, _, path_str = last_handle.partition("://")
            slurm_job_id = path_str.split("/")[1]
            int(slurm_job_id)  # validate it's numeric
        except (ValueError, IndexError):
            console.warning(
                f"Could not extract Slurm job ID from handle '{last_handle}', "
                "skipping judge cleanup job"
            )
            return

        prefix = cluster_config.get("job_name_prefix", "")
        judge_job_name = f"{prefix}{prepared.expname}-judge"
        account = cluster_config.get("account", "")
        partition = cluster_config.get("cpu_partition") or cluster_config.get("partition", "batch")
        log_file = f"{prepared.output_dir}/training-logs/judge-cleanup-%j.log"

        sbatch_script = (
            "#!/bin/bash\n"
            f"#SBATCH --job-name={prefix}{prepared.expname}-judge-cleanup\n"
            f"#SBATCH --account={account}\n"
            f"#SBATCH --partition={partition}\n"
            "#SBATCH --nodes=1\n"
            "#SBATCH --ntasks=1\n"
            "#SBATCH --time=00:05:00\n"
            f"#SBATCH --output={log_file}\n"
            f"#SBATCH --error={log_file}\n"
            f"#SBATCH --dependency=afterany:{slurm_job_id}\n"
            f"scancel --name={judge_job_name} --user=$USER 2>/dev/null || true\n"
        )

        try:
            result = subprocess.run(
                ["sbatch"],
                input=sbatch_script,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                console.detail(
                    "Judge cleanup job",
                    f"submitted ({result.stdout.strip()})",
                )
            else:
                console.warning(f"Failed to submit judge cleanup job: {result.stderr.strip()}")
        except Exception as e:
            console.warning(f"Could not submit judge cleanup job: {e}")

    def _save_grpo_metadata(self, prepared: PreparedGRPOConfig, config: dict[str, Any]) -> None:
        """Save run metadata YAML for reproducibility."""
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        run_id = f"job_{slurm_job_id}" if slurm_job_id else datetime.now().strftime("%Y%m%d_%H%M%S")

        judge_info = prepared.judge_job_info
        metadata = {
            "start_time": datetime.now().isoformat(),
            "slurm_job_id": slurm_job_id,
            "status": "submitted",
            "format": "nemo_rl_grpo",
            "preset": config.get("preset"),
            "backend": prepared.backend,
            "run_name": prepared.run_name,
            "output_dir": prepared.output_dir,
            "hf_model_name": prepared.hf_model_name,
            "num_nodes": prepared.num_nodes,
            "num_gpus": prepared.num_gpus,
            "installation_command": config.get("installation_command"),
            "extra_arguments": config.get("extra_arguments"),
            "judge_mode": prepared.judge_mode,
            "judge_job_info": {
                "num_gpus": judge_info["num_gpus"],
                "port": judge_info["port"],
                "host_file": judge_info["host_file"],
            }
            if judge_info
            else None,
            "nemo_rl_config": prepared.nemo_rl_config,
        }

        output_path = resolve_host_path(prepared.output_dir)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            metadata_file = output_path / f"run_metadata_{run_id}.yaml"

            with open(metadata_file, "w") as f:
                f.write(f"# GRPO Run Metadata - {run_id}\n")
                f.write("# Auto-generated for reproducibility (NeMo-RL + NeMo-Gym)\n\n")
                yaml.dump(
                    metadata, f, default_flow_style=False, sort_keys=False, allow_unicode=True
                )

            console.detail("Run metadata saved", str(metadata_file))
        except (OSError, PermissionError) as e:
            console.warning(f"Could not save metadata: {e}")

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Prepare config, display summary, and submit GRPO training job."""
        prepared = self._prepare_grpo_config(config, cluster, expname, run_after=run_after)
        self._display_grpo_summary(prepared, config)
        self._submit_grpo_job(prepared, cluster, config)

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate configuration."""
        required = ["output_dir", "model_name"]
        for required_field in required:
            if required_field not in config:
                raise ValueError(f"'{required_field}' is required in GRPO config")

        preset_name = config.get("preset")
        if preset_name and preset_name not in self.presets:
            raise ValueError(
                f"Unknown preset '{preset_name}'. Available: {', '.join(self.presets.keys())}"
            )

        if config.get("dependent_jobs", 0) > 0 and not config.get("training_data"):
            raise ValueError("'training_data' is required when dependent_jobs > 0.")
