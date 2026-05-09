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
"""Prepare data for GRPO training by running NeMo-Gym's ng_prepare_data.

ng_prepare_data stamps each JSONL record with an ``agent_ref`` field that
tells NeMo-Gym which agent server to route the example to during training.

This stage derives agent definitions from the top-level ``environments``
dict, generates an agent-config overlay YAML inside the Slurm job at
``{output_dir}/agent_config_overlay.yaml``, and passes it as the last
entry in ``+config_paths``.  Supports multiple environments/agents.
"""

import base64
from typing import Any

import yaml

from nvflow.core import BaseStage, StageRegistry, console


@StageRegistry.register(recipe="finance", workflow="grpo", stage="prepare_data")
class PrepareDataForGRPOStage(BaseStage):
    """Run ng_prepare_data to add agent_ref routing fields to JSONL data.

    CPU-only stage (no GPU needed).  Derives agent definitions from the
    top-level ``environments`` dict and supports multiple environments.

    Execution flow:
      1. Derive agent definitions from ``environments``.
      2. Build a shell snippet that writes the agent-config overlay to
         {output_dir}/agent_config_overlay.yaml at job runtime.
      3. Submit a Slurm job that first writes the overlay, then runs
         ``ng_prepare_data`` with the overlay appended to ``+config_paths``.
    """

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _agents_from_environments(environments: dict[str, Any]) -> list[dict[str, Any]]:
        """Derive the agents list from the ``environments`` dict.

        ``env_cfg["datasets"]`` is a dict keyed by NeMo-Gym DatasetType:
          - ``"train"`` — required.
          - ``"val"`` / ``"validation"`` — optional.  Omitted in the current
            GRPO pipeline because the post-rollout ``train_validation_split``
            stage produces the final val set (see base.yaml prepare_data
            docstring).  ng_prepare_data gracefully skips absent types
            (see Gym/nemo_gym/train_data_utils.py:in_scope_dataset_types
            + collate_samples).
        """
        agents = []
        for env_name, env_cfg in environments.items():
            env_datasets = env_cfg.get("datasets") or {}
            dataset_entries: list[dict[str, Any]] = []
            if "train" in env_datasets:
                dataset_entries.append(
                    {
                        "name": "train",
                        "type": "train",
                        "license": "TBD",
                        "jsonl_fpath": env_datasets["train"],
                    }
                )
            # Accept either "val" or "validation" as the key for backward compat.
            val_fpath = env_datasets.get("val") or env_datasets.get("validation")
            if val_fpath:
                dataset_entries.append(
                    {
                        "name": "validation",
                        "type": "validation",
                        "license": "TBD",
                        "jsonl_fpath": val_fpath,
                    }
                )
            agents.append(
                {
                    "name": env_cfg["agent_name"],
                    "agent_type": env_cfg.get("agent_type", "simple_agent"),
                    "entrypoint": "app.py",
                    "resources_server": {
                        "type": "resources_servers",
                        "name": env_cfg.get("resources_server_name", env_name),
                    },
                    "model_server": {"type": "responses_api_models", "name": "policy_model"},
                    "datasets": dataset_entries,
                }
            )
        return agents

    @staticmethod
    def _config_paths_from_environments(environments: dict[str, Any]) -> list[str]:
        """Collect environment config_paths.

        No model config is needed here -- Gym's NO_MODEL_GLOBAL_CONFIG_DICT
        provides a dummy policy_model for data-only operations like
        ng_prepare_data."""
        config_paths: list[str] = []
        for env_cfg in environments.values():
            config_paths.extend(env_cfg.get("config_paths", []))
        return config_paths

    @staticmethod
    def _build_overlay(agents: list[dict[str, Any]]) -> dict:
        """Build a NeMo-Gym agent-config overlay from the agents list."""
        overlay: dict = {}
        for agent_cfg in agents:
            overlay[agent_cfg["name"]] = {
                "responses_api_agents": {
                    agent_cfg["agent_type"]: {
                        "entrypoint": agent_cfg["entrypoint"],
                        "resources_server": agent_cfg["resources_server"],
                        "model_server": agent_cfg["model_server"],
                        "datasets": agent_cfg["datasets"],
                    }
                }
            }
        return overlay

    def _overlay_shell_snippet(
        self, output_dir: str, agents: list[dict[str, Any]]
    ) -> tuple[str, str]:
        """Return a shell snippet that writes the overlay YAML at job runtime."""
        overlay = self._build_overlay(agents)

        agent_names = [a["name"] for a in agents]
        header = f"# Auto-generated by PrepareDataForGRPOStage.\n# Agents: {agent_names}\n"
        content = header + yaml.dump(overlay, default_flow_style=False, sort_keys=False)

        encoded = base64.b64encode(content.encode()).decode()
        overlay_path = f"{output_dir}/agent_config_overlay.yaml"
        snippet = f"mkdir -p {output_dir} && echo {encoded} | base64 -d > {overlay_path}"
        return snippet, overlay_path

    # -- main entry points ----------------------------------------------------

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Submit per-environment ng_prepare_data Slurm jobs."""
        from nvflow.lib.rl.helpers import resolve_environments

        environments = resolve_environments(config)
        base_output_dir = config["output_dir"]
        # input_dir points at the upstream stage that produced the per-env
        # file to be prepared (typically convert_to_responses_api).  Only a
        # single "train" dataset is fed to ng_prepare_data -- the post-rollout
        # train_validation_split stage produces the final val set after
        # collect_rollouts / compute_rewards, so we don't pre-split here.
        # ng_prepare_data tolerates a single-dataset agent config (see
        # Gym/nemo_gym/train_data_utils.py::in_scope_dataset_types and
        # collate_samples -- absent types are silently skipped).
        input_dir = config["input_dir"]
        input_filename = config.get("input_filename", "final_result.jsonl")

        for env_name, env_cfg in environments.items():
            if not env_cfg.get("raw_train_data"):
                console.warning(f"Skipping environment '{env_name}': no raw_train_data configured")
                continue
            env_output_dir = f"{base_output_dir}/{env_name}"
            env_datasets = {
                "train": f"{input_dir}/{env_name}/{input_filename}",
            }
            single_env = {env_name: {**env_cfg, "datasets": env_datasets}}

            self._submit_prepare_job(
                single_env=single_env,
                env_name=env_name,
                env_output_dir=env_output_dir,
                config=config,
                cluster=cluster,
                expname=f"{expname}-{env_name}",
                run_after=run_after,
            )

    def _submit_prepare_job(
        self,
        *,
        single_env: dict[str, Any],
        env_name: str,
        env_output_dir: str,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None,
    ) -> None:
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        gym_path = config["gym_path"]
        container = config["container"]
        installation_command = config.get("installation_command")
        mode = config.get("mode", "train_preparation")
        should_download = config.get("should_download", False)

        agents = self._agents_from_environments(single_env)
        config_paths = self._config_paths_from_environments(single_env)

        overlay_snippet, overlay_path = self._overlay_shell_snippet(env_output_dir, agents)
        config_paths.append(overlay_path)

        config_paths_str = ",".join(config_paths)
        cmd = (
            f"{overlay_snippet} && "
            f"cd {gym_path} && "
            f'ng_prepare_data "+config_paths=[{config_paths_str}]" '
            f"+output_dirpath={env_output_dir} "
            f"+mode={mode} "
            f"+error_on_almost_servers=false"
        )
        if should_download:
            cmd += " +should_download=true"
        extra_args = config.get("extra_args", "")
        if extra_args:
            cmd += f" {extra_args}"

        # Deterministic post-shuffle of train.jsonl.  ng_prepare_data preserves
        # SDG's per-filing clustering, which means collect_rollouts' contiguous
        # chunking (num_chunks > 1) and ``head -n max_num_samples`` truncation
        # see unbalanced question_type / date / company mixes.  Shuffling here
        # restores what the old pre-rollout train_validation_split used to
        # provide implicitly.  Seed-based + in-place; rerun-safe.
        shuffle = config.get("shuffle", True)
        random_seed = config.get("random_seed", 42)
        if shuffle:
            cmd += (
                f" && python -m nvflow.recipes.finance.utils.rl.shuffle_jsonl"
                f" --input_file {env_output_dir}/train.jsonl"
                f" --random_seed {random_seed}"
            )

        console.status(f"Preparing data for environment: {env_name}")
        console.detail("Mode", mode)
        for agent in agents:
            all_datasets = ", ".join(d["name"] for d in agent["datasets"])
            console.detail(f"Agent [{agent['agent_type']}]", agent["name"])
            console.detail("  Datasets", all_datasets)
        console.detail("Output", env_output_dir)
        console.detail("Post-shuffle", f"seed={random_seed}" if shuffle else "disabled")
        console.blank()

        run_cmd(
            ctx=wrap_arguments(cmd),
            cluster=cluster,
            container=container,
            num_gpus=config.get("num_gpus", 0),
            log_dir=f"{env_output_dir}/logs",
            expname=expname,
            run_after=run_after,
            installation_command=installation_command,
        )
        console.success(f"Data preparation job submitted → {env_output_dir}/")

    def validate_config(self, config: dict[str, Any]) -> None:
        """Check that all required fields are present."""
        for field in ("output_dir", "gym_path", "container", "input_dir"):
            if not config.get(field):
                raise ValueError(f"'{field}' is required in prepare_data config")

        environments = config.get("environments")
        if not isinstance(environments, dict) or not environments:
            raise ValueError("'environments' must be a non-empty dict in prepare_data config")

        for env_name, env_cfg in environments.items():
            if not env_cfg.get("config_paths"):
                raise ValueError(f"environments.{env_name}.config_paths is required")
            if not env_cfg.get("agent_name"):
                raise ValueError(
                    f"environments.{env_name}.agent_name is required. "
                    f"Set it to the agent Server ID from the Gym config YAML "
                    f"(the top-level key above 'responses_api_agents')."
                )

        mode = config.get("mode", "train_preparation")
        if mode not in ("train_preparation", "example_validation"):
            raise ValueError(
                f"'mode' must be 'train_preparation' or 'example_validation', got: {mode}"
            )
