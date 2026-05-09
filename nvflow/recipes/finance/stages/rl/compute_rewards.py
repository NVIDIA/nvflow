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
"""Compute rewards stage (finance recipe).

Thin wrapper around :func:`nvflow.lib.rl.verify` that registers the
stage with the finance/grpo workflow and supplies recipe-specific
utility module names.
"""

from typing import Any

from nvflow.core import BaseStage, StageRegistry

_UTILS = "nvflow.recipes.finance.utils.rl"


@StageRegistry.register(recipe="finance", workflow="grpo", stage="compute_rewards")
class ComputeRewardsStage(BaseStage):
    """Re-compute rewards on existing rollouts using a different judge.

    Delegates all orchestration to :func:`nvflow.lib.rl.verify`.
    This stage only handles registration, config validation, and
    passing recipe-specific module names.
    """

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        from nvflow.lib.rl.helpers import resolve_environments
        from nvflow.lib.rl.verify import verify

        environments = resolve_environments(config)
        base_output_dir = config["output_dir"]
        rollouts_dir = config["rollouts_dir"]
        prepare_data_dir = config["prepare_data_dir"]

        for env_name, env_cfg in environments.items():
            env_output_dir = f"{base_output_dir}/{env_name}"
            env_rollouts_dir = f"{rollouts_dir}/{env_name}"
            env_prepare_dir = f"{prepare_data_dir}/{env_name}"

            env_config = {
                **config,
                "output_dir": env_output_dir,
                "environments": {env_name: env_cfg},
            }
            env_judge_vllm = env_cfg.get("judge_vllm") or {}
            env_config["rejudge"] = {
                **config.get("rejudge", {}),
                "input_dir": f"{env_rollouts_dir}/rollout",
                "prepare_data_dir": env_prepare_dir,
                "judge_vllm": env_judge_vllm,
            }
            env_config["filter"] = {
                **config.get("filter", {}),
                "input_data": f"{env_prepare_dir}/train.jsonl",
                # prepare_data emits a single train.jsonl; the post-rollout
                # train_validation_split stage produces the final val set.
                # filter_training_data treats validation_path as optional.
            }

            verify(
                env_config,
                cluster,
                f"{expname}-{env_name}",
                run_after,
                analyze_module=f"{_UTILS}.analyze_rollouts",
                aggregate_module=f"{_UTILS}.aggregate_seeds",
                filter_module=f"{_UTILS}.filter_training_data",
            )

    def validate_config(self, config: dict[str, Any]) -> None:
        from nvflow.lib.rl.helpers import (
            determine_judge_mode,
            resolve_environments,
            validate_judge_config,
        )

        for field in ("output_dir", "gym_path", "container", "rollouts_dir", "prepare_data_dir"):
            if not config.get(field):
                raise ValueError(f"'{field}' is required in compute_rewards config")

        if not config.get("environments"):
            raise ValueError("'environments' dict is required in compute_rewards config")

        environments = resolve_environments(config)
        rcfg = config.get("rejudge") or {}
        for _env_name, env_cfg in environments.items():
            env_judge = {**rcfg, "judge_vllm": env_cfg.get("judge_vllm") or {}}
            determine_judge_mode(env_judge, allow_policy_as_judge=False)
            validate_judge_config(env_judge)
