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

"""Regression tests for finance checkpoint paths."""

from pathlib import Path

import pytest

from nvflow import load_workflow
from nvflow.execution import WorkflowRunner
from nvflow.integrations.nemo_gym.environments import resolve_environments
from nvflow.stages.nemo_rl.grpo import GRPOStage

REPO_ROOT = Path(__file__).resolve().parents[1]
NEMOTRON_GRPO_DIR = REPO_ROOT / "projects/finance/workflows/grpo"
NEMOTRON_CONFIGS = ["nemotron_nano_30b_a3b.yaml"]


@pytest.mark.parametrize("config_name", NEMOTRON_CONFIGS)
def test_nemotron_eval_path_matches_generated_training_run_name(config_name: str) -> None:
    config = WorkflowRunner(str(NEMOTRON_GRPO_DIR / config_name)).config
    training = config["stages"]["training"]
    policy = training["overrides"]["policy"]
    megatron = policy["megatron_cfg"]

    model_short = Path(training["model_name"]).name.lower().replace("_", "-")
    expected_run_name = (
        f"grpo-{model_short}-{training['total_gpus']}g-"
        f"tp{megatron['tensor_model_parallel_size']}-"
        f"cp{megatron['context_parallel_size']}-"
        f"seq{policy['max_total_sequence_length'] // 1024}k"
    )
    checkpoint_path = Path(config["stages"]["eval"]["checkpoint_path"])

    assert checkpoint_path.parent.name == "finance_sec_search"
    assert checkpoint_path.name == expected_run_name


GRPO_WORKFLOWS = sorted(
    path.name
    for path in NEMOTRON_GRPO_DIR.glob("*.yaml")
    if path.name not in {"base.yaml", "grpo_presets.yaml"}
)


@pytest.mark.parametrize("workflow", GRPO_WORKFLOWS)
def test_every_eval_step_is_a_checkpoint_training_keeps(workflow: str) -> None:
    definition = load_workflow(NEMOTRON_GRPO_DIR / workflow)
    if "eval" not in definition.pipeline_stages:
        pytest.skip("workflow does not run evaluation")
    stages = definition.config["stages"]
    training = definition.project_context.render_launcher_paths(dict(stages["training"]))
    presets = GRPOStage._load_presets(training)
    checked = 0
    for selection in [None, *resolve_environments(training)]:
        environment = training["environments"][selection] if selection else {}
        if selection and not environment.get("raw_train_data"):
            continue
        selected = {**training, "_environment": selection} if selection else training
        eval_steps = (environment.get("eval") or {}).get("eval_steps", stages["eval"]["eval_steps"])
        for _label, run, _expname in GRPOStage._resolve_run_configs(selected, "check"):
            merged = GRPOStage()._resolve_nemo_rl_config(run, presets)
            save_period = merged["checkpointing"]["save_period"]
            keep_top_k = merged["checkpointing"].get("keep_top_k")
            max_num_steps = merged["grpo"]["max_num_steps"]
            # NeMo-RL keeps the newest checkpoint and, with keep_top_k, only the
            # best-ranked save_period checkpoints. An eval step at max_num_steps
            # also assumes the data lasts that long: max_num_epochs can end
            # training first.
            pruned = keep_top_k is not None and max_num_steps // save_period > keep_top_k
            missing = [
                step
                for step in eval_steps
                if step > max_num_steps
                or (step != max_num_steps and (step % save_period or pruned))
            ]
            assert not missing, (
                f"{workflow} -e {selection or '<all>'}: training does not keep eval steps "
                f"{missing} (save_period={save_period}, keep_top_k={keep_top_k}, "
                f"max_num_steps={max_num_steps})"
            )
            checked += 1
    assert checked
