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
"""One-shot script to dump current renderer outputs into the test fixtures dir.

Used to capture the initial baseline for the bash-renderer snapshot tests
in tests/test_rollout.py.  Re-run after any *intentional* change to the
renderers to refresh fixtures, then audit the diff before committing.

Usage:  uv run python3 scripts/_dump_rollout_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

from nvflow.lib.rl.rollout import (
    _build_client_cmd,
    _build_merge_cmd,
    build_aggregate_cmd,
    build_filter_cmd,
)

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures" / "rollout"
FIXTURES.mkdir(parents=True, exist_ok=True)


def _client_cmd(**overrides: object) -> str:
    """Render _build_client_cmd with a known-good baseline + overrides."""
    base: dict[str, object] = {
        "output_dir": "/out/rollout",
        "gym_path": "/opt/Gym",
        "model_path": "/hf_models/Qwen/Qwen3-30B-A3B",
        "agent_name": "finance_agent",
        "input_data": "/data/train.jsonl",
        "output_file": "/out/rollout/rs0/chunk_0.jsonl",
        "done_file": "/out/rollout/rs0/chunk_0.jsonl.done",
        "config_paths": "vllm.yaml,env.yaml,overlay.yaml",
        "num_parallel": 512,
        "job_label": "rs0_chunk0",
        "policy_vllm_url": "http://policy:8000/v1",
        "judge_vllm_url": "http://judge:8001/v1",
        "judge_ng_run_overrides": (
            '    "+judge_model.responses_api_models.vllm_model.entrypoint=app.py" \\\n'
            '    "+judge_model.responses_api_models.vllm_model.base_url=http://judge:8001/v1" \\\n'
        ),
        "max_num_samples": 0,
        "chunk_id": 0,
        "num_chunks": 8,
        "responses_create_params": {"max_output_tokens": 32768, "temperature": 1.0},
    }
    base.update(overrides)
    return _build_client_cmd(**base)  # type: ignore[arg-type]


# 1. Dual-server Qwen3-style: policy URL + judge URL + multi-chunk
(FIXTURES / "client_cmd_dual_server.txt").write_text(_client_cmd())

# 2. Policy-only (policy_as_judge or no judge): empty judge URL + overrides
(FIXTURES / "client_cmd_policy_only.txt").write_text(
    _client_cmd(judge_vllm_url="", judge_ng_run_overrides="")
)

# 3. Single-chunk path: chunk slicing branch must NOT emit
(FIXTURES / "client_cmd_no_chunk.txt").write_text(_client_cmd(num_chunks=1))

# 4. With max_num_samples cap (truncated input)
(FIXTURES / "client_cmd_max_samples.txt").write_text(_client_cmd(max_num_samples=10000))

# 5. Empty responses_create_params: no extra +responses_create_params.* lines
(FIXTURES / "client_cmd_no_rcp.txt").write_text(_client_cmd(responses_create_params={}))


# --- _build_merge_cmd ---
def _merge_cmd(**overrides: object) -> str:
    base: dict[str, object] = {
        "gym_path": "/opt/Gym",
        "merged_file": "/out/rollout/output-rs0.jsonl",
        "analysis_dir": "/out/rollout/analysis_rs0",
        "seed_label": "rs0",
        "num_chunks": 8,
        "chunk_file_pattern": "/out/rollout/rs0/chunk_$i.jsonl",
        "merged_done_file": "/out/rollout/output-rs0.jsonl.done",
        "analyze_module": "nvflow.recipes.finance.utils.rl.analyze_rollouts",
        "enrich_module": "nvflow.recipes.finance.utils.rl.enrich_rollouts",
        "input_data": "/data/train.jsonl",
    }
    base.update(overrides)
    return _build_merge_cmd(**base)  # type: ignore[arg-type]


(FIXTURES / "merge_cmd_8chunks.txt").write_text(_merge_cmd())
(FIXTURES / "merge_cmd_1chunk.txt").write_text(_merge_cmd(num_chunks=1))


# --- build_aggregate_cmd ---
(FIXTURES / "aggregate_cmd_default.txt").write_text(
    build_aggregate_cmd(
        rollout_dir="/out/rollout",
        aggregate_module="nvflow.recipes.finance.utils.rl.aggregate_seeds",
    )
)
(FIXTURES / "aggregate_cmd_custom_filename.txt").write_text(
    build_aggregate_cmd(
        rollout_dir="/out/rollout",
        aggregate_module="nvflow.recipes.finance.utils.rl.aggregate_seeds",
        difficulty_filename="custom_difficulty.jsonl",
    )
)


# --- build_filter_cmd ---
(FIXTURES / "filter_cmd_minimal.txt").write_text(
    build_filter_cmd(
        output_dir="/out",
        difficulty_dir="/out/rollout",
        filter_module="nvflow.recipes.finance.utils.rl.filter_training_data",
        train_data="/in/train.jsonl",
        validation_data="",
    )
)
(FIXTURES / "filter_cmd_full.txt").write_text(
    build_filter_cmd(
        output_dir="/out",
        difficulty_dir="/out/rollout",
        filter_module="nvflow.recipes.finance.utils.rl.filter_training_data",
        train_data="/in/train.jsonl",
        validation_data="/in/val.jsonl",
        min_reward_std=1e-6,
        policy_model="/hf_models/Qwen/Qwen3-30B-A3B",
        judge_model="/hf_models/openai/gpt-oss-120b",
        train_filename="train.jsonl",
        val_filename="validation.jsonl",
        difficulty_filename="difficulty.jsonl",
        report_filename="filter_report.json",
    )
)


print("Wrote fixtures:")
for p in sorted(FIXTURES.iterdir()):
    print(f"  {p.name}: {p.stat().st_size} bytes")
