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
"""One-shot script to dump current verify-renderer outputs into fixtures.

Used to capture the baseline for the bash-renderer snapshot tests in
``tests/test_verify.py``.  Re-run after any *intentional* change to
``_build_verify_cmd`` or ``_build_analysis_cmd``, then audit the diff
under ``tests/fixtures/verify/`` carefully before committing.

Usage:  uv run python3 scripts/_dump_verify_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

from nvflow.lib.rl.verify import _build_analysis_cmd, _build_verify_cmd

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures" / "verify"
FIXTURES.mkdir(parents=True, exist_ok=True)


# Override blocks mirror what build_judge_ng_run_overrides() returns
# for each judge_mode -- keep them in sync so the snapshots reflect
# realistic call-site shapes rather than a stripped placeholder.

_LOCAL_VLLM_OVERRIDES = (
    '    "+judge_model.responses_api_models.vllm_model.entrypoint=app.py" \\\n'
    '    "+judge_model.responses_api_models.vllm_model.base_url=http://127.0.0.1:$JUDGE_PORT/v1" \\\n'
    '    "+judge_model.responses_api_models.vllm_model.api_key=EMPTY" \\\n'
    '    "+judge_model.responses_api_models.vllm_model.model=/hf_models/openai/gpt-oss-120b" \\\n'
    '    "+judge_model.responses_api_models.vllm_model.return_token_id_information=false" \\\n'
    '    "+judge_model.responses_api_models.vllm_model.uses_reasoning_parser=true" \\\n'
    '    "+finance_env.resources_servers.finance_env.judge_model_server.name=judge_model" \\\n'
)

_OPENAI_OVERRIDES = (
    '    "+judge_model.responses_api_models.openai_model.base_url=https://api.openai.com/v1" \\\n'
    '    "+judge_model.responses_api_models.openai_model.api_key_env_var=OPENAI_API_KEY" \\\n'
    '    "+judge_model.responses_api_models.openai_model.model=gpt-4o-mini" \\\n'
    '    "+finance_env.resources_servers.finance_env.judge_model_server.name=judge_model" \\\n'
)


def _verify_cmd(**overrides: object) -> str:
    """Render _build_verify_cmd with a known-good baseline + overrides."""
    base: dict[str, object] = {
        "output_dir": "/out/verify",
        "gym_path": "/opt/Gym",
        "input_file": "/in/rollouts/output-rs0.jsonl",
        "output_file": "/out/verify/rejudge/output-rs0.jsonl",
        "done_file": "/out/verify/rejudge/output-rs0.jsonl.done",
        "config_paths": "vllm.yaml,env.yaml,overlay.yaml",
        "num_parallel": 8,
        "job_label": "rejudge_rs0",
        "judge_mode": "local_vllm",
        "environment_name": "finance_env",
        "judge_ng_run_overrides": _LOCAL_VLLM_OVERRIDES,
    }
    base.update(overrides)
    return _build_verify_cmd(**base)  # type: ignore[arg-type]


# 1. Local-vLLM judge: full overrides w/ uses_reasoning_parser
(FIXTURES / "verify_cmd_local_judge.txt").write_text(_verify_cmd())

# 2. OpenAI-API judge: shape of overrides differs (different keys)
(FIXTURES / "verify_cmd_openai_judge.txt").write_text(
    _verify_cmd(judge_mode="openai", judge_ng_run_overrides=_OPENAI_OVERRIDES)
)


# --- _build_analysis_cmd ---
def _analysis_cmd(**overrides: object) -> str:
    base: dict[str, object] = {
        "rejudge_dir": "/out/verify/rejudge",
        "gym_path": "/opt/Gym",
        "analyze_module": "nvflow.recipes.finance.utils.rl.analyze_rollouts",
        "analysis_entries": [
            ("rs0", "/out/verify/rejudge/output-rs0.jsonl"),
            ("rs1", "/out/verify/rejudge/output-rs1.jsonl"),
        ],
    }
    base.update(overrides)
    return _build_analysis_cmd(**base)  # type: ignore[arg-type]


(FIXTURES / "analysis_cmd_multi_seed.txt").write_text(_analysis_cmd())
(FIXTURES / "analysis_cmd_single_seed.txt").write_text(
    _analysis_cmd(analysis_entries=[("rs0", "/out/verify/rejudge/output-rs0.jsonl")])
)
(FIXTURES / "analysis_cmd_empty_entries.txt").write_text(_analysis_cmd(analysis_entries=[]))


print("Wrote fixtures:")
for p in sorted(FIXTURES.iterdir()):
    print(f"  {p.name}: {p.stat().st_size} bytes")
