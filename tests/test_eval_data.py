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
"""Tests for the Gym benchmark preparation stage.

The stage emits shell that runs unattended inside a Slurm job, so the tests
cover the two decisions that shape it and are invisible until a job fails:

  - QA benchmarks render through a prompt template while agentic ones are
    copied verbatim, because their prompts and tool schemas are already baked in.
  - ``overrides`` reach the Gym CLI unquoted, which is what lets a gated dataset
    read a credential from the job's environment instead of from a checked-in
    file.

The skip-when-present guard matters just as much: re-running the stage must not
re-download a gated dataset.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from nvflow.stages.nemo_gym.prepare_benchmark import (
    _benchmark_command,
    _resolve_entry,
)

# ---------------------------------------------------------------------------
# _resolve_entry
# ---------------------------------------------------------------------------


def test_bare_string_entry_expands_to_gyms_layout():
    assert asdict(_resolve_entry("secque")) == {
        "benchmark": "secque",
        "prepared_file": "benchmarks/secque/data/secque_benchmark.jsonl",
        "prompt_config": "benchmarks/secque/prompts/default.yaml",
        "output_file": "secque/gym_eval.jsonl",
        "overrides": (),
    }


def test_config_scoped_selector_seeds_defaults_from_the_benchmark_dir():
    """`<name>/<config>` selects a variant, but the files still live under <name>."""
    got = _resolve_entry({"benchmark": "finance_sec_search/config_no_web_search"})
    assert got.prepared_file == (
        "benchmarks/finance_sec_search/data/finance_sec_search_benchmark.jsonl"
    )
    assert got.output_file == "finance_sec_search/gym_eval.jsonl"
    assert got.benchmark == "finance_sec_search/config_no_web_search"


def test_explicit_fields_win_over_the_defaults():
    got = _resolve_entry(
        {
            "benchmark": "finance_agent_v2",
            "prepared_file": "benchmarks/finance_agent_v2/data/vals_v2_public_27q.jsonl",
            "prompt_config": None,
            "output_file": "fabv2/gym_eval.jsonl",
        }
    )
    assert got.prepared_file.endswith("vals_v2_public_27q.jsonl")
    assert got.prompt_config is None
    assert got.output_file == "fabv2/gym_eval.jsonl"


def test_null_overrides_normalise_to_a_list():
    assert _resolve_entry({"benchmark": "secque", "overrides": None}).overrides == ()


@pytest.mark.parametrize("entry", [{}, {"benchmark": ""}, {"prepared_file": "x.jsonl"}])
def test_entry_without_a_selector_is_rejected(entry):
    with pytest.raises(ValueError, match="requires a selector"):
        _resolve_entry(entry)


# ---------------------------------------------------------------------------
# _benchmark_command
# ---------------------------------------------------------------------------


def test_qa_benchmark_renders_through_its_prompt_template():
    cmd = _benchmark_command(_resolve_entry("secque"), "/data")
    assert "gym dataset render" in cmd
    assert "+prompt_config=benchmarks/secque/prompts/default.yaml" in cmd
    assert "+output_jsonl_fpath=/data/secque/gym_eval.jsonl" in cmd
    assert "cp " not in cmd


def test_agentic_benchmark_is_copied_verbatim():
    cmd = _benchmark_command(
        _resolve_entry(
            {
                "benchmark": "finance_agent_v2",
                "prepared_file": "benchmarks/finance_agent_v2/data/vals_v2_public_27q.jsonl",
                "prompt_config": None,
                "output_file": "fabv2/gym_eval.jsonl",
            }
        ),
        "/data",
    )
    assert "gym dataset render" not in cmd
    assert (
        "cp benchmarks/finance_agent_v2/data/vals_v2_public_27q.jsonl /data/fabv2/gym_eval.jsonl"
        in cmd
    )


def test_existing_output_short_circuits_the_download():
    cmd = _benchmark_command(_resolve_entry("secque"), "/data")
    assert cmd.startswith("if [ -s /data/secque/gym_eval.jsonl ]; then")
    assert "Using prepared data: /data/secque/gym_eval.jsonl" in cmd


def test_destination_directory_is_created_before_the_file_is_placed():
    cmd = _benchmark_command(_resolve_entry("secque"), "/data")
    assert cmd.index("mkdir -p /data/secque") < cmd.index("gym dataset render")


def test_overrides_stay_unquoted_so_the_job_shell_expands_them():
    """Quoting here would ship the literal $VAR to Gym instead of the secret."""
    cmd = _benchmark_command(
        _resolve_entry(
            {
                "benchmark": "finance_sec_search_200",
                "overrides": ["+mlflow_tracking_uri=\"'$MLFLOW_TRACKING_URI'\""],
            }
        ),
        "/data",
    )
    assert "--benchmark finance_sec_search_200 +mlflow_tracking_uri=\"'$MLFLOW_TRACKING_URI'\"" in (
        cmd
    )


def test_prepare_reuses_a_cached_download():
    cmd = _benchmark_command(_resolve_entry("secque"), "/data")
    assert "+use_cached_prepared_benchmarks=true" in cmd
