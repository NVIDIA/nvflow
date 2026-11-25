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
"""nemo_gym.render: tool stamping, Gym rendering, resume, and plans."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from nvflow.execution import PlanningContext, TargetBinding
from nvflow.integrations.nemo_gym.workers.render_prompts import main
from nvflow.stages.nemo_gym.render_prompts import RenderPromptsStage

TEMPLATE = """\
user: |
  The current date is {current_date}.
  Question:
  {problem}
tools:
  - {type: function, name: lookup}
  - {type: function, name: submit_final_result}
parallel_tool_calls: true
"""

# Stand-in for `gym dataset render`: Gym fills user/system from row fields,
# rejects rows that already have an input, and keeps other request fields.
FAKE_GYM = """\
import json, sys, yaml
args = dict(arg.lstrip("+").split("=", 1) for arg in sys.argv[3:])
template = yaml.safe_load(open(args["prompt_config"]))
with open(args["input_jsonl_fpath"]) as reader, open(args["output_jsonl_fpath"], "w") as writer:
    for line in reader:
        row = json.loads(line)
        assert not row["responses_create_params"].get("input"), "input is already rendered"
        messages = [{"role": "user", "content": template["user"].format_map(row)}]
        row["responses_create_params"]["input"] = messages
        row["prompt_config_used"] = args["prompt_config"]
        writer.write(json.dumps(row) + "\\n")
"""


@pytest.fixture
def gym(tmp_path: Path) -> str:
    script = tmp_path / "bin" / "gym"
    script.parent.mkdir()
    script.write_text(f"#!{sys.executable}\n{FAKE_GYM}", encoding="utf-8")
    script.chmod(0o755)
    return str(script)


def _write(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def test_render_stamps_tools_then_fills_the_prompt(tmp_path: Path, gym: str) -> None:
    source = _write(
        tmp_path / "pool.jsonl",
        [
            {
                "uuid": "a",
                "problem": "What was the rainfall?",
                "current_date": "March 1, 2025",
                "responses_create_params": {
                    "input": [{"role": "user", "content": "old"}],
                    "tools": [],
                },
            },
            {"uuid": "b", "problem": "How many stores?", "current_date": "May 2, 2024"},
        ],
    )
    template = tmp_path / "template.yaml"
    template.write_text(TEMPLATE, encoding="utf-8")
    output = tmp_path / "out" / "web_off.jsonl"

    assert (
        main(
            [
                "--input",
                str(source),
                "--prompt",
                str(template),
                "--output",
                str(output),
                "--gym",
                gym,
            ]
        )
        == 0
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["uuid"] for row in rows] == ["a", "b"]
    request = rows[0]["responses_create_params"]
    assert request["input"] == [
        {
            "role": "user",
            "content": "The current date is March 1, 2025.\nQuestion:\nWhat was the rainfall?\n",
        }
    ]
    assert [tool["name"] for tool in request["tools"]] == [
        "lookup",
        "submit_final_result",
    ]
    assert request["parallel_tool_calls"] is True
    assert output.with_suffix(".done").exists()

    finished = output.stat().st_mtime_ns
    assert (
        main(
            [
                "--input",
                str(source),
                "--prompt",
                str(template),
                "--output",
                str(output),
                "--gym",
                gym,
            ]
        )
        == 0
    )
    assert output.stat().st_mtime_ns == finished


def test_row_fields_replace_the_rows_own_and_change_the_render(tmp_path: Path, gym: str) -> None:
    pool_row = {"problem": "q", "current_date": "d", "verifier": {"type": "x"}, "_ng_task_index": 7}
    source = _write(tmp_path / "pool.jsonl", [pool_row])
    template = tmp_path / "template.yaml"
    template.write_text(TEMPLATE, encoding="utf-8")
    output = tmp_path / "out" / "checks.jsonl"
    base = ["--input", str(source), "--prompt", str(template), "--output", str(output)]
    verifier = {"type": "string_match", "expected_markers": ["Verdict: PASS"]}

    assert main([*base, "--row_fields", json.dumps({"verifier": verifier}), "--gym", gym]) == 0
    [row] = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert row["verifier"] == verifier
    assert "_ng_task_index" not in row

    changed = {"verifier": {**verifier, "expected_markers": ["Verdict: YES"]}}
    assert main([*base, "--row_fields", json.dumps(changed), "--gym", gym]) == 0
    [row] = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert row["verifier"]["expected_markers"] == ["Verdict: YES"]


def test_render_requires_the_gym_cli(tmp_path: Path) -> None:
    source = _write(tmp_path / "pool.jsonl", [{"problem": "q", "current_date": "d"}])
    template = tmp_path / "template.yaml"
    template.write_text(TEMPLATE, encoding="utf-8")

    with pytest.raises(RuntimeError, match="not on PATH"):
        main(
            [
                "--input",
                str(source),
                "--prompt",
                str(template),
                "--output",
                str(tmp_path / "out.jsonl"),
                "--gym",
                str(tmp_path / "missing-gym"),
            ]
        )


def _context(config: dict[str, Any], *, present: set[str] | None = None) -> PlanningContext:
    class Filesystem:
        def batch_exists(self, paths: list[str]) -> dict[str, bool]:
            return {path: path in (present or set()) for path in paths}

    return PlanningContext(
        stage="render",
        config=config,
        launcher_config=config,
        target=TargetBinding.resolve("test"),
        expname="render",
        run_after=("split",),
        live=present is not None,
        filesystem=None if present is None else Filesystem(),  # type: ignore[arg-type]
        engine=None if present is None else "slurm",
    )


CONFIG = {
    "contract_version": 1,
    "output_dir": "/out/rendered",
    "installation_command": "export PATH=/opt/gym/bin:$PATH",
    "renders": {
        "web_off": {"input_file": "/split/web_off.jsonl", "prompt": "/prompts/off.yaml"},
        "web_on": {"input_file": "/split/web_on.jsonl", "prompt": "/prompts/on.yaml"},
    },
}


def test_plan_runs_every_render_in_one_gym_cpu_task() -> None:
    RenderPromptsStage().validate_config(CONFIG)
    [task] = RenderPromptsStage().plan(_context(CONFIG)).tasks

    command = task.commands[0].command
    assert task.commands[0].container == "nemo-gym"
    assert task.resources[0].gpus_per_node == 0
    assert command.startswith("set -e\nexport PATH=/opt/gym/bin:$PATH\n")
    assert command.count("workers.render_prompts") == 2
    assert {output.path for output in task.outputs} >= {
        "/out/rendered/web_off.jsonl",
        "/out/rendered/web_on.done",
    }


def test_live_plan_leaves_finished_renders_for_the_worker_to_check() -> None:
    present = {"/out/rendered/web_off.done", "/out/rendered/web_on.done"}

    [task] = RenderPromptsStage().plan(_context(CONFIG, present=present)).tasks
    assert task.commands[0].command.count("workers.render_prompts") == 2


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"contract_version": 1, "output_dir": "/o"}, "renders"),
        ({**CONFIG, "renders": {"a/b": {"input_file": "i", "prompt": "p"}}}, "invalid"),
        ({**CONFIG, "renders": {"a": {"input_file": "i"}}}, "prompt"),
        (
            {**CONFIG, "renders": {"a": {"input_file": "i", "prompt": "p", "row_field": {}}}},
            "unknown",
        ),
        ({**CONFIG, "contract_version": 2}, "contract_version"),
        ({key: value for key, value in CONFIG.items() if key != "contract_version"}, "contract"),
    ],
)
def test_stage_config_fails_closed(config: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RenderPromptsStage().validate_config(config)
