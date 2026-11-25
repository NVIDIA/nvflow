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
"""VALS 1.1 question pool: agent-template parity with Gym and the check prompts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from nvflow import load_workflow
from nvflow.execution import TargetBinding, WorkflowPlanner
from nvflow.integrations.nemo_gym.responses import load_prompt_template, render_prompt

PROJECT = Path(__file__).parents[1]
WORKFLOW = yaml.safe_load((PROJECT / "workflows/sdg/vals11-prepare.yaml").read_text())
GYM_ROWS = json.loads((PROJECT / "tests/data/fabv1_gym_rows.json").read_text())
TEMPLATES = {
    "web": PROJECT / "prompts/finance_sec_search_template_edgar_search_with_web.yaml",
    "noweb": PROJECT / "prompts/finance_sec_search_template_edgar_search_without_web.yaml",
}
# Known differences from Gym's fabv1 prompt, still to be decided: one wording
# change, and our templates end the prompt with a newline.
KNOWN_DIFFERENCE = ("to apply answer questions", "to answer questions")
# Our agent searches the local SEC index instead of fabv1's live SEC.gov search.
SEARCH_TOOL = {"sec_filing_search": "edgar_search"}


def test_pools_render_with_the_edgar_search_agent_prompts() -> None:
    renders = WORKFLOW["stages"]["build_pools"]["substages"]["render"]["renders"]

    assert {name: render["prompt"] for name, render in renders.items()} == {
        name: f"project://prompts/{TEMPLATES['web' if name.endswith('web_on') else 'noweb'].name}"
        for name in ("web_off", "web_on", "held_out_web_off", "held_out_web_on")
    }


@pytest.mark.parametrize("variant", ["web", "noweb"])
def test_agent_templates_match_gym_fabv1_rows_but_the_search_tool(variant: str) -> None:
    gym = GYM_ROWS["rows"][variant]["responses_create_params"]
    template = yaml.safe_load(TEMPLATES[variant].read_text())
    [message] = gym["input"]
    date = re.search(r"current date is ([^.\n]+)", message["content"])
    assert date is not None

    ours = template["user"].format(current_date=date.group(1), problem=GYM_ROWS["question"])

    assert [tool["name"] for tool in template["tools"]] == [
        SEARCH_TOOL.get(tool["name"], tool["name"]) for tool in gym["tools"]
    ]
    assert [tool for tool in template["tools"] if tool["name"] not in SEARCH_TOOL.values()] == [
        tool for tool in gym["tools"] if tool["name"] not in SEARCH_TOOL
    ]
    assert template["parallel_tool_calls"] is True
    assert ours.endswith("\n") and not message["content"].endswith("\n")
    assert ours.replace(*KNOWN_DIFFERENCE).removesuffix("\n") == message["content"]


def _gym_string_match(text: str, verifier: dict[str, Any]) -> float:
    """NeMo Gym v0.7.0 format_verification's string_match rule."""
    expected = verifier["expected_markers"]
    missing = [marker for marker in expected if marker not in text]
    spurious = [
        found.group(0)
        for pattern in verifier.get("patterns", [])
        for found in re.finditer(pattern, text)
        if found.group(0) not in expected
    ]
    return 1.0 if not missing and not spurious else 0.0


@pytest.mark.parametrize(
    ("reply", "reward"),
    [
        ("Reason: all checks pass\nVerdict: PASS", 1.0),
        ("Reason: 3. the question names two different fiscal years\nVerdict: FAIL", 0.0),
        ("Verdict: PASS\nVerdict: FAIL", 0.0),
        ("Reason: unsure\nVerdict: <PASS|FAIL>", 0.0),
        ("The answer looks right.", 0.0),
    ],
)
def test_expert_check_rewards_only_a_lone_pass(reply: str, reward: float) -> None:
    assert _gym_string_match(reply, WORKFLOW["expert_verifier"]) == reward


@pytest.mark.parametrize(
    ("reply", "reward"),
    [
        ("Reason: different fiscal year\nVerdict: DIFFERENT", 1.0),
        ("Reason: same company, period, and metric\nVerdict: SAME", 0.0),
        ("Reason: unclear\nVerdict: <SAME|DIFFERENT>", 0.0),
    ],
)
def test_contamination_judge_keeps_only_different_questions(reply: str, reward: float) -> None:
    assert _gym_string_match(reply, WORKFLOW["judge_verifier"]) == reward


@pytest.mark.parametrize(
    ("prompt", "verifier"),
    [
        ("contamination_judge.yaml", "judge_verifier"),
        ("question_answer_expert_check.yaml", "expert_verifier"),
    ],
)
def test_check_prompts_ask_for_the_verdicts_their_verifiers_match(
    prompt: str, verifier: str
) -> None:
    _system, user = load_prompt_template(PROJECT / "prompts" / prompt)
    [choices] = re.findall(r"Verdict: <([^>]+)>", user)

    assert WORKFLOW[verifier]["patterns"] == [f"Verdict: ({choices})"]


def test_pools_keep_the_fields_gym_and_the_templates_need() -> None:
    pools = WORKFLOW["stages"]["build_pools"]["substages"]
    fields = set(pools["split"]["fields"])
    for render in pools["render"]["renders"].values():
        path = PROJECT / render["prompt"].removeprefix("project://")
        text = "\n".join(part or "" for part in load_prompt_template(path))
        assert set(re.findall(r"(?<!\{)\{(\w+)\}(?!\})", text)) <= fields, path.name
    assert {"uuid", "question", "expected_answer"} <= fields


@pytest.mark.parametrize(
    ("question_set", "splits"),
    [
        ("sft", {"held_out": 0.05, "web_off": 0.475, "web_on": 0.475}),
        ("rl", {"held_out": 0.05, "web_off": 0.85, "web_on": 0.10}),
    ],
)
def test_each_set_splits_its_questions_in_its_own_shares(
    question_set: str, splits: dict[str, float]
) -> None:
    definition = load_workflow(PROJECT / f"workflows/sdg/vals11-prepare-{question_set}.yaml")
    split = definition.config["stages"]["build_pools"]["substages"]["split"]

    assert dict(split["splits"]) == splits
    assert list(split["splits"]) == ["held_out", "web_off", "web_on"]


def test_check_prompts_render_from_pool_and_candidate_fields() -> None:
    checks = WORKFLOW["stages"]["verify_questions"]["substages"]
    expert = checks["expert_requests"]["renders"]["pool"]
    judge = checks["judge_requests"]["renders"]["candidates"]
    assert expert["prompt"] == "project://prompts/question_answer_expert_check.yaml"
    assert judge["prompt"] == "project://prompts/contamination_judge.yaml"

    _system, user = load_prompt_template(PROJECT / "prompts/question_answer_expert_check.yaml")
    rendered = render_prompt({"problem": "What was X?", "expected_answer": "7"}, user)
    assert "What was X?" in rendered and "7" in rendered

    _system, user = load_prompt_template(PROJECT / "prompts/contamination_judge.yaml")
    rendered = render_prompt({"text": "train question", "reference_text": "eval question"}, user)
    assert "train question" in rendered and "eval question" in rendered


@pytest.fixture
def planning_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    containers = {name: f"{name}-image" for name in ("nemo-gym", "nemo-skills", "vllm")}
    cluster = {
        "executor": "none",
        "gpus_per_node": 8,
        "cpu_partition": "cpu",
        "containers": containers,
    }
    monkeypatch.setattr(
        "nemo_skills.pipeline.utils.cluster.get_cluster_config",
        lambda *_args, **_kwargs: cluster,
    )


@pytest.mark.requires_nemo
@pytest.mark.usefixtures("planning_cluster")
@pytest.mark.parametrize(("question_set", "chunk_jobs"), [("sft", 0), ("rl", 3 + 1)])
def test_only_the_rl_expert_check_runs_in_chunks(question_set: str, chunk_jobs: int) -> None:
    from nvflow.execution.engines.slurm import _grouped_task, _job_groups

    definition = load_workflow(PROJECT / f"workflows/sdg/vals11-prepare-{question_set}.yaml")
    plan = WorkflowPlanner(
        definition, TargetBinding.resolve(definition.cluster, "planning-cluster")
    ).plan()
    jobs = [_grouped_task(group) for group in _job_groups(plan.tasks)]

    assert [
        (job.stage, job.commands[0].container, job.resources[0].gpus_per_node) for job in jobs
    ] == [
        ("merge_questions", "nemo-skills", 0),
        ("merge_questions", "vllm", 1),
        ("verify_questions", "nemo-gym", 0),
        *[("verify_questions", "nemo-gym", 0)] * chunk_jobs,
        ("build_pools", "nemo-gym", 0),
    ]
    assert f"/workflow-1-prepare/{question_set}/" in plan.tasks[0].commands[0].command
