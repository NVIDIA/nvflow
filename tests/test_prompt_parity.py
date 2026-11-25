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
"""SECQUE is prompted identically by training and by evaluation, and the QA
judge is prompted the way its rubric expects.

Two copies of the question prompt exist because the training data calls the
question field ``problem`` while the prepared benchmark record calls it
``question``.  Nothing at runtime compares them, and a divergence produces no
error: both pipelines still succeed, and the scores simply stop being
comparable.

The judge checks guard the same class of silent failure.  Folding the rubric
into the user turn instead of the system turn grades every QA answer slightly
more leniently, with no warning anywhere.
"""

from pathlib import Path

import yaml

from nvflow.project import ProjectResolver

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "projects/finance/workflows/eval"
EVAL_BASE = EVAL_DIR / "base.yaml"
TRAIN_PROMPT = REPO_ROOT / "projects/finance/prompts/secque_template.yaml"
EVAL_PROMPT = EVAL_DIR / "prompts/secque.yaml"
JUDGE_USER_PROMPT = EVAL_DIR / "prompts/judge_user.txt"

QA_BENCHMARKS = ("secque", "financebench")
FINANCE_CONTEXT = ProjectResolver().resolve_reference("finance")


def _user_message(path: Path) -> str:
    return yaml.safe_load(path.read_text())["user"]


def _judge_settings() -> dict:
    base = yaml.safe_load(EVAL_BASE.read_text())
    return base["benchmarks"]["secque"]["gym"]["equivalence_llm_judge"]["resources_servers"][
        "equivalence_llm_judge"
    ]


def test_prompt_files_exist():
    assert TRAIN_PROMPT.is_file(), TRAIN_PROMPT
    assert EVAL_PROMPT.is_file(), EVAL_PROMPT


def test_train_and_eval_prompts_match():
    train = _user_message(TRAIN_PROMPT).replace("{problem}", "{question}")
    assert train == _user_message(EVAL_PROMPT), (
        f"{TRAIN_PROMPT.name} and {EVAL_PROMPT.name} have diverged. Evaluation would "
        f"no longer measure the prompt the model was trained on."
    )


def test_placeholders_are_the_expected_ones():
    train = _user_message(TRAIN_PROMPT)
    eval_ = _user_message(EVAL_PROMPT)

    assert "{problem}" in train and "{question}" not in train
    assert "{question}" in eval_ and "{problem}" not in eval_
    assert "{context}" in train and "{context}" in eval_


def test_eval_prompt_is_the_one_base_yaml_renders():
    base = yaml.safe_load((REPO_ROOT / "projects/finance/workflows/eval/base.yaml").read_text())
    entries = base["stages"]["eval_data"]["benchmarks"]
    secque = next(e for e in entries if e.get("benchmark") == "secque")

    assert FINANCE_CONTEXT.launcher_path(secque["prompt_config"]) == EVAL_PROMPT


def test_judge_rubric_is_delivered_as_a_system_message():
    rubric = _judge_settings()["judge_system_message"]

    assert rubric.strip(), "an empty system message pushes the rubric into the user turn"
    assert "impartial judge" in rubric
    assert "[[rating]]" in rubric


def test_judge_user_turn_carries_only_the_question_and_answers():
    template = JUDGE_USER_PROMPT.read_text()

    assert template.startswith("QUESTION:")
    assert "impartial judge" not in template, "the rubric belongs in the system message"
    # The names Gym substitutes; a typo here renders the placeholder literally.
    for placeholder in ("{question}", "{expected_answer}", "{generated_answer}"):
        assert placeholder in template


def test_judge_settings_are_inline_and_shared_by_both_qa_benchmarks():
    base = yaml.safe_load(EVAL_BASE.read_text())
    configured = _judge_settings()["judge_prompt_template_fpath"]
    assert FINANCE_CONTEXT.launcher_path(configured) == JUDGE_USER_PROMPT

    settings = [
        base["benchmarks"][name]["gym"]["equivalence_llm_judge"]["resources_servers"][
            "equivalence_llm_judge"
        ]
        for name in QA_BENCHMARKS
    ]
    assert settings[0] == settings[1]
    assert all(
        paths == ["resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml"]
        for paths in (base["benchmarks"][name]["config_paths"] for name in QA_BENCHMARKS)
    )
