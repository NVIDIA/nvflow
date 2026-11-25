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
"""Tests for prompt template loading and parity with NeMo-Gym's converter."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
from nvfinance.utils.rl.prompt_template_applier import load_prompt_template

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"

SEARCH_TOOLS = ("edgar_search", "sec_filing_search")
WEB_SUFFIXES = ("without_web", "with_web")

# Gym renders this date into the prompt; nvflow renders {current_date} instead.
GYM_CURRENT_DATE = "February 23, 2026"
# Gym says "to answer questions" where vals-ai/finance-agent, the harness the
# policy is scored against, says "to apply answer questions". nvflow follows
# vals-ai, so the comparison normalises Gym rather than the template.
GYM_WORDING = "retrieve_information tool to answer questions"
VALS_WORDING = "retrieve_information tool to apply answer questions"


def template_path(search_tool: str, web_suffix: str) -> Path:
    return PROMPTS / f"finance_sec_search_template_{search_tool}_{web_suffix}.yaml"


def expected_tool_order(search_tool: str, web: bool) -> list[str]:
    tools = ["retrieve_information", "parse_html_page", search_tool, "submit_final_result"]
    return ["web_search"] + tools if web else tools


ALL_COMBINATIONS = [
    (search_tool, web_suffix) for search_tool in SEARCH_TOOLS for web_suffix in WEB_SUFFIXES
]


class TestShippedTemplates:
    """Each template fully determines what a training record advertises."""

    @pytest.mark.parametrize(("search_tool", "web_suffix"), ALL_COMBINATIONS)
    def test_template_exists(self, search_tool: str, web_suffix: str):
        assert template_path(search_tool, web_suffix).is_file()

    @pytest.mark.parametrize(("search_tool", "web_suffix"), ALL_COMBINATIONS)
    def test_tool_order_is_pinned(self, search_tool: str, web_suffix: str):
        loaded = load_prompt_template(str(template_path(search_tool, web_suffix)))
        names = [tool["name"] for tool in loaded["response_params"]["tools"]]

        assert names == expected_tool_order(search_tool, web_suffix == "with_web")

    @pytest.mark.parametrize(("search_tool", "web_suffix"), ALL_COMBINATIONS)
    def test_exactly_one_filing_search_tool_is_offered(self, search_tool: str, web_suffix: str):
        loaded = load_prompt_template(str(template_path(search_tool, web_suffix)))
        names = [tool["name"] for tool in loaded["response_params"]["tools"]]

        assert [n for n in names if n in SEARCH_TOOLS] == [search_tool]

    @pytest.mark.parametrize(("search_tool", "web_suffix"), ALL_COMBINATIONS)
    def test_prompt_renders_without_leftover_placeholders(self, search_tool: str, web_suffix: str):
        loaded = load_prompt_template(str(template_path(search_tool, web_suffix)))

        rendered = loaded["user_template"].format(problem="Q?", current_date="April 07, 2025")

        assert "{problem}" not in rendered
        assert "{current_date}" not in rendered
        # The sources example must survive as literal braces for the model.
        assert '{{\n    "sources"' in rendered


class TestGymParity:
    """Gym's converter is the source of truth for prompt and tool text.

    A drifted description or reordered tool silently changes what the policy
    sees, and would make a dataset built by Gym's CLI differ from one built
    here.
    """

    @pytest.mark.parametrize(("search_tool", "web_suffix"), ALL_COMBINATIONS)
    def test_tools_match_gym_exactly(self, search_tool: str, web_suffix: str):
        gym = _gym_convert_questions()
        web = web_suffix == "with_web"
        gym_tools = [
            gym.RETRIEVE_INFORMATION_TOOL,
            gym.PARSE_HTML_TOOL,
            gym.SEARCH_TOOLS[search_tool],
        ]
        if web:
            gym_tools = [gym.WEB_TOOL] + gym_tools
        gym_tools.append(gym.SUBMIT_TOOL)

        loaded = load_prompt_template(str(template_path(search_tool, web_suffix)))

        assert loaded["response_params"]["tools"] == gym_tools

    @pytest.mark.parametrize(("search_tool", "web_suffix"), ALL_COMBINATIONS)
    def test_prompt_matches_gym_exactly(self, search_tool: str, web_suffix: str):
        gym = _gym_convert_questions()
        expected = gym.PROMPT.replace(GYM_WORDING, VALS_WORDING)

        loaded = load_prompt_template(str(template_path(search_tool, web_suffix)))
        rendered = loaded["user_template"].format(problem="", current_date=GYM_CURRENT_DATE)

        assert rendered.rstrip() == expected.rstrip()

    def test_gym_still_spells_the_wording_this_recipe_overrides(self):
        """If Gym adopts the vals-ai wording, the override above becomes a no-op."""
        gym = _gym_convert_questions()

        assert GYM_WORDING in gym.PROMPT or VALS_WORDING in gym.PROMPT


def _gym_convert_questions():
    """Load NeMo-Gym's question converter, or skip.

    Loaded from a checkout rather than imported so the test runs against the Gym
    the records will actually be rolled out against. Point NEMO_GYM_PATH at one,
    or run inside the nemo-gym container where /opt/nemo-gym exists.
    """
    root = Path(os.environ.get("NEMO_GYM_PATH", "/opt/nemo-gym"))
    path = root / "resources_servers/finance_sec_search/scripts/convert_questions.py"
    if not path.is_file():
        pytest.skip(f"NeMo-Gym not available at {root}")
    name = "gym_convert_questions"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
