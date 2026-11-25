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
"""SEC identity is required by the one SDG stage that calls EDGAR."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from nvfinance.stages.sdg.create_seed_data import CreateSeedDataStage
from omegaconf import OmegaConf

from nvflow.core.workflow_definition import load_config_with_inheritance

WORKFLOW = Path("projects/finance/workflows/sdg/template-based-sdg.yaml")

SEC_CALLING_STAGE = "create_seed_data"


def _config() -> dict[str, Any]:
    return {
        "output_file": "/runs/seed/seed_questions.jsonl",
        "company_info_file": "/runs/seed/company_info.tsv",
        "sec_identity_email": "analyst@example.com",
        "sec_identity_company": "Example Corp",
    }


def test_validation_accepts_a_complete_identity() -> None:
    CreateSeedDataStage().validate_config(_config())


@pytest.mark.parametrize("absent", ["sec_identity_email", "sec_identity_company"])
def test_validation_rejects_a_partial_identity(absent: str) -> None:
    config = _config()
    config[absent] = None

    with pytest.raises(ValueError, match="SEC EDGAR"):
        CreateSeedDataStage().validate_config(config)


def test_validation_names_both_variables_when_neither_is_set() -> None:
    config = _config() | {"sec_identity_email": None, "sec_identity_company": None}

    with pytest.raises(ValueError) as raised:
        CreateSeedDataStage().validate_config(config)

    message = str(raised.value)
    assert "export SEC_IDENTITY_EMAIL=<value> SEC_IDENTITY_COMPANY=<value>" in message


def test_workflow_resolves_without_an_sec_identity(monkeypatch) -> None:
    """Every other stage must stay runnable in a shell that exported nothing."""
    monkeypatch.delenv("SEC_IDENTITY_EMAIL", raising=False)
    monkeypatch.delenv("SEC_IDENTITY_COMPANY", raising=False)

    config = OmegaConf.to_container(load_config_with_inheritance(WORKFLOW), resolve=True)

    stage = config["stages"][SEC_CALLING_STAGE]
    assert stage["sec_identity_email"] is None
    assert stage["sec_identity_company"] is None
