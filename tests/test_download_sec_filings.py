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
"""Tests for nvflow.recipes.finance.stages.download.download_sec_filings.

``DownloadSecFilingsStage`` does not import ``nemo_skills`` at module
level (only inside ``execute()``), so this module is importable in the
lightweight CI environment. The command-rendering tests stub
``nemo_skills.pipeline.cli`` in ``sys.modules`` so ``execute()`` can run
end-to-end without the real dependency -- they assert only on what this
stage passes to ``run_cmd``/``wrap_arguments``, not on nemo-skills'
internal behavior.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml

from nvflow.recipes.finance.stages.download.download_sec_filings import (
    DownloadSecFilingsStage,
)


def _base_config(**overrides: Any) -> dict[str, Any]:
    config = {
        "output_dir": "/data/sec",
        "sec_identity_email": "test@example.com",
        "sec_identity_company": "Test Co",
        "tickers": ["AAPL", "MSFT"],
        "start_year": 2020,
        "end_year": 2023,
    }
    config.update(overrides)
    return config


class TestValidateConfig:
    def test_valid_direct_config_passes(self) -> None:
        DownloadSecFilingsStage().validate_config(_base_config())

    def test_valid_config_file_reference_does_not_require_tickers(self) -> None:
        config = {
            "output_dir": "/data/sec",
            "sec_identity_email": "test@example.com",
            "sec_identity_company": "Test Co",
            "config": "filings.yaml",
        }
        DownloadSecFilingsStage().validate_config(config)

    @pytest.mark.parametrize(
        "missing_field", ["output_dir", "sec_identity_email", "sec_identity_company"]
    )
    def test_missing_always_required_field_raises(self, missing_field: str) -> None:
        config = _base_config()
        del config[missing_field]
        with pytest.raises(ValueError, match=missing_field):
            DownloadSecFilingsStage().validate_config(config)

    @pytest.mark.parametrize("missing_field", ["tickers", "start_year", "end_year"])
    def test_missing_field_without_config_file_raises(self, missing_field: str) -> None:
        config = _base_config()
        del config[missing_field]
        with pytest.raises(ValueError, match=missing_field):
            DownloadSecFilingsStage().validate_config(config)


class TestExecuteCommandRendering:
    """Exercises execute() with nemo_skills.pipeline.cli stubbed out.

    Captures the ``ctx`` string passed to ``run_cmd`` (our stub
    ``wrap_arguments`` is the identity function) to assert on exactly
    what shell command this stage builds.
    """

    @pytest.fixture
    def captured_run_cmd(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        captured: dict[str, Any] = {}

        def fake_run_cmd(**kwargs: Any) -> None:
            captured.update(kwargs)

        fake_cli = types.ModuleType("nemo_skills.pipeline.cli")
        fake_cli.run_cmd = fake_run_cmd  # type: ignore[attr-defined]
        fake_cli.wrap_arguments = lambda cmd: cmd  # type: ignore[attr-defined]

        fake_pipeline = types.ModuleType("nemo_skills.pipeline")
        fake_pipeline.cli = fake_cli  # type: ignore[attr-defined]

        fake_nemo_skills = types.ModuleType("nemo_skills")
        fake_nemo_skills.pipeline = fake_pipeline  # type: ignore[attr-defined]

        monkeypatch.setitem(sys.modules, "nemo_skills", fake_nemo_skills)
        monkeypatch.setitem(sys.modules, "nemo_skills.pipeline", fake_pipeline)
        monkeypatch.setitem(sys.modules, "nemo_skills.pipeline.cli", fake_cli)
        return captured

    def test_renders_python3_module_invocation(self, captured_run_cmd: dict[str, Any]) -> None:
        DownloadSecFilingsStage().execute(_base_config(), cluster="my_cluster", expname="exp")
        assert captured_run_cmd["ctx"].startswith(
            "python3 -m nvflow.recipes.finance.utils.download.download_sec_filings "
        )

    def test_renders_all_expected_flags(self, captured_run_cmd: dict[str, Any]) -> None:
        DownloadSecFilingsStage().execute(_base_config(), cluster="my_cluster", expname="exp")
        cmd = captured_run_cmd["ctx"]
        assert "--tickers 'AAPL MSFT'" in cmd
        assert "--start_year 2020" in cmd
        assert "--end_year 2023" in cmd
        assert "--output_dir /data/sec" in cmd
        assert "--sec_email test@example.com" in cmd
        assert "--sec_company 'Test Co'" in cmd

    def test_shell_metacharacters_in_ticker_are_safely_quoted(
        self, captured_run_cmd: dict[str, Any]
    ) -> None:
        malicious_config = _base_config(tickers=["AAPL; rm -rf /", "$(whoami)"])
        DownloadSecFilingsStage().execute(malicious_config, cluster="my_cluster", expname="exp")
        cmd = captured_run_cmd["ctx"]
        # shlex.quote wraps the whole space-joined tickers string in single
        # quotes, so the shell sees one literal argument -- ``;`` and
        # ``$(...)`` cannot terminate the command or trigger substitution.
        assert "--tickers 'AAPL; rm -rf / $(whoami)'" in cmd

    def test_loads_tickers_from_referenced_config_file(
        self, captured_run_cmd: dict[str, Any], tmp_path: Path
    ) -> None:
        filings_config = tmp_path / "filings.yaml"
        filings_config.write_text(
            yaml.dump({"tickers": ["NVDA"], "start_year": 2021, "end_year": 2022})
        )
        config = {
            "output_dir": "/data/sec",
            "sec_identity_email": "test@example.com",
            "sec_identity_company": "Test Co",
            "config": str(filings_config),
        }
        DownloadSecFilingsStage().execute(config, cluster="my_cluster", expname="exp")
        cmd = captured_run_cmd["ctx"]
        assert "--tickers NVDA" in cmd
        assert "--start_year 2021" in cmd
        assert "--end_year 2022" in cmd
