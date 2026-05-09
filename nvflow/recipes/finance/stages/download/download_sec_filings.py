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
"""Download SEC filings and extract sections using in-tree utilities."""

from pathlib import Path
from typing import Any

import yaml

from nvflow.core import BaseStage, StageRegistry, console


@StageRegistry.register(recipe="finance", workflow="download-sec", stage="sap-500")
@StageRegistry.register(recipe="finance", workflow="download-sec", stage="demo")
@StageRegistry.register(recipe="finance", workflow="download-sec", stage="smoke")
class DownloadSecFilingsStage(BaseStage):
    """Download SEC filings (10-K, 10-Q, 8-K) from EDGAR and extract sections."""

    workflow = "download-sec"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Download SEC filings for specified tickers and date range."""
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        # Load separate config file if specified
        if "config" in config:
            config_path = Path(config["config"])

            console.detail("Loading config from", str(config_path))
            with open(config_path) as f:
                filings_config = yaml.safe_load(f)

            tickers = filings_config.get("tickers")
            start_year = filings_config.get("start_year")
            end_year = filings_config.get("end_year")
            forms = filings_config.get("forms", ["10-K", "10-Q", "8-K"])
        else:
            tickers = config["tickers"]
            start_year = config["start_year"]
            end_year = config["end_year"]
            forms = config.get("forms", ["10-K", "10-Q", "8-K"])

        output_dir = config["output_dir"]
        sec_identity_email = config["sec_identity_email"]
        sec_identity_company = config["sec_identity_company"]

        console.status("Downloading SEC filings")
        console.detail("Output directory", output_dir)
        console.detail("Tickers", ", ".join(tickers) if isinstance(tickers, list) else tickers)
        console.detail("Year range", f"{start_year} - {end_year}")
        console.detail("Form types", ", ".join(forms) if isinstance(forms, list) else forms)
        console.detail("SEC Identity", f"{sec_identity_company} <{sec_identity_email}>")

        tickers_str = " ".join(tickers) if isinstance(tickers, list) else tickers
        forms_str = " ".join(forms) if isinstance(forms, list) else forms
        log_dir = Path(output_dir) / "download-logs"

        run_cmd(
            ctx=wrap_arguments(
                f"python -m nvflow.recipes.finance.utils.download.download_sec_filings "
                f'--tickers "{tickers_str}" '
                f'--forms "{forms_str}" '
                f"--start_year {start_year} "
                f"--end_year {end_year} "
                f"--output_dir {output_dir} "
                f'--sec_email "{sec_identity_email}" '
                f'--sec_company "{sec_identity_company}"'
            ),
            cluster=cluster,
            **config.get("stage_kwargs", {}),
            expname=expname,
            log_dir=str(log_dir),
            run_after=run_after,
        )

        console.success("Downloaded SEC filings")
