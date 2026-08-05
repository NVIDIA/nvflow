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
"""Data preprocessing stage for Document-Grounded SDG."""

from typing import Any

from nvflow.core import BaseStage, console
from nvflow.lib.rl.helpers import resolve_host_path


class DGSDGPreprocessStage(BaseStage):
    """Preprocess domain documents into structured JSONL data."""

    workflow = "document_grounded_sdg"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute the data preprocessing pipeline."""
        from nemo_skills.pipeline.cli import run_cmd, wrap_arguments

        input_dir = config["input_dir"]
        output_dir = config["output_dir"]
        distribution_dir = config["distribution_dir"]

        max_tokens = config.get("max_tokens", 2000)
        overlap_tokens = config.get("overlap_tokens", 100)
        total_samples = config.get("total_samples", 150000)
        max_skip_count = config.get("max_skip_count", 20000)
        seed = config.get("seed", 42)
        preprocess_module = config["preprocess_module"]
        rerun_done = config.get("rerun_done", False)

        # Domain-agnostic passthrough: arbitrary extra CLI args forwarded verbatim
        # to the preprocess_module. Lets domain recipes pass module-specific flags
        # (e.g. the SEC recipe's --forms) without this generic stage knowing about
        # them. Bool True -> bare flag; other values -> "--key value" (quoted).
        extra_args = config.get("extra_args") or {}
        extra_parts: list[str] = []
        for key, value in extra_args.items():
            if isinstance(value, bool):
                if value:
                    extra_parts.append(f"--{key}")
            elif isinstance(value, list | tuple):
                extra_parts.append(f"--{key} '{' '.join(str(v) for v in value)}'")
            elif isinstance(value, str):
                extra_parts.append(f"--{key} '{value}'")
            else:
                extra_parts.append(f"--{key} {value}")
        extra_args_str = " ".join(extra_parts)

        console.status("Document data preprocessing")
        console.detail("Input dir", input_dir)
        console.detail("Output dir", output_dir)
        console.detail("Distribution dir", distribution_dir)
        console.detail("Preprocess module", preprocess_module)
        console.detail("Max tokens", str(max_tokens))
        console.detail("Overlap tokens", str(overlap_tokens))
        console.detail("Total samples", str(total_samples))
        console.detail("Max skip count", str(max_skip_count))
        console.detail("Seed", str(seed))
        if extra_args_str:
            console.detail("Extra args", extra_args_str)
        console.blank()

        # Reuse previously materialized sampling output by default.
        # Set rerun_done=true to force a full regenerate.
        #
        # ``execute()`` runs on the orchestrator/login node, so ``output_dir``
        # (a container path like ``/workspace/...``) must be resolved to its
        # host path before the existence check -- otherwise it never matches and
        # sampling re-runs on every launch.
        forms_arg = str((extra_args or {}).get("forms", "10-K 10-Q"))
        forms = [f for f in forms_arg.split() if f]
        host_jsonl_dir = resolve_host_path(f"{output_dir}/jsonl")
        if forms and not rerun_done:
            expected_outputs = [host_jsonl_dir / f"{form.lower()}-data.jsonl" for form in forms]
            all_present = all(p.exists() and p.stat().st_size > 0 for p in expected_outputs)
            if all_present:
                console.success("Data preprocessing skipped (reusing existing sampled output)")
                console.detail("Output directory", output_dir)
                return

        full_cmd = (
            f"python3 -m {preprocess_module} "
            f"--input_dir {input_dir} "
            f"--output_dir {output_dir} "
            f"--distribution_dir {distribution_dir} "
            f"--max_tokens {max_tokens} "
            f"--overlap_tokens {overlap_tokens} "
            f"--total_samples {total_samples} "
            f"--max_skip_count {max_skip_count} "
            f"--seed {seed}"
        )
        if extra_args_str:
            full_cmd += f" {extra_args_str}"

        run_cmd(
            ctx=wrap_arguments(full_cmd),
            cluster=cluster,
            expname=expname,
            run_after=run_after,
        )

        console.success("Data preprocessing job submitted")
        console.detail("Output directory", output_dir)

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate stage configuration."""
        required = ["input_dir", "output_dir", "distribution_dir", "preprocess_module"]
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
