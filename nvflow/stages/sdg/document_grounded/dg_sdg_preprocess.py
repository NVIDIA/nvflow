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
"""Data preprocessing stage for Document-Grounded SDG.

Supports two sampling strategies via ``sampling_strategy``:
  random     — single-document random sampling weighted by distribution CSVs
               (original DG-SDG behaviour; requires ``distribution_dir``).
  relational — multi-document sampling along structural relation axes
               (temporal / cross-form / same-year / multi-company).
               Mode weights and guidance filter are passed via ``extra_args``.
               ``distribution_dir`` is optional; the relational engine
               auto-generates uniform CSVs if the directory is absent.

Project configuration maps sampling strategies to project-owned preprocessing
modules. ``preprocess_module`` remains available for a direct one-off selection.
"""

from typing import Any

from nvflow.core import console
from nvflow.execution import CapturedExecutionStage, declare_task_outputs
from nvflow.execution.capture import reusable_output


class DGSDGPreprocessStage(CapturedExecutionStage):
    """Preprocess domain documents into structured JSONL data."""

    capture_strategy = "run_cmd"
    capture_empty_reason = "existing sampled output is reused"

    def build_tasks(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        """Execute the data preprocessing pipeline."""
        from nvflow.integrations.nemo_skills.commands import run_cmd, wrap_arguments

        input_dir = config["input_dir"]
        output_dir = config["output_dir"]

        max_tokens = config.get("max_tokens", 2000)
        overlap_tokens = config.get("overlap_tokens", 100)
        total_samples = config.get("total_samples", 150000)
        max_skip_count = config.get("max_skip_count", 20000)
        seed = config.get("seed", 42)

        sampling_strategy = config.get("sampling_strategy")
        if sampling_strategy:
            preprocess_modules = config.get("preprocess_modules", {})
            if sampling_strategy not in preprocess_modules:
                raise ValueError(
                    f"sampling_strategy must be one of {list(preprocess_modules)}, "
                    f"got {sampling_strategy!r}"
                )
            preprocess_module = preprocess_modules[sampling_strategy]
        else:
            preprocess_module = config["preprocess_module"]

        # distribution_dir: required for random (feeds hand-crafted section
        # weights); optional for relational (engine auto-generates uniform CSVs).
        if sampling_strategy == "relational":
            distribution_dir = config.get("distribution_dir", f"{output_dir}/auto_distribution")
        else:
            distribution_dir = config["distribution_dir"]

        # rerun_done: the skip-check looks for {form}-data.jsonl, which the
        # relational engine doesn't emit (it writes mode-specific filenames).
        # Auto-enable for the relational path so the check is always bypassed.
        rerun_done = config.get("rerun_done", sampling_strategy == "relational")

        # Domain-agnostic passthrough: arbitrary extra CLI args forwarded verbatim
        # to the preprocess_module. Lets domain projects pass module-specific flags
        # (e.g. the SEC project's --forms) without this generic stage knowing about
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
        console.detail("Sampling strategy", sampling_strategy or "explicit")
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
        forms_arg = str((extra_args or {}).get("forms", "10-K 10-Q"))
        forms = [f for f in forms_arg.split() if f]
        if forms and not rerun_done:
            if all(
                reusable_output(f"{output_dir}/jsonl/{form.lower()}-data.jsonl") for form in forms
            ):
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

        with declare_task_outputs(output_dir=output_dir):
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
        for field in ("input_dir", "output_dir"):
            if field not in config:
                raise ValueError(f"Missing required field: {field}")

        has_strategy = "sampling_strategy" in config
        has_module = "preprocess_module" in config
        if not has_strategy and not has_module:
            raise ValueError("Either 'sampling_strategy' or 'preprocess_module' must be set")

        strategy = config.get("sampling_strategy")
        preprocess_modules = config.get("preprocess_modules", {})
        if strategy and strategy not in preprocess_modules:
            raise ValueError(
                f"sampling_strategy must be one of {list(preprocess_modules)}, got {strategy!r}"
            )

        # distribution_dir required for random sampling (relational engine
        # auto-generates uniform CSVs when the directory is absent).
        if strategy == "random" and "distribution_dir" not in config:
            raise ValueError("distribution_dir is required for sampling_strategy: random")
