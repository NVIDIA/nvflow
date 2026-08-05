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
"""Validate SDG questions before they enter the GRPO pipeline.

Two-phase per-environment pipeline:

1. **Regex prefilter (CPU).** Drops questions that use "the company"/
   "the firm"/etc. with no named company or ticker anywhere in the text.
   Intentionally narrow -- recall over precision.

2. **LLM classifier (GPU).** For every question that survived Phase 1,
   asks a judge model (GPT-OSS-120B by default) to return
   ``Answer: VALID`` / ``Answer: INVALID``.  A post-processing chain
   parses the tag and splits records into kept (VALID) and dropped
   (INVALID) streams.  Parse failures default to VALID.

The kept stream is written to ``{output_dir}/{env}/final_result.jsonl``
so downstream ``data_transformation`` can consume it via the normal
``env.raw_train_data`` path (re-pointed at this stage's output in the
model config).

Per-env output layout:

  {output_dir}/{env_name}/
  ├── final_result.jsonl        <- VALID records, read by data_transformation
  ├── phase1_regex/
  │   ├── prefiltered.jsonl     <- regex kept
  │   ├── regex_dropped.jsonl   <- regex dropped (audit)
  │   ├── prefilter_stats.json  <- regex phase stats
  │   └── logs/                 <- regex Slurm logs
  └── phase2_llm/
      ├── output.jsonl          <- raw LLM generation (nemo-skills)
      ├── parsed.jsonl          <- LLM output with validate_tag attached
      ├── parsed_parse_log.txt  <- parse audit
      ├── llm_dropped.jsonl     <- LLM dropped (audit)
      ├── llm_filter_stats.json <- LLM phase stats
      └── generation-logs/      <- LLM Slurm logs (nemo-skills names this)
"""

from pathlib import Path
from typing import Any

from nvflow.core import BaseStage, StageRegistry, console
from nvflow.lib.cli_cmd import build_python_cmd

# Internal artefact names (not user-facing -- see module docstring for layout).
_PHASE1_SUBDIR = "phase1_regex"
_PHASE2_SUBDIR = "phase2_llm"
_PREFILTERED = "prefiltered.jsonl"
_REGEX_DROPPED = "regex_dropped.jsonl"
_PREFILTER_STATS = "prefilter_stats.json"
_LLM_GEN_OUTPUT = "output.jsonl"
_PARSED = "parsed.jsonl"
_LLM_DROPPED = "llm_dropped.jsonl"
_LLM_FILTER_STATS = "llm_filter_stats.json"

_UTILS_MODULE = "nvflow.recipes.finance.utils.rl"


@StageRegistry.register(recipe="finance", workflow="grpo", stage="validate_questions")
class ValidateQuestionsStage(BaseStage):
    """GRPO data-quality pre-filter that runs before data_transformation."""

    workflow = "grpo"

    def execute(
        self,
        config: dict[str, Any],
        cluster: str,
        expname: str,
        run_after: list[str] | None = None,
    ) -> None:
        from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments

        from nvflow.lib.rl.helpers import resolve_environments

        environments = resolve_environments(config)
        source_data = config["source_data"]
        output_dir = config["output_dir"]
        prompt_config = config["prompt_config"]
        inline_args = config.get("inline_args", "")
        source_filename = config.get("source_filename", "final_result.jsonl")
        final_filename = config.get("final_filename", "final_result.jsonl")

        stage_kwargs = config.get("stage_kwargs", {})

        console.status("Validating SDG questions before GRPO data_transformation")
        console.detail("Source data", source_data)
        console.detail("Output base dir", output_dir)
        console.detail("Prompt config", str(prompt_config))
        console.detail("Environments", ", ".join(environments.keys()))
        console.blank()

        for env_name in environments:
            env_output_dir = Path(output_dir) / env_name
            phase1_dir = env_output_dir / _PHASE1_SUBDIR
            phase2_dir = env_output_dir / _PHASE2_SUBDIR
            source_file = Path(source_data) / source_filename
            prefiltered_file = phase1_dir / _PREFILTERED
            final_file = env_output_dir / final_filename

            console.status(f"validate_questions: environment '{env_name}'")
            console.detail("Input", str(source_file))
            console.detail("Final output (VALID)", str(final_file))
            console.blank()

            # Step A: regex prefilter (CPU) -> phase1_regex/.  Script is
            # skip-by-default when outputs already exist (rerun-safe after
            # a phase 2 crash: prefiltered.jsonl is left untouched so phase
            # 2's ``skip_filled=True`` resume by row index stays
            # consistent).  Pass ``--force`` here to bypass the skip.
            regex_cmd = build_python_cmd(
                f"{_UTILS_MODULE}.regex_prefilter_questions",
                input_file=source_file,
                output_kept=prefiltered_file,
                output_dropped=phase1_dir / _REGEX_DROPPED,
                stats_file=phase1_dir / _PREFILTER_STATS,
            )
            run_cmd(
                ctx=wrap_arguments(regex_cmd),
                cluster=cluster,
                log_dir=str(phase1_dir / "logs"),
                expname=f"{expname}-{env_name}-phase1-regex",
                run_after=run_after,
            )

            # Step B: LLM classifier + in-process postprocess -> phase2_llm/ and final_result.jsonl.
            # Resume is controlled by ``++skip_filled=True`` in inline_args (see base.yaml):
            # nemo-skills reads output.jsonl-async for already-filled indices and skips them.
            #
            # The postprocess orchestrator (``postprocess_validate``) runs parse + apply
            # in one Python process with sentinel ``PHASE: parse`` / ``PHASE: apply``
            # log lines so post-mortem analysis stays grep-able.  ``--raw_sdg_source``
            # is required: it restores the SDG-original reasoning_content per VALID
            # record (nemo-skills generate() overwrites it with LLM provider reasoning).
            postprocess_cmd = build_python_cmd(
                f"{_UTILS_MODULE}.postprocess_validate",
                llm_output=phase2_dir / _LLM_GEN_OUTPUT,
                parsed_jsonl=phase2_dir / _PARSED,
                final_kept=final_file,
                dropped=phase2_dir / _LLM_DROPPED,
                stats=phase2_dir / _LLM_FILTER_STATS,
                raw_sdg_source=source_data,
                raw_sdg_filename=source_filename,
            )
            generate(
                ctx=wrap_arguments(f"++prompt_config={prompt_config} {inline_args}".strip()),
                cluster=cluster,
                input_file=str(prefiltered_file),
                output_dir=str(phase2_dir),
                expname=f"{expname}-{env_name}-phase2-llm",
                run_after=[f"{expname}-{env_name}-phase1-regex"],
                postprocess_cmd=postprocess_cmd,
                **stage_kwargs,
            )

            console.success(f"validate_questions submitted for '{env_name}' -> {final_file}")

    def validate_config(self, config: dict[str, Any]) -> None:
        """Basic sanity checks before Slurm submission."""
        for field in ("output_dir", "source_data", "prompt_config", "environments"):
            if not config.get(field):
                raise ValueError(
                    f"stages.validate_questions.{field} is required "
                    "(see validate_questions stage docstring for details)."
                )
        stage_kwargs = config.get("stage_kwargs") or {}
        if not stage_kwargs.get("model"):
            raise ValueError(
                "stages.validate_questions.stage_kwargs.model is required "
                "(e.g., /hf_models/openai/gpt-oss-120b)."
            )
