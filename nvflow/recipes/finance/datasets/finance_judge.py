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
"""Custom LLM judge for finance benchmarks with optional answer extraction.

By default, extracts answers using regex pattern "Answer:\\s*([\\s\\S]*)".
Set skip_extraction=True to use full generation as answer (no extraction).
Uses finance-specific judge prompt (secque_judge.yaml) for comparison.
"""

from dataclasses import field

import hydra
from nemo_skills.evaluation.math_grader import extract_answer
from nemo_skills.inference.generate import GenerationTask, GenerationTaskConfig, InferenceConfig
from nemo_skills.utils import nested_dataclass, prefill_judgement

from nvflow.utils import setup_logger

logger = setup_logger(__name__)


@nested_dataclass(kw_only=True)
class FinanceLLMJudgeConfig(GenerationTaskConfig):
    """Finance LLM judge configuration."""

    # LLM call parameters
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: dict = field(default_factory=dict)

    # Judge prompt and output key
    prompt_config: str = "nvflow/recipes/finance/prompts/secque_judge"
    generation_key: str = "judgement"
    add_generation_stats: bool = False

    # Answer extraction configuration
    skip_extraction: bool = False  # If True, use full generation as answer (no extraction)
    extract_from_boxed: bool = False
    extract_regex: str = r"Answer:\s*([\s\S]*)"


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_finance_llm_judge_config", node=FinanceLLMJudgeConfig)


class FinanceLLMJudgeTask(GenerationTask):
    """Custom judge task with finance-specific answer extraction."""

    def __init__(self, cfg: FinanceLLMJudgeConfig):
        super().__init__(cfg)
        self.skip_extraction = cfg.skip_extraction
        self.extract_from_boxed = cfg.extract_from_boxed
        self.extract_regex = cfg.extract_regex

    def preprocess_data(self, data):
        """Extract predicted answer from generation before judging.

        If skip_extraction=True, uses full generation as answer.
        Otherwise, extracts using regex with fallback to full generation.
        """
        logger.info("Preprocessing %d samples for answer extraction", len(data))

        if self.skip_extraction:
            logger.info("skip_extraction=True, using full generation as answer")
            for data_point in data:
                if "predicted_answer" not in data_point:
                    generation = data_point.get("generation", "")
                    data_point["predicted_answer"] = generation.strip() if generation else None
            return data

        # Normal extraction flow
        extracted_count = 0
        for data_point in data:
            if "predicted_answer" not in data_point:
                extract_from_boxed = data_point.get("extract_from_boxed", self.extract_from_boxed)
                extract_regex = data_point.get("extract_regex", self.extract_regex)

                generation = data_point.get("generation", "")
                predicted_answer = extract_answer(
                    generation,
                    extract_from_boxed=extract_from_boxed,
                    extract_regex=extract_regex,
                )

                if predicted_answer:
                    data_point["predicted_answer"] = predicted_answer
                    extracted_count += 1
                    logger.debug(
                        "Extracted: %s",
                        predicted_answer[:100] if predicted_answer else None,
                    )
                else:
                    # Fallback: use full generation if no Answer: pattern found
                    data_point["predicted_answer"] = generation.strip() if generation else None
                    logger.warning(
                        "No 'Answer:' found, using full generation as answer: %s...",
                        generation[:100] if generation else "empty",
                    )

        fallback_count = len(data) - extracted_count
        logger.info(
            "Extracted answers: %d with 'Answer:' pattern, %d fallback to full generation",
            extracted_count,
            fallback_count,
        )
        return data

    def prefill_generation(self, data_point):
        """Prefill judgement if already available."""
        judgement = prefill_judgement(data_point)
        return {"generation": judgement} if judgement else None


# nemo-skills discovers this class via module import
GENERATION_TASK_CLASS = FinanceLLMJudgeTask


@hydra.main(version_base=None, config_name="base_finance_llm_judge_config")
def generate(cfg: FinanceLLMJudgeConfig):
    """Main entry point for the finance LLM judge."""
    cfg = FinanceLLMJudgeConfig(_init_nested=True, **cfg)
    logger.debug("Finance LLM Judge config: %s", cfg)
    FinanceLLMJudgeTask(cfg).generate()


if __name__ == "__main__":
    generate()
