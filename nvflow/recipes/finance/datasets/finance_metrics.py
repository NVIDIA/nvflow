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
"""Custom 3-tier metrics for finance benchmarks.

Parses Rating: [[X]] format where X is 0, 1, or 2:
- [[2]] = Correct
- [[1]] = Partially correct
- [[0]] = Incorrect

Reports both strict (only [[2]]) and lenient ([[1]] + [[2]]) accuracy.
"""

import re

from nemo_skills.evaluation.metrics.base import as_int, as_percentage
from nemo_skills.evaluation.metrics.math_metrics import MathMetrics

from nvflow.utils import setup_logger

logger = setup_logger(__name__)


def parse_rating(judgement: str) -> int | None:
    """Extract rating from judge response.

    Looks for pattern: [[X]] where X is 0, 1, or 2

    Args:
        judgement: Raw judgement text from LLM judge

    Returns:
        0, 1, or 2 if found, None if parsing fails
    """
    if not judgement:
        return None

    # Look for [[0]], [[1]], or [[2]] -- take the last match so reasoning
    # judges that mention earlier ratings are recorded correctly.
    matches = list(re.finditer(r"\[\[([012])\]\]", judgement))
    if matches:
        return int(matches[-1].group(1))

    # Fallback: try to find rating number in various formats (last match wins)
    fallback = list(re.finditer(r"rating[:\s]+([012])", judgement.lower()))
    if fallback:
        return int(fallback[-1].group(1))

    return None


class FinanceMetrics(MathMetrics):
    """3-tier metrics for finance benchmarks.

    Extends MathMetrics to parse [[rating]] format and report:
    - judge_correct: Strict accuracy (only [[2]] counts)
    - judge_lenient: Lenient accuracy ([[1]] and [[2]] count)
    - judge_partial: Partial answer rate (only [[1]])
    - avg_score: Weighted average score (0=0%, 1=50%, 2=100%)
    """

    def __init__(self, compute_no_answer: bool = True, answer_key: str = "predicted_answer"):
        super().__init__(compute_no_answer=compute_no_answer, answer_key=answer_key)

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        """Parse 3-tier rating from judgement field.

        Returns:
            Dictionary with boolean scoring methods (required for pass@k):
            - judge_correct: True if rating is 2 (strict)
            - judge_lenient: True if rating is 1 or 2 (lenient)
            - judge_partial: True if rating is 1 (partial only)
        """
        correctness_dict: dict[str, bool | int | float] = {}

        if "judgement" in prediction:
            rating = parse_rating(prediction["judgement"])

            if rating is not None:
                # Strict: only [[2]] is correct
                correctness_dict["judge_correct"] = rating == 2

                # Lenient: [[1]] and [[2]] are correct
                correctness_dict["judge_lenient"] = rating >= 1

                # Partial: only [[1]]
                correctness_dict["judge_partial"] = rating == 1

                # Weighted score: 0→0%, 1→50%, 2→100%
                correctness_dict["avg_score"] = rating / 2.0
            else:
                # Failed to parse - treat as incorrect
                logger.warning(
                    "Failed to parse rating from judgement: %s...",
                    prediction["judgement"][:100] if prediction["judgement"] else "empty",
                )
                correctness_dict["judge_correct"] = False
                correctness_dict["judge_lenient"] = False
                correctness_dict["judge_partial"] = False
                correctness_dict["avg_score"] = 0.0

        # Also check for symbolic_correct if present (from other graders)
        if "symbolic_correct" in prediction:
            correctness_dict["symbolic_correct"] = prediction["symbolic_correct"]

        return correctness_dict

    def metrics_to_print(self):
        """Define which metrics to print in summary."""
        metrics_to_print = {
            "num_entries": as_int,
            "avg_tokens": as_int,
            "gen_seconds": as_int,
            "judge_correct": as_percentage,  # Strict: only [[2]]
            "judge_lenient": as_percentage,  # Lenient: [[1]] + [[2]]
            "judge_partial": as_percentage,  # Only [[1]]
            "avg_score": as_percentage,  # Weighted: 0→0%, 1→50%, 2→100%
        }
        if self.compute_no_answer:
            metrics_to_print["no_answer"] = as_percentage
        return metrics_to_print
