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
"""Preprocess reconciled HopChain queries for Omni difficulty filter.

Each accepted record is expanded into k copies so that nemo-skills inference
runs k independent samples per question. The consensus answer and original
record are carried in _metadata so the postprocess step can score and filter.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from nvflow.recipes.multimodal.utils.hopchain_llm_judge_common import load_prompt_template
from nvflow.recipes.multimodal.utils.hopchain_sdg_models import ReconciledHopChainQuery

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand reconciled queries into k difficulty-filter requests"
    )
    parser.add_argument(
        "--input", required=True, help="reconciled_queries.jsonl from reconcile step"
    )
    parser.add_argument(
        "--output", required=True, help="OpenAI-format JSONL for nemo-skills generate"
    )
    parser.add_argument("--prompt", required=True, help="Prompt template file")
    parser.add_argument("--k", type=int, default=5, help="Samples per question")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Inject chat_template_kwargs={enable_thinking: true} per datapoint (note: not read by nemo_skills inference; omni-step70 always reasons via vLLM reasoning parser)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_template = load_prompt_template(args.prompt)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    total_requests = 0
    with Path(args.input).open("r") as f_in, output_path.open("w") as f_out:
        for line_num, line in enumerate(f_in, start=1):
            if not line.strip():
                continue
            try:
                record = ReconciledHopChainQuery.model_validate(json.loads(line))
                if record.llm_judge_reconciliation_status != "accepted":
                    continue
                prompt_text = prompt_template.format(question=record.question)
                for i in range(args.k):
                    message: dict = {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        # NeMo-Skills resolves local paths and encodes them as
                                        # data URLs before calling the vLLM OpenAI endpoint.
                                        "image_url": {"url": record.image_fullpath},
                                    },
                                    {"type": "text", "text": prompt_text},
                                ],
                            }
                        ],
                        "_metadata": {
                            "query_id": record.query_id,
                            "sample_index": i,
                            "k": args.k,
                            "consensus_normalized_answer": record.llm_judge_consensus_normalized_answer,
                            "record": record.model_dump(),
                        },
                    }
                    if args.enable_thinking:
                        message["chat_template_kwargs"] = {"enable_thinking": True}
                    f_out.write(json.dumps(message) + "\n")
                    total_requests += 1
                total_records += 1
            except Exception as exc:
                logger.exception("Failed to preprocess line %s: %s", line_num, exc)

    logger.info(
        "Preprocessed %d accepted records → %d inference requests (k=%d)",
        total_records,
        total_requests,
        args.k,
    )


if __name__ == "__main__":
    main()
