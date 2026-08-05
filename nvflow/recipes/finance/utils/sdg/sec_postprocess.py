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
"""Thin CLI wrapper around ``dgsdg_post_process`` for the finance recipe.

Currently identical to the lib entry-point -- no SEC-specific cleaning
overrides are needed now that the SFT-eligibility filter (which used
``difficulty_score``) has been removed alongside the difficulty stage.
Kept as its own module so the workflow YAML can keep pointing at a
recipe-owned path (``recipes/finance/utils/sdg/sec_postprocess.py``);
adding SEC-specific behavior later is then a one-file edit.

Usage:
    python sec_postprocess.py --input_file /path/to/input.jsonl --output_dir /path/to/output --seed 42
"""

import argparse
import os
import sys

from nvflow.lib.sdg.document_grounded.postprocess import dgsdg_post_process
from nvflow.utils import setup_logger

logger = setup_logger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-process DG-SDG data for the finance (SEC) recipe."
    )
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)

    dgsdg_post_process(
        args.input_file,
        args.output_dir,
        seed=args.seed,
    )
