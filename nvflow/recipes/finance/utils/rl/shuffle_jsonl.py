#!/usr/bin/env python3
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
"""Deterministic in-place shuffle of a JSONL file.

Used as a post-pass in ``prepare_data`` so that ``collect_rollouts``
(which takes contiguous slices for ``num_chunks > 1`` and
``head -n N`` for ``max_num_samples``) sees a representative ordering
regardless of the per-filing clustering that SDG produces.

The previous pre-rollout ``train_validation_split`` stage (now moved
post-rollout) used to shuffle implicitly; this script restores that
property for the rollout input without reintroducing a pre-rollout
split.

Usage::

    python -m nvflow.recipes.finance.utils.rl.shuffle_jsonl \\
        --input_file /path/to/train.jsonl \\
        --random_seed 42
"""

import argparse
import random
import sys
from pathlib import Path

from nvflow.utils import setup_logger

logger = setup_logger(__name__)


def shuffle_file(input_file: str, random_seed: int) -> int:
    """Shuffle the JSONL file at ``input_file`` in place.

    Uses a deterministic seed so reruns produce the same order.  Reads
    the full file into memory (jsonl is line-oriented so this is safe
    for multi-GB files on the prepare_data Slurm node).  Preserves
    blank lines at EOF by operating on raw ``readlines()`` bytes.
    """
    path = Path(input_file)
    if not path.exists():
        logger.error("Input file does not exist: %s", path)
        sys.exit(1)

    with open(path, "rb") as f:
        lines = f.readlines()

    rng = random.Random(random_seed)
    rng.shuffle(lines)

    with open(path, "wb") as f:
        f.write(b"".join(lines))

    return len(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic in-place JSONL shuffle.")
    parser.add_argument("--input_file", required=True, help="JSONL file to shuffle in place.")
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Seed for deterministic shuffle (default: 42, matches train_validation_split).",
    )
    args = parser.parse_args()

    n = shuffle_file(args.input_file, args.random_seed)
    logger.info("Shuffled %d rows with seed=%d -> %s", n, args.random_seed, args.input_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
