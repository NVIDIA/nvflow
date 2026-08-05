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

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

from nvflow.utils import setup_logger

logger = setup_logger(__name__)


def shuffle_file(input_file: str | Path, random_seed: int) -> int:
    """Shuffle the JSONL file at ``input_file`` in place, atomically.

    Crash semantics: writes the shuffled content to ``{path}.tmp`` and
    then renames it over ``{path}`` via :func:`os.replace`.  POSIX
    guarantees the rename is atomic on the same mount, so a process
    killed at any point sees one of two states:

      1. Original ``train.jsonl`` is fully intact (rename did not
         happen yet).
      2. Shuffled ``train.jsonl`` is fully written (rename completed).

    Without this, an interrupted in-place rewrite would leave a
    partially-written ``train.jsonl`` -- step-5 ``collect_rollouts``
    would silently consume the corrupted file (``iter_jsonl`` drops
    the cut-off last record), giving an off-by-N rollout count that
    is invisible without manual auditing.

    Uses a deterministic seed so reruns produce the same order.  Reads
    the full file into memory (jsonl is line-oriented so this is safe
    for multi-GB files on the prepare_data Slurm node).  Preserves
    blank lines at EOF by operating on raw ``readlines()`` bytes.
    """
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    with path.open("rb") as f:
        lines = f.readlines()

    rng = random.Random(random_seed)
    rng.shuffle(lines)

    # Sibling tmp file (same directory) -- ``os.replace`` is atomic
    # only when source and destination share a mount point.
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("wb") as f:
            f.write(b"".join(lines))
        os.replace(tmp_path, path)
    except BaseException:
        # Clean up the tmp file on any failure path (KeyboardInterrupt,
        # disk-full IOError, etc.) so reruns start from a clean state.
        # ``replace`` is the last operation, so if we got here either
        # the write or the replace failed -- in both cases the original
        # ``path`` is still untouched and the tmp may be partial.
        if tmp_path.exists():
            tmp_path.unlink()
        raise

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

    try:
        n = shuffle_file(args.input_file, args.random_seed)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1
    logger.info("Shuffled %d rows with seed=%d -> %s", n, args.random_seed, args.input_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
