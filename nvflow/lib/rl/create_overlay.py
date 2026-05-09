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
"""Create a symlinked model overlay directory with a patched ``config.json``.

All files from the original model directory are symlinked into the
overlay; only ``config.json`` is replaced with a real file that merges
the original config with the supplied overrides.

The overlay name is expected to be content-addressed (caller provides
it), and a marker file (``.overlay_overrides.json``) tracks the current
overrides so the overlay is only recreated when they change.

Standalone script that runs inside the Slurm container with
``PYTHONPATH=/workspace``.

Usage::

    python3 -m nvflow.lib.rl.create_overlay \\
        --model-path /hf_models/Qwen/Qwen3-4B \\
        --overlay-path /hf_models/Qwen/Qwen3-4B-overlay-a1b2c3d4e5f6 \\
        --overrides '{"rope_scaling": {"rope_type": "yarn", "factor": 3.2, "original_max_position_embeddings": 40960}}'
"""

import argparse
import json
import os
import sys

from nvflow.utils import setup_logger

logger = setup_logger(__name__)


def create_overlay(model_path: str, overlay_path: str, overrides: dict) -> None:
    """Create or verify a symlinked model overlay directory."""
    marker = os.path.join(overlay_path, ".overlay_overrides.json")
    expected = json.dumps(overrides, sort_keys=True)

    if os.path.exists(marker):
        with open(marker) as f:
            if f.read() == expected:
                logger.info("Model overlay up-to-date: %s", overlay_path)
                return

    os.makedirs(overlay_path, exist_ok=True)

    for name in os.listdir(model_path):
        link = os.path.join(overlay_path, name)
        if os.path.exists(link) or os.path.islink(link):
            os.unlink(link)
        target = os.path.join(model_path, name)
        os.symlink(os.path.relpath(target, overlay_path), link)

    cfg_path = os.path.join(overlay_path, "config.json")
    if os.path.islink(cfg_path):
        os.unlink(cfg_path)

    with open(os.path.join(model_path, "config.json")) as f:
        cfg = json.load(f)
    cfg.update(overrides)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    with open(marker, "w") as f:
        f.write(expected)

    logger.info("Created model overlay: %s", overlay_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a symlinked model overlay with patched config.json",
    )
    parser.add_argument("--model-path", required=True, help="Path to the original model directory")
    parser.add_argument("--overlay-path", required=True, help="Path for the overlay directory")
    parser.add_argument("--overrides", required=True, help="JSON string of HF config overrides")
    args = parser.parse_args()

    try:
        overrides = json.loads(args.overrides)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in --overrides: %s", e)
        sys.exit(1)

    create_overlay(args.model_path, args.overlay_path, overrides)
