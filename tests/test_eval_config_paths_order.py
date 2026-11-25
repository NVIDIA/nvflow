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
"""Gym merges ``config_paths`` later-wins, so an overlay listed before the Gym
config it customises loads, loses the merge, and scores against Gym's defaults.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_BASE = REPO_ROOT / "projects/finance/workflows/eval/base.yaml"

# Overlays are absolute container paths; Gym's own configs are relative.
OVERLAY_PREFIX = "/workspace/"


def _catalog() -> dict:
    return yaml.safe_load(EVAL_BASE.read_text())["benchmarks"]


def test_overlays_follow_the_gym_configs_they_customise():
    for name, benchmark in _catalog().items():
        paths = benchmark.get("config_paths") or []
        overlays = [i for i, p in enumerate(paths) if p.startswith(OVERLAY_PREFIX)]
        gym_configs = [i for i, p in enumerate(paths) if not p.startswith(OVERLAY_PREFIX)]
        if not overlays or not gym_configs:
            continue
        assert min(overlays) > max(gym_configs), (
            f"{name} lists an overlay before a stock Gym config: {paths}"
        )
