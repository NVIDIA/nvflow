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
"""Generic weighted distribution sampling utilities for DG-SDG."""

import csv
import random

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def load_distribution(
    distribution_path: str,
    key_column: str = "item_section",
    count_column: str = "count",
) -> dict[str, int]:
    """Load a distribution CSV file into a ``{key: count}`` dictionary.

    The column names are parameterized so this stays domain-agnostic: finance
    uses the default ``item_section`` / ``count`` columns, but any domain can
    point at its own column layout (e.g. ``topic`` / ``weight``).

    Args:
        distribution_path: Path to the CSV file
        key_column: Name of the column holding the distribution keys
        count_column: Name of the column holding the integer weights

    Returns:
        dict mapping each key to its integer count/weight
    """
    if not HAS_PANDAS:
        dist = {}
        with open(distribution_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                dist[row[key_column]] = int(row[count_column])
        return dist

    df = pd.read_csv(distribution_path)
    return {row[key_column]: row[count_column] for _, row in df.iterrows()}


def weighted_random_choice(distribution_dict: dict[str, int]) -> str:
    """Choose an item based on weighted distribution.

    Args:
        distribution_dict: dict mapping item names to weight counts

    Returns:
        A randomly chosen item key, weighted by its count
    """
    items = sorted(distribution_dict.keys())
    weights = [distribution_dict[item] for item in items]
    total_weight = sum(weights)
    if total_weight == 0:
        return random.choice(items)
    return random.choices(items, weights=weights, k=1)[0]
