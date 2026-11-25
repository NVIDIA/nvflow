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
"""Utility functions for handling metadata in generated responses."""

# Common metadata keys to nest
METADATA_KEYS = [
    "num_generated_tokens",
    "reasoning_content",
    "finish_reason",
    "serialized_output",
    "generation_start_time",
    "generation_end_time",
    "generation_time",
    "generation",
]


def nest_metadata_fields(entry, metadata_field_name):
    """Nest metadata fields under a specific key for later debugging.

    Args:
        entry: Dictionary containing the record data
        metadata_field_name: Name of the nested key to store metadata
                            (e.g., "generate_questions_metadata", "generate_answers_metadata")

    Returns:
        Modified entry with metadata nested under metadata_field_name
    """
    metadata = {}

    # Extract metadata keys that exist in the entry
    for key in METADATA_KEYS:
        if key in entry:
            metadata[key] = entry[key]

    # Only add metadata dict if we found any keys
    if metadata:
        entry[metadata_field_name] = metadata

    return entry
