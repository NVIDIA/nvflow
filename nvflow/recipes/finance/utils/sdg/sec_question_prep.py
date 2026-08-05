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
"""Thin CLI wrapper: construct_question_generate_input with SEC context_builder.

Usage:
    python sec_question_prep.py --input_folder /path/to/jsonl --output_file /path/to/output.jsonl
"""

import argparse
from pathlib import Path

from nvflow.lib.sdg.document_grounded.preprocess import construct_question_generate_input
from nvflow.recipes.finance.utils.sdg.sec_callbacks import sec_context_builder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare question generation input with SEC context."
    )
    parser.add_argument("--input_folder", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    args = parser.parse_args()

    construct_question_generate_input(
        args.input_folder,
        args.output_file,
        context_builder=sec_context_builder,
    )
