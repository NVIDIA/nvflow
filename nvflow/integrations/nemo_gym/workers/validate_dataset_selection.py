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
"""Validate an atomically published profiled/training dataset pair."""

from __future__ import annotations

import argparse
import json

from nvflow.integrations.nemo_gym.reward_profiles import validate_dataset_completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiled", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--completion", required=True)
    parser.add_argument("--diagnostics")
    parser.add_argument("--selection-sha256")
    args = parser.parse_args()

    result = validate_dataset_completion(
        args.profiled,
        args.train,
        args.report,
        args.completion,
        diagnostics_path=args.diagnostics,
        expected_selection_sha256=args.selection_sha256,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
