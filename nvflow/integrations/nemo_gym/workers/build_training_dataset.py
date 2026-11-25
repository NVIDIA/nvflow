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
"""Build profiled and training datasets from a native Gym reward profile."""

from __future__ import annotations

import argparse
import json

from nvflow.integrations.nemo_gym.reward_profiles import (
    DatasetSelectionSpec,
    build_training_dataset,
    parse_dataset_selection_spec,
)


def _load_spec(value: str) -> DatasetSelectionSpec:
    raw = json.loads(value)
    if not isinstance(raw, dict):
        raise ValueError("dataset selection spec must be a JSON object")
    return parse_dataset_selection_spec(raw)


def _load_overrides(value: str) -> dict:
    raw = json.loads(value)
    if not isinstance(raw, dict):
        raise ValueError("responses_create_params overrides must be a JSON object")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--materialized-inputs")
    parser.add_argument("--rollouts")
    parser.add_argument("--rollout-completion")
    parser.add_argument("--profiled-output", required=True)
    parser.add_argument("--training-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--completion-output", required=True)
    parser.add_argument("--diagnostics-output", required=True)
    parser.add_argument("--spec", default="{}")
    parser.add_argument("--responses-create-params-overrides", default="{}")
    args = parser.parse_args()
    report = build_training_dataset(
        args.source,
        args.profile,
        args.profiled_output,
        args.training_output,
        args.report_output,
        args.completion_output,
        args.diagnostics_output,
        _load_spec(args.spec),
        materialized_inputs_path=args.materialized_inputs,
        rollouts_path=args.rollouts,
        rollout_completion_path=args.rollout_completion,
        responses_create_params_overrides=_load_overrides(args.responses_create_params_overrides),
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
