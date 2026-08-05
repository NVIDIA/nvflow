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
"""Evaluate-stage enrich: identical to ``enrich_rollouts`` but writes the
extracted text under ``evaluate_generation`` (what ``evaluate.py`` parses).

``rollout()``'s enrich hook is a fixed 2-positional CLI
(``python -m <mod> <input> <merged>``), so the per-stage ``generation_key``
cannot be passed through it.  This thin module pins it for the evaluate stage.

    python -m nvflow.lib.sdg.document_grounded.enrich_rollouts_evaluate <input.jsonl> <rollouts.jsonl>
"""

from __future__ import annotations

import argparse
import sys

from nvflow.lib.sdg.document_grounded.enrich_rollouts import enrich


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file")
    parser.add_argument("rollouts_file")
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    args = parser.parse_args(argv)
    enrich(
        args.input_file,
        args.rollouts_file,
        generation_key="evaluate_generation",
        strict=args.strict,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
