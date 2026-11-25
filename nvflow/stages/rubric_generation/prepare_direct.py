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

"""Prepare simple or already-paired normalized records for rubric generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import METHODS, read_jsonl


def prepare_direct(
    *, input_file: str | Path, output_file: str | Path, method: str
) -> dict[str, int]:
    """Select the canonical fields consumed by the method-specific prompt."""

    if method not in METHODS:
        raise ValueError(f"method must be one of {sorted(METHODS)}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output_path.open("w", encoding="utf-8") as stream:
        for _, record in read_jsonl(input_file):
            total += 1
            if method == "openrubric_contrastive":
                preferred = record.get("preferred_answer")
                rejected = record.get("rejected_answer")
                if not isinstance(preferred, str) or not preferred.strip():
                    raise ValueError(f"record {record.get('id')} has no preferred_answer")
                if not isinstance(rejected, str) or not rejected.strip():
                    raise ValueError(f"record {record.get('id')} has no rejected_answer")
                if preferred.strip() == rejected.strip():
                    raise ValueError(f"record {record.get('id')} has an equivalent direct pair")
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    if not total:
        raise ValueError("no records found while preparing rubric method input")
    return {"total": total, "prepared": total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", "--input_file", required=True)
    parser.add_argument("--output-file", "--output_file", required=True)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_direct(
                input_file=args.input_file,
                output_file=args.output_file,
                method=args.method,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
