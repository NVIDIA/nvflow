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

"""Regression tests for complete difficulty-filter sample groups."""

from __future__ import annotations

import argparse
import json

import pytest
from nvhopchain.utils import hopchain_difficulty_filter_postprocess as postprocess


def _sample(query_id: str, sample_index: int) -> dict:
    return {
        "generation": "<final_answer>2</final_answer>",
        "_metadata": {
            "query_id": query_id,
            "sample_index": sample_index,
            "consensus_normalized_answer": "2",
            "record": {"query_id": query_id},
        },
    }


def _run_postprocess(tmp_path, monkeypatch, samples: list[dict], *, k: int) -> None:
    input_path = tmp_path / "output.jsonl"
    output_dir = tmp_path / "filtered"
    summary_path = tmp_path / "summary.json"
    input_path.write_text("\n".join(json.dumps(sample) for sample in samples) + "\n")
    monkeypatch.setattr(
        postprocess,
        "parse_args",
        lambda: argparse.Namespace(
            input=str(input_path),
            output_dir=str(output_dir),
            summary=str(summary_path),
            k=k,
            min_pass_rate=0.0,
            max_pass_rate=1.0,
        ),
    )
    postprocess.main()


def test_difficulty_filter_rejects_incomplete_sample_group(tmp_path, monkeypatch) -> None:
    with pytest.raises(
        ValueError,
        match=r"exactly 2 distinct sample indexes.*missing indexes=\[1\]",
    ):
        _run_postprocess(tmp_path, monkeypatch, [_sample("query-1", 0)], k=2)


def test_difficulty_filter_rejects_duplicate_sample_indexes(tmp_path, monkeypatch) -> None:
    with pytest.raises(
        ValueError,
        match=r"exactly 2 distinct sample indexes.*duplicate indexes=\[0\]",
    ):
        _run_postprocess(
            tmp_path,
            monkeypatch,
            [_sample("query-1", 0), _sample("query-1", 0)],
            k=2,
        )


def test_difficulty_filter_uses_configured_k_for_complete_group(tmp_path, monkeypatch) -> None:
    _run_postprocess(
        tmp_path,
        monkeypatch,
        [_sample("query-1", 0), _sample("query-1", 1)],
        k=2,
    )

    summary = json.loads((tmp_path / "summary.json").read_text())
    result = json.loads((tmp_path / "filtered/final_output.jsonl").read_text())
    assert summary["pass_rate_histogram"] == {"2/2": 1}
    assert result["difficulty_pass_rate"] == 1.0
