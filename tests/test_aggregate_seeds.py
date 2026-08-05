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
"""Tests for nvflow.recipes.finance.utils.rl.aggregate_seeds.

Pins the F2 contract: when ``--expected-seeds N`` is provided, missing
per-seed rollout files surface as a loud RuntimeError instead of
silently shrinking ``num_seeds`` in metrics.json.

This is the second line of defense against the silent-success cascade
documented in the F1 commit message.  The cluster's default Slurm dep
type is ``afterany`` (see cluster_configs/template-slurm.yaml note on
dependency_type), so a FAILED upstream merge does NOT prevent
aggregate from running.  Without this validation, aggregate would
glob whatever ``output-rs*.jsonl`` files happened to be present and
proceed with the diminished set, producing partial difficulty data
that filter then passes through silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nvflow.recipes.finance.utils.rl.aggregate_seeds import aggregate


def _write_seed_file(path: Path, num_rows: int = 3) -> None:
    """Emit a minimal rollout file with the fields aggregate inspects."""
    rows = [
        {"uuid": f"q-{i}", "reward": float(i % 2), "question_type": "test"} for i in range(num_rows)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_aggregate_passes_when_expected_seeds_matches_found(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    for s in range(3):
        _write_seed_file(rollout_dir / f"output-rs{s}.jsonl")
    out_dir = tmp_path / "agg"

    aggregate(str(rollout_dir), str(out_dir), expected_seeds=3)

    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert metrics["num_seeds"] == 3


def test_aggregate_raises_when_seed_missing(tmp_path: Path) -> None:
    """Production scenario: rs0's merge job FAILED so output-rs0.jsonl is
    absent, but rs1 and rs2 succeeded.  Without --expected-seeds the
    pre-F2 behaviour was to silently set num_seeds=2 and exit 0.  With
    F2 plumbed through (build_aggregate_cmd always passes
    p.num_random_seeds), this raises so Slurm marks the aggregate job
    FAILED -- visible signal that an upstream merge dropped a seed.
    """
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    # Only seeds 1 and 2 -- seed 0 missing.
    _write_seed_file(rollout_dir / "output-rs1.jsonl")
    _write_seed_file(rollout_dir / "output-rs2.jsonl")
    out_dir = tmp_path / "agg"

    with pytest.raises(RuntimeError, match="Expected 3 per-seed rollout files"):
        aggregate(str(rollout_dir), str(out_dir), expected_seeds=3)

    # No partial output should be written when the precondition fails.
    assert not (out_dir / "metrics.json").exists()
    assert not (out_dir / "summary.txt").exists()


def test_aggregate_raises_when_extra_seeds_present(tmp_path: Path) -> None:
    """Symmetric guard: extra files (e.g. a stale output-rs7.jsonl from a
    larger previous run) also fail the precondition.  Otherwise an
    operator could silently aggregate a mix of fresh + stale data.
    """
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    for s in range(5):  # 5 seeds present but config expects 3
        _write_seed_file(rollout_dir / f"output-rs{s}.jsonl")
    out_dir = tmp_path / "agg"

    with pytest.raises(RuntimeError, match="Expected 3 per-seed rollout files"):
        aggregate(str(rollout_dir), str(out_dir), expected_seeds=3)


def test_aggregate_back_compat_no_expected_seeds(tmp_path: Path) -> None:
    """Default expected_seeds=None preserves pre-F2 behaviour: aggregate
    accepts whatever per-seed files exist.  External callers (verify.py
    rejudge path, ad-hoc scripts) that don't know the expected count
    must continue to work.
    """
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    _write_seed_file(rollout_dir / "output-rs1.jsonl")
    _write_seed_file(rollout_dir / "output-rs2.jsonl")
    out_dir = tmp_path / "agg"

    aggregate(str(rollout_dir), str(out_dir))  # no expected_seeds kwarg

    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert metrics["num_seeds"] == 2  # accepts the diminished set


def test_aggregate_excludes_chunk_and_async_files(tmp_path: Path) -> None:
    """The seed-count comparison must be on canonical merged outputs
    only.  Per-chunk intermediates (output-rs0_chunk_0.jsonl) and
    in-flight async files (output-rs0.jsonl-async) must be excluded
    BEFORE the precondition check, otherwise stale intermediates
    could mask a truly-missing seed.
    """
    rollout_dir = tmp_path / "rollout"
    rollout_dir.mkdir()
    _write_seed_file(rollout_dir / "output-rs1.jsonl")
    _write_seed_file(rollout_dir / "output-rs2.jsonl")
    # Spurious intermediates that happen to glob-match.
    _write_seed_file(rollout_dir / "output-rs0_chunk_0.jsonl")
    _write_seed_file(rollout_dir / "output-rs0.jsonl-async")
    out_dir = tmp_path / "agg"

    with pytest.raises(RuntimeError, match="found 2"):
        aggregate(str(rollout_dir), str(out_dir), expected_seeds=3)
