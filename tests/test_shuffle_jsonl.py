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
"""Tests for shuffle_jsonl: determinism, content equivalence, atomicity.

The atomicity test is the load-bearing one: ``prepare_data`` runs this
script as a post-pass on the multi-GB ``train.jsonl`` produced by
``ng_prepare_data``.  Without atomic writes, an interrupted slurm job
(timeout, OOM, manual cancel) would leave a partially-written
``train.jsonl`` that step-5 ``collect_rollouts`` silently consumes
(``iter_jsonl`` drops the cut-off last record), giving an off-by-N
rollout count that is invisible without manual auditing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from nvflow.recipes.finance.utils.rl.shuffle_jsonl import shuffle_file


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _make_records(n: int) -> list[dict]:
    """Distinct records keyed by uuid so we can detect order changes."""
    return [{"uuid": f"uuid-{i:04d}", "problem": f"problem {i}", "idx": i} for i in range(n)]


def test_shuffle_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    """Same seed must produce the same order across runs.

    Pins the contract that ``random_seed`` controls the order
    deterministically -- without this, debugging a rollout-distribution
    issue would require capturing the shuffled file at job time.
    """
    records = _make_records(100)
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"
    _write_jsonl(path_a, records)
    _write_jsonl(path_b, records)

    shuffle_file(path_a, random_seed=42)
    shuffle_file(path_b, random_seed=42)

    assert _read_jsonl(path_a) == _read_jsonl(path_b)


def test_different_seeds_produce_different_orders(tmp_path: Path) -> None:
    """Anti-bug guard: a stuck rng (e.g. seed silently ignored) would
    return the same order regardless of seed.  100 records is enough
    that two random seeds match by accident only with vanishing
    probability (1/100! ≈ 10^-158).
    """
    records = _make_records(100)
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"
    _write_jsonl(path_a, records)
    _write_jsonl(path_b, records)

    shuffle_file(path_a, random_seed=42)
    shuffle_file(path_b, random_seed=43)

    assert _read_jsonl(path_a) != _read_jsonl(path_b), (
        "different seeds produced the same order -- seed is being ignored"
    )


def test_shuffle_preserves_record_set(tmp_path: Path) -> None:
    """Content equivalence: only order changes, no records added/lost.

    Pins that the shuffle is a permutation of the input set.  Without
    this, a regression that drops blank lines / EOF newlines / treats
    JSONL as text rather than bytes would silently corrupt the dataset.
    """
    records = _make_records(50)
    path = tmp_path / "in.jsonl"
    _write_jsonl(path, records)

    n = shuffle_file(path, random_seed=42)
    out = _read_jsonl(path)

    assert n == 50
    assert len(out) == 50
    # Set equivalence (uuids), not list equivalence -- order is expected
    # to differ.
    assert {r["uuid"] for r in out} == {r["uuid"] for r in records}
    # And the order DID change (probability of accidental fixed-point
    # for n=50 is ~1/n! ≈ 3 * 10^-65).
    assert [r["uuid"] for r in out] != [r["uuid"] for r in records]


def test_shuffle_preserves_byte_level_content(tmp_path: Path) -> None:
    """Round-trip the bytes of each line, not just the parsed dict.

    ``ng_prepare_data`` writes via orjson; ``shuffle_jsonl`` operates
    on raw ``readlines()`` bytes precisely so the orjson key-ordering
    and float formatting reach step-5 unmodified.  This test pins
    that contract: a regression that re-serialises records via stdlib
    ``json`` would silently change byte-level output even though
    ``json.loads(...)`` round-trips would still pass.
    """
    records = _make_records(20)
    path = tmp_path / "in.jsonl"
    _write_jsonl(path, records)
    original_lines = path.read_bytes().splitlines(keepends=True)

    shuffle_file(path, random_seed=42)
    shuffled_lines = path.read_bytes().splitlines(keepends=True)

    # Same set of byte-strings, just reordered.
    assert sorted(original_lines) == sorted(shuffled_lines), (
        "byte-level content changed: the shuffle re-serialised records "
        "instead of treating lines as opaque bytes"
    )


def test_atomic_write_leaves_original_on_failure(tmp_path: Path) -> None:
    """Pin the load-bearing atomicity contract.

    Simulates a crash by patching ``os.replace`` to raise.  A
    non-atomic implementation would have already truncated the
    original via ``open(path, "wb")`` and would leave a corrupted
    file behind.  The atomic implementation writes to ``{path}.tmp``
    first, so the original is untouched until the rename succeeds.

    Why this matters operationally: prepare_data runs in an ~11-hour
    slurm job; the shuffle phase is a ~2-minute window where a
    timeout / preemption / OOM kill could land.  An interrupted
    in-place rewrite produces silent corruption: step-5
    ``collect_rollouts`` reads via ``iter_jsonl(on_error="skip")``,
    which drops the cut-off last record and proceeds with an
    off-by-N rollout count that no health check catches.
    """
    records = _make_records(50)
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, records)
    original_bytes = path.read_bytes()

    # Patch ``os.replace`` only inside the shuffle module so the test
    # framework's own usage is untouched.  Raises after the tmp file
    # has been written but before it replaces the original -- this is
    # the exact failure window the atomic-rename protects against.
    with mock.patch(
        "nvflow.recipes.finance.utils.rl.shuffle_jsonl.os.replace",
        side_effect=OSError("simulated crash mid-rename"),
    ):
        with pytest.raises(OSError, match="simulated crash"):
            shuffle_file(path, random_seed=42)

    # Original content must be byte-identical -- not just parseable.
    # A non-atomic implementation would have truncated this file.
    assert path.read_bytes() == original_bytes, (
        "atomic-write contract violated: original file was modified before the rename completed"
    )

    # The cleanup-on-failure path must remove the tmp file so reruns
    # start from a clean state (avoids confusing operators who see a
    # stale ``.tmp`` and wonder if a previous run is still in flight).
    tmp_path_file = path.with_name(path.name + ".tmp")
    assert not tmp_path_file.exists(), f"tmp file leaked on failure path: {tmp_path_file}"


def test_missing_input_file_raises_clean_exit(tmp_path: Path) -> None:
    """A non-existent input must fail loudly with exit code 1.

    Pins the ``FileNotFoundError -> main() -> exit 1`` translation
    so a regression that lets the stack trace bubble up (or returns
    0 on missing input) gets caught at the test layer.
    """
    cmd = [
        sys.executable,
        "-m",
        "nvflow.recipes.finance.utils.rl.shuffle_jsonl",
        "--input_file",
        str(tmp_path / "does_not_exist.jsonl"),
        "--random_seed",
        "42",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    assert result.returncode == 1, f"expected exit 1, got {result.returncode}"
    assert "does not exist" in result.stderr or "does not exist" in result.stdout


def test_main_smoke_via_subprocess(tmp_path: Path) -> None:
    """End-to-end smoke test through ``python -m`` invocation.

    Mirrors how ``prepare_data`` actually invokes the shuffle (via
    ``build_python_cmd`` which renders ``python3 -m
    nvflow.recipes.finance.utils.rl.shuffle_jsonl --input_file ...``).
    Catches argparse-level regressions that unit-level tests miss.
    """
    records = _make_records(30)
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, records)

    cmd = [
        sys.executable,
        "-m",
        "nvflow.recipes.finance.utils.rl.shuffle_jsonl",
        "--input_file",
        str(path),
        "--random_seed",
        "7",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    out = _read_jsonl(path)
    assert len(out) == 30
    assert {r["uuid"] for r in out} == {r["uuid"] for r in records}
    # The completion sentinel log line is the operator's signal that
    # the shuffle ran -- pin it so a silent regression is loud.
    assert (
        "Shuffled 30 rows with seed=7" in result.stderr
        or "Shuffled 30 rows with seed=7" in result.stdout
    )
