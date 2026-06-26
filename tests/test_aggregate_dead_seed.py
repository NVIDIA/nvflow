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
"""Regression tests for the dead-seed policy in cross-seed aggregation (NV-5).

Policy: a rollout seed that dies / yields no usable data is skipped with a
logged warning and aggregation proceeds over the survivors, as long as at
least one seed survives.  A total wipe-out (zero usable seeds) fails loudly
(``SystemExit``), so pass@k is never silently computed over nothing.
"""

import json

import pytest

from nvflow.recipes.finance.utils.rl.aggregate_seeds import aggregate


def _write_seed(d, seed, rows):
    p = d / f"output-rs{seed}.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def _rows(uuid, reward):
    return [{"uuid": uuid, "reward": reward, "question_type": "qa", "question": "q"}]


class TestDeadSeedPolicy:
    def test_all_seeds_survive(self, tmp_path):
        _write_seed(tmp_path, 0, _rows("u1", 1.0))
        _write_seed(tmp_path, 1, _rows("u1", 0.0))
        out = tmp_path / "agg"
        aggregate(str(tmp_path), str(out), expected_num_seeds=2)
        metrics = json.loads((out / "metrics.json").read_text())
        assert metrics["num_seeds"] == 2

    def test_partial_dead_seed_warns_and_proceeds(self, tmp_path, capsys):
        # Two seed files but launcher expected three -> one seed died.
        _write_seed(tmp_path, 0, _rows("u1", 1.0))
        _write_seed(tmp_path, 1, _rows("u1", 0.0))
        out = tmp_path / "agg"
        aggregate(str(tmp_path), str(out), expected_num_seeds=3)
        captured = capsys.readouterr()
        assert "dead-seed" in captured.err
        metrics = json.loads((out / "metrics.json").read_text())
        # Aggregation proceeded over the 2 survivors.
        assert metrics["num_seeds"] == 2

    def test_empty_seed_file_is_dropped(self, tmp_path, capsys):
        _write_seed(tmp_path, 0, _rows("u1", 1.0))
        (tmp_path / "output-rs1.jsonl").write_text("")  # dead/empty seed
        out = tmp_path / "agg"
        aggregate(str(tmp_path), str(out), expected_num_seeds=2)
        captured = capsys.readouterr()
        assert "empty seed file" in captured.err
        metrics = json.loads((out / "metrics.json").read_text())
        assert metrics["num_seeds"] == 1

    def test_all_seeds_dead_fails_loudly(self, tmp_path):
        # Every seed file is empty -> total wipe-out.
        (tmp_path / "output-rs0.jsonl").write_text("")
        (tmp_path / "output-rs1.jsonl").write_text("")
        out = tmp_path / "agg"
        with pytest.raises(SystemExit) as exc:
            aggregate(str(tmp_path), str(out), expected_num_seeds=2)
        assert exc.value.code == 1

    def test_no_uuid_rows_fails_loudly(self, tmp_path):
        # Seed has rows but none are uuid-keyed -> no usable data.
        _write_seed(tmp_path, 0, [{"reward": 1.0}])
        out = tmp_path / "agg"
        with pytest.raises(SystemExit) as exc:
            aggregate(str(tmp_path), str(out), expected_num_seeds=1)
        assert exc.value.code == 1
