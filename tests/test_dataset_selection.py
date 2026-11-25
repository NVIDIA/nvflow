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
"""Shared native Gym reward-profile normalization and selection."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import pytest

import nvflow.integrations.nemo_gym.reward_profiles as reward_profiles
from nvflow.integrations.nemo_gym.native_contract import NativeGymOutputLayout
from nvflow.integrations.nemo_gym.native_sharding import (
    publish_rollout_completion,
)
from nvflow.integrations.nemo_gym.reward_profiles import (
    DatasetSelectionSpec,
    RewardProfileError,
    build_training_dataset,
    parse_dataset_selection_spec,
    validate_dataset_completion,
)
from nvflow.integrations.nemo_gym.workers.build_training_dataset import _load_spec


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, allow_nan=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _publish(
    source: Path,
    profile: Path,
    output: Path,
    spec: DatasetSelectionSpec,
    **kwargs: Any,
) -> dict[str, Any]:
    layout = NativeGymOutputLayout.collection(output)
    return build_training_dataset(
        source,
        profile,
        layout.profiled_dataset,
        layout.training_dataset,
        layout.dataset_selection_report,
        layout.dataset_completion,
        layout.unprofiled_samples,
        spec,
        **kwargs,
    )


def _validate_publication(output: Path) -> dict[str, Any]:
    layout = NativeGymOutputLayout.collection(output)
    return validate_dataset_completion(
        layout.profiled_dataset,
        layout.training_dataset,
        layout.dataset_selection_report,
        layout.dataset_completion,
        diagnostics_path=layout.unprofiled_samples,
    )


def _profile(
    uuid: str,
    rewards: list[float],
    *,
    task_index: int = 0,
    agent: str | None = None,
    rollout_indices: list[int] | None = None,
) -> dict:
    infos = []
    for index, reward in enumerate(rewards):
        rollout_index = (rollout_indices or list(range(len(rewards))))[index]
        info = {
            "_ng_task_index": task_index,
            "_ng_rollout_index": rollout_index,
            "reward": reward,
            "input_tokens": 10 + index,
            "output_tokens": 20 + index,
            "total_tokens": 30 + 2 * index,
        }
        infos.append(info)
    row = {
        "_ng_task_index": task_index,
        "sample": {
            "uuid": uuid,
            "responses_create_params": {"input": [{"role": "user", "content": uuid}]},
            "custom": uuid,
        },
        "num_rollouts": len(rewards),
        "expected_num_rollouts": len(rewards),
        "missing_num_rollouts": 0,
        "reward_profile_completion_pct": 100.0,
        "rollout_infos": infos,
        "mean/reward": statistics.fmean(rewards),
        "std/reward": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
        "min/reward": min(rewards),
        "max/reward": max(rewards),
    }
    if agent:
        row["agent_ref"] = {"name": agent}
    return row


def _profile_sample(uuid: str) -> dict[str, Any]:
    return {
        "uuid": uuid,
        "responses_create_params": {"input": [{"role": "user", "content": uuid}]},
        "custom": uuid,
    }


def _materialized(path: Path, rows: list[dict], repeats: int = 2) -> Path:
    expanded = []
    for task_index, row in enumerate(rows):
        for rollout_index in range(repeats):
            expanded.append(
                {
                    **row,
                    "_ng_task_index": task_index,
                    "_ng_rollout_index": rollout_index,
                }
            )
    _write(path, expanded)
    return path


def _completed_materialized(
    path: Path,
    rows: list[dict],
    repeats: int = 2,
) -> tuple[Path, Path, Path]:
    materialized = _materialized(path, rows, repeats)
    rollouts = path.with_name("rollouts.jsonl")
    marker = rollouts.with_name("completion.json")
    rollout_rows = [
        {
            "_ng_task_index": row["_ng_task_index"],
            "_ng_rollout_index": row["_ng_rollout_index"],
            "response": {"output": []},
            "reward": 0.0,
        }
        for row in (
            json.loads(line) for line in materialized.read_text().splitlines() if line.strip()
        )
    ]
    _write(rollouts, rollout_rows)
    publish_rollout_completion(materialized, rollouts, marker)
    return materialized, rollouts, marker


def test_synthetic_repeat_seeds_are_ignored_but_other_extra_body_fields_survive(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    output = tmp_path / "output"
    sample = _profile_sample("a")
    sample["responses_create_params"]["metadata"] = {"extra_body": json.dumps({"top_k": 20})}
    profiled = _profile("a", [0.0, 0.5, 1.0])
    profiled["sample"]["responses_create_params"]["metadata"] = {
        "extra_body": json.dumps({"top_k": 20, "seed": 0})
    }
    materialized_rows = []
    for rollout_index in range(3):
        row = json.loads(json.dumps(sample))
        row["_ng_task_index"] = 0
        row["_ng_rollout_index"] = rollout_index
        row["responses_create_params"]["metadata"]["extra_body"] = json.dumps(
            {"top_k": 20, "seed": rollout_index}
        )
        materialized_rows.append(row)
    materialized = tmp_path / "materialized.jsonl"
    rollouts = tmp_path / "rollouts.jsonl"
    marker = tmp_path / "completion.json"
    _write(source, [sample])
    _write(profile, [profiled])
    _write(materialized, materialized_rows)
    _write(
        rollouts,
        [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": rollout_index,
                "response": {"output": []},
                "reward": float(rollout_index),
            }
            for rollout_index in range(3)
        ],
    )
    publish_rollout_completion(materialized, rollouts, marker)

    _publish(
        source,
        profile,
        output,
        DatasetSelectionSpec(),
        materialized_inputs_path=materialized,
        rollouts_path=rollouts,
        rollout_completion_path=marker,
    )

    selected = json.loads((output / "train.jsonl").read_text())
    extra_body = json.loads(selected["responses_create_params"]["metadata"]["extra_body"])
    assert extra_body == {"top_k": 20}


def test_synthetic_seed_only_metadata_is_removed() -> None:
    sample = _profile_sample("a")
    sample["responses_create_params"]["metadata"] = {"extra_body": json.dumps({"seed": 2})}

    stable = reward_profiles._stable_sample(sample)

    assert "metadata" not in stable["responses_create_params"]


def test_generated_task_index_can_select_sources_without_uuid(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    output = tmp_path / "output"
    sample = {
        "question": "What was revenue?",
        "responses_create_params": {"input": [{"role": "user", "content": "What was revenue?"}]},
    }
    profiled = _profile("unused", [0.0, 1.0])
    profiled["sample"] = sample
    _write(source, [sample])
    _write(profile, [profiled])

    _publish(
        source,
        profile,
        output,
        DatasetSelectionSpec(join_key=None),
    )

    assert json.loads((output / "train.jsonl").read_text())["question"] == ("What was revenue?")


def test_configured_response_overrides_are_expected_profile_mutations(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    sample = _profile_sample("a")
    sample["responses_create_params"]["max_output_tokens"] = 1024
    profiled = _profile("a", [0.0, 1.0])
    profiled["sample"]["responses_create_params"]["max_output_tokens"] = 32768
    profiled["sample"]["responses_create_params"]["metadata"] = {
        "extra_body": json.dumps({"seed": 0})
    }
    _write(source, [sample])
    _write(profile, [profiled])

    with pytest.raises(
        RewardProfileError,
        match="responses_create_params.max_output_tokens",
    ):
        _publish(
            source,
            profile,
            tmp_path / "strict",
            DatasetSelectionSpec(),
        )

    output = tmp_path / "configured"
    _publish(
        source,
        profile,
        output,
        DatasetSelectionSpec(),
        responses_create_params_overrides={"max_output_tokens": 32768},
    )

    selected = json.loads((output / "train.jsonl").read_text())
    assert selected["responses_create_params"]["max_output_tokens"] == 32768
    assert "metadata" not in selected["responses_create_params"]


def test_nested_response_override_matches_gyms_shallow_merge(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    output = tmp_path / "output"
    sample = _profile_sample("a")
    sample["responses_create_params"]["metadata"] = {"purpose": "source"}
    profiled = _profile("a", [0.0, 1.0])
    profiled["sample"]["responses_create_params"]["metadata"] = {
        "request_id": "run",
        "extra_body": json.dumps({"seed": 0}),
    }
    _write(source, [sample])
    _write(profile, [profiled])

    _publish(
        source,
        profile,
        output,
        DatasetSelectionSpec(),
        responses_create_params_overrides={"metadata": {"request_id": "run"}},
    )

    selected = json.loads((output / "train.jsonl").read_text())
    assert selected["responses_create_params"]["metadata"] == {"request_id": "run"}


def test_run_level_skills_reference_is_preserved(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    output = tmp_path / "output"
    sample = _profile_sample("a")
    profiled = _profile("a", [0.0, 1.0])
    profiled["sample"]["skills_ref"] = {"name": "tools"}
    _write(source, [sample])
    _write(profile, [profiled])

    _publish(source, profile, output, DatasetSelectionSpec())

    selected = json.loads((output / "train.jsonl").read_text())
    assert selected["skills_ref"] == {"name": "tools"}


def test_normalizes_before_filter_and_publishes_atomic_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    output = tmp_path / "output"
    _write(
        source,
        [
            {
                "uuid": "a",
                "custom": "a",
                "_ng_task_index": 99,
                "_ng_rollout_index": 7,
            },
            {"uuid": "b"},
        ],
    )
    _write(
        profile,
        [
            _profile("a", [0.0, 1.0], task_index=0),
            _profile("b", [1.0, 1.0], task_index=1),
        ],
    )

    report = _publish(
        source,
        profile,
        output,
        DatasetSelectionSpec(
            mode="scalar_variance",
            min_std=1e-6,
            success_threshold=1.0,
            provenance={"policy_model": "policy-a"},
        ),
    )

    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert report["selected"] == 1
    assert rows[0]["uuid"] == "a"
    assert rows[0]["custom"] == "a"
    assert "_ng_task_index" not in rows[0]
    assert "_ng_rollout_index" not in rows[0]
    assert rows[0]["pass_rate"] == 0.5
    assert rows[0]["pass_rate_passed"] == 1
    assert rows[0]["pass_rate_total"] == 2
    assert rows[0]["input_tokens/std"] == 0.5
    assert rows[0]["difficulty_profile"]["reward_std"] == pytest.approx(2**-0.5)
    assert rows[0]["difficulty_profile"]["std_convention"] == "sample"
    assert rows[0]["difficulty_profile"]["provenance"]["policy_model"] == "policy-a"
    assert len(rows[0]["raw_rewards_and_lengths"]) == 2
    layout = NativeGymOutputLayout.collection(output)
    assert Path(layout.dataset_completion).is_file()
    marker = json.loads(Path(layout.dataset_completion).read_text())
    assert marker["profiled"]["rows"] == 2
    assert marker["train"]["rows"] == 1
    assert len(marker["source_sha256"]) == 64
    assert len(marker["profile_sha256"]) == 64
    assert len(marker["selection_sha256"]) == 64
    assert not (output / "difficulty.jsonl").exists()
    dataset_job_dir = Path(layout.dataset_job_dir)
    assert not list(dataset_job_dir.glob(".dataset-selection.*.sqlite3"))
    assert (dataset_job_dir / ".dataset-publication.lock").is_file()


def test_dataset_completion_validates_published_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    output = tmp_path / "output"
    _write(source, [{"uuid": "a"}])
    _write(profile, [_profile("a", [0.0, 1.0])])
    _publish(source, profile, output, DatasetSelectionSpec())

    marker = _validate_publication(output)

    assert marker["kind"] == "dataset_completion"
    assert marker["schema_version"] == 1
    assert marker["profiled"]["rows"] == 1
    assert marker["train"]["rows"] == 1
    assert marker["source_rows"] == 1


def test_dataset_completion_rejects_tampered_train(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    output = tmp_path / "output"
    _write(source, [{"uuid": "a"}])
    _write(profile, [_profile("a", [0.0, 1.0])])
    _publish(source, profile, output, DatasetSelectionSpec())
    train = output / "train.jsonl"
    train.write_text(train.read_text() + '{"uuid":"tampered"}\n', encoding="utf-8")

    with pytest.raises(RewardProfileError, match="row count mismatch"):
        _validate_publication(output)


def test_dataset_completion_rejects_same_row_count_hash_change(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    output = tmp_path / "output"
    _write(source, [{"uuid": "a"}])
    _write(profile, [_profile("a", [0.0, 1.0])])
    _publish(source, profile, output, DatasetSelectionSpec())
    train = output / "train.jsonl"
    train.write_bytes(train.read_bytes() + b" ")

    with pytest.raises(RewardProfileError, match="SHA-256"):
        _validate_publication(output)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "unsupported.*schema_version"),
        ("source_rows", 0, "completion.source_rows"),
        ("selection_sha256", None, "completion.selection_sha256"),
    ],
)
def test_dataset_completion_rejects_invalid_metadata(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    output = tmp_path / "output"
    _write(source, [{"uuid": "a"}])
    _write(profile, [_profile("a", [0.0, 1.0])])
    _publish(source, profile, output, DatasetSelectionSpec())
    marker_path = Path(NativeGymOutputLayout.collection(output).dataset_completion)
    marker = json.loads(marker_path.read_text())
    marker[field] = value
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(RewardProfileError, match=message):
        _validate_publication(output)


def test_dataset_completion_binds_selection_hash_to_report(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    output = tmp_path / "output"
    _write(source, [{"uuid": "a"}])
    _write(profile, [_profile("a", [0.0, 1.0])])
    _publish(source, profile, output, DatasetSelectionSpec())
    completion = Path(NativeGymOutputLayout.collection(output).dataset_completion)
    payload = json.loads(completion.read_text())
    payload["selection_sha256"] = "b" * 64
    completion.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RewardProfileError, match="selection report does not match"):
        _validate_publication(output)


def test_continuous_rewards_require_explicit_success_semantics(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    output = tmp_path / "output"
    _write(source, [{"uuid": "a"}])
    _write(profile, [_profile("a", [0.25, 0.75])])
    _publish(source, profile, output, DatasetSelectionSpec(mode="all"))
    row = json.loads((output / "train.jsonl").read_text())
    assert "pass_rate" not in row
    assert row["difficulty_profile"]["avg_reward"] == 0.5


def test_empty_selection_fails_without_publishing_train(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    output = tmp_path / "output"
    _write(source, [{"uuid": "a"}])
    _write(profile, [_profile("a", [1.0, 1.0])])

    with pytest.raises(RewardProfileError, match="retained no source rows"):
        _publish(
            source,
            profile,
            output,
            DatasetSelectionSpec(mode="scalar_variance", min_std=1e-6),
        )
    assert not (output / "train.jsonl").exists()
    assert not Path(NativeGymOutputLayout.collection(output).dataset_completion).exists()


def test_fanout_requires_an_agent_selection_for_uuid_join(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    _write(source, [{"uuid": "a"}])
    _write(
        profile,
        [
            _profile("a", [0.0, 1.0], agent="agent-a", rollout_indices=[0, 2]),
            _profile("a", [1.0, 1.0], agent="agent-b", rollout_indices=[1, 3]),
        ],
    )
    with pytest.raises(RewardProfileError, match="ambiguous duplicate"):
        _publish(source, profile, tmp_path / "out", DatasetSelectionSpec())

    _write(
        source,
        [{"uuid": "a", "agent_ref": {"name": "agent-a", "type": "old"}}],
    )
    report = _publish(
        source,
        profile,
        tmp_path / "selected",
        DatasetSelectionSpec(agent="agent-b"),
    )
    assert report["selected"] == 1
    row = json.loads((tmp_path / "selected" / "train.jsonl").read_text())
    assert row["agent_ref"]["name"] == "agent-b"


def test_changed_source_is_not_joined_to_stale_profile(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    _write(source, [{"uuid": "a", "custom": "changed"}])
    _write(profile, [_profile("a", [0.0, 1.0])])

    with pytest.raises(RewardProfileError, match="source data changed after rollout"):
        _publish(source, profile, tmp_path / "out", DatasetSelectionSpec())


def test_incomplete_or_nonfinite_profile_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    _write(source, [{"uuid": "a"}])
    incomplete = _profile("a", [0.0, 1.0])
    incomplete["missing_num_rollouts"] = 1
    _write(profile, [incomplete])
    with pytest.raises(RewardProfileError, match="incomplete"):
        _publish(source, profile, tmp_path / "out", DatasetSelectionSpec())

    invalid = _profile("a", [0.0, 1.0])
    invalid["rollout_infos"][0]["reward"] = float("nan")
    _write(profile, [invalid])
    with pytest.raises(RewardProfileError, match="finite"):
        _publish(source, profile, tmp_path / "out", DatasetSelectionSpec())


def test_profile_orders_raw_entries_by_rollout_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    output = tmp_path / "output"
    _write(source, [{"uuid": "a"}])
    _write(profile, [_profile("a", [0.2, 0.8], rollout_indices=[2, 0])])

    _publish(source, profile, output, DatasetSelectionSpec())

    selected = json.loads((output / "train.jsonl").read_text())
    assert [entry["_ng_rollout_index"] for entry in selected["raw_rewards_and_lengths"]] == [0, 2]
    assert [entry["reward"] for entry in selected["raw_rewards_and_lengths"]] == [0.8, 0.2]


def test_duplicate_rollout_identity_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    _write(source, [{"uuid": "a"}])
    _write(profile, [_profile("a", [0.0, 1.0], rollout_indices=[0, 0])])

    with pytest.raises(RewardProfileError, match="duplicate rollout index"):
        _publish(source, profile, tmp_path / "out", DatasetSelectionSpec())


def test_malformed_profile_json_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    _write(source, [{"uuid": "a"}])
    profile.write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(RewardProfileError, match="malformed JSON"):
        _publish(source, profile, tmp_path / "out", DatasetSelectionSpec())


def test_token_sum_mismatch_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    _write(source, [{"uuid": "a"}])
    row = _profile("a", [0.0, 1.0])
    row["rollout_infos"][0]["total_tokens"] = 999
    _write(profile, [row])

    with pytest.raises(RewardProfileError, match="total_tokens does not equal"):
        _publish(source, profile, tmp_path / "out", DatasetSelectionSpec())


@pytest.mark.parametrize(
    ("field", "message"), [("mean/reward", "native mean"), ("std/reward", "native std")]
)
def test_native_reward_statistics_must_match_rollouts(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    _write(source, [{"uuid": "a"}])
    row = _profile("a", [0.0, 1.0])
    row[field] = 99.0
    _write(profile, [row])

    with pytest.raises(RewardProfileError, match=message):
        _publish(source, profile, tmp_path / "out", DatasetSelectionSpec())


@pytest.mark.parametrize("location", ["profile", "rollout"])
def test_missing_native_identity_fails_closed(tmp_path: Path, location: str) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    _write(source, [{"uuid": "a"}])
    row = _profile("a", [0.0, 1.0])
    if location == "profile":
        row.pop("_ng_task_index")
    else:
        row["rollout_infos"][0].pop("_ng_rollout_index")
    _write(profile, [row])

    with pytest.raises(RewardProfileError, match="missing"):
        _publish(source, profile, tmp_path / "out", DatasetSelectionSpec())


def test_missing_token_usage_is_nullable_and_reports_coverage(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    output = tmp_path / "output"
    _write(source, [{"uuid": "a"}])
    row = _profile("a", [0.0, 1.0])
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        row["rollout_infos"][0].pop(field)
    _write(profile, [row])

    _publish(source, profile, output, DatasetSelectionSpec())

    selected = json.loads((output / "train.jsonl").read_text())
    assert selected["input_tokens/count"] == 1
    assert selected["input_tokens/coverage"] == 0.5
    assert selected["raw_rewards_and_lengths"][0]["input_tokens"] is None


def test_duplicate_source_and_destination_alias_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    profile = tmp_path / "profile.jsonl"
    _write(source, [{"uuid": "a"}, {"uuid": "a"}])
    _write(profile, [_profile("a", [0.0, 1.0])])
    with pytest.raises(RewardProfileError, match="duplicate source"):
        _publish(source, profile, tmp_path / "out", DatasetSelectionSpec())
    with pytest.raises(RewardProfileError, match="must be distinct"):
        _publish(source, profile, tmp_path, DatasetSelectionSpec())


def test_unprofiled_sample_policy_is_explicit(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    _write(source, [{"uuid": "a"}, {"uuid": "b"}])
    _write(profile, [_profile("a", [0.0, 1.0])])
    materialized, rollouts, marker = _completed_materialized(
        tmp_path / "materialized.jsonl",
        [_profile_sample("a")],
    )
    with pytest.raises(RewardProfileError, match="no reward profile"):
        _publish(source, profile, tmp_path / "error", DatasetSelectionSpec())
    with pytest.raises(RewardProfileError, match="requires materialized inputs"):
        _publish(
            source,
            profile,
            tmp_path / "unsafe-drop",
            DatasetSelectionSpec(unprofiled_samples="exclude"),
        )
    report = _publish(
        source,
        profile,
        tmp_path / "drop",
        DatasetSelectionSpec(unprofiled_samples="exclude"),
        materialized_inputs_path=materialized,
        rollouts_path=rollouts,
        rollout_completion_path=marker,
    )
    assert report["unprofiled"] == 1
    assert report["selected"] == 1

    separate_output = tmp_path / "separate"
    _publish(
        source,
        profile,
        separate_output,
        DatasetSelectionSpec(unprofiled_samples="separate"),
        materialized_inputs_path=materialized,
        rollouts_path=rollouts,
        rollout_completion_path=marker,
    )
    diagnostics = Path(NativeGymOutputLayout.collection(separate_output).unprofiled_samples)
    [diagnostic] = [json.loads(line) for line in diagnostics.read_text().splitlines()]
    assert diagnostic["reason"] == "outside_rollout_scope"


def test_include_policy_scrubs_stale_profile_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    output = tmp_path / "output"
    _write(
        source,
        [
            {"uuid": "a"},
            {
                "uuid": "b",
                "pass_rate": 1.0,
                "input_tokens/mean": 99,
                "raw_rewards_and_lengths": [{"reward": 1.0}],
            },
        ],
    )
    _write(profile, [_profile("a", [0.0, 1.0])])

    _publish(
        source,
        profile,
        output,
        DatasetSelectionSpec(unprofiled_samples="include"),
    )

    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    kept = next(row for row in rows if row["uuid"] == "b")
    assert kept["difficulty_profile"] == {"profiled": False}
    assert "pass_rate" not in kept
    assert "input_tokens/mean" not in kept
    assert "raw_rewards_and_lengths" not in kept


def test_include_policy_applies_response_overrides_to_unprofiled_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    output = tmp_path / "output"
    _write(source, [_profile_sample("a"), _profile_sample("b")])
    profiled = _profile("a", [0.0, 1.0])
    profiled["sample"]["responses_create_params"]["max_output_tokens"] = 32768
    _write(profile, [profiled])

    _publish(
        source,
        profile,
        output,
        DatasetSelectionSpec(unprofiled_samples="include"),
        responses_create_params_overrides={"max_output_tokens": 32768},
    )

    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert all(row["responses_create_params"]["max_output_tokens"] == 32768 for row in rows)


def test_scalar_variance_include_keeps_unprofiled_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    output = tmp_path / "output"
    _write(source, [{"uuid": "a"}, {"uuid": "b"}])
    _write(profile, [_profile("a", [0.0, 1.0])])

    _publish(
        source,
        profile,
        output,
        DatasetSelectionSpec(
            mode="scalar_variance",
            min_std=1e-6,
            unprofiled_samples="include",
        ),
    )

    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert {row["uuid"] for row in rows} == {"a", "b"}
    unprofiled = next(row for row in rows if row["uuid"] == "b")
    assert unprofiled["difficulty_profile"] == {"profiled": False}


def test_exclude_policy_rejects_profile_truncated_within_materialized_scope(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    rows = [_profile_sample("a"), _profile_sample("b")]
    _write(source, rows)
    _write(profile, [_profile("a", [0.0, 1.0])])
    materialized, rollouts, marker = _completed_materialized(
        tmp_path / "materialized.jsonl",
        rows,
    )

    with pytest.raises(RewardProfileError, match="do not exactly match"):
        _publish(
            source,
            profile,
            tmp_path / "out",
            DatasetSelectionSpec(unprofiled_samples="exclude"),
            materialized_inputs_path=materialized,
            rollouts_path=rollouts,
            rollout_completion_path=marker,
        )


def test_exclude_policy_rejects_tampered_completion_scope(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    sample = _profile_sample("a")
    _write(source, [{"uuid": "a"}])
    _write(profile, [_profile("a", [0.0, 1.0])])
    materialized, rollouts, marker = _completed_materialized(
        tmp_path / "materialized.jsonl",
        [sample],
    )
    with materialized.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    **sample,
                    "_ng_task_index": 1,
                    "_ng_rollout_index": 0,
                }
            )
            + "\n"
        )

    with pytest.raises(RewardProfileError, match="invalid rollout completion evidence"):
        _publish(
            source,
            profile,
            tmp_path / "out",
            DatasetSelectionSpec(unprofiled_samples="exclude"),
            materialized_inputs_path=materialized,
            rollouts_path=rollouts,
            rollout_completion_path=marker,
        )


def test_failed_publish_preserves_existing_train_and_omits_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    output = tmp_path / "output"
    output.mkdir()
    existing = output / "train.jsonl"
    existing.write_text("old\n", encoding="utf-8")
    completion = Path(NativeGymOutputLayout.collection(output).dataset_completion)
    completion.parent.mkdir(parents=True)
    completion.write_text("old marker\n", encoding="utf-8")
    _write(source, [{"uuid": "a"}])
    _write(profile, [_profile("a", [0.0, 1.0])])
    real_replace = reward_profiles.os.replace

    def fail_train(source_path, destination_path):
        if Path(destination_path) == existing:
            raise OSError("simulated publication failure")
        return real_replace(source_path, destination_path)

    monkeypatch.setattr(reward_profiles.os, "replace", fail_train)
    with pytest.raises(OSError, match="simulated publication failure"):
        _publish(source, profile, output, DatasetSelectionSpec())

    assert existing.read_text(encoding="utf-8") == "old\n"
    assert not completion.exists()


def test_marker_failure_leaves_new_train_uncommitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.jsonl"
    profile = tmp_path / "profile.jsonl"
    output = tmp_path / "output"
    output.mkdir()
    existing = output / "train.jsonl"
    existing.write_text("old\n", encoding="utf-8")
    _write(source, [{"uuid": "a"}])
    _write(profile, [_profile("a", [0.0, 1.0])])
    marker = Path(NativeGymOutputLayout.collection(output).dataset_completion)
    real_replace = reward_profiles.os.replace

    def fail_marker(source_path, destination_path):
        if Path(destination_path) == marker:
            raise OSError("simulated marker publication failure")
        return real_replace(source_path, destination_path)

    monkeypatch.setattr(reward_profiles.os, "replace", fail_marker)
    with pytest.raises(OSError, match="simulated marker publication failure"):
        _publish(source, profile, output, DatasetSelectionSpec())

    assert existing.read_text(encoding="utf-8") != "old\n"
    assert not marker.exists()
    with pytest.raises(RewardProfileError, match="completion record is missing"):
        _validate_publication(output)


def test_worker_accepts_long_inline_json_spec_without_treating_it_as_a_path() -> None:
    value = json.dumps(
        {
            "mode": "all",
            "provenance": {"description": "x" * 512},
        }
    )
    assert _load_spec(value).provenance["description"] == "x" * 512


@pytest.mark.parametrize(
    "kwargs",
    [
        {"unprofiled_samples": "typo"},
        {"min_std": True},
        {"success_threshold": False},
        {"provenance": []},
    ],
)
def test_selection_policy_rejects_ambiguous_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(RewardProfileError):
        DatasetSelectionSpec(**kwargs)


def test_component_metric_configuration_is_rejected() -> None:
    with pytest.raises(RewardProfileError, match="unknown dataset selection field"):
        parse_dataset_selection_spec({"metrics": ["reward", "safety_score"]})
