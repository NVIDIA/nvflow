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
"""Fault-focused tests for native Gym input sharding and resume guards."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from nvflow.integrations.nemo_gym.native_sharding import (
    NativeGymShardError,
    ShardIdentityError,
    bind_behavior_manifest,
    main,
    prepare_input_shard,
    publish_rollout_completion,
    repair_truncated_jsonl_tail,
    validate_rollout_completion,
)

_ROLLOUT_CONFIG_SHA256 = hashlib.sha256(b"rollout config").hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _rollout(task_index: int, rollout_index: int, **extra: object) -> dict[str, object]:
    return {
        "_ng_task_index": task_index,
        "_ng_rollout_index": rollout_index,
        **extra,
    }


def test_prepare_empty_shard_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    shard = tmp_path / "chunk-2" / "input.jsonl"
    manifest_path = tmp_path / "chunk-2" / "manifest.json"
    _write_jsonl(source, [{"value": "only-row"}])

    manifest = prepare_input_shard(
        source,
        shard,
        manifest_path,
        chunk_index=2,
        num_chunks=4,
        num_repeats=3,
        rollout_config_sha256=_ROLLOUT_CONFIG_SHA256,
    )

    assert shard.read_bytes() == b""
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {
        "kind": "shard_manifest",
        "schema_version": 1,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "shard_sha256": hashlib.sha256(b"").hexdigest(),
        "effective_row_count": 1,
        "chunk_index": 2,
        "num_chunks": 4,
        "selected_start": 0,
        "selected_end": 0,
        "selected_row_count": 0,
        "num_repeats": 3,
        "rollout_config_sha256": _ROLLOUT_CONFIG_SHA256,
    }
    assert manifest.selected_row_count == 0


def test_prepare_rejects_entirely_empty_source_unless_allowed(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"")
    options: dict[str, Any] = {
        "chunk_index": 0,
        "num_chunks": 1,
        "num_repeats": 1,
        "rollout_config_sha256": _ROLLOUT_CONFIG_SHA256,
    }

    with pytest.raises(NativeGymShardError, match="zero usable rows"):
        prepare_input_shard(source, tmp_path / "shard.jsonl", tmp_path / "manifest.json", **options)

    manifest = prepare_input_shard(
        source, tmp_path / "shard.jsonl", tmp_path / "manifest.json", allow_empty=True, **options
    )
    assert (manifest.effective_row_count, manifest.selected_row_count) == (0, 0)
    assert (tmp_path / "shard.jsonl").read_bytes() == b""


def test_balanced_shards_retain_remainder_and_global_indices(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_jsonl(
        source,
        [
            {"value": "zero"},
            {"value": "one"},
            {"value": "two"},
            {"_ng_task_index": "preserved", "value": "three"},
            {"value": "four"},
        ],
    )

    shards: list[list[dict[str, object]]] = []
    manifests = []
    for chunk_index in range(2):
        shard = tmp_path / f"chunk-{chunk_index}.jsonl"
        manifest = tmp_path / f"chunk-{chunk_index}.manifest.json"
        manifests.append(
            prepare_input_shard(
                source,
                shard,
                manifest,
                chunk_index=chunk_index,
                num_chunks=2,
                num_repeats=2,
                rollout_config_sha256=_ROLLOUT_CONFIG_SHA256,
            )
        )
        shards.append(_read_jsonl(shard))

    assert [item.selected_row_count for item in manifests] == [2, 3]
    assert [(item.selected_start, item.selected_end) for item in manifests] == [
        (0, 2),
        (2, 5),
    ]
    assert [row["_ng_task_index"] for row in shards[0]] == [0, 1]
    assert [row["_ng_task_index"] for row in shards[1]] == [2, "preserved", 4]
    assert [row["value"] for shard in shards for row in shard] == [
        "zero",
        "one",
        "two",
        "three",
        "four",
    ]


def test_max_samples_applies_before_balanced_split(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    rows: list[dict[str, object]] = [{"value": index} for index in range(8)]
    rows[5]["_ng_task_index"] = 0  # Outside the effective input and intentionally ignored.
    _write_jsonl(source, rows)
    shard = tmp_path / "limited.jsonl"

    manifest = prepare_input_shard(
        source,
        shard,
        tmp_path / "limited.manifest.json",
        chunk_index=2,
        num_chunks=3,
        num_repeats=4,
        rollout_config_sha256=_ROLLOUT_CONFIG_SHA256,
        max_num_samples=5,
    )

    assert manifest.effective_row_count == 5
    assert (manifest.selected_start, manifest.selected_end) == (3, 5)
    assert [row["_ng_task_index"] for row in _read_jsonl(shard)] == [3, 4]


def test_prepare_rejects_duplicate_effective_task_ids(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [{"value": "generated-zero"}, {"_ng_task_index": 0}])

    with pytest.raises(NativeGymShardError, match="duplicate _ng_task_index"):
        prepare_input_shard(
            source,
            tmp_path / "shard.jsonl",
            tmp_path / "manifest.json",
            chunk_index=0,
            num_chunks=1,
            num_repeats=1,
            rollout_config_sha256=_ROLLOUT_CONFIG_SHA256,
        )

    assert not (tmp_path / "shard.jsonl").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_prepare_refuses_changed_resume_identity_without_overwriting(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    shard = tmp_path / "shard.jsonl"
    manifest = tmp_path / "manifest.json"
    _write_jsonl(source, [{"value": "original"}])
    prepare_input_shard(
        source,
        shard,
        manifest,
        chunk_index=0,
        num_chunks=1,
        num_repeats=2,
        rollout_config_sha256=_ROLLOUT_CONFIG_SHA256,
    )
    original_shard = shard.read_bytes()
    original_manifest = manifest.read_bytes()

    with pytest.raises(ShardIdentityError, match="num_repeats"):
        prepare_input_shard(
            source,
            shard,
            manifest,
            chunk_index=0,
            num_chunks=1,
            num_repeats=3,
            rollout_config_sha256=_ROLLOUT_CONFIG_SHA256,
        )

    with pytest.raises(ShardIdentityError, match="rollout_config_sha256"):
        prepare_input_shard(
            source,
            shard,
            manifest,
            chunk_index=0,
            num_chunks=1,
            num_repeats=2,
            rollout_config_sha256=hashlib.sha256(b"changed config").hexdigest(),
        )

    _write_jsonl(source, [{"value": "changed"}])
    with pytest.raises(ShardIdentityError, match="source_sha256"):
        prepare_input_shard(
            source,
            shard,
            manifest,
            chunk_index=0,
            num_chunks=1,
            num_repeats=2,
            rollout_config_sha256=_ROLLOUT_CONFIG_SHA256,
        )

    assert shard.read_bytes() == original_shard
    assert manifest.read_bytes() == original_manifest


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("invalid", "invalid JSONL row"),
        ("count", "row count differs"),
        ("task_id", "unexpected _ng_task_index"),
        ("payload", "digest differs"),
    ],
)
def test_prepare_rejects_corrupt_reusable_shard_without_overwriting(
    tmp_path: Path,
    corruption: str,
    message: str,
) -> None:
    source = tmp_path / "source.jsonl"
    shard = tmp_path / "shard.jsonl"
    manifest = tmp_path / "manifest.json"
    _write_jsonl(source, [{"value": "zero"}, {"value": "one"}])
    created = prepare_input_shard(
        source,
        shard,
        manifest,
        chunk_index=0,
        num_chunks=1,
        num_repeats=2,
        rollout_config_sha256=_ROLLOUT_CONFIG_SHA256,
    )
    assert (
        prepare_input_shard(
            source,
            shard,
            manifest,
            chunk_index=0,
            num_chunks=1,
            num_repeats=2,
            rollout_config_sha256=_ROLLOUT_CONFIG_SHA256,
        )
        == created
    )

    rows = _read_jsonl(shard)
    if corruption == "invalid":
        shard.write_bytes(b'{"_ng_task_index":0\n')
    elif corruption == "count":
        _write_jsonl(shard, rows[:1])
    elif corruption == "task_id":
        rows[0]["_ng_task_index"] = 99
        _write_jsonl(shard, rows)
    else:
        rows[0]["value"] = "tampered"
        _write_jsonl(shard, rows)
    corrupt_shard = shard.read_bytes()
    original_manifest = manifest.read_bytes()

    with pytest.raises(NativeGymShardError, match=message):
        prepare_input_shard(
            source,
            shard,
            manifest,
            chunk_index=0,
            num_chunks=1,
            num_repeats=2,
            rollout_config_sha256=_ROLLOUT_CONFIG_SHA256,
        )

    assert shard.read_bytes() == corrupt_shard
    assert manifest.read_bytes() == original_manifest


def test_repair_appends_newline_to_valid_final_json(tmp_path: Path) -> None:
    output = tmp_path / "rollouts.jsonl"
    original = b'{"value":1}\n{"value":2}'
    output.write_bytes(original)

    result = repair_truncated_jsonl_tail(output)

    assert result.action == "appended_newline"
    assert output.read_bytes() == original + b"\n"


def test_repair_truncates_only_invalid_final_tail(tmp_path: Path) -> None:
    output = tmp_path / "rollouts.jsonl"
    complete_rows = b'{"value":1}\n{"value":2}\n'
    output.write_bytes(complete_rows + b'{"value":')

    result = repair_truncated_jsonl_tail(output)

    assert result.action == "truncated_invalid_tail"
    assert result.repaired_size == len(complete_rows)
    assert output.read_bytes() == complete_rows


@pytest.mark.parametrize("duplicate_file", ["materialized", "successes"])
def test_completion_rejects_duplicate_rollout_keys(
    tmp_path: Path,
    duplicate_file: str,
) -> None:
    materialized = tmp_path / "materialized.jsonl"
    successes = tmp_path / "successes.jsonl"
    marker = tmp_path / "completion.json"
    unique = [_rollout(0, 0)]
    duplicate = [_rollout(0, 0), _rollout(0, 0, retry=True)]
    _write_jsonl(materialized, duplicate if duplicate_file == "materialized" else unique)
    _write_jsonl(successes, duplicate if duplicate_file == "successes" else unique)

    with pytest.raises(NativeGymShardError, match="duplicate"):
        publish_rollout_completion(materialized, successes, marker)

    assert not marker.exists()


def test_completion_rejects_missing_successes(tmp_path: Path) -> None:
    materialized = tmp_path / "materialized.jsonl"
    successes = tmp_path / "successes.jsonl"
    marker = tmp_path / "completion.json"
    _write_jsonl(materialized, [_rollout(0, 0), _rollout(1, 0)])
    _write_jsonl(successes, [_rollout(0, 0)])

    with pytest.raises(NativeGymShardError, match="missing=1, unexpected=0"):
        publish_rollout_completion(materialized, successes, marker)

    assert not marker.exists()


def test_completion_rejects_empty_materialized_workload(tmp_path: Path) -> None:
    materialized = tmp_path / "materialized.jsonl"
    successes = tmp_path / "successes.jsonl"
    marker = tmp_path / "completion.json"
    materialized.write_bytes(b"")
    successes.write_bytes(b"")

    with pytest.raises(NativeGymShardError, match="zero rollout keys"):
        publish_rollout_completion(materialized, successes, marker)

    assert not marker.exists()


def test_completion_accepts_manifest_authorized_empty_shard(tmp_path: Path) -> None:
    materialized = tmp_path / "materialized.jsonl"
    successes = tmp_path / "successes.jsonl"
    marker = tmp_path / "completion.json"
    materialized.write_bytes(b"")
    successes.write_bytes(b"")

    evidence = publish_rollout_completion(
        materialized,
        successes,
        marker,
        expected_count=0,
    )

    assert evidence.materialized_input_count == 0
    assert evidence.success_count == 0


def test_completion_rejects_truncated_materialization_with_matching_successes(
    tmp_path: Path,
) -> None:
    materialized = tmp_path / "materialized.jsonl"
    successes = tmp_path / "successes.jsonl"
    marker = tmp_path / "completion.json"
    rows = [_rollout(0, 0)]
    _write_jsonl(materialized, rows)
    _write_jsonl(successes, rows)

    with pytest.raises(NativeGymShardError, match="expected=2, found=1"):
        publish_rollout_completion(
            materialized,
            successes,
            marker,
            expected_count=2,
        )

    assert not marker.exists()


def test_completion_ignores_prior_failures_and_records_evidence(
    tmp_path: Path,
) -> None:
    materialized = tmp_path / "rollouts_materialized_inputs.jsonl"
    successes = tmp_path / "rollouts.jsonl"
    failures = tmp_path / "rollouts_failures.jsonl"
    marker_path = tmp_path / "completion.json"
    expected = [_rollout(7, 0, input="a"), _rollout(7, 1, input="a")]
    completed = [_rollout(7, 1, reward=1), _rollout(7, 0, reward=0)]
    _write_jsonl(materialized, expected)
    _write_jsonl(successes, completed)
    _write_jsonl(
        failures,
        [
            _rollout(7, 0, attempt=0, error="timeout"),
            _rollout(7, 0, attempt=1, error="timeout"),
            _rollout(7, 1, attempt=0, error="connection"),
        ],
    )

    marker = publish_rollout_completion(materialized, successes, marker_path)

    expected_marker = {
        "kind": "rollout_completion",
        "schema_version": 1,
        "status": "complete",
        "materialized_inputs_sha256": hashlib.sha256(materialized.read_bytes()).hexdigest(),
        "successes_sha256": hashlib.sha256(successes.read_bytes()).hexdigest(),
        "materialized_input_count": 2,
        "success_count": 2,
    }
    assert marker.as_json_dict() == expected_marker
    assert json.loads(marker_path.read_text(encoding="utf-8")) == expected_marker


def test_completion_marker_validation_rejects_stale_artifacts(tmp_path: Path) -> None:
    materialized = tmp_path / "rollouts_materialized_inputs.jsonl"
    successes = tmp_path / "rollouts.jsonl"
    marker = tmp_path / "completion.json"
    rows = [_rollout(0, 0, reward=1)]
    _write_jsonl(materialized, rows)
    _write_jsonl(successes, rows)
    expected = publish_rollout_completion(materialized, successes, marker)

    assert validate_rollout_completion(materialized, successes, marker) == expected

    _write_jsonl(successes, [_rollout(0, 0, reward=0)])
    with pytest.raises(NativeGymShardError, match="does not match current rollout artifacts"):
        validate_rollout_completion(materialized, successes, marker)


def test_completion_marker_can_bind_evidence_to_behavior(tmp_path: Path) -> None:
    materialized = tmp_path / "rollouts_materialized_inputs.jsonl"
    successes = tmp_path / "rollouts.jsonl"
    marker = tmp_path / "rejudge.done"
    rows = [_rollout(0, 0, reward=1)]
    _write_jsonl(materialized, rows)
    _write_jsonl(successes, rows)

    evidence = publish_rollout_completion(
        materialized,
        successes,
        marker,
        "a" * 64,
    )

    assert evidence.as_json_dict()["schema_version"] == 1
    assert evidence.as_json_dict()["behavior_sha256"] == "a" * 64
    assert validate_rollout_completion(materialized, successes, marker) == evidence
    assert (
        validate_rollout_completion(
            materialized,
            successes,
            marker,
            "a" * 64,
        )
        == evidence
    )
    with pytest.raises(NativeGymShardError, match="does not match current rollout artifacts"):
        validate_rollout_completion(
            materialized,
            successes,
            marker,
            "b" * 64,
        )


def test_completion_record_validates_owned_artifacts(tmp_path: Path) -> None:
    materialized = tmp_path / "materialized.jsonl"
    successes = tmp_path / "successes.jsonl"
    report = tmp_path / "quality_summary.json"
    blade = tmp_path / "blade"
    blade.mkdir()
    (blade / "report.json").write_text('{"score":1}\n', encoding="utf-8")
    completion = tmp_path / "completion.json"
    _write_jsonl(materialized, [_rollout(0, 0)])
    _write_jsonl(successes, [_rollout(0, 0)])
    report.write_text('{"status":"healthy"}\n', encoding="utf-8")

    evidence = publish_rollout_completion(
        materialized,
        successes,
        completion,
        artifacts={"blade_analysis": blade, "quality_summary": report},
    )

    assert set(evidence.artifacts_sha256 or {}) == {
        "blade_analysis",
        "quality_summary",
    }
    validate_rollout_completion(
        materialized,
        successes,
        completion,
        artifacts={"blade_analysis": blade, "quality_summary": report},
    )
    report.write_text('{"status":"changed"}\n', encoding="utf-8")
    with pytest.raises(NativeGymShardError, match="does not match"):
        validate_rollout_completion(
            materialized,
            successes,
            completion,
            artifacts={"blade_analysis": blade, "quality_summary": report},
        )


def test_validate_command_checks_recorded_artifacts_only_when_given(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    materialized = tmp_path / "rollouts_materialized_inputs.jsonl"
    successes = tmp_path / "rollouts.jsonl"
    report = tmp_path / "quality_summary.json"
    completion = tmp_path / "completion.json"
    rows = [_rollout(0, 0, reward=1)]
    _write_jsonl(materialized, rows)
    _write_jsonl(successes, rows)
    report.write_text('{"status":"healthy"}\n', encoding="utf-8")
    publish_rollout_completion(
        materialized, successes, completion, artifacts={"quality_summary": report}
    )
    command = [
        "validate",
        "--materialized-inputs",
        str(materialized),
        "--successes",
        str(successes),
        "--completion",
        str(completion),
    ]

    assert main(command) == 0
    report.write_text('{"status":"changed"}\n', encoding="utf-8")
    assert main(command) == 0
    assert main([*command, "--artifact", f"quality_summary={report}"]) == 1
    assert "does not match current rollout artifacts" in capsys.readouterr().err


def test_completion_rejects_changed_source_bound_to_shard_manifest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    shard = tmp_path / "shard.jsonl"
    manifest = tmp_path / "manifest.json"
    materialized = tmp_path / "materialized.jsonl"
    successes = tmp_path / "successes.jsonl"
    completion = tmp_path / "completion.json"
    _write_jsonl(source, [{"uuid": "a"}])
    prepare_input_shard(
        source,
        shard,
        manifest,
        chunk_index=0,
        num_chunks=1,
        num_repeats=1,
        rollout_config_sha256=_ROLLOUT_CONFIG_SHA256,
    )
    rows = [_rollout(0, 0)]
    _write_jsonl(materialized, rows)
    _write_jsonl(successes, rows)
    publish_rollout_completion(
        materialized,
        successes,
        completion,
        source_path=source,
        shard_manifests=[manifest],
        manifest_behavior_sha256=_ROLLOUT_CONFIG_SHA256,
    )
    _write_jsonl(source, [{"uuid": "changed"}])

    with pytest.raises(ShardIdentityError, match="source dataset changed"):
        validate_rollout_completion(
            materialized,
            successes,
            completion,
            source_path=source,
            shard_manifests=[manifest],
            manifest_behavior_sha256=_ROLLOUT_CONFIG_SHA256,
        )


def test_partial_output_behavior_manifest_rejects_config_drift(tmp_path: Path) -> None:
    manifest = tmp_path / "state" / "rejudge.json"

    assert bind_behavior_manifest(manifest, "a" * 64) == {
        "kind": "behavior_manifest",
        "schema_version": 1,
        "behavior_sha256": "a" * 64,
    }
    assert bind_behavior_manifest(manifest, "a" * 64)["behavior_sha256"] == "a" * 64
    with pytest.raises(ShardIdentityError, match="rejudge behavior differs"):
        bind_behavior_manifest(manifest, "b" * 64)


def test_partial_output_behavior_manifest_binds_source_completion(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "state" / "rejudge.json"
    source_marker = tmp_path / "source-completion.json"
    source_marker.write_text('{"materialized_input_count":2}\n', encoding="utf-8")

    expected = bind_behavior_manifest(manifest, "a" * 64, source_marker)

    assert expected == {
        "kind": "behavior_manifest",
        "schema_version": 1,
        "behavior_sha256": "a" * 64,
        "source_completion_sha256": hashlib.sha256(source_marker.read_bytes()).hexdigest(),
    }
    source_marker.write_text('{"materialized_input_count":3}\n', encoding="utf-8")
    with pytest.raises(ShardIdentityError, match="rejudge behavior differs"):
        bind_behavior_manifest(manifest, "a" * 64, source_marker)


def test_prepare_reuses_a_shard_recorded_under_an_accepted_fingerprint(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    shard = tmp_path / "shard.jsonl"
    manifest = tmp_path / "manifest.json"
    _write_jsonl(source, [{"value": "original"}])
    prepare_input_shard(
        source,
        shard,
        manifest,
        chunk_index=0,
        num_chunks=1,
        num_repeats=2,
        rollout_config_sha256=_ROLLOUT_CONFIG_SHA256,
    )
    original_shard = shard.read_bytes()
    current = hashlib.sha256(b"current config").hexdigest()

    reused = prepare_input_shard(
        source,
        shard,
        manifest,
        chunk_index=0,
        num_chunks=1,
        num_repeats=2,
        rollout_config_sha256=current,
        accepted_rollout_config_sha256=(_ROLLOUT_CONFIG_SHA256,),
    )

    assert reused.rollout_config_sha256 == current
    assert json.loads(manifest.read_text(encoding="utf-8"))["rollout_config_sha256"] == current
    assert shard.read_bytes() == original_shard
    with pytest.raises(ShardIdentityError, match="rollout_config_sha256"):
        prepare_input_shard(
            source,
            shard,
            manifest,
            chunk_index=0,
            num_chunks=1,
            num_repeats=2,
            rollout_config_sha256=hashlib.sha256(b"changed config").hexdigest(),
            accepted_rollout_config_sha256=(_ROLLOUT_CONFIG_SHA256,),
        )


@pytest.mark.parametrize("recorded", [None, ["not", "a", "digest"]])
def test_a_manifest_without_a_usable_fingerprint_is_refused_cleanly(
    tmp_path: Path, recorded: object
) -> None:
    source = tmp_path / "source.jsonl"
    shard = tmp_path / "shard.jsonl"
    manifest = tmp_path / "manifest.json"
    _write_jsonl(source, [{"value": "original"}])
    prepare_input_shard(
        source,
        shard,
        manifest,
        chunk_index=0,
        num_chunks=1,
        num_repeats=2,
        rollout_config_sha256=_ROLLOUT_CONFIG_SHA256,
    )
    value = json.loads(manifest.read_text(encoding="utf-8"))
    if recorded is None:
        del value["rollout_config_sha256"]
    else:
        value["rollout_config_sha256"] = recorded
    manifest.write_text(json.dumps(value), encoding="utf-8")
    current = hashlib.sha256(b"current config").hexdigest()

    with pytest.raises(ShardIdentityError, match="rollout_config_sha256"):
        prepare_input_shard(
            source,
            shard,
            manifest,
            chunk_index=0,
            num_chunks=1,
            num_repeats=2,
            rollout_config_sha256=current,
            accepted_rollout_config_sha256=(_ROLLOUT_CONFIG_SHA256,),
        )
    rows = [_rollout(0, 0, reward=1), _rollout(0, 1, reward=1)]
    materialized = tmp_path / "materialized.jsonl"
    successes = tmp_path / "successes.jsonl"
    _write_jsonl(materialized, rows)
    _write_jsonl(successes, rows)
    with pytest.raises(ShardIdentityError, match="rollout configuration changed"):
        publish_rollout_completion(
            materialized,
            successes,
            tmp_path / "completion.json",
            shard_manifests=[manifest],
            manifest_behavior_sha256=current,
            accepted_manifest_behavior_sha256=(_ROLLOUT_CONFIG_SHA256,),
        )


def test_completion_evidence_recorded_under_an_accepted_fingerprint_validates(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    manifest = tmp_path / "manifest.json"
    materialized = tmp_path / "materialized.jsonl"
    successes = tmp_path / "successes.jsonl"
    completion = tmp_path / "completion.json"
    _write_jsonl(source, [{"uuid": "a"}])
    prepare_input_shard(
        source,
        tmp_path / "shard.jsonl",
        manifest,
        chunk_index=0,
        num_chunks=1,
        num_repeats=1,
        rollout_config_sha256=_ROLLOUT_CONFIG_SHA256,
    )
    rows = [_rollout(0, 0)]
    _write_jsonl(materialized, rows)
    _write_jsonl(successes, rows)
    evidence = publish_rollout_completion(
        materialized,
        successes,
        completion,
        _ROLLOUT_CONFIG_SHA256,
        source_path=source,
        shard_manifests=[manifest],
        manifest_behavior_sha256=_ROLLOUT_CONFIG_SHA256,
    )
    current = hashlib.sha256(b"current config").hexdigest()

    accepted = validate_rollout_completion(
        materialized,
        successes,
        completion,
        current,
        source_path=source,
        shard_manifests=[manifest],
        manifest_behavior_sha256=current,
        accepted_behavior_sha256=(_ROLLOUT_CONFIG_SHA256,),
        accepted_manifest_behavior_sha256=(_ROLLOUT_CONFIG_SHA256,),
    )

    assert accepted == evidence
    with pytest.raises(ShardIdentityError, match="rollout configuration changed"):
        validate_rollout_completion(
            materialized,
            successes,
            completion,
            current,
            source_path=source,
            shard_manifests=[manifest],
            manifest_behavior_sha256=current,
            accepted_behavior_sha256=(_ROLLOUT_CONFIG_SHA256,),
        )
    with pytest.raises(NativeGymShardError, match="does not match current rollout artifacts"):
        validate_rollout_completion(materialized, successes, completion, current)


def test_rejudge_behavior_manifest_moves_from_an_accepted_fingerprint(tmp_path: Path) -> None:
    manifest = tmp_path / "state" / "rejudge.json"
    source_marker = tmp_path / "source-completion.json"
    source_marker.write_text('{"materialized_input_count":2}\n', encoding="utf-8")
    bind_behavior_manifest(manifest, "a" * 64, source_marker)

    rebound = bind_behavior_manifest(
        manifest, "b" * 64, source_marker, accepted_behavior_sha256=("a" * 64,)
    )

    assert rebound["behavior_sha256"] == "b" * 64
    assert json.loads(manifest.read_text(encoding="utf-8")) == rebound
    with pytest.raises(ShardIdentityError, match="rejudge behavior differs"):
        bind_behavior_manifest(
            manifest, "c" * 64, source_marker, accepted_behavior_sha256=("a" * 64,)
        )
    source_marker.write_text('{"materialized_input_count":3}\n', encoding="utf-8")
    with pytest.raises(ShardIdentityError, match="rejudge behavior differs"):
        bind_behavior_manifest(
            manifest, "d" * 64, source_marker, accepted_behavior_sha256=("b" * 64,)
        )


def test_completion_strictly_rejects_malformed_success_row(tmp_path: Path) -> None:
    materialized = tmp_path / "materialized.jsonl"
    successes = tmp_path / "successes.jsonl"
    marker = tmp_path / "completion.json"
    _write_jsonl(materialized, [_rollout(0, 0)])
    successes.write_bytes(b'{"_ng_task_index":0,"_ng_rollout_index":')

    with pytest.raises(NativeGymShardError, match="invalid JSONL row"):
        publish_rollout_completion(materialized, successes, marker)

    assert not marker.exists()


def test_module_cli_runs_worker_subcommands(tmp_path: Path) -> None:
    module_command = [
        sys.executable,
        "-m",
        "nvflow.integrations.nemo_gym.native_sharding",
    ]
    source = tmp_path / "source.jsonl"
    shard = tmp_path / "input.jsonl"
    manifest = tmp_path / "manifest.json"
    repair_target = tmp_path / "partial.jsonl"
    materialized = tmp_path / "materialized.jsonl"
    successes = tmp_path / "successes.jsonl"
    marker = tmp_path / "completion.json"
    _write_jsonl(source, [{"value": "cli"}])
    repair_target.write_bytes(b'{"valid":true}')
    _write_jsonl(materialized, [_rollout(0, 0)])
    _write_jsonl(successes, [_rollout(0, 0, reward=1)])

    prepared = subprocess.run(
        [
            *module_command,
            "prepare",
            "--source",
            str(source),
            "--output",
            str(shard),
            "--manifest",
            str(manifest),
            "--chunk-index",
            "0",
            "--num-chunks",
            "1",
            "--num-repeats",
            "1",
            "--rollout-config-sha256",
            _ROLLOUT_CONFIG_SHA256,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    repaired = subprocess.run(
        [*module_command, "repair", "--jsonl", str(repair_target)],
        check=True,
        capture_output=True,
        text=True,
    )
    completed = subprocess.run(
        [
            *module_command,
            "complete",
            "--materialized-inputs",
            str(materialized),
            "--successes",
            str(successes),
            "--completion",
            str(marker),
            "--shard-manifest",
            str(manifest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    validated = subprocess.run(
        [
            *module_command,
            "validate",
            "--materialized-inputs",
            str(materialized),
            "--successes",
            str(successes),
            "--completion",
            str(marker),
            "--shard-manifest",
            str(manifest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(prepared.stdout)["selected_row_count"] == 1
    assert json.loads(repaired.stdout)["action"] == "appended_newline"
    assert json.loads(completed.stdout)["status"] == "complete"
    assert json.loads(validated.stdout) == json.loads(completed.stdout)
    assert marker.is_file()
