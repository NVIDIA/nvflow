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
"""Validate Gym reward profiles and atomically publish standardized datasets."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sqlite3
import statistics
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

SelectionMode = Literal["all", "scalar_variance"]
UnprofiledSamplesPolicy = Literal["error", "exclude", "include", "separate"]
SuccessComparator = Literal["ge", "gt", "le", "lt", "eq"]

_TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens")
_PROFILE_FIELDS = {
    "difficulty_profile",
    "pass_rate",
    "pass_rate_passed",
    "pass_rate_total",
    "raw_rewards_and_lengths",
}
_PROFILE_PREFIXES = tuple(f"{token}/" for token in _TOKEN_FIELDS)


class RewardProfileError(ValueError):
    """Raised when profile data cannot be safely normalized or selected."""


@dataclass(frozen=True, slots=True)
class DatasetSelectionSpec:
    """Dataset publication policy over canonical native Gym reward profiles."""

    mode: SelectionMode = "all"
    min_std: float = 0.0
    unprofiled_samples: UnprofiledSamplesPolicy = "error"
    join_key: str | None = "uuid"
    agent: str | None = None
    success_threshold: float | None = None
    success_comparator: SuccessComparator = "ge"
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in {"all", "scalar_variance"}:
            raise RewardProfileError(f"unsupported selection mode: {self.mode!r}")
        if (
            isinstance(self.min_std, bool)
            or not isinstance(self.min_std, int | float)
            or not math.isfinite(self.min_std)
            or self.min_std < 0
        ):
            raise RewardProfileError("min_std must be a finite non-negative number")
        if self.unprofiled_samples not in {
            "error",
            "exclude",
            "include",
            "separate",
        }:
            raise RewardProfileError(
                f"unsupported unprofiled_samples policy: {self.unprofiled_samples!r}"
            )
        if self.join_key is not None and (not isinstance(self.join_key, str) or not self.join_key):
            raise RewardProfileError("join_key must be null or a non-empty string")
        if self.agent is not None and (not isinstance(self.agent, str) or not self.agent):
            raise RewardProfileError("agent must be null or a non-empty string")
        if self.success_threshold is not None and (
            isinstance(self.success_threshold, bool)
            or not isinstance(self.success_threshold, int | float)
            or not math.isfinite(self.success_threshold)
        ):
            raise RewardProfileError("success_threshold must be a finite number")
        if self.success_comparator not in {"ge", "gt", "le", "lt", "eq"}:
            raise RewardProfileError(f"unsupported success comparator: {self.success_comparator!r}")
        if not isinstance(self.provenance, dict):
            raise RewardProfileError("provenance must be a mapping")
        try:
            json.dumps(self.provenance, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise RewardProfileError("provenance must be finite JSON data") from exc


@dataclass(frozen=True, slots=True)
class _Profile:
    key: object
    agent_name: str | None
    sample: dict[str, Any]
    native: dict[str, Any]
    rollout_infos: tuple[dict[str, Any], ...]


def parse_dataset_selection_spec(value: Mapping[str, Any]) -> DatasetSelectionSpec:
    """Parse a closed mapping into the dataset-selection contract."""
    allowed = {definition.name for definition in fields(DatasetSelectionSpec)}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise RewardProfileError(f"unknown dataset selection field(s): {', '.join(unknown)}")
    return DatasetSelectionSpec(**dict(value))


def dataset_selection_sha256(
    spec: DatasetSelectionSpec | Mapping[str, Any],
    responses_create_params_overrides: Mapping[str, Any] | None = None,
) -> str:
    spec_payload = asdict(spec) if isinstance(spec, DatasetSelectionSpec) else dict(spec)
    payload = {
        "dataset_selection": spec_payload,
        "responses_create_params_overrides": dict(responses_create_params_overrides or {}),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            sort_keys=True,
            default=list,
        ).encode()
    ).hexdigest()


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise RewardProfileError(f"{label} must be a finite number")
    return float(value)


def _non_negative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RewardProfileError(f"{label} must be a non-negative integer")
    return value


def _identity_token(value: object, label: str) -> str:
    try:
        return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise RewardProfileError(f"{label} must be a JSON value") from exc


def _identity_sort_key(value: object) -> tuple[int, object]:
    if isinstance(value, int) and not isinstance(value, bool):
        return (0, value)
    if isinstance(value, float) and math.isfinite(value):
        return (1, value)
    if isinstance(value, str):
        return (2, value)
    return (3, _identity_token(value, "_ng_rollout_index"))


def _profile_key(
    row: Mapping[str, Any],
    sample: Mapping[str, Any],
    spec: DatasetSelectionSpec,
) -> object:
    if spec.join_key is None:
        return _identity_token(row.get("_ng_task_index"), "_ng_task_index")
    if spec.join_key not in sample:
        raise RewardProfileError(f"sample is missing join key {spec.join_key!r}")
    return _identity_token(sample[spec.join_key], f"sample.{spec.join_key}")


def _agent_name(row: Mapping[str, Any]) -> str | None:
    reference = row.get("agent_ref")
    if reference is None:
        reference = (row.get("sample") or {}).get("agent_ref")
    if reference is None:
        return None
    if (
        not isinstance(reference, Mapping)
        or not isinstance(reference.get("name"), str)
        or not reference["name"]
    ):
        raise RewardProfileError("agent_ref must contain a non-empty string name")
    return str(reference["name"])


def _iter_reward_profiles(
    profile_path: str | os.PathLike[str],
    spec: DatasetSelectionSpec,
) -> Iterator[_Profile]:
    """Yield validated complete native Gym per-task profiles."""
    path = Path(profile_path)
    count_profiles = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RewardProfileError(f"{path}:{line_number}: malformed JSON: {exc}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("sample"), dict):
                raise RewardProfileError(f"{path}:{line_number}: profile requires object sample")
            if "_ng_task_index" not in row:
                raise RewardProfileError(f"{path}:{line_number}: profile is missing _ng_task_index")
            task_index = row.get("_ng_task_index")
            task_index_token = _identity_token(task_index, "_ng_task_index")
            agent_name = _agent_name(row)
            if spec.agent is not None and agent_name != spec.agent:
                continue
            sample = dict(row["sample"])
            key = _profile_key(row, sample, spec)
            count = _non_negative_integer(row.get("num_rollouts"), "num_rollouts")
            expected = _non_negative_integer(
                row.get("expected_num_rollouts"),
                "expected_num_rollouts",
            )
            missing = _non_negative_integer(
                row.get("missing_num_rollouts"),
                "missing_num_rollouts",
            )
            completion = _finite_number(
                row.get("reward_profile_completion_pct"),
                "reward_profile_completion_pct",
            )
            infos = row.get("rollout_infos")
            if (
                count == 0
                or count != expected
                or missing != 0
                or not math.isclose(completion, 100.0, abs_tol=1e-9)
                or not isinstance(infos, list)
                or len(infos) != count
            ):
                raise RewardProfileError(
                    f"{path}:{line_number}: incomplete reward profile for {key!r}"
                )
            normalized_infos: list[dict[str, Any]] = []
            seen_indices: set[str] = set()
            for info_index, info in enumerate(infos):
                if not isinstance(info, dict):
                    raise RewardProfileError(
                        f"{path}:{line_number}: rollout_infos[{info_index}] must be an object"
                    )
                missing_identity = [
                    name for name in ("_ng_task_index", "_ng_rollout_index") if name not in info
                ]
                if missing_identity:
                    raise RewardProfileError(
                        f"{path}:{line_number}: rollout_infos[{info_index}] is missing "
                        + ", ".join(missing_identity)
                    )
                if (
                    _identity_token(
                        info.get("_ng_task_index"),
                        f"rollout_infos[{info_index}]._ng_task_index",
                    )
                    != task_index_token
                ):
                    raise RewardProfileError(
                        f"{path}:{line_number}: rollout task index does not match profile"
                    )
                rollout_index = _identity_token(
                    info.get("_ng_rollout_index"),
                    f"rollout_infos[{info_index}]._ng_rollout_index",
                )
                if rollout_index in seen_indices:
                    raise RewardProfileError(
                        f"{path}:{line_number}: duplicate rollout index {rollout_index}"
                    )
                seen_indices.add(rollout_index)
                normalized_infos.append(dict(info))
            count_profiles += 1
            yield _Profile(
                key=key,
                agent_name=agent_name,
                sample=sample,
                native={
                    name: value
                    for name, value in row.items()
                    if name not in {"_ng_task_index", "sample", "rollout_infos"}
                },
                rollout_infos=tuple(
                    sorted(
                        normalized_infos,
                        key=lambda item: _identity_sort_key(item["_ng_rollout_index"]),
                    )
                ),
            )
    if not count_profiles:
        raise RewardProfileError(f"{path}: reward profile contains no selected task rows")


def _metric_values(profile: _Profile, metric: str) -> list[float]:
    values = [
        _finite_number(info.get(metric), f"rollout_info.{metric}") for info in profile.rollout_infos
    ]
    native_mean = _finite_number(profile.native.get(f"mean/{metric}"), f"mean/{metric}")
    native_std = _finite_number(profile.native.get(f"std/{metric}"), f"std/{metric}")
    expected_mean = statistics.fmean(values)
    expected_std = statistics.stdev(values) if len(values) > 1 else 0.0
    if not math.isclose(native_mean, expected_mean, rel_tol=1e-9, abs_tol=1e-9):
        raise RewardProfileError(f"native mean/{metric} does not match rollout values")
    if not math.isclose(native_std, expected_std, rel_tol=1e-9, abs_tol=1e-9):
        raise RewardProfileError(f"native std/{metric} does not match rollout values")
    return values


def _token_values(profile: _Profile, token: str) -> list[int | None]:
    values: list[int | None] = []
    for index, info in enumerate(profile.rollout_infos):
        raw = info.get(token)
        if raw is None:
            values.append(None)
            continue
        value = _non_negative_integer(raw, f"rollout_infos[{index}].{token}")
        values.append(value)
    return values


def _success_predicate(spec: DatasetSelectionSpec) -> Callable[[float], bool]:
    threshold = spec.success_threshold
    assert threshold is not None
    return {
        "ge": lambda value: value >= threshold,
        "gt": lambda value: value > threshold,
        "le": lambda value: value <= threshold,
        "lt": lambda value: value < threshold,
        "eq": lambda value: math.isclose(value, threshold, rel_tol=1e-9, abs_tol=1e-9),
    }[spec.success_comparator]


def _scrub_profile_fields(
    source: Mapping[str, Any],
    *,
    drop_agent: bool = False,
) -> dict[str, Any]:
    record = {
        key: value
        for key, value in source.items()
        if key not in _PROFILE_FIELDS
        and key not in {"_ng_task_index", "_ng_rollout_index"}
        and not key.startswith(_PROFILE_PREFIXES)
    }
    if drop_agent:
        record.pop("agent_ref", None)
    return record


def _stable_sample(
    source: Mapping[str, Any],
    *,
    drop_agent: bool = False,
) -> dict[str, Any]:
    """Remove profile fields and Gym's synthetic per-repeat seed."""
    record = _scrub_profile_fields(source, drop_agent=drop_agent)
    responses = record.get("responses_create_params")
    if not isinstance(responses, Mapping):
        return record
    metadata = responses.get("metadata")
    if not isinstance(metadata, Mapping):
        return record
    raw_extra_body = metadata.get("extra_body")
    if not isinstance(raw_extra_body, str):
        return record
    try:
        extra_body = json.loads(raw_extra_body)
    except json.JSONDecodeError:
        return record
    if not isinstance(extra_body, dict):
        return record

    record = copy.deepcopy(record)
    responses = record["responses_create_params"]
    metadata = responses["metadata"]
    extra_body.pop("seed", None)
    if extra_body:
        metadata["extra_body"] = json.dumps(
            extra_body,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    else:
        metadata.pop("extra_body")
    if not metadata:
        responses.pop("metadata")
    return record


def _apply_response_overrides(
    source: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if not overrides:
        return source
    responses = source.get("responses_create_params")
    if not isinstance(responses, Mapping):
        raise RewardProfileError(
            "source responses_create_params must be a mapping when rollout overrides are configured"
        )
    record = copy.deepcopy(dict(source))
    record["responses_create_params"] = {
        **copy.deepcopy(dict(responses)),
        **copy.deepcopy(dict(overrides)),
    }
    return record


def _canonical_json(value: Mapping[str, Any], label: str) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise RewardProfileError(f"{label} must contain finite JSON data") from exc


def _nested_mismatches(
    expected: object,
    actual: object,
    path: str,
) -> list[str]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        mismatches = []
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}" if path else str(key)
            if key not in expected or key not in actual:
                mismatches.append(child)
            else:
                mismatches.extend(_nested_mismatches(expected[key], actual[key], child))
        return mismatches
    return [] if expected == actual else [path]


def _validate_source_matches_profile(
    source: Mapping[str, Any],
    profile: _Profile,
    label: str,
    responses_create_params_overrides: Mapping[str, Any] | None = None,
) -> None:
    supplied = _stable_sample(
        _apply_response_overrides(source, responses_create_params_overrides),
        drop_agent=True,
    )
    canonical = _stable_sample(profile.sample, drop_agent=True)
    canonical.pop("skills_ref", None)
    mismatched = []
    for key, value in supplied.items():
        if key not in canonical:
            mismatched.append(key)
        else:
            mismatched.extend(_nested_mismatches(value, canonical[key], key))
    if mismatched:
        raise RewardProfileError(
            f"{label}: source data changed after rollout for field(s): " + ", ".join(mismatched)
        )


def _normalize_profile(
    source: Mapping[str, Any],
    profile: _Profile,
    spec: DatasetSelectionSpec,
) -> dict[str, Any]:
    """Return one standardized, self-contained profiled dataset record."""
    record = _stable_sample(source)
    reward_values = _metric_values(profile, "reward")
    if profile.agent_name is not None:
        reference = profile.native.get("agent_ref") or profile.sample.get("agent_ref")
        record["agent_ref"] = (
            dict(reference) if isinstance(reference, Mapping) else {"name": profile.agent_name}
        )

    raw_entries = []
    token_values = {token: _token_values(profile, token) for token in _TOKEN_FIELDS}
    for index, info in enumerate(profile.rollout_infos):
        input_tokens = token_values["input_tokens"][index]
        output_tokens = token_values["output_tokens"][index]
        total_tokens = token_values["total_tokens"][index]
        if (
            input_tokens is not None
            and output_tokens is not None
            and total_tokens is not None
            and total_tokens != input_tokens + output_tokens
        ):
            raise RewardProfileError(
                f"rollout {info.get('_ng_rollout_index')!r}: total_tokens does not equal "
                "input_tokens + output_tokens"
            )
        raw_entries.append(
            {
                "_ng_rollout_index": info.get("_ng_rollout_index"),
                "reward": reward_values[index],
                **{token: values[index] for token, values in token_values.items()},
            }
        )
    record["raw_rewards_and_lengths"] = raw_entries

    token_coverage = {}
    for token, values in token_values.items():
        observed = [value for value in values if value is not None]
        token_coverage[token] = {
            "observed": len(observed),
            "total": len(values),
        }
        record[f"{token}/count"] = len(observed)
        record[f"{token}/coverage"] = len(observed) / len(values)
        record[f"{token}/mean"] = statistics.fmean(observed) if observed else None
        record[f"{token}/std"] = (
            statistics.pstdev(observed) if len(observed) > 1 else (0.0 if observed else None)
        )
        record[f"{token}/max"] = max(observed) if observed else None
        record[f"{token}/min"] = min(observed) if observed else None

    difficulty = {
        "n": len(reward_values),
        "avg_reward": statistics.fmean(reward_values),
        "reward_std": statistics.stdev(reward_values) if len(reward_values) > 1 else 0.0,
        "reward_min": min(reward_values),
        "reward_max": max(reward_values),
        "std_convention": "sample",
        "token_std_convention": "population",
        "selection_metric": "reward",
        "provenance": dict(spec.provenance),
        "token_coverage": token_coverage,
    }
    if spec.success_threshold is not None:
        successes = sum(_success_predicate(spec)(value) for value in reward_values)
        record["pass_rate_passed"] = successes
        record["pass_rate_total"] = len(reward_values)
        record["pass_rate"] = successes / len(reward_values)
        difficulty.update(
            {
                "c": successes,
                "pass@1": successes / len(reward_values),
                "successes": successes,
                "success_threshold": spec.success_threshold,
                "success_comparator": spec.success_comparator,
            }
        )
    difficulty["metric_std"] = {"reward": difficulty["reward_std"]}
    record["difficulty_profile"] = difficulty
    return record


def _selected(record: Mapping[str, Any], spec: DatasetSelectionSpec) -> bool:
    difficulty = record.get("difficulty_profile")
    if isinstance(difficulty, Mapping) and difficulty.get("profiled") is False:
        return spec.unprofiled_samples == "include"
    if spec.mode == "all":
        return True
    return record["difficulty_profile"]["reward_std"] >= spec.min_std


def _source_key(
    row: Mapping[str, Any],
    index: int,
    spec: DatasetSelectionSpec,
) -> object:
    if spec.join_key is None:
        identity = row.get("_ng_task_index", index)
        return _identity_token(identity, "source _ng_task_index")
    if spec.join_key not in row:
        raise RewardProfileError(f"source row {index + 1} is missing {spec.join_key!r}")
    return _identity_token(row[spec.join_key], f"source.{spec.join_key}")


def _iter_materialized_scope(
    path: Path,
    spec: DatasetSelectionSpec,
) -> Iterator[tuple[str, str, str, str]]:
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RewardProfileError(
                    f"{path}:{line_number}: malformed materialized input: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise RewardProfileError(
                    f"{path}:{line_number}: materialized input must be an object"
                )
            if spec.agent is not None and _agent_name(row) != spec.agent:
                continue
            missing_identity = [
                name for name in ("_ng_task_index", "_ng_rollout_index") if name not in row
            ]
            if missing_identity:
                raise RewardProfileError(
                    f"{path}:{line_number}: missing " + ", ".join(missing_identity)
                )
            if spec.join_key is None:
                key = _identity_token(row["_ng_task_index"], "_ng_task_index")
            else:
                if spec.join_key not in row:
                    raise RewardProfileError(
                        f"{path}:{line_number}: missing join key {spec.join_key!r}"
                    )
                key = _identity_token(
                    row[spec.join_key],
                    f"materialized.{spec.join_key}",
                )
            count += 1
            yield (
                str(key),
                _identity_token(row["_ng_task_index"], "_ng_task_index"),
                _identity_token(row["_ng_rollout_index"], "_ng_rollout_index"),
                _canonical_json(
                    _stable_sample(row),
                    f"{path}:{line_number}",
                ),
            )
    if not count:
        raise RewardProfileError(f"{path}: materialized inputs contain no selected tasks")


def _encode_profile(profile: _Profile) -> str:
    try:
        return json.dumps(
            {
                "key": profile.key,
                "agent_name": profile.agent_name,
                "sample": profile.sample,
                "native": profile.native,
                "rollout_infos": list(profile.rollout_infos),
            },
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise RewardProfileError(
            f"profile {profile.key!r} contains non-finite or non-JSON data"
        ) from exc


def _decode_profile(payload: str) -> _Profile:
    value = json.loads(payload)
    return _Profile(
        key=value["key"],
        agent_name=value["agent_name"],
        sample=value["sample"],
        native=value["native"],
        rollout_infos=tuple(value["rollout_infos"]),
    )


def _build_profile_index(
    profile_path: Path,
    spec: DatasetSelectionSpec,
    output_dir: Path,
    materialized_inputs_path: Path | None,
) -> tuple[sqlite3.Connection, Path]:
    """Build a disk-backed identity index and validate materialized scope."""
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=".dataset-selection.",
        suffix=".sqlite3",
        dir=output_dir,
    )
    os.close(descriptor)
    index_path = Path(raw_path)
    connection = sqlite3.connect(index_path)
    try:
        connection.execute(
            "CREATE TABLE profiles "
            "(identity TEXT PRIMARY KEY, payload TEXT NOT NULL, materialized_sample TEXT NOT NULL, "
            "used INTEGER NOT NULL DEFAULT 0)"
        )
        connection.execute(
            "CREATE TABLE profile_rollouts "
            "(identity TEXT NOT NULL, task_index TEXT NOT NULL, rollout_index TEXT NOT NULL, "
            "PRIMARY KEY(identity, task_index, rollout_index))"
        )
        connection.execute("CREATE TABLE source (identity TEXT PRIMARY KEY)")
        for profile in _iter_reward_profiles(profile_path, spec):
            try:
                connection.execute(
                    "INSERT INTO profiles(identity, payload, materialized_sample) VALUES (?, ?, ?)",
                    (
                        str(profile.key),
                        _encode_profile(profile),
                        _canonical_json(
                            _stable_sample(profile.sample),
                            f"profile {profile.key!r} sample",
                        ),
                    ),
                )
                for info in profile.rollout_infos:
                    connection.execute(
                        "INSERT INTO profile_rollouts(identity, task_index, rollout_index) "
                        "VALUES (?, ?, ?)",
                        (
                            str(profile.key),
                            _identity_token(info["_ng_task_index"], "_ng_task_index"),
                            _identity_token(info["_ng_rollout_index"], "_ng_rollout_index"),
                        ),
                    )
            except sqlite3.IntegrityError as exc:
                raise RewardProfileError(
                    f"ambiguous duplicate profile identity {profile.key!r}; "
                    "select an agent or use a unique join key"
                ) from exc
        if materialized_inputs_path is not None:
            connection.execute(
                "CREATE TABLE scope "
                "(identity TEXT NOT NULL, task_index TEXT NOT NULL, rollout_index TEXT NOT NULL, "
                "sample TEXT NOT NULL, PRIMARY KEY(identity, task_index, rollout_index))"
            )
            for key, task_index, rollout_index, sample in _iter_materialized_scope(
                materialized_inputs_path,
                spec,
            ):
                try:
                    connection.execute(
                        "INSERT INTO scope(identity, task_index, rollout_index, sample) "
                        "VALUES (?, ?, ?, ?)",
                        (key, task_index, rollout_index, sample),
                    )
                except sqlite3.IntegrityError as exc:
                    raise RewardProfileError(
                        "materialized inputs contain a duplicate task/rollout identity"
                    ) from exc
            missing = connection.execute(
                "SELECT COUNT(*) FROM scope LEFT JOIN profile_rollouts "
                "USING(identity, task_index, rollout_index) "
                "WHERE profile_rollouts.identity IS NULL"
            ).fetchone()[0]
            extra = connection.execute(
                "SELECT COUNT(*) FROM profile_rollouts LEFT JOIN scope "
                "USING(identity, task_index, rollout_index) WHERE scope.identity IS NULL"
            ).fetchone()[0]
            mismatched_samples = connection.execute(
                "SELECT COUNT(*) FROM scope JOIN profiles USING(identity) "
                "WHERE scope.sample != profiles.materialized_sample"
            ).fetchone()[0]
            if missing or extra or mismatched_samples:
                raise RewardProfileError(
                    "reward profile rollouts do not exactly match materialized input "
                    f"scope: missing={missing}, extra={extra}, "
                    f"sample_mismatches={mismatched_samples}"
                )
        connection.commit()
        return connection, index_path
    except BaseException:
        connection.close()
        index_path.unlink(missing_ok=True)
        raise


def _atomic_temporary(path: Path, lines: Iterable[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        for line in lines:
            stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())
    return temporary


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, label: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RewardProfileError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_jsonl_evidence(
    path: Path,
    *,
    expected_rows: object,
    expected_sha256: object,
    label: str,
) -> None:
    expected_digest = _require_sha256(expected_sha256, f"{label}.sha256")
    if isinstance(expected_rows, bool) or not isinstance(expected_rows, int) or expected_rows <= 0:
        raise RewardProfileError(f"{label}.rows must be a positive integer")
    digest = hashlib.sha256()
    rows = 0
    try:
        with path.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                digest.update(line)
                if not line.strip():
                    continue
                rows += 1
                try:
                    value = json.loads(line)
                except (UnicodeError, json.JSONDecodeError) as exc:
                    raise RewardProfileError(
                        f"{path}:{line_number}: malformed JSON: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise RewardProfileError(f"{path}:{line_number}: row must be a JSON object")
    except FileNotFoundError as exc:
        raise RewardProfileError(f"{label} file is missing: {path}") from exc
    except OSError as exc:
        raise RewardProfileError(f"unable to read {label} file {path}: {exc}") from exc
    if rows != expected_rows:
        raise RewardProfileError(f"{label} row count mismatch: marker={expected_rows}, file={rows}")
    if digest.hexdigest() != expected_digest:
        raise RewardProfileError(f"{label} SHA-256 does not match completion evidence")


def validate_dataset_completion(
    profiled_path: str | os.PathLike[str],
    train_path: str | os.PathLike[str],
    report_path: str | os.PathLike[str],
    completion_path: str | os.PathLike[str],
    *,
    diagnostics_path: str | os.PathLike[str] | None = None,
    expected_selection_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate dataset-publication evidence and return its metadata."""
    profiled = Path(profiled_path)
    train = Path(train_path)
    report = Path(report_path)
    completion = Path(completion_path)
    try:
        payload = json.loads(completion.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RewardProfileError(f"dataset completion record is missing: {completion}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RewardProfileError(
            f"unable to read dataset completion record {completion}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RewardProfileError("dataset completion record must contain a JSON object")
    if payload.get("kind") != "dataset_completion" or payload.get("schema_version") != 1:
        raise RewardProfileError(
            f"unsupported dataset completion schema_version {payload.get('schema_version')!r}"
        )
    profiled_evidence = payload.get("profiled")
    train_evidence = payload.get("train")
    if not isinstance(profiled_evidence, dict) or not isinstance(train_evidence, dict):
        raise RewardProfileError("dataset completion must describe profiled and train outputs")
    source_rows = payload.get("source_rows")
    profiled_rows = profiled_evidence.get("rows")
    if (
        isinstance(source_rows, bool)
        or not isinstance(source_rows, int)
        or isinstance(profiled_rows, bool)
        or not isinstance(profiled_rows, int)
        or source_rows < profiled_rows
    ):
        raise RewardProfileError("completion.source_rows must cover every profiled output row")
    for field_name in ("source_sha256", "profile_sha256"):
        _require_sha256(payload.get(field_name), f"completion.{field_name}")
    selection_sha256 = _require_sha256(
        payload.get("selection_sha256"),
        "completion.selection_sha256",
    )
    if expected_selection_sha256 is not None and selection_sha256 != expected_selection_sha256:
        raise RewardProfileError("dataset selection configuration differs from completion evidence")
    for field_name in (
        "materialized_inputs_sha256",
        "rollouts_sha256",
        "rollout_completion_sha256",
    ):
        _require_sha256(
            payload.get(field_name),
            f"completion.{field_name}",
            nullable=True,
        )
    _validate_jsonl_evidence(
        profiled,
        expected_rows=profiled_evidence.get("rows"),
        expected_sha256=profiled_evidence.get("sha256"),
        label="profiled",
    )
    _validate_jsonl_evidence(
        train,
        expected_rows=train_evidence.get("rows"),
        expected_sha256=train_evidence.get("sha256"),
        label="train",
    )
    report_sha256 = _require_sha256(
        payload.get("report_sha256"),
        "completion.report_sha256",
    )
    if _sha256_file(report) != report_sha256:
        raise RewardProfileError("selection report SHA-256 does not match completion evidence")
    try:
        report_payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RewardProfileError(f"unable to read selection report {report}: {exc}") from exc
    if (
        not isinstance(report_payload, dict)
        or report_payload.get("kind") != "dataset_selection_report"
        or report_payload.get("schema_version") != 1
        or report_payload.get("selection_sha256") != selection_sha256
    ):
        raise RewardProfileError("selection report does not match dataset completion evidence")
    diagnostics = payload.get("diagnostics")
    if diagnostics is not None:
        if diagnostics_path is None or not isinstance(diagnostics, dict):
            raise RewardProfileError("dataset completion diagnostics are invalid")
        _validate_jsonl_evidence(
            Path(diagnostics_path),
            expected_rows=diagnostics.get("rows"),
            expected_sha256=diagnostics.get("sha256"),
            label="unprofiled samples",
        )
    return payload


@contextmanager
def _publication_lock(output_dir: Path) -> Iterator[None]:
    """Serialize the final train/report/marker publication on POSIX workers."""
    if os.name == "nt":  # Worker jobs run on Linux; keep local Windows inspection importable.
        yield
        return
    import fcntl

    lock_path = output_dir / ".dataset-publication.lock"
    with lock_path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def build_training_dataset(
    source_path: str | os.PathLike[str],
    profile_path: str | os.PathLike[str],
    profiled_output_path: str | os.PathLike[str],
    training_output_path: str | os.PathLike[str],
    report_output_path: str | os.PathLike[str],
    completion_output_path: str | os.PathLike[str],
    diagnostics_output_path: str | os.PathLike[str],
    spec: DatasetSelectionSpec,
    *,
    materialized_inputs_path: str | os.PathLike[str] | None = None,
    rollouts_path: str | os.PathLike[str] | None = None,
    rollout_completion_path: str | os.PathLike[str] | None = None,
    responses_create_params_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize, select, and atomically publish profiled and training datasets."""
    source = Path(source_path)
    profile = Path(profile_path)
    profiled_path = Path(profiled_output_path)
    train_path = Path(training_output_path)
    report_path = Path(report_output_path)
    completion_path = Path(completion_output_path)
    diagnostics_path = Path(diagnostics_output_path)
    destinations = {
        profiled_path.resolve(),
        train_path.resolve(),
        report_path.resolve(),
        completion_path.resolve(),
        diagnostics_path.resolve(),
    }
    if source.resolve() in destinations or len(destinations) != 5:
        raise RewardProfileError("source and dataset-publication output paths must be distinct")

    completion_path.unlink(missing_ok=True)
    profile_sha256 = _sha256_file(profile)
    scope_path = Path(materialized_inputs_path) if materialized_inputs_path else None
    rollouts = Path(rollouts_path) if rollouts_path else None
    rollout_completion = Path(rollout_completion_path) if rollout_completion_path else None
    if spec.unprofiled_samples in {"exclude", "separate"} and (
        scope_path is None or rollouts is None or rollout_completion is None
    ):
        raise RewardProfileError(
            f"unprofiled_samples={spec.unprofiled_samples!r} requires materialized "
            "inputs, rollouts, and their completion record to prove scope"
        )
    if scope_path is not None and (rollouts is None) != (rollout_completion is None):
        raise RewardProfileError("rollouts and rollout completion must be provided together")
    if rollouts is not None and rollout_completion is not None:
        if scope_path is None:
            raise RewardProfileError(
                "materialized inputs are required with rollout completion evidence"
            )
        from .native_sharding import NativeGymShardError, validate_rollout_completion

        try:
            validate_rollout_completion(scope_path, rollouts, rollout_completion)
        except (NativeGymShardError, OSError) as exc:
            raise RewardProfileError(f"invalid rollout completion evidence: {exc}") from exc
    materialized_sha256 = _sha256_file(scope_path) if scope_path else None
    rollouts_sha256 = _sha256_file(rollouts) if rollouts else None
    rollout_completion_sha256 = _sha256_file(rollout_completion) if rollout_completion else None
    unprofiled_count = 0
    total = 0
    profiled_count = 0
    selected_count = 0
    for path in (
        profiled_path,
        train_path,
        report_path,
        completion_path,
        diagnostics_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    profile_index, profile_index_path = _build_profile_index(
        profile,
        spec,
        completion_path.parent,
        scope_path,
    )
    profiled_temporary: Path | None = None
    train_temporary: Path | None = None
    diagnostics_temporary: Path | None = None
    auxiliary_temporaries: list[tuple[Path, Path]] = []
    try:
        with (
            tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=profiled_path.parent,
                prefix=f".{profiled_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as profiled_destination,
            tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=train_path.parent,
                prefix=f".{train_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as train_destination,
        ):
            profiled_temporary = Path(profiled_destination.name)
            train_temporary = Path(train_destination.name)
            profiled_digest = hashlib.sha256()
            train_digest = hashlib.sha256()
            diagnostics_stream = None
            if spec.unprofiled_samples == "separate":
                diagnostics_stream = tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=diagnostics_path.parent,
                    prefix=f".{diagnostics_path.name}.",
                    suffix=".tmp",
                    delete=False,
                )
                diagnostics_temporary = Path(diagnostics_stream.name)
            diagnostics_digest = hashlib.sha256()
            source_digest = hashlib.sha256()
            try:
                with source.open(encoding="utf-8", newline="") as stream:
                    for index, line in enumerate(stream):
                        source_digest.update(line.encode())
                        if not line.strip():
                            continue
                        total += 1
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise RewardProfileError(
                                f"{source}:{index + 1}: malformed JSON: {exc}"
                            ) from exc
                        if not isinstance(row, dict):
                            raise RewardProfileError(
                                f"{source}:{index + 1}: source row must be an object"
                            )
                        key = _source_key(row, total - 1, spec)
                        try:
                            profile_index.execute(
                                "INSERT INTO source(identity) VALUES (?)",
                                (str(key),),
                            )
                        except sqlite3.IntegrityError as exc:
                            raise RewardProfileError(
                                f"{source}:{index + 1}: duplicate source identity {key}"
                            ) from exc
                        indexed = profile_index.execute(
                            "SELECT payload FROM profiles WHERE identity = ?",
                            (str(key),),
                        ).fetchone()
                        if indexed is None:
                            if (
                                scope_path is not None
                                and profile_index.execute(
                                    "SELECT 1 FROM scope WHERE identity = ?",
                                    (str(key),),
                                ).fetchone()
                            ):
                                raise RewardProfileError(
                                    f"{source}:{index + 1}: scoped row has no "
                                    f"reward profile for {key}"
                                )
                            unprofiled_count += 1
                            if spec.unprofiled_samples == "error":
                                raise RewardProfileError(
                                    f"{source}:{index + 1}: no reward profile for {key}"
                                )
                            if spec.unprofiled_samples == "separate":
                                assert diagnostics_stream is not None
                                diagnostic = (
                                    json.dumps(
                                        {
                                            "source_index": total - 1,
                                            "identity": key,
                                            "reason": "outside_rollout_scope",
                                        },
                                        ensure_ascii=False,
                                        allow_nan=False,
                                    )
                                    + "\n"
                                )
                                diagnostics_stream.write(diagnostic)
                                diagnostics_digest.update(diagnostic.encode())
                            if spec.unprofiled_samples in {"exclude", "separate"}:
                                continue
                            normalized = _stable_sample(
                                _apply_response_overrides(
                                    row,
                                    responses_create_params_overrides,
                                )
                            )
                            normalized["difficulty_profile"] = {"profiled": False}
                        else:
                            selected_profile = _decode_profile(str(indexed[0]))
                            _validate_source_matches_profile(
                                row,
                                selected_profile,
                                f"{source}:{index + 1}",
                                responses_create_params_overrides,
                            )
                            profile_index.execute(
                                "UPDATE profiles SET used = 1 WHERE identity = ?",
                                (str(key),),
                            )
                            normalized = _normalize_profile(
                                selected_profile.sample,
                                selected_profile,
                                spec,
                            )
                        rendered = (
                            json.dumps(
                                normalized,
                                ensure_ascii=False,
                                allow_nan=False,
                            )
                            + "\n"
                        )
                        profiled_destination.write(rendered)
                        profiled_digest.update(rendered.encode())
                        profiled_count += 1
                        if not _selected(normalized, spec):
                            continue
                        train_destination.write(rendered)
                        train_digest.update(rendered.encode())
                        selected_count += 1
                unused_profiles = profile_index.execute(
                    "SELECT COUNT(*) FROM profiles WHERE used = 0"
                ).fetchone()[0]
                if unused_profiles:
                    raise RewardProfileError(
                        f"reward profile contains {unused_profiles} task(s) absent "
                        "from the source dataset"
                    )
                if profiled_count == 0:
                    raise RewardProfileError("dataset publication produced no profiled rows")
                if selected_count == 0:
                    raise RewardProfileError("dataset selection retained no source rows")
                profiled_destination.flush()
                os.fsync(profiled_destination.fileno())
                train_destination.flush()
                os.fsync(train_destination.fileno())
                if diagnostics_stream is not None:
                    diagnostics_stream.flush()
                    os.fsync(diagnostics_stream.fileno())
            finally:
                if diagnostics_stream is not None:
                    diagnostics_stream.close()
        if source_digest.hexdigest() != _sha256_file(source):
            raise RewardProfileError("source dataset changed during dataset publication")
        if profile_sha256 != _sha256_file(profile):
            raise RewardProfileError("reward profile changed during dataset publication")
        if scope_path and materialized_sha256 != _sha256_file(scope_path):
            raise RewardProfileError("materialized inputs changed while selection was running")
        if rollouts and rollouts_sha256 != _sha256_file(rollouts):
            raise RewardProfileError("rollouts changed while selection was running")
        if rollout_completion and rollout_completion_sha256 != _sha256_file(rollout_completion):
            raise RewardProfileError("completion record changed during dataset publication")

        spec_payload = asdict(spec)
        selection_sha256 = dataset_selection_sha256(
            spec,
            responses_create_params_overrides,
        )
        report = {
            "kind": "dataset_selection_report",
            "schema_version": 1,
            "selection": spec_payload,
            "selection_sha256": selection_sha256,
            "mode": spec.mode,
            "metric": "reward",
            "min_std": spec.min_std,
            "unprofiled_samples": spec.unprofiled_samples,
            "total": total,
            "profiled": profiled_count,
            "selected": selected_count,
            "removed": profiled_count - selected_count,
            "unprofiled": unprofiled_count,
            "source": str(source),
            "profile": str(profile),
            "materialized_inputs": str(scope_path) if scope_path else None,
            "rollouts": str(rollouts) if rollouts else None,
            "rollout_completion": (str(rollout_completion) if rollout_completion else None),
        }
        report_text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        report_temporary = _atomic_temporary(report_path, [report_text])
        diagnostics_evidence = None
        if unprofiled_count and spec.unprofiled_samples == "separate":
            diagnostics_evidence = {
                "rows": unprofiled_count,
                "sha256": diagnostics_digest.hexdigest(),
            }
        completion_text = (
            json.dumps(
                {
                    "kind": "dataset_completion",
                    "schema_version": 1,
                    "profiled": {
                        "sha256": profiled_digest.hexdigest(),
                        "rows": profiled_count,
                    },
                    "train": {
                        "sha256": train_digest.hexdigest(),
                        "rows": selected_count,
                    },
                    "diagnostics": diagnostics_evidence,
                    "report_sha256": _sha256_file(report_temporary),
                    "source_rows": total,
                    "source_sha256": source_digest.hexdigest(),
                    "profile_sha256": profile_sha256,
                    "materialized_inputs_sha256": materialized_sha256,
                    "rollouts_sha256": rollouts_sha256,
                    "rollout_completion_sha256": rollout_completion_sha256,
                    "selection_sha256": selection_sha256,
                },
                sort_keys=True,
            )
            + "\n"
        )
        auxiliary_temporaries.append((report_path, report_temporary))
        auxiliary_temporaries.append(
            (
                completion_path,
                _atomic_temporary(completion_path, [completion_text]),
            )
        )
        if diagnostics_evidence is not None:
            assert diagnostics_temporary is not None
            auxiliary_temporaries.insert(
                0,
                (diagnostics_path, diagnostics_temporary),
            )
            diagnostics_temporary = None
        else:
            diagnostics_path.unlink(missing_ok=True)
        assert train_temporary is not None
        assert profiled_temporary is not None
        with _publication_lock(completion_path.parent):
            os.replace(profiled_temporary, profiled_path)
            profiled_temporary = None
            os.replace(train_temporary, train_path)
            train_temporary = None
            for destination, temporary in auxiliary_temporaries:
                os.replace(temporary, destination)
    finally:
        if profiled_temporary is not None:
            profiled_temporary.unlink(missing_ok=True)
        if train_temporary is not None:
            train_temporary.unlink(missing_ok=True)
        if diagnostics_temporary is not None:
            diagnostics_temporary.unlink(missing_ok=True)
        for _destination, temporary in auxiliary_temporaries:
            temporary.unlink(missing_ok=True)
        profile_index.close()
        profile_index_path.unlink(missing_ok=True)
    return report


__all__ = [
    "DatasetSelectionSpec",
    "RewardProfileError",
    "SelectionMode",
    "SuccessComparator",
    "UnprofiledSamplesPolicy",
    "build_training_dataset",
    "parse_dataset_selection_spec",
    "validate_dataset_completion",
]
