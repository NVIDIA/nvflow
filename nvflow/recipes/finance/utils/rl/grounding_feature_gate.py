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
"""Validate repeated native finance rollouts and GroundingVerifier sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"
OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
VALID_STATUSES = frozenset({"allow", "block", "unavailable"})
_TRUTHY = frozenset({"1", "true", "yes", "on"})


class GroundingFeatureGateError(RuntimeError):
    """Raised when a repeated native finance run violates the gate contract."""


def _require_complete(path: Path) -> None:
    if not path.is_file():
        raise GroundingFeatureGateError(f"missing file: {path}")
    marker = Path(f"{path}.done")
    if not marker.is_file():
        raise GroundingFeatureGateError(f"missing completion marker: {marker}")


def _check_offline_contract(row: dict[str, Any], model_root: Path) -> None:
    model_root = model_root.resolve(strict=False)
    missing_env = [
        name for name in OFFLINE_ENV_VARS if os.environ.get(name, "").lower() not in _TRUTHY
    ]
    if missing_env:
        raise GroundingFeatureGateError(
            "offline environment is not enforced: " + ", ".join(missing_env)
        )

    models = row.get("models") or {}
    for key in ("routing_model", "nli_model"):
        model_path = Path(str(models.get(key) or "")).resolve(strict=False)
        if not model_path.is_absolute() or not model_path.is_relative_to(model_root):
            raise GroundingFeatureGateError(f"{key} is not under {model_root}: {model_path}")
        for filename in ("config.json", "model.safetensors"):
            if not (model_path / filename).is_file():
                raise GroundingFeatureGateError(f"missing {key} file: {model_path / filename}")


def validate_grounding_feature(
    *,
    feature: str,
    environment: str,
    rollouts_dir: Path,
    sidecars_dir: Path,
    starting_seed: int,
    required_runs: int,
    expected_rows_per_run: int,
    max_unavailable_rate: float,
    require_offline: bool = False,
    model_root: Path = Path("/hf_models"),
) -> dict[str, Any]:
    """Return an audit summary or fail on incomplete or inconsistent output."""
    if required_runs < 1 or expected_rows_per_run < 1:
        raise ValueError("required_runs and expected_rows_per_run must be positive")
    if not 0 <= max_unavailable_rate <= 1:
        raise ValueError("max_unavailable_rate must be between 0 and 1")

    counts: Counter[str] = Counter()
    evaluation_ids: set[str] = set()
    seeds = list(range(starting_seed, starting_seed + required_runs))

    for seed in seeds:
        rollout = rollouts_dir / environment / "rollout" / f"output-rs{seed}.jsonl"
        sidecar = sidecars_dir / environment / f"grounding-verifier-rs{seed}.jsonl"
        _require_complete(rollout)
        _require_complete(sidecar)

        with rollout.open(encoding="utf-8") as stream:
            rollout_lines = list(stream)
        with sidecar.open(encoding="utf-8") as stream:
            sidecar_lines = list(stream)
        if len(rollout_lines) != expected_rows_per_run:
            raise GroundingFeatureGateError(
                f"seed {seed}: expected {expected_rows_per_run} rollout rows, "
                f"found {len(rollout_lines)}"
            )
        if len(sidecar_lines) != len(rollout_lines):
            raise GroundingFeatureGateError(
                f"seed {seed}: rollout and sidecar row counts differ "
                f"({len(rollout_lines)} != {len(sidecar_lines)})"
            )

        for line_number, (raw_rollout, raw_sidecar) in enumerate(
            zip(rollout_lines, sidecar_lines, strict=True)
        ):
            try:
                row = json.loads(raw_sidecar)
            except json.JSONDecodeError as exc:
                raise GroundingFeatureGateError(
                    f"seed {seed} line {line_number}: invalid sidecar JSON"
                ) from exc
            fingerprint = hashlib.sha256(raw_rollout.encode("utf-8")).hexdigest()[:16]
            if row.get("raw_line_fingerprint") != fingerprint:
                raise GroundingFeatureGateError(
                    f"seed {seed} line {line_number}: rollout fingerprint mismatch"
                )
            if row.get("seed") != seed or row.get("line_number") != line_number:
                raise GroundingFeatureGateError(
                    f"seed {seed} line {line_number}: sidecar identity mismatch"
                )
            evaluation_id = row.get("evaluation_uuid")
            if not evaluation_id or evaluation_id in evaluation_ids:
                raise GroundingFeatureGateError(
                    f"seed {seed} line {line_number}: missing or duplicate evaluation_uuid"
                )
            evaluation_ids.add(evaluation_id)

            status = (row.get("verdict") or {}).get("status")
            if status not in VALID_STATUSES:
                raise GroundingFeatureGateError(
                    f"seed {seed} line {line_number}: invalid verdict status {status!r}"
                )
            counts[status] += 1
            if require_offline:
                _check_offline_contract(row, model_root)

    total_rows = sum(counts.values())
    unavailable_rate = counts["unavailable"] / total_rows
    if unavailable_rate > max_unavailable_rate:
        raise GroundingFeatureGateError(
            f"unavailable rate {unavailable_rate:.3f} exceeds {max_unavailable_rate:.3f}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "feature": feature,
        "environment": environment,
        "required_runs": required_runs,
        "seeds": seeds,
        "rows_per_run": expected_rows_per_run,
        "total_rows": total_rows,
        "verdicts": {status: counts[status] for status in sorted(VALID_STATUSES)},
        "unavailable_rate": unavailable_rate,
        "passed": True,
    }


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    done_path = Path(f"{path}.done")
    done_path.unlink(missing_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".grounding_gate_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
        done_path.touch()
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--rollouts_dir", type=Path, required=True)
    parser.add_argument("--sidecars_dir", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--starting_seed", type=int, required=True)
    parser.add_argument("--required_runs", type=int, required=True)
    parser.add_argument("--expected_rows_per_run", type=int, required=True)
    parser.add_argument("--max_unavailable_rate", type=float, default=0.0)
    parser.add_argument("--require_offline", type=int, choices=(0, 1), default=0)
    parser.add_argument("--model_root", type=Path, default=Path("/hf_models"))
    args = parser.parse_args(argv)

    summary = validate_grounding_feature(
        feature=args.feature,
        environment=args.environment,
        rollouts_dir=args.rollouts_dir,
        sidecars_dir=args.sidecars_dir,
        starting_seed=args.starting_seed,
        required_runs=args.required_runs,
        expected_rows_per_run=args.expected_rows_per_run,
        max_unavailable_rate=args.max_unavailable_rate,
        require_offline=bool(args.require_offline),
        model_root=args.model_root,
    )
    _write_summary(args.output_file, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
