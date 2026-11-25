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
"""Remove exact and near-duplicate JSONL records.

Exact duplicates share normalized text; near duplicates have a MinHash Jaccard
estimate at or above a threshold (``datasketch.MinHashLSH``). Records whose
guard fields differ are never duplicates. ``signatures`` runs once per chunk,
then ``group``; ``run`` does both in one process.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nvflow.stages.data.chunks import (
    atomic_output,
    atomic_write_json,
    completion_matches,
    iter_rows,
    prepare_manifest,
    publish_completion,
)
from nvflow.stages.data.text_keys import (
    exact_key,
    field_names,
    field_value,
    folded_text,
    record_text,
    word_shingles,
)
from nvflow.utils import setup_logger

if TYPE_CHECKING:
    import numpy as np

logger = setup_logger(__name__)

ON_CONFLICT = ("keep_priority", "drop_group", "keep_all")
MINHASH_SCHEME = "affine32"
# Every LSH candidate is re-checked against the threshold, so the index favors
# recall: it proposes ~95% of pairs at the threshold, versus ~40% when balanced.
LSH_WEIGHTS = (0.05, 0.95)
COMPLETION_NAME = "dedup.done"
OUTPUT_NAMES = {
    "kept": "kept.jsonl",
    "dropped": "dropped.jsonl",
    "groups": "groups.jsonl",
    "report": "report.json",
}


@dataclass(frozen=True)
class DedupSettings:
    """What counts as a duplicate and which record of a group is kept."""

    text_fields: tuple[str, ...]
    text_after: str | None = None
    id_field: str | None = None
    exact: bool = True
    near_threshold: float | None = None
    shingle_size: int = 5
    num_perm: int = 128
    seed: int = 1
    guard_fields: tuple[str, ...] = ()
    keep_field: str | None = None
    keep_order: tuple[str, ...] = ()
    conflict_field: str | None = None
    on_conflict: str = "keep_priority"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> DedupSettings:
        """Read the ``data.dedup`` stage options."""
        text_fields = field_names(config.get("text_fields"), "text_fields")
        near_options = config.get("near")
        if near_options is not None and not isinstance(near_options, Mapping):
            raise ValueError("'near' must be a mapping, or null to disable near duplicates")
        near = dict(near_options or {})
        unknown = sorted(set(near) - {"threshold", "shingle_size", "num_perm", "seed"})
        if unknown:
            raise ValueError(f"unknown 'near' options: {', '.join(unknown)}")
        threshold = near.get("threshold", 0.8) if near_options is not None else None
        if threshold is not None and (
            isinstance(threshold, bool)
            or not isinstance(threshold, int | float)
            or not 0 < threshold <= 1
        ):
            raise ValueError("'near.threshold' must be in (0, 1]")
        shingle_size = near.get("shingle_size", 5)
        num_perm = near.get("num_perm", 128)
        seed = near.get("seed", 1)
        if isinstance(shingle_size, bool) or not isinstance(shingle_size, int) or shingle_size < 1:
            raise ValueError("'near.shingle_size' must be a positive integer")
        if isinstance(num_perm, bool) or not isinstance(num_perm, int) or num_perm < 16:
            raise ValueError("'near.num_perm' must be an integer of at least 16")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("'near.seed' must be an integer")
        exact = config.get("exact", True)
        if not isinstance(exact, bool):
            raise ValueError("'exact' must be true or false")
        if not exact and threshold is None:
            raise ValueError("enable 'exact', 'near', or both")
        keep = config.get("keep") or {}
        if not isinstance(keep, Mapping) or set(keep) - {"field", "order"}:
            raise ValueError("'keep' must be a mapping with 'field' and 'order'")
        keep_order = keep.get("order") or []
        if bool(keep.get("field")) != bool(keep_order) or not isinstance(keep_order, list):
            raise ValueError("'keep' requires both 'field' and an 'order' list")
        on_conflict = config.get("on_conflict", "keep_priority")
        if on_conflict not in ON_CONFLICT:
            raise ValueError(f"'on_conflict' must be one of {', '.join(ON_CONFLICT)}")
        if on_conflict != "keep_priority" and not config.get("conflict_field"):
            raise ValueError(f"'on_conflict: {on_conflict}' requires 'conflict_field'")
        return cls(
            text_fields=text_fields,
            text_after=config.get("text_after"),
            id_field=config.get("id_field"),
            exact=exact,
            near_threshold=None if threshold is None else float(threshold),
            shingle_size=shingle_size,
            num_perm=num_perm,
            seed=seed,
            guard_fields=field_names(
                config.get("guard_fields") or [], "guard_fields", required=False
            ),
            keep_field=keep.get("field"),
            keep_order=tuple(keep_order),
            conflict_field=config.get("conflict_field"),
            on_conflict=on_conflict,
        )

    def to_json(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self)))


@dataclass
class Signatures:
    """Per-record keys for one chunk, plus MinHash values when near dedup is on."""

    meta: list[dict[str, Any]] = field(default_factory=list)
    minhash: np.ndarray | None = None


def compute_signatures(
    rows: Iterable[tuple[int, Mapping[str, Any]]],
    settings: DedupSettings,
) -> Signatures:
    meta: list[dict[str, Any]] = []
    shingles: list[list[bytes]] = []
    for index, record in rows:
        text = record_text(record, settings.text_fields, text_after=settings.text_after)
        guard = [field_value(record, name) for name in settings.guard_fields]
        priority = 0
        if settings.keep_field:
            value = field_value(record, settings.keep_field)
            order = settings.keep_order
            priority = order.index(value) if value in order else len(order)
        conflict = field_value(record, settings.conflict_field) if settings.conflict_field else None
        meta.append(
            {
                "row": index,
                "id": field_value(record, settings.id_field) if settings.id_field else index,
                "exact": exact_key(text, guard) if text else None,
                "guard": json.dumps(guard, ensure_ascii=False, default=str),
                "priority": priority,
                "conflict": None if conflict is None else folded_text(str(conflict)),
            }
        )
        if settings.near_threshold is not None:
            shingles.append(
                [
                    item.encode("utf-8")
                    for item in sorted(word_shingles(text, settings.shingle_size))
                ]
            )
    minhash = None
    if settings.near_threshold is not None:
        import numpy as np
        from datasketch import MinHash

        hashes = MinHash.bulk(
            shingles, num_perm=settings.num_perm, seed=settings.seed, scheme=MINHASH_SCHEME
        )
        minhash = (
            np.stack([item.hashvalues for item in hashes])
            if hashes
            else np.zeros((0, settings.num_perm), dtype=np.uint32)
        )
    return Signatures(meta, minhash)


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, first: int, second: int) -> None:
        first, second = self.find(first), self.find(second)
        if first != second:
            self.parent[max(first, second)] = min(first, second)


def _near_links(signatures: Signatures, settings: DedupSettings, links: _UnionFind) -> None:
    if settings.near_threshold is None or signatures.minhash is None:
        return
    import numpy as np
    from datasketch import LeanMinHash, MinHashLSH

    # Rows with the same exact key have the same words and so the same
    # signature; one row stands in for all its copies.
    buckets: dict[str, dict[str, list[int]]] = defaultdict(dict)
    for position, item in enumerate(signatures.meta):
        if item["exact"] is not None:
            buckets[item["guard"]].setdefault(item["exact"], []).append(position)
    values = signatures.minhash
    for copies in buckets.values():
        for first, *rest in copies.values():
            for position in rest:
                links.union(first, position)
        positions = [first for first, *_rest in copies.values()]
        if len(positions) < 2:
            continue
        lsh = MinHashLSH(
            threshold=settings.near_threshold, num_perm=settings.num_perm, weights=LSH_WEIGHTS
        )
        hashes = {
            position: LeanMinHash(
                seed=settings.seed, hashvalues=values[position], scheme=MINHASH_SCHEME
            )
            for position in positions
        }
        with lsh.insertion_session() as session:
            for position in positions:
                session.insert(position, hashes[position])
        for position in positions:
            others = np.array([other for other in lsh.query(hashes[position]) if other > position])
            if others.size:
                similarity = (values[others] == values[position]).mean(axis=1)
                for other in others[similarity >= settings.near_threshold]:
                    links.union(position, int(other))


def find_duplicates(
    signatures: Signatures,
    settings: DedupSettings,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Return drop decisions by row index and one summary per duplicate group."""
    meta = signatures.meta
    for position, item in enumerate(meta):
        if item["row"] != position:
            raise ValueError("signatures must cover every row once, in input order")
    links = _UnionFind(len(meta))
    if settings.exact:
        first: dict[str, int] = {}
        for position, item in enumerate(meta):
            if item["exact"] is not None:
                links.union(first.setdefault(item["exact"], position), position)
    _near_links(signatures, settings, links)

    members: dict[int, list[int]] = defaultdict(list)
    for position in range(len(meta)):
        members[links.find(position)].append(position)
    decisions: dict[int, dict[str, Any]] = {}
    groups = []
    for number, group in enumerate(sorted(rows for rows in members.values() if len(rows) > 1)):
        keeper = min(group, key=lambda position: (meta[position]["priority"], position))
        values = sorted({meta[p]["conflict"] for p in group if meta[p]["conflict"]})
        conflict = len(values) > 1
        if conflict and settings.on_conflict == "keep_all":
            dropped = []
        elif conflict and settings.on_conflict == "drop_group":
            dropped = list(group)
        else:
            dropped = [position for position in group if position != keeper]
        for position in dropped:
            if conflict and settings.on_conflict == "drop_group":
                reason = "conflict"
            elif meta[position]["exact"] == meta[keeper]["exact"]:
                reason = "exact"
            else:
                reason = "near"
            decision: dict[str, Any] = {
                "group": number,
                "duplicate_of": meta[keeper]["id"],
                "reason": reason,
            }
            if reason == "near" and signatures.minhash is not None:
                decision["similarity"] = round(
                    float((signatures.minhash[position] == signatures.minhash[keeper]).mean()), 4
                )
            decisions[position] = decision
        groups.append(
            {
                "group": number,
                "ids": [meta[position]["id"] for position in group],
                "kept": [meta[p]["id"] for p in group if p not in decisions],
                "conflict": conflict,
                "conflict_values": values[:10],
            }
        )
    return decisions, groups


def _report(
    total: int,
    decisions: Mapping[int, Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    signatures: Signatures,
    settings: DedupSettings,
) -> dict[str, Any]:
    return {
        "input_rows": total,
        "kept_rows": total - len(decisions),
        "dropped_rows": len(decisions),
        "dropped_by_reason": dict(sorted(Counter(d["reason"] for d in decisions.values()).items())),
        "groups": len(groups),
        "conflict_groups": sum(1 for group in groups if group["conflict"]),
        "rows_without_text": sum(1 for item in signatures.meta if item["exact"] is None),
        "settings": settings.to_json(),
    }


def _chunk_paths(output_dir: Path, chunk: int) -> dict[str, Path]:
    root = output_dir / "chunks" / f"{chunk:05d}"
    return {"meta": root / "meta.jsonl", "minhash": root / "minhash.npy"}


def chunk_completion(output_dir: Path, chunk: int) -> Path:
    return output_dir / "chunks" / f"{chunk:05d}" / "signatures.done"


def _chunk_outputs(output_dir: Path, chunk: int, settings: DedupSettings) -> dict[str, Path]:
    paths = _chunk_paths(output_dir, chunk)
    if settings.near_threshold is None:
        paths.pop("minhash")
    return paths


def run_signatures(
    output_dir: Path,
    inputs: Sequence[Path],
    settings: DedupSettings,
    num_chunks: int,
    chunk: int,
) -> None:
    """Compute and publish one chunk's signatures unless they are already complete."""
    manifest = prepare_manifest(output_dir, inputs, num_chunks, settings.to_json())
    if not 0 <= chunk < num_chunks:
        raise ValueError(f"chunk {chunk} is outside 0..{num_chunks - 1}")
    outputs = _chunk_outputs(output_dir, chunk, settings)
    completion = chunk_completion(output_dir, chunk)
    if completion_matches(completion, manifest, outputs):
        logger.info(f"Chunk {chunk} signatures are complete; skipping")
        return
    completion.unlink(missing_ok=True)
    start, end = manifest["ranges"][chunk]
    signatures = compute_signatures(iter_rows(manifest, start, end), settings)
    with atomic_output(outputs["meta"]) as stream:
        for item in signatures.meta:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    if signatures.minhash is not None:
        import numpy as np

        with atomic_output(outputs["minhash"], "wb") as stream:
            np.save(stream, signatures.minhash, allow_pickle=False)
    publish_completion(completion, manifest, outputs, chunk=chunk, rows=end - start)
    logger.info(f"Chunk {chunk}: signatures for rows {start}..{end} published")


def _load_signatures(
    output_dir: Path,
    manifest: Mapping[str, Any],
    settings: DedupSettings,
) -> Signatures:
    meta: list[dict[str, Any]] = []
    blocks = []
    missing = []
    for chunk in range(int(manifest["num_chunks"])):
        outputs = _chunk_outputs(output_dir, chunk, settings)
        if not completion_matches(chunk_completion(output_dir, chunk), manifest, outputs):
            missing.append(chunk)
            continue
        with outputs["meta"].open(encoding="utf-8") as stream:
            meta.extend(json.loads(line) for line in stream if line.strip())
        if "minhash" in outputs:
            import numpy as np

            blocks.append(np.load(outputs["minhash"], allow_pickle=False))
    if missing:
        raise ValueError(f"signature chunks are incomplete or stale: {missing}")
    minhash = None
    if settings.near_threshold is not None:
        import numpy as np

        minhash = np.concatenate(blocks)
    return Signatures(meta, minhash)


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {name: output_dir / filename for name, filename in OUTPUT_NAMES.items()}


def run_group(
    output_dir: Path,
    inputs: Sequence[Path],
    settings: DedupSettings,
    num_chunks: int,
) -> dict[str, Any]:
    """Group all chunks' signatures, then write kept and dropped records."""
    manifest = prepare_manifest(output_dir, inputs, num_chunks, settings.to_json())
    outputs = output_paths(output_dir)
    completion = output_dir / COMPLETION_NAME
    if completion_matches(completion, manifest, outputs):
        logger.info(f"{completion} is complete; skipping")
        return dict(json.loads(outputs["report"].read_text(encoding="utf-8")))
    completion.unlink(missing_ok=True)
    signatures = _load_signatures(output_dir, manifest, settings)
    decisions, groups = find_duplicates(signatures, settings)
    with atomic_output(outputs["kept"]) as kept, atomic_output(outputs["dropped"]) as dropped:
        for index, record in iter_rows(manifest):
            decision = decisions.get(index)
            if decision is None:
                kept.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                dropped.write(json.dumps({**record, "_dedup": decision}, ensure_ascii=False) + "\n")
    with atomic_output(outputs["groups"]) as stream:
        for group in groups:
            stream.write(json.dumps(group, ensure_ascii=False, default=str) + "\n")
    report = _report(int(manifest["total_rows"]), decisions, groups, signatures, settings)
    atomic_write_json(outputs["report"], report)
    publish_completion(completion, manifest, outputs)
    logger.info(
        f"Kept {report['kept_rows']:,} of {report['input_rows']:,} rows; "
        f"dropped {report['dropped_by_reason']} in {report['groups']:,} groups"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=["signatures", "group", "run"])
    parser.add_argument("inputs", nargs="+", type=Path, help="Input JSONL files, read in order")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--settings", required=True, help="data.dedup options as JSON")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk", type=int, help="Chunk index for the signatures step")
    args = parser.parse_args(argv)
    settings = DedupSettings.from_config(json.loads(args.settings))
    if args.step == "signatures":
        if args.chunk is None:
            parser.error("the signatures step requires --chunk")
        run_signatures(args.output_dir, args.inputs, settings, args.num_chunks, args.chunk)
        return 0
    if args.step == "run":
        for chunk in range(args.num_chunks):
            run_signatures(args.output_dir, args.inputs, settings, args.num_chunks, chunk)
    run_group(args.output_dir, args.inputs, settings, args.num_chunks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
