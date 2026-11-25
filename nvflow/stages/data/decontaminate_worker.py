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
"""Find records that overlap evaluation (reference) sets.

Layers: exact normalized text, n-gram containment (an exact index of the
references' word n-grams), and embedding similarity. Scores at or above ``drop``
drop a record; scores at or above ``review`` list the pair in
``candidates.jsonl``. ``lexical`` and ``embed`` run per chunk, then ``merge``.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nvflow.integrations.embeddings.settings import EmbeddingSettings
from nvflow.stages.data.chunks import (
    atomic_output,
    atomic_write_json,
    completion_matches,
    digest_value,
    file_evidence,
    iter_rows,
    prepare_manifest,
    publish_completion,
)
from nvflow.stages.data.text_keys import (
    exact_key,
    field_names,
    field_value,
    record_text,
    words,
)
from nvflow.utils import setup_logger

if TYPE_CHECKING:
    from nvflow.integrations.embeddings.encoders import Encoder

logger = setup_logger(__name__)

COMPLETION_NAME = "decontaminate.done"
OUTPUT_NAMES = {
    "kept": "kept.jsonl",
    "dropped": "dropped.jsonl",
    "candidates": "candidates.jsonl",
    "report": "report.json",
}
PLACEMENT_KEYS = {"container", "num_gpus"}
_DECISION_KEYS = {"review", "drop", "top_k"}


def _threshold(options: Mapping[str, Any], key: str, layer: str) -> float | None:
    value = options.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value <= 1:
        raise ValueError(f"'{layer}.{key}' must be in (0, 1]")
    return float(value)


@dataclass(frozen=True)
class Reference:
    name: str
    path: str
    text_fields: tuple[str, ...]
    text_after: str | None
    id_field: str | None


@dataclass(frozen=True)
class Layer:
    review: float | None
    drop: float | None

    @property
    def floor(self) -> float:
        return min(value for value in (self.review, self.drop) if value is not None)

    def verdict(self, score: float) -> str | None:
        if self.drop is not None and score >= self.drop:
            return "drop"
        if self.review is not None and score >= self.review:
            return "review"
        return None


@dataclass(frozen=True)
class DecontaminationSettings:
    """Record text, reference sets, and each layer's thresholds."""

    text_fields: tuple[str, ...]
    text_after: str | None
    id_field: str | None
    references: tuple[Reference, ...]
    exact: bool
    ngram: Layer | None
    ngram_size: int
    ngram_top_k: int
    embedding: EmbeddingSettings | None
    embedding_layer: Layer | None
    top_k: int
    cache_dir: str | None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> DecontaminationSettings:
        """Read the ``data.decontaminate`` stage options."""
        text_fields = field_names(config.get("text_fields"), "text_fields")
        text_after = config.get("text_after")
        id_field = config.get("id_field")
        raw_references = config.get("references")
        if not isinstance(raw_references, Mapping) or not raw_references:
            raise ValueError("'references' must map each reference set name to its file")
        references = []
        for name, value in raw_references.items():
            spec = {"path": value} if isinstance(value, str) else value
            if not isinstance(spec, Mapping) or not spec.get("path"):
                raise ValueError(f"reference set {name!r} needs a 'path'")
            unknown = sorted(set(spec) - {"path", "text_fields", "text_after", "id_field"})
            if unknown:
                raise ValueError(
                    f"reference set {name!r} has unknown options: {', '.join(unknown)}"
                )
            references.append(
                Reference(
                    name=str(name),
                    path=str(spec["path"]),
                    text_fields=field_names(
                        spec.get("text_fields", text_fields), f"references.{name}.text_fields"
                    ),
                    text_after=spec.get("text_after", text_after),
                    id_field=spec.get("id_field", id_field),
                )
            )
        exact = config.get("exact", True)
        if not isinstance(exact, bool):
            raise ValueError("'exact' must be true or false")
        ngram_options = config.get("ngram", {"drop": 0.5})
        ngram = None
        ngram_size = 8
        ngram_top_k = 5
        if ngram_options is not None:
            if not isinstance(ngram_options, Mapping):
                raise ValueError("'ngram' must be a mapping, or null to disable it")
            unknown = sorted(set(ngram_options) - {"size", "review", "drop", "top_k"})
            if unknown:
                raise ValueError(f"unknown 'ngram' options: {', '.join(unknown)}")
            ngram = Layer(
                _threshold(ngram_options, "review", "ngram"),
                _threshold(ngram_options, "drop", "ngram"),
            )
            ngram_size = ngram_options.get("size", 8)
            if isinstance(ngram_size, bool) or not isinstance(ngram_size, int) or ngram_size < 1:
                raise ValueError("'ngram.size' must be a positive integer")
            ngram_top_k = ngram_options.get("top_k", 5)
            if isinstance(ngram_top_k, bool) or not isinstance(ngram_top_k, int) or ngram_top_k < 1:
                raise ValueError("'ngram.top_k' must be a positive integer")
        embedding_options = config.get("embedding")
        embedding = embedding_layer = None
        top_k = 5
        if embedding_options is not None:
            if not isinstance(embedding_options, Mapping):
                raise ValueError("'embedding' must be a mapping, or null to disable it")
            embedding = EmbeddingSettings.from_config(
                {
                    key: value
                    for key, value in embedding_options.items()
                    if key not in _DECISION_KEYS | PLACEMENT_KEYS
                }
            )
            embedding_layer = Layer(
                _threshold(embedding_options, "review", "embedding"),
                _threshold(embedding_options, "drop", "embedding"),
            )
            top_k = embedding_options.get("top_k", 5)
            if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
                raise ValueError("'embedding.top_k' must be a positive integer")
        for name, layer in (("ngram", ngram), ("embedding", embedding_layer)):
            if layer is not None and layer.review is None and layer.drop is None:
                raise ValueError(f"'{name}' needs a 'review' or 'drop' threshold")
            if (
                layer is not None
                and layer.review is not None
                and layer.drop is not None
                and layer.review >= layer.drop
            ):
                raise ValueError(f"'{name}.review' must be below '{name}.drop'")
        if not exact and ngram is None and embedding is None:
            raise ValueError("enable at least one of 'exact', 'ngram', or 'embedding'")
        return cls(
            text_fields=text_fields,
            text_after=text_after,
            id_field=id_field,
            references=tuple(references),
            exact=exact,
            ngram=ngram,
            ngram_size=ngram_size,
            ngram_top_k=ngram_top_k,
            embedding=embedding,
            embedding_layer=embedding_layer,
            top_k=top_k,
            cache_dir=config.get("cache_dir"),
        )

    def manifest_settings(self) -> dict[str, Any]:
        """Settings that change results, with each reference file's digest."""
        settings = asdict(self)
        settings.pop("cache_dir")
        if self.embedding is not None:
            settings["embedding"] = self.embedding.identity()
        settings["reference_digests"] = {
            reference.name: file_evidence(Path(reference.path))["sha256"]
            for reference in self.references
        }
        return settings


@dataclass(frozen=True)
class ReferenceText:
    reference: str
    id: Any
    text: str


def load_references(settings: DecontaminationSettings) -> list[ReferenceText]:
    """Every reference record's text; a record without text fails the run."""
    texts = []
    for reference in settings.references:
        rows = missing = 0
        with open(reference.path, encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record_text(record, reference.text_fields, text_after=reference.text_after)
                if text:
                    identifier = (
                        field_value(record, reference.id_field) if reference.id_field else None
                    )
                    texts.append(
                        ReferenceText(
                            reference.name, rows if identifier is None else identifier, text
                        )
                    )
                else:
                    missing += 1
                rows += 1
        if not rows:
            raise ValueError(f"reference set {reference.name!r} has no records")
        if missing:
            raise ValueError(
                f"reference set {reference.name!r}: {missing:,} of {rows:,} records have no "
                f"text in {list(reference.text_fields)}"
            )
    return texts


def _chunk_dir(output_dir: Path, chunk: int) -> Path:
    return output_dir / "chunks" / f"{chunk:05d}"


def lexical_completion(output_dir: Path, chunk: int) -> Path:
    return _chunk_dir(output_dir, chunk) / "lexical.done"


def embedding_completion(output_dir: Path, chunk: int) -> Path:
    return _chunk_dir(output_dir, chunk) / "embedding.done"


def _hits_path(output_dir: Path, chunk: int, layer: str) -> Path:
    return _chunk_dir(output_dir, chunk) / f"{layer}.jsonl"


def _chunk_texts(
    manifest: Mapping[str, Any],
    chunk: int,
    settings: DecontaminationSettings,
) -> dict[int, str]:
    start, end = manifest["ranges"][chunk]
    texts = {}
    for index, record in iter_rows(manifest, start, end):
        text = record_text(record, settings.text_fields, text_after=settings.text_after)
        if text:
            texts[index] = text
    return texts


def _write_hits(path: Path, hits: list[dict[str, Any]]) -> None:
    hits.sort(key=lambda hit: (hit["row"], hit["layer"], hit["reference_index"]))
    with atomic_output(path) as stream:
        for hit in hits:
            stream.write(json.dumps(hit, ensure_ascii=False, default=str) + "\n")


def _lexical_hits(
    texts: Mapping[int, str],
    references: Sequence[ReferenceText],
    settings: DecontaminationSettings,
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    if settings.exact:
        by_key: dict[str, list[int]] = defaultdict(list)
        for position, reference in enumerate(references):
            by_key[exact_key(reference.text)].append(position)
        for index, text in texts.items():
            for position in by_key.get(exact_key(text), []):
                hits.append(_hit(index, "exact", 1.0, position, references[position]))
    if settings.ngram is None or not texts:
        return hits
    size, floor = settings.ngram_size, settings.ngram.floor
    # A reference shorter than `size` words is indexed by all of its words, so a
    # row containing it matches.
    by_width: defaultdict[int, defaultdict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    totals: list[int] = []
    for position, reference in enumerate(references):
        tokens = words(reference.text)
        width = min(size, len(tokens))
        shingles = _shingles(tokens, width) if tokens else set()
        totals.append(len(shingles))
        for shingle in shingles:
            by_width[width][shingle].append(position)
    for row, text in texts.items():
        tokens = words(text)
        shared: Counter[int] = Counter()
        for width, table in by_width.items():
            for shingle in _shingles(tokens, width):
                shared.update(table.get(shingle, ()))
        matches = [
            (count / totals[position], position)
            for position, count in shared.items()
            if count / totals[position] >= floor
        ]
        matches.sort(key=lambda match: (-match[0], match[1]))
        for share, position in matches[: settings.ngram_top_k]:
            hits.append(_hit(row, "ngram", share, position, references[position]))
    return hits


def _shingles(tokens: Sequence[str], width: int) -> set[str]:
    return {" ".join(tokens[start : start + width]) for start in range(len(tokens) - width + 1)}


def _hit(
    row: int, layer: str, score: float, position: int, reference: ReferenceText
) -> dict[str, Any]:
    return {
        "row": row,
        "layer": layer,
        "score": score,
        "reference": reference.reference,
        "reference_id": reference.id,
        "reference_index": position,
    }


def _manifest(
    output_dir: Path,
    inputs: Sequence[Path],
    settings: DecontaminationSettings,
    num_chunks: int,
) -> dict[str, Any]:
    return prepare_manifest(output_dir, inputs, num_chunks, settings.manifest_settings())


def run_lexical(
    output_dir: Path,
    inputs: Sequence[Path],
    settings: DecontaminationSettings,
    num_chunks: int,
    chunk: int,
) -> None:
    """Exact and n-gram hits for one chunk."""
    manifest = _manifest(output_dir, inputs, settings, num_chunks)
    outputs = {"hits": _hits_path(output_dir, chunk, "lexical")}
    completion = lexical_completion(output_dir, chunk)
    if completion_matches(completion, manifest, outputs):
        logger.info(f"Chunk {chunk} lexical layers are complete; skipping")
        return
    completion.unlink(missing_ok=True)
    texts = _chunk_texts(manifest, chunk, settings)
    hits = _lexical_hits(texts, load_references(settings), settings)
    _write_hits(outputs["hits"], hits)
    publish_completion(completion, manifest, outputs, chunk=chunk, hits=len(hits))
    logger.info(f"Chunk {chunk}: {len(hits):,} lexical hits over {len(texts):,} rows")


def _read_hits(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _verdict(hit: Mapping[str, Any], settings: DecontaminationSettings) -> str | None:
    if hit["layer"] == "exact":
        return "drop"
    layer = settings.ngram if hit["layer"] == "ngram" else settings.embedding_layer
    return None if layer is None else layer.verdict(float(hit["score"]))


def _cached_embeddings(
    cache_dir: Path,
    kind: str,
    texts: Sequence[str],
    settings: EmbeddingSettings,
    encoder: Callable[[], Encoder],
) -> tuple[Any, Any]:
    """Load or compute window vectors, cached by model settings and text digest."""
    import numpy as np

    from nvflow.integrations.embeddings.encoders import embed_texts

    key = digest_value({"embedding": settings.identity(), "texts": list(texts)})
    path = cache_dir / kind / f"{key}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            return cached["vectors"], cached["owners"]
    vectors, owners = embed_texts(texts, settings, encoder())
    with atomic_output(path, "wb") as stream:
        np.savez(stream, vectors=vectors, owners=owners)
    return vectors, owners


def run_embed(
    output_dir: Path,
    inputs: Sequence[Path],
    settings: DecontaminationSettings,
    num_chunks: int,
    chunk: int,
) -> None:
    """Embedding nearest neighbors for every row of one chunk.

    Rows a lexical layer drops are embedded too, so the cached row vectors
    depend only on the rows and the model, and survive new references or
    thresholds.
    """
    if settings.embedding is None or settings.embedding_layer is None:
        raise ValueError("the embed step requires 'embedding' settings")
    from nvflow.integrations.embeddings.encoders import load_encoder, top_k

    manifest = _manifest(output_dir, inputs, settings, num_chunks)
    outputs = {"hits": _hits_path(output_dir, chunk, "embedding")}
    completion = embedding_completion(output_dir, chunk)
    if completion_matches(completion, manifest, outputs):
        logger.info(f"Chunk {chunk} embeddings are complete; skipping")
        return
    completion.unlink(missing_ok=True)
    texts = _chunk_texts(manifest, chunk, settings)
    references = load_references(settings)
    cache_dir = Path(settings.cache_dir) if settings.cache_dir else output_dir / "cache"
    embedding = settings.embedding
    encoder = cache(lambda: load_encoder(embedding))
    rows = sorted(texts)
    reference_vectors, reference_owners = _cached_embeddings(
        cache_dir, "references", [item.text for item in references], settings.embedding, encoder
    )
    hits: list[dict[str, Any]] = []
    if rows:
        row_vectors, row_owners = _cached_embeddings(
            cache_dir, "rows", [texts[index] for index in rows], settings.embedding, encoder
        )
        query_ids, neighbor_ids, scores = top_k(
            row_vectors, row_owners, reference_vectors, reference_owners, settings.top_k
        )
        floor = settings.embedding_layer.floor
        for query, neighbors, values in zip(query_ids, neighbor_ids, scores, strict=True):
            for neighbor, score in zip(neighbors, values, strict=True):
                if score >= floor:
                    hits.append(
                        _hit(
                            rows[int(query)],
                            "embedding",
                            float(score),
                            int(neighbor),
                            references[int(neighbor)],
                        )
                    )
    _write_hits(outputs["hits"], hits)
    publish_completion(completion, manifest, outputs, chunk=chunk, hits=len(hits))
    logger.info(f"Chunk {chunk}: {len(hits):,} embedding hits over {len(rows):,} rows")


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {name: output_dir / filename for name, filename in OUTPUT_NAMES.items()}


def _collect_hits(
    output_dir: Path,
    manifest: Mapping[str, Any],
    settings: DecontaminationSettings,
    num_chunks: int,
) -> dict[int, list[dict[str, Any]]]:
    hits_by_row: dict[int, list[dict[str, Any]]] = defaultdict(list)
    missing = []
    for chunk in range(num_chunks):
        steps = [("lexical", lexical_completion(output_dir, chunk))]
        if settings.embedding is not None:
            steps.append(("embedding", embedding_completion(output_dir, chunk)))
        for layer, path in steps:
            hits_file = {"hits": _hits_path(output_dir, chunk, layer)}
            if not completion_matches(path, manifest, hits_file):
                missing.append(f"{chunk}:{layer}")
                continue
            for hit in _read_hits(hits_file["hits"]):
                hits_by_row[int(hit["row"])].append(hit)
    if missing:
        raise ValueError(f"decontamination chunks are incomplete or stale: {missing}")
    return hits_by_row


def _evidence(
    hits: Sequence[Mapping[str, Any]],
    settings: DecontaminationSettings,
) -> list[dict[str, Any]]:
    evidence = []
    for hit in hits:
        verdict = _verdict(hit, settings)
        if verdict:
            evidence.append({**{k: v for k, v in hit.items() if k != "row"}, "verdict": verdict})
    return evidence


def _review_pairs(evidence: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The best-scoring review hit for each reference record."""
    best: dict[int, Mapping[str, Any]] = {}
    for item in evidence:
        key = int(item["reference_index"])
        if key not in best or item["score"] > best[key]["score"]:
            best[key] = item
    return [best[key] for key in sorted(best)]


def _candidate(
    index: int,
    record: Mapping[str, Any],
    item: Mapping[str, Any],
    settings: DecontaminationSettings,
    references: Sequence[ReferenceText],
) -> str:
    candidate = {
        "row": index,
        "id": field_value(record, settings.id_field) if settings.id_field else index,
        "text": record_text(record, settings.text_fields, text_after=settings.text_after),
        "reference": item["reference"],
        "reference_id": item["reference_id"],
        "reference_text": references[int(item["reference_index"])].text,
        "layer": item["layer"],
        "score": item["score"],
    }
    return json.dumps(candidate, ensure_ascii=False, default=str) + "\n"


def run_merge(
    output_dir: Path,
    inputs: Sequence[Path],
    settings: DecontaminationSettings,
    num_chunks: int,
) -> dict[str, Any]:
    """Apply every chunk's hits: write kept, dropped, and review-candidate records."""
    manifest = _manifest(output_dir, inputs, settings, num_chunks)
    outputs = output_paths(output_dir)
    completion = output_dir / COMPLETION_NAME
    if completion_matches(completion, manifest, outputs):
        logger.info(f"{completion} is complete; skipping")
        return dict(json.loads(outputs["report"].read_text(encoding="utf-8")))
    completion.unlink(missing_ok=True)
    hits_by_row = _collect_hits(output_dir, manifest, settings, num_chunks)
    references = load_references(settings)
    counts: Counter[tuple[str, str, str]] = Counter()
    kept = dropped_rows = candidates = 0
    with (
        atomic_output(outputs["kept"]) as kept_stream,
        atomic_output(outputs["dropped"]) as dropped,
        atomic_output(outputs["candidates"]) as review,
    ):
        for index, record in iter_rows(manifest):
            evidence = _evidence(hits_by_row.get(index, []), settings)
            for item in evidence:
                counts[(item["reference"], item["layer"], item["verdict"])] += 1
            if any(item["verdict"] == "drop" for item in evidence):
                matches = [
                    {key: value for key, value in item.items() if key != "reference_index"}
                    for item in evidence
                ]
                row = {**record, "_contamination": matches}
                dropped.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                dropped_rows += 1
                continue
            kept_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
            for item in _review_pairs(evidence):
                review.write(_candidate(index, record, item, settings, references))
                candidates += 1
    report = {
        "input_rows": int(manifest["total_rows"]),
        "kept_rows": kept,
        "dropped_rows": dropped_rows,
        "review_candidates": candidates,
        "reference_rows": dict(Counter(item.reference for item in references)),
        "hits": [
            {"reference": reference, "layer": layer, "verdict": verdict, "count": count}
            for (reference, layer, verdict), count in sorted(counts.items())
        ],
        "settings": manifest["settings"],
    }
    atomic_write_json(outputs["report"], report)
    publish_completion(completion, manifest, outputs)
    logger.info(
        f"Kept {kept:,} of {report['input_rows']:,} rows, dropped {dropped_rows:,}, "
        f"{candidates:,} pairs to review"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=["lexical", "embed", "merge", "run"])
    parser.add_argument("inputs", nargs="+", type=Path, help="Input JSONL files, read in order")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--settings", required=True, help="data.decontaminate options as JSON")
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk", type=int, help="Chunk index for the lexical and embed steps")
    args = parser.parse_args(argv)
    settings = DecontaminationSettings.from_config(json.loads(args.settings))
    if args.step in {"lexical", "embed"}:
        if args.chunk is None or not 0 <= args.chunk < args.num_chunks:
            parser.error(f"the {args.step} step requires --chunk in 0..{args.num_chunks - 1}")
        step = run_lexical if args.step == "lexical" else run_embed
        step(args.output_dir, args.inputs, settings, args.num_chunks, args.chunk)
        return 0
    if args.step == "run":
        for chunk in range(args.num_chunks):
            run_lexical(args.output_dir, args.inputs, settings, args.num_chunks, chunk)
            if settings.embedding is not None:
                run_embed(args.output_dir, args.inputs, settings, args.num_chunks, chunk)
    run_merge(args.output_dir, args.inputs, settings, args.num_chunks)
    return 0


if __name__ == "__main__":
    status = main()
    if "vllm" in sys.modules:
        # vLLM's engine threads can abort interpreter shutdown after the results
        # are written, failing a finished step: release the engine, then exit
        # without that shutdown.
        gc.collect()
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(status)
    sys.exit(status)
