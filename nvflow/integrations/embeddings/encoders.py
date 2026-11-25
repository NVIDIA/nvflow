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
"""Embedding backends, text windows, and exact top-k cosine search."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from nvflow.integrations.embeddings.settings import EmbeddingSettings

Encoder = Callable[[list[str]], np.ndarray]
BackendFactory = Callable[[EmbeddingSettings], Encoder]


def _vllm(settings: EmbeddingSettings) -> Encoder:
    from vllm import LLM

    options: dict[str, Any] = {"runner": "pooling", **settings.options}
    if isinstance(options.get("pooler_config"), Mapping):
        from vllm.config import PoolerConfig

        options["pooler_config"] = PoolerConfig(**options["pooler_config"])
    llm = LLM(model=settings.model, **options)

    def vectors(outputs: Sequence[Any]) -> np.ndarray:
        return np.asarray([output.outputs.embedding for output in outputs], dtype=np.float32)

    if not settings.chat_template:
        return lambda texts: vectors(llm.embed(texts))

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(settings.model)

    def encode(texts: list[str]) -> np.ndarray:
        prompts = [
            processor.apply_chat_template(
                [{"role": "system", "content": settings.prefix}, {"role": "user", "content": text}],
                tokenize=False,
            )
            for text in texts
        ]
        return vectors(llm.embed(prompts, tokenization_kwargs={"add_special_tokens": False}))

    return encode


def _sentence_transformers(settings: EmbeddingSettings) -> Encoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.model, device=settings.device, **settings.options)

    def encode(texts: list[str]) -> np.ndarray:
        vectors = model.encode(
            texts,
            batch_size=settings.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    return encode


_FACTORIES: dict[str, BackendFactory] = {
    "vllm": _vllm,
    "sentence_transformers": _sentence_transformers,
}


def load_encoder(settings: EmbeddingSettings) -> Encoder:
    try:
        factory = _FACTORIES[settings.backend]
    except KeyError:
        raise ValueError(f"no embedding backend named {settings.backend!r}") from None
    return factory(settings)


def windows(text: str, max_chars: int | None) -> list[str]:
    """Overlapping windows of at most *max_chars* characters covering *text*."""
    if max_chars is None or len(text) <= max_chars:
        return [text]
    step = max(1, max_chars * 3 // 4)
    pieces = []
    for start in range(0, len(text), step):
        pieces.append(text[start : start + max_chars])
        if start + max_chars >= len(text):
            break
    return pieces


def normalize(vectors: np.ndarray, dimensions: int | None = None) -> np.ndarray:
    """Keep the first *dimensions* components, then scale rows to unit length."""
    if dimensions is not None:
        vectors = vectors[:, :dimensions]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.asarray(vectors / np.maximum(norms, 1e-12), dtype=np.float32)


def embed_texts(
    texts: Sequence[str],
    settings: EmbeddingSettings,
    encoder: Encoder,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unit window vectors and the index of the text each window came from."""
    pieces: list[str] = []
    owners: list[int] = []
    for index, text in enumerate(texts):
        for piece in windows(text, settings.max_chars):
            pieces.append(piece if settings.chat_template else settings.prefix + piece)
            owners.append(index)
    if not pieces:
        return np.zeros((0, settings.dimensions or 1), dtype=np.float32), np.zeros(0, np.int64)
    vectors: np.ndarray | None = None
    for start in range(0, len(pieces), settings.batch_size):
        batch = pieces[start : start + settings.batch_size]
        unit = normalize(encoder(batch), settings.dimensions)
        if len(unit) != len(batch):
            raise ValueError(f"the encoder returned {len(unit):,} vectors for {len(batch):,} texts")
        if not np.isfinite(unit).all() or not np.linalg.norm(unit, axis=1).all():
            raise ValueError("the encoder returned vectors that are not finite or have zero length")
        if vectors is None:
            vectors = np.empty((len(pieces), unit.shape[1]), dtype=np.float32)
        vectors[start : start + len(batch)] = unit
    assert vectors is not None
    return vectors, np.asarray(owners, dtype=np.int64)


def _starts(owners: np.ndarray) -> np.ndarray:
    if not len(owners):
        return np.zeros(0, dtype=np.int64)
    if np.any(owners[1:] < owners[:-1]):
        raise ValueError("windows must be grouped by text, in text order")
    return np.flatnonzero(np.r_[True, owners[1:] != owners[:-1]])


def top_k(
    queries: np.ndarray,
    query_owners: np.ndarray,
    references: np.ndarray,
    reference_owners: np.ndarray,
    k: int,
    *,
    batch_texts: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact cosine top-k between texts, each the best match over its windows.

    Returns the query text ids, and for each one the ids and scores of its *k*
    nearest reference texts, best first.
    """
    query_starts = _starts(query_owners)
    reference_starts = _starts(reference_owners)
    query_ids = query_owners[query_starts]
    reference_ids = reference_owners[reference_starts]
    k = min(k, len(reference_ids))
    if not len(query_ids) or k == 0:
        empty = np.zeros((len(query_ids), 0))
        return query_ids, empty.astype(np.int64), empty.astype(np.float32)
    bounds = np.r_[query_starts, len(query_owners)]
    neighbor_ids, neighbor_scores = [], []
    for first in range(0, len(query_ids), batch_texts):
        last = min(first + batch_texts, len(query_ids))
        window_scores = queries[bounds[first] : bounds[last]] @ references.T
        text_scores = np.maximum.reduceat(window_scores, reference_starts, axis=1)
        text_scores = np.maximum.reduceat(
            text_scores, query_starts[first:last] - bounds[first], axis=0
        )
        best = np.argpartition(-text_scores, k - 1, axis=1)[:, :k]
        best_scores = np.take_along_axis(text_scores, best, axis=1)
        order = np.argsort(-best_scores, axis=1, kind="stable")
        neighbor_ids.append(reference_ids[np.take_along_axis(best, order, axis=1)])
        neighbor_scores.append(np.take_along_axis(best_scores, order, axis=1))
    return (
        query_ids,
        np.concatenate(neighbor_ids),
        np.concatenate(neighbor_scores).astype(np.float32),
    )


__all__ = [
    "Encoder",
    "embed_texts",
    "load_encoder",
    "normalize",
    "top_k",
    "windows",
]
