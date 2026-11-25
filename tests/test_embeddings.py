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
"""Embedding settings, windows, and exact top-k search."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from nvflow.integrations.embeddings import encoders
from nvflow.integrations.embeddings.settings import EmbeddingSettings


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"backend": "faiss", "model": "m"}, "backend"),
        ({"backend": "vllm"}, "model"),
        ({"backend": "vllm", "model": "m", "dimensions": 0}, "dimensions"),
        ({"backend": "vllm", "model": "m", "dimensions": True}, "dimensions"),
        ({"backend": "vllm", "model": "m", "batch_size": True}, "batch_size"),
        ({"backend": "vllm", "model": "m", "pooling": "mean"}, "unknown"),
        ({"backend": "vllm", "model": "m", "chat_template": "yes"}, "chat_template"),
        ({"backend": "sentence_transformers", "model": "m", "chat_template": True}, "vllm"),
    ],
)
def test_settings_fail_closed(config: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EmbeddingSettings.from_config(config)


def test_identity_ignores_settings_that_do_not_change_vectors() -> None:
    first = EmbeddingSettings.from_config({"backend": "vllm", "model": "m", "batch_size": 8})
    second = EmbeddingSettings.from_config(
        {"backend": "vllm", "model": "m", "batch_size": 512, "device": "cuda"}
    )
    prefixed = EmbeddingSettings.from_config({"backend": "vllm", "model": "m", "prefix": "query: "})

    assert first.identity() == second.identity()
    assert first.identity() != prefixed.identity()


def test_windows_cover_long_text_with_overlap() -> None:
    text = "abcdefghijklmnopqrstuvwxy"

    assert encoders.windows("short", 10) == ["short"]
    assert encoders.windows(text, 10) == [text[0:10], text[7:17], text[14:24], text[21:]]


def test_normalize_truncates_then_scales_to_unit_length() -> None:
    vectors = encoders.normalize(np.array([[3.0, 4.0, 12.0]]), dimensions=2)

    assert vectors.shape == (1, 2)
    np.testing.assert_allclose(vectors, [[0.6, 0.8]], rtol=1e-6)


def test_top_k_scores_texts_by_their_best_window() -> None:
    rng = np.random.default_rng(0)
    queries = encoders.normalize(rng.normal(size=(5, 8)))
    references = encoders.normalize(rng.normal(size=(6, 8)))
    query_owners = np.array([0, 0, 1, 2, 2])
    reference_owners = np.array([0, 1, 1, 2, 3, 3])

    ids, neighbors, scores = encoders.top_k(
        queries, query_owners, references, reference_owners, k=2, batch_texts=2
    )

    similarity = queries @ references.T
    assert ids.tolist() == [0, 1, 2]
    for row, text in enumerate(ids):
        best = {
            reference: similarity[query_owners == text][:, reference_owners == reference].max()
            for reference in range(4)
        }
        expected = sorted(best, key=lambda reference: -best[reference])[:2]
        assert neighbors[row].tolist() == expected
        np.testing.assert_allclose(
            scores[row], [best[reference] for reference in expected], rtol=1e-6
        )


def test_embed_texts_adds_the_prefix_and_batches() -> None:
    settings = EmbeddingSettings.from_config(
        {"backend": "vllm", "model": "m", "prefix": "query: ", "batch_size": 2, "max_chars": 4}
    )
    batches: list[list[str]] = []

    def encode(texts: list[str]) -> np.ndarray:
        batches.append(texts)
        return np.ones((len(texts), 3), dtype=np.float32)

    vectors, owners = encoders.embed_texts(["abc", "abcdefg"], settings, encode)

    assert batches == [["query: abc", "query: abcd"], ["query: defg"]]
    assert owners.tolist() == [0, 1, 1]
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-6)


@pytest.mark.parametrize("bad", [np.nan, 0.0])
def test_embed_texts_rejects_vectors_without_a_direction(bad: float) -> None:
    settings = EmbeddingSettings.from_config({"backend": "vllm", "model": "m"})

    def encode(texts: list[str]) -> np.ndarray:
        vectors = np.ones((len(texts), 3), dtype=np.float32)
        vectors[0] = bad
        return vectors

    with pytest.raises(ValueError, match="not finite or have zero length"):
        encoders.embed_texts(["a", "b"], settings, encode)


def test_embed_texts_rejects_a_missing_vector() -> None:
    settings = EmbeddingSettings.from_config({"backend": "vllm", "model": "m"})

    with pytest.raises(ValueError, match="returned 1 vectors for 2 texts"):
        encoders.embed_texts(["a", "b"], settings, lambda texts: np.ones((1, 3), np.float32))


def test_chat_template_puts_the_prefix_in_the_system_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    class LLM:
        def __init__(self, **options: Any) -> None:
            calls["options"] = options

        def embed(self, prompts: list[str], **kwargs: Any) -> list[Any]:
            calls["embed"] = (prompts, kwargs)
            return [SimpleNamespace(outputs=SimpleNamespace(embedding=[1.0, 0.0])) for _ in prompts]

    class Processor:
        def apply_chat_template(self, messages: list[dict[str, str]], *, tokenize: bool) -> str:
            assert tokenize is False
            return "|".join(f"{message['role']}={message['content']}" for message in messages)

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=LLM))
    monkeypatch.setitem(sys.modules, "vllm.config", SimpleNamespace(PoolerConfig=dict))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=lambda _model: Processor())),
    )
    settings = EmbeddingSettings.from_config(
        {
            "backend": "vllm",
            "model": "/models/embed",
            "prefix": "query:",
            "chat_template": True,
            "options": {"pooler_config": {"pooling_type": "MEAN"}},
        }
    )

    vectors, _owners = encoders.embed_texts(
        ["What was X?"], settings, encoders.load_encoder(settings)
    )

    assert calls["options"]["pooler_config"] == {"pooling_type": "MEAN"}
    assert calls["embed"] == (
        ["system=query:|user=What was X?"],
        {"tokenization_kwargs": {"add_special_tokens": False}},
    )
    assert vectors.shape == (1, 2)
