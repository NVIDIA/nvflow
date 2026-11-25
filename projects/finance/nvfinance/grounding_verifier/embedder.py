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
"""Lazy Hugging Face embedder for claim/evidence routing.

Default public model: ``sentence-transformers/all-MiniLM-L6-v2``
(Apache 2.0).  Model construction is strictly lazy — ``torch`` and
``transformers`` imports happen inside ``_ensure_loaded``, not at
module import time.

Uses the raw Transformers ``AutoTokenizer`` / ``AutoModel`` recipe with
mean pooling over ``last_hidden_state`` with attention mask, L2-normalized,
returning plain float lists — so no ``sentence-transformers`` runtime
dependency is needed.
"""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class HFEmbedder:
    """Transformers embedder with lazy model loading and mean pooling.

    Args:
        model_id: Hugging Face model ID.  Defaults to
            ``sentence-transformers/all-MiniLM-L6-v2``.
        revision: Git revision to pin.  ``None`` uses the latest
            available revision.  Production deployments should pin a
            specific revision fetched from Hugging Face Hub metadata
            (do not hardcode unverified hashes).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_EMBEDDING_MODEL,
        revision: str | None = None,
    ) -> None:
        self._model_id = model_id
        self._revision = revision
        self._model = None
        self._tokenizer = None
        self._dim: int | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        kwargs: dict = {}
        if self._revision:
            kwargs["revision"] = self._revision

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id, **kwargs)
        self._model = AutoModel.from_pretrained(self._model_id, **kwargs)
        self._model.eval()
        self._dim = self._model.config.hidden_size

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        assert self._dim is not None
        return self._dim

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        import torch

        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None

        encoded = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self._model(**encoded)
            token_embeddings = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"]
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            sentence_embeddings = sum_embeddings / sum_mask
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return [[float(x) for x in row] for row in sentence_embeddings.detach().cpu().tolist()]
