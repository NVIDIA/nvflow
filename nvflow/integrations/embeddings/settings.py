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
"""Embedding backend settings, validated without importing any backend."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

BACKENDS = ("vllm", "sentence_transformers")
_KEYS = {
    "backend",
    "model",
    "prefix",
    "chat_template",
    "dimensions",
    "batch_size",
    "max_chars",
    "device",
    "options",
}


@dataclass(frozen=True)
class EmbeddingSettings:
    """One embedding model and how to call it.

    ``prefix`` is prepended to each text or, with ``chat_template`` (vLLM only),
    becomes the system message of the model's chat template. ``dimensions``
    keeps the leading vector components (Matryoshka truncation) before
    normalizing. Text longer than ``max_chars`` is split into overlapping
    windows, and its similarity is that of its best window. ``options`` go to
    ``vllm.LLM`` or ``SentenceTransformer``.
    """

    backend: str
    model: str
    prefix: str = ""
    chat_template: bool = False
    dimensions: int | None = None
    batch_size: int = 128
    max_chars: int | None = None
    device: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> EmbeddingSettings:
        unknown = sorted(set(config) - _KEYS)
        if unknown:
            raise ValueError(f"unknown embedding settings: {', '.join(unknown)}")
        backend = config.get("backend")
        if backend not in BACKENDS:
            raise ValueError(f"embedding backend must be one of {', '.join(BACKENDS)}")
        model = config.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("embedding model is required")
        for key in ("dimensions", "max_chars"):
            value = config.get(key)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"embedding {key} must be a positive integer")
        batch_size = config.get("batch_size", 128)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("embedding batch_size must be a positive integer")
        options = config.get("options") or {}
        if not isinstance(options, Mapping):
            raise ValueError("embedding options must be a mapping")
        chat_template = config.get("chat_template", False)
        if not isinstance(chat_template, bool):
            raise ValueError("embedding chat_template must be true or false")
        if chat_template and backend != "vllm":
            raise ValueError("embedding chat_template needs the vllm backend")
        return cls(
            backend=backend,
            model=model,
            prefix=str(config.get("prefix") or ""),
            chat_template=chat_template,
            dimensions=config.get("dimensions"),
            batch_size=batch_size,
            max_chars=config.get("max_chars"),
            device=config.get("device"),
            options=dict(options),
        )

    def identity(self) -> dict[str, Any]:
        """The settings that change vectors; batch size and device do not."""
        values = asdict(self)
        for key in ("batch_size", "device"):
            values.pop(key)
        values["options"] = dict(self.options)
        return values


__all__ = ["BACKENDS", "EmbeddingSettings"]
