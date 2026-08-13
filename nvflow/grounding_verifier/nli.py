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
"""Lazy Hugging Face NLI scorer for claim-evidence entailment checking.

Default public model: ``MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`` (MIT).

Validates the exact unique label set (entailment, neutral, contradiction)
from the model config — fails on duplicates, extras, or missing labels.
Model construction is strictly lazy — the ``transformers`` import happens
inside ``__init__``, not at module import time.

No lexical/hash runtime fallback.  If model loading fails, the caller
must catch the exception and produce an ``unavailable`` decision.
"""

from __future__ import annotations

from nvflow.grounding_verifier.types import NLIResult

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

_VALID_NLI_LABELS = frozenset({"entailment", "neutral", "contradiction"})


def _normalize_nli_label(raw_label: str) -> str:
    """Normalize a model config label to one of the three canonical NLI classes.

    Validates the *actual* label name from the model config
    (``config.id2label``).  Never assumes arbitrary ``LABEL_0`` ordering.

    Raises ``ValueError`` if the label does not match entailment,
    neutral, or contradiction (case-insensitive).
    """
    normalized = raw_label.strip().lower()
    if normalized in _VALID_NLI_LABELS:
        return normalized
    raise ValueError(
        f"Unknown NLI label '{raw_label}': expected one of "
        f"entailment, neutral, contradiction (case-insensitive). "
        "Refusing to assume arbitrary label ordering."
    )


class HFNLI:
    """DeBERTa sequence-classification NLI scorer with lazy model loading.

    Args:
        model_id: Hugging Face model ID.
        revision: Git revision to pin.  ``None`` uses the latest
            available revision.  Production deployments should pin a
            specific revision fetched from Hugging Face Hub metadata
            (do not hardcode unverified hashes).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_NLI_MODEL,
        revision: str | None = None,
    ) -> None:
        self._model_id = model_id
        self._revision = revision
        self._model = None
        self._tokenizer = None
        self._label_map: dict[int, str] | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        kwargs: dict = {}
        if self._revision:
            kwargs["revision"] = self._revision

        # Load into locals first — do not publish _model / _tokenizer /
        # _label_map until ALL validation succeeds.  This ensures that
        # a failed duplicate, missing, or incomplete id2label does not
        # poison state: subsequent evaluations re-attempt loading and
        # fail again rather than reusing partial/cached state.
        tokenizer = AutoTokenizer.from_pretrained(self._model_id, **kwargs)
        model = AutoModelForSequenceClassification.from_pretrained(self._model_id, **kwargs)
        model.eval()

        # Build label map from model config — validate actual label names,
        # never assume arbitrary LABEL_0 ordering.
        id2label = model.config.id2label
        label_map: dict[int, str] = {}
        for idx, label in id2label.items():
            normalized = _normalize_nli_label(str(label))
            label_map[int(idx)] = normalized
        # Validate exact unique label set: exactly 3 labels, one each
        # of entailment / neutral / contradiction, no duplicates,
        # extras, or missing.
        seen_labels: list[str] = []
        for idx in sorted(label_map):
            seen_labels.append(label_map[idx])
        if len(seen_labels) != 3:
            raise ValueError(
                f"NLI model has {len(seen_labels)} labels; expected "
                f"exactly 3 (entailment, neutral, contradiction)."
            )
        label_set = set(seen_labels)
        if label_set != _VALID_NLI_LABELS:
            missing = _VALID_NLI_LABELS - label_set
            extra = label_set - _VALID_NLI_LABELS
            parts: list[str] = []
            if missing:
                parts.append(f"missing: {sorted(missing)}")
            if extra:
                parts.append(f"extra: {sorted(extra)}")
            if len(label_set) < len(seen_labels):
                parts.append("duplicate labels detected")
            raise ValueError("NLI label set validation failed: " + "; ".join(parts))

        # Publish only after all validation succeeds.
        self._tokenizer = tokenizer
        self._model = model
        self._label_map = label_map

    def score(self, *, premise: str, hypothesis: str) -> NLIResult:
        import torch

        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._label_map is not None

        # NLI convention: premise is the evidence (premise), hypothesis
        # is the claim to verify.
        encoded = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            logits = self._model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[0]

        probabilities: list[tuple[str, float]] = []
        for idx in range(len(probs)):
            label = self._label_map[int(idx)]
            probabilities.append((label, float(probs[idx])))

        best_idx = int(probs.argmax().item())
        best_label = self._label_map[best_idx]
        best_score = float(probs[best_idx].item())

        return NLIResult(
            label=best_label,
            score=best_score,
            probabilities=tuple(probabilities),
        )
