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
"""Tests for hopchain_sample_instance_combinations sampling strategies.

Run: uv run pytest nvflow/recipes/multimodal/tests/test_hopchain_sample_instance_combinations.py -v
"""

from __future__ import annotations

import random
from types import SimpleNamespace

from nvflow.recipes.multimodal.utils.hopchain_sample_instance_combinations import (
    _bbox_iou,
    _instance_area_confidence_score,
    _instance_area_px,
    deduplicate_instances_by_bbox_iou,
    sample_instance_combinations_for_image,
    select_instances_balanced_by_category_then_area_confidence,
    select_instances_balanced_by_category_then_size,
    select_instances_by_area_confidence,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_instance(
    instance_id: str,
    category: str,
    *,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    image_width: int = 1000,
    image_height: int = 1000,
    confidence: float | None = 0.8,
    use_crop_context: bool = True,
) -> SimpleNamespace:
    """Build a minimal fake LocalizedInstance for testing."""
    bbox_xyxy = SimpleNamespace(x1=x1, y1=y1, x2=x2, y2=y2)
    # bbox_norm_1000: scale pixel coords to 0-1000 space
    bbox_norm_1000 = SimpleNamespace(
        x1=int(x1 / image_width * 1000),
        y1=int(y1 / image_height * 1000),
        x2=int(x2 / image_width * 1000),
        y2=int(y2 / image_height * 1000),
    )
    crop_bbox_context = (
        SimpleNamespace(image_width=image_width, image_height=image_height)
        if use_crop_context
        else None
    )
    return SimpleNamespace(
        instance_id=instance_id,
        category=category,
        bbox_xyxy=bbox_xyxy,
        bbox_norm_1000=bbox_norm_1000,
        crop_bbox_context=crop_bbox_context,
        confidence=confidence,
    )


# ── _instance_area_confidence_score ──────────────────────────────────────────


class TestAreaConfidenceScore:
    def _score(self, inst, max_area_px=None, area_weight=1.0, confidence_weight=0.0, null_conf=0.5):
        if max_area_px is None:
            max_area_px = _instance_area_px(inst) or 1.0
        return _instance_area_confidence_score(
            inst,
            area_weight=area_weight,
            confidence_weight=confidence_weight,
            null_confidence_fallback=null_conf,
            max_area_px=max_area_px,
        )

    def test_largest_instance_gets_norm_area_1(self):
        # When max_area_px equals the instance area, norm_area = 1.0
        inst = make_instance("i1", "cat", x1=0, y1=0, x2=500, y2=500, confidence=0.0)
        score = self._score(inst, max_area_px=500 * 500, area_weight=1.0, confidence_weight=0.0)
        assert abs(score - 1.0) < 1e-6

    def test_smaller_instance_gets_proportional_score(self):
        # Instance area = 100x100 = 10000; max = 200x200 = 40000 → norm_area = 0.25
        inst = make_instance("i1", "cat", x1=0, y1=0, x2=100, y2=100, confidence=0.0)
        score = self._score(inst, max_area_px=40_000, area_weight=1.0, confidence_weight=0.0)
        assert abs(score - 0.25) < 1e-6

    def test_uses_null_confidence_fallback(self):
        inst = make_instance("i1", "cat", x1=0, y1=0, x2=100, y2=100, confidence=None)
        score = _instance_area_confidence_score(
            inst,
            area_weight=0.0,
            confidence_weight=1.0,
            null_confidence_fallback=0.7,
            max_area_px=10_000,
        )
        assert abs(score - 0.7) < 1e-6

    def test_combined_score_equal_weights(self):
        # Instance is the largest (norm_area=1.0), confidence=0.8
        # score = 0.5*1.0 + 0.5*0.8 = 0.9
        inst = make_instance("i1", "cat", x1=0, y1=0, x2=200, y2=200, confidence=0.8)
        score = self._score(inst, max_area_px=200 * 200, area_weight=0.5, confidence_weight=0.5)
        assert abs(score - 0.9) < 1e-6

    def test_area_clamped_to_1(self):
        # max_area_px smaller than instance area → norm_area clamped to 1.0
        inst = make_instance("i1", "cat", x1=0, y1=0, x2=500, y2=500, confidence=0.0)
        score = self._score(inst, max_area_px=100, area_weight=1.0, confidence_weight=0.0)
        assert score <= 1.0

    def test_zero_max_area_returns_zero_area_component(self):
        inst = make_instance("i1", "cat", x1=100, y1=100, x2=101, y2=101, confidence=0.9)
        score = _instance_area_confidence_score(
            inst,
            area_weight=0.5,
            confidence_weight=0.5,
            null_confidence_fallback=0.5,
            max_area_px=0,
        )
        # area component = 0, confidence component = 0.5 * 0.9 = 0.45
        assert abs(score - 0.45) < 1e-6


# ── IOU deduplication ─────────────────────────────────────────────────────────


class TestIouDeduplication:
    def test_bbox_iou_uses_intersection_over_union(self):
        inst_a = make_instance("a", "icon", x1=0, y1=0, x2=100, y2=100)
        inst_b = make_instance("b", "logo", x1=50, y1=50, x2=150, y2=150)

        assert abs(_bbox_iou(inst_a, inst_b) - (2500 / 17500)) < 1e-6

    def test_dedup_keeps_earlier_ranked_instance(self):
        instances = [
            make_instance("keep", "icon", x1=10, y1=10, x2=110, y2=110),
            make_instance("drop", "logo", x1=12, y1=12, x2=112, y2=112),
            make_instance("other", "button", x1=300, y1=300, x2=360, y2=360),
        ]

        deduped, dropped = deduplicate_instances_by_bbox_iou(instances, iou_threshold=0.9)

        assert [instance.instance_id for instance in deduped] == ["keep", "other"]
        assert dropped == 1

    def test_dedup_refills_from_later_ranked_instances(self):
        instances = [
            make_instance("keep", "icon", x1=10, y1=10, x2=110, y2=110),
            make_instance("drop", "logo", x1=12, y1=12, x2=112, y2=112),
            make_instance("refill", "button", x1=300, y1=300, x2=360, y2=360),
        ]

        deduped, dropped = deduplicate_instances_by_bbox_iou(
            instances,
            iou_threshold=0.9,
            max_instances=2,
        )

        assert [instance.instance_id for instance in deduped] == ["keep", "refill"]
        assert dropped == 1

    def test_dedup_can_be_disabled(self):
        instances = [
            make_instance("a", "icon", x1=0, y1=0, x2=100, y2=100),
            make_instance("b", "logo", x1=0, y1=0, x2=100, y2=100),
        ]

        deduped, dropped = deduplicate_instances_by_bbox_iou(instances, iou_threshold=None)

        assert deduped == instances
        assert dropped == 0


# ── select_instances_balanced_by_category_then_size (baseline) ───────────────


class TestStrategyA:
    def test_no_capping_when_under_limit(self):
        instances = [
            make_instance(f"i{i}", "cat", x1=0, y1=0, x2=10 + i, y2=10 + i) for i in range(5)
        ]
        selected, capped = select_instances_balanced_by_category_then_size(
            instances, max_instances_considered_per_image=12, max_instances_per_category=2
        )
        assert selected == instances
        assert capped is False

    def test_capping_triggered_above_limit(self):
        instances = [
            make_instance(f"i{i}", f"cat{i}", x1=0, y1=0, x2=10 + i, y2=10 + i) for i in range(15)
        ]
        selected, capped = select_instances_balanced_by_category_then_size(
            instances, max_instances_considered_per_image=12, max_instances_per_category=2
        )
        assert len(selected) == 12
        assert capped is True

    def test_accepts_extra_kwargs(self):
        """Backward-compat: extra kwargs must not raise."""
        instances = [
            make_instance(f"i{i}", "cat", x1=0, y1=0, x2=10 + i, y2=10 + i) for i in range(3)
        ]
        selected, _ = select_instances_balanced_by_category_then_size(
            instances,
            max_instances_considered_per_image=12,
            max_instances_per_category=2,
            area_weight=0.5,
            confidence_weight=0.5,
        )
        assert len(selected) == 3


# ── select_instances_by_area_confidence (Strategy B) ─────────────────────────


class TestStrategyB:
    def test_no_capping_when_under_limit(self):
        instances = [
            make_instance(f"i{i}", "cat", x1=0, y1=0, x2=10 + i, y2=10 + i) for i in range(5)
        ]
        selected, capped = select_instances_by_area_confidence(
            instances,
            max_instances_considered_per_image=12,
            area_weight=0.5,
            confidence_weight=0.5,
            null_confidence_fallback=0.5,
        )
        assert selected == instances
        assert capped is False

    def test_selects_top_n_by_score(self):
        # Instances with varying confidence: highest confidence should be preferred
        # when area is equal.
        instances = [
            make_instance("low", "cat", x1=0, y1=0, x2=100, y2=100, confidence=0.3),
            make_instance("mid", "cat", x1=0, y1=0, x2=100, y2=100, confidence=0.6),
            make_instance("high", "cat", x1=0, y1=0, x2=100, y2=100, confidence=0.9),
        ]
        selected, capped = select_instances_by_area_confidence(
            instances,
            max_instances_considered_per_image=2,
            area_weight=0.0,
            confidence_weight=1.0,
            null_confidence_fallback=0.5,
        )
        ids = {i.instance_id for i in selected}
        assert "high" in ids
        assert "mid" in ids
        assert "low" not in ids
        assert capped is True

    def test_confidence_threshold_filters_instances(self):
        instances = [
            make_instance("keep1", "a", x1=0, y1=0, x2=100, y2=100, confidence=0.8),
            make_instance("keep2", "b", x1=0, y1=0, x2=100, y2=100, confidence=0.7),
            make_instance("drop", "c", x1=0, y1=0, x2=100, y2=100, confidence=0.4),
        ]
        # cap=10 so no capping, but threshold removes "drop"
        selected, _ = select_instances_by_area_confidence(
            instances,
            max_instances_considered_per_image=10,
            area_weight=0.5,
            confidence_weight=0.5,
            min_confidence_threshold=0.6,
            null_confidence_fallback=0.5,
        )
        ids = {i.instance_id for i in selected}
        assert "drop" not in ids
        assert "keep1" in ids
        assert "keep2" in ids

    def test_null_confidence_instances_kept_when_no_threshold(self):
        instances = [
            make_instance("with_conf", "cat", x1=0, y1=0, x2=100, y2=100, confidence=0.8),
            make_instance("null_conf", "cat", x1=0, y1=0, x2=100, y2=100, confidence=None),
        ]
        selected, _ = select_instances_by_area_confidence(
            instances,
            max_instances_considered_per_image=10,
            area_weight=0.5,
            confidence_weight=0.5,
            min_confidence_threshold=None,
            null_confidence_fallback=0.5,
        )
        assert len(selected) == 2


# ── sample_instance_combinations_for_image ───────────────────────────────────


class TestSampleInstanceCombinationsForImage:
    def test_largest_first_keeps_the_original_max_size_preference(self):
        instances = [
            make_instance(f"i{i}", f"cat{i}", x1=0, y1=0, x2=100, y2=100) for i in range(10)
        ]
        sampled = sample_instance_combinations_for_image(
            instances=instances,
            min_instances=5,
            max_instances=8,
            max_combinations_per_image=3,
            combination_size_strategy="largest_first",
            combination_size_weights={},
            per_image_random=random.Random("seed"),
        )
        assert [size for size, _, _ in sampled] == [8, 8, 8]

    def test_stratified_random_covers_sizes_before_weighted_fill(self):
        instances = [
            make_instance(f"i{i}", f"cat{i}", x1=0, y1=0, x2=100, y2=100) for i in range(10)
        ]
        sampled = sample_instance_combinations_for_image(
            instances=instances,
            min_instances=5,
            max_instances=8,
            max_combinations_per_image=6,
            combination_size_strategy="stratified_random",
            combination_size_weights={5: 1, 6: 2, 7: 3, 8: 3},
            per_image_random=random.Random("seed"),
        )
        sizes = [size for size, _, _ in sampled]
        assert len(sizes) == 6
        assert {5, 6, 7, 8} <= set(sizes)

    def test_weighted_random_does_not_force_each_size(self):
        instances = [
            make_instance(f"i{i}", f"cat{i}", x1=0, y1=0, x2=100, y2=100) for i in range(10)
        ]
        sampled = sample_instance_combinations_for_image(
            instances=instances,
            min_instances=3,
            max_instances=8,
            max_combinations_per_image=6,
            combination_size_strategy="weighted_random",
            combination_size_weights={3: 1, 4: 1, 5: 2, 6: 3, 7: 3, 8: 3},
            per_image_random=random.Random("seed"),
        )
        sizes = [size for size, _, _ in sampled]
        assert len(sizes) == 6
        assert set(sizes) != {3, 4, 5, 6, 7, 8}

    def test_stratified_random_respects_zero_weighted_sizes(self):
        instances = [
            make_instance(f"i{i}", f"cat{i}", x1=0, y1=0, x2=100, y2=100) for i in range(10)
        ]
        sampled = sample_instance_combinations_for_image(
            instances=instances,
            min_instances=5,
            max_instances=8,
            max_combinations_per_image=4,
            combination_size_strategy="stratified_random",
            combination_size_weights={6: 1},
            per_image_random=random.Random("seed"),
        )
        assert {size for size, _, _ in sampled} == {6}


# ── select_instances_balanced_by_category_then_area_confidence (Strategy C) ──


class TestStrategyC:
    def test_no_capping_when_under_limit(self):
        instances = [
            make_instance(f"i{i}", f"cat{i}", x1=0, y1=0, x2=100, y2=100) for i in range(5)
        ]
        selected, capped = select_instances_balanced_by_category_then_area_confidence(
            instances,
            max_instances_considered_per_image=12,
            max_instances_per_category=2,
            area_weight=0.5,
            confidence_weight=0.5,
            null_confidence_fallback=0.5,
        )
        assert selected == instances
        assert capped is False

    def test_ranks_even_when_under_limit(self):
        instances = [
            make_instance("low", "cat", x1=0, y1=0, x2=100, y2=100, confidence=0.1),
            make_instance("high", "cat", x1=0, y1=0, x2=100, y2=100, confidence=0.9),
        ]

        selected, capped = select_instances_balanced_by_category_then_area_confidence(
            instances,
            max_instances_considered_per_image=10,
            max_instances_per_category=None,
            area_weight=0.0,
            confidence_weight=1.0,
            null_confidence_fallback=0.5,
        )

        assert [instance.instance_id for instance in selected] == ["high", "low"]
        assert capped is False

    def test_can_build_uncapped_per_category_candidate_pool(self):
        instances = [
            make_instance(f"inst{i}", "cat", x1=0, y1=0, x2=100, y2=100, confidence=0.9 - i * 0.1)
            for i in range(4)
        ]

        selected, capped = select_instances_balanced_by_category_then_area_confidence(
            instances,
            max_instances_considered_per_image=4,
            max_instances_per_category=None,
            area_weight=0.0,
            confidence_weight=1.0,
            null_confidence_fallback=0.5,
        )

        assert [instance.instance_id for instance in selected] == [
            "inst0",
            "inst1",
            "inst2",
            "inst3",
        ]
        assert capped is False

    def test_respects_cap(self):
        instances = [
            make_instance(f"i{i}", f"cat{i}", x1=0, y1=0, x2=100, y2=100) for i in range(15)
        ]
        selected, capped = select_instances_balanced_by_category_then_area_confidence(
            instances,
            max_instances_considered_per_image=12,
            max_instances_per_category=2,
            area_weight=0.5,
            confidence_weight=0.5,
            null_confidence_fallback=0.5,
        )
        assert len(selected) == 12
        assert capped is True

    def test_category_diversity_preserved(self):
        # 5 categories, 4 instances each = 20 total; cap=10
        # Should select at least one from each of the 5 categories
        instances = []
        for cat_idx in range(5):
            for inst_idx in range(4):
                instances.append(
                    make_instance(
                        f"cat{cat_idx}_inst{inst_idx}",
                        f"cat{cat_idx}",
                        x1=0,
                        y1=0,
                        x2=100,
                        y2=100,
                        confidence=0.8,
                    )
                )
        selected, capped = select_instances_balanced_by_category_then_area_confidence(
            instances,
            max_instances_considered_per_image=10,
            max_instances_per_category=2,
            area_weight=0.5,
            confidence_weight=0.5,
            null_confidence_fallback=0.5,
        )
        selected_cats = {i.category for i in selected}
        assert len(selected_cats) == 5
        assert capped is True

    def test_prefers_higher_confidence_within_category(self):
        # One category, two instances — the high-confidence one should be in first pass
        instances = [
            make_instance("low_conf", "cat", x1=0, y1=0, x2=200, y2=200, confidence=0.3),
            make_instance("high_conf", "cat", x1=0, y1=0, x2=200, y2=200, confidence=0.9),
            # Pad to 13 to trigger capping
        ] + [
            make_instance(f"other{i}", f"other{i}", x1=0, y1=0, x2=50, y2=50, confidence=0.5)
            for i in range(11)
        ]
        selected, capped = select_instances_balanced_by_category_then_area_confidence(
            instances,
            max_instances_considered_per_image=12,
            max_instances_per_category=1,
            area_weight=0.0,
            confidence_weight=1.0,
            null_confidence_fallback=0.5,
        )
        ids = {i.instance_id for i in selected}
        assert "high_conf" in ids
        assert "low_conf" not in ids
        assert capped is True

    def test_confidence_threshold_applied_before_selection(self):
        instances = [
            make_instance("keep", "a", x1=0, y1=0, x2=100, y2=100, confidence=0.9),
            make_instance("drop", "b", x1=0, y1=0, x2=100, y2=100, confidence=0.4),
        ] + [
            make_instance(f"pad{i}", f"pad{i}", x1=0, y1=0, x2=100, y2=100, confidence=0.8)
            for i in range(11)
        ]
        selected, _ = select_instances_balanced_by_category_then_area_confidence(
            instances,
            max_instances_considered_per_image=12,
            max_instances_per_category=2,
            area_weight=0.5,
            confidence_weight=0.5,
            min_confidence_threshold=0.6,
            null_confidence_fallback=0.5,
        )
        ids = {i.instance_id for i in selected}
        assert "drop" not in ids
