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
"""Tests for ``nvflow.lib.gpu_layout``."""

from __future__ import annotations

import pytest

from nvflow.lib.gpu_layout import GpuLayout, resolve_gpu_layout


class TestGpuLayout:
    def test_total_gpus_property(self) -> None:
        layout = GpuLayout(num_nodes=2, gpus_per_node=8)
        assert layout.total_gpus == 16


class TestResolveGpuLayoutTotalGpus:
    def test_divides_evenly_with_cluster_config(self) -> None:
        layout = resolve_gpu_layout({"total_gpus": 16}, {"gpus_per_node": 4})
        assert layout == GpuLayout(num_nodes=4, gpus_per_node=4)

    def test_uses_default_gpus_per_node_without_cluster_config(self) -> None:
        layout = resolve_gpu_layout({"total_gpus": 16})
        assert layout == GpuLayout(num_nodes=2, gpus_per_node=8)

    def test_not_evenly_divisible_raises(self) -> None:
        with pytest.raises(ValueError, match="not evenly divisible"):
            resolve_gpu_layout({"total_gpus": 10}, {"gpus_per_node": 8})

    def test_zero_total_gpus_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a positive integer"):
            resolve_gpu_layout({"total_gpus": 0}, {"gpus_per_node": 8})

    def test_negative_total_gpus_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a positive integer"):
            resolve_gpu_layout({"total_gpus": -8}, {"gpus_per_node": 8})


class TestResolveGpuLayoutGpusPerNode:
    def test_zero_gpus_per_node_raises_value_error_not_zero_division(self) -> None:
        with pytest.raises(ValueError, match="gpus_per_node.*must be a positive integer"):
            resolve_gpu_layout({"total_gpus": 16}, {"gpus_per_node": 0})

    def test_negative_gpus_per_node_raises(self) -> None:
        with pytest.raises(ValueError, match="gpus_per_node.*must be a positive integer"):
            resolve_gpu_layout({"total_gpus": 16}, {"gpus_per_node": -4})


class TestResolveGpuLayoutLegacyNumNodes:
    def test_num_nodes_with_explicit_num_gpus(self) -> None:
        layout = resolve_gpu_layout({"num_nodes": 3, "num_gpus": 4})
        assert layout == GpuLayout(num_nodes=3, gpus_per_node=4)

    def test_num_nodes_without_num_gpus_falls_back_to_gpus_per_node(self) -> None:
        layout = resolve_gpu_layout({"num_nodes": 3}, {"gpus_per_node": 2})
        assert layout == GpuLayout(num_nodes=3, gpus_per_node=2)

    def test_zero_num_nodes_raises(self) -> None:
        with pytest.raises(ValueError, match="num_nodes.*must be a positive integer"):
            resolve_gpu_layout({"num_nodes": 0})

    def test_zero_num_gpus_raises(self) -> None:
        with pytest.raises(ValueError, match="num_gpus.*must be a positive integer"):
            resolve_gpu_layout({"num_nodes": 2, "num_gpus": 0})


class TestResolveGpuLayoutDefault:
    def test_neither_field_defaults_to_single_node(self) -> None:
        layout = resolve_gpu_layout({})
        assert layout == GpuLayout(num_nodes=1, gpus_per_node=8)

    def test_neither_field_uses_cluster_gpus_per_node(self) -> None:
        layout = resolve_gpu_layout({}, {"gpus_per_node": 4})
        assert layout == GpuLayout(num_nodes=1, gpus_per_node=4)
