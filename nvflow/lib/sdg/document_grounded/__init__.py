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
"""Document-Grounded Synthetic Data Generation (DG-SDG) library.

Submodules:
    metadata    -- shared metadata nesting utilities
    chunking    -- HTML-to-Markdown chunking engine
    sampling    -- weighted distribution sampling
    preprocess  -- QA pipeline preprocessing (question/answer/verify stages)
    genselect   -- merge multi-seed answers and select best via judgment
    evaluate    -- parse LLM evaluation responses (answerable/correct tags)
    aggregate   -- multi-seed evaluation aggregation with majority voting
    difficulty  -- difficulty estimation via small-model probing
    postprocess -- final record cleaning and subset splitting

Worker scripts (evaluate, aggregate, difficulty, genselect, preprocess,
postprocess) run inside the Slurm container with ``PYTHONPATH=/workspace``.
This __init__.py is intentionally kept import-free so that
``python -m nvflow.lib.sdg.document_grounded.<worker>`` does not trigger the
dependency chain.
"""
