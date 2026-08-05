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
"""Utility functions and helpers.

Only lightweight, dependency-free helpers are re-exported here.  In
particular, the JSONL helpers in :mod:`nvflow.utils.jsonl` import
``orjson`` at module top-level and MUST be imported directly via
``from nvflow.utils.jsonl import ...``.  Re-exporting them from this
package would force every consumer of :func:`setup_logger` (including
standalone workers like :mod:`nvflow.lib.rl.create_overlay`, which run
inside container images that do NOT ship ``orjson`` such as the vLLM
server container) to pay the ``orjson`` import cost -- and crash on
``ModuleNotFoundError`` when the dep is absent.

If you need the JSONL helpers, import them explicitly::

    from nvflow.utils.jsonl import iter_jsonl, write_jsonl, write_stats_json
"""

from nvflow.utils.logging_setup import setup_logger

__all__ = [
    "setup_logger",
]
