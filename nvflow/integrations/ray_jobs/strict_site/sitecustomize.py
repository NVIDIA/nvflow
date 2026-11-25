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
"""Fail closed when a Ray-native child tries to start a fallback cluster."""

from __future__ import annotations

import atexit
import os

if os.environ.get("NVFLOW_REQUIRE_SUPPLIED_RAY") == "1":
    import ray

    _original_init = ray.init
    _exiting = False

    def _mark_exiting() -> None:
        global _exiting
        _exiting = True

    def _strict_init(*args, **kwargs):
        if _exiting:
            # Destructors that run after Ray's exit handler reconnect only to clean up
            # workers that exit with the job anyway.
            raise RuntimeError(
                "Ray already shut down as this process exits; NVFlow does not reconnect"
            )
        address = kwargs.get("address")
        if address is None and not args:
            kwargs["address"] = "auto"
        try:
            result = _original_init(*args, **kwargs)
        except Exception as exc:
            # Callers often print only the message, so it carries Ray's reason.
            raise RuntimeError(
                "NVFlow Ray-native workload could not attach to supplied cluster: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        # Registered after the exit handler ray.init just added, so it runs first.
        atexit.unregister(_mark_exiting)
        atexit.register(_mark_exiting)
        expected = os.environ.get("NVFLOW_EXPECTED_GCS_ADDRESS")
        actual = ray.get_runtime_context().gcs_address
        if expected and actual != expected:
            ray.shutdown()
            raise RuntimeError(f"NVFlow expected supplied GCS {expected!r}, attached to {actual!r}")
        return result

    ray.init = _strict_init
