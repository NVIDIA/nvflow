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
"""Shared vLLM server argument helpers."""

import json
import shlex
from typing import Any

NON_VLLM_KEYS = frozenset(
    {
        "num_gpus",
        "server_nodes",
        "model_path",
        "base_url",
        "openai_base_url",
        "openai_model",
        "openai_api_key",
        "uses_reasoning_parser",
        "tensor_parallel_size",
        "trust_remote_code",
        "hf_config_overrides",
        "server_entrypoint",
        "num_samples_in_parallel",
        "dependent_jobs",
        "chat_template_kwargs",
    }
)
"""Keys in vLLM config dicts that are NOT ``vllm serve`` CLI flags.

:func:`vllm_serve_overrides` strips them, passing ``hf_config_overrides`` to
vLLM as ``--hf-overrides``.
"""


def vllm_serve_overrides(serve_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return the ``vllm serve`` flags in *serve_kwargs* for :func:`build_vllm_server_args`."""
    overrides = {key: value for key, value in serve_kwargs.items() if key not in NON_VLLM_KEYS}
    if hf_overrides := serve_kwargs.get("hf_config_overrides"):
        overrides["hf_overrides"] = hf_overrides
    return overrides


def build_vllm_server_args(overrides: dict[str, Any]) -> str:
    """Convert a dict of vLLM overrides into CLI flags.

    Every key in *overrides* is emitted as a flag -- the caller is responsible
    for stripping non-vLLM keys before calling this function.

    Underscores are converted to hyphens (``max_model_len`` → ``--max-model-len``).
    Boolean ``True`` emits a bare flag; ``False`` suppresses it.
    """
    parts: list[str] = []
    for key, value in overrides.items():
        cli_key = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                parts.append(cli_key)
        else:
            encoded = (
                json.dumps(value, separators=(",", ":"), sort_keys=True)
                if isinstance(value, dict | list)
                else str(value)
            )
            parts.append(f"{cli_key} {shlex.quote(encoded)}")
    return " ".join(parts)
