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
"""Patched vLLM entrypoint for standalone ``vllm serve`` processes.

Applies runtime workarounds **before** vLLM is imported, then builds and
runs the ``vllm.entrypoints.openai.api_server`` command directly.

The upstream ``nemo_skills.inference.server.serve_vllm`` hardcodes
``tensor_parallel_size = num_gpus * num_nodes``, which assumes all GPUs
are for TP.  This entrypoint replaces that with the formula::

    tp = (num_gpus * num_nodes) // dp_size

When ``--data-parallel-size`` is absent, ``dp_size`` defaults to 1 and the
formula gives the same result as the original code.  This is model-agnostic
and backward-compatible.

Tested with vLLM 0.17.1 and 0.18.1.
Other vLLM versions are safe -- patches skip gracefully if the expected
code snippets are not found, and vLLM starts normally unpatched.

Active workarounds
------------------
WORKAROUND(vllm-0.17-hermes)
    Patches ``vllm/tool_parsers/hermes_tool_parser.py`` on disk so that
    ``Hermes2ProToolParser.__init__`` caches tokenizer encode/decode results
    behind a ``threading.Lock``.  Fixes ``RuntimeError: Already borrowed``
    under concurrent chat-completion requests with tool calling enabled.
    Remove when: vLLM ships fix from https://github.com/vllm-project/vllm/pull/35034

WORKAROUND(harmony-aarch64)
    Pre-downloads tiktoken vocab files for gpt-oss models on aarch64.
    The ``openai_harmony`` Rust binary cannot download them at runtime.
    Remove when: openai_harmony ships a fixed aarch64 binary.
    Tracking: https://github.com/openai/harmony/issues/71
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import urllib.request
from importlib.util import find_spec
from pathlib import Path

_TAG = "[serve_vllm_patched]"

# ---------------------------------------------------------------------------
# WORKAROUND(vllm-0.17-hermes) -- hermes tool parser thread-safety
#
# Identical logic to RL/nemo_rl/models/generation/vllm/vllm_worker.py but
# applied on disk before vLLM is imported (standalone ``vllm serve`` has no
# in-process hook).
#
# Remove this entire section when vLLM ships the upstream fix.
# ---------------------------------------------------------------------------

# --- Exact string snippets to locate and replace in hermes_tool_parser.py ---

_OLD_IMPORT = "import json\nfrom collections.abc import Sequence"

_NEW_IMPORT = "import json\nimport threading\nfrom collections.abc import Sequence"

_OLD_CLASS_LINE = "class Hermes2ProToolParser(ToolParser):"

_NEW_CLASS_LINE = (
    "class Hermes2ProToolParser(ToolParser):\n"
    "    _tokenizer_lock = threading.Lock()\n"
    "    _tokenizer_cache = {}"
)

_OLD_INIT = (
    "        self.tool_call_start_token_ids = self.model_tokenizer.encode(\n"
    "            self.tool_call_start_token, add_special_tokens=False\n"
    "        )\n"
    "        self.tool_call_end_token_ids = self.model_tokenizer.encode(\n"
    "            self.tool_call_end_token, add_special_tokens=False\n"
    "        )\n"
    "\n"
    "        self.tool_call_start_token_array = [\n"
    "            self.model_tokenizer.decode([token_id])\n"
    "            for token_id in self.tool_call_start_token_ids\n"
    "        ]\n"
    "\n"
    "        self.tool_call_end_token_array = [\n"
    "            self.model_tokenizer.decode([token_id])\n"
    "            for token_id in self.tool_call_end_token_ids\n"
    "        ]"
)

_NEW_INIT = (
    "        _tid = id(self.model_tokenizer)\n"
    "        if _tid in Hermes2ProToolParser._tokenizer_cache:\n"
    "            _cached = Hermes2ProToolParser._tokenizer_cache[_tid]\n"
    "            self.tool_call_start_token_ids = _cached['start_ids']\n"
    "            self.tool_call_end_token_ids = _cached['end_ids']\n"
    "            self.tool_call_start_token_array = _cached['start_array']\n"
    "            self.tool_call_end_token_array = _cached['end_array']\n"
    "        else:\n"
    "            with Hermes2ProToolParser._tokenizer_lock:\n"
    "                if _tid in Hermes2ProToolParser._tokenizer_cache:\n"
    "                    _cached = Hermes2ProToolParser._tokenizer_cache[_tid]\n"
    "                    self.tool_call_start_token_ids = _cached['start_ids']\n"
    "                    self.tool_call_end_token_ids = _cached['end_ids']\n"
    "                    self.tool_call_start_token_array = _cached['start_array']\n"
    "                    self.tool_call_end_token_array = _cached['end_array']\n"
    "                else:\n"
    "                    self.tool_call_start_token_ids = self.model_tokenizer.encode(\n"
    "                        self.tool_call_start_token, add_special_tokens=False\n"
    "                    )\n"
    "                    self.tool_call_end_token_ids = self.model_tokenizer.encode(\n"
    "                        self.tool_call_end_token, add_special_tokens=False\n"
    "                    )\n"
    "                    self.tool_call_start_token_array = [\n"
    "                        self.model_tokenizer.decode([token_id])\n"
    "                        for token_id in self.tool_call_start_token_ids\n"
    "                    ]\n"
    "                    self.tool_call_end_token_array = [\n"
    "                        self.model_tokenizer.decode([token_id])\n"
    "                        for token_id in self.tool_call_end_token_ids\n"
    "                    ]\n"
    "                    Hermes2ProToolParser._tokenizer_cache[_tid] = {\n"
    "                        'start_ids': self.tool_call_start_token_ids,\n"
    "                        'end_ids': self.tool_call_end_token_ids,\n"
    "                        'start_array': self.tool_call_start_token_array,\n"
    "                        'end_array': self.tool_call_end_token_array,\n"
    "                    }"
)


def _patch_hermes_tool_parser() -> None:  # WORKAROUND(vllm-0.17-hermes)
    """Patch hermes_tool_parser.py on disk before vLLM imports it."""
    spec = find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        print(f"{_TAG} vLLM not found -- skipping hermes patch.")
        return

    base_dir = next(iter(spec.submodule_search_locations))
    target = os.path.join(base_dir, "tool_parsers", "hermes_tool_parser.py")

    if not os.path.exists(target):
        print(f"{_TAG} {target} not found -- skipping hermes patch.")
        return

    with open(target) as f:
        content = f.read()

    if "_tokenizer_cache" in content:
        print(f"{_TAG} Hermes patch already applied.")
        return

    if _OLD_INIT not in content:
        print(f"{_TAG} WARNING: Expected code snippet not found in {target}.")
        print(f"{_TAG} The vLLM version may have changed -- skipping hermes patch.")
        return

    content = content.replace(_OLD_IMPORT, _NEW_IMPORT, 1)
    content = content.replace(_OLD_CLASS_LINE, _NEW_CLASS_LINE, 1)
    content = content.replace(_OLD_INIT, _NEW_INIT, 1)

    with open(target, "w") as f:
        f.write(content)

    print(f"{_TAG} Successfully patched {target} for thread-safety.")


# ---------------------------------------------------------------------------
# WORKAROUND(harmony-aarch64) -- tiktoken vocab download for gpt-oss on ARM
#
# Remove this entire section when openai_harmony ships a fixed aarch64 binary.
# ---------------------------------------------------------------------------

_TIKTOKEN_FILES = {
    "o200k_base.tiktoken": "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
    "cl100k_base.tiktoken": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
}


def _ensure_tiktoken_cache() -> None:  # WORKAROUND(harmony-aarch64)
    """Download tiktoken vocab files if on aarch64 and set env vars."""
    if platform.machine() not in ("aarch64", "arm64"):
        return

    cache_dir = Path("/tmp/tiktoken-encodings")
    cache_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in _TIKTOKEN_FILES.items():
        dest = cache_dir / filename
        if dest.exists() and dest.stat().st_size > 0:
            continue
        try:
            print(f"{_TAG} Downloading {filename} for aarch64 workaround...")
            urllib.request.urlretrieve(url, dest)
            print(f"{_TAG}   {dest.stat().st_size:,} bytes -> {dest}")
        except Exception as e:
            print(f"{_TAG} WARNING: Failed to download {filename}: {e}")
            return

    os.environ["TIKTOKEN_ENCODINGS_BASE"] = str(cache_dir)
    os.environ.setdefault("TIKTOKEN_RS_CACHE_DIR", str(cache_dir))
    print(f"{_TAG} TIKTOKEN_ENCODINGS_BASE={cache_dir}")


# ---------------------------------------------------------------------------
# TP / DP arithmetic
# ---------------------------------------------------------------------------


def _extract_int_flag(args: list[str], flag: str, default: int = 1) -> int:
    """Read an integer flag value from *args* without removing it.

    Supports both ``--flag N`` and ``--flag=N`` forms.
    """
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return int(args[i + 1])
        if a.startswith(f"{flag}="):
            return int(a.split("=", 1)[1])
    return default


def _has_flag(args: list[str], flag: str) -> bool:
    """Return True if *flag* is already present in *args*."""
    return any(a == flag or a.startswith(f"{flag}=") for a in args)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    _patch_hermes_tool_parser()
    _ensure_tiktoken_cache()

    parser = argparse.ArgumentParser(
        description="Patched vLLM server entrypoint with DP-aware TP calculation",
    )
    parser.add_argument("--model", required=True, help="Model path or HF name")
    parser.add_argument("--num_gpus", type=int, required=True)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--no_verbose", action="store_true", help="Suppress request logs")
    args, unknown = parser.parse_known_args()

    dp_size = _extract_int_flag(unknown, "--data-parallel-size")
    total_gpus = args.num_gpus * args.num_nodes
    tp_size = total_gpus // dp_size

    # Multi-node DP: vLLM must know how many DP ranks fit on the local
    # (master) node, otherwise it tries to place all ranks locally.
    dp_size_local = args.num_gpus // tp_size
    if dp_size > 1 and not _has_flag(unknown, "--data-parallel-size-local"):
        unknown.extend(["--data-parallel-size-local", str(dp_size_local)])

    print(f"{_TAG} Deploying model {args.model}")
    print(
        f"{_TAG} GPUs: {total_gpus} total"
        f" (num_gpus={args.num_gpus} x num_nodes={args.num_nodes})"
        f" -> TP={tp_size}, DP={dp_size}, DP_local={dp_size_local}"
    )

    cmd_list = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        f"--model={args.model}",
        f"--served-model-name={args.model}",
        "--trust-remote-code",
        "--host=0.0.0.0",
        f"--port={args.port}",
        f"--tensor-parallel-size={tp_size}",
    ]
    if args.no_verbose:
        cmd_list.extend(["--disable-log-requests", "--disable-log-stats"])
    cmd_list.extend(unknown)

    print(f"{_TAG} Starting OpenAI Server")
    print(f"{_TAG} cmd: {' '.join(cmd_list)}")
    subprocess.run(cmd_list, check=True)


if __name__ == "__main__":
    main()
