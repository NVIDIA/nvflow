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
import glob
import os
import platform
import subprocess
import sys
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


def _vllm_base_dir_for(python: str) -> str | None:
    """Return the on-disk ``vllm`` package dir as seen by *python*.

    Used so the on-disk hermes patch targets the SAME vLLM install the serve
    interpreter will import, even when that interpreter is a separate per-actor
    venv (see WORKAROUND(nemo-rl-vllm-separate-venv)).
    """
    try:
        proc = subprocess.run(
            [python, "-c", "import vllm, os; print(os.path.dirname(vllm.__file__))"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    base_dir = proc.stdout.strip()
    return base_dir or None


def _patch_hermes_tool_parser_via(python: str) -> None:  # WORKAROUND(vllm-0.17-hermes)
    """Patch hermes_tool_parser.py for the vLLM install used by *python*."""
    base_dir = _vllm_base_dir_for(python)
    if base_dir is None:
        print(f"{_TAG} Could not locate vllm via {python} -- skipping hermes patch.")
        return
    _patch_hermes_tool_parser(base_dir=base_dir)


def _patch_hermes_tool_parser(base_dir: str | None = None) -> None:  # WORKAROUND(vllm-0.17-hermes)
    """Patch hermes_tool_parser.py on disk before vLLM imports it.

    When *base_dir* is given it is used as the vLLM package directory (e.g.
    discovered via a separate per-actor venv); otherwise it is resolved from
    the current interpreter via ``find_spec``.
    """
    if base_dir is None:
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
    """Download tiktoken vocab files if on aarch64 and set env vars.

    Skips the download when TIKTOKEN_CACHE_DIR or TIKTOKEN_RS_CACHE_DIR is
    already set (e.g. pointing at files baked into the container), which is
    required for airgap / offline environments.
    """
    if platform.machine() not in ("aarch64", "arm64"):
        return

    existing = os.environ.get("TIKTOKEN_CACHE_DIR") or os.environ.get("TIKTOKEN_RS_CACHE_DIR")
    if existing:
        print(f"{_TAG} Tiktoken cache already configured ({existing}), skipping download.")
        os.environ.setdefault("TIKTOKEN_ENCODINGS_BASE", existing)
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
# WORKAROUND(nemo-rl-vllm-separate-venv) -- vLLM lives in a per-actor venv
#
# nemo-skills builds the standalone serve cmd as ``python3 {entrypoint} ...``
# (nemo_skills/pipeline/utils/server.py), so this script runs under whatever
# ``python3`` resolves to first on PATH. On a NeMo-RL v0.6.0 Ray-on-Slurm
# cluster that is the head's RAY_VENV (/opt/nemo_rl_venv), which does NOT have
# vLLM installed -- the container keeps vLLM (~0.17.1) in a SEPARATE per-actor
# uv venv created at runtime under /opt/ray_venvs/<hash>/ for NeMo-RL's
# VllmGenerationWorker. A bare ``python3 -m vllm...`` therefore fails with
# ``No module named 'vllm'`` for the standalone GRPO rollout *policy* serve.
#
# Fix: resolve an interpreter that can ``import vllm`` for the inner serve
# subprocess. Resolution order (first hit wins):
#   1. NVFLOW_VLLM_PYTHON  -- explicit interpreter path (operator override).
#   2. NVFLOW_VLLM_VENV    -- explicit venv dir; uses <venv>/bin/python.
#   3. The current interpreter (sys.executable) if it can import vllm -- this
#      keeps the working Slurm / in-container path UNCHANGED (vLLM already
#      importable in the running python).
#   4. Auto-discovery: scan candidate globs (NVFLOW_VLLM_VENV_GLOB, default
#      /opt/ray_venvs/*/bin/python*) and pick the first python that can
#      ``import vllm``.
#   5. Fallback to "python3" (preserves prior behavior / error if vLLM is
#      genuinely absent everywhere).
#
# Remove when: the standalone policy serve runs in the same venv as vLLM
# (e.g. nemo-skills launches the serve under the per-actor venv, or NeMo-RL
# installs vLLM into RAY_VENV).
# ---------------------------------------------------------------------------

# Default glob for per-actor venvs that may contain vLLM. Overridable via
# NVFLOW_VLLM_VENV_GLOB for containers that lay venvs out differently.
_DEFAULT_VLLM_VENV_GLOB = "/opt/ray_venvs/*/bin/python*"


def _python_has_vllm(python: str) -> bool:
    """Return True if *python* can ``import vllm`` (best-effort, quiet)."""
    try:
        proc = subprocess.run(
            [python, "-c", "import vllm"],
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _resolve_vllm_python() -> str:  # WORKAROUND(nemo-rl-vllm-separate-venv)
    """Return a python interpreter that can ``import vllm``.

    See the WORKAROUND(nemo-rl-vllm-separate-venv) section above for the
    full resolution order and rationale.
    """
    # 1. Explicit interpreter override.
    explicit = os.environ.get("NVFLOW_VLLM_PYTHON")
    if explicit:
        print(f"{_TAG} Using NVFLOW_VLLM_PYTHON={explicit} for vLLM serve.")
        return explicit

    # 2. Explicit venv override.
    venv = os.environ.get("NVFLOW_VLLM_VENV")
    if venv:
        candidate = os.path.join(venv, "bin", "python")
        print(f"{_TAG} Using NVFLOW_VLLM_VENV={venv} -> {candidate} for vLLM serve.")
        return candidate

    # 3. Current interpreter already has vLLM -> keep Slurm/in-container path
    #    UNCHANGED. This is the common, working case.
    if _python_has_vllm(sys.executable):
        return sys.executable

    # 4. Auto-discover a per-actor venv python that can import vLLM.
    venv_glob = os.environ.get("NVFLOW_VLLM_VENV_GLOB", _DEFAULT_VLLM_VENV_GLOB)
    print(
        f"{_TAG} '{sys.executable}' cannot import vllm; scanning '{venv_glob}'"
        f" for a python that can (set NVFLOW_VLLM_PYTHON to skip this)."
    )
    for candidate in sorted(glob.glob(venv_glob)):
        if _python_has_vllm(candidate):
            print(f"{_TAG} Discovered vLLM in {candidate}.")
            return candidate

    # 5. Fallback: preserve prior behavior (and its error) if nothing works.
    print(
        f"{_TAG} WARNING: no python with vllm found via override, sys.executable,"
        f" or glob '{venv_glob}'. Falling back to 'python3'"
        f" (set NVFLOW_VLLM_PYTHON=/path/to/python to fix)."
    )
    return "python3"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Resolve which interpreter runs the inner ``-m vllm...`` serve. On a
    # NeMo-RL Ray-on-Slurm cluster this script may run under a python WITHOUT
    # vLLM (RAY_VENV=/opt/nemo_rl_venv); vLLM lives in a per-actor venv.
    # WORKAROUND(nemo-rl-vllm-separate-venv) -- see helper above.
    vllm_python = _resolve_vllm_python()

    # The on-disk hermes patch must target the SAME vLLM the serve will use.
    # When the resolved interpreter differs from the current one, run the patch
    # under that interpreter so find_spec("vllm") locates the right install;
    # otherwise patch in-process (the working Slurm/in-container case).
    if vllm_python == sys.executable:
        _patch_hermes_tool_parser()
    else:
        _patch_hermes_tool_parser_via(vllm_python)
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
        vllm_python,  # WORKAROUND(nemo-rl-vllm-separate-venv) -- not bare "python3"
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
