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
"""Shared shell command builders for stage submission.

Stages that submit ``python3 -m <module> [positional ...] --flag <value> ...``
shell commands to nemo-skills' ``run_cmd`` / ``generate`` should use
:func:`build_python_cmd` to build the command string rather than
concatenating raw f-strings.  ``shlex.quote`` ensures values containing
spaces, single quotes, or shell metacharacters do not break the rendered
command -- this matters because nemo-skills interpolates the command
into a Slurm shell wrapper at submission time.

Usage::

    from nvflow.lib.cli_cmd import build_python_cmd

    # Flag-only invocation:
    rendered = build_python_cmd(
        "nvflow.recipes.finance.utils.rl.regex_prefilter_questions",
        input_file=Path("/lustre/foo/in.jsonl"),
        output_kept=Path("/lustre/foo/kept.jsonl"),
    )

    # With positional args (e.g. for ``argparse`` scripts that take
    # ``input_files`` positionally):
    rendered = build_python_cmd(
        "nvflow.recipes.finance.utils.shared.dataset_transformer",
        Path("/lustre/sdg/final_result.jsonl"),
        output_file="/lustre/out/final.jsonl",
        num_chunks=10,
    )
    # → "python3 -m ...dataset_transformer "
    #   "/lustre/sdg/final_result.jsonl "
    #   "--output_file /lustre/out/final.jsonl --num_chunks 10"
"""

from __future__ import annotations

import shlex
from pathlib import Path

# All cluster containers in this repo provide ``python3`` (it is the
# canonical interpreter on every modern Linux base image we ship).  We
# standardise on ``python3`` rather than ``python`` so ambiguity around
# the unversioned ``python`` symlink (absent in some minimal images) can
# never bite us.
_INTERPRETER = "python3"


def build_python_cmd(
    module: str,
    *positional: str | int | float | Path,
    **flags: str | int | float | Path,
) -> str:
    """Build a ``python3 -m <module> [positional ...] --flag value ...`` shell command.

    Each positional and flag value is passed through :func:`shlex.quote`
    so paths containing spaces, single quotes, or shell metacharacters
    do not break the rendered command -- this command string is
    interpolated by nemo-skills into a Slurm shell wrapper, so safe
    quoting matters.

    Accepts ``str``, numeric types, or :class:`pathlib.Path` values;
    non-string values are stringified via :func:`str` before quoting.
    Positional args are emitted in argument order, then flags in
    declaration order.  This keeps the rendered command stable for
    log-grepping and diffing across reruns.

    Args:
        module: Fully-qualified Python module name (e.g.
            ``"nvflow.recipes.finance.utils.shared.dataset_transformer"``).
        *positional: Positional arguments emitted before any flags --
            useful for ``argparse``-style scripts that accept positional
            inputs (e.g. one or more input file paths).
        **flags: Keyword arguments rendered as ``--<key> <quoted-value>``
            pairs in declaration order.  Boolean flags (no value) must
            be appended manually by the caller; this helper does not
            support them because Python kwargs cannot express
            "value-less" flags unambiguously.

    Returns:
        A single-line shell command string suitable for nemo-skills'
        ``run_cmd`` / ``generate`` ``ctx`` argument.

    Examples:
        >>> build_python_cmd("foo.bar", input_file="/a/b.jsonl")
        'python3 -m foo.bar --input_file /a/b.jsonl'
        >>> build_python_cmd("foo.bar", "/a/in.jsonl", output_file="/a/out.jsonl")
        'python3 -m foo.bar /a/in.jsonl --output_file /a/out.jsonl'
        >>> build_python_cmd("foo.bar", input_file="/a path/with spaces.jsonl")
        "python3 -m foo.bar --input_file '/a path/with spaces.jsonl'"
    """
    parts = [_INTERPRETER, "-m", module]
    for arg in positional:
        parts.append(shlex.quote(str(arg)))
    for flag, value in flags.items():
        parts.extend([f"--{flag}", shlex.quote(str(value))])
    return " ".join(parts)


def build_python_script_cmd(
    script: str | Path,
    *positional: str | int | float | Path,
    **flags: str | int | float | Path,
) -> str:
    """Build a ``python3 <script> [positional ...] --flag value ...`` shell command.

    Sibling of :func:`build_python_cmd`; use when invoking a path-based
    script (typically a vendored external tool inside a third-party
    repo such as NeMo-Gym) rather than a Python module.  Same shlex
    quoting + interpreter contract as ``build_python_cmd`` -- the
    only difference is the invocation form (``python3 <path>`` vs
    ``python3 -m <module>``).

    The script path itself is also ``shlex``-quoted so external tool
    paths containing spaces or shell metacharacters do not break the
    rendered command.  Most paths are static config values today, but
    quoting keeps the helper robust for future overlays that might
    interpolate paths from environment variables or symlinks.

    Args:
        script: Path to the Python script to execute.  Accepts ``str``
            or :class:`pathlib.Path`; relative paths are passed through
            verbatim and resolved by the shell against the active
            ``cd`` (callers typically prepend ``cd <gym_path> &&`` for
            scripts that import config files relative to the repo
            root).
        *positional: Positional arguments emitted before any flags.
        **flags: Keyword arguments rendered as ``--<key> <quoted-value>``
            pairs in declaration order.  Boolean flags (no value) must
            be appended manually by the caller; same caveat as
            ``build_python_cmd``.

    Returns:
        A single-line shell command string suitable for nemo-skills'
        ``run_cmd`` / ``generate`` ``ctx`` argument, or for shell
        composition (e.g. ``cd /opt/foo && <returned-cmd>``).

    Examples:
        >>> build_python_script_cmd("scripts/prefetch.py", cache_dir="/x")
        'python3 scripts/prefetch.py --cache_dir /x'
        >>> build_python_script_cmd("/a path/run.py", input_file="/b/in.json")
        "python3 '/a path/run.py' --input_file /b/in.json"
    """
    parts = [_INTERPRETER, shlex.quote(str(script))]
    for arg in positional:
        parts.append(shlex.quote(str(arg)))
    for flag, value in flags.items():
        parts.extend([f"--{flag}", shlex.quote(str(value))])
    return " ".join(parts)


__all__ = ["build_python_cmd", "build_python_script_cmd"]
