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
"""Tests for nvflow.lib.cli_cmd.build_python_cmd.

Pins the shlex-quoting + interpreter contract so future refactors can't
accidentally reintroduce a path-injection vector via the rendered shell
command.  This helper is shared across multiple stages
(validate_questions, data_transformation, apply_prompt_template,
convert_to_responses_api, prepare_data, prefetch_cache) so a regression
here would fan out across the entire data pipeline.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from nvflow.lib.cli_cmd import build_python_cmd, build_python_script_cmd


def test_emits_module_invocation() -> None:
    out = build_python_cmd("foo.bar.baz", input_file="/a/b.jsonl")
    assert out.startswith("python3 -m foo.bar.baz ")
    # Should be a single space-joined string.
    assert "\n" not in out
    assert "  " not in out  # no doubled spaces


def test_renders_simple_paths_unchanged() -> None:
    """Plain paths (no special chars) shouldn't gain extra quotes -- shlex.quote
    only quotes when needed.  This makes log-grepping the rendered command
    straightforward.
    """
    out = build_python_cmd(
        "m", input_file="/lustre/foo/bar.jsonl", stats_file="/lustre/foo/stats.json"
    )
    assert "--input_file /lustre/foo/bar.jsonl" in out
    assert "--stats_file /lustre/foo/stats.json" in out
    # No surrounding single quotes around the safe paths.
    assert "'/lustre/foo/bar.jsonl'" not in out


def test_quotes_path_with_spaces() -> None:
    """Spaces in a path must be properly quoted so the shell parses one arg."""
    out = build_python_cmd("m", input_file="/a path/with spaces.jsonl")
    # shlex.quote surrounds the value with single quotes when needed.
    assert "--input_file '/a path/with spaces.jsonl'" in out
    # And shlex.split must round-trip to the original token.
    tokens = shlex.split(out)
    idx = tokens.index("--input_file")
    assert tokens[idx + 1] == "/a path/with spaces.jsonl"


def test_quotes_path_with_single_quote() -> None:
    """Single-quote in a value is the canonical injection vector for naive
    f-string command builders -- shlex.quote handles it correctly.
    """
    out = build_python_cmd("m", input_file="/a/file's name.jsonl")
    tokens = shlex.split(out)
    idx = tokens.index("--input_file")
    assert tokens[idx + 1] == "/a/file's name.jsonl"


def test_quotes_shell_metacharacters() -> None:
    """``$``, ``;``, ``|``, backticks, ``&`` etc. must not be interpreted by
    the shell as control characters when they appear in a value.
    """
    dangerous = "/path; rm -rf /;$(echo pwned)`evil`|cat&"
    out = build_python_cmd("m", input_file=dangerous)
    tokens = shlex.split(out)
    idx = tokens.index("--input_file")
    assert tokens[idx + 1] == dangerous


def test_accepts_pathlib_values() -> None:
    """Stage code passes pathlib.Path -- helper must stringify them."""
    out = build_python_cmd("m", input_file=Path("/a/b.jsonl"))
    assert "--input_file /a/b.jsonl" in out


def test_accepts_numeric_values() -> None:
    """Stages pass ints (e.g. ``num_chunks=10``) and floats (e.g.
    ``context_min_percentile=1.0``) directly -- helper must stringify them.
    """
    out = build_python_cmd("m", num_chunks=10, context_min_percentile=1.0)
    assert "--num_chunks 10" in out
    assert "--context_min_percentile 1.0" in out


def test_preserves_flag_order() -> None:
    """Ordered output keeps log-grep diffs minimal across reruns.  Python
    3.7+ preserves kwarg order so this comes for free, but the test pins it.
    """
    out = build_python_cmd("m", alpha="1", beta="2", gamma="3")
    assert out.index("--alpha") < out.index("--beta") < out.index("--gamma")


def test_handles_empty_value() -> None:
    """Empty string still needs quoting so the flag's value isn't lost."""
    out = build_python_cmd("m", input_file="")
    tokens = shlex.split(out)
    idx = tokens.index("--input_file")
    assert tokens[idx + 1] == ""


def test_positional_args_emitted_before_flags() -> None:
    """Positional inputs (e.g. dataset_transformer's input_files) must appear
    between ``-m <module>`` and the first flag, in argument order.
    """
    out = build_python_cmd(
        "m",
        "/a/in1.jsonl",
        "/a/in2.jsonl",
        output_file="/a/out.jsonl",
    )
    tokens = shlex.split(out)
    assert tokens[:3] == ["python3", "-m", "m"]
    assert tokens[3] == "/a/in1.jsonl"
    assert tokens[4] == "/a/in2.jsonl"
    assert tokens[5] == "--output_file"
    assert tokens[6] == "/a/out.jsonl"


def test_positional_args_quoted() -> None:
    """Positional args must use the same shlex.quote treatment as flag values
    so ``input_files`` containing spaces or metacharacters do not break.
    """
    out = build_python_cmd("m", "/a path/file.jsonl", "/b/'evil'.jsonl")
    tokens = shlex.split(out)
    assert tokens[3] == "/a path/file.jsonl"
    assert tokens[4] == "/b/'evil'.jsonl"


def test_positional_args_accept_pathlib() -> None:
    """Stage code may pass Path positionals -- they must be stringified."""
    out = build_python_cmd("m", Path("/a/b.jsonl"), output_file=Path("/a/out.jsonl"))
    tokens = shlex.split(out)
    # Layout: python3 -m m /a/b.jsonl --output_file /a/out.jsonl
    #         [0]    [1] [2] [3]      [4]           [5]
    assert tokens[3] == "/a/b.jsonl"
    assert tokens[4] == "--output_file"
    assert tokens[5] == "/a/out.jsonl"


def test_no_positional_no_flags_renders_bare_module() -> None:
    """Bare module invocation (no args at all) is a valid edge case --
    the helper should not append trailing spaces or empty tokens.
    """
    out = build_python_cmd("m")
    assert out == "python3 -m m"


def test_uses_python3_interpreter_not_python() -> None:
    """All cluster containers in this repo use ``python3`` -- standardising
    avoids ambiguity around the unversioned ``python`` symlink (absent in
    some minimal images).  Pin this so future refactors don't silently
    flip back to ``python``.
    """
    out = build_python_cmd("any.module")
    assert out.startswith("python3 ")
    assert not out.startswith("python ")


# --- build_python_script_cmd (path-based variant for vendored external tools)


def test_script_cmd_emits_path_invocation() -> None:
    """``python3 <script>`` form -- contrast with build_python_cmd's
    ``python3 -m <module>`` form.  Used by stages that wrap external
    scripts shipped inside vendored repos (e.g. NeMo-Gym) where the
    upstream tool is a path-based script, not an importable module.
    """
    out = build_python_script_cmd("scripts/prefetch.py", cache_dir="/x")
    # No ``-m`` flag -- this is the load-bearing distinction.
    assert " -m " not in out
    assert out.startswith("python3 scripts/prefetch.py ")
    assert "--cache_dir /x" in out


def test_script_cmd_quotes_path_with_spaces() -> None:
    """Script paths with spaces must be quoted -- this is the same
    injection vector as flag values, but for the script path itself.
    Pinning this keeps the helper safe for environment-derived paths
    (e.g. vendored repos under user home dirs that may contain spaces).
    """
    out = build_python_script_cmd("/a path/run.py", input_file="/b/in.json")
    tokens = shlex.split(out)
    assert tokens == ["python3", "/a path/run.py", "--input_file", "/b/in.json"]


def test_script_cmd_accepts_pathlib_script() -> None:
    """Stage code may pass a pathlib.Path script -- helper must stringify."""
    out = build_python_script_cmd(
        Path("/opt/Gym/scripts/prefetch.py"),
        cache_dir=Path("/cache/finance"),
        ticker_config=Path("/configs/sp500.yaml"),
    )
    tokens = shlex.split(out)
    assert tokens[0] == "python3"
    assert tokens[1] == "/opt/Gym/scripts/prefetch.py"
    assert "--cache_dir" in tokens
    assert tokens[tokens.index("--cache_dir") + 1] == "/cache/finance"


def test_script_cmd_quotes_metacharacters_in_flag_value() -> None:
    """Same shlex.quote treatment as build_python_cmd -- verifies the
    helpers share their quoting contract end-to-end (no copy-paste
    drift when the path-based variant was added).
    """
    dangerous = "/path; rm -rf /;$(echo pwned)`evil`|cat&"
    out = build_python_script_cmd("scripts/run.py", cache_dir=dangerous)
    tokens = shlex.split(out)
    idx = tokens.index("--cache_dir")
    assert tokens[idx + 1] == dangerous


def test_script_cmd_uses_python3_interpreter() -> None:
    """Same interpreter standardisation as build_python_cmd -- pin so a
    future refactor doesn't silently diverge between the two helpers.
    """
    out = build_python_script_cmd("scripts/run.py")
    assert out.startswith("python3 ")
    assert not out.startswith("python ")
