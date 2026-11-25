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
"""TTY-aware stdout payloads and stderr diagnostics for the NVFlow CLI."""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator
from typing import TextIO

from rich.console import RenderableType
from rich.measure import Measurement
from rich.segment import Segment, Segments
from rich.syntax import Syntax
from rich.text import Text

from nvflow.core.console import (
    ACCENT_STYLE,
    ERROR_STYLE,
    ERROR_SYMBOL,
    HEADING_STYLE,
    MUTED_STYLE,
    STATUS_STYLE,
    STATUS_SYMBOL,
    SUCCESS_STYLE,
    SUCCESS_SYMBOL,
    WARNING_STYLE,
    WARNING_SYMBOL,
    create_console,
    message_text,
)

stdout_console = create_console()
stderr_console = create_console(stderr=True)

_YAML_SYNTAX_THEME = "ansi_dark"
_UNBOUNDED_WIDTH = 100_000


@contextlib.contextmanager
def route_stdout_to_stderr() -> Iterator[None]:
    """Route Python and inherited process stdout to stderr for action commands."""
    stdout_fd = _fileno(sys.stdout)
    stderr_fd = _fileno(sys.stderr)
    saved_stdout_fd: int | None = None
    if stdout_fd is not None and stderr_fd is not None and stdout_fd != stderr_fd:
        sys.stdout.flush()
        sys.stderr.flush()
        saved_stdout_fd = os.dup(stdout_fd)
        os.dup2(stderr_fd, stdout_fd)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        if saved_stdout_fd is not None and stdout_fd is not None:
            sys.stderr.flush()
            os.dup2(saved_stdout_fd, stdout_fd)
            os.close(saved_stdout_fd)


def _fileno(stream: TextIO) -> int | None:
    try:
        return stream.fileno()
    except (AttributeError, OSError, ValueError):
        return None


def write_stdout(value: object = "", *, end: str = "\n") -> None:
    """Write an exact, unstyled payload to stdout."""
    sys.stdout.write(f"{value}{end}")


def write_stderr(value: object = "", *, end: str = "\n") -> None:
    """Write an exact, unstyled diagnostic to stderr."""
    sys.stderr.write(f"{value}{end}")


def emit(renderable: RenderableType) -> None:
    """Render a human-oriented payload on stdout."""
    stdout_console.print(renderable)


def emit_wide(renderable: RenderableType) -> None:
    """Render a payload at its natural width, leaving long lines to the terminal.

    Unlike :func:`emit`, cells are never wrapped or cropped to fit the console.
    """
    width = Measurement.get(
        stdout_console,
        stdout_console.options.update_width(_UNBOUNDED_WIDTH),
        renderable,
    ).maximum
    lines = stdout_console.render_lines(
        renderable,
        stdout_console.options.update_width(max(width, 1)),
        pad=False,
    )
    segments = [segment for rendered in lines for segment in (*rendered, Segment.line())]
    stdout_console.print(Segments(segments), soft_wrap=True)


def line(value: object = "", *, style: str | None = None) -> None:
    """Render one literal human-oriented line on stdout."""
    stdout_console.print(message_text(value, style=style), soft_wrap=True)


def heading(value: object) -> None:
    """Render a CLI payload heading on stdout."""
    line(value, style=HEADING_STYLE)


def success(value: object) -> None:
    """Render a successful CLI result on stdout."""
    stdout_console.print(
        message_text(value, style=SUCCESS_STYLE, symbol=SUCCESS_SYMBOL),
        soft_wrap=True,
    )


def status(value: object) -> None:
    """Render progress on stderr."""
    stderr_console.print(
        message_text(value, style=STATUS_STYLE, symbol=STATUS_SYMBOL),
        soft_wrap=True,
    )


def diagnostic(value: object = "", *, style: str | None = None) -> None:
    """Render one literal diagnostic line on stderr."""
    stderr_console.print(message_text(value, style=style), soft_wrap=True)


def warning(value: object) -> None:
    """Render a warning on stderr."""
    stderr_console.print(
        message_text(value, style=WARNING_STYLE, symbol=WARNING_SYMBOL),
        soft_wrap=True,
    )


def error(value: object) -> None:
    """Render an error on stderr."""
    stderr_console.print(
        message_text(f"Error: {value}", style=ERROR_STYLE, symbol=ERROR_SYMBOL),
        soft_wrap=True,
    )


def emit_plan_text(value: str) -> None:
    """Render trusted plan labels with color while preserving literal values."""
    if not stdout_console.is_terminal:
        write_stdout(value)
        return

    rendered = Text()
    source_statuses = {
        "READY": SUCCESS_STYLE,
        "BLOCKED": ERROR_STYLE,
        "UNKNOWN": WARNING_STYLE,
    }
    coverage_statuses = {
        "COMPLETE": SUCCESS_STYLE,
        "PARTIAL": WARNING_STYLE,
    }
    model_path_statuses = {
        "NOT CHECKED": WARNING_STYLE,
        "NOT APPLICABLE": MUTED_STYLE,
    }
    for index, raw_line in enumerate(value.splitlines()):
        if index:
            rendered.append("\n")
        if raw_line.startswith("Mode: "):
            label, mode = raw_line.split(": ", 1)
            rendered.append(f"{label}: ", style=MUTED_STYLE)
            rendered.append(mode, style=ACCENT_STYLE)
        elif raw_line.startswith("Filesystem inspection: "):
            label, state = raw_line.split(": ", 1)
            rendered.append(f"{label}: ", style=MUTED_STYLE)
            style = ACCENT_STYLE if state.startswith("ENABLED") else MUTED_STYLE
            rendered.append(state, style=style)
        elif raw_line.startswith("Source eligibility: "):
            _append_status_line(rendered, raw_line, source_statuses)
        elif raw_line.startswith("Managed model paths: "):
            _append_status_line(rendered, raw_line, model_path_statuses)
        elif raw_line.startswith("Plan coverage: "):
            _append_status_line(rendered, raw_line, coverage_statuses)
        elif raw_line.startswith("NVFlow plan: "):
            label, value_text = raw_line.split(": ", 1)
            rendered.append(f"{label}: ", style=HEADING_STYLE)
            rendered.append(value_text)
        elif raw_line.startswith("● "):
            rendered.append("●", style=ACCENT_STYLE)
            rendered.append(raw_line[1:])
        else:
            rendered.append(raw_line)
    stdout_console.print(rendered, soft_wrap=True)


def emit_yaml(value: str) -> None:
    """Highlight YAML and its indentation on TTYs, preserving exact redirected output."""
    if not stdout_console.is_terminal:
        write_stdout(value, end="" if value.endswith("\n") else "\n")
        return

    syntax = Syntax(
        value.rstrip("\n"),
        "yaml",
        theme=_YAML_SYNTAX_THEME,
        background_color="default",
        indent_guides=False,
        padding=0,
        word_wrap=True,
    )
    stdout_console.print(syntax, soft_wrap=False)


def _append_status_line(rendered: Text, raw_line: str, styles: dict[str, str]) -> None:
    label, value = raw_line.split(": ", 1)
    rendered.append(f"{label}: ", style=MUTED_STYLE)
    status_value, separator, detail = value.partition(" — ")
    rendered.append(status_value, style=styles.get(status_value, STATUS_STYLE))
    if separator:
        rendered.append(f"{separator}{detail}")


__all__ = [
    "emit",
    "emit_plan_text",
    "emit_wide",
    "emit_yaml",
    "diagnostic",
    "error",
    "heading",
    "line",
    "route_stdout_to_stderr",
    "status",
    "stderr_console",
    "stdout_console",
    "success",
    "warning",
    "write_stderr",
    "write_stdout",
]
