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
"""Console utilities for consistent terminal output across NVFlow.

This module provides simple wrappers around Rich for consistent logging
in the orchestration layer (local machine). All functions print to the
user's terminal during workflow execution.

Note: These are for ORCHESTRATION logs (what's being submitted to cluster),
not execution logs (which are captured by Slurm on the cluster).
"""

from typing import TextIO

from rich.console import Console
from rich.text import Text
from rich.theme import Theme

SUCCESS_STYLE = "nvflow.success"
INFO_STYLE = "nvflow.info"
WARNING_STYLE = "nvflow.warning"
ERROR_STYLE = "nvflow.error"
STATUS_STYLE = "nvflow.status"
HEADING_STYLE = "nvflow.heading"
ACCENT_STYLE = "nvflow.accent"
MUTED_STYLE = "nvflow.muted"

SUCCESS_SYMBOL = "✓"
INFO_SYMBOL = "ℹ"
WARNING_SYMBOL = "⚠"
ERROR_SYMBOL = "✗"
STATUS_SYMBOL = "▶"

NVFLOW_THEME = Theme(
    {
        SUCCESS_STYLE: "green",
        INFO_STYLE: "blue",
        WARNING_STYLE: "yellow",
        ERROR_STYLE: "red",
        STATUS_STYLE: "cyan",
        HEADING_STYLE: "bold",
        ACCENT_STYLE: "bold cyan",
        MUTED_STYLE: "dim",
    }
)


def create_console(
    *,
    stderr: bool = False,
    file: TextIO | None = None,
    force_terminal: bool | None = None,
) -> Console:
    """Create a Rich console with NVFlow's shared theme and safe defaults."""
    if file is None:
        return Console(
            theme=NVFLOW_THEME,
            highlight=False,
            force_terminal=force_terminal,
            stderr=stderr,
        )
    return Console(
        theme=NVFLOW_THEME,
        highlight=False,
        force_terminal=force_terminal,
        file=file,
    )


# Orchestration diagnostics are not command payloads. Keeping them on stderr
# makes structured CLI stdout (for example,
# ``nvflow plan --cluster <target> --format json``) directly machine-readable.
console = create_console(stderr=True)

# Convenience: alias for console.print
print = console.print


def message_text(
    message: object,
    *,
    style: str | None = None,
    symbol: str | None = None,
) -> Text:
    """Build a styled message while treating caller-provided values literally."""
    rendered = Text()
    if symbol is not None:
        rendered.append(symbol, style=style)
        rendered.append(" ")
    rendered.append(str(message), style=style if symbol is None else None)
    return rendered


def header(message: str) -> None:
    """Print a bold header with separator lines.

    Use for major workflow sections.

    Args:
        message: Header text to display

    Example:
        >>> header("Starting SDG Pipeline")
        ================================================================================
        Starting SDG Pipeline
        ================================================================================
    """
    console.print()
    console.print("=" * 80)
    console.print(message_text(message, style=HEADING_STYLE))
    console.print("=" * 80)
    console.print()


def section(message: str) -> None:
    """Print a section header with lighter separator.

    Use for stage execution sections.

    Args:
        message: Section text to display

    Example:
        >>> section("Running Stage: sdg.generate_qas")
        ────────────────────────────────────────────────────────────────────────────────
        ▶ Running Stage: sdg.generate_qas
        ────────────────────────────────────────────────────────────────────────────────
    """
    console.print()
    console.print("─" * 80)
    console.print(message_text(message, style=STATUS_STYLE, symbol=STATUS_SYMBOL))
    console.print(f"{'─' * 80}")


def success(message: str) -> None:
    """Print a success message with green checkmark.

    Use for successful operations.

    Args:
        message: Success message to display

    Example:
        >>> success("Job submitted successfully")
        ✓ Job submitted successfully
    """
    console.print(message_text(message, style=SUCCESS_STYLE, symbol=SUCCESS_SYMBOL))


def info(message: str) -> None:
    """Print an informational message.

    Use for general information and status updates.

    Args:
        message: Info message to display

    Example:
        >>> info("Found 3 input files")
        ℹ Found 3 input files
    """
    console.print(message_text(message, style=INFO_STYLE, symbol=INFO_SYMBOL))


def warning(message: str) -> None:
    """Print a warning message.

    Use for non-critical issues that users should be aware of.

    Args:
        message: Warning message to display

    Example:
        >>> warning("Job may take 10+ minutes")
        ⚠ Job may take 10+ minutes
    """
    console.print(message_text(message, style=WARNING_STYLE, symbol=WARNING_SYMBOL))


def error(message: str) -> None:
    """Print an error message with red X.

    Use for errors and failures.

    Args:
        message: Error message to display

    Example:
        >>> error("Job submission failed")
        ✗ Job submission failed
    """
    console.print(message_text(message, style=ERROR_STYLE, symbol=ERROR_SYMBOL))


def status(message: str) -> None:
    """Print a status message.

    Use for ongoing operations and progress updates.

    Args:
        message: Status message to display

    Example:
        >>> status("Submitting job to cluster...")
        ▶ Submitting job to cluster...
    """
    console.print(message_text(message, style=STATUS_STYLE, symbol=STATUS_SYMBOL))


def detail(key: str, value: str) -> None:
    """Print a key-value detail line.

    Use for showing configuration details and parameters.

    Args:
        key: Detail key/label
        value: Detail value

    Example:
        >>> detail("Model", "/path/to/model")
        >>> detail("GPUs", "8")
          Model: /path/to/model
          GPUs: 8
    """
    rendered = Text("  ")
    rendered.append(f"{key}:", style=MUTED_STYLE)
    rendered.append(f" {value}")
    console.print(rendered)


def blank() -> None:
    """Print a blank line for spacing."""
    console.print()


def rule(title: str = "") -> None:
    """Print a horizontal rule with optional title.

    Args:
        title: Optional title to display in the rule

    Example:
        >>> rule("Configuration")
        ────────────── Configuration ──────────────
    """
    console.rule(title, style="dim")


# Convenience: Export console for advanced usage
__all__ = [
    "console",
    "create_console",
    "message_text",
    "print",
    "header",
    "section",
    "success",
    "info",
    "warning",
    "error",
    "status",
    "detail",
    "blank",
    "rule",
]
