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

from rich.console import Console

# Single console instance for the entire application
console = Console()

# Convenience: alias for console.print
print = console.print


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
    console.print(f"\n{'=' * 80}")
    console.print(f"[bold]{message}[/bold]")
    console.print(f"{'=' * 80}\n")


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
    console.print(f"\n{'─' * 80}")
    console.print(f"[cyan]▶[/cyan] {message}")
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
    console.print(f"[green]✓[/green] {message}")


def info(message: str) -> None:
    """Print an informational message.

    Use for general information and status updates.

    Args:
        message: Info message to display

    Example:
        >>> info("Found 3 input files")
        ℹ Found 3 input files
    """
    console.print(f"[blue]ℹ[/blue] {message}")


def warning(message: str) -> None:
    """Print a warning message.

    Use for non-critical issues that users should be aware of.

    Args:
        message: Warning message to display

    Example:
        >>> warning("Job may take 10+ minutes")
        ⚠ Job may take 10+ minutes
    """
    console.print(f"[yellow]⚠[/yellow]  {message}")


def error(message: str) -> None:
    """Print an error message with red X.

    Use for errors and failures.

    Args:
        message: Error message to display

    Example:
        >>> error("Job submission failed")
        ✗ Job submission failed
    """
    console.print(f"[red]✗[/red] {message}")


def status(message: str) -> None:
    """Print a status message.

    Use for ongoing operations and progress updates.

    Args:
        message: Status message to display

    Example:
        >>> status("Submitting job to cluster...")
        ▶ Submitting job to cluster...
    """
    console.print(f"[cyan]▶[/cyan] {message}")


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
    console.print(f"  [dim]{key}:[/dim] {value}")


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
