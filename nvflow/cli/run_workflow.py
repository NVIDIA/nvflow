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
"""Python debugging entry point for direct WorkflowRunner invocation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from nvflow.cli.workflow_selection import require_workflow


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an NVFlow workflow in the current Python interpreter. "
            "Use nvflow for automatic project-environment isolation."
        )
    )
    parser.add_argument("stages", nargs="*", help="Stage name(s) to run")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--config", "-c", help="Workflow YAML path")
    selection.add_argument("--workflow", "-w", help="Project workflow selector")
    parser.add_argument("--all", action="store_true", help="Run every configured stage")
    parser.add_argument(
        "--environment",
        "-e",
        action="extend",
        nargs="+",
        help="Run one or more environments; the option may be repeated",
    )
    parser.add_argument("--cluster", required=True, help="Target configuration name")
    parser.add_argument("--project-dir", help="Explicit external project root")
    parser.add_argument(
        "--project-path",
        action="append",
        default=[],
        help="Ordered project search path; repeatable",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run selected stages through the public Python API."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.all == bool(args.stages):
        parser.error("provide stage name(s) or --all, but not both")

    try:
        selection = require_workflow(
            config=args.config,
            workflow=args.workflow,
            project_dir=args.project_dir,
            project_paths=tuple(args.project_path),
        )
        from nvflow.execution import WorkflowRunner

        runner = WorkflowRunner(
            selection.config_path,
            project_dir=selection.project_dir,
            cluster=args.cluster,
            project_search_paths=tuple(args.project_path),
        )
        source_root = str(runner.project_context.launcher_python_root)
        added_source_root = source_root not in sys.path
        if added_source_root:
            sys.path.insert(0, source_root)
        try:
            runner.run(
                stages=None if args.all else args.stages,
                environment=args.environment,
            )
        finally:
            if added_source_root:
                try:
                    sys.path.remove(source_root)
                except ValueError:
                    pass
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
