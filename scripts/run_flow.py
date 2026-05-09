#!/usr/bin/env python3
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
"""Direct Python interface for running NVFlow stages.

Usage:
    uv run python scripts/run_flow.py STAGE [STAGE...] --config CONFIG
    uv run python scripts/run_flow.py --all --config CONFIG
    uv run python scripts/run_flow.py --help

Examples:
    # Run a single SFT stage
    uv run python scripts/run_flow.py sft --config nvflow/recipes/finance/workflows/training_sft.yaml

    # Run a single GRPO stage
    uv run python scripts/run_flow.py collect_rollouts --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml

    # Run a GRPO stage for one environment
    uv run python scripts/run_flow.py collect_rollouts -c nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml -e equivalence_llm_judge

    # Run a GRPO stage for multiple environments
    uv run python scripts/run_flow.py training -c nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml -e mcqa equivalence_llm_judge

    # Run all stages in workflow
    uv run python scripts/run_flow.py --all --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to Python path so we can import nvflow
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Auto-discover all recipes and stages
import nvflow.recipes  # noqa: F401, E402
from nvflow.core import WorkflowRunner  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Direct Python interface for running NVFlow stages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single SFT stage
  uv run python scripts/run_flow.py sft --config nvflow/recipes/finance/workflows/training_sft.yaml

  # Run a single GRPO stage
  uv run python scripts/run_flow.py collect_rollouts --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml

  # Run a GRPO stage for one environment
  uv run python scripts/run_flow.py collect_rollouts -c nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml -e equivalence_llm_judge

  # Run a GRPO stage for multiple environments
  uv run python scripts/run_flow.py training -c nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml -e mcqa equivalence_llm_judge

  # Run all stages
  uv run python scripts/run_flow.py --all --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml
        """,
    )

    parser.add_argument(
        "stages",
        nargs="*",
        help="Stage name(s) to run (e.g., download, generate_qa) - short names from config",
    )

    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to workflow configuration file",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all stages defined in the workflow config",
    )

    parser.add_argument(
        "--environment",
        "-e",
        nargs="+",
        default=None,
        help="Run for specific environment(s) only (default: all environments)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all and not args.stages:
        parser.error("Either provide stage name(s) or use --all flag")

    if args.all and args.stages:
        parser.error("Cannot specify both --all and stage names")

    try:
        # Initialize workflow runner
        runner = WorkflowRunner(args.config)

        # Run stages
        if args.all:
            print(f"Running all stages from {args.config}...")
            runner.run(environment=args.environment)
        else:
            print(f"Running {len(args.stages)} stage(s) from {args.config}...")
            runner.run(stages=args.stages, environment=args.environment)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
