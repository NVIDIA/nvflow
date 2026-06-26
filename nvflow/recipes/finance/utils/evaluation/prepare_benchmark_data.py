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
"""Utility script to prepare finance benchmark datasets.

Runs prepare.py scripts for custom finance benchmarks.

Usage:
    python -m nvflow.recipes.finance.utils.evaluation.prepare_benchmark_data \
        --benchmarks secque finqa tatqa \
        --output_dir /workspace/finance_data
"""

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

from nvflow.utils import setup_logger

# Initialize logger
logger = setup_logger(__name__)


def prepare_benchmarks(benchmarks: list[str], output_dir: str) -> None:
    """Prepare multiple benchmark datasets.

    Args:
        benchmarks: List of benchmark names (e.g., ["secque", "finqa"])
        output_dir: Output directory for prepared datasets
    """
    logger.info(f"Preparing {len(benchmarks)} benchmark(s): {', '.join(benchmarks)}")
    logger.info(f"Output directory: {output_dir}\n")

    failed = []
    for bench_name in benchmarks:
        logger.info(f"▶ Preparing {bench_name}...")

        try:
            # Import the dataset module to get its path
            dataset_module = importlib.import_module(
                f"nvflow.recipes.finance.datasets.{bench_name}.prepare"
            )
            prepare_script = Path(dataset_module.__file__)

            # Create benchmark-specific output directory
            benchmark_output_dir = Path(output_dir) / bench_name
            benchmark_output_dir.mkdir(parents=True, exist_ok=True)

            # Airgap: prepare.py downloads the benchmark from HuggingFace
            # (load_dataset). If the prepared eval.jsonl is already present
            # (baked into the deploy / produced by a prior run), skip the
            # download so this runs offline.  Pair with HF_*_OFFLINE in the
            # cluster/job env so any unguarded load_dataset fails fast.
            existing = benchmark_output_dir / "eval.jsonl"
            if existing.exists() and existing.stat().st_size > 0:
                logger.info(f"  {bench_name}: eval.jsonl already prepared — skipping download")
                logger.info(f"✓ {bench_name} prepared successfully\n")
                continue

            # Run prepare.py with benchmark-specific output_dir
            result = subprocess.run(
                [sys.executable, str(prepare_script), "--output_dir", str(benchmark_output_dir)],
                check=True,
                capture_output=True,
                text=True,
            )

            # Log output from prepare.py
            if result.stdout.strip():
                for line in result.stdout.strip().split("\n"):
                    logger.info(f"  {line}")

            logger.info(f"✓ {bench_name} prepared successfully\n")

        except ImportError as e:
            logger.error(f"✗ {bench_name} module not found: {e}\n")
            failed.append(bench_name)

        except subprocess.CalledProcessError as e:
            logger.error(f"✗ {bench_name} failed with exit code {e.returncode}")
            if e.stderr:
                logger.error(f"  {e.stderr}")
            failed.append(bench_name)
            logger.info("")

    # Summary
    logger.info("=" * 80)
    logger.info(f"Summary: {len(benchmarks) - len(failed)}/{len(benchmarks)} benchmarks prepared")

    if failed:
        logger.error(f"Failed: {', '.join(failed)}")
        sys.exit(1)

    logger.info("All benchmarks prepared successfully! ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare finance benchmark datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        required=True,
        help="List of benchmark names to prepare (e.g., secque finqa tatqa)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for prepared datasets",
    )

    args = parser.parse_args()

    prepare_benchmarks(args.benchmarks, args.output_dir)
