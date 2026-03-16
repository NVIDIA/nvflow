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
"""Megatron and DCP (FSDP) → HuggingFace checkpoint conversion.

Megatron conversion runs in this module (via CLI or convert_checkpoint).
DCP conversion builds a bash script that resolves the run directory and
invokes NeMo-RL's convert_dcp_to_hf.py on the cluster.

Usage (from eval stage):
    from nvflow.recipes.finance.utils.evaluation.checkpoint_converter import (
        build_conversion_script,       # Megatron
        build_dcp_conversion_script,   # DCP/FSDP
        get_hf_output_paths,
    )
"""

import argparse
import sys
from pathlib import Path

from nvflow.utils import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# Core Conversion Logic (runs on cluster)
# ============================================================================


def convert_checkpoint(
    megatron_path: str | Path,
    hf_output_path: str | Path,
    model_name: str,
) -> bool:
    """
    Convert Megatron checkpoint to HuggingFace format.

    This function runs ON THE CLUSTER where paths are valid. It:
    1. Checks if HF model already exists → skip (idempotent)
    2. Checks if Megatron checkpoint exists → convert
    3. Neither exists → error

    Args:
        megatron_path: Path to Megatron checkpoint (e.g., .../checkpoints/step_5000)
        hf_output_path: Where to save HF model (e.g., .../hf_models/step_5000)
        model_name: HF model name for tokenizer/architecture (e.g., "Qwen/Qwen3-14B")

    Returns:
        True if conversion was performed, False if skipped (already exists)

    Raises:
        FileNotFoundError: If Megatron checkpoint doesn't exist
        RuntimeError: If conversion fails
    """
    megatron_path = Path(megatron_path)
    hf_output_path = Path(hf_output_path)

    logger.info("=" * 60)
    logger.info("CHECKPOINT CONVERSION")
    logger.info("=" * 60)
    logger.info(f"Megatron:  {megatron_path}")
    logger.info(f"HF output: {hf_output_path}")
    logger.info(f"Model:     {model_name}")
    logger.info("")

    # Check 1: HF model already exists? Skip.
    if (hf_output_path / "config.json").exists():
        logger.info(f"✓ HF model already exists at {hf_output_path}")
        logger.info("Skipping conversion (idempotent)")
        return False

    # Check 2: Megatron checkpoint exists?
    weights_path = megatron_path / "policy" / "weights"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Megatron checkpoint not found at {megatron_path}\nExpected: {weights_path}/"
        )

    # Convert Megatron → HuggingFace
    logger.info("Converting Megatron checkpoint to HuggingFace format...")

    # Import here to avoid loading nemo-rl when not needed
    from nemo_rl.models.megatron.community_import import export_model_from_megatron

    # Note: hf_overrides is not passed due to nemo-rl 0.7.1 bug.
    # The bug is patched via installation_command in eval/base.yaml.
    input_path = weights_path / "iter_0000000"
    export_model_from_megatron(
        hf_model_name=model_name,
        input_path=str(input_path),
        output_path=str(hf_output_path),
        hf_tokenizer_path=model_name,
        overwrite=False,
    )

    # Verify conversion succeeded
    if not (hf_output_path / "config.json").exists():
        raise RuntimeError(f"Conversion failed - {hf_output_path}/config.json not found")

    logger.info("")
    logger.info(f"✓ Conversion complete: {hf_output_path}")
    logger.info("=" * 60)
    return True


# ============================================================================
# Helper Functions (used by eval stage)
# ============================================================================


def get_hf_output_paths(run_path: str | Path, step: int) -> tuple[Path, Path]:
    """
    Derive HF model and log paths for a given training run and step.

    Given a training run at:
        .../model-qwen3-14b-32n-tp4-pp1-cp8-seq48k

    Returns paths for step 5000:
        HF model: .../hf_models/step_5000
        Logs: .../hf_models/convert-logs/step_5000

    Args:
        run_path: Path to training run directory
        step: Checkpoint step number

    Returns:
        Tuple of (hf_model_path, convert_log_dir)
    """
    run_path = Path(run_path)
    step_name = f"step_{step}"

    hf_model_path = run_path / "hf_models" / step_name
    convert_log_dir = run_path / "hf_models" / "convert-logs" / step_name

    return hf_model_path, convert_log_dir


def build_conversion_script(
    megatron_path: str | Path,
    hf_output_path: str | Path,
    model_name: str,
) -> str:
    """
    Build a bash script that runs the conversion.

    The script invokes this module as a CLI tool on the cluster.

    Args:
        megatron_path: Path to Megatron checkpoint
        hf_output_path: Where to save HF model
        model_name: HF model name for tokenizer/architecture

    Returns:
        Bash script as a string
    """
    script = f"""
set -e

export UV_PROJECT=/opt/NeMo-RL
uv run --extra mcore python -m nvflow.recipes.finance.utils.evaluation.checkpoint_converter \\
    --megatron-path "{megatron_path}" \\
    --hf-output-path "{hf_output_path}" \\
    --model-name "{model_name}"
"""
    return script


def build_dcp_conversion_script(
    checkpoint_path: str | Path,
    step: int,
    hf_output_path: str | Path,
) -> str:
    """Build a bash script that converts a DCP (FSDP) checkpoint to HF format.

    The script runs ON THE CLUSTER. It resolves the run subdirectory under
    checkpoint_path (flat or GRPO layout), then calls NeMo-RL's converter.

    Args:
        checkpoint_path: Parent dir (e.g. .../step-7-training) — may contain
            a nested run dir like grpo-qwen3-4b-.../checkpoints/step_N/...
        step: Checkpoint step number
        hf_output_path: Where to write HF model (known at submit time)
    """
    step_name = f"step_{step}"
    script = f"""
set -euo pipefail

CKPT_ROOT="{checkpoint_path}"
STEP_NAME="{step_name}"
HF_OUTPUT="{hf_output_path}"

# Skip if already converted
if [ -f "$HF_OUTPUT/config.json" ]; then
    echo "HF model already exists at $HF_OUTPUT — skipping"
    exit 0
fi

# Resolve run_path: flat layout or nested GRPO subdirectory
if [ -d "$CKPT_ROOT/checkpoints/$STEP_NAME/policy/weights" ]; then
    RUN_PATH="$CKPT_ROOT"
else
    RUN_PATH=""
    for d in "$CKPT_ROOT"/*/; do
        if [ -d "${{d}}checkpoints/$STEP_NAME/policy/weights" ]; then
            RUN_PATH="${{d%/}}"
            break
        fi
    done
    if [ -z "$RUN_PATH" ]; then
        echo "ERROR: No DCP checkpoint for $STEP_NAME under $CKPT_ROOT" >&2
        exit 1
    fi
fi

STEP_DIR="$RUN_PATH/checkpoints/$STEP_NAME"
echo "Resolved run path: $RUN_PATH"
echo "Converting DCP checkpoint: $STEP_DIR -> $HF_OUTPUT"

cd /opt/NeMo-RL
uv run examples/converters/convert_dcp_to_hf.py \\
    --config="$STEP_DIR/config.yaml" \\
    --dcp-ckpt-path="$STEP_DIR/policy/weights" \\
    --hf-ckpt-path="$HF_OUTPUT"

rsync -ahP "$STEP_DIR/policy/tokenizer/" "$HF_OUTPUT/"
echo "DCP conversion complete: $HF_OUTPUT"
"""
    return script


# ============================================================================
# CLI Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Convert Megatron checkpoint to HuggingFace format"
    )
    parser.add_argument(
        "--megatron-path",
        required=True,
        help="Path to Megatron checkpoint (e.g., .../checkpoints/step_5000)",
    )
    parser.add_argument(
        "--hf-output-path",
        required=True,
        help="Where to save HF model (e.g., .../hf_models/step_5000)",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="HF model name for tokenizer/architecture (e.g., Qwen/Qwen3-14B)",
    )
    args = parser.parse_args()

    try:
        convert_checkpoint(
            megatron_path=args.megatron_path,
            hf_output_path=args.hf_output_path,
            model_name=args.model_name,
        )
        return 0
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(f"✗ ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
