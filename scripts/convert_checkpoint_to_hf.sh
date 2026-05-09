#!/bin/bash
# ============================================================================
# Convert Training Checkpoint to HuggingFace Format
# ============================================================================
# Converts a Megatron or DTensor checkpoint to HuggingFace format.
#
# Supported backends:
#   megatron (default): Megatron DCP checkpoints (.distcp shards)
#                       Uses checkpoint_converter.py which auto-copies
#                       tokenizer/config from the base model.
#   dtensor:            DTensor/FSDP checkpoints
#                       Uses NeMo-RL's convert_dcp_to_hf.py + tokenizer rsync.
#
# Usage:
#   ./scripts/convert_checkpoint_to_hf.sh [OPTIONS]
#
# Required:
#   --model-name NAME        HF model name or path (e.g., Qwen/Qwen3-30B-A3B)
#   --checkpoint-dir PATH    Path to checkpoints dir (contains step_N/)
#   --step NUMBER            Step number to convert
#
# Optional:
#   --backend BACKEND        megatron (default) or dtensor
#   --output-dir PATH        Output path (default: ../manual_conversion/hf/step_N)
#   --cluster NAME           Cluster config name (default: my_cluster)
#   --num-gpus N             GPUs for conversion job (default: 4)
#
# Examples:
#   # Convert Megatron checkpoint (GSPO Qwen3-30B-A3B)
#   ./scripts/convert_checkpoint_to_hf.sh \
#       --model-name /hf_models/Qwen/Qwen3-30B-A3B \
#       --checkpoint-dir /workspace/outputs/.../checkpoints \
#       --step 18
#
#   # Convert DTensor checkpoint
#   ./scripts/convert_checkpoint_to_hf.sh \
#       --model-name Qwen/Qwen3-14B \
#       --checkpoint-dir /workspace/outputs/.../checkpoints \
#       --step 500 \
#       --backend dtensor
# ============================================================================

set -euo pipefail

# ============================================================================
# Parse Arguments
# ============================================================================
MODEL_NAME=""
CHECKPOINT_DIR=""
STEP_NUMBER=""
OUTPUT_DIR=""
BACKEND="megatron"
CLUSTER="my_cluster"
NUM_GPUS=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)      MODEL_NAME="$2"; shift 2 ;;
        --checkpoint-dir)  CHECKPOINT_DIR="$2"; shift 2 ;;
        --step)            STEP_NUMBER="$2"; shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2"; shift 2 ;;
        --backend)         BACKEND="$2"; shift 2 ;;
        --cluster)         CLUSTER="$2"; shift 2 ;;
        --num-gpus)        NUM_GPUS="$2"; shift 2 ;;
        -h|--help)         head -41 "$0" | tail -40; exit 0 ;;
        *)                 echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# Validate required args
for arg_name in MODEL_NAME CHECKPOINT_DIR STEP_NUMBER; do
    if [[ -z "${!arg_name}" ]]; then
        echo "Error: --$(echo "$arg_name" | tr '_' '-' | tr '[:upper:]' '[:lower:]') is required" >&2
        echo "Run with --help for usage" >&2
        exit 1
    fi
done

if [[ "$BACKEND" != "megatron" && "$BACKEND" != "dtensor" ]]; then
    echo "Error: --backend must be 'megatron' or 'dtensor'" >&2
    exit 1
fi

# Derive paths
STEP_PATH="${CHECKPOINT_DIR}/step_${STEP_NUMBER}"
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$CHECKPOINT_DIR")/manual_conversion/hf/step_${STEP_NUMBER}}"
LOG_DIR="$(dirname "$CHECKPOINT_DIR")/manual_conversion/logs/step_${STEP_NUMBER}"
EXPNAME="convert-$(basename "$(dirname "$CHECKPOINT_DIR")")-step${STEP_NUMBER}"

echo "============================================"
echo "Checkpoint to HuggingFace Conversion"
echo "============================================"
echo ""
echo "  Model:       ${MODEL_NAME}"
echo "  Backend:     ${BACKEND}"
echo "  Step:        ${STEP_NUMBER}"
echo "  Input:       ${STEP_PATH}"
echo "  Output:      ${OUTPUT_DIR}"
echo "  Cluster:     ${CLUSTER}"
echo "  GPUs:        ${NUM_GPUS}"
echo ""

# ============================================================================
# Build Conversion Command
# ============================================================================
cd "$(dirname "$0")/.."

if [[ "$BACKEND" == "megatron" ]]; then
    # checkpoint_converter.py handles:
    #   1. Megatron DCP -> HF weight conversion
    #   2. Auto-copy tokenizer/config from base model (fixes Bridge corruption)
    #   3. Idempotent skip if output already exists
    CONVERT_CALL="convert_checkpoint("
    CONVERT_CALL+="megatron_path=\\\"${STEP_PATH}\\\", "
    CONVERT_CALL+="hf_output_path=\\\"${OUTPUT_DIR}\\\", "
    CONVERT_CALL+="model_name=\\\"${MODEL_NAME}\\\")"

    FULL_CMD="export UV_PROJECT=/opt/NeMo-RL \
        && export PYTHONPATH=\$PYTHONPATH:/nemo_run/code \
        && uv run --extra mcore python -c \
        \"from nvflow.recipes.finance.utils.evaluation.checkpoint_converter import convert_checkpoint; ${CONVERT_CALL}\""
else
    # DTensor/FSDP: NeMo-RL's converter + rsync tokenizer from checkpoint
    FULL_CMD="export UV_PROJECT=/opt/NeMo-RL \
        && cd /opt/NeMo-RL \
        && uv run examples/converters/convert_dcp_to_hf.py \
            --config=\"${STEP_PATH}/config.yaml\" \
            --dcp-ckpt-path=\"${STEP_PATH}/policy/weights\" \
            --hf-ckpt-path=\"${OUTPUT_DIR}\" \
        && rsync -ahP \"${STEP_PATH}/policy/tokenizer/\" \"${OUTPUT_DIR}/\""
fi

# ============================================================================
# Submit Job
# ============================================================================
echo "Submitting conversion job..."
echo ""

uv run ns run_cmd \
    --cluster "${CLUSTER}" \
    --num_gpus "${NUM_GPUS}" \
    --container "nemo-rl" \
    --expname "${EXPNAME}" \
    --log_dir "${LOG_DIR}" \
    --command "${FULL_CMD}"

echo ""
echo "============================================"
echo "Job submitted!"
echo "============================================"
echo ""
echo "Monitor logs at: ${LOG_DIR}"
echo "Output HF model: ${OUTPUT_DIR}"
echo ""
