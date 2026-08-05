# Workflow 5: Evaluation

## Purpose

Evaluate fine-tuned models and baselines on financial reasoning benchmarks.

Evaluation runs in two modes:

- **Baseline evaluation** — standalone workflow (`eval/baselines.yaml`) that evaluates pre-trained models for comparison.
- **Checkpoint evaluation** — embedded as the last stage of SFT and GRPO training pipelines, so checkpoints are evaluated immediately after training completes.

## Pipeline Flow

```
SFT Pipeline                          GRPO Pipeline
┌──────────────────────┐              ┌──────────────────────┐
│ data_transformation  │              │ data_transformation  │
│ prepare_for_sft      │              │ apply_prompt_template│
│ train_validation_split│             │ convert_to_responses_api│
│ sequence_length_group│              │ train_validation_split│
│ training             │              │ prepare_data         │
│ ─────────────────────│              │ collect_rollouts     │
│ eval  ← checkpoints  │              │ training             │
│   + baseline         │              │ ─────────────────────│
└──────────────────────┘              │ eval  ← checkpoints  │
                                      │   + baseline         │
Standalone Baselines                  └──────────────────────┘
┌──────────────────────┐
│ prepare_data         │
│ qwen3-4b             │
│ qwen3-14b            │
│ qwen3-32b            │
│ gpt-oss-120b         │
│ nemotron-nano-9b-v2  │
│ nemotron-3-nano-30b  │
└──────────────────────┘
```

> **Offline clusters:** `prepare_data` downloads the benchmark datasets from HuggingFace, so temporarily clear `HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE` and `TRANSFORMERS_OFFLINE` for its first run, then restore them. The datasets persist and are reused afterwards. See [Offline runtime](../troubleshooting.md#offline-runtime).

## Configuration

**Directory:** `workflows/eval/`

| File | Purpose |
|------|---------|
| `eval/base.yaml` | Shared: benchmarks, judges, datasets, conversion settings |
| `eval/baselines.yaml` | Standalone workflow for all baseline (pre-trained) models |

Checkpoint evaluation is configured directly in the training configs:

| File | Eval Config |
|------|-------------|
| `sft/qwen3_4b.yaml` | `stages.eval` with `eval_steps: [10]` |
| `sft/qwen3_14b.yaml` | `stages.eval` with `eval_steps: [2600, 5000, 7408]` |
| `grpo/qwen3_4b.yaml` (equivalence, FSDP) | `stages.eval` with `eval_steps: [20]` |
| `grpo/qwen3_4b_finsec.yaml` (finance_sec_search, Megatron) | `stages.eval` with `eval_steps: [20]` |

## Usage

### Training + Evaluation (Recommended)

Run training and evaluation together — eval runs automatically after training:

```bash
# SFT training + checkpoint eval
uv run nflow run-all --config nvflow/recipes/finance/workflows/sft/qwen3_4b.yaml

# GRPO training + checkpoint eval
uv run nflow run-all --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml
```

### Eval Stage Only

Run just the eval stage from a training config:

```bash
# Evaluate SFT checkpoints only (skip training)
uv run nflow run eval --config nvflow/recipes/finance/workflows/sft/qwen3_14b.yaml
```

### Baseline Evaluation

Evaluate all baseline (pre-trained) models:

```bash
# All baselines
uv run nflow run-all --config nvflow/recipes/finance/workflows/eval/baselines.yaml

# Specific baseline
uv run nflow run gpt-oss-120b --config nvflow/recipes/finance/workflows/eval/baselines.yaml
```

## Evaluation Benchmarks

Configured in `eval/base.yaml`, shared across all evaluation contexts:

- **SEC-QUE**: SEC filing comprehension (565 samples)
- **FinanceBench**: Financial question answering (150 samples)

`finance_agent` (multi-turn agentic financial QA from [vals-ai/finance-agent](https://github.com/vals-ai/finance-agent)) is **disabled** — `eval/base.yaml` sets it to `null` pending validation of the multi-turn tool-calling path. See [finance-agent-eval](../stages/finance-agent-eval.md) to re-enable it.

## Eval Stage Configuration (in Training YAMLs)

Add to your SFT or GRPO model config:

```yaml
stages:
  eval:
    eval_steps: [1000, 3000, 5000]
    checkpoint_path: ${directories.step-4-training}/model-name
    format: megatron        # Match the checkpoint's training backend: "fsdp" or "megatron"
    baseline_model: /hf_models/Qwen/Qwen3-14B
    server_type: vllm
    gpus: 1
    inference_args: >-
      ++prompt_config=nvflow/recipes/finance/prompts/secque_template.yaml
      ++inference.temperature=0.6
      ++inference.top_p=0.95
      ++inference.top_k=20
      ++inference.tokens_to_generate=16384
      ++chat_template_kwargs.enable_thinking=true
    server_args: "--max-model-len 40960 --async-scheduling --reasoning-parser qwen3"
```

The eval stage will:
1. Evaluate each checkpoint at the specified steps
2. Evaluate the baseline model for comparison
3. Skip benchmarks that already have `metrics.json` results

## Output Structure

Checkpoint evaluations output alongside the training results:

```
outputs/finance/demo/workflow-4-sft/qwen3_4b/
├── step-4-training/          # Training checkpoints
└── eval/
    ├── step-10/
    │   └── eval-results/{benchmark}/metrics.json
    └── baseline/
        └── eval-results/{benchmark}/metrics.json
```

Baseline evaluations output to the standalone eval directory:

```
outputs/finance/sap-500/workflow-1-baseline-eval/
└── baselines/
    ├── qwen3-14b/
    │   └── eval-results/{benchmark}/metrics.json
    ├── qwen3-32b/
    └── gpt-oss-120b/
```

## Adding Eval to a New Model

### Step 1: Add eval to pipeline_stages

```yaml
pipeline_stages:
  - data_transformation
  - prepare_for_sft
  - train_validation_split
  - sequence_length_grouping
  - training
  - eval                    # Add this
```

### Step 2: Add eval stage config

```yaml
stages:
  eval:
    eval_steps: [100, 500, 1000]
    checkpoint_path: ${directories.step-4-training}/model-my-model-name
    format: megatron        # Match the checkpoint's training backend: "fsdp" or "megatron"
    baseline_model: /hf_models/MyOrg/MyModel
    server_type: vllm
    gpus: 1
    inference_args: >-
      ++prompt_config=nvflow/recipes/finance/prompts/secque_template.yaml
    server_args: "--max-model-len 40960"
```

### Step 3: Run

```bash
uv run nflow run-all --config nvflow/recipes/finance/workflows/sft/my_model.yaml
```

## Common Issues

### Model checkpoint not found

**Solution:**
- Verify `checkpoint_path` matches the actual training output directory
- Check if training completed successfully
- Use absolute paths in config

### OOM during evaluation

**Solution:**
- Reduce `--max-model-len` in `server_args`
- Increase `gpus` for larger models
- Use tensor parallelism for 32B+ models (`gpus: 2`)

### Low scores on all models

**Solution:**
- Check evaluation data quality
- Verify judge prompt is appropriate
- Review prediction outputs manually

## Technical Details

For comprehensive stage-by-stage documentation:
- **[Eval Stages Reference](../stages/eval.md)**

For benchmark details:
- FinanceBench: [Link to benchmark]
- SEC-QUE: [Link to dataset]
