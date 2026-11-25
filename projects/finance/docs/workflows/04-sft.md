# Supervised Fine-Tuning (SFT)

## Purpose

Fine-tune language models on synthetic financial Q&A data generated from SDG workflows.

> **Note:** Current production workflow uses [Template-Based SDG](02-template-based-sdg.md) data (~300K pairs).

## Quick Navigation

- [Prerequisites](#prerequisites)
- [Workflow Overview](#workflow-overview)
- [Usage](#usage)
- [Data & Results](#data--results)
- [Customization](#customization)
- [Additional Resources](#additional-resources)

---

## Prerequisites

- ✅ Q&A data from [Template-Based SDG](02-template-based-sdg.md)
- ✅ Base model accessible on cluster (Qwen3-14B)
- ✅ GPU resources (256 GPUs / 32 nodes)

## Workflow Overview

### Stages

```
┌─────────────────────────┐
│ 0. data_transformation  │  Conversion: Q&A format → Training format
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 1. prepare_for_sft      │  Preparation: Apply prompt template & format for SFT
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 2. train_validation_    │  Splitting: Create train/validation sets
│    split                │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 3. sequence_length_     │  Optimization: Group by length for efficiency
│    grouping             │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 4. training             │  Fine-tuning: Train the model
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ 5. eval                 │  Evaluation: Score checkpoints on finance benchmarks
└─────────────────────────┘
```

**6 Stages:**
1. **data_transformation** (Step 0): Convert Q&A format to training format
2. **prepare_for_sft** (Step 1): Prepare data for SFT (formatting, filtering)
3. **train_validation_split** (Step 2): Split into train/validation sets
4. **sequence_length_grouping** (Step 3): Group by sequence length for efficiency
5. **training** (Step 4): Fine-tune the model
6. **eval** (Step 5): Evaluate checkpoints on finance benchmarks

> **Qwen3 models add a seventh stage.** `qwen3_14b.yaml` inserts `convert_to_messages` between `training` and `eval` to convert checkpoints to the OpenAI messages format. Other configs, including the `qwen3_4b.yaml` demo, run the six stages above.

**See [technical reference](#detailed-stage-reference) for detailed stage documentation.**

### Model Configurations

Pre-configured workflows for supported models:

| Config | Model | Architecture | Params | GPUs | Cluster Config | Status |
|--------|-------|-------------|--------|------|----------------|--------|
| `sft/qwen3_14b.yaml` | Qwen3-14B | Dense | 14B | 256 (32 nodes) | `my_cluster` | Production |
| `sft/qwen3_4b.yaml` | Qwen3-4B | Dense | 4B | 8 (1 node) | `my_cluster` | Demo |
| `sft/gemma3_1b_it.yaml` | Gemma3-1B-IT | Dense | 1B | 8 (1 node) | `my_cluster` | Example / Demo |
| `sft/gemma3_4b_it.yaml` | Gemma3-4B-IT | Dense | 4B | 8 (1 node) | `my_cluster` | Example / Demo |
| `sft/gemma3_27b_it.yaml` | Gemma3-27B-IT | Dense | 27B | 16 (2 nodes) | `my_cluster` | Example / Demo |
| `sft/nemotron-3-nano.yaml` | Nemotron-3-Nano-30B | MoE Hybrid | 30B (3.5B active) | 256 (32 nodes) | `my_cluster` | Production |
| `sft/base.yaml` | Base template | - | - | - | `my_cluster` | - |

> **Note:** NeMo-RL v0.8.0 contains Nemotron-3-Nano support; do not overlay a
> separate NeMo-RL checkout.
>
> **Gemma3 configs** (1B, 4B, 27B) are **example / demo** only: they demonstrate Gemma SFT with FSDP2 and validated max sequence lengths. They are not production workflows.

## Usage

> **📋 Before You Start:** Make sure to update the `raw_train_data` path in `base.yaml` or your model config to point to your SDG output directory. See [Input Data](#input-data) section below for details.

### Run Complete Workflow

```bash
# Qwen3-14B (production)
uv run nvflow run --all --config projects/finance/workflows/sft/qwen3_14b.yaml --cluster my_cluster

# Qwen3-4B (demo)
uv run nvflow run --all --config projects/finance/workflows/sft/qwen3_4b.yaml --cluster my_cluster

# Gemma3-1B-IT (example / demo)
uv run nvflow run --all --config projects/finance/workflows/sft/gemma3_1b_it.yaml --cluster my_cluster

# Gemma3-4B-IT (example / demo)
uv run nvflow run --all --config projects/finance/workflows/sft/gemma3_4b_it.yaml --cluster my_cluster

# Gemma3-27B-IT (example / demo, 2 nodes)
uv run nvflow run --all --config projects/finance/workflows/sft/gemma3_27b_it.yaml --cluster my_cluster

# Nemotron-3-Nano-30B (production)
uv run nvflow run --all --config projects/finance/workflows/sft/nemotron-3-nano.yaml --cluster my_cluster
```

### Stage-by-Stage Execution

Replace `CONFIG` with the path to your model config (e.g.,
`projects/finance/workflows/sft/qwen3_14b.yaml`).

```bash
# Step 0: Transform data
uv run nvflow run data_transformation --config CONFIG --cluster my_cluster

# Step 1: Prepare for SFT
uv run nvflow run prepare_for_sft --config CONFIG --cluster my_cluster

# Step 2: Split train/val
uv run nvflow run train_validation_split --config CONFIG --cluster my_cluster

# Step 3: Group by length
uv run nvflow run sequence_length_grouping --config CONFIG --cluster my_cluster

# Step 4: Train
uv run nvflow run training --config CONFIG --cluster my_cluster

# Step 5: Convert to messages (qwen3_14b.yaml only)
uv run nvflow run convert_to_messages --config CONFIG --cluster my_cluster

# Step 5: Evaluate checkpoints
uv run nvflow run eval --config CONFIG --cluster my_cluster
```

## Data & Results

### Input Data

#### Preparing SDG Output for SFT

The SFT workflow expects input data at a specific location. The `base.yaml` config defines:

```yaml
directories:
  raw_train_data: /workspace/outputs/finance/sp500/workflow-3-template-based-sdg/step-5-filter-answers
```

The `data_transformation` stage reads from: `${raw_train_data}/final_result.jsonl`

#### Setting Up Input Data

The default config points to the template-based SDG output. You can override if needed:

```yaml
# In your SFT config or command line
directories:
  raw_train_data: /workspace/outputs/finance/sp500/workflow-3-template-based-sdg/step-5-filter-answers
```

The input file should be the `final_result.jsonl` from the template-based SDG workflow (step-5-filter-answers).

### Output Structure

```
outputs/finance/sp500/workflow-4-sft/qwen3_14b/
├── step-0-data-transformation/
│   └── chunks/
│       └── final_result_chunk*.jsonl
├── step-1-prepare-for-sft/
│   └── final_result.jsonl           # ~366K samples
├── step-2-train-validation-split/
│   ├── train.jsonl                  # ~330K samples
│   └── val.jsonl                    # ~37K samples
├── step-3-sequence-length-grouping/  # [Optional]
│   └── train_bucket_*.jsonl
├── step-4-training/
│   └── model-qwen3-14b-256g-tp4-pp1-cp8-seq48k/
│       ├── checkpoints/
│       │   ├── step_100/            # Megatron checkpoint every save_period steps
│       │   ├── step_200/
│       │   └── ...
│       ├── hf_models/
│       │   └── step_<N>/            # HF export of each evaluated step
│       └── training-logs/
├── step-5-convert-to-messages/       # [Optional]
│   ├── train.jsonl
│   └── val.jsonl
└── step-5-eval/
    └── step-<N>/collect-rollouts/{benchmark}/
```

### Expected Results

| Model | Training Samples | Validation Samples | Training Duration | Final Checkpoint |
|-------|------------------|-------------------|-------------------|------------------|
| Qwen3-14B (256 GPUs) | ~330K | ~37K | 12-16 hours (4 jobs) | `outputs/finance/sp500/workflow-4-sft/qwen3_14b/step-4-training/.../checkpoints/step_<N>/` (highest step) |

**Note:** Training runs as 4 sequential jobs (~4 hours each max). Total time depends on cluster availability and whether the final job finishes early.

### Validation

`$WORKSPACE` is the host directory your target mounts at `/workspace`, as in the
[quick start](../quick-start.md).

```bash
# Check training outputs
OUTPUT_DIR="$WORKSPACE/outputs/finance/sp500/workflow-4-sft/qwen3_14b"

# Verify train/val split
wc -l $OUTPUT_DIR/step-2-train-validation-split/train.jsonl  # Should be ~330K
wc -l $OUTPUT_DIR/step-2-train-validation-split/val.jsonl    # Should be ~37K

# Check prepared data
wc -l $OUTPUT_DIR/step-1-prepare-for-sft/final_result.jsonl  # Should be ~366K

# List checkpoints (step_<N>/ every save_period steps)
ls $OUTPUT_DIR/step-4-training/model-qwen3-14b-256g-tp4-pp1-cp8-seq48k/checkpoints/

# Check a Hugging Face export, written when eval converts that step
ls $OUTPUT_DIR/step-4-training/model-qwen3-14b-256g-tp4-pp1-cp8-seq48k/hf_models/step_7408/
# Should contain: config.json, model weights, tokenizer files
```

## Customization

The production configuration (`sft/qwen3_14b.yaml`) serves as a reference implementation. You can customize it for:
- **Different models** - Train models other than Qwen3-14B
- **Resource constraints** - Adjust for available GPU/node counts
- **Training behavior** - Modify epochs, checkpointing, validation frequency

All customization follows the same pattern: inherit from `base.yaml` and override specific settings.

### Training a Different Model

To train a model other than Qwen3-14B, create a new config following the `qwen3_14b.yaml` pattern. Name it `private_*.yaml` so git ignores it and `run` accepts it, or commit it:

```yaml
# projects/finance/workflows/sft/private_my_model.yaml
_base_: base.yaml

# Model-specific output directory
base_output_dir: /workspace/outputs/finance/sp500/workflow-4-sft/my_model

stages:
  # Step 1: Update tokenizer for your model
  prepare_for_sft:
    prepare_data_kwargs:
      ctx_args: >-
        ++tokenizer=/hf_models/MyOrg/MyModel
        ++prompt_config=project://prompts/secque_template.yaml

  # Step 3: Use same tokenizer for sequence grouping
  sequence_length_grouping:
    tokenizer_path: /hf_models/MyOrg/MyModel

  # Step 4: Training configuration
  training:
    model_name: MyOrg/MyModel
    hf_checkpoint_path: /hf_models/MyOrg/MyModel
    total_gpus: 64                  # Adjust for your model size
    preset: "sft-base"              # The preset sft_presets.yaml defines

    overrides:
      sft:
        max_num_epochs: 3
      checkpointing:
        save_period: 100
      policy:
        train_global_batch_size: 128
        max_total_sequence_length: 49152
        sequence_packing:           # Required by Megatron when context_parallel_size > 1
          enabled: true
          train_mb_tokens: 49152
        # Adjust parallelism for your model size
        megatron_cfg:
          tensor_model_parallel_size: 2
          pipeline_model_parallel_size: 1
          context_parallel_size: 4
          scheduler:
            lr_warmup_iters: 2000
```

`overrides` is deep-merged into the preset without checking key names, so a key
outside NeMo-RL's config schema is silently ignored. Use the NeMo-RL paths
listed under [training](#training).

`sft-base` is the only preset in `sft_presets.yaml`, and an unknown `preset` is
rejected before submission. A custom preset must start as a copy of `sft-base`: it
carries keys that NeMo-RL v0.8.0 requires, such as
`policy.offload_policy_before_refit` and `policy.offload_optimizer_for_refit`.

**Key configuration points:**
1. Update `tokenizer` path in `prepare_for_sft` (Step 1) and `sequence_length_grouping` (Step 3)
2. Set `model_name` and `hf_checkpoint_path` to your model in `training` (Step 4)
3. Adjust `total_gpus` based on model size and available resources
4. Tune `policy.megatron_cfg` parallelism for your model architecture

### Adjusting Training Parameters

You can modify training behavior without changing the model. Common adjustments:

```yaml
training:
  # === Resource Configuration ===
  total_gpus: 128                   # Reduce for smaller runs (vs 256 in production)
  dependent_jobs: 1                 # Fewer job splits (vs 3 in production)

  overrides:
    sft:
      max_num_epochs: 5             # More epochs for smaller datasets
      val_period: 100               # Validate less frequently

    policy:
      train_global_batch_size: 64   # Smaller for limited resources
      megatron_cfg:
        scheduler:
          lr_warmup_iters: 1000     # Adjust warmup period

    checkpointing:
      save_period: 200              # Save checkpoints less frequently

  # Experiment tracking
  wandb_mode: offline               # Use offline mode, or 'disabled' to turn off
```

**Parameter guide:**
- **`total_gpus`**: Total GPUs for training (auto-split across nodes using `gpus_per_node` from cluster config)
- **`dependent_jobs`**: Training split count (higher = more job restarts, lower max
  time per job). Only continuation jobs use `afterany`; the first job retains
  the cluster's fail-closed `afterok` dependency on upstream stages.
- **`sft.max_num_epochs`**: Total training passes through the dataset
- **`policy.megatron_cfg.scheduler.lr_warmup_iters`**: Learning rate warmup steps on the Megatron backend (typically 5-10% of total steps)
- **`policy.train_global_batch_size`**: Effective batch size across all GPUs
- **`sft.val_period`**: Steps between validation runs
- **`checkpointing.save_period`**: Steps between checkpoint saves

## Model-Specific Notes

### Nemotron-3-Nano Special Requirements

Nemotron-3-Nano-30B is a **Mixture-of-Experts (MoE) Hybrid** model with a unique architecture (23 Mamba-2 + MoE layers, 6 Attention layers). It requires special infrastructure setup:

#### NeMo-RL runtime

NVFlow's `nemo-rl` image, `nvflow-nemo-rl:v0.8.0`, builds on NeMo-RL 0.8.0,
which contains Nemotron-3-Nano MoE support, so no NeMo-RL clone, source
overlay, or `my_cluster_nemotron` target is required.

Do not mount a host checkout over `/opt/nemo-rl`; that would change the source
without changing the image's locked actor environments. Verify the selected
runtime from the job log or an allocation:

```bash
/opt/nemo_rl_venv/bin/python -c \
  'import nemo_rl, platform, ray; print(platform.python_version(), ray.__version__)'
```

#### Key Architecture Differences

| Property | Nemotron-3-Nano | Qwen3 / Gemma3 (Dense) |
|----------|-----------------|------------------------|
| Architecture | MoE Hybrid (Mamba-2 + MoE) | Dense Transformer |
| Total / Active params | 30B / 3.5B | Full params active |
| Expert Parallelism (EP) | 8 | N/A (ignored) |
| Expert Tensor Parallelism (ETP) | 1 | N/A (ignored) |
| Tensor Parallelism (TP) | 4 | 1–8 (model-dependent; Gemma3: 1/4/8 for 1B/4B/27B) |
| Context Parallelism (CP) | 8 | 1–8 (model-dependent; Gemma3 uses 1, no sequence packing) |
| Cluster config | `my_cluster` | `my_cluster` |
| Sequence packing | Required (CP > 1) | Optional |

### Qwen3 Models

- **Qwen3-14B** (production): 32 nodes, TP=4, CP=8, 48K sequence length
- **Qwen3-4B** (demo): 1 node, TP=2, CP=2, 32K sequence length
- Only `qwen3_14b.yaml` adds the `convert_to_messages` stage for OpenAI messages format conversion; the Qwen3-4B demo runs the six standard stages
- Chat template: `<|im_start|>role\ncontent<|im_end|>`

### Gemma3 models (1B, 4B, 27B) — example / demo

These configs are **example runs** for testing Gemma SFT (demo data, small scale). They are not production workflows.

**Recommendation (Gemma only):** Use **FSDP2 over Megatron** for Gemma3 SFT. Megatron has known issues with Gemma3 (262K-vocab logprob OOM, activation checkpointing failure); we use FSDP2 in these configs. Stay within the **max sequence lengths we validated** below; longer sequences may OOM without code changes (e.g. chunked logprob). This recommendation applies to **Gemma only**; other models (e.g. Qwen3) continue to use Megatron.

All Gemma3 variants in these configs use the **FSDP2 backend**. Megatron blocks are kept as reference only in the YAML.

| Model | Nodes | Backend | TP | Max sequence length (validated) | Notes |
|-------|-------|---------|-----|----------------------------------|-------|
| Gemma3-1B-IT | 1 | FSDP2 | 1 | 16K | 1 GQA KV head → TP=1 only |
| Gemma3-4B-IT | 1 | FSDP2 | 4 | 16K | 4 GQA KV heads; 16K max we tried successfully |
| Gemma3-27B-IT | 2 | FSDP2 | 8 | 16K (32K likely with TP=8) | 16 GQA KV heads; 16K configured, 32K not yet tried |

- **Max token limit we tried:** 16K for 4B and 1B (runs successfully). 27B is set to 16K conservatively; 32K is expected to fit with TP=8 vocab split but was not validated in our runs.
- **Sequence packing:** Disabled (Gemma3 does not support `packed_seq_params`).
- Does **not** include `convert_to_messages` (Qwen3-specific chat template parser).
- Chat template: `<start_of_turn>role\ncontent<end_of_turn>`

## Additional Resources

### Monitoring Training

#### Weights & Biases (wandb)

Training metrics can be tracked in a wandb dashboard. `sft/base.yaml` leaves it
off:

```yaml
# sft/base.yaml, under stages.training
wandb_project: finance-training
wandb_mode: disabled  # online | offline | disabled
```

`sft/nemotron-3-nano.yaml`, `grpo/nemotron_nano_30b_a3b.yaml` and
`grpo/qwen3_30b_a3b.yaml` set `wandb_mode: online`. Before running them, export
`WANDB_API_KEY` in the shell that runs `nvflow`; a Slurm target forwards it to
the job (see the API-key notes in
[`template-slurm.yaml`](../../../../cluster_configs/template-slurm.yaml)), and on
Ray it must be set in the clusters' environment.

**If you do not have WandB credentials, set the mode to `disabled`** in an overlay:

```yaml
stages:
  training:
    wandb_mode: disabled  # online | offline | disabled
```

**View metrics:**
- Loss curves
- Learning rate schedule
- Validation metrics
- GPU utilization

#### Check Training Progress

```bash
# Count completed checkpoints
ls ${OUTPUT_DIR}/step-4-training/model-qwen3-14b-256g-tp4-pp1-cp8-seq48k/checkpoints/ | grep '^step_' | wc -l

# View latest checkpoint
ls -lht ${OUTPUT_DIR}/step-4-training/model-qwen3-14b-256g-tp4-pp1-cp8-seq48k/checkpoints/ | head
```

### Troubleshooting

For common issues and solutions, see:
- **[Troubleshooting Guide](../troubleshooting.md)** - Comprehensive solutions for:
  - OOM errors
  - Training loss not decreasing
  - Checkpoint saving failures
  - Model loading errors

### Next Steps

After training:

- **[Evaluation](05-eval.md)** - Evaluate the fine-tuned model
- **Compare Models** - Train different sizes and compare
- **Hyperparameter Tuning** - Experiment with learning rates, batch sizes

### Technical Details

This workflow uses **NeMo-RL** for distributed training. Backend is model-dependent: **Megatron** for Qwen3 and Nemotron-3-Nano; **FSDP2 (Automodel)** for Gemma3 (1B, 4B, 27B), which avoids Megatron’s known Gemma3 issues (logprob OOM, activation checkpointing).

For comprehensive stage-by-stage documentation:
- **[SFT Stages Reference](#detailed-stage-reference)**

For NeMo training details:
- Check NeMo documentation for advanced training options
- See `projects/finance/nvfinance/stages/sft/` and `projects/finance/nvfinance/stages/shared/` for stage implementations

## Detailed stage reference
Technical reference for all 6 stages in the SFT (Supervised Fine-Tuning) workflow.

> **Note:** Current production pipeline uses template-based SDG data (~300K pairs).

### Quick Navigation

- [data_transformation](#data_transformation)
- [prepare_for_sft](#prepare_for_sft)
- [train_validation_split](#train_validation_split)
- [sequence_length_grouping](#sequence_length_grouping)
- [training](#training)
- [convert_to_messages](#convert_to_messages)

---

### data_transformation

**File:** `projects/finance/nvfinance/stages/shared/data_transformation.py`
**Binding:** workflow alias `data_transformation` selects `DataTransformationStage`

#### Purpose

Transform SDG Q&A data to standardized training format. Renames fields, generates UUIDs, moves metadata, and computes length statistics.

#### Inputs

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input_files` | list | Q&A data from SDG workflows (can be single string) | Required |
| `output_file` | path | Single file output (unused when chunking) | Required |
| `output_dir` | path | Output directory (chunks saved here) | Required |
| `num_chunks` | int | Number of output chunks for memory efficiency | 1 |
| `source_format` | string | SDG data structure: "separated" \| "think_tags" \| "inline" | "separated" |
| `reasoning_mode` | string | Generation formatting: "thinking" \| "natural" \| "none" | "none" |
| `filter_outliers` | bool | Enable percentile-based outlier filtering (`sft/base.yaml` sets `true`) | false |
| `filter_config` | dict | Percentile thresholds for filtering | See config |

#### Source Format Options

- **"separated"**: Reasoning model SDG with separate `answer_reasoning_content` field
- **"think_tags"**: Reasoning model SDG with `<think>` tags in answer
- **"inline"**: Non-reasoning model SDG (keep as-is)

#### Reasoning Mode Options

- **"thinking"**: Wrap reasoning in `<think>` tags (Qwen3 style): `<think>{reasoning}</think>\n\n{answer}`
- **"natural"**: Concatenate reasoning + answer: `{reasoning}\n\n{answer}`
- **"none"**: Answer only, discard reasoning

> **Note:** `source_format="inline"` + `reasoning_mode="thinking"` is blocked (invalid combination)

#### Field Transformations

Renames fields from SDG format:

| SDG Field | Training Field | Description |
|-----------|----------------|-------------|
| `question` | `problem` | Question text |
| `source_info` | `context` | Context/source information |
| `answer` | `generation` | Answer text |
| `answer_reasoning_content` | `reasoning_content` | Reasoning content (if separated) |
| - | `uuid` | Generated deterministic hash |
| Other fields | `metadata` | Moved to metadata object |

**Input Format (SDG):**
```json
{
  "question": "What was NVIDIA's revenue in 2023?",
  "source_info": "...SEC filing context...",
  "answer": "NVIDIA's revenue was $26.9 billion.",
  "answer_reasoning_content": "Based on the 10-K filing..."
}
```

**Output Format (Training):**
```json
{
  "problem": "What was NVIDIA's revenue in 2023?",
  "context": "...SEC filing context...",
  "generation": "NVIDIA's revenue was $26.9 billion.",
  "reasoning_content": "Based on the 10-K filing...",
  "uuid": "abc123...",
  "metadata": {"QID": "...", "question_type": "..."}
}
```

#### Outputs

- `${output_dir}/chunks/final_result_chunk{N}.jsonl` - Chunked output files, numbered from 1 (e.g., `final_result_chunk1.jsonl`, `final_result_chunk2.jsonl`)
- `${output_dir}/errors.jsonl` - Records that failed transformation (if any)

#### Resources

- **Compute:** CPU only
- **Runtime:** ~8 min (366K records, 10 chunks, 35GB output)

---

### prepare_for_sft

**File:** `projects/finance/nvfinance/stages/sft/prepare_for_sft.py`
**Binding:** workflow alias `prepare_for_sft` selects `PrepareForSFTStage`

#### Purpose

Apply prompt template formatting and prepare data for SFT training. Calls `nemo_skills.training.prepare_data` to format data with chat templates and optionally add token counts.

#### Inputs

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input_dir` | path | Chunked data from data_transformation (`${output_dir}/chunks/`) | Required |
| `output_dir` | path | Output directory for prepared data | Required |
| `prepare_data_kwargs` | dict | Data preparation arguments (see below) | `{}` |
| `add_token_counts` | bool | Add token counts to output for filtering | true |

#### prepare_data_kwargs Structure

| Parameter | Description |
|-----------|-------------|
| `ctx_args` | Context arguments for nemo_skills.training.prepare_data:<br>- `++tokenizer=/path/to/tokenizer`<br>- `++prompt_config=path/to/prompt.yaml`<br>- `++chat_template_kwargs.enable_thinking=true` |

#### Process

1. **Process chunks in parallel:** Reads all `*.jsonl` files from `input_dir`, processes each separately to avoid OOM
2. **Apply formatting:** Uses `nemo_skills.training.prepare_data` to apply:
   - Prompt template from `prompt_config`
   - Chat template formatting via tokenizer
   - Deduplication (if configured)
   - Field filtering/validation
3. **Concatenate results:** Merges all processed chunks into `${output_dir}/final_result.jsonl`
4. **Add token counts (optional):** If `add_token_counts=true` and tokenizer provided, adds `num_tokens` field

#### Outputs

- `${output_dir}/final_result.jsonl` - Formatted training data ready for SFT
- `${output_dir}/logs/` - Processing logs

#### Resources

- **Compute:** CPU only
- **Runtime:** ~34 min (366K records, 10 chunks, 69GB output)

---

### train_validation_split

**Binding:** shared stage type `data.train_validation_split` (see [Train with NVFlow's shared stages](../../../../docs/guides/training-stages.md))

#### Purpose

Split prepared data into training and validation sets with stratified sampling to maintain distribution across sets. Filters out samples exceeding maximum token length.

#### Inputs

| Parameter | Type | Description | Finance setting |
|-----------|------|-------------|-----------------|
| `input_file` | path | Prepared data from prepare_for_sft | Required |
| `output_dir` | path | Output directory for train/val splits | Required |
| `val_ratio` | float | Validation split ratio (0-1) | 0.1 |
| `stratify_by` | string | Metadata field for stratified sampling | `question_type` |
| `random_seed` | int | Random seed for reproducibility | 42 |
| `fields` | list | Fields kept in each output record | `[input, output, uuid, total_token_length]` |
| `max_token_length` | int | Filter samples exceeding this length | 49152 |
| `token_length_field` | string | Field compared with `max_token_length` | `total_token_length` |

#### Process

1. **Load data:** Read from `input_file`
2. **Filter by length:** Remove samples with `total_token_length > max_token_length`
3. **Stratified split:** Maintain distribution of `stratify_by` field across train/val sets
4. **Shuffle:** Randomize order within each set for better training
5. **Save splits:** Write `train.jsonl`, `val.jsonl`, and the `split.done` completion marker to `output_dir`

#### Outputs

- `${output_dir}/train.jsonl` - Training examples (329,620 records, ~90%)
- `${output_dir}/val.jsonl` - Validation examples (36,623 records, ~10%)
- `${output_dir}/split.done` - Row counts and SHA-256 digests of both files
- `${output_dir}/logs/` - Split statistics and logs

> **Note:** Records exceeding `max_token_length` are filtered during split (e.g., 10 records filtered from 366,253)

#### Resources

- **Compute:** CPU only
- **Runtime:** ~13 min (366K records, 34GB output)

---

### sequence_length_grouping

**File:** `projects/finance/nvfinance/stages/sft/sequence_length_grouping.py`
**Binding:** workflow alias `sequence_length_grouping` selects `SequenceLengthGroupingStage`

#### Purpose

Group training examples by total sequence length (input + output tokens) to reduce padding overhead and improve training efficiency.

#### Inputs

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input_file` | path | Training data from train_validation_split | Required |
| `output_dir` | path | Directory for grouped/bucketed data | Required |
| `tokenizer_path` | path | Tokenizer for computing lengths (optional if pre-computed) | None |
| `bucket_sizes` | list | Token length boundaries for buckets | `[16000, 24000, 32000, 48000]` |

#### Bucket Configuration

Examples are grouped into buckets based on total sequence length (`bucket_sizes: [16000, 24000, 32000, 48000]` in `sft/base.yaml`):
- **Bucket 1:** ≤ 16,000 tokens
- **Bucket 2:** 16,001 - 24,000 tokens
- **Bucket 3:** 24,001 - 32,000 tokens
- **Bucket 4:** 32,001 - 48,000 tokens
- **Overflow:** > 48,000 tokens

> **Note:** If `prepare_for_sft` added `num_tokens` field (via `add_token_counts=true`), tokenizer is not needed for grouping.

#### Benefits

- **Reduces padding:** Minimizes wasted computation on padding tokens
- **Improves efficiency:** Batches similar-length examples together
- **Speeds training:** Can improve training speed by 20-30%

#### Outputs

- `${output_dir}/train_bucket_16000.jsonl` - Examples ≤ 16K tokens (23,586 records, ~7%)
- `${output_dir}/train_bucket_24000.jsonl` - Examples 16K-24K tokens (20,787 records, ~6%)
- `${output_dir}/train_bucket_32000.jsonl` - Examples 24K-32K tokens (26,705 records, ~8%)
- `${output_dir}/train_bucket_48000.jsonl` - Examples 32K-48K tokens (258,542 records, ~78%)
- `${output_dir}/train_bucket_overflow.jsonl` - Examples > 48K tokens (0 records in production run)
- `${output_dir}/logs/` - Grouping statistics

> **Note:** Most examples (~78%) fall in the 32K-48K bucket, reflecting the long-context nature of financial reasoning tasks.

#### Resources

- **Compute:** CPU only
- **Runtime:** ~3 min (330K training records, 30GB output)

#### Note

This stage is optional and can be skipped for simpler workflows.

---

### training

**Binding:** shared stage type `nemo_rl.sft` (see [Train with NVFlow's shared stages](../../../../docs/guides/training-stages.md))

#### Purpose

Fine-tune the language model on financial Q&A data using supervised learning.

#### Inputs

Training uses NeMo-RL's config schema: pick a `preset` from Finance's `presets_file` (`project://workflows/sft/sft_presets.yaml`, set in `sft/base.yaml`), then patch it through `overrides`, which is passed to NeMo-RL nested and unflattened.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_name` | string | Model identifier, e.g. `Qwen/Qwen3-14B` |
| `hf_checkpoint_path` | path | Base model on disk, e.g. `/hf_models/Qwen/Qwen3-14B` |
| `backend` | string | `megatron` or `fsdp` |
| `total_gpus` | int | GPUs for the job; data parallelism is derived from it |
| `dependent_jobs` | int | Extra chained jobs, for training longer than one time limit |
| `preset` | string | Base config to start from, e.g. `sft-base` |
| `overrides` | dict | Nested patch over the preset, grouped into `sft`, `checkpointing`, `policy` and `data` |

Commonly overridden keys:

| Key | Description |
|-----|-------------|
| `sft.max_num_epochs` | Number of epochs |
| `sft.val_period` | Validate every N steps |
| `checkpointing.save_period` | Save every N steps |
| `checkpointing.keep_top_k` | Checkpoints to retain |
| `policy.train_global_batch_size` | Global batch size |
| `policy.train_micro_batch_size` | Per-rank micro batch |
| `policy.max_total_sequence_length` | Sequence budget |
| `policy.megatron_cfg.*` | Parallelism (`tensor_model_parallel_size`, `context_parallel_size`, …) |
| `policy.megatron_cfg.optimizer.lr` | Learning rate |

#### Training Configuration

```yaml
stages:
  training:
    model_name: Qwen/Qwen3-14B
    hf_checkpoint_path: /hf_models/Qwen/Qwen3-14B
    backend: megatron
    total_gpus: 256
    preset: "sft-base"
    overrides:
      sft:
        max_num_epochs: 3
      policy:
        train_global_batch_size: 128
        max_total_sequence_length: 49152
        megatron_cfg:
          tensor_model_parallel_size: 4
          context_parallel_size: 8
          optimizer:
            lr: 5e-6
```

#### Outputs

```
${output_dir}/model-{model}-{total_gpus}g-tp{tp}-pp{pp}-cp{cp}-seq{seq}k/
├── checkpoints/
│   ├── step_10/
│   └── step_20/
├── training-logs/
└── run_metadata_*.yaml
```

Checkpoints are step-numbered; there is no `final/` directory. The `eval` stage converts a chosen step to HuggingFace format when it needs one.

#### Resources

| Model Size | GPUs | Memory/GPU | Runtime |
|------------|------|------------|---------|
| 7-14B | 8 | 40GB | 2-3h |
| 32B | 16 | 40GB | 4-5h |
| 70B+ | 32 | 80GB | 8-12h |

#### Monitoring

```bash
# Watch training logs
tail -f ${output_dir}/model-*/training-logs/ray-*-job.log

# GPU utilization
watch -n 1 nvidia-smi

# Tensorboard (finds the event files under each run's training-logs/)
tensorboard --logdir ${output_dir}
```

---

### convert_to_messages

**File:** `projects/finance/nvfinance/stages/sft/convert_to_messages.py`
**Binding:** workflow alias `convert_to_messages` selects `ConvertToMessagesStage`

#### Purpose

Convert Qwen3 chat-templated training data to OpenAI messages format. Parses Qwen3 special tokens and extracts reasoning content into a structured format.

#### Inputs

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input_dir` | path | Directory with train.jsonl and val.jsonl from train_validation_split | Required |
| `output_dir` | path | Output directory for converted messages | Required |
| `extract_reasoning` | bool | Extract reasoning content as separate field | true |
| `sdg_model` | string | SDG model name for metadata tracking (optional) | None |

#### Process

1. **Load data:** Read `train.jsonl` and `val.jsonl` from `input_dir`
2. **Parse Qwen3 tokens:** Extract content from `<|im_start|>` and `<|im_end|>` markers
3. **Extract roles:** Identify system, user, and assistant messages
4. **Extract reasoning:** If enabled, parse reasoning from assistant response
5. **Structure output:** Convert to OpenAI messages array format
6. **Add metadata:** Include UUID and SDG model info

#### Input Format (from train_validation_split)

```json
{
  "input": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>",
  "output": "<reasoning>...\n<answer>...</answer>",
  "uuid": "abc123..."
}
```

#### Output Format (OpenAI Messages)

```json
{
  "messages": [
    {"role": "system", "content": "You are a financial reasoning assistant..."},
    {"role": "user", "content": "What was NVIDIA's revenue in 2023?"},
    {
      "role": "assistant",
      "reasoning_content": "Based on the 10-K filing...",
      "content": "NVIDIA's revenue was $26.9 billion."
    }
  ],
  "metadata": {
    "uuid": "abc123...",
    "sdg_model": "Qwen/Qwen2.5-72B-Instruct"
  }
}
```

#### Outputs

- `${output_dir}/train.jsonl` - Training data in messages format (329,620 records)
- `${output_dir}/val.jsonl` - Validation data in messages format (36,623 records)
- `${output_dir}/logs/` - Conversion logs

#### Resources

- **Compute:** CPU only
- **Runtime:** ~16 min (366K records, 34GB output)

---

### Pipeline Summary

| Stage | Purpose | Compute | Runtime (366K records) |
|-------|---------|---------|------------------------|
| data_transformation | Format conversion | CPU | ~8 min |
| prepare_for_sft | Apply prompt template | CPU | ~34 min |
| train_validation_split | Split train/val | CPU | ~13 min |
| sequence_length_grouping | [Optional] Group by length | CPU | ~3 min |
| training | Fine-tune model | 8× GPU | 2-8h |
| convert_to_messages | Convert to messages format | CPU | ~16 min |

**Total (CPU stages):** ~74 min
**Total (with training):** ~3-9 hours (depending on model size and training config)

### Common Training Parameters

All of these live under `overrides` in the training stage.

#### Learning Rate

Set at `policy.megatron_cfg.optimizer.lr`. The shipped configs use `5e-6` with `min_lr: 5e-7`, cosine decay, and warmup from `1e-7`. Treat `5e-6` as the starting point rather than scaling by model size.

#### Batch Size

`policy.train_global_batch_size` is the global batch, and `policy.train_micro_batch_size` the per-rank micro batch; gradient accumulation is derived from the two together with the data-parallel width. The production 14B config uses `128` global and `1` micro.

#### Checkpointing

- **`checkpointing.save_period`**: save every N steps — `100` for full training, `10` in the demo
- **`checkpointing.keep_top_k`**: checkpoints to retain
- **`sft.val_period`**: validate every N steps

See [SFT Workflow](../workflows/04-sft.md) for usage examples and configuration details.
