# Workflow 4: Supervised Fine-Tuning (SFT)

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
│ 5. convert_to_messages  │  Conversion: Convert to OpenAI messages format
└─────────────────────────┘
```

> **Note:** All 6 stages run in the production `qwen3_14b.yaml` configuration. Some stages (`sequence_length_grouping`, `convert_to_messages`) may be optional for custom configurations.

**6 Stages:**
1. **data_transformation** (Step 0): Convert Q&A format to training format
2. **prepare_for_sft** (Step 1): Prepare data for SFT (formatting, filtering)
3. **train_validation_split** (Step 2): Split into train/validation sets
4. **sequence_length_grouping** (Step 3): Group by sequence length for efficiency
5. **training** (Step 4): Fine-tune the model
6. **convert_to_messages** (Step 5): Convert to message format for chat interfaces

**See [technical reference](../stages/sft.md) for detailed stage documentation.**

### Model Configurations

Pre-configured workflows for supported models:

| Config | Model | Architecture | Params | GPUs | Cluster Config | Status |
|--------|-------|-------------|--------|------|----------------|--------|
| `sft/qwen3_14b.yaml` | Qwen3-14B | Dense | 14B | 256 (32 nodes) | `my_cluster` | Production |
| `sft/qwen3_4b.yaml` | Qwen3-4B | Dense | 4B | 8 (1 node) | `my_cluster` | Demo |
| `sft/gemma3_1b_it.yaml` | Gemma3-1B-IT | Dense | 1B | 8 (1 node) | `my_cluster` | Example / Demo |
| `sft/gemma3_4b_it.yaml` | Gemma3-4B-IT | Dense | 4B | 8 (1 node) | `my_cluster` | Example / Demo |
| `sft/gemma3_27b_it.yaml` | Gemma3-27B-IT | Dense | 27B | 16 (2 nodes) | `my_cluster` | Example / Demo |
| `sft/nemotron-3-nano.yaml` | Nemotron-3-Nano-30B | MoE Hybrid | 30B (3.5B active) | 256 (32 nodes) | `my_cluster_nemotron` | Production |
| `sft/base.yaml` | Base template | - | - | - | `my_cluster` | - |

> **Note:** Nemotron-3-Nano requires a dedicated cluster config (`my_cluster_nemotron`) with a NeMo-RL overlay mount for MoE support. See [Nemotron-3-Nano Special Requirements](#nemotron-3-nano-special-requirements) below.
>
> **Gemma3 configs** (1B, 4B, 27B) are **example / demo** only: they demonstrate Gemma SFT with FSDP2 and validated max sequence lengths. They are not production workflows.

## Usage

> **📋 Before You Start:** Make sure to update the `raw_train_data` path in `base.yaml` or your model config to point to your SDG output directory. See [Input Data](#input-data) section below for details.

### Run Complete Workflow

```bash
# Qwen3-14B (production)
uv run nflow run-all --config nvflow/recipes/finance/workflows/sft/qwen3_14b.yaml

# Qwen3-4B (demo)
uv run nflow run-all --config nvflow/recipes/finance/workflows/sft/qwen3_4b.yaml

# Gemma3-1B-IT (example / demo)
uv run nflow run-all --config nvflow/recipes/finance/workflows/sft/gemma3_1b_it.yaml

# Gemma3-4B-IT (example / demo)
uv run nflow run-all --config nvflow/recipes/finance/workflows/sft/gemma3_4b_it.yaml

# Gemma3-27B-IT (example / demo, 2 nodes)
uv run nflow run-all --config nvflow/recipes/finance/workflows/sft/gemma3_27b_it.yaml

# Nemotron-3-Nano-30B (production) — requires my_cluster_nemotron
uv run nflow run-all --config nvflow/recipes/finance/workflows/sft/nemotron-3-nano.yaml
```

### Stage-by-Stage Execution

Replace `CONFIG` with the path to your model config (e.g., `nvflow/recipes/finance/workflows/sft/qwen3_14b.yaml`):

```bash
# Step 0: Transform data
uv run nflow run data_transformation --config CONFIG

# Step 1: Prepare for SFT
uv run nflow run prepare_for_sft --config CONFIG

# Step 2: Split train/val
uv run nflow run train_validation_split --config CONFIG

# Step 3: Group by length
uv run nflow run sequence_length_grouping --config CONFIG

# Step 4: Train
uv run nflow run training --config CONFIG

# Step 5: Convert to messages (Qwen3 configs only)
uv run nflow run convert_to_messages --config CONFIG
```

## Data & Results

### Input Data

#### Preparing SDG Output for SFT

The SFT workflow expects input data at a specific location. The `base.yaml` config defines:

```yaml
directories:
  raw_train_data: /workspace/outputs/finance/sap-500/workflow-3-template-based-sdg/step-5-filter-answers
```

The `data_transformation` stage reads from: `${raw_train_data}/final_result.jsonl`

#### Setting Up Input Data

The default config points to the template-based SDG output. You can override if needed:

```yaml
# In your SFT config or command line
directories:
  raw_train_data: /workspace/outputs/finance/sap-500/workflow-3-template-based-sdg/step-5-filter-answers
```

The input file should be the `final_result.jsonl` from the template-based SDG workflow (step-5-filter-answers).

### Output Structure

```
outputs/finance/sap-500/workflow-4-sft/qwen3_14b/
├── step-0-data-transformation/
│   └── chunks/
│       └── final_result_chunk*.jsonl
├── step-1-prepare-for-sft/
│   └── final_result.jsonl           # ~366K samples
├── step-2-train-validation-split/
│   ├── train.jsonl                  # ~330K samples
│   └── val.jsonl                    # ~37K samples
├── step-3-sequence-length-grouping/  # [Optional]
│   └── grouped_data/
├── step-4-training/
│   └── model-qwen3-14b-256g-tp4-pp1-cp8-seq48k/
│       ├── checkpoints/
│       │   ├── checkpoint-100/
│       │   ├── checkpoint-200/
│       │   └── final/               # ← Final model
│       ├── training-logs/
│       └── hf_models/
└── step-5-convert-to-messages/       # [Optional]
    └── messages_data.jsonl
```

### Expected Results

| Model | Training Samples | Validation Samples | Training Duration | Final Checkpoint |
|-------|------------------|-------------------|-------------------|------------------|
| Qwen3-14B (256 GPUs) | ~330K | ~37K | 12-16 hours (4 jobs) | `outputs/finance/sap-500/workflow-4-sft/qwen3_14b/step-4-training/.../checkpoints/final/` |

**Note:** Training runs as 4 sequential jobs (~4 hours each max). Total time depends on cluster availability and whether the final job finishes early.

### Validation

```bash
# Check training outputs (from nvflow directory)
OUTPUT_DIR="outputs/finance/sap-500/workflow-4-sft/qwen3_14b"

# Verify train/val split
wc -l $OUTPUT_DIR/step-2-train-validation-split/train.jsonl  # Should be ~330K
wc -l $OUTPUT_DIR/step-2-train-validation-split/val.jsonl    # Should be ~37K

# Check prepared data
wc -l $OUTPUT_DIR/step-1-prepare-for-sft/final_result.jsonl  # Should be ~366K

# List checkpoints
ls $OUTPUT_DIR/step-4-training/model-qwen3-14b-256g-tp4-pp1-cp8-seq48k/checkpoints/

# Check final model
ls $OUTPUT_DIR/step-4-training/model-qwen3-14b-256g-tp4-pp1-cp8-seq48k/checkpoints/final/
# Should contain: config.json, model weights, tokenizer files
```

## Customization

The production configuration (`sft/qwen3_14b.yaml`) serves as a reference implementation. You can customize it for:
- **Different models** - Train models other than Qwen3-14B
- **Resource constraints** - Adjust for available GPU/node counts
- **Training behavior** - Modify epochs, checkpointing, validation frequency

All customization follows the same pattern: inherit from `base.yaml` and override specific settings.

### Training a Different Model

To train a model other than Qwen3-14B, create a new config following the `qwen3_14b.yaml` pattern:

```yaml
# my-custom-model.yaml
_base_: base.yaml

# Model-specific output directory
base_output_dir: /workspace/outputs/finance/sap-500/workflow-4-sft/my_model

stages:
  # Step 1: Update tokenizer for your model
  prepare_for_sft:
    prepare_data_kwargs:
      ctx_args: >-
        ++tokenizer=/hf_models/MyOrg/MyModel
        ++prompt_config=nvflow/recipes/finance/prompts/secque_template.yaml

  # Step 3: Use same tokenizer for sequence grouping
  sequence_length_grouping:
    tokenizer_path: /hf_models/MyOrg/MyModel

  # Step 4: Training configuration
  training:
    model_name: MyOrg/MyModel
    hf_checkpoint_path: /hf_models/MyOrg/MyModel
    total_gpus: 64                  # Adjust for your model size

    # Use existing preset or create custom
    preset: "qwen-3-14b"            # Or your custom preset

    overrides:
      training:
        max_num_epochs: 3
        warmup_steps: 2000
        global_batch_size: 128

      # Adjust parallelism for your model size
      parallelism:
        tensor_model_parallel_size: 2
        pipeline_model_parallel_size: 1
        context_parallel_size: 4

      checkpointing:
        save_period: 100
```

**Key configuration points:**
1. Update `tokenizer` path in `prepare_for_sft` (Step 1) and `sequence_length_grouping` (Step 3)
2. Set `model_name` and `hf_checkpoint_path` to your model in `training` (Step 4)
3. Adjust `total_gpus` based on model size and available resources
4. Tune `parallelism` settings for your model architecture

### Adjusting Training Parameters

You can modify training behavior without changing the model. Common adjustments:

```yaml
training:
  # === Resource Configuration ===
  total_gpus: 128                   # Reduce for smaller runs (vs 256 in production)
  dependent_jobs: 1                 # Fewer job splits (vs 3 in production)

  overrides:
    training:
      # Training duration
      max_num_epochs: 5             # More epochs for smaller datasets
      warmup_steps: 1000            # Adjust warmup period

      # Batch size
      global_batch_size: 64         # Smaller for limited resources

      # Validation and logging
      val_period: 100               # Validate less frequently

    checkpointing:
      save_period: 200              # Save checkpoints less frequently

  # Experiment tracking
  wandb_mode: offline               # Use offline mode, or 'disabled' to turn off
```

**Parameter guide:**
- **`total_gpus`**: Total GPUs for training (auto-split across nodes using `gpus_per_node` from cluster config)
- **`dependent_jobs`**: Training split count (higher = more job restarts, lower max time per job)
- **`max_num_epochs`**: Total training passes through the dataset
- **`warmup_steps`**: Learning rate warmup (typically 5-10% of total steps)
- **`global_batch_size`**: Effective batch size across all GPUs
- **`val_period`**: Steps between validation runs
- **`save_period`**: Steps between checkpoint saves

## Model-Specific Notes

### Nemotron-3-Nano Special Requirements

Nemotron-3-Nano-30B is a **Mixture-of-Experts (MoE) Hybrid** model with a unique architecture (23 Mamba-2 + MoE layers, 6 Attention layers). It requires special infrastructure setup:

#### 1. Clone NeMo-RL (nano-v3 branch)

Nemotron-3-Nano requires the `nano-v3` branch of NeMo-RL, which has MoE-aware training support not yet available in the main branch. See the [official guide](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/nemotron-3-nano.md) for reference.

```bash
# Clone the nano-v3 branch (MoE support for Nemotron-3-Nano)
git clone -b nano-v3 https://github.com/NVIDIA-NeMo/RL.git

# Initialize submodules
cd RL
git submodule update --init --recursive
```

#### 2. Dedicated Cluster Config

Nemotron-3-Nano uses `cluster: my_cluster_nemotron` (set in `nemotron-3-nano.yaml`), which is identical to `my_cluster.yaml` **except** for one additional mount that overlays the container's built-in `/opt/NeMo-RL` with the cloned `nano-v3` branch:

```yaml
# my_cluster_nemotron.yaml — extra mount (vs my_cluster.yaml)
mounts:
  # ... standard mounts ...
  - /path/to/RL:/opt/NeMo-RL  # NeMo-RL overlay for MoE support
```

Update the mount path to point to wherever you cloned the `RL` repository in step 1.

**Why is this needed?**
- The `nano-v3` branch provides MoE-aware training support (expert parallelism, MoE routing, etc.)
- This support is **not yet merged into the main NeMo-RL branch**
- The mount overlays the container's built-in `/opt/NeMo-RL` with the local `nano-v3` code
- Standard dense models (Qwen3, Gemma3) use the container's built-in NeMo-RL and do not need this mount

#### 3. Verify the Setup

After launching a training job, confirm the overlay is working:

```bash
# Inside the container (via srun or job logs), check that the mounted code is present
ls /opt/NeMo-RL/examples/nemo_gym/  # Should contain grpo_nanov3.yaml
```

**When can this be retired?**
Once MoE support is merged into the main NeMo-RL branch and the container image is updated, `my_cluster_nemotron.yaml` can be retired and all models can use `my_cluster.yaml`.

#### Key Architecture Differences

| Property | Nemotron-3-Nano | Qwen3 / Gemma3 (Dense) |
|----------|-----------------|------------------------|
| Architecture | MoE Hybrid (Mamba-2 + MoE) | Dense Transformer |
| Total / Active params | 30B / 3.5B | Full params active |
| Expert Parallelism (EP) | 8 | N/A (ignored) |
| Expert Tensor Parallelism (ETP) | 1 | N/A (ignored) |
| Tensor Parallelism (TP) | 4 | 1–8 (model-dependent; Gemma3: 1/4/8 for 1B/4B/27B) |
| Context Parallelism (CP) | 8 | 1–8 (model-dependent; Gemma3 uses 1, no sequence packing) |
| Cluster config | `my_cluster_nemotron` | `my_cluster` |
| Sequence packing | Required (CP > 1) | Optional |

### Qwen3 Models

- **Qwen3-14B** (production): 32 nodes, TP=4, CP=8, 48K sequence length
- **Qwen3-4B** (demo): 1 node, TP=2, CP=2, 32K sequence length
- Both include `convert_to_messages` stage for OpenAI messages format conversion
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
- **DTensor:** Configs use v1 worker (`_v2: false`) for correct Gemma3 sharding; activation checkpointing enabled.
- Does **not** include `convert_to_messages` (Qwen3-specific chat template parser).
- Chat template: `<start_of_turn>role\ncontent<end_of_turn>`

## Additional Resources

### Monitoring Training

#### Weights & Biases (wandb)

Training metrics are tracked via wandb dashboard:

```yaml
# Configured in workflow (see base.yaml)
wandb_project: finance-training
wandb_mode: online  # online | offline | disabled
```

**If you do not have WandB credentials, please set the mode to `disabled`**

```yaml
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
ls ${OUTPUT_DIR}/step-4-training/model-qwen3-14b-256g-tp4-pp1-cp8-seq48k/checkpoints/ | grep checkpoint | wc -l

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

This workflow uses **NeMo-RL** for distributed training. Backend is model-dependent: **Megatron** for Qwen3 and Nemotron-3-Nano; **FSDP2 (DTensor)** for Gemma3 (1B, 4B, 27B), which avoids Megatron’s known Gemma3 issues (logprob OOM, activation checkpointing).

For comprehensive stage-by-stage documentation:
- **[SFT Stages Reference](../stages/sft.md)**

For NeMo training details:
- Check NeMo documentation for advanced training options
- See `nvflow/recipes/finance/stages/sft/` and `nvflow/recipes/finance/stages/shared/` for stage implementations
