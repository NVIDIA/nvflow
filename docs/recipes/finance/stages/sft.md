# SFT Stages Reference

Technical reference for all 6 stages in the SFT (Supervised Fine-Tuning) workflow.

> **Note:** Current production pipeline uses template-based SDG data (~300K pairs). Document-grounded SDG integration in progress.

## Quick Navigation

- [data_transformation](#data_transformation)
- [prepare_for_sft](#prepare_for_sft)
- [train_validation_split](#train_validation_split)
- [sequence_length_grouping](#sequence_length_grouping)
- [training](#training)
- [convert_to_messages](#convert_to_messages)

---

## data_transformation

**File:** `nvflow/recipes/finance/stages/shared/data_transformation.py`
**Registry:** `recipe="finance"`, `workflow="sft"`, `stage="data_transformation"` (also registered for `workflow="grpo"`)

### Purpose

Transform SDG Q&A data to standardized training format. Renames fields, generates UUIDs, moves metadata, and computes length statistics.

### Inputs

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input_files` | list | Q&A data from SDG workflows (can be single string) | Required |
| `output_file` | path | Single file output (unused when chunking) | Required |
| `output_dir` | path | Output directory (chunks saved here) | Required |
| `num_chunks` | int | Number of output chunks for memory efficiency | 1 |
| `source_format` | string | SDG data structure: "separated" \| "think_tags" \| "inline" | "separated" |
| `reasoning_mode` | string | Generation formatting: "thinking" \| "natural" \| "none" | "none" |
| `filter_outliers` | bool | Enable percentile-based outlier filtering | true |
| `filter_config` | dict | Percentile thresholds for filtering | See config |

### Source Format Options

- **"separated"**: Reasoning model SDG with separate `answer_reasoning_content` field
- **"think_tags"**: Reasoning model SDG with `<think>` tags in answer
- **"inline"**: Non-reasoning model SDG (keep as-is)

### Reasoning Mode Options

- **"thinking"**: Wrap reasoning in `<think>` tags (Qwen3 style): `<think>{reasoning}</think>\n\n{answer}`
- **"natural"**: Concatenate reasoning + answer: `{reasoning}\n\n{answer}`
- **"none"**: Answer only, discard reasoning

> **Note:** `source_format="inline"` + `reasoning_mode="thinking"` is blocked (invalid combination)

### Field Transformations

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

### Outputs

- `${output_dir}/chunks/*.jsonl` - Chunked output files (e.g., `chunk_0000.jsonl`, `chunk_0001.jsonl`)
- `${output_dir}/errors.jsonl` - Records that failed transformation (if any)

### Resources

- **Compute:** CPU only
- **Runtime:** ~8 min (366K records, 10 chunks, 35GB output)

---

## prepare_for_sft

**File:** `nvflow/recipes/finance/stages/sft/prepare_for_sft.py`
**Registry:** `recipe="finance"`, `workflow="sft"`, `stage="prepare_for_sft"`

### Purpose

Apply prompt template formatting and prepare data for SFT training. Calls `nemo_skills.training.prepare_data` to format data with chat templates and optionally add token counts.

### Inputs

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input_dir` | path | Chunked data from data_transformation (`${output_dir}/chunks/`) | Required |
| `output_dir` | path | Output directory for prepared data | Required |
| `prepare_data_kwargs` | dict | Data preparation arguments (see below) | `{}` |
| `add_token_counts` | bool | Add token counts to output for filtering | true |

### prepare_data_kwargs Structure

| Parameter | Description |
|-----------|-------------|
| `ctx_args` | Context arguments for nemo_skills.training.prepare_data:<br>- `++tokenizer=/path/to/tokenizer`<br>- `++prompt_config=path/to/prompt.yaml`<br>- `++chat_template_kwargs.enable_thinking=true` |

### Process

1. **Process chunks in parallel:** Reads all `*.jsonl` files from `input_dir`, processes each separately to avoid OOM
2. **Apply formatting:** Uses `nemo_skills.training.prepare_data` to apply:
   - Prompt template from `prompt_config`
   - Chat template formatting via tokenizer
   - Deduplication (if configured)
   - Field filtering/validation
3. **Concatenate results:** Merges all processed chunks into `${output_dir}/final_result.jsonl`
4. **Add token counts (optional):** If `add_token_counts=true` and tokenizer provided, adds `num_tokens` field

### Outputs

- `${output_dir}/final_result.jsonl` - Formatted training data ready for SFT
- `${output_dir}/logs/` - Processing logs

### Resources

- **Compute:** CPU only
- **Runtime:** ~34 min (366K records, 10 chunks, 69GB output)

---

## train_validation_split

**File:** `nvflow/recipes/finance/stages/shared/train_validation_split.py`
**Registry:** `recipe="finance"`, `workflow="sft"`, `stage="train_validation_split"` (also registered for `workflow="grpo"`)

### Purpose

Split prepared data into training and validation sets with stratified sampling to maintain distribution across sets. Filters out samples exceeding maximum token length.

### Inputs

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input_file` | path | Prepared data from prepare_for_sft | Required |
| `output_dir` | path | Output directory for train/val splits | Required |
| `val_ratio` | float | Validation split ratio (0-1) | 0.1 |
| `stratify_by` | string | Metadata field for stratified sampling | "question_type" |
| `random_seed` | int | Random seed for reproducibility | 42 |
| `max_token_length` | int | Filter samples exceeding this length (optional) | None |

### Process

1. **Load data:** Read from `input_file`
2. **Filter by length:** Remove samples with `num_tokens > max_token_length` (if specified)
3. **Stratified split:** Maintain distribution of `stratify_by` field across train/val sets
4. **Shuffle:** Randomize order within each set for better training
5. **Save splits:** Write `train.jsonl` and `val.jsonl` to `output_dir`

### Outputs

- `${output_dir}/train.jsonl` - Training examples (329,620 records, ~90%)
- `${output_dir}/val.jsonl` - Validation examples (36,623 records, ~10%)
- `${output_dir}/logs/` - Split statistics and logs

> **Note:** Records exceeding `max_token_length` are filtered during split (e.g., 10 records filtered from 366,253)

### Resources

- **Compute:** CPU only
- **Runtime:** ~13 min (366K records, 34GB output)

---

## sequence_length_grouping

**File:** `nvflow/recipes/finance/stages/sft/sequence_length_grouping.py`
**Registry:** `recipe="finance"`, `workflow="sft"`, `stage="sequence_length_grouping"`

### Purpose

Group training examples by total sequence length (input + output tokens) to reduce padding overhead and improve training efficiency.

### Inputs

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input_file` | path | Training data from train_validation_split | Required |
| `output_dir` | path | Directory for grouped/bucketed data | Required |
| `tokenizer_path` | path | Tokenizer for computing lengths (optional if pre-computed) | None |
| `bucket_sizes` | list | Token length boundaries for buckets | `[16000, 32000, 64000]` |

### Bucket Configuration

Examples are grouped into buckets based on total sequence length:
- **Bucket 1:** ≤ 16,000 tokens
- **Bucket 2:** 16,001 - 32,000 tokens
- **Bucket 3:** 32,001 - 64,000 tokens
- **Overflow:** > 64,000 tokens

> **Note:** If `prepare_for_sft` added `num_tokens` field (via `add_token_counts=true`), tokenizer is not needed for grouping.

### Benefits

- **Reduces padding:** Minimizes wasted computation on padding tokens
- **Improves efficiency:** Batches similar-length examples together
- **Speeds training:** Can improve training speed by 20-30%

### Outputs

- `${output_dir}/train_bucket_16000.jsonl` - Examples ≤ 16K tokens (23,586 records, ~7%)
- `${output_dir}/train_bucket_24000.jsonl` - Examples 16K-24K tokens (20,787 records, ~6%)
- `${output_dir}/train_bucket_32000.jsonl` - Examples 24K-32K tokens (26,705 records, ~8%)
- `${output_dir}/train_bucket_48000.jsonl` - Examples 32K-48K tokens (258,542 records, ~78%)
- `${output_dir}/train_bucket_overflow.jsonl` - Examples > 48K tokens (0 records in production run)
- `${output_dir}/logs/` - Grouping statistics

> **Note:** Most examples (~78%) fall in the 32K-48K bucket, reflecting the long-context nature of financial reasoning tasks.

### Resources

- **Compute:** CPU only
- **Runtime:** ~3 min (330K training records, 30GB output)

### Note

This stage is optional and can be skipped for simpler workflows.

---

## training

**File:** `nvflow/recipes/finance/stages/sft/training.py`
**Registry:** `recipe="finance"`, `workflow="sft"`, `stage="training"`

### Purpose

Fine-tune the language model on financial Q&A data using supervised learning.

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_name_or_path` | path | Base model to fine-tune |
| `train_file` | path | Training data |
| `val_file` | path | Validation data |
| `output_dir` | path | Directory for checkpoints and logs |
| `num_train_epochs` | int | Number of training epochs (default: 3) |
| `learning_rate` | float | Learning rate (default: 2e-5) |
| `per_device_train_batch_size` | int | Batch size per GPU (default: 4) |
| `gradient_accumulation_steps` | int | Gradient accumulation (default: 8) |
| `save_steps` | int | Checkpoint save frequency (default: 500) |
| `eval_steps` | int | Evaluation frequency (default: 500) |

### Training Configuration

```yaml
training:
  learning_rate: 2e-5
  global_batch_size: 128
  max_num_epochs: 5
```

### Outputs

```
${output_dir}/
├── checkpoints/
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   ├── checkpoint-1500/
│   └── final/              # ← Final model
├── logs/
│   └── training.log
├── runs/                   # Tensorboard logs
└── training_args.json
```

### Resources

| Model Size | GPUs | Memory/GPU | Runtime |
|------------|------|------------|---------|
| 7-14B | 8 | 40GB | 2-3h |
| 32B | 16 | 40GB | 4-5h |
| 70B+ | 32 | 80GB | 8-12h |

### Monitoring

```bash
# Watch training logs
tail -f ${output_dir}/logs/training.log

# GPU utilization
watch -n 1 nvidia-smi

# Tensorboard
tensorboard --logdir ${output_dir}/runs/
```

---

## convert_to_messages

**File:** `nvflow/recipes/finance/stages/sft/convert_to_messages.py`
**Registry:** `recipe="finance"`, `workflow="sft"`, `stage="convert_to_messages"`

### Purpose

Convert Qwen3 chat-templated training data to OpenAI messages format. Parses Qwen3 special tokens and extracts reasoning content into a structured format.

### Inputs

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input_dir` | path | Directory with train.jsonl and val.jsonl from train_validation_split | Required |
| `output_dir` | path | Output directory for converted messages | Required |
| `extract_reasoning` | bool | Extract reasoning content as separate field | true |
| `sdg_model` | string | SDG model name for metadata tracking (optional) | None |

### Process

1. **Load data:** Read `train.jsonl` and `val.jsonl` from `input_dir`
2. **Parse Qwen3 tokens:** Extract content from `<|im_start|>` and `<|im_end|>` markers
3. **Extract roles:** Identify system, user, and assistant messages
4. **Extract reasoning:** If enabled, parse reasoning from assistant response
5. **Structure output:** Convert to OpenAI messages array format
6. **Add metadata:** Include UUID and SDG model info

### Input Format (from train_validation_split)

```json
{
  "input": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>",
  "output": "<reasoning>...\n<answer>...</answer>",
  "uuid": "abc123..."
}
```

### Output Format (OpenAI Messages)

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

### Outputs

- `${output_dir}/train.jsonl` - Training data in messages format (329,620 records)
- `${output_dir}/val.jsonl` - Validation data in messages format (36,623 records)
- `${output_dir}/logs/` - Conversion logs

### Resources

- **Compute:** CPU only
- **Runtime:** ~16 min (366K records, 34GB output)

---

## Pipeline Summary

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

## Common Training Parameters

### Learning Rate

| Model Size | Recommended LR |
|------------|----------------|
| 7-14B | 2e-5 |
| 32B | 1e-5 |
| 70B+ | 5e-6 |

### Batch Size

Effective batch size = `per_device_train_batch_size` × `gradient_accumulation_steps` × `total_gpus`

Recommended: 32-128 for most models

### Checkpointing

- **save_steps**: 500-1000 (more frequent for smaller datasets)
- **save_total_limit**: 3-5 (keep only recent checkpoints to save space)
- **eval_steps**: Same as save_steps

See [SFT Workflow](../workflows/04-sft.md) for usage examples and configuration details.
