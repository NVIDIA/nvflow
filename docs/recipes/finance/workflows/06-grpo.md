# Workflow 6: GRPO Reinforcement Learning

## Purpose

Further improve fine-tuned models using Group Relative Policy Optimization (GRPO) with LLM-as-judge reward signals from NeMo-Gym.

> **Note:** GRPO training builds on an SFT checkpoint (or base model). For best results, run [SFT](04-sft.md) first.

## Quick Navigation

- [Prerequisites](#prerequisites)
- [Workflow Overview](#workflow-overview)
- [Usage](#usage)
- [Data & Results](#data--results)
- [Customization](#customization)
- [Additional Resources](#additional-resources)

---

## Prerequisites

- ✅ Base model or SFT checkpoint accessible on cluster (Qwen3-4B for demo)
- ✅ NeMo-RL container with NeMo-Gym (`nemo-rl` container)
- ✅ GPU resources (16 GPUs / 2 nodes for demo, 64 GPUs / 8 nodes for production)

## Workflow Overview

### Stages

```
┌──────────────────────────────┐
│ 0. validate_questions        │  Validate format + deduplicate (CPU)
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ 1. data_transformation       │  SDG cleanup → model-agnostic schema (CPU)
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ 2. apply_prompt_template     │  Apply prompt template + extract answer (CPU)
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ 3. convert_to_responses_api  │  Convert to NeMo-Gym Responses API format (CPU)
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ 4. prepare_data              │  Add agent_ref routing fields (CPU)
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ 5. prefetch_cache            │  Prefetch SEC filings cache (CPU/Network)
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ 6. collect_rollouts          │  Rollout collection + reward profiling + filter (GPU)
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ 7. train_validation_split    │  Split into train/val sets (CPU)
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ 8. training                  │  GRPO training with NeMo-Gym environment (GPU)
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ 9. eval                      │  Evaluate checkpoints on finance benchmarks (GPU)
└──────────────────────────────┘
```

**10 Stages (9 active + 1 optional):**
1. **validate_questions** (Step 0): Validate input questions for format compliance and deduplication
2. **data_transformation** (Step 1): Normalize raw SDG data to model-agnostic schema (shared with SFT pipeline)
3. **apply_prompt_template** (Step 2): Format the problem field using a prompt template and extract the concise expected answer
4. **convert_to_responses_api** (Step 3): Convert prompted data to NeMo-Gym Responses API format (lossless)
5. **prepare_data** (Step 4): Run `ng_prepare_data` to stamp JSONL records with agent routing fields for NeMo-Gym
6. **prefetch_cache** (Step 5): Prefetch SEC filings cache for finance_sec_search environment
7. **collect_rollouts** (Step 6): Collect model rollouts with reward scoring — includes enrichment (restore metadata), analysis (reward distribution, difficulty), and filtering
8. **train_validation_split** (Step 7): Split data into train/val sets with stratified sampling (shared with SFT pipeline)
9. **training** (Step 8): GRPO training using NeMo-RL with online NeMo-Gym environment rewards
10. **eval** (Step 9): Evaluate GRPO checkpoints on finance benchmarks

> **Optional:** **compute_rewards** — Re-judge existing rollouts with a different/stronger judge model without re-generating responses

**See [technical reference](../stages/grpo.md) for detailed stage documentation.**

### Model Configurations

| Config | Model | Environment | Backend | GPUs | Status |
|--------|-------|-------------|---------|------|--------|
| `grpo/qwen3_4b.yaml` | Qwen3-4B | equivalence_llm_judge | FSDP v2 (32K) | 16 (2 nodes) | Demo |
| `grpo/qwen3_4b_finsec.yaml` | Qwen3-4B | finance_sec_search | Megatron (TP2×CP8, 131K) | 16 (2 nodes) | Demo |
| `grpo/qwen3_30b_a3b.yaml` | Qwen3-30B-A3B (MoE) | — | Megatron | 64 (8 nodes) | Production |

## Usage

### Run Complete Workflow (Demo)

```bash
# Qwen3-4B (demo — skips compute_rewards)
uv run nflow run-all --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml
```

### Stage-by-Stage Execution

```bash
CONFIG=nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml

# Step 0: Validate + deduplicate questions (GPU)
uv run nflow run validate_questions --config $CONFIG

# Step 1: SDG cleanup → model-agnostic schema (CPU)
uv run nflow run data_transformation --config $CONFIG

# Step 2: Apply prompt template + extract expected answer (CPU)
uv run nflow run apply_prompt_template --config $CONFIG

# Step 3: Convert to NeMo-Gym Responses API format (CPU)
uv run nflow run convert_to_responses_api --config $CONFIG

# Step 4: Prepare data — add agent routing fields (CPU)
uv run nflow run prepare_data --config $CONFIG

# Prefetch SEC filings cache (CPU/Network)
uv run nflow run prefetch_cache --config $CONFIG

# Step 5: Collect rollouts (inference + reward scoring + filter) (GPU)
uv run nflow run collect_rollouts --config $CONFIG

# Step 7: Split into train/val sets (CPU)
uv run nflow run train_validation_split --config $CONFIG

# Step 8: GRPO training (GPU)
uv run nflow run training --config $CONFIG

# Step 9: Evaluate checkpoints (GPU)
uv run nflow run eval --config $CONFIG
```

### Optional: Re-judge Rollouts

To re-judge rollouts with a different judge model, enable `compute_rewards` in `pipeline_stages` and configure the judge:

```yaml
# In your model config
pipeline_stages:
  - validate_questions
  - data_transformation
  - apply_prompt_template
  - convert_to_responses_api
  - prepare_data
  - prefetch_cache
  - collect_rollouts
  - compute_rewards          # Uncomment to enable
  - train_validation_split
  - training
  - eval

stages:
  compute_rewards:
    rejudge:
      judge_vllm:
        num_gpus: 0
        openai_base_url: "https://api.openai.com/v1"
        openai_model: "gpt-4o"
```

## Data & Results

### Output Structure

```
outputs/finance/demo/workflow-5-grpo/
├── step-0-validate-questions/
│   └── {env_name}/                      # Validated + deduplicated questions
├── step-1-data-transformation/
│   ├── final_result.jsonl               # Normalized SDG data (model-agnostic schema)
│   ├── chunks/                          # Chunked input for parallel processing
│   └── logs/
├── step-2-apply-prompt-template/
│   ├── *.jsonl                          # Prompted data with extracted answers
│   └── logs/
├── step-3-convert-to-responses-api/
│   ├── final_result.jsonl               # Data in Responses API format
│   └── logs/
├── step-4-prepare-data/
│   ├── train.jsonl                      # Training data with agent_ref
│   ├── validation.jsonl                 # Validation data with agent_ref
│   └── agent_config_overlay.yaml        # Auto-generated agent config
└── qwen3_4b/                            # Model-specific outputs
    ├── step-5-collect-rollouts/
    │   ├── {env_name}/                  # Per-environment subdirectory
    │   │   ├── rollout/
    │   │   │   ├── output-rs0.jsonl     # Merged rollouts (enriched)
    │   │   │   ├── analysis_rs0/
    │   │   │   │   ├── summary.txt      # Reward distribution report
    │   │   │   │   ├── best.jsonl       # Samples with highest reward
    │   │   │   │   ├── worst.jsonl      # Samples with lowest reward
    │   │   │   │   └── difficulty.jsonl # Per-question reward_std, reward_min, reward_max
    │   │   │   └── aggregate/           # Cross-seed aggregation (reward_std, pass@k)
    │   │   ├── scripts/                 # Generated Slurm scripts
    │   │   └── logs/                    # vLLM and ng_run logs
    │   ├── train.jsonl                  # Filtered training data (from filter sub-job)
    │   └── validation.jsonl             # Filtered validation data
    ├── step-6-compute-rewards/          # (only if compute_rewards enabled)
    │   ├── train.jsonl                  # Re-judged + filtered training data
    │   ├── validation.jsonl             # Re-judged + filtered validation data
    │   └── ...
    ├── step-7-train-validation-split/
    │   ├── train.jsonl                  # Training split
    │   ├── val.jsonl                    # Validation split
    │   └── logs/
    ├── step-8-training/
    │   ├── equivalence_llm_judge/
    │   │   └── grpo-qwen3-4b-16g-tp2-cp1-seq32k/    # Demo, FSDP v2
    │   └── finance_sec_search/
    │       └── grpo-qwen3-4b-16g-tp2-cp8-seq128k/   # Demo, Megatron (YaRN 131K)
    │           ├── checkpoints/         # GRPO model checkpoints
    │           └── training-logs/
    └── step-9-eval/
        └── ...                          # Benchmark evaluation results
```

### Expected Results (Demo)

| Stage | Output | Notes |
|-------|--------|-------|
| data_transformation | `final_result.jsonl` — normalized SDG schema | CPU-only, ~1 min |
| apply_prompt_template | Prompted JSONL with extracted answers | CPU-only, ~1 min |
| convert_to_responses_api | `final_result.jsonl` in Responses API format | CPU-only, ~1 min |
| train_validation_split | `train.jsonl` + `val.jsonl` | CPU-only, ~1 min |
| prepare_data | `train.jsonl` + `validation.jsonl` with agent_ref fields | CPU-only, ~1 min |
| collect_rollouts | `output-rs0.jsonl` + analysis + filtered train/val | GPU inference, ~10 min |
| training | GRPO checkpoint | GPU training, ~20 min |
| eval | Benchmark scores | GPU inference, ~10 min |

### Validation

```bash
BASE_DIR="outputs/finance/demo/workflow-5-grpo"
MODEL_DIR="$BASE_DIR/qwen3_4b"

# Check normalized SDG data (Step 1)
head -1 $BASE_DIR/step-1-data-transformation/final_result.jsonl | jq 'keys'

# Check prompted data (Step 2)
head -1 $BASE_DIR/step-2-apply-prompt-template/*.jsonl | jq '.problem' | head -c 200

# Check Responses API conversion (Step 3)
head -1 $BASE_DIR/step-3-convert-to-responses-api/final_result.jsonl | jq 'keys'

# Check prepared data with agent_ref (Step 4)
head -1 $BASE_DIR/step-4-prepare-data/train.jsonl | jq '.agent_ref'

# Check rollout analysis (Step 5 — replace {env_name} with your environment)
cat $MODEL_DIR/step-5-collect-rollouts/{env_name}/rollout/analysis_rs0/summary.txt

# Check train/val split (Step 7)
wc -l $MODEL_DIR/step-7-train-validation-split/train.jsonl $MODEL_DIR/step-7-train-validation-split/val.jsonl

# Check training checkpoint (Step 8)
ls $MODEL_DIR/step-8-training/grpo-*/checkpoints/

# Check eval results (Step 9)
ls $MODEL_DIR/step-9-eval/
```

## Customization

### Judge Configuration

Three judge modes for `collect_rollouts`:

```yaml
stages:
  collect_rollouts:
    # Option A: Local vLLM judge (needs extra GPUs)
    judge_model_path: /hf_models/Qwen/Qwen3-30B-A3B-Instruct-2507
    judge_tensor_parallel_size: 4
    num_gpus: 8   # Must cover policy TP + judge TP

    # Option B: OpenAI API judge (no local GPU for judge)
    # judge_openai_base_url: "https://api.openai.com/v1"
    # judge_openai_model: "gpt-4o"

    # Option C: Policy-as-judge (default, testing only)
    # Neither set — judge reuses the policy model
```

### Scaling for Production

For large-scale rollout collection (300K+ samples):

```yaml
stages:
  collect_rollouts:
    num_chunks: 8               # Split input into 8 parallel Slurm jobs
    num_random_seeds: 1         # Independent runs per chunk
    num_repeats: 5              # 5 repeats per sample (for variance-based difficulty filtering)
    num_samples_in_parallel: 512  # Concurrent requests per job
    dependent_jobs: 2           # Chain 3 Slurm jobs per chunk (afterany) for timeout recovery
    rerun_done: false           # Resume from .done files
    responses_create_params:
      max_output_tokens: 32768  # Max generation length per response
```

**`dependent_jobs`**: Each chunk spawns `dependent_jobs + 1` Slurm jobs chained via `afterany` dependency. When a job hits its time limit, the next job in the chain picks up from the last `.done` checkpoint. This avoids losing progress on long-running collections.

### External vLLM Server (SDG Mode)

Use a pre-launched vLLM server (any version) instead of the self-contained launch:

```yaml
stages:
  collect_rollouts:
    vllm_base_url: "http://<host>:<port>/v1"
```

### Training Parameters

```yaml
stages:
  training:
    total_gpus: 32              # Scale up for production
    dependent_jobs: 3           # Multi-job chaining for long runs

    overrides:
      grpo:
        num_prompts_per_step: 64
        num_generations_per_prompt: 16
        max_num_steps: 1000
      policy:
        train_global_batch_size: 1024
```

### Training Backends

The demo runs two environments with different backends: `qwen3_4b.yaml` (equivalence_llm_judge) uses **FSDP v2** at 32K, while `qwen3_4b_finsec.yaml` (finance_sec_search) uses **Megatron** (TP2×CP8) for YaRN context extension to 131K. The production config (`qwen3_30b_a3b.yaml`) uses **Megatron** for the Qwen3-30B-A3B MoE model at 64 GPUs.

**Production (Megatron):**

```yaml
stages:
  training:
    backend: megatron

    overrides:
      policy:
        megatron_cfg:
          tensor_model_parallel_size: 4
          pipeline_model_parallel_size: 1
          context_parallel_size: 1
          activation_checkpointing: true
          converter_type: "Qwen2ForCausalLM"
```

## Additional Resources

### Monitoring

```bash
# Check Slurm jobs
squeue --me

# View rollout collection logs (replace {env_name} with your environment)
tail -f $MODEL_DIR/step-5-collect-rollouts/{env_name}/logs/ng_run_rs0_chunk0.log

# View training logs
tail -f $MODEL_DIR/step-8-training/grpo-*/training-logs/*.log
```

### Troubleshooting

**vLLM server fails to start:**
- Check GPU availability: `sinfo -p interactive`
- Review vLLM logs: `cat $MODEL_DIR/step-5-collect-rollouts/{env_name}/logs/vllm_server_rs0_chunk0.log`
- Ensure `tensor_parallel_size` doesn't exceed available GPUs

**All rewards are 0.0 or 1.0:**
- Check the rollout analysis: `cat $MODEL_DIR/step-5-collect-rollouts/{env_name}/rollout/analysis_rs0/summary.txt`
- Policy-as-judge produces circular evaluation — use a separate judge for meaningful rewards
- Review judge logs: `cat $MODEL_DIR/step-5-collect-rollouts/{env_name}/logs/ng_run_rs0_chunk0.log`

**Rollouts missing metadata (uuid, question):**
- The enrichment step automatically restores fields dropped by NeMo-Gym environments
- Check enrichment output in the merge job logs

### Next Steps

After GRPO training:

- **[Evaluate GRPO Model](05-eval.md)** - Compare GRPO checkpoint with SFT baseline
- **Scale Up** - Increase data, nodes, and training steps for production
- **Judge Iteration** - Try different judge models via `compute_rewards`

### Technical Details

This workflow uses **NeMo-RL** for GRPO training and **NeMo-Gym** for environment-based reward computation.

**Supported NeMo-Gym environments:**

| Environment | Reward Mode | Description |
|-------------|-------------|-------------|
| `equivalence_llm_judge` | Binary (0.0 / 1.0) | Semantic equivalence scoring via an LLM judge |
| `finance_sec_search` | Scaled (0.0 / 0.5 / 1.0) | Real SEC filing retrieval + LLM judge with partial credit |
| `mcqa` | Binary (0.0 / 1.0) | Multiple-choice QA with exact-match scoring (production only, not in demo) |

The `finance_sec_search` environment requires prefetching SEC filings cache to a shared mounted path (see [INSTALL.md](../../../../INSTALL.md#prefetch-sec-filings-cache-for-finance_sec_search)).

For comprehensive stage-by-stage documentation:
- **[GRPO Stages Reference](../stages/grpo.md)**
