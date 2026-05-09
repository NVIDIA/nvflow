# GRPO Stages Reference

Technical reference for all 10 stages in the GRPO RL training workflow (9 active + 1 optional).

## Quick Navigation

- [data_transformation](#data_transformation)
- [apply_prompt_template](#apply_prompt_template)
- [convert_to_responses_api](#convert_to_responses_api)
- [train_validation_split](#train_validation_split)
- [prepare_data](#prepare_data)
- [collect_rollouts](#collect_rollouts)
- [compute_rewards](#compute_rewards)
- [training](#training)
- [eval](#eval)

---

## data_transformation

**File:** `nvflow/recipes/finance/stages/shared/data_transformation.py`
**Script:** `nvflow/recipes/finance/utils/shared/dataset_transformer.py`
**Registry:** `recipe="finance"`, `workflow="sft"` + `workflow="grpo"`, `stage="data_transformation"`

### Purpose

Normalize raw SDG dataset to a model-agnostic schema. Shared between SFT and GRPO workflows. Standardizes field names to:

- `problem`, `context`, `reasoning_content`, `generation`, `uuid`, `question_type`

Generates deterministic UUIDs based on problem + generation content. Computes length statistics for key fields. Supports outlier filtering by percentile.

### Inputs

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input_files` | list | One or more input JSONL file paths | Required |
| `output_file` | path | Output JSONL file path | Required |
| `output_dir` | path | Output directory (chunks written to `${output_dir}/chunks/`) | Required |
| `source_format` | string | SDG data structure: `"separated"`, `"think_tags"`, `"inline"` | `"separated"` |
| `reasoning_mode` | string | Generation formatting: `"thinking"`, `"natural"`, `"none"` | `"none"` |
| `num_chunks` | int | Split output into N chunks for parallel downstream processing | `1` |
| `filter_outliers` | bool | Enable percentile-based outlier filtering | `false` |
| `filter_config` | dict | Percentile thresholds for context/reasoning length filtering | `{}` |

### Outputs

- `${output_dir}/chunks/` — Chunked JSONL files with normalized schema

### Resources

- **Compute:** CPU only (no GPU)
- **Runtime:** ~1 min

---

## apply_prompt_template

**File:** `nvflow/recipes/finance/stages/rl/apply_prompt_template.py`
**Script:** `nvflow/recipes/finance/utils/rl/prompt_template_applier.py`
**Registry:** `recipe="finance"`, `workflow="grpo"`, `stage="apply_prompt_template"`

### Purpose

Format the `problem` field using a YAML prompt template (merging instruction + context + question) so the model receives the same structured prompt it was SFT-trained on. Optionally extracts the concise expected answer after a configurable prefix (e.g. `"Answer:"`) from `generation` for cleaner judge evaluation.

### Inputs

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input_dir` | path | Directory with chunked JSONL from data_transformation | Required |
| `output_dir` | path | Output directory | Required |
| `prompt_template` | path | Path to YAML prompt template file | Required |
| `answer_prefix` | string | Prefix to extract concise answer from generation (e.g. `"Answer:"`) | `null` |

### Outputs

- `${output_dir}/` — Processed JSONL chunks with `prompt` and `expected_answer` fields

### Resources

- **Compute:** CPU only (no GPU)
- **Runtime:** ~1 min

---

## convert_to_responses_api

**File:** `nvflow/recipes/finance/stages/rl/convert_to_responses_api.py`
**Script:** `nvflow/recipes/finance/utils/rl/responses_api_converter.py`
**Registry:** `recipe="finance"`, `workflow="grpo"`, `stage="convert_to_responses_api"`

### Purpose

Lossless conversion of prompted data to the NeMo-Gym `responses_create_params` format required by `ng_prepare_data` and downstream stages. All original fields are preserved alongside the new `responses_create_params` wrapper.

### Inputs

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input_path` | path | Input JSONL file or directory from apply_prompt_template | Required |
| `output_dir` | path | Output directory | Required |
| `container` | string | Container name (e.g., `nemo-rl`) | Required |

### Outputs

- `${output_dir}/final_result.jsonl` — Data in Responses API format

### Resources

- **Compute:** CPU only (no GPU)
- **Runtime:** ~1 min

---

## train_validation_split

**File:** `nvflow/recipes/finance/stages/shared/train_validation_split.py`
**Script:** `nvflow/recipes/finance/utils/shared/dataset_splitter.py`
**Registry:** `recipe="finance"`, `workflow="sft"` + `workflow="grpo"`, `stage="train_validation_split"`

### Purpose

Split data into training and validation sets using stratified sampling to maintain `question_type` distribution. Shared between SFT and GRPO workflows. Output files are shuffled for better training. When `keep_all_fields` is true (GRPO), all input fields including `responses_create_params` are preserved.

### Inputs

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input_file` | path | Input JSONL file (e.g. from convert_to_responses_api) | Required |
| `output_dir` | path | Output directory | Required |
| `val_ratio` | float | Fraction of data for validation | `0.1` |
| `stratify_by` | string | Field to stratify split on | `"question_type"` |
| `random_seed` | int | Random seed for reproducibility | `42` |
| `keep_all_fields` | bool | Preserve all fields (needed for Responses API data) | `false` |
| `max_token_length` | int | Maximum token length filter | `null` |

### Outputs

- `${output_dir}/train.jsonl` — Training split
- `${output_dir}/val.jsonl` — Validation split

### Resources

- **Compute:** CPU only (no GPU)
- **Runtime:** ~1 min

---

## prepare_data

**File:** `nvflow/recipes/finance/stages/rl/prepare_data.py`
**Registry:** `recipe="finance"`, `workflow="grpo"`, `stage="prepare_data"`

### Purpose

Run `ng_prepare_data` to stamp each JSONL record with an `agent_ref` field that tells NeMo-Gym which agent server to route the example to during training. Auto-generates an agent config overlay YAML from the workflow's `agents` list.

### Inputs

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `output_dir` | path | Output directory for prepared data and overlay | Required |
| `gym_path` | path | Path to NeMo-Gym inside the container | Required |
| `container` | string | Container name (e.g., `nemo-rl`) | Required |
| `nemo_gym_config_paths` | list | Base NeMo-Gym config paths | Required |
| `agents` | list | Agent definitions (name, type, datasets) | Required |
| `mode` | string | `"train_preparation"` | `"train_preparation"` |
| `should_download` | bool | Download missing datasets from HuggingFace | `false` |
| `installation_command` | string | Container setup command | Optional |

### Modes

- **`train_preparation`**: Produces `train.jsonl` + `validation.jsonl`

### Agent Configuration

Each agent maps to one NeMo-Gym environment:

```yaml
agents:
  - name: "equivalence_llm_judge_simple_agent"
    agent_type: "simple_agent"
    entrypoint: "app.py"
    resources_server:
      type: resources_servers
      name: equivalence_llm_judge
    model_server:
      type: responses_api_models
      name: policy_model
    datasets:
      - name: train
        type: train
        license: "TBD"
        jsonl_fpath: ${directories.step-3-train-validation-split}/train.jsonl
```

### Outputs

- `${output_dir}/agent_config_overlay.yaml` — Auto-generated agent config
- `${output_dir}/train.jsonl` + `validation.jsonl` — with `agent_ref` routing fields

### Resources

- **Compute:** CPU only
- **Runtime:** ~1 min

---

## collect_rollouts

**File:** `nvflow/recipes/finance/stages/rl/collect_rollouts.py`
**Registry:** `recipe="finance"`, `workflow="grpo"`, `stage="collect_rollouts"`

### Purpose

Collect model rollouts against a NeMo-Gym environment with reward scoring. Supports both small verifiability runs and SDG-scale collection (300K+ samples) through the same execution path.

### Inputs

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `output_dir` | path | Output directory | Required |
| `gym_path` | path | Path to NeMo-Gym | Required |
| `container` | string | Container name | Required |
| `input_data` | path | Prepared JSONL from prepare_data | Required |
| `agent_name` | string | Agent name (must match prepare_data) | Required |
| `model_path` | path | Model to collect rollouts from | Required |
| `nemo_gym_config_paths` | list | NeMo-Gym config paths | Required |
| `num_repeats` | int | Repeats per sample | `1` |
| `num_samples_in_parallel` | int | Concurrent requests | `4` |
| `num_chunks` | int | Split input into N parallel jobs | `1` |
| `num_random_seeds` | int | Independent runs per chunk | `1` |
| `starting_seed` | int | First seed value | `0` |
| `dependent_jobs` | int | Chain N+1 Slurm jobs per chunk via `afterany` for timeout recovery | `0` |
| `responses_create_params` | dict | Pass-through params for NeMo-Gym (e.g., `max_output_tokens`) | `{}` |
| `rerun_done` | bool | Force re-execution | `false` |
| `num_gpus` | int | GPUs per Slurm job | `8` |
| `tensor_parallel_size` | int | Policy vLLM TP | `2` |
| `max_model_len` | int | Max sequence length | `32768` |
| `vllm_base_url` | string | External vLLM URL (optional) | None |

### Judge Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `judge_model_path` | path | Local vLLM judge model |
| `judge_tensor_parallel_size` | int | Judge TP size |
| `judge_max_model_len` | int | Judge max sequence length |
| `judge_openai_base_url` | string | External OpenAI API URL |
| `judge_openai_model` | string | OpenAI model name |
| `judge_openai_api_key` | string | API key override (defaults to `$OPENAI_API_KEY`) |

### Execution Model

```
Total Slurm jobs = num_chunks × num_random_seeds × (dependent_jobs + 1)
```

Each job is self-contained:
1. Start policy vLLM server (or connect to external URL)
2. Start judge vLLM server (if local judge configured)
3. Start NeMo-Gym environment (`ng_run`)
4. Run `ng_collect_rollouts`
5. Write `.done` file on completion

After all chunk jobs complete, a merge job per seed:
1. Concatenates chunk files → `output-rs{seed}.jsonl`
2. Enriches rollouts with input metadata (uuid, question, etc.)
3. Runs reward analysis (distribution, verdicts, difficulty)
4. Deletes individual chunk files

### Post-Collection Pipeline

**Enrichment** (`enrich_rollouts.py`): NeMo-Gym environments may drop extra input fields. The enrichment step restores them by matching output rows to input rows using `expected_answer` as the join key, with prompt content for disambiguation. Fields already present in the output are never overwritten.

**Analysis** (`analyze_rollouts.py`): Produces a summary report with:
- Reward distribution (correct / incorrect / partial)
- Judge verdict breakdown
- RL signal assessment (warns if rewards are too uniform)
- Difficulty analysis (per-question `reward_std`, `reward_min`, `reward_max` when `num_repeats > 1`)

**Aggregation** (`aggregate_seeds.py`): When multiple random seeds are used, aggregates per-question statistics across seeds to compute cross-seed `reward_std` and `pass@k` metrics. Output is written to the `aggregate/` subdirectory.

**Filtering** (`filter_training_data.py`): Filters the training data based on difficulty metrics (e.g., `min_reward_std`) to remove questions that are too easy (all correct) or too hard (all incorrect), keeping only questions with meaningful reward variance for RL training.

### Resume and Robustness

- **`.done` markers**: Each chunk writes a `.done` file on successful completion. Subsequent runs (including `dependent_jobs` chains) skip completed chunks automatically.
- **`.prev` backup**: Before merging, the previous merged output is saved as `.prev` to prevent data loss if the merge job is interrupted.
- **Self-heal**: If a chunk job is interrupted mid-write, the next job in the `dependent_jobs` chain detects incomplete files and re-runs from the last checkpoint.
- **`dependent_jobs` retry**: Jobs are chained via Slurm `afterany`, so the next job runs regardless of how the previous one exited (success, timeout, or failure).

### Outputs

```
${output_dir}/
├── rollout/
│   ├── output-rs0.jsonl         # Merged, enriched rollouts
│   ├── analysis_rs0/
│   │   ├── summary.txt          # Human-readable analysis
│   │   ├── best.jsonl           # Samples with highest reward
│   │   ├── worst.jsonl          # Samples with lowest reward
│   │   ├── partial.jsonl        # Samples with 0 < reward < 1
│   │   ├── judge_failed.jsonl   # Samples with no judge evaluations
│   │   └── difficulty.jsonl     # Per-question reward_std, reward_min, reward_max
│   └── aggregate/               # Cross-seed aggregation (reward_std, pass@k)
├── scripts/                     # Generated Slurm scripts
└── logs/                        # vLLM, ng_run, and merge logs
```

### Resources

- **Compute:** GPU (inference)
- **Runtime:** ~10 min (demo, 10 samples), scales with data size and parallelism

---

## compute_rewards

**File:** `nvflow/recipes/finance/stages/rl/compute_rewards.py`
**Registry:** `recipe="finance"`, `workflow="grpo"`, `stage="compute_rewards"`

### Purpose

Re-evaluate existing rollouts from `collect_rollouts` using a different judge model. Calls the NeMo-Gym `/verify` endpoint — original model responses are preserved, only rewards and judge evaluations are replaced.

### Inputs

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `output_dir` | path | Output directory | Required |
| `gym_path` | path | Path to NeMo-Gym | Required |
| `container` | string | Container name | Required |
| `input_dir` | path | Directory with rollout files from collect_rollouts | Required |
| `nemo_gym_config_paths` | list | NeMo-Gym config paths | Required |
| `environment_name` | string | NeMo-Gym environment name | `"equivalence_llm_judge"` |
| `num_samples_in_parallel` | int | Concurrent `/verify` calls | `8` |
| `num_gpus` | int | GPUs per job (0 for OpenAI API judge) | `0` |
| `rerun_done` | bool | Force re-execution | `false` |

### Judge Configuration

Same as `collect_rollouts`, except **policy-as-judge is not supported** (there's no policy vLLM to reuse). Either `judge_model_path` or `judge_openai_base_url` must be set.

### Outputs

```
${output_dir}/
├── rewards-rs0.jsonl            # Re-judged rollouts
├── analysis_rs0/
│   ├── summary.txt              # Reward analysis
│   └── *.jsonl                  # Categorized samples
├── scripts/                     # Generated Slurm scripts
└── logs/                        # Server and job logs
```

### Resources

- **Compute:** GPU (local vLLM judge) or CPU-only (OpenAI API judge)
- **Runtime:** Depends on judge model and data size

---

## training

**File:** `nvflow/recipes/finance/stages/rl/training.py`
**Registry:** `recipe="finance"`, `workflow="grpo"`, `stage="training"`

### Purpose

Run GRPO reinforcement learning using NeMo-RL with online NeMo-Gym environment rewards.

### Inputs

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `output_dir` | path | Output directory | Required |
| `model_name` | string | HuggingFace model identifier | Required |
| `hf_checkpoint_path` | path | Model weights path | Required |
| `preset` | string | GRPO preset name from `grpo_presets.yaml` | Required |
| `backend` | string | `"fsdp"` or `"megatron"` | `"fsdp"` |
| `total_gpus` | int | Total GPUs for training (auto-split across nodes) | `16` (demo) / `64` (production) |
| `dependent_jobs` | int | Multi-job chaining for long runs | `0` |
| `training_data` | path | Training JSONL (from prepare_data) | Required |
| `validation_data` | path | Validation JSONL | Required |
| `wandb_project` | string | W&B project name | `"finance-grpo"` |
| `wandb_mode` | string | `"online"`, `"offline"`, or `"disabled"` | `"disabled"` |
| `overrides` | dict | NeMo-RL config overrides (deep-merged with preset) | `{}` |
| `nemo_rl_base_config` | path | Hydra base config for run_grpo_nemo_gym.py (uses built-in default if not set) | Optional |
| `extra_arguments` | string | Additional CLI arguments | Optional |
| `installation_command` | string | Container setup command | Optional |

### Preset System

The `grpo_presets.yaml` file provides a full NeMo-RL config (policy, GRPO algorithm, loss function, checkpointing, environment). Model configs deep-merge overrides on top:

```yaml
# In model config (e.g., qwen3_4b.yaml)
training:
  preset: "grpo-base"
  overrides:
    grpo:
      num_prompts_per_step: 4
      num_generations_per_prompt: 2
    policy:
      train_global_batch_size: 8
```

### Parallelism Validation

The stage validates parallelism before job submission:

- **Dense models**: `world_size % (TP × PP × CP) == 0`
- **MoE models (FSDP)**: `world_size % (TP × CP × EP) == 0`
- **MoE models (Megatron)**: Validates both regular and expert DP groups
- **Auto-corrections**: Disables sequence parallelism when TP=1

### Outputs

```
${output_dir}/grpo-{model}-{nodes}n-tp{tp}-cp{cp}-seq{seq}k/
├── checkpoints/
│   ├── step_1/
│   └── step_2/
├── training-logs/
└── run_metadata_*.yaml          # Full config for reproducibility
```

### Resources

| Model Size | GPUs | Runtime (demo) |
|------------|------|----------------|
| 4B | 16 (2 nodes) | ~20 min |
| 14B | 64 (8 nodes) | TBD |

---

## eval

**File:** `nvflow/recipes/finance/stages/evaluation/evaluate.py`
**Registry:** `recipe="finance"`, `workflow="sft"` + `workflow="grpo"`, `stage="eval"`

### Purpose

Evaluate GRPO training checkpoints on finance benchmarks using nemo-skills. Supports multiple checkpoint formats (`hf`, `fsdp`, `megatron`) with automatic checkpoint conversion on the cluster. Shared evaluation settings (benchmarks, judges, datasets) are loaded from `eval/base.yaml` at runtime.

Also registered for the SFT workflow, making it a shared evaluation stage across both training pipelines.

### Inputs

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `eval_output_dir` | path | Output directory for evaluation results | Required |
| `eval_steps` | list | Checkpoint steps to evaluate | `[]` |
| `checkpoint_path` | path | Path to training checkpoints | Required |
| `format` | string | Checkpoint format: `"hf"`, `"fsdp"`, `"megatron"` | `"fsdp"` (demo) / `"megatron"` (production) |
| `baseline_model` | path | Baseline model for comparison evaluation | Optional |
| `server_type` | string | Inference server type | `"vllm"` |
| `gpus` | int | GPUs for inference server | `1` |
| `inference_args` | string | Extra inference arguments | Optional |
| `server_args` | string | Extra server arguments | Optional |

### Outputs

```
${eval_output_dir}/
├── step-{N}/
│   └── eval-results/
│       └── {benchmark}/
│           └── metrics.json     # Benchmark scores
├── baseline/                    # Baseline model results (if configured)
└── logs/
```

### Resources

- **Compute:** GPU (inference)
- **Runtime:** Depends on number of benchmarks and checkpoint steps

---

## Pipeline Summary

| Stage | Purpose | Compute | Runtime (demo) |
|-------|---------|---------|----------------|
| data_transformation | Normalize SDG data to model-agnostic schema | CPU | ~1 min |
| apply_prompt_template | Apply prompt template + extract expected answer | CPU | ~1 min |
| convert_to_responses_api | Convert to NeMo-Gym Responses API format | CPU | ~1 min |
| train_validation_split | Stratified split into train/val sets | CPU | ~1 min |
| prepare_data | Add agent routing fields | CPU | ~1 min |
| collect_rollouts | Rollout collection + analysis | GPU | ~10 min |
| compute_rewards | [Optional] Re-judge rollouts | GPU/CPU | Varies |
| training | GRPO training | GPU | ~20 min |
| eval | Evaluate checkpoints on finance benchmarks | GPU | Varies |

**Total (demo):** ~35 min

## Shared Infrastructure

### Helper Module (`nvflow/lib/rl/helpers.py`)

Reusable utilities shared across all GRPO stages:

| Function | Purpose |
|----------|---------|
| `resolve_host_path` | Map container paths to host filesystem |
| `build_config_paths_str` | Assemble NeMo-Gym config_paths with overlay |
| `build_vllm_server_args` | Convert vLLM YAML config to CLI flags |
| `compute_num_gpus` | Auto-compute Slurm GPU request from per-endpoint config |
| `determine_judge_mode` | Detect judge mode from config |
| `validate_judge_config` | Validate judge-related fields |
| `build_judge_ng_run_overrides` | Generate ng_run CLI overrides for judge |
| `build_judge_nemo_gym_config` | Generate NeMo-Gym config dict fragment for judge |
| `log_judge_details` | Console output for judge config |

### Standalone Scripts

Stdlib-only Python scripts that run inside Slurm containers:

| Script | Location | Purpose |
|--------|----------|---------|
| `dataset_transformer.py` | `utils/shared/` | Normalize SDG data to model-agnostic schema |
| `prompt_template_applier.py` | `utils/rl/` | Apply prompt templates and extract expected answers |
| `responses_api_converter.py` | `utils/rl/` | Convert to NeMo-Gym Responses API format |
| `dataset_splitter.py` | `utils/shared/` | Stratified train/validation split |
| `enrich_rollouts.py` | `utils/rl/` | Restore input metadata dropped by NeMo-Gym environments |
| `analyze_rollouts.py` | `utils/rl/` | Reward distribution, judge verdicts, difficulty analysis |
| `filter_training_data.py` | `utils/rl/` | Filter training data by pass rate |
| `aggregate_seeds.py` | `utils/rl/` | Aggregate results across random seeds |

See [GRPO Workflow](../workflows/06-grpo.md) for usage examples and configuration details.
