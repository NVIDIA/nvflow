# Quick Start

Get hands-on experience with the finance recipe by running a complete end-to-end pipeline with 7 demo companies.

**Total time:** ~3 hours (GPU jobs run in background)

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Prerequisites](#prerequisites)
- [Step 1: Evaluate Baseline Models (optional)](#step-1-evaluate-baseline-models-optional) — ~7 min
- [Step 2: Download SEC Filings](#step-2-download-sec-filings) — ~3 min
- [Step 3: Generate Synthetic Q&A Data](#step-3-generate-synthetic-qa-data) — ~20 min
- [Step 4: Fine-Tune Model + Evaluate (SFT)](#step-4-fine-tune-model--evaluate-sft) — ~25 min
- [Step 5: GRPO RL Training + Evaluate](#step-5-grpo-rl-training--evaluate) — ~2 hr
- [Next Steps](#next-steps)
- [Troubleshooting](#troubleshooting)

---

## Pipeline Overview

```
  ┌─────────────────┐               ┌─────────────────┐     ┌─────────────────┐
  │  1. Evaluate    │   informs     │  2. Download    │────▶│  3. SDG         │
  │  Baselines      │· · · · · · · ▶│  SEC Filings    │     │  Generate Q&A   │
  │  (optional)     │   strategy    │  (~3 min)       │     │  (~20 min)      │
  └─────────────────┘               └─────────────────┘     └────────┬────────┘
                                                                     │
                                                           ┌─────────┴─────────┐
                                                           ▼                   ▼
                                               ┌─────────────────┐  ┌─────────────────┐
                                               │  4. SFT + Eval  │  │  5. GRPO + Eval │
                                               │  (~25 min)      │  │  (~1 hr)        │
                                               └────────┬────────┘  └─────────────────┘
                                                        ·                    ▲
                                                        · · · optional · · ·
                                                          use SFT checkpoint
```

---

## Prerequisites

- Cluster access configured (see [INSTALL.md](../../../INSTALL.md))
- You are in the `nvflow` directory: `cd /path/to/nvflow`

---

<a id="step-1-evaluate-baseline-models-optional" name="step-1-evaluate-baseline-models-optional"></a>
<details open>
<summary><h2>Step 1: Evaluate Baseline Models (Optional) — ~7 min</h2></summary>

> Config: `eval/demo.yaml` | Output: `outputs/finance/demo/workflow-1-baseline-eval/`

Evaluate three baseline models on finance benchmarks to understand pre-fine-tuning performance across different model families. This step is optional — SFT and GRPO pipelines include their own baseline evaluation automatically.

| Model | Family | Parameters | Reasoning |
|-------|--------|-----------|-----------|
| Qwen3-4B | Qwen3 | 4B | Thinking mode (`enable_thinking`) |
| Gemma 3 4B IT | Gemma 3 | 4B | Default sampling (no official recommendation) |
| GPT-OSS 20B | GPT-OSS (MoE) | 21B (3.6B active) | Harmony format (`reasoning_effort=high`) |

**Preview stages:**
```bash
uv run nflow list-stages --config nvflow/recipes/finance/workflows/eval/demo.yaml
```

1. `prepare_data` — Prepare benchmark datasets (SecQUE, FinanceBench) into `outputs/finance/eval-datasets/`
2. `qwen3-4b` — Evaluate Qwen3-4B on SecQUE and FinanceBench
3. `gemma-3-4b-it` — Evaluate Gemma 3 4B IT on SecQUE and FinanceBench
4. `gpt-oss-20b` — Evaluate GPT-OSS 20B on SecQUE and FinanceBench

**Run:**

First, prepare the benchmark datasets and wait for the Slurm job to complete:
```bash
uv run nflow run prepare_data --config nvflow/recipes/finance/workflows/eval/demo.yaml
```

Verify the data is ready (expect ~565 SecQUE and ~150 FinanceBench examples):
```bash
wc -l outputs/finance/eval-datasets/secque/eval.jsonl outputs/finance/eval-datasets/financebench/eval.jsonl
```

**prepare_data output:**
```
outputs/finance/eval-datasets/   # shared across workflows
├── secque/
│   └── eval.jsonl
├── financebench/
│   └── eval.jsonl
└── logs/                              # prepare_data Slurm job logs
```

Then launch all three baseline evaluations (they run as independent Slurm jobs):
```bash
uv run nflow run qwen3-4b gemma-3-4b-it gpt-oss-20b --config nvflow/recipes/finance/workflows/eval/demo.yaml
```

**Verify:**
```bash
ls outputs/finance/demo/workflow-1-baseline-eval/qwen3-4b/eval-results/
ls outputs/finance/demo/workflow-1-baseline-eval/gemma-3-4b-it/eval-results/
ls outputs/finance/demo/workflow-1-baseline-eval/gpt-oss-20b/eval-results/
cat outputs/finance/demo/workflow-1-baseline-eval/qwen3-4b/eval-results/secque/metrics.json
```

**Evaluation output:**
```
outputs/finance/demo/workflow-1-baseline-eval/
├── qwen3-4b/
│   ├── eval-results/
│   │   ├── secque/
│   │   │   └── metrics.json
│   │   └── financebench/
│   │       └── metrics.json
│   └── logs/                              # Slurm job logs (stdout/stderr)
├── gemma-3-4b-it/
│   ├── eval-results/
│   │   ├── secque/
│   │   │   └── metrics.json
│   │   └── financebench/
│   │       └── metrics.json
│   └── logs/
└── gpt-oss-20b/
    ├── eval-results/
    │   ├── secque/
    │   │   └── metrics.json
    │   └── financebench/
    │       └── metrics.json
    └── logs/
```

**Reference results** (`judge_correct` %, pass@1 averaged over 5 seeds):

| Model | SecQUE | FinanceBench |
|-------|--------|--------------|
| Qwen3-4B | ~50% | ~79% |
| Gemma 3 4B IT | ~25% | ~58% |
| GPT-OSS 20B | ~62% | ~82% |

Qwen3-4B and GPT-OSS 20B leverage reasoning (thinking mode and Harmony format respectively), which significantly improves financial analysis accuracy. Gemma 3 4B IT, without built-in reasoning, shows the gap that SFT and GRPO fine-tuning aim to close.

> **Tip:** For production (all baselines), use `eval/baselines.yaml` which evaluates Qwen3-4B/14B/32B, GPT-OSS-120B, and Nemotron models.

</details>

---

<a id="step-2-download-sec-filings" name="step-2-download-sec-filings"></a>
<details open>
<summary><h2>Step 2: Download SEC Filings — ~3 min</h2></summary>

> Config: `download_sec_filings.yaml` | Output: `outputs/finance/demo/workflow-2-download-sec/`

Download 10-K and 10-Q filings for 7 demo companies from SEC EDGAR. The download utility is built into nvflow and uses the `edgartools` library to fetch filings and extract sections.

> **Required first:** SEC EDGAR rejects requests that don't identify the caller, and the config ships with placeholders. Edit the `demo` stage in `nvflow/recipes/finance/workflows/download_sec_filings.yaml` before running — there is no command-line override:
>
> ```yaml
> stages:
>   demo:
>     sec_identity_email: your.email@company.com
>     sec_identity_company: YourCompany
> ```
>
> See the [SEC Fair Access Policy](https://www.sec.gov/os/accessing-edgar-data).

**Preview stages:**
```bash
uv run nflow list-stages --config nvflow/recipes/finance/workflows/download_sec_filings.yaml
```

1. `demo` — Download filings for 7 companies (NVDA, AAPL, GOOG, CSCO, IBM, META, MSFT)
2. `sap-500` — Download filings for full S&P 500 (production)

**Run:**
```bash
uv run nflow run demo --config nvflow/recipes/finance/workflows/download_sec_filings.yaml
```

**Verify:**
```bash
ls outputs/finance/demo/workflow-2-download-sec/step-0-download/data/
# Expected: AAPL/ CSCO/ GOOG/ IBM/ META/ MSFT/ NVDA/

ls outputs/finance/demo/workflow-2-download-sec/step-0-download/sec_metadata.parquet
```

**Output:**
```
outputs/finance/demo/workflow-2-download-sec/
└── step-0-download/
    ├── data/
    │   ├── AAPL/                  # 10-K and 10-Q filings (2020-2024)
    │   ├── CSCO/
    │   ├── GOOG/
    │   ├── IBM/
    │   ├── META/
    │   ├── MSFT/
    │   └── NVDA/
    └── sec_metadata.parquet       # Filing metadata index
```

> **Tip:** For production (S&P 500), use `sap-500` stage instead of `demo`.

</details>

---

<a id="step-3-generate-synthetic-qa-data" name="step-3-generate-synthetic-qa-data"></a>
<details open>
<summary><h2>Step 3: Generate Synthetic Q&A Data — ~20 min</h2></summary>

> Config: `template-based-sdg-demo.yaml` | Output: `outputs/finance/demo/workflow-3-template-based-sdg/`

Generate financial Q&A pairs using the template-based SDG workflow.

> **Required first:** `create_seed_data` reaches both SEC EDGAR and HuggingFace, so before running:
>
> 1. Set your SEC identity in `nvflow/recipes/finance/workflows/sdg/template-based-sdg.yaml` (inherited by the demo config, and shipped with placeholders):
>
>    ```yaml
>    stages:
>      create_seed_data:
>        sec_identity_email: your.email@company.com
>        sec_identity_company: YourCompany
>    ```
>
> 2. Temporarily clear `HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE` and `TRANSFORMERS_OFFLINE` in your cluster config, since the seed dataset is pulled from HuggingFace. Re-enable them afterwards. See [Offline runtime](troubleshooting.md#offline-runtime).

**Preview stages:**
```bash
uv run nflow list-stages --config nvflow/recipes/finance/workflows/sdg/template-based-sdg-demo.yaml
```

1. `create_seed_data` — Create seed questions for demo companies
2. `generate_questions` — Expand seed questions using LLM
3. `map_questions_to_context` — Find relevant SEC filing sections
4. `generate_answers` — Generate answer candidates
5. `genselect_answers` — Select best answers
6. `filter_answers` — Remove unanswerable questions

**Run:**
```bash
uv run nflow run-all --config nvflow/recipes/finance/workflows/sdg/template-based-sdg-demo.yaml
```

**Verify:**
```bash
ls outputs/finance/demo/workflow-3-template-based-sdg/step-5-filter-answers/final_result.jsonl
wc -l outputs/finance/demo/workflow-3-template-based-sdg/step-5-filter-answers/final_result.jsonl
head -1 outputs/finance/demo/workflow-3-template-based-sdg/step-5-filter-answers/final_result.jsonl | jq .
```

**Output:**
```
outputs/finance/demo/workflow-3-template-based-sdg/
├── step-0-create-seed-data/
│   ├── seed_questions_demo.jsonl
│   └── company_info_demo.tsv
├── step-1-generate-questions/
│   └── final_result.jsonl
├── step-2-map-questions-to-context/
│   └── final_result.jsonl
├── step-3-generate-answers/
│   ├── output-rs0.jsonl ... output-rs2.jsonl
│   └── generation-logs/
├── step-4-genselect-answers/
│   └── final_result.jsonl
└── step-5-filter-answers/
    └── final_result.jsonl          # ~1000-1200 Q&A pairs (input for SFT/GRPO)
```

> **Tip:** For production (S&P 500), use `template-based-sdg.yaml` which uses larger models and the full filing set.

</details>

---

<a id="step-4-fine-tune-model--evaluate-sft" name="step-4-fine-tune-model--evaluate-sft"></a>
<details open>
<summary><h2>Step 4: Fine-Tune Model + Evaluate (SFT) — ~25 min</h2></summary>

> Config: `sft/qwen3_4b.yaml` | Output: `outputs/finance/demo/workflow-4-sft/qwen3_4b/`

Fine-tune Qwen3-4B on the generated Q&A data and evaluate checkpoints on finance benchmarks. Uses the Q&A pairs from Step 3 (`workflow-3-template-based-sdg/step-5-filter-answers/final_result.jsonl`) as training input.

**Preview stages:**
```bash
uv run nflow list-stages --config nvflow/recipes/finance/workflows/sft/qwen3_4b.yaml
```

1. `data_transformation` — Transform raw SDG data to standardized format (CPU)
2. `prepare_for_sft` — Apply prompt template and tokenizer formatting (CPU)
3. `train_validation_split` — Split into train/val sets (CPU)
4. `sequence_length_grouping` — Group by sequence length for efficient training (CPU)
5. `training` — Run SFT training on 1 node × 8 GPUs (GPU, ~12 min)
6. `eval` — Evaluate checkpoint on finance benchmarks (GPU, ~6 min)

**Pre-check:** If you skipped Step 1 (baseline eval), ensure benchmark datasets exist:
```bash
wc -l outputs/finance/eval-datasets/secque/eval.jsonl outputs/finance/eval-datasets/financebench/eval.jsonl
# Expected: 565 secque + 150 financebench
```

If files are missing, prepare them first and wait for the Slurm job to complete:
```bash
uv run nflow run prepare_data --config nvflow/recipes/finance/workflows/eval/demo.yaml
```

**Run:**
```bash
uv run nflow run-all --config nvflow/recipes/finance/workflows/sft/qwen3_4b.yaml
```

**Monitor training:**
```bash
squeue --me
tail -f outputs/finance/demo/workflow-4-sft/qwen3_4b/step-4-training/model-qwen3-4b-*/training-logs/ray-*-job.log
```

**Verify training:**
```bash
ls outputs/finance/demo/workflow-4-sft/qwen3_4b/step-4-training/model-qwen3-4b-8g-tp2-pp1-cp2-seq32k/checkpoints/
# Expected: step_10/ step_16/ (checkpoint at save_period=10 and final epoch)

ls outputs/finance/demo/workflow-4-sft/qwen3_4b/step-4-training/model-qwen3-4b-8g-tp2-pp1-cp2-seq32k/hf_models/
# Expected: step_10/ (HF-format model, converted during eval)
```

**Verify evaluation:**
```bash
cat outputs/finance/demo/workflow-4-sft/qwen3_4b/step-5-eval/step-10/eval-results/secque/metrics.json
cat outputs/finance/demo/workflow-4-sft/qwen3_4b/step-5-eval/step-10/eval-results/financebench/metrics.json
```

**Output:**
```
outputs/finance/demo/workflow-4-sft/qwen3_4b/
├── step-0-data-transformation/
│   ├── chunks/                           # 10 chunked JSONL files
│   ├── filtered_outliers.jsonl
│   └── logs/
├── step-1-prepare-for-sft/
│   ├── final_result.jsonl
│   └── logs/
├── step-2-train-validation-split/
│   ├── train.jsonl                       # ~1060 training examples
│   ├── val.jsonl                         # ~120 validation examples
│   └── logs/
├── step-3-sequence-length-grouping/
│   ├── train_bucket_*.jsonl              # Grouped by sequence length
│   └── logs/
├── step-4-training/
│   └── model-qwen3-4b-8g-tp2-pp1-cp2-seq32k/
│       ├── checkpoints/
│       │   ├── step_10/                  # Megatron checkpoint (save_period=10)
│       │   └── step_16/                  # Final epoch checkpoint
│       ├── hf_models/                   # HF-format models (converted during eval)
│       │   └── step_10/                 # Per-step HF checkpoint
│       └── training-logs/
└── step-5-eval/
    └── step-10/
        ├── eval-results/
        │   ├── secque/
        │   │   └── metrics.json
        │   └── financebench/
        │       └── metrics.json
        └── logs/
```

> **Note:** Demo results will vary due to limited training data (7 companies). Production training shows ~11% improvement over baseline.

> **Tip:** For production, use `sft/qwen3_14b.yaml` for Qwen3-14B or create a custom model config inheriting from `sft/base.yaml`.

</details>

---

<a id="step-5-grpo-rl-training--evaluate" name="step-5-grpo-rl-training--evaluate"></a>
<details open>
<summary><h2>Step 5: GRPO RL Training + Evaluate — ~2 hr</h2></summary>

> Config: `grpo/qwen3_4b.yaml` | Output: `outputs/finance/demo/workflow-5-grpo/qwen3_4b/`

Run GRPO reinforcement learning using LLM-as-judge rewards from NeMo-Gym environments, then evaluate checkpoints on finance benchmarks. Uses the same Q&A pairs from Step 3 as training input — GRPO does **not** depend on the SFT checkpoint.

**Two environments:** This demo trains on two independent NeMo-Gym environments, each producing a separate model:

| Environment | Reward Signal | Context |
|-------------|--------------|---------|
| `equivalence_llm_judge` | LLM judges answer equivalence to gold | Question + SEC context provided |
| `finance_sec_search` | Multi-turn tool-calling agent retrieves SEC data | Agent must find context via tools |

Each environment has its own data pipeline, rollout collection, and training. Use `-e <env>` to run a specific environment.

**Preview stages:**
```bash
uv run nflow list-stages --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml
```

1. `validate_questions` — Filter ambiguous questions for finance_sec_search (GPU, GPT-OSS-120B)
2. `data_transformation` — Clean raw SDG data to model-agnostic format (CPU)
3. `apply_prompt_template` — Apply prompt template + extract expected answer (CPU)
4. `convert_to_responses_api` — Convert to NeMo-Gym Responses API format (CPU)
5. `prepare_data` — Add agent routing fields for NeMo-Gym (CPU)
6. `prefetch_cache` — Pre-warm SEC metadata cache for finance_sec_search (CPU)
7. `collect_rollouts` — Collect rollouts, profile rewards, and filter training data (GPU)
8. `train_validation_split` — Split reward-filtered data into train/val (CPU)
9. `training` — GRPO training with NeMo-Gym environment (GPU, ~20 min per env)
10. `eval` — Evaluate checkpoint on finance benchmarks (GPU, ~6 min)

Stages 1-8 run **per-environment**: outputs are written to `{step_dir}/{env_name}/`.

Stage 7 (`collect_rollouts`) includes automatic sub-jobs:
- **Rollout** (GPU) — policy + judge vLLM servers + NeMo-Gym client, per seed (8 seeds)
- **Merge + Analyze** (CPU) — merge chunks and compute per-seed reward distributions
- **Aggregate** (CPU) — cross-seed pass@k metrics and `difficulty.jsonl`
- **Filter** (CPU) — curate training data by removing too-hard and too-easy questions

**Pre-check:** If you skipped Step 1 (baseline eval), ensure benchmark datasets exist:
```bash
wc -l outputs/finance/eval-datasets/secque/eval.jsonl outputs/finance/eval-datasets/financebench/eval.jsonl
# Expected: 565 secque + 150 financebench
```

If files are missing, prepare them first and wait for the Slurm job to complete:
```bash
uv run nflow run prepare_data --config nvflow/recipes/finance/workflows/eval/demo.yaml
```

**Run:**

We recommend running one environment at a time so you can inspect the full lifecycle before moving to the next:

**Environment 1: `equivalence_llm_judge` (~45 min)**

The simpler environment — LLM judges whether the model's answer is equivalent to the gold answer. Context is provided directly.

```bash
# Data preparation (CPU, fast)
uv run nflow run data_transformation apply_prompt_template convert_to_responses_api prepare_data \
  --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml -e equivalence_llm_judge

# Rollout collection (GPU, ~20 min) — inspect reward distributions before proceeding
uv run nflow run collect_rollouts --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml -e equivalence_llm_judge

# Post-rollout train/val split (CPU)
uv run nflow run train_validation_split --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml -e equivalence_llm_judge

# Training (FSDP v2, 16 GPUs, ~60 min)
uv run nflow run training --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml -e equivalence_llm_judge
```

**Environment 2: `finance_sec_search` (~1 hr)**

The multi-turn tool-calling environment — the agent must retrieve SEC filings via tools before answering. Includes question validation and SEC cache prefetch.

```bash
# Question validation (GPU, uses GPT-OSS-120B judge — multi-job, wait for completion)
uv run nflow run validate_questions --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml -e finance_sec_search
# Wait for all validate_questions Slurm jobs to finish (check: squeue --me)

# Data preparation + cache prefetch (CPU + GPU)
uv run nflow run data_transformation apply_prompt_template convert_to_responses_api prepare_data prefetch_cache \
  --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml -e finance_sec_search

# Rollout collection (GPU, ~30 min)
uv run nflow run collect_rollouts --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml -e finance_sec_search

# Post-rollout train/val split (CPU)
uv run nflow run train_validation_split --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml -e finance_sec_search

# Training (Megatron, 16 GPUs — uses separate config for YaRN + CP=8)
uv run nflow run training --config nvflow/recipes/finance/workflows/grpo/qwen3_4b_finsec.yaml -e finance_sec_search
```

**Evaluation (~6 min each):**
```bash
# Equivalence eval (FSDP checkpoint)
uv run nflow run eval --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml -e equivalence_llm_judge

# Finance SEC search eval (Megatron checkpoint)
uv run nflow run eval --config nvflow/recipes/finance/workflows/grpo/qwen3_4b_finsec.yaml -e finance_sec_search
```

**Monitor:**
```bash
squeue --me
# Rollout logs (one per seed per environment)
tail -f outputs/finance/demo/workflow-5-grpo/qwen3_4b/step-5-collect-rollouts/*/logs/*.log
# Training logs
tail -f outputs/finance/demo/workflow-5-grpo/qwen3_4b/step-8-training/*/grpo-qwen3-4b-*/training-logs/ray-*-job.log
```

**Verify rollouts (both environments):**
```bash
# equivalence_llm_judge
cat outputs/finance/demo/workflow-5-grpo/qwen3_4b/step-5-collect-rollouts/equivalence_llm_judge/rollout/aggregate/summary.txt
cat outputs/finance/demo/workflow-5-grpo/qwen3_4b/step-5-collect-rollouts/equivalence_llm_judge/filter/filter_report.json
wc -l outputs/finance/demo/workflow-5-grpo/qwen3_4b/step-5-collect-rollouts/equivalence_llm_judge/train.jsonl

# finance_sec_search
cat outputs/finance/demo/workflow-5-grpo/qwen3_4b/step-5-collect-rollouts/finance_sec_search/rollout/aggregate/summary.txt
cat outputs/finance/demo/workflow-5-grpo/qwen3_4b/step-5-collect-rollouts/finance_sec_search/filter/filter_report.json
wc -l outputs/finance/demo/workflow-5-grpo/qwen3_4b/step-5-collect-rollouts/finance_sec_search/train.jsonl
```

**Verify training (per-environment):**
```bash
# equivalence_llm_judge model
ls outputs/finance/demo/workflow-5-grpo/qwen3_4b/step-8-training/equivalence_llm_judge/grpo-qwen3-4b-*/checkpoints/
# Expected: step_10/ step_20/ (save_period=10, max_num_steps=20)

# finance_sec_search model
ls outputs/finance/demo/workflow-5-grpo/qwen3_4b/step-8-training/finance_sec_search/grpo-qwen3-4b-*/checkpoints/
```

**Verify evaluation** (results are per-environment, matching the training checkpoints):
```bash
cat outputs/finance/demo/workflow-5-grpo/qwen3_4b/step-9-eval/finance_sec_search/step-20/eval-results/secque/metrics.json
cat outputs/finance/demo/workflow-5-grpo/qwen3_4b/step-9-eval/finance_sec_search/step-20/eval-results/financebench/metrics.json
```

**Output:**
```
outputs/finance/demo/workflow-5-grpo/
├── step-0-validate-questions/
│   └── finance_sec_search/              # Only finance_sec_search (equivalence skips validation)
│       └── final_result.jsonl
├── step-1-data-transformation/
│   ├── equivalence_llm_judge/
│   │   └── chunks/
│   └── finance_sec_search/
│       └── chunks/
├── step-2-apply-prompt-template/
│   ├── equivalence_llm_judge/
│   └── finance_sec_search/
├── step-3-convert-to-responses-api/
│   ├── equivalence_llm_judge/
│   └── finance_sec_search/
├── step-4-prepare-data/
│   ├── equivalence_llm_judge/
│   │   ├── train.jsonl
│   │   └── agent_config_overlay.yaml
│   └── finance_sec_search/
│       ├── train.jsonl
│       └── agent_config_overlay.yaml
├── qwen3_4b/
│   ├── step-5-collect-rollouts/
│   │   ├── equivalence_llm_judge/
│   │   │   ├── rollout/aggregate/summary.txt
│   │   │   ├── filter/filter_report.json
│   │   │   └── train.jsonl
│   │   └── finance_sec_search/
│   │       ├── rollout/aggregate/summary.txt
│   │       ├── filter/filter_report.json
│   │       └── train.jsonl
│   ├── step-8-training/
│   │   ├── equivalence_llm_judge/      # Per-env model
│   │   │   └── grpo-qwen3-4b-*/
│   │   │       ├── checkpoints/
│   │   │       └── training-logs/
│   │   └── finance_sec_search/         # Per-env model
│   │       └── grpo-qwen3-4b-*/
│   │           ├── checkpoints/
│   │           └── training-logs/
│   └── step-9-eval/
│       ├── equivalence_llm_judge/      # Per-env results
│       │   └── step-20/
│       │       └── eval-results/
│       │           ├── secque/metrics.json
│       │           └── financebench/metrics.json
│       └── finance_sec_search/
│           └── step-20/
│               └── eval-results/
│                   ├── secque/metrics.json
│                   └── financebench/metrics.json
```

> **Per-environment training:** Each environment produces a separate model checkpoint. To train a single combined model on both environments, omit `-e` in the training command.

> **Note:** Demo results will vary due to limited training data (7 companies) and rollout stochasticity. The filter stage typically keeps 25-40% of questions (removing too-hard and too-easy), which provides the best RL training signal.

> **Tip:** For production (Qwen3-30B-A3B on S&P 500 data), use `grpo/qwen3_30b_a3b.yaml`.

</details>

---

## Next Steps

**Learn More:**
[Template-Based SDG](workflows/02-template-based-sdg.md) |
[SFT Workflow](workflows/04-sft.md) |
[GRPO Workflow](workflows/06-grpo.md) |
[Evaluation](workflows/05-eval.md)

**Scale to Production:**
[Download SEC (S&P 500)](workflows/01-download-sec.md#production) |
[SDG (Production)](workflows/02-template-based-sdg.md#production-run-all-stages-together) |
[SFT (Qwen3-14B)](workflows/04-sft.md) |
[GRPO (Scaling)](workflows/06-grpo.md#customization)

---

## Troubleshooting

See the comprehensive **[Finance Recipe Troubleshooting](troubleshooting.md)** guide for issues across all workflows (SEC rate limits, missing `sec_metadata.parquet`, jobs not starting, eval metrics `N/A`, `Address already in use`, offline-runtime errors, resuming interrupted runs, and more).

---

[Workflow Documentation](workflows/) | [Stage Reference](stages/) | [Main README](README.md)
