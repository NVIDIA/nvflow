# Finance project quick start

Research scientists often assemble data preparation, generation, training, and
evaluation as separate scripts. This walkthrough shows how the NVFlow finance
project composes the same research process from reusable workflows, using seven
demo companies.

**Total time:** ~6 hours (GPU jobs run in background)

## Contents

- [Workflow Overview](#workflow-overview)
- [Prerequisites](#prerequisites)
- [Understand the quickstart recipe (optional)](#understand-the-quickstart-recipe-optional)
- [Run the workflows](#run-the-workflows)
  - [1. Prepare Evaluation Data and Evaluate Baselines](#step-1-prepare-evaluation-data-and-evaluate-baselines) — ~45 min
  - [2. Download SEC Filings](#step-2-download-sec-filings) — ~10 min
  - [3. Generate Synthetic Q&A Data](#step-3-generate-synthetic-qa-data) — ~45 min
  - [4. Fine-Tune Model + Evaluate (SFT)](#step-4-fine-tune-model--evaluate-sft) — ~1 hr
  - [5. GRPO RL Training + Evaluate](#step-5-grpo-rl-training--evaluate) — ~5 hr
- [Next Steps](#next-steps)
- [Troubleshooting](#troubleshooting)

---

## Workflow Overview

```
  ┌─────────────────┐               ┌─────────────────┐     ┌─────────────────┐
  │  1. Eval data + │   informs     │  2. Download    │────▶│  3. SDG         │
  │  Baselines      │· · · · · · · ▶│  SEC Filings    │     │  Generate Q&A   │
  │  (~45 min)      │   strategy    │  (~10 min)      │     │  (~45 min)      │
  └─────────────────┘               └─────────────────┘     └────────┬────────┘
                                                                     │
                                                           ┌─────────┴─────────┐
                                                           ▼                   ▼
                                               ┌─────────────────┐  ┌─────────────────┐
                                               │  4. SFT + Eval  │  │  5. GRPO + Eval │
                                               │  (~1 hr)        │  │  (~5 hr)        │
                                               └────────┬────────┘  └─────────────────┘
                                                        ·                    ▲
                                                        · · · optional · · ·
                                                          use SFT checkpoint
```

Steps 4 and 5 also evaluate their checkpoints on the benchmark data that Step 1's
`eval_data` stage prepares.

`project.yaml` records this connection as the `finance:quickstart` recipe
DAG. The recipe can be validated and planned as a graph, but it is not submitted
as one scheduler operation yet. Each workflow selector below is still planned
and submitted independently.

---

## Prerequisites

- [INSTALL.md](../../../INSTALL.md) completed, and a target from
  [Slurm execution](../../../docs/slurm/README.md) or
  [Ray execution](../../../docs/ray/README.md#set-up-a-ray-target); on Ray,
  pass your Ray target instead of `my_cluster`.
- You are at the NVFlow checkout root or a standalone finance-project root.

### What each step needs

| Step | Models under `/hf_models` | Images (roles) | Largest job | Also needs |
|---|---|---|---|---|
| 1. Baseline eval (QA) | Qwen3-4B, gpt-oss-20b, gemma-3-4b-it; gpt-oss-120b judge | `nemo-gym`, `vllm` | 8 GPUs: 4 for the model, 4 for the judge | Network (Hugging Face and GitHub) for the first `eval_data` run; the optional agentic run also needs sec.gov and Step 2's SEC cache |
| 2. Download SEC filings | None | `nemo-skills`, `nemo-gym` | CPU only | Network; `SEC_IDENTITY_EMAIL` and `SEC_IDENTITY_COMPANY` on the launcher |
| 3. Synthetic Q&A | gpt-oss-20b | `nemo-skills`, `vllm` | 4 GPUs | Network; the SEC identity |
| 4. SFT + eval | Qwen3-4B; gpt-oss-120b judge | `nemo-rl`, `nemo-gym`, `vllm`, `nemo-skills` | 8 GPUs on 1 node | Step 1's `eval_data` output |
| 5. GRPO + eval | Qwen3-4B; gpt-oss-120b judge | `nemo-rl`, `nemo-gym`, `vllm`, `nemo-skills` | 16 GPUs on 2 nodes (32 on 4 nodes for `finance_sec_search`), plus a 4-GPU judge alongside | Step 1's `eval_data` output |

Download the models with [INSTALL.md → Download Models](../../../INSTALL.md#download-models).
On Ray, route every listed role to a cluster with that capacity, and keep the
control cluster available for every step.

---

## Understand the quickstart recipe (optional)

> These commands are for orientation only. They do not run or submit workflows.
> Skip to [Run the workflows](#run-the-workflows) when you are ready to begin.

For the quickest overview of the connected demo workflows:

```bash
uv run nvflow list-workflows --recipe finance:quickstart
```

The output shows the complete workflow definitions in dependency order,
including their types, stage counts, and optional status.

<details>
<summary>Additional validation and inspection commands</summary>

Validate recipe references or inspect dependency edges and provenance:

```bash
uv run nvflow validate --recipe finance:quickstart
uv run nvflow plan-recipe finance:quickstart
```

Inspect one workflow without submitting it:

```bash
uv run nvflow validate --workflow finance:eval/demo
uv run nvflow list-stages --workflow finance:eval/demo
uv run nvflow plan --workflow finance:eval/demo --cluster my_cluster
```

Replace `my_cluster` and the selector as needed. Use
`show-config -w <selector>` to inspect resolved configuration. See the
[CLI guide](../../../docs/platform/cli-and-python-api.md) for static versus live
planning.

</details>

### Compare with production recipes

```bash
uv run nvflow list-recipes --project finance
uv run nvflow list-workflows --project finance
```

- `finance:quickstart` — template-based demo SDG for Qwen3-4B SFT and GRPO.
- `finance:template-based-sft` — template-based production SDG for single-turn
  Qwen3-14B and Nemotron-3-Nano SFT.
- `finance:document-grounded-local-search-grpo` — document-grounded production
  SDG for multi-turn Nemotron-3-Nano GRPO with `finance_sec_search`, served from
  the local full-text index.

These are the three documented recipes. `project.yaml` also defines
`finance:vals-no-web` and `finance:vals11`, which `list-recipes` shows.

Qwen3-4B is demo-only. The production recipes include the required S&P 500
download and SDG workflows.

---

## Run the workflows

For the shortest end-to-end path, run Step 1's `eval_data` stage,
[Download SEC Filings](#step-2-download-sec-filings), then
[Generate Synthetic Q&A Data](#step-3-generate-synthetic-qa-data), and choose
[SFT](#step-4-fine-tune-model--evaluate-sft),
[GRPO](#step-5-grpo-rl-training--evaluate), or both. Run the
[baseline evaluations](#step-1-prepare-evaluation-data-and-evaluate-baselines)
only when you want a pre-training comparison.

Each workflow is planned and submitted independently. None of the optional
inspection commands above is a prerequisite for running a workflow.

Output paths below are relative to the host directory your target mounts at
`/workspace`. The **Verify** commands read it through `$WORKSPACE`, so set it
once; on Ray, use a host that mounts it:

```bash
export WORKSPACE=<host directory mounted at /workspace>
```

<a id="step-1-prepare-evaluation-data-and-evaluate-baselines" name="step-1-prepare-evaluation-data-and-evaluate-baselines"></a>
<details>
<summary><h2>1. Prepare Evaluation Data and Evaluate Baselines — ~45 min</h2></summary>

> - **Selector:** `finance:eval/demo`
> - **Config:** `eval/demo.yaml`
> - **Output:** `outputs/finance/demo/workflow-1-baseline-eval/`

Prepare the finance benchmarks, then evaluate three baseline models on them to
understand pre-fine-tuning performance across different model families. The
`eval_data` stage is required: Steps 4 and 5 evaluate their checkpoints on the
data it prepares. The baseline evaluations are optional; run them when you want a
pre-training comparison.

| Model | Family | Parameters | Reasoning |
|-------|--------|-----------|-----------|
| Qwen3-4B | Qwen3 | 4B | Thinking mode (`enable_thinking`) |
| Gemma 3 4B IT | Gemma 3 | 4B | Default sampling (no official recommendation) |
| GPT-OSS 20B | GPT-OSS (MoE) | 21B (3.6B active) | Harmony format (`reasoning_effort=high`) |

This walkthrough uses the two QA benchmarks, selected with `-e`, because they
grade with a locally served judge and need no API keys. `demo.yaml` can also run
the agentic benchmarks — see [Agentic benchmarks](#agentic-demo) at the end of
this workflow section.

**Inspect available stages:**
```bash
uv run nvflow list-stages --workflow finance:eval/demo
```

**Workflow stages:**

1. `eval_data` — Prepare all three benchmark datasets into `step-1-eval-data/`
2. `qwen3-4b` — Evaluate Qwen3-4B on the selected benchmarks
3. `gemma-3-4b-it` — Evaluate Gemma 3 4B IT (QA benchmarks only — no tool-call parser)
4. `gpt-oss-20b` — Evaluate GPT-OSS 20B on the selected benchmarks

The config also provides optional `rejudge` settings, but does not include that
stage in the default pipeline. Set its source/output paths and invoke it
explicitly to apply a different judge to existing eval rollouts.

**Run:**

First, prepare the benchmark datasets and wait for the job to complete. The
first run downloads them from Hugging Face and GitHub, so it needs network:
```bash
uv run nvflow run eval_data --workflow finance:eval/demo --cluster my_cluster
uv run nvflow status --cluster my_cluster   # wait until the job finishes
```

The stage prepares the two QA benchmarks and the agentic Vals set, `fabv1`.
Verify the data is ready:
```bash
wc -l $WORKSPACE/outputs/finance/demo/workflow-1-baseline-eval/step-1-eval-data/*/eval.jsonl
# Expected: 565 secque, 150 financebench, 50 fabv1_noweb
```

**eval_data output:**
```
outputs/finance/demo/workflow-1-baseline-eval/step-1-eval-data/   # also read by Steps 4 and 5
├── secque/
│   └── eval.jsonl
├── financebench/
│   └── eval.jsonl
├── fabv1_noweb/                       # name tracks fabv1_variant
│   └── eval.jsonl
└── logs/                              # eval_data job logs
```

Then launch all three baseline evaluations (they run as independent jobs):
```bash
uv run nvflow run qwen3-4b gemma-3-4b-it gpt-oss-20b \
  --workflow finance:eval/demo \
  --cluster my_cluster \
  -e secque -e financebench
```

**Monitor:**
```bash
uv run nvflow status --cluster my_cluster
```

**Verify:**
```bash
ls $WORKSPACE/outputs/finance/demo/workflow-1-baseline-eval/qwen3-4b/collect-rollouts/secque/
cat $WORKSPACE/outputs/finance/demo/workflow-1-baseline-eval/qwen3-4b/collect-rollouts/secque/jobs/job-1-analyze-rollouts/analysis/evaluation/summary.txt
```

**Evaluation output** — one subtree per (model, benchmark) pair:
```
outputs/finance/demo/
├── qwen3-4b/
│   └── collect-rollouts/
│       ├── secque/
│       │   └── jobs/
│       │       ├── job-0-collect-rollouts/
│       │       └── job-1-analyze-rollouts/
│       │           ├── outputs/            # native Gym bundle
│       │           ├── analysis/evaluation/
│       │           │   ├── metrics.json    # finance leaderboard
│       │           │   └── summary.txt
│       │           ├── completion.json
│       │           └── logs/
│       └── financebench/
│           └── ...
├── gemma-3-4b-it/
│   └── ...
└── gpt-oss-20b/
    └── ...
```

**Reference results** (`judge_correct` %, pass@1 averaged over 5 repeats):

| Model | SecQUE | FinanceBench |
|-------|--------|--------------|
| Qwen3-4B | ~53% | ~79% |
| Gemma 3 4B IT | ~29% | ~53% |
| GPT-OSS 20B | ~62% | ~82% |

Qwen3-4B and GPT-OSS 20B leverage reasoning (thinking mode and Harmony format respectively), which significantly improves financial analysis accuracy. Gemma 3 4B IT, without built-in reasoning, shows the gap that SFT and GRPO fine-tuning aim to close.

<a id="agentic-demo" name="agentic-demo"></a>
**Agentic benchmark.** The same config also runs the multi-turn tool-use
benchmark `fabv1`. The quickstart uses NVFlow/NeMo-Skills-managed
vLLM services for both policy and judge. `fabv1` reads the SEC filing cache that
[Download SEC Filings](#step-2-download-sec-filings) writes, so run that step
first; companies the cache lacks are fetched from sec.gov on first use and added
to it.

With the default `noweb` tool set, `fabv1` needs no service keys. The `web` tool
set adds web search, which needs `TAVILY_API_KEY` exported in the shell that runs
`nvflow`; see [fabv1 tool sets](workflows/05-eval.md#fabv1-tool-sets).

```bash
uv run nvflow run --all --workflow finance:eval/demo \
  --cluster my_cluster \
  -e fabv1
```

Gemma 3 4B IT is skipped here: the agentic benchmark needs a tool-call parser
and vLLM ships none for its tool format. Omit `-e` entirely to run both scopes
at once.

`fabv1` runs without web search by default, so `TAVILY_API_KEY` only matters once
you set `fabv1_variant: web` — see "fabv1 tool sets" in
[the eval workflow guide](workflows/05-eval.md).

> **Tip:** For production baseline coverage, use `finance:eval/baselines`. It
> evaluates Qwen3-4B/14B/32B, GPT-OSS-120B, and Nemotron models across every
> benchmark and requires `NVIDIA_API_KEY` for its hosted judge on the NVIDIA
> inference hub. To use OpenAI instead, see
> [Hosted model endpoints](../../../docs/hosted-endpoints.md#using-openai).

</details>

---

<a id="step-2-download-sec-filings" name="step-2-download-sec-filings"></a>
<details>
<summary><h2>2. Download SEC Filings — ~10 min</h2></summary>

> - **Selector:** `finance:download/demo`
> - **Config:** `download/demo.yaml`
> - **Output:** `outputs/finance/demo/workflow-2-download-sec/`

Download 10-K, 10-Q, 8-K and DEF 14A filings for 7 demo companies from SEC EDGAR. The
finance project uses `edgartools` to fetch filings and extract sections.

> **Required first:** SEC EDGAR rejects requests that don't identify the caller.
> The workflow requires both identity values from the launcher environment, and
> neither has a default:
>
> ```bash
> export SEC_IDENTITY_EMAIL=your.email@company.com
> export SEC_IDENTITY_COMPANY="Your Company"
> ```
>
> To avoid re-exporting them on every new shell, put them in a gitignored `.env`
> and let `uv` load it, either per command with `--env-file .env` or once with
> `export UV_ENV_FILE=.env`:
>
> ```bash
> printf 'SEC_IDENTITY_EMAIL=your.email@company.com\nSEC_IDENTITY_COMPANY="Your Company"\n' > .env
> uv run --env-file .env nvflow run --all --workflow finance:download/demo --cluster my_cluster
> ```
>
> Quote any value containing spaces; an unquoted `.env` value is skipped with a
> parse warning and then reported as unset.
>
> If either is missing, the command stops before submitting. `plan` reports only
> `Workflow planning failed before configuration could be loaded`; see
> [SEC Identity Not Set](troubleshooting.md#sec-identity-not-set).
>
> See the [SEC Fair Access Policy](https://www.sec.gov/os/accessing-edgar-data).

**Inspect available stages:**
```bash
uv run nvflow list-stages --workflow finance:download/demo
```

**Workflow variants:**

- `finance:download/smoke` — two-company smoke scope
- `finance:download/demo` — seven-company demo scope (NVDA, AAPL, GOOG, CSCO, IBM, META, MSFT)
- `finance:download/sp500` — full S&P 500 production scope

Each variant runs `download` to fetch filings and extract sections, then
`build_index` to build the full-text search index over them. The demo and sp500
variants also run `prefetch_cache`, which warms the filing-metadata cache that
the same scope's GRPO and `fabv1` evaluation read; it needs no corpus, so it
runs alongside the download.

**Run:**
```bash
uv run nvflow run --all --workflow finance:download/demo --cluster my_cluster
```

**Monitor:**
```bash
uv run nvflow status --cluster my_cluster
```

**Verify:**
```bash
ls $WORKSPACE/outputs/finance/demo/workflow-2-download-sec/step-0-download/data/
# Expected: AAPL/ CSCO/ GOOG/ IBM/ META/ MSFT/ NVDA/

ls $WORKSPACE/outputs/finance/demo/workflow-2-download-sec/step-1-index/
# Includes: demo_sec_fts.sqlite  demo_sec_fts.sqlite.metadata

ls $WORKSPACE/outputs/finance/demo/workflow-2-download-sec/cache/finance_sec_search/
# Includes: filings_metadata/  tickers.json
```

**Output:**
```
outputs/finance/demo/workflow-2-download-sec/
├── step-0-download/
│   ├── data/
│   │   ├── AAPL/                        # 10-K, 10-Q, 8-K and DEF 14A filings (2020-2024)
│   │   ├── CSCO/
│   │   ├── GOOG/
│   │   ├── IBM/
│   │   ├── META/
│   │   ├── MSFT/
│   │   └── NVDA/
│   └── sec_metadata.parquet             # Filing metadata index
├── step-1-index/
│   ├── demo_sec_fts.sqlite              # Full-text search index
│   └── demo_sec_fts.sqlite.metadata     # Metadata companion
└── cache/finance_sec_search/
    ├── filings_metadata/                # One JSON per company, keyed by CIK
    └── tickers.json                     # Ticker-to-CIK registry
```

> **Tip:** For production, select `finance:download/sp500`.

</details>

---

<a id="step-3-generate-synthetic-qa-data" name="step-3-generate-synthetic-qa-data"></a>
<details>
<summary><h2>3. Generate Synthetic Q&A Data — ~45 min</h2></summary>

> - **Selector:** `finance:sdg/template-based-sdg-demo`
> - **Config:** `template-based-sdg-demo.yaml`
> - **Output:** `outputs/finance/demo/workflow-3-template-based-sdg/`

Generate financial Q&A pairs using the template-based SDG workflow.

> **Required first:** `create_seed_data` reaches SEC EDGAR, Hugging Face and Wikipedia, so before running:
>
> 1. Export the SEC identity consumed by the inherited workflow:
>
>    ```bash
>    export SEC_IDENTITY_EMAIL=your.email@company.com
>    export SEC_IDENTITY_COMPANY="Your Company"
>    ```
>
> 2. The seed dataset comes from Hugging Face, so this run needs network and the three offline flags (`HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE`, `TRANSFORMERS_OFFLINE`) cleared; re-enable them afterwards. Where the flags live depends on your backend: [Slurm](../../../docs/slurm/README.md#where-settings-live) or [Ray](../../../docs/ray/README.md#where-settings-live).

**Inspect available stages:**
```bash
uv run nvflow list-stages --workflow finance:sdg/template-based-sdg-demo
```

**Workflow stages:**

1. `create_seed_data` — Create seed questions for demo companies
2. `generate_questions` — Expand seed questions using LLM
3. `map_questions_to_context` — Find relevant SEC filing sections
4. `generate_answers` — Generate answer candidates
5. `genselect_answers` — Select best answers
6. `filter_answers` — Remove unanswerable questions

**Run:**
```bash
uv run nvflow run --all --workflow finance:sdg/template-based-sdg-demo --cluster my_cluster
```

**Monitor:**
```bash
uv run nvflow status --cluster my_cluster
```

**Verify:**
```bash
ls $WORKSPACE/outputs/finance/demo/workflow-3-template-based-sdg/step-5-filter-answers/final_result.jsonl
wc -l $WORKSPACE/outputs/finance/demo/workflow-3-template-based-sdg/step-5-filter-answers/final_result.jsonl
head -1 $WORKSPACE/outputs/finance/demo/workflow-3-template-based-sdg/step-5-filter-answers/final_result.jsonl | jq .
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

> **Tip:** For production S&P 500 generation, use
> `finance:sdg/template-based-sdg`, which selects larger models and the full
> filing set.

</details>

---

<a id="step-4-fine-tune-model--evaluate-sft" name="step-4-fine-tune-model--evaluate-sft"></a>
<details>
<summary><h2>4. Fine-Tune Model + Evaluate (SFT) — ~1 hr</h2></summary>

> - **Selector:** `finance:sft/qwen3_4b`
> - **Config:** `sft/qwen3_4b.yaml`
> - **Output:** `outputs/finance/demo/workflow-4-sft/qwen3_4b/`

Fine-tune Qwen3-4B on the generated Q&A data and evaluate checkpoints on
finance benchmarks. This workflow uses the Q&A pairs from the template-based
SDG workflow (`workflow-3-template-based-sdg/step-5-filter-answers/final_result.jsonl`).

**Inspect available stages:**
```bash
uv run nvflow list-stages --workflow finance:sft/qwen3_4b
```

**Workflow stages:**

1. `data_transformation` — Transform raw SDG data to standardized format (CPU)
2. `prepare_for_sft` — Apply prompt template and tokenizer formatting (CPU)
3. `train_validation_split` — Split into train/val sets with the shared `data.train_validation_split` stage (CPU)
4. `sequence_length_grouping` — Report sequence-length groups for the training data; training reads the split directly (CPU)
5. `training` — Run SFT training with the shared `nemo_rl.sft` stage on 1 node × 8 GPUs (GPU, ~12 min)
6. `eval` — Evaluate checkpoint on finance benchmarks (GPU, ~40 min)

**Note:** The `eval` stage reads the benchmark datasets that Step 1's `eval_data` stage prepared, so run that stage first. Check that they are there:
```bash
wc -l $WORKSPACE/outputs/finance/demo/workflow-1-baseline-eval/step-1-eval-data/{secque,financebench}/eval.jsonl
# Expected: 565 secque + 150 financebench
```

**Run:**
```bash
uv run nvflow run --all --workflow finance:sft/qwen3_4b --cluster my_cluster
```

**Monitor training:**
```bash
uv run nvflow status --cluster my_cluster
tail -f $WORKSPACE/outputs/finance/demo/workflow-4-sft/qwen3_4b/step-4-training/model-qwen3-4b-*/training-logs/ray-*-job.log
```

**Verify training:**
```bash
ls $WORKSPACE/outputs/finance/demo/workflow-4-sft/qwen3_4b/step-4-training/model-qwen3-4b-8g-tp2-pp1-cp2-seq32k/checkpoints/
# Expected: step_10/ and a final step_N/ (save_period=10; N depends on how many rows the split keeps)

ls $WORKSPACE/outputs/finance/demo/workflow-4-sft/qwen3_4b/step-4-training/model-qwen3-4b-8g-tp2-pp1-cp2-seq32k/hf_models/
# Expected: step_10/ (HF-format model, converted during eval)
```

**Verify evaluation:**
```bash
cat $WORKSPACE/outputs/finance/demo/workflow-4-sft/qwen3_4b/step-5-eval/step-10/collect-rollouts/secque/jobs/job-1-analyze-rollouts/analysis/evaluation/metrics.json
cat $WORKSPACE/outputs/finance/demo/workflow-4-sft/qwen3_4b/step-5-eval/step-10/collect-rollouts/financebench/jobs/job-1-analyze-rollouts/analysis/evaluation/metrics.json
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
│       │   └── step_<N>/                 # Final checkpoint
│       ├── hf_models/                   # HF-format models (converted during eval)
│       │   └── step_10/                 # Per-step HF checkpoint
│       └── training-logs/
└── step-5-eval/
    └── step-10/
        └── collect-rollouts/
            ├── secque/
            │   └── jobs/
            │       ├── job-0-collect-rollouts/
            │       └── job-1-analyze-rollouts/
            │           ├── outputs/
            │           ├── analysis/evaluation/metrics.json
            │           ├── completion.json
            │           └── logs/
            └── financebench/
                └── jobs/{job-0-collect-rollouts,job-1-analyze-rollouts}/
```

> **Note:** Demo results will vary due to limited training data (7 companies). Production training shows ~11% improvement over baseline.

> **Tip:** For production SFT, use `finance:sft/qwen3_14b` or
> `finance:sft/nemotron-3-nano`.

</details>

---

<a id="step-5-grpo-rl-training--evaluate" name="step-5-grpo-rl-training--evaluate"></a>
<details>
<summary><h2>5. GRPO RL Training + Evaluate — ~5 hr</h2></summary>

> - **Selector:** `finance:grpo/qwen3_4b`
> - **Config:** `grpo/qwen3_4b.yaml`
> - **Output:** `outputs/finance/demo/workflow-5-grpo/qwen3_4b/`

Run GRPO reinforcement learning using LLM-as-judge rewards from NeMo-Gym
environments, then evaluate checkpoints on finance benchmarks. It uses the same
Q&A pairs as the SFT workflow; GRPO does **not** depend on the SFT checkpoint.

**Two environments:** This demo trains on two independent NeMo-Gym environments, each producing a separate model:

| Environment | Reward Signal | Context |
|-------------|--------------|---------|
| `equivalence_llm_judge` | LLM judges answer equivalence to gold | Question + SEC context provided |
| `finance_sec_search` | Multi-turn tool-calling agent retrieves SEC data | Agent must find context via tools |

Each environment has its own data pipeline, rollout collection, and training. Use `-e <env>` to run a specific environment.

**Inspect available stages:**
```bash
uv run nvflow list-stages --workflow finance:grpo/qwen3_4b
```

**Workflow stages** (the number is the stage's `step-N` output directory):

- 0 `validate_questions` — Filter ambiguous questions for finance_sec_search (GPU, GPT-OSS-120B)
- 1 `data_transformation` — Clean raw SDG data to model-agnostic format (CPU)
- 2 `apply_prompt_template` — Apply prompt template + extract expected answer (CPU)
- 3 `convert_to_responses_api` — Convert to NeMo-Gym Responses API format (CPU)
- 4 `prepare_data` — Add agent routing fields for NeMo-Gym (CPU)
- 5 `collect_rollouts` — Collect rollouts, profile rewards, and filter training data (GPU)
- 7 `train_validation_split` — Split reward-filtered data into train/val with the shared `nemo_gym.train_validation_split` stage (CPU)
- 8 `training` — GRPO training with NeMo-Gym environment through the shared `nemo_rl.grpo` stage (GPU)
- 9 `eval` — Evaluate checkpoint on finance benchmarks (GPU, ~40 min)

Step 6 is the optional `rejudge` stage, which the demo leaves off. Every stage
writes **per-environment** outputs under `step-N-<stage>/<env_name>/`.

The `collect_rollouts` stage includes automatic sub-jobs:
- **Native Gym rollout** (GPU) — two NVFlow chunks demonstrate parallel/resumable
  execution while Gym runs eight repeats per sample; NVFlow supplies managed
  policy and judge endpoints
- **Aggregate + profile + health** — Gym merges canonical chunks, profiles rewards, and writes quality verdicts
- **BLADE** (optional, inline with CPU finalization) — inspect unfiltered artifacts when enabled
- **Dataset selection** (CPU) — normalize the Gym profile into standardized
  `train.jsonl` and optionally retain only prompts with useful reward variance

The quickstart launches separate policy and judge services through
NVFlow/NeMo-Skills. Gym consumes their OpenAI-compatible endpoints.

**Note:** The `eval` stage reads the benchmark datasets that Step 1's `eval_data`
stage prepared, so run that stage first. Check that they are there:
```bash
wc -l $WORKSPACE/outputs/finance/demo/workflow-1-baseline-eval/step-1-eval-data/{secque,financebench}/eval.jsonl
# Expected: 565 secque + 150 financebench
```

**Run:**

Submit each environment with its own command; the two can run side by side
when the cluster has room. Each command submits the environment's stages
in order and returns; every stage starts only after the one it depends on
succeeds, so there is nothing to wait for between stages. Follow progress with
the **Monitor** commands below. Environment 1 takes about 2 hours and
environment 2 about 5, most of it in its multi-turn rollouts.

**Environment 1: `equivalence_llm_judge`**

The simpler environment — LLM judges whether the model's answer is equivalent to the gold answer. Context is provided directly, so it skips question validation.

```bash
# Data prep (CPU) → rollouts (GPU) → train/val split (CPU) → training (FSDP, 16 GPUs) → eval
uv run nvflow run data_transformation apply_prompt_template convert_to_responses_api prepare_data \
  collect_rollouts train_validation_split training eval \
  --workflow finance:grpo/qwen3_4b --cluster my_cluster -e equivalence_llm_judge
```

**Environment 2: `finance_sec_search`**

The multi-turn tool-calling environment — the agent must retrieve SEC filings via tools before answering. It runs every stage, starting with question validation, and trains with Megatron on 32 GPUs, using YaRN for 131K context.

The agent's filing-search tools read the index and cache that Step 2 builds
(see the [download workflow guide](workflows/01-download-sec.md)), so run Step 2
first.

```bash
uv run nvflow run --all --workflow finance:grpo/qwen3_4b --cluster my_cluster -e finance_sec_search
```

To inspect the rollouts before training starts, list only the stages up to
`train_validation_split` (environment 2 starts with `validate_questions`), check
them as shown below, then run `training eval` with the same `--workflow` and
`-e`.

**Monitor:**
```bash
uv run nvflow status --cluster my_cluster
# Rollout logs (one per chunk/attempt per environment)
tail -f $WORKSPACE/outputs/finance/demo/workflow-5-grpo/qwen3_4b/step-5-collect-rollouts/*/jobs/job-0-collect-rollouts/chunks/chunk-*/attempts/attempt-*/logs/*.log
# Training logs
tail -f $WORKSPACE/outputs/finance/demo/workflow-5-grpo/qwen3_4b/step-8-training/*/grpo-qwen3-4b-*/training-logs/ray-*-job.log
```

**Verify rollouts (both environments):**
```bash
# equivalence_llm_judge
cat $WORKSPACE/outputs/finance/demo/workflow-5-grpo/qwen3_4b/step-5-collect-rollouts/equivalence_llm_judge/jobs/job-2-build-training-dataset/selection_report.json
wc -l $WORKSPACE/outputs/finance/demo/workflow-5-grpo/qwen3_4b/step-5-collect-rollouts/equivalence_llm_judge/train.jsonl

# finance_sec_search
cat $WORKSPACE/outputs/finance/demo/workflow-5-grpo/qwen3_4b/step-5-collect-rollouts/finance_sec_search/jobs/job-2-build-training-dataset/selection_report.json
wc -l $WORKSPACE/outputs/finance/demo/workflow-5-grpo/qwen3_4b/step-5-collect-rollouts/finance_sec_search/train.jsonl
```

**Verify training (per-environment):**
```bash
# equivalence_llm_judge model
ls $WORKSPACE/outputs/finance/demo/workflow-5-grpo/qwen3_4b/step-8-training/equivalence_llm_judge/grpo-qwen3-4b-*/checkpoints/
# Expected: step_10/ (save_period=10, max_num_steps=10)

# finance_sec_search model
ls $WORKSPACE/outputs/finance/demo/workflow-5-grpo/qwen3_4b/step-8-training/finance_sec_search/grpo-qwen3-4b-*/checkpoints/
# Expected: step_10/
```

**Verify evaluation** (results are per-environment, matching the training checkpoints):
```bash
for env in equivalence_llm_judge finance_sec_search; do
  for bench in secque financebench; do
    cat $WORKSPACE/outputs/finance/demo/workflow-5-grpo/qwen3_4b/step-9-eval/$env/step-10/collect-rollouts/$bench/jobs/job-1-analyze-rollouts/analysis/evaluation/metrics.json
  done
done
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
│   │   └── agent_config_overlay.yaml       # generated internal Gym config
│   └── finance_sec_search/
│       ├── train.jsonl
│       └── agent_config_overlay.yaml       # generated internal Gym config
├── qwen3_4b/
│   ├── step-5-collect-rollouts/
│   │   ├── equivalence_llm_judge/
│   │   │   ├── profiled.jsonl
│   │   │   ├── train.jsonl
│   │   │   └── jobs/{job-0-collect-rollouts,job-1-analyze-rollouts,job-2-build-training-dataset}/
│   │   └── finance_sec_search/
│   │       ├── profiled.jsonl
│   │       ├── train.jsonl
│   │       └── jobs/{job-0-collect-rollouts,job-1-analyze-rollouts,job-2-build-training-dataset}/
│   ├── step-7-train-validation-split/
│   │   ├── equivalence_llm_judge/      # train.jsonl, val.jsonl, split.done
│   │   └── finance_sec_search/         # train.jsonl, val.jsonl, split.done
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
│       │   └── step-10/
│       │       └── collect-rollouts/
│       │           ├── secque/jobs/job-1-analyze-rollouts/analysis/evaluation/metrics.json
│       │           └── financebench/jobs/job-1-analyze-rollouts/analysis/evaluation/metrics.json
│       └── finance_sec_search/
│           └── step-10/
│               └── collect-rollouts/
│                   ├── secque/jobs/job-1-analyze-rollouts/analysis/evaluation/metrics.json
│                   └── financebench/jobs/job-1-analyze-rollouts/analysis/evaluation/metrics.json
```

> **Per-environment training:** Each environment produces a separate model checkpoint.

> **Note:** Demo results will vary due to limited training data (7 companies) and rollout stochasticity. The filter stage typically keeps 20-40% of questions (removing too-hard and too-easy), which provides the best RL training signal.

> **Tip:** The production recipe uses `finance:grpo/nemotron_nano_30b_a3b`.
> `finance:grpo/qwen3_30b_a3b` is also available for Qwen3-30B-A3B experiments.

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
[SDG (Production)](workflows/02-template-based-sdg.md#production-run-complete-workflow) |
[SFT (Qwen3-14B or Nemotron-3-Nano)](workflows/04-sft.md) |
[GRPO (Nemotron-3-Nano)](workflows/06-grpo.md#production-nemotron-3-nano-30b-a3b)

---

## Troubleshooting

See the comprehensive **[Finance Project Troubleshooting](troubleshooting.md)** guide for issues across all workflows (SEC rate limits, missing `sec_metadata.parquet`, out-of-memory and disk-space errors, offline-runtime errors, refused reuse of earlier output, and more).

---

[Workflow documentation](workflows/) | [Finance project README](../README.md)
