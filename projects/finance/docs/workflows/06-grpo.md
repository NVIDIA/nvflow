# GRPO Reinforcement Learning

## Purpose

Further improve fine-tuned models using Group Relative Policy Optimization (GRPO) with LLM-as-judge reward signals from NeMo-Gym.

> **Note:** The shipped configs start GRPO from the base model, so GRPO does not depend on [SFT](04-sft.md). To start from an SFT checkpoint instead, point `training.hf_checkpoint_path` at its Hugging Face export (`hf_models/step_<N>/`).

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
- ✅ GPU resources (16 GPUs / 2 nodes for equivalence demo, 32 GPUs / 4 nodes for finance SEC demo, 128 GPUs / 16 nodes for [production](#production-nemotron-3-nano-30b-a3b) training)
- ✅ Filing corpus and search index from [Download SEC Filings](01-download-sec.md), for the `finance_sec_search` environment (see [Filing Search](#filing-search))

## Workflow Overview

### Stages

```
┌──────────────────────────────┐
│ 0. validate_questions        │  Regex prefilter (CPU) + LLM classifier (GPU)
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
│ 4. prepare_data              │  Add task_source routing fields (CPU)
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│ 5. collect_rollouts          │  Rollout collection + reward profiling + filter (GPU)
└───────────┬──────────────────┘
            │
            ▼   (6. rejudge, optional)
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

The step numbers match the `step-N-<stage>` output directories.

**9 active stages plus 1 optional stage:**
1. **validate_questions** (Step 0): Drop structurally broken questions with a regex prefilter, then a GPU LLM classifier
2. **data_transformation** (Step 1): Normalize raw SDG data to model-agnostic schema (shared with SFT pipeline)
3. **apply_prompt_template** (Step 2): Format the problem field using a prompt template and extract the concise expected answer
4. **convert_to_responses_api** (Step 3): Convert prompted data to NeMo-Gym Responses API format (lossless)
5. **prepare_data** (Step 4): Run `gym dataset collate` to stamp JSONL records with agent routing fields for NeMo-Gym
6. **collect_rollouts** (Step 5): Collect and profile model rollouts, run health checks, and publish standardized selected training data
7. **train_validation_split** (Step 7): Split data into train/val sets with stratified sampling (shared with SFT pipeline)
8. **training** (Step 8): GRPO training using NeMo-RL with online NeMo-Gym environment rewards
9. **eval** (Step 9): Evaluate GRPO checkpoints on finance benchmarks

> **Optional:** **rejudge** — Apply a different/stronger judge to existing rollouts without regenerating responses

**See [technical reference](#detailed-stage-reference) for detailed stage documentation.**

### Model Configurations

| Config | Model | Environment | Backend | GPUs | Status |
|--------|-------|-------------|---------|------|--------|
| `grpo/qwen3_4b.yaml` with `-e equivalence_llm_judge` | Qwen3-4B | equivalence_llm_judge | FSDP (32K) | 16 (2 nodes) | Demo |
| `grpo/qwen3_4b.yaml` with `-e finance_sec_search` | Qwen3-4B | finance_sec_search | Megatron (TP2×CP8, 131K) | 32 (4 nodes) | Demo |
| `grpo/nemotron_nano_30b_a3b.yaml` with `-e finance_sec_search` | Nemotron-3-Nano-30B-A3B (MoE) | finance_sec_search | Megatron (TP4×PP4×CP8, EP32, 48K) | 128 (16 nodes) | Production |
| `grpo/qwen3_30b_a3b.yaml` | Qwen3-30B-A3B (MoE) | — | Megatron (TP4×CP8, EP8, 131K) | 64 (8 nodes) | Smoke-scale example |

## Usage

### Run Complete Workflow (Demo)

Each environment trains its own model, so run one command per environment:

```bash
CONFIG=projects/finance/workflows/grpo/qwen3_4b.yaml

# equivalence_llm_judge: the prompt carries the context, so it skips validate_questions
uv run nvflow run data_transformation apply_prompt_template convert_to_responses_api prepare_data \
  collect_rollouts train_validation_split training eval \
  --config $CONFIG --cluster my_cluster -e equivalence_llm_judge

# finance_sec_search: every stage, starting with validate_questions
uv run nvflow run --all --config $CONFIG --cluster my_cluster -e finance_sec_search
```

### Production (Nemotron-3-Nano-30B-A3B)

The `finance:document-grounded-local-search-grpo` recipe trains
`grpo/nemotron_nano_30b_a3b.yaml` on the `finance_sec_search` environment,
which answers from the local full-text index:

```bash
uv run nvflow run --all \
  --config projects/finance/workflows/grpo/nemotron_nano_30b_a3b.yaml \
  --cluster my_cluster \
  -e finance_sec_search
```

Run the recipe's upstream workflows first: `finance:download/sp500` for the
filings, index and cache, `finance:sdg/document-grounded-sdg-sp500` for the
training data, and `finance:eval/baselines` for the benchmark data the
checkpoint eval reads. The models it serves are
`/hf_models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` and
`/hf_models/openai/gpt-oss-120b`.

- **Data:** the S&P 500 document-grounded SDG output
  (`/workspace/outputs/finance/sp500/workflow-3-document-grounded-sdg/step-6-post-process`),
  which `validate_questions` filters with GPT-OSS-120B on 4 GPUs.
- **Rollouts:** up to 36,000 samples × 8 Gym repeats in 24 chunks. Each chunk is
  a 5-GPU job (a 4-GPU policy and a 1-GPU GPT-OSS-120B judge) with
  `dependent_jobs: 3`, which gives it four chained attempts for a 4-hour job limit.
- **Training:** Megatron on 128 GPUs (16 nodes × 8), TP4 × PP4 × CP8 with EP32
  and 48K sequences, plus a 1-GPU judge. Each step uses 128 prompts × 16
  generations. `max_num_steps` is 300, but the five chained jobs
  (`dependent_jobs: 4`) are sized for about 50 steps; raise `dependent_jobs`
  to train longer.
- **Eval:** the step-20 checkpoint on `secque` and `financebench`.

The training stage logs to W&B (`wandb_mode: online`). Export `WANDB_API_KEY`
in the shell that runs `nvflow`; a Slurm target forwards it to the job (see the
API-key notes in [`template-slurm.yaml`](../../../../cluster_configs/template-slurm.yaml)),
and on Ray it must be set in the clusters' environment. Without a W&B account,
set `stages.training.wandb_mode: disabled` in an overlay.

### Stage-by-Stage Execution

```bash
CONFIG=projects/finance/workflows/grpo/qwen3_4b.yaml
ENVIRONMENT=finance_sec_search

# Step 0: Drop broken questions: regex prefilter (CPU), then LLM classifier (GPU)
uv run nvflow run validate_questions --config $CONFIG --cluster my_cluster -e "$ENVIRONMENT"

# Step 1: SDG cleanup → model-agnostic schema (CPU)
uv run nvflow run data_transformation --config $CONFIG --cluster my_cluster -e "$ENVIRONMENT"

# Step 2: Apply prompt template + extract expected answer (CPU)
uv run nvflow run apply_prompt_template --config $CONFIG --cluster my_cluster -e "$ENVIRONMENT"

# Step 3: Convert to NeMo-Gym Responses API format (CPU)
uv run nvflow run convert_to_responses_api --config $CONFIG --cluster my_cluster -e "$ENVIRONMENT"

# Step 4: Prepare data — add agent routing fields (CPU)
uv run nvflow run prepare_data --config $CONFIG --cluster my_cluster -e "$ENVIRONMENT"

# Step 5: Collect rollouts (inference + reward scoring + filter) (GPU)
uv run nvflow run collect_rollouts --config $CONFIG --cluster my_cluster -e "$ENVIRONMENT"

# Step 7: Split into train/val sets (CPU)
uv run nvflow run train_validation_split --config $CONFIG --cluster my_cluster -e "$ENVIRONMENT"

# Step 8: GRPO training (GPU)
uv run nvflow run training --config $CONFIG --cluster my_cluster -e "$ENVIRONMENT"

# Step 9: Evaluate checkpoints (GPU)
uv run nvflow run eval --config $CONFIG --cluster my_cluster -e "$ENVIRONMENT"
```

With `ENVIRONMENT=equivalence_llm_judge`, start at Step 1: `qwen3_4b.yaml` runs
`validate_questions` for `finance_sec_search` only.

### Optional: Re-judge Rollouts

To re-judge rollouts with a different judge model, enable `rejudge` in `pipeline_stages` and configure the judge:

```yaml
# In your model config
reward_source_stage: rejudge
pipeline_stages:
  - validate_questions
  - data_transformation
  - apply_prompt_template
  - convert_to_responses_api
  - prepare_data
  - collect_rollouts
  - rejudge                  # Optional; listing it here enables it
  - train_validation_split
  - training
  - eval

stages:
  rejudge:
    verification:
      models:
        judge:
          deployment: external
          model_type: openai_model
          model: openai/openai/gpt-5.2
          base_url: https://inference-api.nvidia.com/v1
          api_key_env: NVIDIA_API_KEY
```

See [Hosted Model Endpoints](../../../../docs/hosted-endpoints.md) to point this
at OpenAI or another provider.

## Data & Results

### Output Structure

```
outputs/finance/demo/workflow-5-grpo/
├── step-0-validate-questions/
│   └── {env_name}/                      # Questions kept by the regex and LLM filters
├── step-1-data-transformation/
│   └── {env_name}/
│       ├── chunks/                      # Normalized, chunked SDG data
│       └── logs/
├── step-2-apply-prompt-template/
│   └── {env_name}/
│       ├── final_result_chunk*.jsonl    # Prompted data with extracted answers
│       └── logs/
├── step-3-convert-to-responses-api/
│   └── {env_name}/
│       ├── final_result.jsonl           # Data in Responses API format
│       └── logs/
├── step-4-prepare-data/
│   └── {env_name}/
│       ├── train.jsonl                  # Pre-rollout data with task_source
│       └── agent_config_overlay.yaml    # Generated internal Gym config
└── qwen3_4b/                            # Model-specific outputs
    ├── step-5-collect-rollouts/
    │   └── {env_name}/                  # Per-environment subdirectory
    │       ├── profiled.jsonl           # All normalized profiled data
    │       ├── train.jsonl              # Selected training data
    │       └── jobs/
    │           ├── job-0-collect-rollouts/
    │           │   └── chunks/chunk-N/  # Per-chunk rollouts; logs in attempts/attempt-*/logs/
    │           ├── job-1-analyze-rollouts/
    │           │   ├── outputs/         # rollouts.jsonl, rollouts_reward_profiling.jsonl,
    │           │   │                    # quality_summary.json, rollout_verdicts.jsonl
    │           │   ├── logs/
    │           │   └── completion.json
    │           └── job-2-build-training-dataset/
    │               ├── selection_report.json
    │               └── completion.json
    ├── step-6-rejudge/                  # (only if rejudge enabled)
    │   └── {env_name}/                  # Re-judged + filtered data
    ├── step-7-train-validation-split/
    │   └── {env_name}/
    │       ├── train.jsonl              # Training split
    │       ├── val.jsonl                # Validation split
    │       └── split.done               # Row counts and digests of both files
    ├── step-8-training/
    │   ├── equivalence_llm_judge/
    │   │   └── grpo-qwen3-4b-16g-tp2-cp1-seq32k/    # Demo, FSDP
    │   └── finance_sec_search/
    │       └── grpo-qwen3-4b-32g-tp2-cp8-seq128k/   # Demo, Megatron (YaRN 131K)
    │           ├── checkpoints/         # GRPO model checkpoints
    │           └── training-logs/
    └── step-9-eval/
        └── {env_name}/step-{N}/collect-rollouts/{benchmark}/   # Benchmark evaluation results
```

### Expected Results (Demo)

| Stage | Output | Notes |
|-------|--------|-------|
| validate_questions | `{env_name}/final_result.jsonl` | `finance_sec_search` only; GPU (GPT-OSS-120B) |
| data_transformation | `{env_name}/chunks/final_result_chunk*.jsonl` | CPU-only, ~1 min |
| apply_prompt_template | `{env_name}/final_result_chunk*.jsonl` | CPU-only, ~1 min |
| convert_to_responses_api | `{env_name}/final_result.jsonl` | CPU-only, ~1 min |
| prepare_data | `{env_name}/train.jsonl` with task_source fields | CPU-only, ~1 min |
| collect_rollouts | Gym rollout bundle + profile/health + selected `train.jsonl` | GPU inference |
| train_validation_split | `{env_name}/train.jsonl` + `val.jsonl` | CPU-only, ~1 min |
| training | GRPO checkpoint | GPU training |
| eval | Benchmark scores | GPU inference, ~40 min |

End to end, `equivalence_llm_judge` takes about 2 hours and `finance_sec_search`
about 5, most of it in its multi-turn rollouts.

### Validation

`$WORKSPACE` is the host directory your target mounts at `/workspace`, as in the
[quick start](../quick-start.md).

```bash
BASE_DIR="$WORKSPACE/outputs/finance/demo/workflow-5-grpo"
MODEL_DIR="$BASE_DIR/qwen3_4b"

# Check normalized SDG data (Step 1)
head -1 $BASE_DIR/step-1-data-transformation/{env_name}/chunks/*.jsonl | jq 'keys'

# Check prompted data (Step 2)
head -1 $BASE_DIR/step-2-apply-prompt-template/{env_name}/*.jsonl | jq '.problem' | head -c 200

# Check Responses API conversion (Step 3)
head -1 $BASE_DIR/step-3-convert-to-responses-api/{env_name}/final_result.jsonl | jq 'keys'

# Check prepared data with task_source (Step 4)
head -1 $BASE_DIR/step-4-prepare-data/{env_name}/train.jsonl | jq '.task_source'

# Check dataset selection (Step 5 — replace {env_name} with your environment)
cat $MODEL_DIR/step-5-collect-rollouts/{env_name}/jobs/job-2-build-training-dataset/selection_report.json

# Check train/val split (Step 7)
wc -l $MODEL_DIR/step-7-train-validation-split/{env_name}/train.jsonl \
  $MODEL_DIR/step-7-train-validation-split/{env_name}/val.jsonl

# Check training checkpoint (Step 8)
ls $MODEL_DIR/step-8-training/{env_name}/grpo-*/checkpoints/

# Check eval results (Step 9)
ls $MODEL_DIR/step-9-eval/{env_name}/
```

## Customization

### Filing Search

The `finance_sec_search` policy answers from SEC filings through one of two tools,
and the prompt template and the environment overlay have to agree on which:

| Setting | `edgar_search` | `sec_filing_search` |
|---------|----------------|---------------------|
| `environments.finance_sec_search.prompt_template` | `..._edgar_search_without_web.yaml` | `..._sec_filing_search_without_web.yaml` |
| `edgar_search_mode` in the environment overlay | `local` | `null` |
| `local_edgar_index_path` in the environment overlay | a built index | `null` |

`prompt_template` decides what the training records advertise, so it decides what
the policy can call; the overlay decides what the environment can serve. If they
disagree the tool returns an error on every call and the rollouts train on
failures.

There is one template per (filing-search tool, `web_search`) combination, under
`projects/finance/prompts/`:

| | without `web_search` | with `web_search` |
|---|---|---|
| `edgar_search` | `finance_sec_search_template_edgar_search_without_web.yaml` | `finance_sec_search_template_edgar_search_with_web.yaml` |
| `sec_filing_search` | `finance_sec_search_template_sec_filing_search_without_web.yaml` | `finance_sec_search_template_sec_filing_search_with_web.yaml` |

Each is generated from NeMo-Gym's
`resources_servers/finance_sec_search/scripts/convert_questions.py`, so a dataset
built by that script (`--search-tool <tool>`, optionally `-w`) presents the same
tools, in the same order, with the same descriptions. `web_search` also needs
`TAVILY_API_KEY`; without it the tool reports itself unavailable and scores drop.

```yaml
environments:
  finance_sec_search:
    prompt_template: project://prompts/finance_sec_search_template_edgar_search_without_web.yaml
    config_paths:
      - resources_servers/finance_sec_search/configs/finance_sec_search.yaml
    gym:
      finance_sec_search_resources_server:
        resources_servers:
          finance_sec_search:
            # Serves edgar_search from the index the download workflow built.
            edgar_search_mode: local
            local_edgar_index_path: ${data_root}/workflow-2-download-sec/step-1-index/${scope}_sec_fts.sqlite
```

`edgar_search` names a tool, not a backend. Training serves it from the
full-text index built by [Download SEC Filings](01-download-sec.md) and reads
filing bodies from the downloaded corpus, so rollouts need no network.
`sec_filing_search` looks up filing metadata by ticker, served from the cache
that the download workflow's `prefetch_cache` warms. In evaluation, the agentic
`fabv1` benchmark offers `sec_filing_search`, plus `web_search` in its `web`
tool set, and the shipped checkpoint evals score only the QA benchmarks.

#### Image prerequisite

`edgar_search` is served by Gym's `finance_sec_search` component. Images built
before that component gained `local_edgar_search.py` cannot serve it, and the
tool reports itself unavailable. The pinned `nemo-rl` and `nemo-gym` images both
carry it. Do not bind-mount Gym source over them: the images bake each
component's venv inside its own directory, and a mount forces a runtime resolve
on a node that may have no network — see [troubleshooting](../troubleshooting.md).

### Judge Configuration

Endpoint ownership is explicit for both policy and judge:

```yaml
stages:
  collect_rollouts:
    rollout:
      models:
        policy:
          deployment: nvflow
          server_type: vllm
          model: /hf_models/Qwen/Qwen3-30B-A3B
          num_gpus: 4
        judge:
          deployment: nvflow
          server_type: vllm
          model: /hf_models/openai/gpt-oss-120b
          num_gpus: 4
```

### Scaling for Production

For large-scale rollout collection (300K+ samples), raise the chunk count and
chain continuation jobs. The production config's settings are listed under
[Production (Nemotron-3-Nano-30B-A3B)](#production-nemotron-3-nano-30b-a3b).

```yaml
stages:
  collect_rollouts:
    rollout:
      num_chunks: 8
      num_repeats: 5
      num_samples_in_parallel: 512
      dependent_jobs: 2
      responses_create_params:
        max_output_tokens: 32768
```

**`dependent_jobs`**: Each chunk spawns `dependent_jobs + 1` Slurm jobs. The
first attempt retains the cluster's fail-closed `afterok` dependency on prior
stages; only continuation attempts and their analyze job use `afterany`. Every
attempt invokes the same native Gym `--resume` command, and analysis
accepts only content-validated `completion.json` records. The row-level
failure budget never drops below Gym's default of three and expands when a
longer Slurm continuation chain is configured.

### Existing vLLM endpoint (SDG mode)

Use a pre-launched vLLM server (any version) instead of the self-contained launch:

```yaml
stages:
  collect_rollouts:
    rollout:
      models:
        policy:
          deployment: external
          model: my-model
          base_url: "http://<host>:<port>/v1"
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

The demo trains each environment in `qwen3_4b.yaml` separately with its own backend: `-e equivalence_llm_judge` uses **FSDP** at 32K, while `-e finance_sec_search` uses **Megatron** (TP2×CP8) for YaRN context extension to 131K. When `-e` selects one environment, its `environments.<name>` entry can override the training stage (`training`), the policy (`training_policy`) and the checkpoint eval (`eval`). The MoE configs use **Megatron** with expert parallelism: the production `nemotron_nano_30b_a3b.yaml` on 128 GPUs, and the smoke-scale Qwen3-30B-A3B example `qwen3_30b_a3b.yaml` on 64 GPUs.

**Qwen3-30B-A3B (Megatron, `qwen3_30b_a3b.yaml`):**

```yaml
stages:
  training:
    backend: megatron
    total_gpus: 64
    overrides:
      policy:
        megatron_cfg:
          tensor_model_parallel_size: 4
          pipeline_model_parallel_size: 1
          context_parallel_size: 8
          expert_model_parallel_size: 8
          expert_tensor_parallel_size: 1
          sequence_parallel: true
          activation_checkpointing: true
```

TP × CP × PP = 32, so data parallelism is 64 / 32 = 2. Context parallelism
above 1 requires `policy.sequence_packing.enabled: true`, which the config sets.

## Additional Resources

### Monitoring

```bash
# Check your queued and running jobs
uv run nvflow status --cluster my_cluster

# View rollout logs, one per chunk and attempt (replace {env_name} with your environment)
tail -f $MODEL_DIR/step-5-collect-rollouts/{env_name}/jobs/job-0-collect-rollouts/chunks/chunk-*/attempts/attempt-*/logs/*.log

# View training logs
tail -f $MODEL_DIR/step-8-training/{env_name}/grpo-*/training-logs/ray-*-job.log
```

### Troubleshooting

**vLLM server fails to start:**
- Check GPU availability: `sinfo -p interactive`
- Review rollout logs: `cat $MODEL_DIR/step-5-collect-rollouts/{env_name}/jobs/job-0-collect-rollouts/chunks/chunk-*/attempts/attempt-*/logs/*.log`
- Ensure `tensor_parallel_size` doesn't exceed available GPUs

**All rewards are 0.0 or 1.0:**
- Check `rollouts_reward_profiling.jsonl` and `quality_summary.json` under `step-5-collect-rollouts/{env_name}/jobs/job-1-analyze-rollouts/outputs/`
- Policy-as-judge produces circular evaluation — use a separate judge for meaningful rewards
- Review the native Gym logs under the environment's `jobs/job-0-collect-rollouts/` and `jobs/job-1-analyze-rollouts/logs/` directories

**Rollouts missing metadata (uuid, question):**
- The enrichment step automatically restores fields dropped by NeMo-Gym environments
- Check enrichment output in the `jobs/job-1-analyze-rollouts/logs/` directory

### Next Steps

After GRPO training:

- **[Evaluate GRPO Model](05-eval.md)** - Compare GRPO checkpoint with SFT baseline
- **Scale Up** - Increase data, nodes, and training steps for production
- **Judge Iteration** - Try different judge models via `rejudge`

### Technical Details

This workflow uses **NeMo-RL** for GRPO training and **NeMo-Gym** for environment-based reward computation.

**Supported NeMo-Gym environments:**

| Environment | Reward Mode | Description |
|-------------|-------------|-------------|
| `equivalence_llm_judge` | Binary (0.0 / 1.0) | Semantic equivalence scoring via an LLM judge |
| `finance_sec_search` | Scaled (0.0 / 0.5 / 1.0) | Real SEC filing retrieval + LLM judge with partial credit |
| `mcqa` | Binary (0.0 / 1.0) | Multiple-choice QA placeholder; disabled in `grpo/base.yaml` until it has training data |

The `finance_sec_search` environment reads the search index and filing cache that [Download SEC Filings](01-download-sec.md) builds. Run that scope before training against it.

For comprehensive stage-by-stage documentation:
- **[GRPO Stages Reference](#detailed-stage-reference)**

## Detailed stage reference
Technical reference for the GRPO RL training workflow: 9 active stages plus `rejudge`, which is optional and commented out by default.

> **Use `-e <environment>` to narrow a run.** A model config's `environments`
> block merges with `grpo/base.yaml`. When no filter is supplied, stages combine
> every environment with `raw_train_data`; unconfigured placeholders such as
> `mcqa` are skipped consistently.

### Quick Navigation

Listed in execution order.

- [validate_questions](#validate_questions) — step 0
- [data_transformation](#data_transformation) — step 1
- [apply_prompt_template](#apply_prompt_template) — step 2
- [convert_to_responses_api](#convert_to_responses_api) — step 3
- [prepare_data](#prepare_data) — step 4
- [collect_rollouts](#collect_rollouts) — step 5
- [rejudge](#rejudge) — step 6, optional
- [train_validation_split](#train_validation_split) — step 7
- [training](#training) — step 8
- [eval](#eval) — step 9

---

### validate_questions

**File:** `projects/finance/nvfinance/stages/rl/validate_questions.py`
**Binding:** workflow alias `validate_questions` selects `ValidateQuestionsStage`

#### Purpose

Drop structurally-broken SDG questions before they enter the pipeline, per environment, in two phases:

1. **Regex prefilter (CPU).** Drops questions that say "the company" / "the firm" with no named company or ticker anywhere in the text. Deliberately narrow — recall over precision.
2. **LLM classifier (GPU).** Asks a judge model (GPT-OSS-120B by default) for `VALID` / `INVALID` on each survivor. Parse failures default to `VALID`.

The kept stream is written where `data_transformation` can read it, so a model config re-points `env.raw_train_data` at this stage's output.

#### Outputs

```
${step-0-validate-questions}/${env_name}/
├── final_result.jsonl          # VALID records, consumed by data_transformation
├── phase1_regex/               # prefiltered + dropped + stats (audit)
└── phase2_llm/                 # raw generation, parsed tags, dropped, stats
```

#### Resources

- **Phase 1:** CPU only
- **Phase 2:** GPU, for the judge model

---

### data_transformation

**File:** `projects/finance/nvfinance/stages/shared/data_transformation.py`
**Script:** `projects/finance/nvfinance/utils/shared/dataset_transformer.py`
**Binding:** workflow alias `data_transformation` selects `DataTransformationStage`

#### Purpose

Normalize raw SDG dataset to a model-agnostic schema. Shared between SFT and GRPO workflows. Standardizes field names to:

- `problem`, `context`, `reasoning_content`, `generation`, `uuid`, `question_type`

Generates deterministic UUIDs based on problem + generation content. Computes length statistics for key fields. Supports outlier filtering by percentile.

#### Inputs

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

#### Outputs

- `${output_dir}/chunks/` — Chunked JSONL files with normalized schema

#### Resources

- **Compute:** CPU only (no GPU)
- **Runtime:** ~1 min

---

### apply_prompt_template

**File:** `projects/finance/nvfinance/stages/rl/apply_prompt_template.py`
**Script:** `projects/finance/nvfinance/utils/rl/prompt_template_applier.py`
**Binding:** workflow alias `apply_prompt_template` selects `ApplyPromptTemplateStage`

#### Purpose

Format the `problem` field using a YAML prompt template (merging instruction + context + question) so the model receives the same structured prompt it was SFT-trained on. Optionally extracts the concise expected answer after a configurable prefix (e.g. `"Answer:"`) from `generation` for cleaner judge evaluation.

#### Inputs

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input_dir` | path | Directory with chunked JSONL from data_transformation | Required |
| `output_dir` | path | Output directory | Required |
| `prompt_template` | path | Path to YAML prompt template file | Required |
| `answer_prefix` | string | Prefix to extract concise answer from generation (e.g. `"Answer:"`) | `null` |

`prompt_template` and `answer_prefix` are set per environment in the
`environments` dict, not on the stage.

A record can only call the tools its template lists, so `prompt_template` is
what decides whether the policy searches the local full-text index
(`edgar_search`) or looks up filing metadata by ticker (`sec_filing_search`),
and whether `web_search` is available. It has to agree with how the environment
is served: `edgar_search` returns an error unless the environment overlay sets
`edgar_search_mode: local` and `local_edgar_index_path`.

#### Outputs

- `${output_dir}/` — Processed JSONL chunks with `prompt` and `expected_answer` fields

#### Resources

- **Compute:** CPU only (no GPU)
- **Runtime:** ~1 min

---

### convert_to_responses_api

**File:** `projects/finance/nvfinance/stages/rl/convert_to_responses_api.py`
**Script:** `projects/finance/nvfinance/utils/rl/responses_api_converter.py`
**Binding:** workflow alias `convert_to_responses_api` selects `ConvertToResponsesAPIStage`

#### Purpose

Lossless conversion of prompted data to the NeMo-Gym
`responses_create_params` format required by `gym dataset collate` and
downstream stages. All original fields are preserved alongside the new
`responses_create_params` wrapper.

#### Inputs

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input_path` | path | Input JSONL file or directory from apply_prompt_template | Required |
| `output_dir` | path | Output directory | Required |
| `container` | string | Container name (`nemo-skills` in `grpo/base.yaml`) | Required |

#### Outputs

- `${output_dir}/final_result.jsonl` — Data in Responses API format

#### Resources

- **Compute:** CPU only (no GPU)
- **Runtime:** ~1 min

---

### prepare_data

**Implementation:** shared `nvflow.stages.nemo_gym.CollateDatasetStage`
**Binding:** workflow alias `prepare_data` uses `nemo_gym.collate`

#### Purpose

Run `gym dataset collate` to stamp each JSONL record with a `task_source` field.
Gym resolves the agent from that source and the generated runtime config rather
than coupling data to an `agent_ref`. The stage auto-generates the agent config
artifact from the workflow environment.

#### Inputs

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `output_dir` | path | Output directory for prepared data and overlay | Required |
| `input_dir` | path | Per-environment output from `convert_to_responses_api` | Required |
| `input_filename` | string | Input JSONL filename under each environment | `final_result.jsonl` |
| `gym_path` | path | Path to NeMo-Gym inside the container | Required |
| `container` | string | Container name (`nemo-gym` in `grpo/base.yaml`) | Required |
| `environments` | mapping | Dataset bindings, Gym config paths, and agent instances | Required |
| `mode` | string | `"train_preparation"` | `"train_preparation"` |
| `should_download` | bool | Download missing datasets from HuggingFace | `false` |
| `installation_command` | string | Container setup command | Optional |

#### Modes

- **`train_preparation`**: Produces `train.jsonl`. Only a single `train` dataset is collated here; the train/validation split happens later, in [train_validation_split](#train_validation_split), on reward-filtered data.

#### Agent Configuration

Each entry references the provider-owned Gym config and selects an agent
instance. Gym resolves the implementation and resources-server route:

```yaml
environments:
  equivalence_llm_judge:
    source_data: ${directories.step-3-convert-to-responses-api}/equivalence_llm_judge/final_result.jsonl
    config_paths:
      - resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml
    agent_name: equivalence_llm_judge_simple_agent
```

#### Outputs

- `${output_dir}/train.jsonl` — with `task_source` routing fields

#### Resources

- **Compute:** CPU only
- **Runtime:** ~1 min

---

### collect_rollouts

**Implementation:** shared `nvflow.stages.nemo_gym.CollectRolloutsStage`

#### Purpose

Collect canonical rollout artifacts through native Gym. NVFlow owns workflow
planning, Slurm resources, deterministic outer chunking, dependencies and
completion evidence. Gym owns model/environment lifecycle, repeats, resume,
rollout execution, aggregation, profiling and health checks.

#### Configuration

```yaml
rollout:
  num_repeats: 8
  num_chunks: 2
  dependent_jobs: 0
  num_samples_in_parallel: 64
  max_num_samples: null
  models:
    policy:
      deployment: gym
      model: /hf_models/Qwen/Qwen3-4B
      num_gpus: 2
      gym_model:
        uses_reasoning_parser: true
        vllm_serve_kwargs:
          tensor_parallel_size: 2
          reasoning_parser: qwen3
    judge:
      deployment: gym
      model: /hf_models/openai/gpt-oss-120b
      num_gpus: 2
      gym_model:
        return_token_id_information: false
        uses_reasoning_parser: true
        vllm_serve_env_vars:
          VLLM_RAY_DP_PACK_STRATEGY: strict
        vllm_serve_kwargs:
          data_parallel_size: 1
          tensor_parallel_size: 2
          pipeline_parallel_size: 1
          reasoning_parser: openai_gptoss
```

Optional BLADE analysis runs inside `job-1-analyze-rollouts` before filtering:

```yaml
blade:
  enabled: true
  command: >-
    /opt/nemo_gym_venv/bin/python
    /opt/nemo-gym/.agents/skills/nemo-gym-blade-analysis/scripts/blade_toolkit.py
    validate --benchmark-dir /path/to/blade-package --phase rollouts
```

The hook receives `NVFLOW_ROLLOUTS`, `NVFLOW_MATERIALIZED_INPUTS`,
`NVFLOW_REWARD_PROFILE`, and `NVFLOW_OUTPUT_DIR`. BLADE expects a benchmark
analysis package, so NVFlow does not invent a default command. A
campaign-level default may be overridden or disabled under
`environments.<name>.blade`.

Each role uses the same deployment contract:

- `deployment: gym`: Gym starts `local_vllm_model` inside its allocation.
- `deployment: nvflow`: NVFlow schedules a separate service; choose
  `server_type: vllm` or `server_type: sglang`.
- `deployment: external`: use `base_url`; neither Gym nor NVFlow launches it.

For policy-as-judge, omit `models.judge` and route the environment's Gym-native
`judge_model_server.name` to `policy_model`. This is the same resource-server
reference Gym uses directly, not an NVFlow deployment mode.

For a Gym-deployed policy plus Gym-deployed judge, NVFlow requests one allocation equal to
the sum of both `num_gpus` values. Gym creates independent
`local_vllm_model` instances and Ray assigns each a disjoint GPU bundle.
NVFlow validates that each role's TP × PP × DP topology equals its declared
GPU count. NVFlow-deployed endpoints retain separate heterogeneous resource
groups.

`gym_model` is passed through using Gym's field names for every deployment;
NVFlow only adds the role/type/URL envelope. Gym-deployed model stdout/stderr is
multiplexed into the rollout job log with `(policy_model)` and `(judge_model)`
prefixes. Set `gym_model.show_vllm_engine_stats: true` for detailed metrics; Gym
suppresses them by default.

Project-owned Gym values live under `environments.<name>.gym` in the workflow
instead of maintained overlay files.

#### Execution model

```text
environment × chunk × (dependent_jobs + 1)
```

Gym expands every input into `num_repeats` stable `_ng_rollout_index` values.
Each chained attempt runs the same `gym eval run --resume`; completed
`(_ng_task_index, _ng_rollout_index)` pairs are skipped.

During live submission, NVFlow omits a chunk only after its marker and current
materialized/rollout content hashes validate on the target filesystem, so
verified chunks request no GPU allocation. Static plans still show the full
worst-case topology. The CPU analyze job rechecks source/chunk identity
before it trusts any skipped shard. Partial or invalid chunks remain scheduled
and resume through Gym's row-level logic.

After collection:

1. every chunk marker is content-validated;
2. `gym eval aggregate` publishes canonical rollouts plus environment-specific
   metrics; a single chunk aggregates before its original Gym fleet shuts down,
   while multiple chunks use a CPU-only Gym environment without relaunching
   policy or judge inference servers;
3. `gym eval profile` writes task/repeat reward profiles;
4. `gym eval health-check` writes quality summary and rollout verdicts;
5. optional BLADE analysis reads the unfiltered artifacts;
6. shared dataset selection publishes profiled rows and retains useful reward variance.

#### Outputs

```text
${output_dir}/
├── profiled.jsonl
├── train.jsonl
└── jobs/
    ├── job-0-collect-rollouts/
    │   └── chunks/chunk-N/{outputs,attempts,manifest.json,completion.json}
    ├── job-1-analyze-rollouts/
    │   ├── outputs/                 # Gym-native canonical basenames
    │   ├── analysis/{evaluation,blade,compare}/
    │   ├── logs/
    │   └── completion.json
    └── job-2-build-training-dataset/
        ├── diagnostics/unprofiled_samples.jsonl
        ├── selection_report.json
        ├── logs/
        └── completion.json
```

Changing the source digest, sample limit or chunk count rejects stale resume.
Legacy `output-rsN.jsonl` trees are not resumed.

This is an intentional clean break:

- replace `num_random_seeds` with Gym-owned `num_repeats`;
- replace `policy_vllm` / `judge_vllm` with `models.policy` / `models.judge`;
- move environment Hydra values into `environments.<name>.gym`;
- treat the job-1 rollout bundle plus its validated `completion.json` as the canonical
  consumer contract.

---

### rejudge

`rejudge` remains optional and independently runnable. It recomputes
rewards over existing canonical rollouts without regenerating policy responses
or mutating the collection artifacts.

- Servers declaring `STATELESS` use native `gym eval reverify`.
- `equivalence_llm_judge` and `finance_sec_search` remain on the audited
  fallback because the pinned Gym declares their safety `UNKNOWN`.
- `--force` is never used.
- Output is a separate canonical bundle under step 6, with its own profile,
  health report, source comparison, dataset-selection output and completion
  evidence.
- Judge deployment and rejudge run first; profile/health run in a
  separate CPU analyze job so judge GPUs are released promptly.
- Optional BLADE and native `gym eval compare` run inside that analyze job;
  shared dataset selection remains a separate downstream CPU task.
- Policy trajectories are preserved; new judge captures are written under the
  rejudge output's own `captures/model-calls`.
- Job-0 `behavior.json` binds partial output to judge behavior;
  job-0 and job-1 `completion.json` records separately commit judge and analyzed
  outputs.

To train from recomputed rewards:

```yaml
reward_source_stage: rejudge
pipeline_stages:
  # ...
  - collect_rollouts
  - rejudge
  - train_validation_split
```

The default is `reward_source_stage: collect_rollouts`.
`rejudge.blade` uses the same optional hook and runs over recomputed, unfiltered
rewards before its filter substage. Every rejudge also writes native comparison
reports under `analysis/compare`.

---

### train_validation_split

**Binding:** shared stage type `nemo_gym.train_validation_split` (see [Train with NVFlow's shared stages](../../../../docs/guides/training-stages.md))

#### Purpose

Split each environment's data into training and validation sets using stratified sampling to maintain `question_type` distribution. The same splitter serves the SFT workflow through `data.train_validation_split`. Output files are shuffled for better training, and every input field, including `responses_create_params`, is preserved.

#### Inputs

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input_dir` | path | Directory holding `{environment}/{input_filename}` | Required |
| `input_filename` | string | Per-environment input file | `final_result.jsonl` |
| `output_dir` | path | Output directory; each environment writes `{output_dir}/{environment}/` | Required |
| `environments` | dict | Environments to split | Required |
| `val_ratio` | float | Fraction of data for validation | `0.1` |
| `stratify_by` | string | Field to stratify split on (Finance sets `question_type`) | unset |
| `random_seed` | int | Random seed for reproducibility | `42` |
| `sort_by` / `sort_order` | string | Curriculum ordering of `train.jsonl` instead of shuffling | unset / `desc` |
| `input_contract` | string | Set to `dataset_selection` to validate the upstream task-local completion record before splitting | unset |

For GRPO, the split waits for all active rollout/retry jobs and then validates
the selected training-data marker. A resumed stage with no active upstream job
fails before submission when the input or marker is absent.

#### Outputs

- `${output_dir}/train.jsonl` — Training split
- `${output_dir}/val.jsonl` — Validation split
- `${output_dir}/split.done` — Hashes and row counts committed after both files
  are complete; GRPO training validates it before reading either split

#### Resources

- **Compute:** CPU only (no GPU)
- **Runtime:** ~1 min

---

### training

**Binding:** shared stage type `nemo_rl.grpo` (see [Train with NVFlow's shared stages](../../../../docs/guides/training-stages.md))

#### Purpose

Run GRPO reinforcement learning using NeMo-RL with online NeMo-Gym environment rewards.

Combined training launches at most one dedicated judge service. Every
judge-bearing environment must therefore declare the same canonical
`models.judge` endpoint; NVFlow validates this and routes each environment to
that shared service. Environments without a dedicated judge retain
policy-as-judge routing.

#### Inputs

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `output_dir` | path | Output directory | Required |
| `model_name` | string | HuggingFace model identifier | Required |
| `data_source_dir` | path | Split output; each environment reads `{data_source_dir}/{env_name}/train.jsonl` and `val.jsonl` (set to `step-7-train-validation-split` in `grpo/base.yaml`) | Required |
| `environments` | dict | Environments to train (set in `grpo/base.yaml`) | Required |
| `hf_checkpoint_path` | path | Model weights path | `model_name` |
| `preset` | string | GRPO preset name from `presets_file` (`project://workflows/grpo/grpo_presets.yaml`, set in `grpo/base.yaml`) | Required |
| `backend` | string | `"fsdp"` or `"megatron"` | `"fsdp"` |
| `total_gpus` | int | Total GPUs for training (auto-split across nodes) | `8` in `grpo/base.yaml`; the demo uses 16 (32 for `finance_sec_search`), production 128 |
| `dependent_jobs` | int | Multi-job chaining for long runs | `0` |
| `wandb_project` | string | W&B project name | `"finance-grpo"` (set in `grpo/base.yaml`) |
| `wandb_mode` | string | `"online"`, `"offline"`, or `"disabled"` | `"disabled"` |
| `overrides` | dict | NeMo-RL config overrides (deep-merged with preset) | `{}` |
| `extra_arguments` | string | Additional CLI arguments | Optional |
| `installation_command` | string | Container setup command | Optional |

#### Preset System

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

#### Parallelism Validation

The stage validates parallelism before job submission:

- **Dense models**: `world_size % (TP × PP × CP) == 0`
- **MoE models (FSDP)**: `world_size % (TP × CP × EP) == 0`
- **MoE models (Megatron)**: Validates both regular and expert DP groups
- **Auto-corrections**: Disables sequence parallelism when TP=1

#### Outputs

```
${output_dir}/grpo-{model}-{total_gpus}g-tp{tp}-cp{cp}-seq{seq}k/
├── checkpoints/
│   ├── step_1/
│   └── step_2/
├── training-logs/
└── run_metadata_*.yaml          # Full config for reproducibility
```

The directory name is built from the resolved layout, so `grpo-qwen3-4b-16g-tp2-cp1-seq32k` means 16 GPUs total, TP=2, CP=1 and a 32K sequence budget.

#### Resources

| Config | GPUs | Jobs |
|--------|------|------|
| `qwen3_4b.yaml` (demo) | 16; 32 for `finance_sec_search` | 1 |
| `nemotron_nano_30b_a3b.yaml` (production) | 128 | 5 (`dependent_jobs: 4`) |
| `qwen3_30b_a3b.yaml` (smoke-scale example) | 64 | 3 (`dependent_jobs: 2`) |

---

### eval

**File:** `projects/finance/nvfinance/stages/rl/embedded_eval.py`
**Binding:** workflow alias `eval` selects `GymEmbeddedEvalStage`

#### Purpose

Score GRPO training checkpoints on NeMo-Gym benchmarks. Supports `fsdp` and `megatron` checkpoints, which it converts to Hugging Face format on the cluster. The benchmark catalog, judge and serving profiles are loaded from `eval/embedded.yaml` at runtime; a training config only names a serving preset and the benchmarks to run.

Both SFT and GRPO workflow YAML select this project class for their `eval` alias.

#### Inputs

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `eval_output_dir` | path | Output directory for evaluation results | Required |
| `eval_steps` | list | Checkpoint steps to evaluate | `[]` |
| `eval_data_root` | path | Benchmark data prepared by the evaluation workflow's `eval_data` stage | Required with `eval_steps` |
| `checkpoint_path` | path | Path to training checkpoints | Required |
| `format` | string | Checkpoint format; match the training backend: `"fsdp"` (equivalence demo), `"megatron"` (finance_sec_search demo + production) | backend-dependent |
| `base_model` | path | HF model the checkpoint was trained from; required to convert a Megatron checkpoint | Optional |
| `baseline_model` | path | Baseline model for comparison evaluation | Optional |
| `preset` | string | Serving profile from `eval/eval_presets.yaml` | Required |
| `benchmarks` | list | Catalog entries to run | `[secque, financebench]` |
| `repeats` | int | Override the catalog repeat count for every selected benchmark | Optional |

#### Outputs

```
${eval_output_dir}/
├── step-{N}/
│   └── collect-rollouts/
│       └── {benchmark}/
│           └── jobs/
│               ├── job-0-collect-rollouts/
│               └── job-1-analyze-rollouts/
│                   ├── outputs/                    # rollouts.jsonl, quality_summary.json, ...
│                   ├── analysis/evaluation/metrics.json
│                   ├── completion.json
│                   └── logs/
└── baseline/                              # Baseline model results (if configured)
```

#### Resources

- **Compute:** GPU (inference)
- **Runtime:** Depends on number of benchmarks and checkpoint steps

---

### Pipeline Summary

| Stage | Purpose | Compute | Runtime (demo) |
|-------|---------|---------|----------------|
| validate_questions | Drop broken questions (`finance_sec_search` only in the demo) | CPU + GPU | Varies |
| data_transformation | Normalize SDG data to model-agnostic schema | CPU | ~1 min |
| apply_prompt_template | Apply prompt template + extract expected answer | CPU | ~1 min |
| convert_to_responses_api | Convert to NeMo-Gym Responses API format | CPU | ~1 min |
| prepare_data | Add agent routing fields | CPU | ~1 min |
| collect_rollouts | Rollout collection + analysis | GPU | Varies |
| rejudge | [Optional] Re-judge rollouts | GPU/CPU | Varies |
| train_validation_split | Stratified split into train/val sets | CPU | ~1 min |
| training | GRPO training | GPU | Varies |
| eval | Evaluate checkpoints on finance benchmarks | GPU | ~40 min |

**Total (demo):** about 2 hours for `equivalence_llm_judge` and about 5 hours for
`finance_sec_search`, most of it in its multi-turn rollouts.

### Shared Infrastructure

#### Integration Helper Modules (`nvflow/integrations/nemo_gym/`)

Reusable Gym integration utilities are split by concern; `native_contract.py`
defines the `jobs/job-N-*` output layout that the stages above write.

| Function | Module | Purpose |
|----------|--------|---------|
| `ModelEndpoint` | `contracts.py` | Validate one policy/judge endpoint schema |
| `render_gym_command` | `native_commands.py` | Render the pinned native Gym CLI contract |
| `build_rollout_jobs` | `native_rollout.py` | Build collect/analyze/dataset job groups |
| `build_reward_jobs` | `native_rewards.py` | Build rejudge/analyze/dataset job groups |
| `validate_rollout_completion` | `native_sharding.py` | Validate a chunk's completion record against its content and behavior |

#### Standalone Scripts

Stdlib-only Python scripts that run inside Slurm containers:

| Script | Location | Purpose |
|--------|----------|---------|
| `dataset_transformer.py` | `utils/shared/` | Normalize SDG data to model-agnostic schema |
| `prompt_template_applier.py` | `utils/rl/` | Apply prompt templates and extract expected answers |
| `responses_api_converter.py` | `utils/rl/` | Convert to NeMo-Gym Responses API format |
| `dataset_splitter.py` | shared `nvflow.stages.data` | Stratified train/validation split |
| `eval_aggregate.py` | `utils/rl/` | Render finance metrics from Gym artifacts |
| `reward_profiles.py` | shared Gym integration | Normalize and select profiled data |

See [GRPO Workflow](../workflows/06-grpo.md) for usage examples and configuration details.
