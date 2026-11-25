# Evaluation

## Purpose

Evaluate fine-tuned models and baselines on financial reasoning benchmarks.

Evaluation runs in two modes, both on **NeMo-Gym**: benchmark data, prompts, tools and judges all come from the Gym container.

- **Baseline evaluation** — standalone campaigns (`eval/baselines.yaml` and its siblings) that score released models for comparison.
- **Checkpoint evaluation** — embedded as the last stage of SFT and GRPO training pipelines, so checkpoints are scored as soon as training finishes.

vLLM serves HuggingFace directories, but training writes Megatron or FSDP shards. Both modes bridge that themselves: set `format` to match your training backend and each checkpoint is converted before being scored, skipping any that is already converted. In a training pipeline see [Eval Stage Configuration](#eval-stage-configuration-in-training-yamls); in a campaign see [Scoring checkpoints](#scoring-checkpoints).

## Pipeline Flow

```
SFT Pipeline                          GRPO Pipeline
┌──────────────────────┐              ┌──────────────────────┐
│ data_transformation  │              │ validate_questions   │
│ prepare_for_sft      │              │ data_transformation  │
│ train_validation_split│             │ apply_prompt_template│
│ sequence_length_group│              │ convert_to_responses_api│
│ training             │              │ prepare_data         │
│ ─────────────────────│              │ collect_rollouts     │
│ eval  ← checkpoints  │              │ train_validation_split│
│   + baseline         │              │ training             │
└──────────────────────┘              │ ─────────────────────│
                                      │ eval  ← checkpoints  │
Standalone Baselines (NeMo-Gym)       │   + baseline         │
┌──────────────────────┐              └──────────────────────┘
│ eval_data            │
│ qwen3-4b             │
│ qwen3-14b            │
│ qwen3-32b            │
│ gpt-oss-120b         │
│ nemotron-3-nano-30b  │
└──────────────────────┘
```

Each baseline stage fans out into one job per (model, benchmark) pair. Narrow a run with `-e <benchmark>`.

> **Offline clusters:** the data-preparation stages download the benchmark datasets from Hugging Face and GitHub on their first run, so that run needs network on the compute nodes. They clear the HF offline flags themselves. The datasets persist and are reused afterwards.

## Configuration

**Directory:** `workflows/eval/`

| File | Purpose |
|------|---------|
| `eval/base.yaml` | Shared NeMo-Gym settings: benchmark catalog, judge, rollout defaults, prep stages. Not runnable |
| `eval/eval_presets.yaml` | How each model is served: weights, GPUs, context length, parsers. Not runnable |
| `eval/baselines.yaml` | The full matrix, graded by the hosted judge. **Add models here** |
| `eval/demo.yaml` | Three small models, locally served judge — QA needs no API key |
| `eval/embedded.yaml` | What training pipelines score checkpoints against: the catalog with a local QA judge. Not runnable |
| `eval/smoke.yaml` | One model, one benchmark, two chunks × two repeats |
| `eval/shared-gym-smoke.yaml` | Full SecQue + FABv1, two-repeat acceptance eval using the standard Qwen3-4B policy preset and shared external GPT-5.2 judge |

`baselines.yaml`, `demo.yaml`, `smoke.yaml`, and `shared-gym-smoke.yaml` are
runnable; they differ in roster, benchmark scope, and judge.
`baselines.yaml` and `demo.yaml` both cover the full benchmark catalog, so
narrowing a run is a `-e` flag rather than another
config file.

The demo's ordered stages are `eval_data`, `qwen3-4b`,
`gemma-3-4b-it`, and `gpt-oss-20b`.

Every model/benchmark pair passes through the same shared Gym environment
normalizer used by `nemo_gym.collect`. The provider file named by
`config_paths` remains authoritative for agent and resource topology; this
workflow supplies datasets, endpoint ownership, execution settings, and
run-specific leaf overrides without deleting provider server definitions.

`fabv1` reads the SEC filing-metadata cache that the matching
download scope's `prefetch_cache` writes: `finance:download/sp500` for
`baselines.yaml`, `finance:download/demo` for `demo.yaml`. Run it once before an
agentic campaign. The QA benchmarks need nothing downloaded.

`baselines.yaml` and `demo.yaml` write to different output trees, so the same
model can appear in both without their results colliding.

Checkpoint evaluation is configured directly in the training configs:

| File | Eval Config |
|------|-------------|
| `sft/qwen3_4b.yaml` | `stages.eval` with `eval_steps: [10]` |
| `sft/qwen3_14b.yaml` | `stages.eval` with `eval_steps: [2600, 5000, 7408]` |
| `grpo/qwen3_4b.yaml` | `stages.eval` with `eval_steps: [10]`; `environments.finance_sec_search.eval` applies with `-e finance_sec_search` |

## Usage

### Training + Evaluation (Recommended)

Run training and evaluation together — eval runs automatically after training:

```bash
# SFT training + checkpoint eval
uv run nvflow run --all --config projects/finance/workflows/sft/qwen3_4b.yaml --cluster my_cluster

# GRPO training + checkpoint eval, one environment per run (quick start, Step 5)
uv run nvflow run --all --config projects/finance/workflows/grpo/qwen3_4b.yaml --cluster my_cluster -e finance_sec_search
```

### Eval Stage Only

Run just the eval stage from a training config:

```bash
# Evaluate SFT checkpoints only (skip training)
uv run nvflow run eval --config projects/finance/workflows/sft/qwen3_14b.yaml --cluster my_cluster
```

### Baseline Evaluation

Evaluate all baseline (pre-trained) models:

```bash
# All baselines
uv run nvflow run --all --config projects/finance/workflows/eval/baselines.yaml --cluster my_cluster

# Specific baseline
uv run nvflow run gpt-oss-120b --config projects/finance/workflows/eval/baselines.yaml --cluster my_cluster

# One model on one benchmark
uv run nvflow run gpt-oss-120b --config projects/finance/workflows/eval/baselines.yaml --cluster my_cluster -e fabv1
```

## Adding a Model

A model is defined once in `eval/eval_presets.yaml` — weights, GPU count, context
length, sampling, parsers — and then named from a campaign. Adding a model to a
run is two lines if the preset already exists.

`baselines.yaml` ships several larger reference models commented out for exactly
this reason. To enable one, uncomment it in both `models:` and `pipeline_stages:`.

### Step 1: Add a preset (skip if the model already has one)

```yaml
# eval_presets.yaml
model_presets:
  my_model_8b:
    model: /hf_models/MyOrg/MyModel-8B
    num_gpus: 4
    gym_model:
      uses_reasoning_parser: true
      vllm_serve_kwargs:
        reasoning_parser: qwen3
        max_model_len: 40960
    responses_create_params:
      temperature: 0.6
      top_p: 0.95
      max_output_tokens: 16384
```

An entry is referenced whole: a campaign cannot merge extra keys into it, because
YAML's `<<:` merge needs an anchor and does not work on an interpolation. Anything
that varies by benchmark belongs in the preset's `benchmark_models` so the
model is served identically everywhere it appears. A model that genuinely needs
two different profiles wants two entries.

To run the **agentic** benchmarks, the preset also needs a `tool_call_parser` that
vLLM ships for the model's tool format, declared per benchmark:

```yaml
    benchmark_models:
      fabv1:
        gym_model:
          vllm_serve_kwargs:
            enable_auto_tool_choice: true
            tool_call_parser: hermes
```

If the model's context window is shorter than the benchmark asks for, add
rope-scaling in the same place — see the Qwen3 entries, which enable YaRN for the
agentic benchmarks only.

### Step 2: Name it from a campaign

<!-- doc-check: yaml-excerpt -->
```yaml
# baselines.yaml
models:
  my-model-8b:
    dependencies: *agentic_deps      # [eval_data] if QA only
    model: ${model_presets.my_model_8b}

pipeline_stages:
  - my-model-8b
```

A model with no tool-call parser is scoped to the QA benchmarks by giving it
`environments: *qa_only` alongside `dependencies: [eval_data]`.

### Step 3: Run it

```bash
uv run nvflow run my-model-8b --config projects/finance/workflows/eval/baselines.yaml --cluster my_cluster
```

The stage name is the key under `models:`, and it fans out into one Slurm job per
benchmark in scope.

### Scoring checkpoints

A campaign can also score training checkpoints, under `checkpoints:` instead of
`models:`. Each step is converted to HuggingFace and then evaluated exactly like
a released model:

```yaml
checkpoints:
  my-sft-run:
    dependencies: [eval_data]
    path: /workspace/outputs/.../model-my-run-8g-tp2-pp1-cp2-seq32k
    format: megatron              # or "fsdp"
    base_model: /hf_models/Qwen/Qwen3-4B
    eval_steps: [25, 50, 75]
    model: ${model_presets.my_model_8b}

pipeline_stages:
  - my-sft-run                    # expands to one stage per step
```

`path` is the run directory holding `checkpoints/step_<n>`. The shared
`nvflow.integrations.nemo_rl.checkpoints` helpers plan the HuggingFace
conversion; exports and conversion logs are written beside it under
`hf_models/`, so it must be writable — to score someone else's run, point
`path` at a directory of your own containing symlinks to the steps you want.

`base_model` supplies the tokenizer and architecture, and is required for
Megatron. `model` is the serving profile: a fine-tune is served like its base,
so the base model's preset applies unchanged and only the weights are swapped
for the converted step.

Conversion is idempotent — a rerun reuses an export that is already there — and
each step's results land in their own `<run>-<step>` subtree.

## Evaluation Benchmarks

The baseline (NeMo-Gym) catalog is in `eval/base.yaml`. Benchmark keys are what you pass to `-e`:

| Key | Benchmark | Notes |
|-----|-----------|-------|
| `secque` | SEC filing comprehension (565 samples) | single-turn QA, equivalence judge |
| `financebench` | Financial question answering (150 samples) | single-turn QA, equivalence judge |
| `fabv1` | Vals AI finance-agent v1.1 | multi-turn SEC tool use; public 50-question set |

The agentic benchmarks need a model-specific `tool_call_parser`, so a model only runs them once its parser is declared. Models without one are scoped to the QA benchmarks; see the opt-in comment at the top of `eval/baselines.yaml`.

### fabv1 tool sets

`fabv1` can offer different tools, and the choice is baked into the prepared data rather than applied at rollout time — so changing it re-prepares the dataset. One tool set runs per invocation, chosen by `fabv1_variant`:

| `fabv1_variant` | Tools |
|---|---|
| `noweb` (default) | `sec_filing_search` |
| `web` | `sec_filing_search` + `web_search` |

Set it in the campaign config; the table of variants lives in `eval/base.yaml`. `web_search` needs `TAVILY_API_KEY`, and a tool whose key is unset reports itself unavailable to the model instead of failing the run — so a missing key shows up as a lower score, not an error.

The chosen tool set names the dataset directory and the output subtree (via the benchmark's `output_name`), so combinations never overwrite or resume each other, and `projects/finance/scripts/eval_metrics.py` gives each its own column when several are present under one root.

Embedded checkpoint eval defaults to the QA benchmarks (SEC-QUE and FinanceBench),
scored against `eval/embedded.yaml` — the catalog with a locally served QA judge,
so a training pipeline needs no hosted-API key.

## Eval Stage Configuration (in Training YAMLs)

Add to your SFT or GRPO model config:

```yaml
stages:
  eval:
    eval_steps: [1000, 3000, 5000]
    eval_data_root: /workspace/outputs/finance/baselines/step-1-eval-data  # prepared by eval/baselines
    checkpoint_path: ${directories.step-4-training}/model-name
    format: megatron        # Match the checkpoint's training backend: "fsdp" or "megatron"
    base_model: ${hf_model_path}
    baseline_model: /hf_models/Qwen/Qwen3-14B
    preset: qwen3_14b       # serving profile from eval/eval_presets.yaml
    benchmarks: [secque, financebench]
```

### Choosing a judge

The judge is a property of the campaign, so that a checkpoint and a baseline are
always graded the same way. `campaign` selects one:

```yaml
stages:
  eval:
    campaign: base.yaml     # default: embedded.yaml (local QA judge)
```

`base.yaml` grades every benchmark with the hosted judge, which needs an API key
in the job environment — see [Hosted Model
Endpoints](../../../../docs/hosted-endpoints.md) for both backends and for OpenAI. Point `campaign` at any campaign in
`eval/`, including one of your own (name it `private_*.yaml` so git ignores it
and `run` accepts it), to score checkpoints against a benchmark set the default
does not cover.

Keys the stage does not recognise are rejected at submission rather than
ignored, so a stale setting cannot silently change how a checkpoint is scored.
Configs written against the previous nemo-skills stage set five keys that have
moved:

| Retired key | Where it lives now |
| --- | --- |
| `judge` | `campaign`, as above |
| `server_type`, `gpus`, `server_args`, `inference_args` | the serving preset named by `preset`, in `eval/eval_presets.yaml` |

The eval stage will:
1. Read the benchmark data that the evaluation campaign's `eval_data` stage
   prepared, from `eval_data_root`
2. Convert each checkpoint to HuggingFace format when needed, skipping the
   conversion if the HF model already exists
3. Evaluate each checkpoint at the specified steps
4. Evaluate the baseline model for comparison

### Rejudge existing eval rollouts

`rejudge` is an optional first-class stage. It changes the judge without
rerunning the policy, tools, or environment:

```yaml
stages:
  rejudge:
    rollouts_dir: /workspace/outputs/finance/smoke/qwen3-4b/collect-rollouts
    output_dir: /workspace/outputs/finance/smoke/qwen3-4b/rejudge/gpt-5
    environments:
      secque: ${benchmarks.secque}
    verification:
      mode: fallback
      models:
        judge:
          deployment: external
          model_type: openai_model
          model: gpt-5
          base_url: https://api.openai.com/v1
          api_key_env: OPENAI_API_KEY
```

Run it for the whole configured benchmark set or select one:

```bash
uv run nvflow run rejudge --workflow finance:eval/private_campaign \
  --cluster my_cluster -e secque
```

Use `mode: native` only when Gym declares the verifier `STATELESS`; maintained
finance verifiers currently use the audited `fallback`. Both modes call the
shared rejudge implementation and publish separate aggregate/profile/health
artifacts. Judge execution is followed by `job-1-analyze-rollouts`, which regenerates
`analysis/evaluation`, runs configured BLADE analysis, and compares the source
and rejudged bundles with native `gym eval compare` under `analysis/compare`.
Original policy trajectories are preserved, while new judge captures stay under
the rejudge output. Job-0 `behavior.json` prevents partial output from resuming
under changed judge behavior; job 0 and job 1 publish separate task-local
`completion.json` records.

### Optional BLADE analysis

BLADE analyzes completed eval rollouts after aggregate/profile/health. Set one
campaign default, then override or disable it per benchmark:

```yaml
blade:
  enabled: true
  command: blade-tool analyze ...

benchmarks:
  secque:
    blade:
      command: blade-tool analyze --benchmark secque ...
  fabv1:
    blade:
      enabled: false
```

Each enabled model/benchmark pair runs BLADE inside `job-1-analyze-rollouts`,
before any downstream dataset publication. BLADE failure prevents job-1
completion; rerunning the stage reuses completed chunk or rejudge outputs and
retries analysis.

### Compare completed runs

Every rejudge analyze job automatically compares its source and rejudged bundles
with native `gym eval compare`. The standalone `compare` CPU stage remains
available for any other completed pair. The pinned Gym version accepts one
baseline and one candidate:

```yaml
stages:
  compare:
    baseline_rollouts: /workspace/outputs/baseline/secque/jobs/job-1-analyze-rollouts/outputs/rollouts.jsonl
    baseline_completion: /workspace/outputs/baseline/secque/jobs/job-1-analyze-rollouts/completion.json
    candidate_rollouts: /workspace/outputs/candidate/secque/jobs/job-1-analyze-rollouts/outputs/rollouts.jsonl
    candidate_completion: /workspace/outputs/candidate/secque/jobs/job-1-analyze-rollouts/completion.json
    output_dir: /workspace/outputs/comparisons/secque
    container: nemo-gym
    report_format: both
```

```bash
uv run nvflow run compare --workflow finance:eval/private_campaign \
  --cluster my_cluster
```

Both runs must use compatible tasks, agents, input identity, and
aggregate-metric semantics. Standalone compare remains separate from rollout
collection so an operator can choose the baseline after both runs complete.

## Output Structure

Checkpoint evaluations output alongside the training results:

```
outputs/finance/demo/workflow-4-sft/qwen3_4b/
├── step-4-training/          # Training checkpoints
└── step-5-eval/
    ├── step-10/
    │   └── collect-rollouts/{benchmark}/jobs/job-1-analyze-rollouts/analysis/evaluation/metrics.json
    └── baseline/
        └── collect-rollouts/{benchmark}/jobs/job-1-analyze-rollouts/analysis/evaluation/metrics.json
```

Baseline evaluations output under the campaign's `data_root`, one subtree per
(model, benchmark):

```
outputs/finance/baselines/
└── {model}/collect-rollouts/{benchmark}/
    └── jobs/
        ├── job-0-collect-rollouts/
        └── job-1-analyze-rollouts/
            ├── outputs/
            ├── analysis/evaluation/{metrics.json,summary.txt}
            ├── completion.json
            └── logs/
```

`{benchmark}` is the catalog key, unless the benchmark sets `output_name` to keep
its variants apart — `fabv1` does, writing to `fabv1_noweb` or `fabv1_web`.

The shared-stage acceptance run keeps its benchmark data and rollouts under its
run root:

```
outputs/finance/shared-gym-suite-smoke/
├── step-1-eval-data/{secque,fabv1_noweb}/eval.jsonl
└── qwen3-4b/collect-rollouts/{secque,fabv1_noweb}/
```

To collate a whole campaign into one CSV — a column per model, a row per metric:

```bash
uv run python projects/finance/scripts/eval_metrics.py outputs/finance/baselines -o baselines.csv
```

## Adding Checkpoint Eval to a Training Config

Embedded checkpoint evaluation uses the same native Gym data plane as standalone
evaluation.

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
    eval_data_root: /workspace/outputs/finance/baselines/step-1-eval-data  # prepared by eval/baselines
    checkpoint_path: ${directories.step-4-training}/model-my-model-name
    format: megatron        # Match the checkpoint's training backend: "fsdp" or "megatron"
    base_model: ${hf_model_path}
    baseline_model: /hf_models/MyOrg/MyModel
    preset: my_model        # add an entry to eval/eval_presets.yaml first
```

### Step 3: Run

```bash
uv run nvflow run --all --config projects/finance/workflows/sft/private_my_model.yaml --cluster my_cluster
```

## Common Issues

### Model checkpoint not found

**Solution:**
- Verify `checkpoint_path` matches the actual training output directory
- Check if training completed successfully
- Use absolute paths in config

### OOM during evaluation

**Solution:**
- Reduce `gym_model.vllm_serve_kwargs.max_model_len` in the preset
- Raise the preset's `num_gpus` for larger models; the shipped presets use at
  least 4, and an NVFlow-deployed server spreads the model over all of them
  (`num_gpus` × `server_nodes`) with tensor parallelism

### Low scores on all models

**Solution:**
- Check evaluation data quality
- Verify judge prompt is appropriate
- Review prediction outputs manually

## Technical Details

For comprehensive stage-by-stage documentation:
- **[Eval Stages Reference](#detailed-stage-reference)**

For benchmark details, see the public Hugging Face datasets behind the QA
benchmarks, which `eval_data` prepares through Gym's `secque` and
`financebench` benchmarks:
- FinanceBench: [PatronusAI/financebench](https://huggingface.co/datasets/PatronusAI/financebench)
- SEC-QUE: [nogabenyoash/SecQue](https://huggingface.co/datasets/nogabenyoash/SecQue)

## Detailed stage reference
Technical reference for the evaluation stages.

### Quick Navigation

- [eval_data](#eval_data) (standalone baselines workflow)
- [Baseline Models](#baseline-model-stages) (standalone baselines workflow)
- [Embedded Eval Stage](#embedded-eval-stage) (SFT and GRPO pipelines)
- [Rejudge Existing Rollouts](#rejudge-existing-eval-rollouts) (optional)
- [BLADE Analysis](#optional-blade-analysis) (optional per benchmark)
- [Compare Completed Runs](#compare-completed-runs) (optional)

---

### eval_data

**Implementation:** shared `nvflow.stages.nemo_gym.PrepareBenchmarkStage`
**Binding:** workflow alias `eval_data` uses `nemo_gym.prepare_benchmark`

#### Purpose

Prepare benchmark datasets with the Gym CLI. Each benchmark is downloaded and
converted, then either rendered through the project's own prompt template (the QA
benchmarks) or copied as-is (the agentic benchmarks, which bake prompts and tool
schemas in at prepare time).

The job is idempotent: re-running when the data already exists is a fast no-op.

#### Inputs

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `benchmarks` | list | Entries naming a Gym benchmark selector (`<name>` or `<name>/<config>`) | Required |
| `output_dir` | path | Directory to write prepared JSONL into | Required |

#### Supported Benchmarks

**SEC-QUE** (565 samples):
- **Analysis** (72): Qualitative analysis of financial information
- **Comparison** (220): Cross-company or temporal comparisons
- **Ratio** (188): Financial ratio calculations and interpretation
- **Risk** (85): Risk factor identification and analysis

**FinanceBench** (150 samples):
- **metrics-generated** (50): Standardized financial metrics
- **domain-relevant** (50): Domain-specific financial questions
- **novel-generated** (50): Novel question generation

#### Resources

- **Compute:** CPU only (HuggingFace download + format conversion)
- **Runtime:** ~3 min (downloads both benchmarks, caches for future runs)

---

### Baseline Model Stages

These stages evaluate baseline (non-fine-tuned) models. Their project class is
selected by workflow-local `uses` and they run via `eval/baselines.yaml`.

Workflow expansion applies the base file's project-class `uses` coordinate to
each entry in the campaign's `models` section.

#### Available Baselines

| Stage | Model | GPUs |
|-------|-------|------|
| `qwen3-4b` | Qwen3-4B | 4 |
| `qwen3-14b` | Qwen3-14B | 4 |
| `qwen3-32b` | Qwen3-32B | 4 |
| `gpt-oss-120b` | GPT-OSS-120B | 4 |
| `nemotron-3-nano-30b` | Nemotron-3-Nano-30B | 4 |

#### Usage

```bash
# All baselines
uv run nvflow run --all --config projects/finance/workflows/eval/baselines.yaml --cluster my_cluster

# Specific baseline
uv run nvflow run gpt-oss-120b --config projects/finance/workflows/eval/baselines.yaml --cluster my_cluster
```

---

### Embedded Eval Stage

**File:** `projects/finance/nvfinance/stages/rl/embedded_eval.py`
**Class:** `GymEmbeddedEvalStage`
**Binding:** workflow alias `eval` selects `GymEmbeddedEvalStage`

#### Purpose

Scores training checkpoints on NeMo-Gym benchmarks, embedded as the last stage of
SFT and GRPO training pipelines.

Checkpoints run through the same rollout matrix as the standalone baseline
campaigns, so a checkpoint and a baseline are directly comparable. The benchmark
catalog, judge and serving profiles come from `eval/embedded.yaml`, which grades the
QA benchmarks with a locally served judge — a training pipeline therefore needs
no hosted-API credentials.

#### Configuration (in training YAMLs)

```yaml
stages:
  eval:
    eval_steps: [2600, 5000, 7408]
    eval_data_root: /workspace/outputs/finance/baselines/step-1-eval-data  # prepared by eval/baselines
    checkpoint_path: ${directories.step-4-training}/model-name
    format: megatron        # Match the checkpoint's training backend: "fsdp" or "megatron"
    base_model: ${hf_model_path}
    baseline_model: /hf_models/Qwen/Qwen3-14B
    preset: qwen3_14b
    benchmarks: [secque, financebench]
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `eval_steps` | list[int] | Training steps to evaluate |
| `eval_data_root` | path | Benchmark data prepared by the evaluation workflow's `eval_data` stage, such as `outputs/finance/demo/workflow-1-baseline-eval/step-1-eval-data`; required with `eval_steps` |
| `checkpoint_path` | path | Base path to training checkpoints |
| `format` | str | Must match the checkpoint's training backend: `"megatron"` (SFT, finance_sec_search GRPO, production) or `"fsdp"` (equivalence_llm_judge GRPO demo) |
| `base_model` | path | HF model the checkpoint was trained from; required to convert a Megatron checkpoint |
| `baseline_model` | path | HF model path for baseline comparison |
| `preset` | str | Serving profile from `eval/eval_presets.yaml` |
| `benchmarks` | list[str] | Catalog entries to run; defaults to `[secque, financebench]` |
| `repeats` | int | Override the catalog repeat count for every selected benchmark |

#### Behavior

1. Reads the benchmarks at `eval_data_root`. It prepares no data itself, so run
   the evaluation workflow's `eval_data` stage first.
2. For each step in `eval_steps`, submits a conversion job, then evaluates the
   converted HF model. Conversion skips itself if the HF model already exists.
3. Evaluates the `baseline_model` for comparison, when set.

#### Output Structure

```
${eval_output_dir}/
├── step-2600/collect-rollouts/{benchmark}/
├── step-5000/collect-rollouts/{benchmark}/
├── step-7408/collect-rollouts/{benchmark}/
└── baseline/collect-rollouts/{benchmark}/
```

---

### Common Evaluation Parameters

Both the standalone campaigns and embedded eval submit rollouts through the same
NeMo-Gym path, configured by the benchmark catalog in `eval/base.yaml`.

#### Core Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `data_root` | path | Root of the rollout output tree |
| `environments` | dict | Catalog subset a campaign runs |
| `rollout.num_repeats` | int | Default rollouts per task; each catalog benchmark's `repeats` overrides it |
| `rollout.num_chunks` | int | NVFlow's outer Slurm chunks |

#### Judge endpoint configuration (`models.judge`)

| Parameter | Description |
|-----------|-------------|
| `deployment` | `gym` (Gym launches local vLLM), `nvflow` (NVFlow launches a service), or `external` (use `base_url`; launch nothing) |
| `server_type` | `vllm` or `sglang`; only for `deployment: nvflow` |
| `model_type` | Gym model component type; for example `local_vllm_model`, `vllm_model`, or `openai_model` |
| `model` | Judge model identity or path |
| `base_url` / `api_key_env` | Existing endpoint location and optional key; only for `deployment: external` |
| `num_gpus` | GPUs reserved for a Gym- or NVFlow-deployed endpoint |
| `gym_model` | Gym-native model component fields, including request/response and vLLM settings |

To use the policy as judge, omit `models.judge` and set the environment's
Gym-native `judge_model_server.name` to `policy_model`. Policy reuse is an
environment route, not a fourth deployment type.

---

### Metrics

The evaluation pipeline computes multiple metrics per benchmark, aggregated across repeats.

#### Primary Metrics

- **pass@1 (judge_correct)**: Percentage of examples accepted by the configured judge
- **predicted_answer_ids**: Number of examples with valid answer extraction
- **missing_answers**: Number of examples with no answer extracted

#### Production Results (SEC-QUE Benchmark)

| Model | judge_correct | Notes |
|-------|---------------|-------|
| **Baselines:** | | |
| Qwen3-14B | 59.7% | Baseline model |
| Qwen3-32B | 61.8% | Larger baseline |
| GPT-OSS-120B | 61.9% | Best baseline |
| **Fine-tuned (Qwen3-14B-48k):** | | |
| Step 2600 | 70.3% | +10.6% vs baseline (end of epoch 1) |
| Step 5000 | 70.3% | Mid-training |
| Step 7408 | 71.0% | +11.3% vs baseline (final) |

---

### Resources per Stage

| Model Size | GPUs | Memory/GPU | Runtime (2 benchmarks × 5 repeats) |
|------------|------|------------|----------------------------------|
| 7-14B | 4 | 40-80GB | 1-2 hours |
| 32B | 4 | 80GB | 2-3 hours |
| 120B | 4 | 80GB | 3-4 hours |

GPUs is the preset's `num_gpus` in `eval/eval_presets.yaml`. An NVFlow-deployed
server uses all of them for tensor parallelism. The larger reference models
there use 8 GPUs or more.

> **Note:** The hosted judge runs over an HTTP API, so it needs no local GPU.

---

### Common Issues

#### Checkpoint Conversion Failures

**Symptom:** Conversion job fails or HF model not created

**Solutions:**
- Check the checkpoint exists at the specified path
- Verify `base_model` points at the HF model the checkpoint was trained from
- Check conversion logs
- Ensure `conversion.num_gpus` clears the cluster's per-job GPU minimum

#### Low Scores

**Check:**
- **Serving preset**: confirm the `preset` matches the model family, so the chat
  template and reasoning parser are right
- **Output budget**: `max_output_tokens` must leave room for the prompt inside
  `max_model_len`, or vLLM rejects every request and each rollout scores zero
- **Judge**: a local judge needs `override_generation_config` to pin temperature

#### vLLM Server Crashes / OOM

**Solutions:**
- Reduce `gym_model.vllm_serve_kwargs.max_model_len` in the preset
- Raise `num_gpus` for larger models
- Check GPU memory: 40GB+ required for 14B models with long context

See [Eval Workflow](../workflows/05-eval.md) for usage examples and detailed configuration.
