# Document-Grounded SDG

## Purpose

Generate high-quality financial Q&A pairs directly from SEC filing documents with built-in question verification, multi-seed answer evaluation, and per-stage field trimming.

> **Note:** This workflow generates ~800K Q&A pairs in a single `final_result.jsonl`. The previous difficulty-stratified outputs (`full_data.jsonl`, `hard_rl_data.jsonl`) and the `difficulty_estimation` stage have been removed; downstream SFT / GRPO workflows read `final_result.jsonl` directly. For the production template-based pipeline, see [Template-Based SDG](02-template-based-sdg.md).

## Prerequisites

- SEC filings downloaded ([Download SEC Filings](01-download-sec.md))
- Will be preprocessed in Stage 0 (dg_sdg_preprocess)

## Key Differences from Template-Based

| Aspect | Template-Based | Document-Grounded |
|--------|----------------|-------------------|
| **Question Source** | Seed questions | Generated from documents |
| **Verification** | None | Built-in verification step |
| **Quality Control** | GenSelect + Filter | GenSelect + Evaluation + Aggregation |
| **Output** | Single dataset | Single `final_result.jsonl` (no stratification) |

## Pipeline Flow
```
┌──────────────────────────────┐
│ 0. dg_sdg_preprocess         │  Preprocessing: SEC HTML → Chunked JSONL
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 1. generate_verified_questions│ Q-pipeline: prep + Q-gen + verify-prep + Q-verify
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 2. generate_answers          │  A-pipeline: a-prep (threshold filter) + A-gen
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 3. gym_genselect_answers     │  Selection: Best answer from candidates
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 4. evaluate_answers          │  Evaluation: Quality scoring (multi-seed)
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 5. aggregate_answers         │  Aggregation: Combine evaluation results
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 6. dgsdg_post_process        │  Output: Cleaned + renamed → final_result.jsonl
└──────────────────────────────┘
```

## 7 Stages (Overview)

0. **dg_sdg_preprocess**: Preprocess SEC filings (chunk HTML → create JSONL data following SecQue distribution)
1. **generate_verified_questions**: Generate questions from documents and verify them (4 internal sub-steps: q-prep + Q-gen + verify-prep + Q-verify)
2. **generate_answers**: Filter questions by verification pass-rate, generate N candidate answers (2 internal sub-steps: a-prep + A-gen)
3. **gym_genselect_answers**: Select best answer from multiple candidates
4. **evaluate_answers**: Evaluate answer quality (multi-seed for robustness)
5. **aggregate_answers**: Aggregate evaluation results
6. **dgsdg_post_process**: Clean + rename fields, emit single `final_result.jsonl` consumed by downstream SFT / GRPO

**See [technical reference](#detailed-stage-reference) for detailed stage documentation.**

## Configuration

**Production:** `workflows/sdg/document-grounded-sdg-sp500.yaml`, the SDG workflow
of the `finance:document-grounded-local-search-grpo` recipe. It inherits
`document-grounded-sdg-qwen-prod.yaml` and:

- reads the full S&P 500 download
  (`/workspace/outputs/finance/sp500/workflow-2-download-sec/step-0-download`)
  and draws 150,000 samples from 10-K, 10-Q and 8-K filings;
- generates questions and answers with Qwen3.5-397B-A17B, served on 2 nodes × 8 GPUs;
- verifies questions, selects answers and evaluates them with GPT-OSS-120B on
  1 node × 8 GPUs, keeping the judge separate from the generator to avoid
  self-preference bias;
- runs 3 repeats for question verification, answer generation and evaluation,
  and keeps questions with a verification pass rate of at least 0.5;
- writes to `/workspace/outputs/finance/sp500/workflow-3-document-grounded-sdg`.

**Base config:** `workflows/sdg/document-grounded-sdg.yaml` holds the stage
wiring the variants inherit. On its own it reads and writes demo-scope paths
under `/workspace/outputs/finance/demo/`, generates with GPT-OSS-120B, and
verifies, selects and evaluates with Qwen3-235B-A22B. The small demo runs are
`document-grounded-sdg-demo.yaml` and `document-grounded-sdg-qwen-demo.yaml`.

## Usage

### Run Complete Workflow

```bash
uv run nvflow run --all --config projects/finance/workflows/sdg/document-grounded-sdg-sp500.yaml --cluster my_cluster
```

### Run Individual Stages

```bash
CONFIG=projects/finance/workflows/sdg/document-grounded-sdg-sp500.yaml

# Stage 0: Preprocess SEC filings
uv run nvflow run dg_sdg_preprocess --config $CONFIG --cluster my_cluster

# Stage 1: Generate + verify questions
uv run nvflow run generate_verified_questions --config $CONFIG --cluster my_cluster

# Stage 2: Generate candidate answers
uv run nvflow run generate_answers --config $CONFIG --cluster my_cluster

# Stage 3: Select best answers
uv run nvflow run gym_genselect_answers --config $CONFIG --cluster my_cluster

# Stage 4: Evaluate answers
uv run nvflow run evaluate_answers --config $CONFIG --cluster my_cluster

# Stage 5: Aggregate results
uv run nvflow run aggregate_answers --config $CONFIG --cluster my_cluster

# Stage 6: Post process
uv run nvflow run dgsdg_post_process --config $CONFIG --cluster my_cluster
```

## Output Structure

```
${base_data_dir}/
├── step-0-preprocess/
│   ├── chunks/                          # Chunked HTML files (markdown, clean HTML, original)
│   ├── csv_lists/                       # CSV manifests of chunks
│   └── jsonl/
│       ├── 10-k-data.jsonl              # Sampled 10-K data
│       ├── 10-q-data.jsonl              # Sampled 10-Q data
│       └── 8-k-data.jsonl               # Sampled 8-K data (production)
├── step-1-questions/
│   ├── generate_input.jsonl             # Q-prep output
│   ├── generated/                       # Generated questions
│   ├── verify_input.jsonl               # Q-verify-prep output
│   └── verified/                        # Verified questions (consumed by step-2)
├── step-2-answers/
│   ├── answer_input.jsonl               # A-prep output (threshold-filtered)
│   └── generated/                       # Generated answers (consumed by step-3)
├── step-3-genselect/
│   └── selected_answers.jsonl
├── step-4-evaluate/
│   └── evaluation results (multi-seed)
├── step-5-aggregate/
│   └── aggregated_answers.jsonl
└── step-6-post-process/
    └── final_result.jsonl               # Cleaned + renamed records consumed by SFT / GRPO
```

## Expected Results

**Production (S&P 500):**

| Metric | Value |
|--------|-------|
| Questions Generated | ~2M+ |
| Verified Questions | ~1.6M |
| Final Q&A Pairs | ~800K |
| Time | ~30 hours, depending on the resources available |

## Output Format

### Final Training Data

**final_result.jsonl** - Cleaned, renamed records consumed by downstream SFT / GRPO. Each line contains the per-stage allowlisted generic fields (see `nvflow/stages/sdg/document_grounded/_schemas.py::STAGE_KEEP["dgsdg_post_process"]`) plus the project-declared `domain_keep_fields`. It also carries the Responses-API *original form* of the selected answer (`response` + `responses_create_params`) and an `expected_answer` mirroring `answer`, so the record is rollout-like and drop-in for SFT / GRPO. Example for the finance project:

```json
{
  "context": "...SEC filing excerpt...",
  "problem": "Based on the risk factors, what are Tesla's main supply chain concerns?",
  "answer": "...",
  "reasoning_content": "...",
  "question_type": "Risk_Factors",
  "answerable": "YES",
  "question_voting_pass_rate": 1.0,
  "question_voting_total": 5,
  "expected_answer": "...",
  "responses_create_params": { "...": "exact answer-gen request (Responses-API)" },
  "response": { "...": "original answer-gen response object (Responses-API)" },
  "company_name0": "Tesla, Inc.",
  "year": "2023",
  "item_section0": "Item 1A",
  "file_path0": ".../10-K/...",
  "file_type": "10-K"
}
```

## Validation

`$WORKSPACE` is the host directory your target mounts at `/workspace`, as in the
[quick start](../quick-start.md).

```bash
# Check all outputs exist
BASE_DIR="$WORKSPACE/outputs/finance/sp500/workflow-3-document-grounded-sdg"

# Stage outputs
ls $BASE_DIR/step-2-answers/generated/
ls $BASE_DIR/step-3-genselect/selected_answers.jsonl
ls $BASE_DIR/step-5-aggregate/aggregated_answers.jsonl

# Final dataset
ls $BASE_DIR/step-6-post-process/
wc -l $BASE_DIR/step-6-post-process/final_result.jsonl

# Inspect samples
head -n 3 $BASE_DIR/step-6-post-process/final_result.jsonl | jq .
```

## Stage 0: dg_sdg_preprocess Details

Converts raw SEC 10-K and 10-Q HTML filings into structured JSONL data for downstream processing.

### Steps

1. **Chunk HTML files**: Split large HTML documents into token-limited chunks with overlap
2. **Generate CSV lists**: Create file manifests tracking all chunks
3. **Create JSONL data**: Sample chunks following SecQue benchmark distribution

### Key Parameters

| Parameter | Description | Value in `document-grounded-sdg.yaml` |
|-----------|-------------|---------|
| `input_dir` | Raw SEC filings directory (10-K and 10-Q HTML files) | `${filings_dir}/data` |
| `output_dir` | Preprocessed data output directory | `${base_data_dir}/step-0-preprocess` |
| `distribution_dir` | Directory with distribution CSVs (SecQue benchmark) | `project://workflows/sdg/dg_sdg_distribution` |
| `sampling_strategy`, `preprocess_modules` | Sampling mode, and the dotted module path of the domain CLI that chunks + samples for each mode | `random`, which selects `nvfinance.utils.sdg.dg_sdg_data_preprocess` |
| `max_tokens` | Maximum tokens per chunk | 3000 |
| `overlap_tokens` | Overlap tokens between chunks for context coverage | 500 |
| `total_samples` | Total samples to generate following distribution | 150000 |
| `max_skip_count` | Stop sampling after this many skips (non-repeatable) | 20000 |
| `seed` | Random seed for reproducibility | 42 |

The stage's own defaults are `max_tokens: 2000` and `overlap_tokens: 100`; the
demo and S&P 500 production configs use 2000 and 300.

### Input Structure

Your SEC filings should follow this structure:

```
${filings_dir}/data/
├── {TICKER}/                      # e.g., AAPL, MSFT, TSLA
│   ├── 10-K/
│   │   └── {YEAR}/                # e.g., 2019, 2020, 2021
│   │       └── {FILING_ID}/       # SEC Filing ID
│   │           ├── 1.html         # Item sections
│   │           ├── 1A.html
│   │           ├── 7.html
│   │           └── exhibits/
│   │               └── EX-*.html  # Exhibits
│   └── 10-Q/
│       └── {YEAR}/
│           └── {FILING_ID}/...
└── {TICKER_2}/...
```

This structure is created automatically by the SEC download workflow ([Download SEC Filings](01-download-sec.md)).

## Stage 1: generate_verified_questions Details

This stage performs 4 internal sub-steps (Q-side of the pipeline):

1. **Q-prep** (CPU): Run the project-supplied `question_prep_script` to attach `context` strings to each chunk
2. **Q-gen** (GPU): Generate questions from documents with the generator (GPT-OSS-120B in the base config, Qwen3.5-397B-A17B in production)
3. **Q-verify-prep** (CPU): Expand each generated question into N verification trials
4. **Q-verify** (GPU): Per-question Yes/No vote with the judge (Qwen3-235B with 5 Gym repeats in the base config, GPT-OSS-120B with 3 in production)

See [technical reference](#generate_verified_questions) for details.

## Stage 2: generate_answers Details

This stage performs 2 internal sub-steps (A-side of the pipeline):

1. **A-prep** (CPU): `construct_answer_generate_input` keeps only questions whose Q-verify pass-rate ≥ `answer_preprocess_kwargs.threshold`
2. **A-gen** (GPU): Generate N candidate answers per surviving question with the generator, for downstream genselect (5 Gym repeats in the base config, 3 in production)

See [technical reference](#generate_answers) for details.

## Customization

### Adjust Resources

Each GPU stage sets its own `num_chunks`, `num_gpus` and `server_nodes`, so
override them per stage in a config that inherits the one you run:

**Example: Increase parallelization for GPU stages**
```yaml
stages:
  generate_verified_questions:
    question_verify_kwargs:
      args:
        num_chunks: 10  # Change from 1 → 10 to run 10 jobs in parallel
  gym_genselect_answers:
    num_chunks: 10
```
> **Note:** Each chunk is one job, so a stage uses `num_chunks × num_gpus × server_nodes` GPUs when all its chunks run at once. Ensure cluster has enough resources.

### Change Models

```yaml
stages:
  generate_verified_questions:
    question_generation_kwargs:
      args:
        model: /path/to/your/model
        num_gpus: 8

  generate_answers:
    answer_generation_kwargs:
      args:
        model: /path/to/your/model
        num_gpus: 8
```

### Modify Prompts

Edit prompts in `projects/finance/prompts/`:
- `document_grounded_generate_questions.yaml` - Question generation
- `document_grounded_verify_questions.yaml` - Question verification
- `secque_template.yaml` - Answer generation
- `genselect_answers.yaml` - GenSelect (best-of-N answer picker)
- `evaluate_answers.yaml` - Answer evaluation

## Common Issues

### "Input folder empty"

**Solution:** Ensure SEC filings downloaded:
```bash
ls $WORKSPACE/outputs/finance/sp500/workflow-2-download-sec/step-0-download/data/
# Should have company directories
```

### Low verification rate

**Solution:**
- Check verification threshold (default: 1.0 means all 5 repeats must verify)
- Lower threshold to 0.6 (3 out of 5 repeats)
- Review question generation prompt

## Combining with Template-Based

You can combine both SDG approaches:

```bash
# Merge datasets
cat $WORKSPACE/outputs/finance/sp500/workflow-3-template-based-sdg/step-5-filter-answers/final_result.jsonl \
    $WORKSPACE/outputs/finance/sp500/workflow-3-document-grounded-sdg/step-6-post-process/final_result.jsonl \
    > combined_training_data.jsonl

# Use combined data for SFT
# Update SFT workflow to point to combined_training_data.jsonl
```

## Next Steps

After completing document-grounded SDG:

- **[SFT Training](04-sft.md)** - Train on `final_result.jsonl`
- **[Evaluation](05-eval.md)** - Test model performance
- Combine with template-based data for more diversity

## Technical Details

For comprehensive stage-by-stage documentation:
- **[Document-Grounded SDG Stages Reference](#detailed-stage-reference)**

## Models Used

| Role | Production (`document-grounded-sdg-sp500.yaml`) | Base config (`document-grounded-sdg.yaml`) |
|------|-------------------------------------------------|--------------------------------------------|
| Question and answer generation | Qwen3.5-397B-A17B | GPT-OSS-120B |
| Question verification, answer selection, evaluation | GPT-OSS-120B | Qwen3-235B-A22B-Instruct-2507 |

All models are configurable in the workflow YAML.

## Detailed stage reference
Technical reference for all 7 stages in the document-grounded-sdg workflow.

### Quick Navigation

- [dg_sdg_preprocess](#dg_sdg_preprocess)
- [generate_verified_questions](#generate_verified_questions)
- [generate_answers](#generate_answers)
- [gym_genselect_answers](#gym_genselect_answers)
- [evaluate_answers](#evaluate_answers)
- [aggregate_answers](#aggregate_answers)
- [dgsdg_post_process](#dgsdg_post_process)

---

### dg_sdg_preprocess

**File:** `nvflow/stages/sdg/document_grounded/dg_sdg_preprocess.py`
**Binding:** `uses: sdg.document_grounded.preprocess`

#### Purpose

Converts raw SEC 10-K and 10-Q HTML filings into structured JSONL data for question/answer generation. This preprocessing stage performs three main operations:

1. **Chunk HTML files**: Split large documents into token-limited chunks with overlap
2. **Generate CSV lists**: Create file manifests tracking all chunks
3. **Create JSONL data**: Sample chunks following SecQue benchmark distribution

#### Inputs

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input_dir` | path | Raw SEC filings directory (10-K and 10-Q HTML files) | Required |
| `output_dir` | path | Preprocessed data output directory | Required |
| `distribution_dir` | path | Directory with distribution CSVs (SecQue benchmark) | Required |
| `preprocess_module` | str | Dotted module path to domain CLI that chunks + samples | Required |
| `max_tokens` | int | Maximum tokens per chunk | 2000 |
| `overlap_tokens` | int | Overlap tokens between chunks for context coverage | 100 |
| `total_samples` | int | Total samples to generate following distribution | 150000 |
| `max_skip_count` | int | Stop sampling after this many skips (non-repeatable) | 20000 |
| `seed` | int | Random seed for reproducibility | 42 |

#### Expected Input Structure

Your SEC filings should follow this structure (created by [Download SEC Filings](01-download-sec.md)):

```
${input_dir}/
├── {TICKER}/                      # e.g., AAPL, MSFT, TSLA
│   ├── 10-K/
│   │   └── {YEAR}/                # e.g., 2019, 2020, 2021
│   │       └── {FILING_ID}/       # SEC Filing ID
│   │           ├── 1.html         # Item sections
│   │           ├── 1A.html
│   │           ├── 7.html
│   │           └── exhibits/
│   │               └── EX-*.html  # Exhibits
│   └── 10-Q/
│       └── {YEAR}/
│           └── {FILING_ID}/...
└── {TICKER_2}/...
```

#### Outputs

```
${output_dir}/
├── chunks/                        # Chunked HTML files (3 formats)
│   ├── markdown/                  # Markdown-converted chunks
│   ├── clean_html/                # Cleaned HTML chunks
│   └── original_html/             # Original HTML chunks
├── csv_lists/                     # CSV manifests of chunks
└── jsonl/                         # Sampled data, one file per form ← Used by next stage
    ├── 10-k-data.jsonl
    ├── 10-q-data.jsonl
    └── 8-k-data.jsonl             # When 8-K is in the sampled forms
```

#### Configuration Example

```yaml
dg_sdg_preprocess:
  input_dir: ${filings_dir}/data
  output_dir: ${base_data_dir}/step-0-preprocess
  distribution_dir: project://workflows/sdg/dg_sdg_distribution
  preprocess_module: nvfinance.utils.sdg.dg_sdg_data_preprocess
  max_tokens: 3000
  overlap_tokens: 500
  total_samples: 150000
  max_skip_count: 20000
  seed: 42

```

#### Resources

- **Runtime:** ~2-4 hours for full S&P 500 dataset
- **Compute:** CPU only (no GPUs required)
- **Output Size:** ~10GB JSONL data for 150K samples

#### Notes

- Distribution CSVs define sampling proportions matching SecQue benchmark
- Chunking with overlap ensures context continuity across boundaries
- `max_skip_count` prevents infinite loops when distribution can't be satisfied

---

### generate_verified_questions

**File:** `nvflow/stages/sdg/document_grounded/generate_verified_questions.py`
**Binding:** `uses: sdg.document_grounded.generate_verified_questions`

#### Purpose

Q-side of the DG-SDG pipeline. Generates questions from SEC filing documents and verifies their quality. Executes 4 internal sub-steps.

#### Internal Sub-Steps

1. **Q-prep** (CPU): Run the project-supplied `question_prep_script` to attach `context` strings to each chunk
2. **Q-gen** (GPU): Generate questions from documents
3. **Q-verify-prep** (CPU): Expand each generated question into N verification trials
4. **Q-verify** (GPU): Per-question Yes/No vote with multiple Gym repeats

#### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_folder` | path | Preprocessed JSONL data directory from `dg_sdg_preprocess` (`${base_data_dir}/step-0-preprocess/jsonl/`) |
| `output_dir` | path | Q-pipeline output directory (e.g. `${base_data_dir}/step-1-questions`) |
| `question_prep_script` | path | Domain wrapper that injects `context_builder` into `lib.sdg.document_grounded.preprocess.construct_question_generate_input` |
| `gym_config_paths` / `gym_agent_name` | various | NeMo-Gym defaults; per-substep overrides allowed |
| `question_generation_kwargs` | dict | GPU settings for question generation |
| `question_verify_kwargs` | dict | GPU settings for verification (typically 5 repeats) |

#### Outputs

```
${output_dir}/
├── generate_input.jsonl   # step 1 output (q-prep)
├── generated/             # step 2 output (Q-gen rollouts)
├── verify_input.jsonl     # step 3 output (q-verify-prep)
└── verified/              # step 4 output (Q-verify rollouts)
                           # ← consumed by generate_answers
```

#### Configuration Example

<!-- doc-check: yaml-excerpt -->
```yaml
generate_verified_questions:
  input_folder: ${base_data_dir}/step-0-preprocess/jsonl
  output_dir: ${base_data_dir}/step-1-questions
  dependencies: [dg_sdg_preprocess]

  question_prep_script: project://nvfinance/utils/sdg/sec_question_prep.py
  gym_config_paths: *gym_config_paths_format_verification
  gym_agent_name: *gym_agent_format_verification

  question_generation_kwargs:
    args:
      model: /hf_models/openai/gpt-oss-120b
      num_gpus: 8
      server_nodes: 1
      num_repeats: 1
      num_chunks: 1
    ctx_args: >-
      ++prompt_config=project://prompts/document_grounded_generate_questions.yaml
      ++inference.temperature=0.9 ++inference.top_p=0.9

  question_verify_kwargs:
    args:
      model: /hf_models/Qwen/Qwen3-235B-A22B-Instruct-2507
      num_gpus: 8
      server_nodes: 1
      num_repeats: 5
      num_chunks: 1
```

#### Resources

- **Runtime:** ~2-4 hours
- **GPUs:** 8 per chunk for each GPU sub-step; the base config runs one chunk of
  each. Production runs 120 chunks of each, with the Qwen3.5-397B-A17B generator
  on 16 GPUs (2 nodes) per chunk.
- **Models:** GPT-OSS-120B (questions) and Qwen3-235B (verification) in the base config

---

### generate_answers

**File:** `nvflow/stages/sdg/document_grounded/generate_answers.py`
**Binding:** `uses: sdg.document_grounded.generate_answers`

#### Purpose

A-side of the DG-SDG pipeline. Filters questions by verification pass-rate, then generates N candidate answers per surviving question. Executes 2 internal sub-steps.

#### Internal Sub-Steps

1. **A-prep** (CPU): `construct_answer_generate_input` keeps only questions whose Q-verify pass-rate ≥ `threshold`
2. **A-gen** (GPU): Generate answers (typically 5 Gym repeats for downstream genselect)

#### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_dir` | path | Verified-questions directory from `generate_verified_questions` (`${base_data_dir}/step-1-questions/verified`) |
| `output_dir` | path | A-pipeline output directory (e.g. `${base_data_dir}/step-2-answers`) |
| `gym_config_paths` / `gym_agent_name` | various | NeMo-Gym defaults; per-substep overrides allowed |
| `answer_preprocess_kwargs` | dict | CPU settings, includes `threshold` (Q-verify pass-rate cutoff) |
| `answer_generation_kwargs` | dict | GPU settings for answer generation |

#### Outputs

```
${output_dir}/
├── answer_input.jsonl   # step 1 output (a-prep)
└── generated/           # step 2 output (A-gen rollouts; consumed by gym_genselect_answers)
    └── jobs/job-1-analyze-rollouts/outputs/
        ├── rollouts.jsonl
        └── rollouts_materialized_inputs.jsonl
```

#### Configuration Example

<!-- doc-check: yaml-excerpt -->
```yaml
generate_answers:
  input_dir: ${base_data_dir}/step-1-questions/verified
  output_dir: ${base_data_dir}/step-2-answers
  dependencies: [generate_verified_questions]

  gym_config_paths: *gym_config_paths_format_verification
  gym_agent_name: *gym_agent_format_verification

  answer_preprocess_kwargs:
    threshold: 1

  answer_generation_kwargs:
    args:
      model: /hf_models/openai/gpt-oss-120b
      num_gpus: 8
      server_nodes: 1
      num_chunks: 1
      num_repeats: 5
    ctx_args: >-
      ++prompt_config=project://prompts/secque_template.yaml
```

#### Resources

- **Runtime:** ~2-4 hours
- **GPUs:** 8 per chunk; the base config runs one chunk, and repeats run inside
  each chunk. Production runs 240 chunks with the generator on 16 GPUs (2 nodes) each.
- **Model:** GPT-OSS-120B in the base config

---

### gym_genselect_answers

**File:** `nvflow/stages/sdg/document_grounded/gym_genselect_answers.py`
**Binding:** `uses: sdg.document_grounded.gym_genselect_answers`

#### Purpose

Select the best answer from the multiple candidates produced by `generate_answers` (DG-SDG-specific best-of-N picker that runs through NeMo-Gym).

#### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_dir` | path | Answer candidates from `generate_answers` |
| `output_file` | path | Selected answers output file |
| `prompt_template` | path | GenSelect prompt |

#### Outputs

JSONL file with selected best answers.

#### Configuration Example

```yaml
  gym_genselect_answers:
    input_dir: ${base_data_dir}/step-2-answers/generated
    output_file: ${base_data_dir}/step-3-genselect/selected_answers.jsonl
    prompt_template: project://prompts/genselect_answers.yaml
    dependencies: [generate_answers]

    models:
      policy:
        model: /hf_models/Qwen/Qwen3-235B-A22B-Instruct-2507
        num_gpus: 8
        server_nodes: 1
    num_chunks: 5
    inference_params:
      max_output_tokens: 16384
```

#### Resources

- **GPUs:** 40 (5 chunks × 8) in the base config; production runs 240 chunks on GPT-OSS-120B
- **Model:** Qwen3-235B in the base config
- **Runtime:** 2-4 hours

---

### evaluate_answers

**File:** `nvflow/stages/sdg/document_grounded/evaluate_answers.py`
**Binding:** `uses: sdg.document_grounded.evaluate_answers`

#### Purpose

Evaluate answer quality using a large model judge. Runs 5 Gym repeats for robustness. Each response ends with a JSON verdict tag `{"answerable": "YES/NO", "correct": "YES/NO"}` (parsed downstream by `aggregate_answers`).

#### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_file` | path | Selected answers from `gym_genselect_answers` |
| `output_dir` | path | Directory for evaluation results |
| `prompt_template` | path | Evaluation prompt |

#### Outputs

```
${output_dir}/
└── jobs/
    ├── job-0-collect-rollouts/chunks/
    └── job-1-analyze-rollouts/
        ├── outputs/rollouts.jsonl
        ├── outputs/rollouts_materialized_inputs.jsonl
        ├── completion.json
        └── logs/
```

Each record carries the judge's raw `evaluate_generation`, whose last line is the JSON verdict tag parsed by `aggregate_answers`:
```json
{
  "problem": "...",
  "generation": "...",
  "evaluate_generation": "...reasoning...\n{\"answerable\": \"YES\", \"correct\": \"YES\"}"
}
```

#### Configuration Example

```yaml
  evaluate_answers:
    input_file: ${base_data_dir}/step-3-genselect/selected_answers.jsonl
    output_dir: ${base_data_dir}/step-4-evaluate
    prompt_template: project://prompts/evaluate_answers.yaml
    generation_key: evaluate_generation
    dependencies: [gym_genselect_answers]

    models:
      policy:
        model: /hf_models/Qwen/Qwen3-235B-A22B-Instruct-2507
        num_gpus: 8
        server_nodes: 1
    num_chunks: 1
    num_repeats: 5
    inference_params:
      top_p: 0.9
      temperature: 0.8
```

#### Resources

- **GPUs:** 8 (one chunk; repeats run inside it)
- **Model:** Qwen3-235B
- **Runtime:** 8 hours

---

### aggregate_answers

**File:** `nvflow/stages/sdg/document_grounded/aggregate_answers.py`
**Binding:** `uses: sdg.document_grounded.aggregate_answers`

#### Purpose

Aggregate the 5 evaluation repeats: keep a question only if **all** repeats voted `correct=YES` with a consistent `answerable`, and attach the consensus `answerable`.

#### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_dir` | path | Evaluation results from `evaluate_answers` |
| `output_file` | path | Aggregated results output |

#### Outputs

A single JSONL file of surviving records. The per-seed `evaluate_generation` / `correct` are dropped and a consensus `answerable` is added:
```json
{
  "problem": "...",
  "generation": "...",
  "reference_answer": "...",
  "answerable": "YES"
}
```

#### Resources

- **Compute:** CPU only
- **Runtime:** 5-10 min

---

### dgsdg_post_process

**File:** `nvflow/stages/sdg/document_grounded/dgsdg_post_process.py`
**Binding:** `uses: sdg.document_grounded.postprocess`

#### Purpose

Clean and rename fields, then emit a single `final_result.jsonl` consumed by downstream SFT / GRPO workflows. Records are not split into subsets; the per-stage trim (see `_schemas.py::STAGE_KEEP["dgsdg_post_process"]`) plus the project's `domain_keep_fields` defines the final allowlist of fields kept in `final_result.jsonl`.

#### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_file` | path | Aggregated answers from `aggregate_answers` (e.g. `${base_data_dir}/step-5-aggregate/aggregated_answers.jsonl`) |
| `output_dir` | path | Output directory for `final_result.jsonl` |
| `postprocess_script` | path | Domain CLI wrapper around `nvflow.stages.sdg.document_grounded.postprocess.dgsdg_post_process` (e.g. `projects/finance/nvfinance/utils/sdg/sec_postprocess.py`) |
| `seed` | int | Random seed for reproducibility (default: 42) |
| `domain_keep_fields` | list[str] | Project-specific fields appended to the generic allowlist before per-stage trim |

#### Outputs

```
${output_dir}/
└── final_result.jsonl    # Single cleaned + renamed dataset consumed by SFT / GRPO
```

#### Resources

- **Compute:** CPU only
- **Runtime:** 5-10 min

---

### Pipeline Summary

| Stage | Purpose | Compute | Runtime |
|-------|---------|---------|---------|
| dg_sdg_preprocess | Chunk + sample documents | CPU | 2-4h |
| generate_verified_questions | Generate + verify questions | GPU | 2-4h |
| generate_answers | Generate candidate answers | GPU | 2-4h |
| gym_genselect_answers | Select best answers | GPU | 1-2h |
| evaluate_answers | Evaluate quality | GPU | 2-3h |
| aggregate_answers | Aggregate scores | CPU | 10m |
| dgsdg_post_process | Clean + rename → final_result.jsonl | CPU | 10m |

**Total:** ~30 hours for the full S&P 500 run, depending on the resources available.
Production chunks can outlive a 4-hour job limit; rerunning `run --all` resumes
them from the rows already generated.

See [Document-Grounded SDG Workflow](../workflows/03-document-grounded-sdg.md) for usage examples and configuration details.
