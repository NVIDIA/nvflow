# Workflow 3: Document-Grounded SDG

## Purpose

Generate high-quality financial Q&A pairs directly from SEC filing documents with built-in question verification, multi-seed answer evaluation, and per-stage field trimming.

> **Note:** This workflow generates ~800K Q&A pairs in a single `final_result.jsonl`. The previous difficulty-stratified outputs (`full_data.jsonl`, `hard_rl_data.jsonl`) and the `difficulty_estimation` stage have been removed; downstream SFT / GRPO workflows read `final_result.jsonl` directly. For the production template-based pipeline, see [Template-Based SDG](02-template-based-sdg.md).

## Prerequisites

- SEC filings downloaded ([Workflow 1](01-download-sec.md))
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

**See [technical reference](../stages/document-grounded-sdg.md) for detailed stage documentation.**

## Configuration

**File:** `workflows/sdg/document-grounded-sdg.yaml`

- Production-ready configuration
- Uses large models (GPT-OSS-120B, Qwen3-235B)
- Configured for `/workspace/outputs/finance/sap-500/workflow-2-download-sec/` input

## Usage

### Run Complete Workflow

```bash
uv run nflow run-all --config nvflow/recipes/finance/workflows/sdg/document-grounded-sdg.yaml
```

### Run Individual Stages

```bash
# Stage 0: Preprocess SEC filings
uv run nflow run dg_sdg_preprocess --config nvflow/recipes/finance/workflows/sdg/document-grounded-sdg.yaml

# Stage 1: Generate + verify questions
uv run nflow run generate_verified_questions --config nvflow/recipes/finance/workflows/sdg/document-grounded-sdg.yaml

# Stage 2: Generate candidate answers
uv run nflow run generate_answers --config nvflow/recipes/finance/workflows/sdg/document-grounded-sdg.yaml

# Stage 3: Select best answers
uv run nflow run gym_genselect_answers --config nvflow/recipes/finance/workflows/sdg/document-grounded-sdg.yaml

# Stage 4: Evaluate answers
uv run nflow run evaluate_answers --config nvflow/recipes/finance/workflows/sdg/document-grounded-sdg.yaml

# Stage 5: Aggregate results
uv run nflow run aggregate_answers --config nvflow/recipes/finance/workflows/sdg/document-grounded-sdg.yaml

# Stage 6: Post process
uv run nflow run dgsdg_post_process --config nvflow/recipes/finance/workflows/sdg/document-grounded-sdg.yaml
```

## Output Structure

```
${base_data_dir}/
├── step-0-preprocess/
│   ├── chunks/                          # Chunked HTML files (markdown, clean HTML, original)
│   ├── csv_lists/                       # CSV manifests of chunks
│   └── jsonl/
│       ├── 10-k-data.jsonl              # Sampled 10-K data
│       └── 10-q-data.jsonl              # Sampled 10-Q data
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
| Time | ~30 hours, affected by resources used |

## Output Format

### Final Training Data

**final_result.jsonl** - Cleaned, renamed records consumed by downstream SFT / GRPO. Each line contains the per-stage allowlisted generic fields (see `nvflow/generic_stage/sdg/document_grounded/_schemas.py::STAGE_KEEP["dgsdg_post_process"]`) plus the recipe-declared `domain_keep_fields`. It also carries the Responses-API *original form* of the selected answer (`response` + `responses_create_params`) and an `expected_answer` mirroring `answer`, so the record is rollout-like and drop-in for SFT / GRPO. Example for the finance recipe:

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

```bash
# Check all outputs exist (from nvflow directory)
BASE_DIR="outputs/finance/sap-500/workflow-3-document-grounded-sdg"

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

| Parameter | Description | Default |
|-----------|-------------|---------|
| `input_dir` | Raw SEC filings directory (10-K and 10-Q HTML files) | `${filings_dir}/data` |
| `output_dir` | Preprocessed data output directory | `${base_data_dir}/step-0-preprocess` |
| `distribution_dir` | Directory with distribution CSVs (SecQue benchmark) | `nvflow/recipes/finance/workflows/sdg/dg_sdg_distribution` |
| `preprocess_module` | Dotted module path to domain CLI that chunks + samples | `nvflow.recipes.finance.utils.sdg.dg_sdg_data_preprocess` |
| `max_tokens` | Maximum tokens per chunk | 3000 |
| `overlap_tokens` | Overlap tokens between chunks for context coverage | 500 |
| `total_samples` | Total samples to generate following distribution | 150000 |
| `max_skip_count` | Stop sampling after this many skips (non-repeatable) | 20000 |
| `seed` | Random seed for reproducibility | 42 |

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

This structure is created automatically by the SEC download workflow ([Workflow 1](01-download-sec.md)).

## Stage 1: generate_verified_questions Details

This stage performs 4 internal sub-steps (Q-side of the pipeline):

1. **Q-prep** (CPU): Run the recipe-supplied `question_prep_script` to attach `context` strings to each chunk
2. **Q-gen** (GPU): Generate questions from documents using GPT-OSS-120B
3. **Q-verify-prep** (CPU): Expand each generated question into N verification trials
4. **Q-verify** (GPU): Per-question Yes/No vote using Qwen3-235B (5 seeds)

See [technical reference](../stages/document-grounded-sdg.md#generate_verified_questions) for details.

## Stage 2: generate_answers Details

This stage performs 2 internal sub-steps (A-side of the pipeline):

1. **A-prep** (CPU): `construct_answer_generate_input` keeps only questions whose Q-verify pass-rate ≥ `answer_preprocess_kwargs.threshold`
2. **A-gen** (GPU): Generate N candidate answers per surviving question using GPT-OSS-120B (5 seeds for downstream genselect)

See [technical reference](../stages/document-grounded-sdg.md#generate_answers) for details.

## Customization

### Adjust Resources

Edit `workflows/sdg/document-grounded-sdg.yaml` to control parallelization and GPU allocation:
**Example: Increase parallelization For GPU Stages**
```yaml
num_chunks: 10  # Change from 1 → 10 to run 10 jobs in parallel
```
> **Note:** Total GPU usage = `num_chunks × server_gpus`. Ensure cluster has enough resources.

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

Edit prompts in `nvflow/recipes/finance/prompts/`:
- `document_grounded_generate_questions.yaml` - Question generation
- `document_grounded_verify_questions.yaml` - Question verification
- `secque_template.yaml` - Answer generation
- `genselect_answers.yaml` - GenSelect (best-of-N answer picker)
- `evaluate_answers.yaml` - Answer evaluation

## Common Issues

### "Input folder empty"

**Solution:** Ensure SEC filings downloaded:
```bash
# From nvflow directory
ls outputs/finance/sap-500/workflow-2-download-sec/step-0-download/data/
# Should have company directories
```

### Low verification rate

**Solution:**
- Check verification threshold (default: 1.0 means all 5 seeds must verify)
- Lower threshold to 0.6 (3 out of 5 seeds)
- Review question generation prompt

## Combining with Template-Based

You can combine both SDG approaches:

```bash
# Merge datasets (from nvflow directory)
cat outputs/finance/sap-500/workflow-3-template-based-sdg/step-5-filter-answers/final_result.jsonl \
    outputs/finance/sap-500/workflow-3-document-grounded-sdg/step-6-post-process/final_result.jsonl \
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
- **[Document-Grounded SDG Stages Reference](../stages/document-grounded-sdg.md)**

## Models Used

| Model | Usage | Size |
|-------|-------|------|
| GPT-OSS-120B | Question generation, answer generation | 120B |
| Qwen3-235B-A22B | Question verification, answer selection, evaluation | 235B |

All models are configurable in the workflow YAML.
