# Workflow 2: Template-Based SDG

## Purpose

Generate financial Q&A pairs by adapting seed questions to different companies and years, then grounding them in actual SEC filing context.

## Prerequisites

Before running this workflow, ensure you have:

- ✅ **SEC filings downloaded** ([Workflow 1](01-download-sec.md))
  - Required: `sec_metadata.parquet` and filing HTML files in `outputs/finance/demo/workflow-2-download-sec/step-0-download`

- ✅ **HuggingFace access token** (optional but recommended):
  ```bash
  export HF_TOKEN=<your-token>
  ```
  - **Why needed:** Stage 0 downloads the [SecQue dataset](https://huggingface.co/datasets/nvidia/SecQue) (seed questions) from HuggingFace
  - **Public dataset:** No token required for public access, but token avoids rate limits
  - **Login alternative:** Run `huggingface-cli login` if you prefer interactive login
  - **Offline clusters:** Because Stage 0 reaches the Hub, temporarily clear `HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE` and `TRANSFORMERS_OFFLINE` in your cluster config for this run, then restore them. See [Offline runtime](../troubleshooting.md#offline-runtime)

- ✅ **SEC EDGAR identity configured** in workflow YAML:
  ```yaml
  stages:
    create_seed_data:
      sec_identity_email: your.email@company.com
      sec_identity_company: YourCompany
  ```
  - **Why needed:** Stage 0 fetches company metadata from SEC EDGAR API, which requires identification per their Fair Access Policy

- ✅ **Inference server models**:
  - **Question/Answer generation:** Requires a large language model (e.g., `gpt-oss-120b`, `Qwen-3`)
  - **Answer selection:** Requires a larger model different from the question answering model for quality evaluation
  - Models must be accessible via configured inference servers (vLLM or SGLang)

## How It Works

This workflow generates synthetic financial Q&A pairs using a template-based approach:

1. **Seed Questions** → Downloaded from SecQue dataset (pre-validated financial questions)
2. **Question Generation** → Adapt seed questions to different companies and years
3. **Context Mapping** → Find relevant SEC filing sections for each question
4. **Answer Generation** → Generate multiple candidate answers (5 variations with different random seeds)
5. **Answer Selection** → Use a large model to select the best answer from candidates
6. **Filtering** → Remove questions that are unanswerable or low-quality

Each stage produces JSONL files that feed into the next stage, with automatic resumption if a stage is re-run (it continues from existing outputs).

## Pipeline Flow

```
┌──────────────────────────┐
│ 0. create_seed_data      │  Setup: Download seed questions + company info
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ 1. generate_questions    │  Generation: Adapt seed questions to companies/years
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ 2. map_questions_to_     │  Mapping: Find relevant SEC filing sections
│    context               │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ 3. generate_answers      │  Generation: Multiple candidate answers (5 seeds)
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ 4. genselect_answers     │  Selection: Best answer from candidates
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ 5. filter_answers        │  Filtering: Remove unanswerable questions
└──────────────────────────┘
```

## 6 Stages (Overview)

0. **create_seed_data**: Download SecQue dataset + S&P 100 company info
1. **generate_questions**: Adapt seed questions to different companies/years
2. **map_questions_to_context**: Find relevant SEC filing sections
3. **generate_answers**: Generate multiple candidate answers (5 random seeds)
4. **genselect_answers**: Select best answer using large model
5. **filter_answers**: Remove unanswerable questions

**See [technical reference](../stages/template-based-sdg.md) for detailed stage documentation.**

## Configurations

| Config | Use Case | Models | Resources | Time |
|--------|----------|--------|-----------|------|
| `template-based-sdg-demo.yaml` | Quick E2E test | Small | 4 GPUs | 30 min |
| `template-based-sdg.yaml` | Production | Large (120B) | 8+ GPUs | 8 hours |

## Configuration Setup

Before running, update the SEC identity in `template-based-sdg.yaml`:

```yaml
stages:
  create_seed_data:
    sec_identity_email: your.email@example.com  # UPDATE: Your email
    sec_identity_company: YourCompany           # UPDATE: Your company name
```

This is required by the SEC EDGAR API for fetching company metadata in stage 0.

### Optional: Customize Output Directory

```yaml
base_output_dir: ${base_data_dir}/sdg/my_custom_run  # Default: prod_run
```

### Optional: Filter Companies (for testing)

```yaml
stages:
  create_seed_data:
    filter_company_list: "NVDA,AAPL,MSFT"  # Only process these tickers
    num_seed_questions: 30                  # Limit seed questions for faster testing
```

## Usage

### Demo (End-to-End)

Run complete pipeline with smaller models for testing:

```bash
uv run nflow run-all --config nvflow/recipes/finance/workflows/sdg/template-based-sdg-demo.yaml
```

The demo config automatically filters to:
- 7 companies (AAPL, GOOG, CSCO, IBM, META, NVDA, MSFT)
- 30 seed questions

### Production (Run All Stages Together)

```bash
uv run nflow run-all --config nvflow/recipes/finance/workflows/sdg/template-based-sdg.yaml
```

Stages will chain via Slurm dependencies.

Each stage continues from whatever partial output is available at that stage, this allows to safely rerun the whole pipeline without redoing any work.

### Production (Stage-by-Stage)

Recommended for production to monitor progress and handle failures:

```bash
# Stage 0: Create seed data (downloads from HuggingFace + SEC EDGAR)
uv run nflow run create_seed_data --config nvflow/recipes/finance/workflows/sdg/template-based-sdg.yaml

# Stage 1: Generate questions
uv run nflow run generate_questions --config nvflow/recipes/finance/workflows/sdg/template-based-sdg.yaml

# Wait for completion, then Stage 2
uv run nflow run map_questions_to_context --config nvflow/recipes/finance/workflows/sdg/template-based-sdg.yaml

# Stage 3
uv run nflow run generate_answers --config nvflow/recipes/finance/workflows/sdg/template-based-sdg.yaml

# Stage 4
uv run nflow run genselect_answers --config nvflow/recipes/finance/workflows/sdg/template-based-sdg.yaml

# Stage 5
uv run nflow run filter_answers --config nvflow/recipes/finance/workflows/sdg/template-based-sdg.yaml
```

## Output Structure

```
${base_output_dir}/workflow-3-template-based-sdg/
├── step-0-create-seed-data/
│   ├── seed_questions.jsonl        # Seed questions from SecQue
│   ├── company_info.tsv            # S&P 100 company metadata
│   └── logs/                       # Stage logs
├── step-1-generate-questions/
│   └── final_result.jsonl          # Generated questions
├── step-2-map-questions-to-context/
│   └── final_result.jsonl          # Questions + context
├── step-3-generate-answers/
│   ├── output-rs0.jsonl            # Multiple candidate answers (rs means random seed)
│   ├── output-rs1.jsonl
│   └── ...
├── step-4-genselect-answers/
│   └── final_result.jsonl          # Best answers selected
└── step-5-filter-answers/
    └── final_result.jsonl          # ← Final output for training
```

## Expected Results

| Config | Questions Generated | Final Q&A Pairs | Time |
|--------|---------------------|-----------------|------|
| Demo | ~2K | ~500-1K | 30 min |
| Production (S&P 500) | ~500K | ~300K | 8 hours |

## Output Format

Each line in `step-5-filter-answers/final_result.jsonl`:

```json
{
  "question": "What was NVIDIA's total revenue in fiscal year 2023?",
  "context": "...relevant excerpt from NVIDIA 10-K filing...",
  "generation": "<reasoning>From the consolidated statements...\n<answer>$26.97 billion</answer>",
  "company": "NVDA",
  "year": 2023,
  "seed_question_id": "sq_001"
}
```

## Validation

```bash
# Set output directory (from nvflow directory)
OUTPUT_DIR="outputs/finance/sap-500/workflow-3-template-based-sdg"

# Check seed data was created
wc -l $OUTPUT_DIR/step-0-create-seed-data/seed_questions.jsonl
wc -l $OUTPUT_DIR/step-0-create-seed-data/company_info.tsv

# Count final Q&A pairs
wc -l $OUTPUT_DIR/step-5-filter-answers/final_result.jsonl

# Inspect sample outputs
head -n 3 $OUTPUT_DIR/step-5-filter-answers/final_result.jsonl | jq .

# Check intermediate stages
for step in step-{0..5}*; do
  echo "=== $step ==="
  ls -lh $OUTPUT_DIR/$step/
done
```

## Customization

### Use Custom Config

Create a custom config inheriting from base:

```yaml
# my-custom-sdg.yaml
_base_: template-based-sdg.yaml

# Override output directory
base_output_dir: ${base_data_dir}/sdg/my_custom_run

# Override SEC identity
stages:
  create_seed_data:
    sec_identity_email: my.email@company.com
    sec_identity_company: MyCompany

  # Override model for answers
  generate_answers:
    stage_kwargs:
      model: /path/to/my/model
      server_gpus: 4
```

Run with:
```bash
uv run nflow run-all --config my-custom-sdg.yaml
```

### Filter to Specific Companies (Demo Mode)

The demo config shows how to filter:

```yaml
stages:
  create_seed_data:
    filter_company_list: "AAPL,GOOG,MSFT"  # Only these companies
    num_seed_questions: 30                  # Sample 30 questions
```

### Modify Prompts

Edit prompts in `nvflow/recipes/finance/prompts/`:
- `generate_questions.yaml` - Question generation prompt
- `generate_answers.yaml` - Answer generation prompt
- `genselect_answers.yaml` - Answer selection prompt
- `filter_answers.yaml` - Filtering criteria

## Common Issues

### "File not found: sec_metadata.parquet"

**Solution:** Ensure download-sec workflow completed successfully:
```bash
# From nvflow directory
ls outputs/finance/demo/workflow-2-download-sec/step-0-download/sec_metadata.parquet
ls outputs/finance/demo/workflow-2-download-sec/step-0-download/data/
```

### Stage fails with OOM (Out of Memory)

**Solution:**
- Use smaller model
- Reduce the number of max_concurrent_requests and/or tokens_to_generate

### No outputs after filter_answers

**Solution:**
- Check genselect_answers output quality
- Adjust filtering threshold in `filter_answers` prompt
- Review earlier stage outputs for data quality

## Next Steps

After completing template-based SDG:

- **[SFT Training](04-sft.md)** - Train model on generated data
- **[Document-Grounded SDG](03-document-grounded-sdg.md)** - Run alternative approach (can combine datasets)
- **Scale to Production** - Use production config examples above with sp500.yaml for full S&P 500

## Technical Details

For comprehensive stage-by-stage documentation:
- **[Template-Based SDG Stages Reference](../stages/template-based-sdg.md)**

For understanding the approach:
- GenSelect paper: [arXiv:2507.17797](https://arxiv.org/abs/2507.17797)
