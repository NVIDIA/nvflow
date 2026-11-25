# Template-Based SDG

## Purpose

Generate financial Q&A pairs by adapting seed questions to different companies and years, then grounding them in actual SEC filing context.

## Prerequisites

Before running this workflow, ensure you have:

- ✅ **SEC filings downloaded** ([Download SEC Filings](01-download-sec.md))
  - Required: `sec_metadata.parquet` and filing HTML files in `outputs/finance/demo/workflow-2-download-sec/step-0-download`

- ✅ **HuggingFace access token** (optional but recommended):
  ```bash
  export HF_TOKEN=<your-token>
  ```
  - **Why needed:** Stage 0 downloads the [SecQue dataset](https://huggingface.co/datasets/nogabenyoash/SecQue) (seed questions) from HuggingFace
  - **Public dataset:** No token required for public access, but token avoids rate limits
  - **Login alternative:** Run `uvx --from huggingface_hub hf auth login` if you prefer interactive login
  - **Offline clusters:** Because Stage 0 reaches the Hub, temporarily clear `HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE` and `TRANSFORMERS_OFFLINE` for this run, then restore them. Where the flags and `HF_TOKEN` live depends on your backend: [Slurm](../../../../docs/slurm/README.md#where-settings-live) or [Ray](../../../../docs/ray/README.md#where-settings-live)

- ✅ **SEC EDGAR identity exported** in the launcher environment:
  ```bash
  export SEC_IDENTITY_EMAIL=your.email@company.com
  export SEC_IDENTITY_COMPANY="Your Company"
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

**See [technical reference](#detailed-stage-reference) for detailed stage documentation.**

## Configurations

| Config | Use Case | Models | Resources | Time |
|--------|----------|--------|-----------|------|
| `template-based-sdg-demo.yaml` | Quick E2E test | Small | 4 GPUs | 30 min |
| `template-based-sdg.yaml` | Production | Large (30B to 235B) | 8+ GPUs; GenSelect uses 16 on 2 nodes | 8 hours |

The production config uses three models:

- **GPT-OSS-120B** generates questions and answers on 8 GPUs.
- **Qwen3-235B-A22B-Instruct-2507** runs `genselect_answers` on 2 nodes × 8 GPUs
  from `/hf_models/Qwen/Qwen3-235B-A22B-Instruct-2507-tp16-sglang`, a copy saved
  in SGLang's sharded-state format for 16-way tensor parallelism and loaded with
  `--load-format sharded_state`. That copy is not on Hugging Face. Without one,
  serve the standard `Qwen/Qwen3-235B-A22B-Instruct-2507` checkpoint: in a
  [custom config](#use-custom-config), set
  `stages.genselect_answers.stage_kwargs.model` to
  `/hf_models/Qwen/Qwen3-235B-A22B-Instruct-2507` and `server_args` to `""`.
- **Qwen3-30B-A3B-Instruct-2507** runs `filter_answers` on 2 GPUs per chunk.

## Configuration Setup

Before running, export the identity required by SEC EDGAR:

```bash
export SEC_IDENTITY_EMAIL=your.email@example.com
export SEC_IDENTITY_COMPANY="Your Company"
```

This is required by the SEC EDGAR API for fetching company metadata in stage 0.

Only stage 0 reads these. Re-running any later stage on its own needs neither, so
a fresh shell can resume mid-pipeline without exporting anything. Stage 0 refuses
to submit when either is unset rather than sending SEC an unidentified request.

### Optional: Customize Output Directory

```yaml
base_output_dir: /workspace/outputs/finance/sp500/my_custom_run  # Default: /workspace/outputs/finance/sp500/workflow-3-template-based-sdg
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
uv run nvflow run --all --config projects/finance/workflows/sdg/template-based-sdg-demo.yaml --cluster my_cluster
```

The demo config automatically filters to:
- 7 companies (AAPL, GOOG, CSCO, IBM, META, NVDA, MSFT)
- 30 seed questions

### Production (Run Complete Workflow)

```bash
uv run nvflow run --all --config projects/finance/workflows/sdg/template-based-sdg.yaml --cluster my_cluster
```

Stages will chain via Slurm dependencies.

Each stage continues from whatever partial output is available at that stage, this allows to safely rerun the whole pipeline without redoing any work.

### Production (Stage-by-Stage)

Recommended for production to monitor progress and handle failures:

```bash
# Stage 0: Create seed data (downloads from HuggingFace + SEC EDGAR)
uv run nvflow run create_seed_data --config projects/finance/workflows/sdg/template-based-sdg.yaml --cluster my_cluster

# Stage 1: Generate questions
uv run nvflow run generate_questions --config projects/finance/workflows/sdg/template-based-sdg.yaml --cluster my_cluster

# Wait for completion, then Stage 2
uv run nvflow run map_questions_to_context --config projects/finance/workflows/sdg/template-based-sdg.yaml --cluster my_cluster

# Stage 3
uv run nvflow run generate_answers --config projects/finance/workflows/sdg/template-based-sdg.yaml --cluster my_cluster

# Stage 4
uv run nvflow run genselect_answers --config projects/finance/workflows/sdg/template-based-sdg.yaml --cluster my_cluster

# Stage 5
uv run nvflow run filter_answers --config projects/finance/workflows/sdg/template-based-sdg.yaml --cluster my_cluster
```

## Output Structure

```
${base_output_dir}/                 # Default: /workspace/outputs/finance/sp500/workflow-3-template-based-sdg
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

`$WORKSPACE` is the host directory your target mounts at `/workspace`, as in the
[quick start](../quick-start.md).

```bash
# Set output directory
OUTPUT_DIR="$WORKSPACE/outputs/finance/sp500/workflow-3-template-based-sdg"

# Check seed data was created
wc -l $OUTPUT_DIR/step-0-create-seed-data/seed_questions.jsonl
wc -l $OUTPUT_DIR/step-0-create-seed-data/company_info.tsv

# Count final Q&A pairs
wc -l $OUTPUT_DIR/step-5-filter-answers/final_result.jsonl

# Inspect sample outputs
head -n 3 $OUTPUT_DIR/step-5-filter-answers/final_result.jsonl | jq .

# Check intermediate stages
for step in $OUTPUT_DIR/step-{0..5}-*/; do
  echo "=== $step ==="
  ls -lh "$step"
done
```

## Customization

### Use Custom Config

Create a custom config inheriting from base. Name it `private_*.yaml` so git
ignores it and `run` accepts it, or commit it:

```yaml
# projects/finance/workflows/sdg/private_custom_sdg.yaml
_base_: template-based-sdg.yaml

# Override output directory
base_output_dir: /workspace/outputs/finance/sp500/my_custom_run

stages:
  # Override model for answers
  generate_answers:
    stage_kwargs:
      model: /path/to/my/model
      server_gpus: 4
```

Run with:
```bash
uv run nvflow run --all --config projects/finance/workflows/sdg/private_custom_sdg.yaml --cluster my_cluster
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

Edit prompts in `projects/finance/prompts/`:
- `generate_questions.yaml` - Question generation prompt
- `secque_template.yaml` - Answer generation prompt
- `genselect_answers.yaml` - Answer selection prompt
- `filter_answers.yaml` - Filtering criteria

## Common Issues

### "File not found: sec_metadata.parquet"

**Solution:** Ensure download-sec workflow completed successfully:
```bash
ls $WORKSPACE/outputs/finance/demo/workflow-2-download-sec/step-0-download/sec_metadata.parquet
ls $WORKSPACE/outputs/finance/demo/workflow-2-download-sec/step-0-download/data/
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
- **[Template-Based SDG Stages Reference](#detailed-stage-reference)**

For understanding the approach:
- GenSelect paper: [arXiv:2507.17797](https://arxiv.org/abs/2507.17797)

## Detailed stage reference
Technical reference for all 6 stages in the template-based-sdg workflow.

### Overview

This workflow scales 565 seed questions into 80K-300K+ synthetic Q&A pairs:

| Stage | Purpose | Key Concept | Input | Output |
|-------|---------|-------------|-------|--------|
| 0. **create_seed_data** | Download seeds + company metadata | Gets **SubIndustry** info for pairing | HuggingFace, Wikipedia, SEC API | seed_questions.jsonl, company_info.tsv |
| 1. **generate_questions** | Generate company×year combinations | Uses **SubIndustry** for comparison pairs | Seed questions + company info | Thousands of company-specific questions |
| 2. **map_questions_to_context** | Find relevant filing sections | Maps to SEC Items 1, 1A, 7, etc. | Generated questions + SEC filings | Questions + filing context |
| 3. **generate_answers** | Generate answer candidates | Creates **5 diverse answers** per question | Questions + context | Multiple answer candidates |
| 4. **genselect_answers** | Select best answer | Large model evaluation | Answer candidates | One best answer per question |
| 5. **filter_answers** | Remove unanswerable/low-quality Q&A | Quality filtering | Generated Q&A pairs | Final training data |

#### Pipeline Flow

**The Template-Based Approach:**

1. **Prepare Seeds** (Stage 0) → Download 565 validated financial questions + S&P 100 companies **with SubIndustry**
2. **Generate Questions** (Stage 1) → Two-step process:
   - **Step 1**: Generate combinations - Create company×year combinations (single companies or pairs from same SubIndustry)
   - **Step 2**: Use LLM to paraphrase - Adapt each seed question to specific companies/years (e.g., "What was NVIDIA's 2023 revenue?" or "Compare NVIDIA vs AMD's 2023 revenue")
3. **Map Context** (Stage 2) → Find relevant SEC filing sections for each question based on the section mapping from SecQue
4. **Generate Answers** (Stage 3) → Create **5 candidate answers** per question (different random seeds for diversity)
5. **Select Best** (Stage 4) → Large model picks highest-quality answer from 5 candidates
6. **Filter** (Stage 5) → Remove unanswerable/low-quality Q&A pairs

**Scale Example:**
- 565 seeds × up to 100 companies (or same-SubIndustry pairs) × 6 years (2019-2024) = ~500K questions in production (before filtering)
- After filtering: 80K-300K final Q&A pairs

**Why SubIndustry matters:** Comparison questions need meaningful company pairs (tech vs tech, bank vs bank). SubIndustry classifications ensure we compare similar companies, making questions realistic and answerable. It also avoid combinatorial explosion from comparing every company with every other company.

### Quick Navigation

- [create_seed_data](#create_seed_data)
- [generate_questions](#generate_questions)
- [map_questions_to_context](#map_questions_to_context)
- [generate_answers](#generate_answers)
- [genselect_answers](#genselect_answers)
- [filter_answers](#filter_answers)

---

### create_seed_data

**File:** `projects/finance/nvfinance/stages/sdg/create_seed_data.py`
**Binding:** workflow alias `create_seed_data` selects `CreateSeedDataStage`

#### Purpose

Create seed data files by downloading the SecQue dataset from HuggingFace, fetching filing metadata from SEC EDGAR API, and scraping S&P 100 company information from Wikipedia. Supports optional filtering by company list and number of questions for demo/testing purposes.

#### What This Stage Does

This stage prepares the foundation for question generation by gathering seed questions and company metadata with industry classifications:

**Downloads performed:**
1. **SecQue dataset** (from HuggingFace): 565 pre-validated financial questions from real SEC filings
2. **S&P 500 company list** (from Wikipedia): To get **GICS SubIndustry** classifications (Technology Hardware, Interactive Media, Banking, etc.)
3. **S&P 100 company list** (from Wikipedia): To filter down to 100 major companies (manageable scale)
4. **SEC EDGAR metadata** (from SEC API): Report dates and form types for each seed question's original filing

**Why both S&P 500 and S&P 100?**
- Fetch **S&P 500** → Get the **SubIndustry** information (not available in S&P 100 data)
- Fetch **S&P 100** → Identify which companies to focus on (avoiding 500 companies = avoiding too many combinations)
- Final output: **S&P 100 companies with their SubIndustry classifications** (merged data)

**Why SubIndustry is critical:**

The next stage (`generate_questions`) creates two types of question variations:

1. **Single-company questions**: Applied to each company individually
   - Seed: "What was the company's revenue in the fiscal year?"
   - Generated: "What was NVIDIA's revenue in fiscal year 2023?"
   - Creates: 1 question per company × year combination

2. **Comparison questions**: Require pairs of companies from the **same SubIndustry**
   - Seed: "Compare the year-over-year revenue growth of two companies"
   - Generated: "Compare NVIDIA vs AMD's year-over-year revenue growth" (both in "Semiconductors" SubIndustry)
   - Avoids: "Compare NVIDIA vs Bank of America" (nonsensical - different industries)
   - Creates: 1 question per company pair × year combination

By pairing companies in the same SubIndustry, we ensure meaningful comparisons: Tech vs tech, bank vs bank, pharmaceutical vs pharmaceutical etc

**Impact on scale:**
- 7 companies (demo) → ~50 possible same-SubIndustry pairs → ~2K questions generated
- 100 companies (S&P 100) → ~1000+ possible same-SubIndustry pairs → ~80K-300K questions generated depending on number of years

#### Inputs

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `output_file` | path | Yes | Output path for seed questions JSONL file |
| `company_info_file` | path | Yes | Output path for company info TSV file |
| `sec_identity_email` | string | Yes | Your email from `SEC_IDENTITY_EMAIL` |
| `sec_identity_company` | string | Yes | Your company from `SEC_IDENTITY_COMPANY` |
| `filter_company_list` | string | No | Comma-separated list of tickers to filter to (e.g., "AAPL,GOOG,MSFT") |
| `num_seed_questions` | int | No | Number of seed questions to sample (uses fixed seed for reproducibility) |

#### Outputs

**Seed questions file** (`seed_questions.jsonl`):
```json
{
  "QID": "q_001",
  "original_question": "Compare the year-over-year change in liquidity ratios...",
  "question_type": "Comparison",
  "accession_number": "0001262039-24-000014;0001560327-24-000021",
  "item": "ITEM 8. Financial Statements...",
  "form_types": "10-K;10-K",
  "report_dates": "2023-12-31;2023-12-31"
}
```

**Company info file** (`company_info.tsv`):
```
Ticker  Company                 Sector              SubIndustry
AAPL    Apple Inc.              Information Tech    Technology Hardware
GOOG    Alphabet Inc. (Class C) Communication Svcs  Interactive Media
...
```

#### Behavior

- **Skips creation if files exist**: If output files already exist, loads them instead of re-fetching
- **Filtering applies after load**: Filtering by company list or question count applies even to existing files
- **Fixed random seed**: Question sampling uses seed=42 for reproducibility

#### Resources

- **Compute:** CPU only
- **Network:** Requires access to HuggingFace, SEC EDGAR API, and Wikipedia
- **Runtime:** 1-2 min (depends on SEC EDGAR API response time)

---

### generate_questions

**File:** `projects/finance/nvfinance/stages/sdg/generate_questions.py`
**Binding:** workflow alias `generate_questions` selects `GenerateQuestionsStage`

#### Purpose

Generate financial questions by adapting seed questions to different companies and years.

#### How It Works

This stage has two sub-steps:

1. **Prepare combinations** (utility script: `prepare_question_gen_data.py`):
   - For each seed question, generate all company×year combinations
   - **Single-company questions**: Generate 1 entry per company × year
   - **Comparison questions**: Generate 1 entry per company **pair** × year
     - Pairs are created from companies in the **same SubIndustry** (from Stage 0 output)
     - Example: NVIDIA + AMD (both "Semiconductors") → valid pair
     - Example: NVIDIA + JPMorgan Chase → **not** a valid pair (different SubIndustries)
   - Output: JSONL with all combinations to generate

2. **Generate questions** (LLM inference):
   - Use LLM to adapt each seed question to the specific company/companies and year
   - Prompt includes company names, year, and the seed question template
   - Output: Company/year-specific questions ready for context mapping

**Scale:** 565 seeds × up to 100 companies × 6 years (2019-2024) in production; the actual number of questions depends on question types and SubIndustry groupings

#### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_file` | path | Seed questions JSONL file |
| `output_file` | path | Output JSONL file for generated questions |
| `company_info_file` | path | Company metadata TSV file |
| `start_year` | int | Start year for questions |
| `end_year` | int | End year for questions |
| `prompt_config` | path | Prompt template YAML |

#### Outputs

JSONL file with generated questions:
```json
{
  "question": "What was NVIDIA's revenue in 2023?",
  "company": "NVDA",
  "year": 2023,
  "seed_question_id": "sq_001"
}
```

#### Resources

- **GPUs:** 8 (configurable)
- **Model:** GPT-OSS-120B or similar
- **Runtime:** 30 min (demo), 2 hours (production)

---

### map_questions_to_context

**File:** `projects/finance/nvfinance/stages/sdg/map_questions_to_context.py`
**Binding:** workflow alias `map_questions_to_context` selects `MapQuestionsToContextStage`

#### Purpose

Map generated questions to relevant sections in SEC filings to provide grounding context. For example if the original question from SECQUE was mapped to Item 1 of form 10-K for a particular year and company, the new question will be mapped to Item 1 of 10-K of the new company and year. However if no such section was found (due to download or parsing error) we drop this question. It is also possible that the mapped section does not actually have the answer to the question, this is handled in the last step `filter_answers`.

#### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_file` | path | Questions from generate_questions |
| `output_file` | path | Output with questions + context |
| `filings_dir` | path | Directory with SEC filings (`/workspace/outputs/finance/demo/workflow-2-download-sec/step-0-download/data`) |
| `filings_metadata` | path | Metadata parquet file |
| `token_limit` | int | Max context length to truncate this section from the start (default: 30000) |

#### Outputs

JSONL file with questions and context:
```json
{
  "question": "What was NVIDIA's revenue in 2023?",
  "context": "NVDA 2023 10-k form Item 1 excerpt...",
  "company": "NVDA",
  "year": 2023,
  "form_type": "10-K",
  "item": "Item 1"
}
```

#### Resources

- **Compute:** CPU only
- **Runtime:** 10-30 min

---

### generate_answers

**File:** `projects/finance/nvfinance/stages/sdg/generate_answers.py`
**Binding:** workflow alias `generate_answers` selects `GenerateAnswersStage`

#### Purpose

Generate multiple candidate answers for each question using an LLM. Uses multiple random seeds to create diverse candidates.

#### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_file` | path | Questions + context from map_questions_to_context |
| `output_dir` | path | Directory for generated answers |
| `prompt_config` | path | Answer generation prompt |
| `num_random_seeds` | int | Number of answer candidates per question (default: 5) |

#### Outputs

Multiple JSONL files in output directory:
```
output_dir/
├── output-rs0.jsonl
├── output-rs1.jsonl
├── output-rs2.jsonl
├── output-rs3.jsonl
└── output-rs4.jsonl
```

Each with answers:
```json
{
  "question": "What was NVIDIA's revenue in 2023?",
  "context": "...",
  "generation": "$X billion",
}
```

#### Resources

- **GPUs:** 8 (configurable)
- **Model:** GPT-OSS-120B or similar
- **Runtime:** 2-4 hours (production)

---

### genselect_answers

**File:** `projects/finance/nvfinance/stages/sdg/genselect_answers.py`
**Binding:** workflow alias `genselect_answers` selects `GenselectAnswersStage`

#### Purpose

Select the best answer from multiple candidates using a large language model as a judge. Implements the [GenSelect](https://arxiv.org/abs/2507.17797) approach.

#### How Selection Works

For each question, this stage:

1. **Receives 5 candidate answers** from the previous stage (generated with different random seeds)
2. **Uses a large evaluation model** to judge quality
   - The evaluation model should be **different from and larger than** the generation model
   - Larger models are better at evaluating answer quality, accuracy, and reasoning
3. **Selects the single best answer** based on:
   - Accuracy (correct information from the filing context)
   - Completeness (fully answers the question)
   - Clarity (well-structured reasoning and explanation)

#### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_dir` | path | Directory with answer candidates from generate_answers |
| `output_file` | path | Output file with selected best answers |
| `prompt_config` | path | GenSelect prompt template |

#### Outputs

JSONL file with best selected answers:
```json
{
  "problem": "What was NVIDIA's revenue in 2023?",
  "context": "...",
  "generation": "$X billion",
  "selected_index": 5,
  "genselect_answer_metadata":{"reasoning_content": "We have to choose from the following candidates..."}
}
```

#### Resources

- **GPUs:** 8-16 (configurable); production uses 16 on 2 nodes
- **Model:** LLM different from answer generation stage; production uses the sharded Qwen3-235B-A22B-Instruct-2507 checkpoint
- **Runtime:** 4-8 hours

---

### filter_answers

**File:** `projects/finance/nvfinance/stages/sdg/filter_answers.py`
**Binding:** workflow alias `filter_answers` selects `FilterAnswersStage`

#### Purpose

Filter out unanswerable questions using an LLM judge.

#### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_file` | path | Selected answers from genselect_answers |
| `output_file` | path | Output file with filtered Q&A pairs |
| `prompt_config` | path | Filtering prompt template |

#### Outputs

JSONL file with final Q&A pairs:
```json
{
  "question": "What was NVIDIA's revenue in 2023?",
  "context": "...",
  "generation": "$X billion",
  "filter_tag": "ANSWERABLE",
}
```

#### Resources

- **GPUs:** 2 per chunk in production, across 16 chunks (configurable)
- **Model:** Qwen3-30B-A3B-Instruct-2507 in production
- **Runtime:** 30 min - 1 hour

---

### Pipeline Summary

| Stage | Input | Output | Compute | Runtime |
|-------|-------|--------|---------|---------|
| create_seed_data | HuggingFace + SEC EDGAR | Seed questions + company info | CPU | 5m |
| generate_questions | Seed questions | Generated questions | GPU | 2h |
| map_questions_to_context | Questions | Questions + context | CPU | 30m |
| generate_answers | Questions + context | Answer candidates (5x) | GPU | 4h |
| genselect_answers | Candidates | Best answers | GPU | 4h |
| filter_answers | Best answers | Final Q&A dataset | GPU | 30m |

**Total:** ~8-10 hours for full production run

See [Template-Based SDG Workflow](../workflows/02-template-based-sdg.md) for usage examples and configuration details.
