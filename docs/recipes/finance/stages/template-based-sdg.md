# Template-Based SDG Stages Reference

Technical reference for all 6 stages in the template-based-sdg workflow.

## Overview

This workflow scales 565 seed questions into 80K-300K+ synthetic Q&A pairs:

| Stage | Purpose | Key Concept | Input | Output |
|-------|---------|-------------|-------|--------|
| 0. **create_seed_data** | Download seeds + company metadata | Gets **SubIndustry** info for pairing | HuggingFace, Wikipedia, SEC API | seed_questions.jsonl, company_info.tsv |
| 1. **generate_questions** | Generate company×year combinations | Uses **SubIndustry** for comparison pairs | Seed questions + company info | Thousands of company-specific questions |
| 2. **map_questions_to_context** | Find relevant filing sections | Maps to SEC Items 1, 1A, 7, etc. | Generated questions + SEC filings | Questions + filing context |
| 3. **generate_answers** | Generate answer candidates | Creates **5 diverse answers** per question | Questions + context | Multiple answer candidates |
| 4. **genselect_answers** | Select best answer | Large model evaluation | Answer candidates | One best answer per question |
| 5. **filter_answers** | Remove unanswerable/low-quality Q&A | Quality filtering | Generated Q&A pairs | Final training data |

### Pipeline Flow

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
- 300 seeds × 100 companies × 4 years = ~120K questions (before filtering)
- After filtering: 80K-300K final Q&A pairs

**Why SubIndustry matters:** Comparison questions need meaningful company pairs (tech vs tech, bank vs bank). SubIndustry classifications ensure we compare similar companies, making questions realistic and answerable. It also avoid combinatorial explosion from comparing every company with every other company.

## Quick Navigation

- [create_seed_data](#create_seed_data)
- [generate_questions](#generate_questions)
- [map_questions_to_context](#map_questions_to_context)
- [generate_answers](#generate_answers)
- [genselect_answers](#genselect_answers)
- [filter_answers](#filter_answers)

---

## create_seed_data

**File:** `nvflow/recipes/finance/stages/sdg/create_seed_data.py`
**Registry:** `recipe="finance"`, `workflow="template_based_sdg"`, `stage="create_seed_data"`

### Purpose

Create seed data files by downloading the SecQue dataset from HuggingFace, fetching filing metadata from SEC EDGAR API, and scraping S&P 100 company information from Wikipedia. Supports optional filtering by company list and number of questions for demo/testing purposes.

### What This Stage Does

This stage prepares the foundation for question generation by gathering seed questions and company metadata with industry classifications:

**Downloads performed:**
1. **SecQue dataset** (from HuggingFace): ~300 pre-validated financial questions from real SEC filings
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

### Inputs

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `output_file` | path | Yes | Output path for seed questions JSONL file |
| `company_info_file` | path | Yes | Output path for company info TSV file |
| `sec_identity_email` | string | Yes | Your email address (required by SEC EDGAR API) |
| `sec_identity_company` | string | Yes | Your company/organization name |
| `filter_company_list` | string | No | Comma-separated list of tickers to filter to (e.g., "AAPL,GOOG,MSFT") |
| `num_seed_questions` | int | No | Number of seed questions to sample (uses fixed seed for reproducibility) |

### Outputs

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

### Behavior

- **Skips creation if files exist**: If output files already exist, loads them instead of re-fetching
- **Filtering applies after load**: Filtering by company list or question count applies even to existing files
- **Fixed random seed**: Question sampling uses seed=42 for reproducibility

### Resources

- **Compute:** CPU only
- **Network:** Requires access to HuggingFace, SEC EDGAR API, and Wikipedia
- **Runtime:** 1-2 min (depends on SEC EDGAR API response time)

---

## generate_questions

**File:** `nvflow/recipes/finance/stages/sdg/generate_questions.py`
**Registry:** `recipe="finance"`, `workflow="template_based_sdg"`, `stage="generate_questions"`

### Purpose

Generate financial questions by adapting seed questions to different companies and years.

### How It Works

This stage has two sub-steps:

1. **Prepare combinations** (utility script: `prepare_question_gen_data.py`):
   - For each seed question, generate all company×year combinations
   - **Single-company questions**: Generate 1 entry per company × year
   - **Comparison questions**: Generate 1 entry per company **pair** × year
     - Pairs are created from companies in the **same SubIndustry** (from Stage 0 output)
     - Example: NVIDIA + AMD (both "Semiconductors") → valid pair
     - Example: NVIDIA + Walmart → **not** a valid pair (different SubIndustries)
   - Output: JSONL with all combinations to generate

2. **Generate questions** (LLM inference):
   - Use LLM to adapt each seed question to the specific company/companies and year
   - Prompt includes company names, year, and the seed question template
   - Output: Company/year-specific questions ready for context mapping

**Scale:** 300 seeds × 100 companies × 4 years = potential for ~120K questions (actual number depends on question types and SubIndustry groupings)

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_file` | path | Seed questions JSONL file |
| `output_file` | path | Output JSONL file for generated questions |
| `company_info_file` | path | Company metadata TSV file |
| `start_year` | int | Start year for questions |
| `end_year` | int | End year for questions |
| `prompt_config` | path | Prompt template YAML |

### Outputs

JSONL file with generated questions:
```json
{
  "question": "What was NVIDIA's revenue in 2023?",
  "company": "NVDA",
  "year": 2023,
  "seed_question_id": "sq_001"
}
```

### Resources

- **GPUs:** 8 (configurable)
- **Model:** GPT-OSS-120B or similar
- **Runtime:** 30 min (demo), 2 hours (production)

---

## map_questions_to_context

**File:** `nvflow/recipes/finance/stages/sdg/map_questions_to_context.py`
**Registry:** `recipe="finance"`, `workflow="template_based_sdg"`, `stage="map_questions_to_context"`

### Purpose

Map generated questions to relevant sections in SEC filings to provide grounding context. For example if the original question from SECQUE was mapped to Item 1 of form 10-K for a particular year and company, the new question will be mapped to Item 1 of 10-K of the new company and year. However if no such section was found (due to download or parsing error) we drop this question. It is also possible that the mapped section does not actually have the answer to the question, this is handled in the last step `filter_answers`.

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_file` | path | Questions from generate_questions |
| `output_file` | path | Output with questions + context |
| `filings_dir` | path | Directory with SEC filings (`/workspace/outputs/finance/demo/workflow-2-download-sec/step-0-download/data`) |
| `filings_metadata` | path | Metadata parquet file |
| `token_limit` | int | Max context length to truncate this section from the start (default: 30000) |

### Outputs

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

### Resources

- **Compute:** CPU only
- **Runtime:** 10-30 min

---

## generate_answers

**File:** `nvflow/recipes/finance/stages/sdg/generate_answers.py`
**Registry:** `recipe="finance"`, `workflow="template_based_sdg"`, `stage="generate_answers"`

### Purpose

Generate multiple candidate answers for each question using an LLM. Uses multiple random seeds to create diverse candidates.

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_file` | path | Questions + context from map_questions_to_context |
| `output_dir` | path | Directory for generated answers |
| `prompt_config` | path | Answer generation prompt |
| `num_random_seeds` | int | Number of answer candidates per question (default: 5) |

### Outputs

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

### Resources

- **GPUs:** 8 (configurable)
- **Model:** GPT-OSS-120B or similar
- **Runtime:** 2-4 hours (production)

---

## genselect_answers

**File:** `nvflow/recipes/finance/stages/sdg/genselect_answers.py`
**Registry:** `recipe="finance"`, `workflow="template_based_sdg"`, `stage="genselect_answers"`

### Purpose

Select the best answer from multiple candidates using a large language model as a judge. Implements the [GenSelect](https://arxiv.org/abs/2507.17797) approach.

### How Selection Works

For each question, this stage:

1. **Receives 5 candidate answers** from the previous stage (generated with different random seeds)
2. **Uses a large evaluation model** to judge quality
   - The evaluation model should be **different from and larger than** the generation model
   - Larger models are better at evaluating answer quality, accuracy, and reasoning
3. **Selects the single best answer** based on:
   - Accuracy (correct information from the filing context)
   - Completeness (fully answers the question)
   - Clarity (well-structured reasoning and explanation)

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_dir` | path | Directory with answer candidates from generate_answers |
| `output_file` | path | Output file with selected best answers |
| `prompt_config` | path | GenSelect prompt template |

### Outputs

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

### Resources

- **GPUs:** 8-16 (configurable)
- **Model:** LLM different from answer generation stage
- **Runtime:** 4-8 hours

---

## filter_answers

**File:** `nvflow/recipes/finance/stages/sdg/filter_answers.py`
**Registry:** `recipe="finance"`, `workflow="template_based_sdg"`, `stage="filter_answers"`

### Purpose

Filter out unanswerable questions using an LLM judge.

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_file` | path | Selected answers from genselect_answers |
| `output_file` | path | Output file with filtered Q&A pairs |
| `prompt_config` | path | Filtering prompt template |

### Outputs

JSONL file with final Q&A pairs:
```json
{
  "question": "What was NVIDIA's revenue in 2023?",
  "context": "...",
  "generation": "$X billion",
  "filter_tag": "ANSWERABLE",
}
```

### Resources

- **GPUs:** 8 (configurable)
- **Model:** GPT-OSS-120B or similar
- **Runtime:** 30 min - 1 hour

---

## Pipeline Summary

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
