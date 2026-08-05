# Document-Grounded SDG Stages Reference

Technical reference for all 7 stages in the document-grounded-sdg workflow.

## Quick Navigation

- [dg_sdg_preprocess](#dg_sdg_preprocess)
- [generate_verified_questions](#generate_verified_questions)
- [generate_answers](#generate_answers)
- [gym_genselect_answers](#gym_genselect_answers)
- [evaluate_answers](#evaluate_answers)
- [aggregate_answers](#aggregate_answers)
- [dgsdg_post_process](#dgsdg_post_process)

---

## dg_sdg_preprocess

**File:** `nvflow/generic_stage/sdg/document_grounded/dg_sdg_preprocess.py`
**Registry:** `recipe="finance"`, `workflow="document_grounded_sdg"`, `stage="dg_sdg_preprocess"`

### Purpose

Converts raw SEC 10-K and 10-Q HTML filings into structured JSONL data for question/answer generation. This preprocessing stage performs three main operations:

1. **Chunk HTML files**: Split large documents into token-limited chunks with overlap
2. **Generate CSV lists**: Create file manifests tracking all chunks
3. **Create JSONL data**: Sample chunks following SecQue benchmark distribution

### Inputs

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

### Expected Input Structure

Your SEC filings should follow this structure (created by Workflow 1):

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

### Outputs

```
${output_dir}/
├── chunks/                        # Chunked HTML files (3 formats)
│   ├── markdown/                  # Markdown-converted chunks
│   ├── clean_html/                # Cleaned HTML chunks
│   └── original_html/             # Original HTML chunks
├── csv/                           # File lists for tracking
│   ├── 10k_1company.csv
│   ├── 10k_2company.csv
│   ├── 10q_1company.csv
│   └── 10q_2company.csv
└── jsonl/                         # Sampled training data ← Used by next stage
    └── training_data.jsonl
```

### Configuration Example

```yaml
dg_sdg_preprocess:
  input_dir: ${filings_dir}/data
  output_dir: ${base_data_dir}/step-0-preprocess
  distribution_dir: nvflow/recipes/finance/workflows/sdg/dg_sdg_distribution
  preprocess_module: nvflow.recipes.finance.utils.sdg.dg_sdg_data_preprocess
  max_tokens: 3000
  overlap_tokens: 500
  total_samples: 150000
  max_skip_count: 20000
  seed: 42

```

### Resources

- **Runtime:** ~2-4 hours for full S&P 500 dataset
- **Compute:** CPU only (no GPUs required)
- **Output Size:** ~10GB JSONL data for 150K samples

### Notes

- Distribution CSVs define sampling proportions matching SecQue benchmark
- Chunking with overlap ensures context continuity across boundaries
- `max_skip_count` prevents infinite loops when distribution can't be satisfied

---

## generate_verified_questions

**File:** `nvflow/generic_stage/sdg/document_grounded/generate_verified_questions.py`
**Registry:** `recipe="finance"`, `workflow="document_grounded_sdg"`, `stage="generate_verified_questions"`

### Purpose

Q-side of the DG-SDG pipeline. Generates questions from SEC filing documents and verifies their quality. Executes 4 internal sub-steps.

### Internal Sub-Steps

1. **Q-prep** (CPU): Run the recipe-supplied `question_prep_script` to attach `context` strings to each chunk
2. **Q-gen** (GPU): Generate questions from documents
3. **Q-verify-prep** (CPU): Expand each generated question into N verification trials
4. **Q-verify** (GPU): Per-question Yes/No vote with multiple random seeds

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_folder` | path | Preprocessed JSONL data directory from `dg_sdg_preprocess` (`${base_data_dir}/step-0-preprocess/jsonl/`) |
| `output_dir` | path | Q-pipeline output directory (e.g. `${base_data_dir}/step-1-questions`) |
| `question_prep_script` | path | Domain wrapper that injects `context_builder` into `lib.sdg.document_grounded.preprocess.construct_question_generate_input` |
| `gym_path` / `gym_config_paths` / `gym_agent_name` | various | NeMo-Gym defaults; per-substep `question_generation_*` / `question_verify_*` overrides allowed |
| `question_generation_kwargs` | dict | GPU settings for question generation |
| `question_verify_kwargs` | dict | GPU settings for verification (typically 5 seeds) |

### Outputs

```
${output_dir}/
├── generate_input.jsonl   # step 1 output (q-prep)
├── generated/             # step 2 output (Q-gen rollouts)
├── verify_input.jsonl     # step 3 output (q-verify-prep)
└── verified/              # step 4 output (Q-verify rollouts)
                           # ← consumed by generate_answers
```

### Configuration Example

```yaml
generate_verified_questions:
  input_folder: ${base_data_dir}/step-0-preprocess/jsonl
  output_dir: ${base_data_dir}/step-1-questions
  dependencies: [dg_sdg_preprocess]

  question_prep_script: nvflow/recipes/finance/utils/sdg/sec_question_prep.py
  gym_path: *gym_path
  gym_config_paths: *gym_config_paths_format_verification
  gym_agent_name: *gym_agent_format_verification

  question_generation_kwargs:
    args:
      model: /models/gpt-oss-120b
      num_gpus: 8
      num_chunks: 5
      num_random_seeds: 1
    ctx_args: >-
      ++prompt_config=nvflow/recipes/finance/prompts/document_grounded_generate_questions.yaml
      ++inference.temperature=0.9

  question_verify_kwargs:
    args:
      model: /models/Qwen3-235B
      num_gpus: 8
      num_chunks: 5
      num_random_seeds: 5
```

### Resources

- **Runtime:** ~2-4 hours
- **GPUs:** 40 for question generation, 200 for question verification
- **Models:** GPT-OSS-120B (questions), Qwen3-235B (verification)

---

## generate_answers

**File:** `nvflow/generic_stage/sdg/document_grounded/generate_answers.py`
**Registry:** `recipe="finance"`, `workflow="document_grounded_sdg"`, `stage="generate_answers"`

### Purpose

A-side of the DG-SDG pipeline. Filters questions by verification pass-rate, then generates N candidate answers per surviving question. Executes 2 internal sub-steps.

### Internal Sub-Steps

1. **A-prep** (CPU): `construct_answer_generate_input` keeps only questions whose Q-verify pass-rate ≥ `threshold`
2. **A-gen** (GPU): Generate answers (typically 5 seeds for downstream genselect)

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_dir` | path | Verified-questions directory from `generate_verified_questions` (`${base_data_dir}/step-1-questions/verified`) |
| `output_dir` | path | A-pipeline output directory (e.g. `${base_data_dir}/step-2-answers`) |
| `gym_path` / `gym_config_paths` / `gym_agent_name` | various | NeMo-Gym defaults; per-substep `answer_generation_*` overrides allowed |
| `answer_preprocess_kwargs` | dict | CPU settings, includes `threshold` (Q-verify pass-rate cutoff) |
| `answer_generation_kwargs` | dict | GPU settings for answer generation |

### Outputs

```
${output_dir}/
├── answer_input.jsonl   # step 1 output (a-prep)
└── generated/           # step 2 output (A-gen rollouts; consumed by gym_genselect_answers)
    ├── output-rs0.jsonl
    └── ...
```

### Configuration Example

```yaml
generate_answers:
  input_dir: ${base_data_dir}/step-1-questions/verified
  output_dir: ${base_data_dir}/step-2-answers
  dependencies: [generate_verified_questions]

  gym_path: *gym_path
  gym_config_paths: *gym_config_paths_format_verification
  gym_agent_name: *gym_agent_format_verification

  answer_preprocess_kwargs:
    threshold: 1

  answer_generation_kwargs:
    args:
      model: /models/gpt-oss-120b
      num_gpus: 8
      num_chunks: 5
      num_random_seeds: 5
    ctx_args: >-
      ++prompt_config=nvflow/recipes/finance/prompts/secque_template.yaml
```

### Resources

- **Runtime:** ~2-4 hours
- **GPUs:** 200 (5 seeds, 5 chunks each)
- **Model:** GPT-OSS-120B

---

## gym_genselect_answers

**File:** `nvflow/generic_stage/sdg/document_grounded/gym_genselect_answers.py`
**Registry:** `recipe="finance"`, `workflow="document_grounded_sdg"`, `stage="gym_genselect_answers"`

### Purpose

Select the best answer from the multiple candidates produced by `generate_answers` (DG-SDG-specific best-of-N picker that runs through NeMo-Gym).

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_dir` | path | Answer candidates from `generate_answers` |
| `output_file` | path | Selected answers output file |
| `prompt_template` | path | GenSelect prompt |

### Outputs

JSONL file with selected best answers.

### Configuration Example

```yaml
  gym_genselect_answers:
    input_dir: ${base_data_dir}/step-2-answers/generated
    output_file: ${base_data_dir}/step-3-genselect/selected_answers.jsonl
    prompt_template: nvflow/recipes/finance/prompts/genselect_answers.yaml
    dependencies: [generate_answers]

    policy_vllm:
      model_path: /models/Qwen3-235B-A22B-Instruct-2507
      num_gpus: 8
      server_nodes: 1
    num_chunks: 5
    inference_params:
      max_output_tokens: 16384
```

### Resources

- **GPUs:** 120
- **Model:** Qwen3-235B
- **Runtime:** 2-4 hours

---

## evaluate_answers

**File:** `nvflow/generic_stage/sdg/document_grounded/evaluate_answers.py`
**Registry:** `recipe="finance"`, `workflow="document_grounded_sdg"`, `stage="evaluate_answers"`

### Purpose

Evaluate answer quality using a large model judge. Runs 5 random seeds for robustness. Each seed's judge response ends with a JSON verdict tag `{"answerable": "YES/NO", "correct": "YES/NO"}` (parsed downstream by `aggregate_answers`).

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_file` | path | Selected answers from `gym_genselect_answers` |
| `output_dir` | path | Directory for evaluation results |
| `prompt_template` | path | Evaluation prompt |

### Outputs

```
${output_dir}/
├── output-rs0.jsonl
├── output-rs1.jsonl
├── output-rs2.jsonl
├── output-rs3.jsonl
└── output-rs4.jsonl
```

Each record carries the judge's raw `evaluate_generation`, whose last line is the JSON verdict tag parsed by `aggregate_answers`:
```json
{
  "problem": "...",
  "generation": "...",
  "evaluate_generation": "...reasoning...\n{\"answerable\": \"YES\", \"correct\": \"YES\"}"
}
```

### Configuration Example

```yaml
  evaluate_answers:
    input_file: ${base_data_dir}/step-3-genselect/selected_answers.jsonl
    output_dir: ${base_data_dir}/step-4-evaluate
    prompt_template: nvflow/recipes/finance/prompts/evaluate_answers.yaml
    generation_key: evaluate_generation
    dependencies: [gym_genselect_answers]

    policy_vllm:
      model_path: /models/Qwen3-235B-A22B-Instruct-2507
      num_gpus: 8
      server_nodes: 1
    num_chunks: 1
    num_random_seeds: 5
    inference_params:
      top_p: 0.9
      temperature: 0.8
```

### Resources

- **GPUs:** 200
- **Model:** Qwen3-235B
- **Runtime:** 8 hours

---

## aggregate_answers

**File:** `nvflow/generic_stage/sdg/document_grounded/aggregate_answers.py`
**Registry:** `recipe="finance"`, `workflow="document_grounded_sdg"`, `stage="aggregate_answers"`

### Purpose

Aggregate the 5 evaluate seeds: keep a question only if **all** seeds voted `correct=YES` with a consistent `answerable`, and attach the consensus `answerable`.

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_dir` | path | Evaluation results from `evaluate_answers` |
| `output_file` | path | Aggregated results output |

### Outputs

A single JSONL file of surviving records. The per-seed `evaluate_generation` / `correct` are dropped and a consensus `answerable` is added:
```json
{
  "problem": "...",
  "generation": "...",
  "reference_answer": "...",
  "answerable": "YES"
}
```

### Resources

- **Compute:** CPU only
- **Runtime:** 5-10 min

---

## dgsdg_post_process

**File:** `nvflow/generic_stage/sdg/document_grounded/dgsdg_post_process.py`
**Registry:** `recipe="finance"`, `workflow="document_grounded_sdg"`, `stage="dgsdg_post_process"`

### Purpose

Clean and rename fields, then emit a single `final_result.jsonl` consumed by downstream SFT / GRPO workflows. Records are not split into subsets; the per-stage trim (see `_schemas.py::STAGE_KEEP["dgsdg_post_process"]`) plus the recipe's `domain_keep_fields` defines the final allowlist of fields kept in `final_result.jsonl`.

### Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_file` | path | Aggregated answers from `aggregate_answers` (e.g. `${base_data_dir}/step-5-aggregate/aggregated_answers.jsonl`) |
| `output_dir` | path | Output directory for `final_result.jsonl` |
| `postprocess_script` | path | Domain CLI wrapper around `nvflow.lib.sdg.document_grounded.postprocess.dgsdg_post_process` (e.g. `recipes/finance/utils/sdg/sec_postprocess.py`) |
| `seed` | int | Random seed for reproducibility (default: 42) |
| `domain_keep_fields` | list[str] | Recipe-specific fields appended to the generic allowlist before per-stage trim |

### Outputs

```
${output_dir}/
└── final_result.jsonl    # Single cleaned + renamed dataset consumed by SFT / GRPO
```

### Resources

- **Compute:** CPU only
- **Runtime:** 5-10 min

---

## Pipeline Summary

| Stage | Purpose | Compute | Runtime |
|-------|---------|---------|---------|
| dg_sdg_preprocess | Chunk + sample documents | CPU | 2-4h |
| generate_verified_questions | Generate + verify questions | GPU | 2-4h |
| generate_answers | Generate candidate answers | GPU | 2-4h |
| gym_genselect_answers | Select best answers | GPU | 1-2h |
| evaluate_answers | Evaluate quality | GPU | 2-3h |
| aggregate_answers | Aggregate scores | CPU | 10m |
| dgsdg_post_process | Clean + rename → final_result.jsonl | CPU | 10m |

**Total:** ~10-12 hours for full production run

See [Document-Grounded SDG Workflow](../workflows/03-document-grounded-sdg.md) for usage examples and configuration details.
