# Finance Recipe

## Overview

End-to-end pipeline for generating synthetic financial Q&A data from SEC filings, training financial reasoning models, and evaluating them on benchmarks.

### Key Capabilities

**Two Independent SDG Approaches:**
- **Template-Based SDG:** Adapts seed questions to different companies/years, maps to relevant context, generates and filters answers
- **Document-Grounded SDG:** Generates questions directly from documents with built-in verification, quality evaluation, and difficulty stratification

**Production-Ready Pipeline:**
- **Data Generation:** Uses GPT-OSS-120B, Qwen3 (14B-235B) models for synthetic Q&A creation
- **Scale:** Processes S&P 500 companies (~100GB filings) → generates 1M+ Q&A pairs
- **Training:** Full SFT pipeline on 256 GPUs (32 nodes) with Qwen3-14B
- **Evaluation:** Benchmark trained models on financial reasoning tasks

## What This Recipe Produces

- **Synthetic Q&A Datasets**: 1M+ high-quality financial question-answer pairs
  - Template-based SDG: ~300K pairs (used in production SFT)
  - Document-grounded SDG: ~800K pairs (SFT integration in progress)
- **Fine-tuned Models**: Financial reasoning models trained via supervised fine-tuning (SFT)
- **RL-trained Models**: Models further improved via GRPO reinforcement learning with LLM-as-judge rewards
- **Evaluation Results**: Model performance on financial benchmarks (SFT and GRPO checkpoints)

## Pipeline Architecture

### High-Level Flow

```
┌─────────────────┐
│  download-sec   │  Download SEC filings (10-K, 10-Q, 8-K)
└────────┬────────┘
         ↓
    ┌────┴────┐
    │         │
    ↓         ↓
┌─────────┐ ┌──────────────────┐
│template-│ │document-grounded-│  Generate synthetic Q&A
│based-sdg│ │      sdg         │  (two independent approaches)
└────┬────┘ └────────┬─────────┘
     │               │
     └───────┬───────┘
             ↓
      ┌──────────────────┐
      │ eval (baselines) │  Evaluate pre-trained models
      └──────────────────┘
             ↓
      ┌──────────────────┐
      │   sft + eval     │  Fine-tune + evaluate checkpoints
      └─────┬────────────┘
            │
            ↓
      ┌──────────────────┐
      │  grpo + eval     │  RL training + evaluate checkpoints
      └──────────────────┘
```

### Detailed Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Download SEC Filings                                         │
├─────────────────────────────────────────────────────────────────┤
│ Input: Ticker list (demo.yaml / sp500.yaml)                     │
│ Output: outputs/finance/{demo,sap-500}/workflow-2-download-sec/ │
│         - Company directories with 10-K, 10-Q, 8-K JSON files   │
│         - sec_metadata.parquet                                  │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
          ┌──────────────┴──────────────┐
          ↓                             ↓
┌──────────────────────┐    ┌───────────────────────────┐
│ 2a. Template-Based   │    │ 2b. Document-Grounded     │
│     SDG (6 stages)   │    │     SDG (7 stages)        │
├──────────────────────┤    ├───────────────────────────┤
│ • Generate questions │    │ • Preprocess filings      │
│ • Map to context     │    │ • Generate verified Q&A   │
│ • Generate answers   │    │ • GenSelect answers       │
│ • GenSelect answers  │    │ • Evaluate quality        │
│ • Filter quality     │    │ • Aggregate results       │
│                      │    │ • Estimate difficulty     │
│                      │    │ • Prepare training data   │
├──────────────────────┤    ├───────────────────────────┤
│ Output: ~300K Q&A    │    │ Output: ~800K Q&A         │
│ [Used in SFT]        │    │ Stratified by difficulty  │
│                      │    │ [Work in progress]        │
│                      │    │                           │
└──────────────────────┘    └───────────────────────────┘
                  │
                  │ (Production path)
                  ↓
           ┌────────────────────────────────────────┐
           │ 3. Supervised Fine-Tuning (6 stages)   │
           ├────────────────────────────────────────┤
           │ • Data transformation                  │
           │ • Prepare for SFT                      │
           │ • Train/validation split               │
           │ • [Optional] Sequence grouping         │
           │ • Training                             │
           │ • Eval (checkpoint + baseline)         │
           ├────────────────────────────────────────┤
           │ Output: Fine-tuned model + eval results│
           └────────────┬───────────────────────────┘
                        │
                        ↓
           ┌────────────────────────────────────────┐
           │ 4. GRPO RL Training (8 stages)         │
           ├────────────────────────────────────────┤
           │ • Prepare data (agent routing)         │
           │ • Collect rollouts + reward analysis   │
           │ • [Optional] Re-compute rewards        │
           │ • GRPO training with NeMo-Gym          │
           │ • Eval (checkpoint + baseline)         │
           ├────────────────────────────────────────┤
           │ Output: RL-trained model + eval results│
           └────────────────────────────────────────┘
```

## 6 Workflows

| # | Workflow | Purpose | Stages | GPUs |
|---|----------|---------|--------|------|
| 1 | [download-sec](workflows/01-download-sec.md) | Download SEC filings from EDGAR | 1 | -- |
| 2 | [template-based-sdg](workflows/02-template-based-sdg.md) | Generate Q&A from seed questions | 6 | 16 |
| 3 | [document-grounded-sdg](workflows/03-document-grounded-sdg.md) | Generate verified Q&A from documents | 7 | 8 |
| 4 | [sft](workflows/04-sft.md) | Supervised fine-tuning + checkpoint eval | 6 | 256 |
| 5 | [eval](workflows/05-eval.md) | Baseline model evaluation | 7 | 8 |
| 6 | [grpo](workflows/06-grpo.md) | GRPO RL training + checkpoint eval | 8 | 8 |

> **Note:** GPU counts show the maximum requirement for any single stage in the workflow (i.e., minimum GPUs needed to run the pipeline).

> **Production Pipeline:** download-sec → template-based-sdg → eval (baselines) → sft (+ eval) → grpo (+ eval)
> **Experimental:** document-grounded-sdg (SFT integration in progress)

## Getting Started

### 🎥 Video Tutorials

> 📹 **Coming Soon:** Video walkthroughs of the complete pipeline
> - [ ] Quick Start Demo
> - [ ] Download SEC Filings
> - [ ] Template-Based SDG Explained
> - [ ] Document-Grounded SDG Explained
> - [ ] Model Training & Evaluation
> - [ ] Production Deployment Guide

### 🚀 First Time Users

**[Quick Start Guide](quick-start.md)** - Run complete demo with 7 companies

### 📖 Understanding the Workflows

Learn what each workflow does and how to use it:

1. **[Download SEC Filings](workflows/01-download-sec.md)** - Download financial documents
2. **[Template-Based SDG](workflows/02-template-based-sdg.md)** - Generate Q&A from templates
3. **[Document-Grounded SDG](workflows/03-document-grounded-sdg.md)** - Generate verified Q&A
4. **[SFT Training](workflows/04-sft.md)** - Fine-tune models on synthetic data
5. **[Evaluation](workflows/05-eval.md)** - Benchmark model performance
6. **[GRPO RL Training](workflows/06-grpo.md)** - Reinforcement learning with NeMo-Gym

### 🔧 Technical Reference

Detailed technical specifications for each stage:

- **[Download-SEC Stages](stages/download-sec.md)** - 1 stage
- **[Template-Based SDG Stages](stages/template-based-sdg.md)** - 6 stages
- **[Document-Grounded SDG Stages](stages/document-grounded-sdg.md)** - 7 stages
- **[SFT Stages](stages/sft.md)** - 6 stages
- **[Eval Stages](stages/eval.md)** - 9 stages
- **[GRPO Stages](stages/grpo.md)** - 9 stages

## Quick Command Reference

```bash
# List all available stages
uv run nflow list-stages --recipe finance

# Run download workflow (S&P 500)
uv run nflow run sap-500 --config nvflow/recipes/finance/workflows/download_sec_filings.yaml

# Run SDG workflow (demo)
uv run nflow run-all --config nvflow/recipes/finance/workflows/sdg/template-based-sdg-demo.yaml

# Run SFT training + checkpoint eval
uv run nflow run-all --config nvflow/recipes/finance/workflows/sft/qwen3_14b.yaml

# Run baseline evaluation only
uv run nflow run-all --config nvflow/recipes/finance/workflows/eval/baselines.yaml

# Run GRPO RL training + checkpoint eval
uv run nflow run-all --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml
```

## Need Help?

- 📖 **[Workflow Guides](workflows/)** - How to run each workflow
- 🔧 **[Stage Reference](stages/)** - Technical specifications
- 🚨 **[Troubleshooting Guide](troubleshooting.md)** - Solutions to common issues
- 🐛 **[GitHub Issues](https://github.com/NVIDIA/nvflow/issues)** - Report bugs or request features
