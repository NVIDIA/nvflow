# NVFlow - Architecture Quick Reference

> **One-page overview of NVFlow architecture**
> **For:** Quick onboarding and reference
> **See also:** [ARCHITECTURE.md](./ARCHITECTURE.md) for comprehensive details

---

## 🏗️ System Architecture (3 Layers)

```
┌─────────────────────────────────────────────────────────┐
│  USER LAYER: CLI, Python API, Scripts                   │
├─────────────────────────────────────────────────────────┤
│  FRAMEWORK LAYER: WorkflowRunner, StageRegistry         │
├─────────────────────────────────────────────────────────┤
│  EXECUTION LAYER: NeMo-Skills, Slurm, Containers        │
└─────────────────────────────────────────────────────────┘
```

---

## 📦 Core Components

| Component | Purpose | Key Methods |
|-----------|---------|-------------|
| **BaseStage** | Abstract base for all stages | `execute()`, `validate_config()` |
| **StageRegistry** | Hierarchical stage registry | `register()`, `get()`, `list_*()` |
| **WorkflowRunner** | Orchestrates workflow execution | `run()`, `validate_config()` |
| **Console** | Rich terminal UI | `header()`, `info()`, `success()` |

---

## 🎯 Hierarchical Organization

```
Recipe (Domain: finance, healthcare, retail)
  ↓
Workflow (Pipeline: download, sdg, sft, eval, grpo)
  ↓
Stage (Task: generate_answers, training, evaluate)
```

**Example Path:** `finance.sft.sft` → `SFTStage` class

---

## 📊 Finance Recipe Pipeline (6 Workflows, 42 Stages)

```
1. download-sec (1 stage)
   └─ Download SEC filings → ~100GB JSON

2. template-based-sdg (6 stages) [PRODUCTION]
   └─ Seed → Questions → Context → Answers → Filter → ~300K Q&A

3. document-grounded-sdg (7 stages) [EXPERIMENTAL]
   └─ Preprocess → Generate → Evaluate → ~800K Q&A

4. sft (4 stages + 2 shared)
   └─ Transform → Prepare → Split → Train → Checkpoints

5. eval (2 stages, dynamically expanded)
   └─ Prepare → Evaluate checkpoints → Compare → Results

6. grpo (10 stages: 9 active + 1 optional)
   └─ GRPO reinforcement learning workflow
```

---

## 🚀 Execution Flow (4 Phases)

```
1. INIT: Load config → Resolve inheritance → Extract context
2. VALIDATE: Check registry → Validate stages
3. EXECUTE: For each stage → Submit to Slurm → Track dependencies
4. MONITOR: Check status → View logs → Collect results
```

---

## 💻 Technology Stack

```yaml
Core:
  - Python 3.12+, OmegaConf, Typer, Rich

Execution:
  - NeMo-Skills (SDG & pipelines)
  - Slurm (cluster scheduling)
  - Enroot (containers)

Models:
  - vLLM, SGLang (inference)
  - HuggingFace (model loading)
```

---

## 🗂️ Directory Structure

```
nvflow/
├── core/              # Framework (BaseStage, Registry, Runner)
├── cli/               # CLI interface (nflow commands)
└── recipes/           # Domain-specific implementations
    ├── finance/       # 42 stages, 6 workflows
    │   ├── stages/    # Stage implementations
    │   ├── workflows/ # YAML configs
    │   └── prompts/   # Prompt templates
    └── example/       # Learning & testing
```

---

## 🔧 Common Commands

```bash
# List all stages
nflow list-stages --recipe finance

# Get stage info
nflow stage-info finance.sft.sft

# Run single stage
nflow run sft --config workflow.yaml

# Run all stages
nflow run-all --config workflow.yaml

# Validate config
nflow validate --config workflow.yaml
```

---

## 📝 Creating a New Stage (3 Steps)

```python
# 1. Create stage file: nvflow/recipes/finance/stages/sdg/my_stage.py
from nvflow.core import BaseStage, StageRegistry

# 2. Implement with decorator
@StageRegistry.register(
    recipe="finance",
    workflow="my_workflow",
    stage="my_stage"
)
class MyStage(BaseStage):
    workflow = "my_workflow"

    def execute(self, config, cluster, expname, run_after=None):
        # Your implementation
        pass
```

```yaml
# 3. Add to workflow YAML
recipe: finance
workflow:
  name: my_workflow
pipeline_stages:
  - my_stage
stages:
  my_stage:
    # Your config
```

---

## 🎨 Design Patterns

| Pattern | Usage | Example |
|---------|-------|---------|
| **Template Method** | BaseStage defines interface | `execute()` method |
| **Registry** | Stage discovery | `StageRegistry.get()` |
| **Decorator** | Stage registration | `@StageRegistry.register()` |
| **Strategy** | Execution modes | Local vs. Slurm |
| **Dependency Injection** | Config passing | `execute(config, cluster, ...)` |

---

## 🏭 Deployment Topology

```
Local Machine                    Slurm Cluster
┌──────────────┐                ┌─────────────────────┐
│ nflow CLI    │───SSH───►      │ Login Node          │
│ Config YAML  │                │   ↓                 │
└──────────────┘                │ Slurm Scheduler     │
                                │   ↓                 │
                                │ Compute Nodes       │
                                │ • 8× H100 GPUs      │
                                │ • Enroot containers │
                                │ • Shared storage    │
                                └─────────────────────┘
```

---

## 📊 Data Flow (Finance Recipe)

```
SEC API → Filings (100GB) → SDG (300K Q&A) →
Data Prep → Training → Checkpoints → Evaluation → Results
```

---

## 🔑 Key Features

✅ **Modular**: Reusable stages across workflows
✅ **Declarative**: YAML-based configuration
✅ **Scalable**: Local to multi-node clusters
✅ **Reproducible**: Version-controlled configs
✅ **Extensible**: Easy to add recipes/stages
✅ **Built on NeMo**: Leverages NVIDIA ecosystem

---

## 📚 Documentation Map

| Document | Purpose | Audience |
|----------|---------|----------|
| **README.md** | Getting started | All users |
| **INSTALL.md** | Cluster setup | DevOps, ML Engineers |
| **ARCHITECTURE.md** | Deep dive (26 KB) | Architects, Contributors |
| **ARCHITECTURE_QUICK_REFERENCE.md** | This page | Quick reference |
| **DIAGRAMS_SUMMARY.md** | Diagram guide | Visual learners |
| **diagrams/*.mmd** | Visual diagrams | All users |
| **docs/recipes/finance/** | Finance recipe | Data Scientists |

---

## 🎯 Use Case: Finance Recipe

**Goal:** Generate synthetic financial Q&A data and train reasoning models

**Input:** SEC filings (10-K, 10-Q, 8-K)

**Process:**
1. Download filings (S&P 500)
2. Generate 300K Q&A pairs (template-based SDG)
3. Prepare training data
4. Fine-tune Qwen3-14B (256 GPUs)
5. Evaluate on benchmarks

**Output:** Fine-tuned financial reasoning model + evaluation metrics

**Scale:** ~100GB data → 300K Q&A → 256 GPU training → Production model

---

## 🔗 Quick Links

- **Full Architecture:** [ARCHITECTURE.md](./ARCHITECTURE.md)
- **Diagrams:** [diagrams/](../diagrams/)
- **Finance Recipe:** [docs/recipes/finance/README.md](../recipes/finance/README.md)
- **Quick Start:** [docs/recipes/finance/quick-start.md](../recipes/finance/quick-start.md)
- **NeMo-Skills:** https://github.com/NVIDIA/NeMo-Skills

---

## 💡 Tips

1. **Start with example recipe** for learning
2. **Use `nflow list-stages`** to discover stages
3. **Check `nflow stage-info`** for stage details
4. **Validate configs** before running: `nflow validate`
5. **Monitor jobs** with `squeue` and log files
6. **Pre-download models** to avoid GPU time waste

---

**Version:** 1.0 | **Updated:** Jan 21, 2026 | **License:** Apache-2.0
