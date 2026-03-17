# NVFlow

**Workflow orchestration for end-to-end ML pipelines (data generation, training, evaluation) built on the NeMo ecosystem.**

NVFlow is a workflow orchestration framework for end-to-end synthetic data generation (SDG), training (SFT), and evaluation pipelines built on NVIDIA's NeMo ecosystem. It exists to standardize how teams build, reproduce, and scale complex ML pipelines across domains with reusable stages and declarative workflows that run locally or on Slurm clusters.

It provides a structured way to build, manage, and execute pipelines through:
- **Recipes** - Domain-specific workflows (e.g., finance)
- **Stages** - Reusable pipeline components for SDG, training, and evaluation
- **Workflows** - YAML-based configurations that chain stages together

Key features:
- **Reusability and reproducibility** with a structured, stage-based architecture
- **Flexible execution** via CLI (`nflow`), Python scripts, or programmatic API
- **Cluster integration** with native Slurm support
- **Built on NeMo** leveraging NeMo-Skills and NeMo-RL infrastructure. Data Designer and NeMo-Gym are coming soon.

Example use case: The finance recipe demonstrates a complete pipeline: download SEC filings → generate synthetic Q&A data → fine-tune models → evaluate performance, producing 300K+ synthetic Q&A pairs.

## 🔑 Core Concepts

Understanding the terminology is key to working effectively with NVFlow:

- **Recipe**: A domain-specific collection of stages and workflows organized around a particular use case (e.g., finance). Recipes provide ready-to-use pipelines for specific problem domains.


- **Workflow**: A declarative pipeline defined in YAML that orchestrates multiple stages in a specific order. Workflows define stage dependencies, data flow between stages, and execution configuration. Think of a workflow as a recipe that connects stages together to accomplish an end-to-end objective.

- **Stage**: A self-contained, reusable component that performs a single, specific task in your ML pipeline. Each stage is an independent unit of work (e.g., downloading data, generating synthetic examples, training a model). Stages are implemented as Python classes and can be composed together.

**Example hierarchy:**
```
Recipe: finance
├── Workflow: download_sec_filings (defined in YAML)
│   └── Stage: demo (or sap-500)
└── Workflow: template_based_sdg (defined in YAML)
    ├── Stage: create_seed_data
    ├── Stage: generate_questions
    ├── Stage: map_questions_to_context
    ├── Stage: generate_answers
    ├── Stage: genselect_answers
    └── Stage: filter_answers
```

In practice, you define workflows in YAML configuration files, reference stages by their short names, and run them via CLI or Python API. This separation allows you to reuse stages across different workflows and maintain clean, modular pipeline code.

## 📁 Understanding the Folder Structure

**How concepts map to folders:**

```
nvflow/recipes/{recipe_name}/
├── workflows/*.yaml          # Workflow definitions (e.g., template_based_sdg.yaml)
├── stages/{category}/*.py    # Stage implementations (e.g., stages/sdg/generate_answers.py)
└── prompts/                  # Prompt templates used by stages
```

**Finding stage implementations:**

When a workflow YAML references a stage like `generate_answers`, the Python implementation is located at:
```
nvflow/recipes/{recipe}/stages/{category}/{stage_name}.py
```

For example:
- Stage name in YAML: `generate_answers`
- Implementation file: `nvflow/recipes/finance/stages/sdg/generate_answers.py`

**About category folders:**

The `{category}` folders (like `sdg/`, `data/`, `training/`) are organizational containers that group related stages together. They help keep the codebase organized but **do not affect stage naming** - stages are always referenced by their short name in workflow YAML files, not by their folder path.

**Example:**
```yaml
# In workflow YAML
pipeline_stages:
  - download_sec_filings  # Short name
  - generate_questions    # Short name

# These map to Python files:
# stages/download/download_sec_filings.py
# stages/sdg/generate_questions.py
```

## 📋 Prerequisites

- **Git** - to clone the repository
- **uv 0.9.22** - Python package manager ([docs](https://docs.astral.sh/uv/))

```bash
# Install uv 0.9.22 (REQUIRED - newer versions have breaking TOML parsing changes)
pip install uv==0.9.22

# Verify installation
uv --version  # Should show: uv 0.9.22
```

> **⚠️ Important:** uv version 0.9.22 is required. Newer versions (0.9.29+) have stricter TOML parsing that's incompatible with upstream dependency (nemo-run) syntax. This is a temporary requirement until nemo-run fixes their `pyproject.toml`.

## 📦 Installation

```bash
git clone https://github.com/NVIDIA/nvflow.git
cd nvflow

# For users
uv sync

# For developers
uv sync --all-extras
uv run pre-commit install
```

> ⚠️ **Developers:** Always run `uv run pre-commit install` after cloning. This enables automatic code quality checks on every commit.

### Activating the Virtual Environment (Optional)

By default, use `uv run <command>` to run commands in the project's virtual environment. If you prefer to activate the environment directly:

```bash
source .venv/bin/activate

# Now you can run commands without 'uv run' prefix
python --version  # 3.12+
nflow --help
pytest
```

## 🔧 Cluster Setup

To run workflows on a Slurm cluster, you need to configure containers and cluster settings.

> **See [INSTALL.md](INSTALL.md)** for complete cluster setup (containers, cluster configuration, verification).

Once cluster setup is complete, set the config directory:

```bash
export NEMO_SKILLS_CONFIG_DIR=/path/to/nvflow/cluster_configs
```

## 🚀 Quick Start

### CLI Invocation

```bash
# List all stages (hierarchical display)
uv run nflow list-stages

# List stages for a specific recipe
uv run nflow list-stages --recipe finance

# Get stage details (full path: recipe.workflow.stage)
uv run nflow stage-info example.sdg_simple.generate_answer

# Or with flags (if you know recipe and workflow)
uv run nflow stage-info generate_answer --recipe example --workflow sdg_simple

# Run specific stage (short name from config)
uv run nflow run generate_answer --config nvflow/recipes/example/workflows/sdg_simple.yaml

# Run all stages in workflow
uv run nflow run-all --config nvflow/recipes/example/workflows/sdg_simple.yaml
```

### Python Invocation

```bash
# Run a single stage (short name from config)
uv run python scripts/run_flow.py generate_answer --config nvflow/recipes/example/workflows/sdg_simple.yaml

# Run all stages
uv run python scripts/run_flow.py --all --config nvflow/recipes/example/workflows/sdg_simple.yaml
```

### Programmatic Python API

```python
from nvflow.core import WorkflowRunner

# Run all stages
runner = WorkflowRunner("nvflow/recipes/example/workflows/sdg_simple.yaml")
runner.run()

# Run specific stage (short name from config)
runner.run(stages=["generate_answer"])
```

## 🏗️ Structure

```
nvflow/
├── core/                      # Framework
├── cli/                       # CLI
└── recipes/                   # Domain recipes
    ├── example/               # Example recipe (learning & testing)
    │   ├── stages/sdg/        # Example SDG stage
    │   ├── prompts/           # Prompt templates
    │   └── workflows/         # Example workflows
    └── finance/               # Finance reasoning recipe
        ├── stages/            # Stage implementations
        │   ├── download/      # SEC filing download
        │   ├── evaluation/    # Evaluation stages
        │   ├── rl/            # GRPO RL training stages
        │   ├── sdg/           # SDG stages
        │   ├── sft/           # SFT training stages
        │   └── shared/        # Shared stages (data_transformation, train_validation_split)
        ├── prompts/           # Prompt templates
        └── workflows/         # Workflow configs
```

## 📝 Creating a Stage

See the `example` recipe for a complete working example. Key steps:

1. **Create stage file** in `nvflow/recipes/{recipe}/stages/{category}/`
2. **Implement with hierarchical decorator:**
   ```python
   @StageRegistry.register(recipe="finance", workflow="sft", stage="training")
   class SFTStage(BaseStage):
       workflow = "sft"
       def execute(self, config, cluster, expname, run_after=None):
           # Your implementation
           pass
   ```
3. **Add to workflow YAML** with short stage name:
   ```yaml
   recipe: finance
   workflow:
     name: "sft"
   pipeline_stages:
     - sft  # Short name!
   stages:
     sft:
       run_name: "baseline-lr5e6"  # Optional: distinguish experiment runs
       # Your stage configuration
   ```
4. **Run it** with `nflow run` or `nflow run-all`

**Terminal output in stages:** Use the `console` helpers for consistent, readable logs when your stage runs (e.g. `console.status()`, `console.detail()`, `console.success()`). See **[Console UI guide](docs/development/console-ui.md)**.

Example: `nvflow/recipes/example/stages/sdg/generate_answer.py`

## 📦 Recipes

### Example Recipe
Simple demonstration recipe for learning the framework:
- **Stage:** `example.sdg_simple.generate_answer` - Basic SDG workflow with nemo-skills integration
- **Config:** `nvflow/recipes/example/workflows/sdg_simple.yaml`
- **Purpose:** Learning NVFlow framework basics

### Finance Recipe

**📚 [Complete Finance Recipe Documentation →](docs/recipes/finance/README.md)**

End-to-end pipeline for generating synthetic financial Q&A data from SEC filings and training financial reasoning models.

**Quick Links:**
- [Quick Start (~1.5 hour demo)](docs/recipes/finance/quick-start.md) - Get started quickly with 7 companies
- [Workflow Guides](docs/recipes/finance/workflows/) - Detailed guides for all 6 workflows
- [Stage Reference](docs/recipes/finance/stages/) - Technical specifications for 27 stages

**Pipeline:**
```
download-sec → template-sdg / document-sdg → sft → eval → grpo
```

**Features:**
- 27 stages across 6 workflows
- Two SDG approaches (template-based & document-grounded)
- Multiple model support (GPT-OSS-120B, Qwen3, Nemotron)
- Produces 80K+ synthetic Q&A pairs
- Complete training and evaluation pipeline

## 📚 CLI Commands

```bash
nflow list-stages                           # List all stages (hierarchical)
nflow list-stages --recipe finance          # Filter by recipe
nflow list-stages --recipe finance --workflow training_sft  # Filter by workflow
nflow stage-info STAGE_PATH                 # Stage details (e.g., finance.training_sft.sft)
nflow stage-info STAGE --recipe R --workflow W  # Or with flags
nflow validate --config FILE                # Validate config
nflow run STAGE --config FILE               # Run specific stage (short name)
nflow run-all --config FILE                 # Run all stages
nflow version                               # Show version
```

## 🛠️ Development

```bash
# Install dev dependencies
uv sync --all-extras
uv run pre-commit install

# Run tests
uv run pytest

# Lint code
uv run ruff check nvflow/ tests/
uv run mypy nvflow/

# Format code
uv run ruff format nvflow/ tests/
```

## 📄 License

Apache-2.0

## 🙏 Acknowledgments

Built on:
- [NeMo-Skills](https://github.com/NVIDIA/NeMo-Skills)
- [NeMo-RL](https://github.com/NVIDIA/NeMo-RL)
