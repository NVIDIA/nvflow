# NVFlow Architecture

> **Version:** 1.0
> **Last Updated:** January 21, 2026
> **Purpose:** Comprehensive architectural overview of the NVFlow orchestration framework

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [System Overview](#2-system-overview)
3. [Core Framework Components](#3-core-framework-components)
4. [Hierarchical Organization](#4-hierarchical-organization)
5. [Execution Flow](#5-execution-flow)
6. [Finance Recipe Architecture](#6-finance-recipe-architecture)
7. [Deployment Architecture](#7-deployment-architecture)
8. [Technology Stack](#8-technology-stack)
9. [Data Flow](#9-data-flow)

---

## 1. High-Level Architecture

```mermaid
graph TB
    subgraph "User Interfaces"
        CLI[CLI - nflow]
        PythonAPI[Python API]
        Script[Python Scripts]
    end

    subgraph "NVFlow Core Framework"
        WR[WorkflowRunner]
        SR[StageRegistry]
        BS[BaseStage]
        Console[Console UI]
    end

    subgraph "Recipe Layer"
        direction TB
        Finance[Finance Recipe]
        Example[Example Recipe]
        Custom[Custom Recipes...]
    end

    subgraph "External Dependencies"
        NemoSkills[NeMo-Skills]
        NemoRL[NeMo-RL]
        Slurm[Slurm Cluster]
        Containers[Container Images]
    end

    subgraph "Storage"
        Data[Data Storage]
        Models[Model Storage]
        Outputs[Job Outputs]
    end

    CLI --> WR
    PythonAPI --> WR
    Script --> WR

    WR --> SR
    WR --> Console
    SR --> BS

    Finance --> BS
    Example --> BS
    Custom --> BS

    WR --> NemoSkills
    WR --> Slurm

    Slurm --> Containers
    Slurm --> Data
    Slurm --> Models
    Slurm --> Outputs

    style CLI fill:#e1f5ff
    style PythonAPI fill:#e1f5ff
    style Script fill:#e1f5ff
    style WR fill:#fff4e6
    style SR fill:#fff4e6
    style BS fill:#fff4e6
    style Finance fill:#e8f5e9
    style Slurm fill:#f3e5f5
```

---

## 2. System Overview

### Architecture Principles

**NVFlow** follows a **layered, modular architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────┐
│               USER INTERFACES                        │
│  CLI (nflow) │ Python API │ Scripts                 │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│            ORCHESTRATION LAYER                       │
│  WorkflowRunner │ Config Management │ Dependencies  │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│             REGISTRY LAYER                           │
│  StageRegistry (Recipe → Workflow → Stage)          │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│              EXECUTION LAYER                         │
│  BaseStage │ Stage Implementations                   │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│          INFRASTRUCTURE LAYER                        │
│  NeMo-Skills │ Slurm │ Containers │ Storage         │
└─────────────────────────────────────────────────────┘
```

### Key Design Patterns

1. **Template Method Pattern**: `BaseStage` defines the interface, concrete stages implement `execute()`
2. **Registry Pattern**: `StageRegistry` provides hierarchical discovery and registration
3. **Strategy Pattern**: Different execution strategies (local, Slurm) via configuration
4. **Decorator Pattern**: `@StageRegistry.register()` for stage registration
5. **Dependency Injection**: Configuration and cluster info injected into stages

---

## 3. Core Framework Components

```mermaid
classDiagram
    class BaseStage {
        <<abstract>>
        +workflow: str
        +execute(config, cluster, expname, run_after)*
        +validate_config(config)
        +get_dependencies(config)
    }

    class StageRegistry {
        -_stages: Dict~Recipe→Workflow→Stage~
        -_recipe_configs: Dict
        +register(recipe, workflow, stage)$
        +get(recipe, workflow, stage)$
        +has(recipe, workflow, stage)$
        +list_recipes()$
        +list_workflows(recipe)$
        +list_stages(recipe, workflow)$
    }

    class WorkflowRunner {
        -config: Dict
        -config_path: Path
        -recipe: str
        -workflow_name: str
        -cluster: str
        +__init__(config_path)
        +run(stages)
        +validate_config()
        -_run_stage(stage_name)
        -_load_config_with_inheritance()
        -_expand_dynamic_stages()
    }

    class ConcreteStage {
        +workflow: str
        +execute(config, cluster, expname, run_after)
        +validate_config(config)
    }

    BaseStage <|-- ConcreteStage
    WorkflowRunner ..> StageRegistry : uses
    StageRegistry --> BaseStage : manages
    WorkflowRunner ..> BaseStage : executes

    note for BaseStage "All stages inherit from\nBaseStage and implement\nthe execute() method"
    note for StageRegistry "Hierarchical registry:\nRecipe → Workflow → Stage"
    note for WorkflowRunner "Loads YAML config,\nmanages dependencies,\nexecutes stage sequence"
```

### Component Responsibilities

#### **BaseStage** (Abstract Base Class)
- Defines the contract for all stages
- Provides template methods for execution
- Handles validation and dependency resolution
- **Key Method**: `execute(config, cluster, expname, run_after)`

#### **StageRegistry** (Singleton Registry)
- Maintains hierarchical mapping: Recipe → Workflow → Stage → Class
- Provides discovery and lookup capabilities
- Supports workflow ordering from configuration
- **Key Methods**: `register()`, `get()`, `list_*()` operations

#### **WorkflowRunner** (Orchestrator)
- Loads and validates YAML configurations
- Supports config inheritance via `_base_` key
- Manages stage execution order and dependencies
- Generates unique experiment names per stage
- **Key Methods**: `run()`, `validate_config()`

#### **Console** (UI Component)
- Provides rich terminal output with colors and formatting
- Displays headers, sections, details, and progress
- Enhances user experience during workflow execution

---

## 4. Hierarchical Organization

### Three-Level Hierarchy

```
Recipe (Domain-specific collection)
  ↓
Workflow (Pipeline definition)
  ↓
Stage (Execution unit)
```

### Structure in Code

```
nvflow/
├── core/                           # Framework layer
│   ├── base_stage.py              # Abstract base class
│   ├── stage_registry.py          # Hierarchical registry
│   ├── workflow_runner.py         # Orchestration engine
│   └── console.py                 # UI components
│
├── cli/                            # User interface layer
│   └── main.py                    # CLI commands
│
├── lib/                            # Shared libraries
│   └── rl/                        # RL utilities (rollout, verify, helpers)
│
└── recipes/                        # Recipe layer
    ├── finance/                   # Finance domain
    │   ├── stages/                # Stage implementations
    │   │   ├── download/         # SEC filing download
    │   │   ├── sdg/              # SDG stages
    │   │   ├── sft/              # SFT training stages
    │   │   ├── shared/           # Shared stages (data_transformation, train_validation_split)
    │   │   ├── rl/               # GRPO RL training stages
    │   │   └── evaluation/       # Eval stages
    │   ├── utils/                 # Utility sub-packages
    │   │   ├── download/         # Download utilities
    │   │   ├── sdg/              # SDG utilities
    │   │   ├── sft/              # SFT utilities
    │   │   ├── rl/               # RL utilities
    │   │   ├── shared/           # Shared utilities
    │   │   └── evaluation/       # Evaluation utilities
    │   ├── workflows/             # Workflow configs
    │   │   ├── download_sec_filings.yaml
    │   │   ├── sdg/
    │   │   │   ├── template-based-sdg.yaml
    │   │   │   ├── template-based-sdg-demo.yaml
    │   │   │   └── document-grounded-sdg.yaml
    │   │   ├── sft/
    │   │   │   └── qwen3_14b.yaml
    │   │   ├── eval/
    │   │   │   ├── base.yaml       # Shared benchmarks, judges, datasets
    │   │   │   ├── demo.yaml       # Demo evaluation
    │   │   │   └── baselines.yaml  # Baseline model evaluation
    │   │   └── grpo/
    │   │       └── qwen3_4b.yaml
    │   └── prompts/               # Prompt templates
    │
    └── example/                   # Example recipe
        ├── stages/sdg/
        ├── workflows/
        └── prompts/
```

### Registration Example

```python
# Stage registration uses decorator pattern
@StageRegistry.register(
    recipe="finance",
    workflow="sft",
    stage="sft"
)
class SFTStage(BaseStage):
    workflow = "sft"

    def execute(self, config, cluster, expname, run_after=None):
        # Implementation
        pass
```

### Workflow Configuration Example

```yaml
recipe: finance
workflow:
  name: sft
  type: training

cluster: my_cluster

pipeline_stages:
  - data_transformation
  - prepare_for_sft
  - train_validation_split
  - training

stages:
  data_transformation:
    input_dir: /data/raw
    output_dir: /data/processed

  training:
    total_gpus: 256
    dependencies:
      - data_transformation
      - prepare_for_sft
      - train_validation_split
```

---

## 5. Execution Flow

### Complete Execution Sequence

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant WorkflowRunner
    participant StageRegistry
    participant Stage
    participant NemoSkills
    participant Slurm

    User->>CLI: nflow run-all --config workflow.yaml
    CLI->>WorkflowRunner: __init__(config_path)
    WorkflowRunner->>WorkflowRunner: Load & validate config
    WorkflowRunner->>WorkflowRunner: Resolve config inheritance
    WorkflowRunner->>WorkflowRunner: Extract recipe/workflow context

    User->>CLI: run()
    CLI->>WorkflowRunner: run(stages=None)

    loop For each stage
        WorkflowRunner->>StageRegistry: get(recipe, workflow, stage)
        StageRegistry-->>WorkflowRunner: Return stage class
        WorkflowRunner->>Stage: __init__()
        WorkflowRunner->>Stage: validate_config(config)
        WorkflowRunner->>Stage: execute(config, cluster, expname, run_after)

        Stage->>NemoSkills: Submit job (NeMo-RL direct for training, nemo-skills for data/eval)
        NemoSkills->>Slurm: Submit job with dependencies
        Slurm-->>NemoSkills: Job ID
        NemoSkills-->>Stage: Job submitted
        Stage-->>WorkflowRunner: Stage complete
    end

    WorkflowRunner-->>CLI: All stages complete
    CLI-->>User: ✅ Workflow Complete!
```

### Dependency Resolution

```mermaid
graph LR
    A[Stage A] --> C[Stage C]
    B[Stage B] --> C
    C --> D[Stage D]

    style A fill:#e8f5e9
    style B fill:#e8f5e9
    style C fill:#fff4e6
    style D fill:#e1f5ff

    note1[run_after: None]
    note2[run_after: None]
    note3[run_after: A, B]
    note4[run_after: C]

    note1 -.-> A
    note2 -.-> B
    note3 -.-> C
    note4 -.-> D
```

**Dependency Handling:**
1. WorkflowRunner reads `dependencies` from stage config
2. Generates experiment names for all dependencies
3. Passes as `run_after` parameter to dependent stage
4. Slurm uses job dependencies to ensure correct execution order

---

## 6. Finance Recipe Architecture

### Pipeline Overview

```mermaid
graph TB
    subgraph "1. Data Acquisition"
        SEC[SEC Filings Download<br/>10-K, 10-Q, 8-K]
    end

    subgraph "2. Synthetic Data Generation"
        direction LR
        TBS[Template-Based SDG<br/>6 stages]
        DGS[Document-Grounded SDG<br/>7 stages]
    end

    subgraph "3. Data Preparation"
        DataPrep[Data Transformation<br/>Prepare for SFT<br/>Train/Val Split]
    end

    subgraph "4. Model Training"
        SFT[Supervised Fine-Tuning<br/>Multi-node GPU training]
    end

    subgraph "5. Evaluation"
        Eval[Benchmark Evaluation<br/>Baseline comparison]
    end

    subgraph "6. RL Training"
        GRPO[GRPO RL Training<br/>NeMo-Gym rewards]
    end

    SEC --> TBS
    SEC --> DGS
    TBS --> DataPrep
    DGS -.-> DataPrep
    DataPrep --> SFT
    SFT --> Eval
    SFT --> GRPO
    GRPO --> Eval

    style SEC fill:#e3f2fd
    style TBS fill:#f1f8e9
    style DGS fill:#fff3e0
    style DataPrep fill:#fce4ec
    style SFT fill:#e8eaf6
    style Eval fill:#f3e5f5
    style GRPO fill:#fff3e0
```

### Workflow Breakdown

#### **Workflow 1: Download SEC Filings**
```
Input: Ticker list (demo/S&P 500)
Output: ~100GB SEC filings (JSON format)
Stages: 1
GPU: None (CPU only)
```

#### **Workflow 2: Template-Based SDG** (Production)
```
Stages:
  1. create_seed_data         - Generate seed questions
  2. generate_questions       - Adapt to companies/years
  3. map_questions_to_context - Find relevant SEC sections
  4. generate_answers         - Generate answers with LLM
  5. genselect_answers        - Self-consistency filtering
  6. filter_answers           - Quality filtering

Output: ~300K Q&A pairs
GPU: 16 GPUs (max per stage)
Models: GPT-OSS-120B, Qwen3-14B
```

#### **Workflow 3: Document-Grounded SDG** (Experimental)
```
Stages:
  1. dg_sdg_preprocess                    - Preprocess filings
  2. document_grounded_qa_generation      - Generate verified Q&A
  3. genselect_answers                    - Self-consistency check
  4. evaluate_answers                     - Quality evaluation
  5. aggregate_answers                    - Combine results
  6. difficulty_estimation                - Stratify by difficulty
  7. document_grounded_data               - Prepare training data

Output: ~800K Q&A pairs (stratified)
GPU: 8 GPUs
Models: Qwen3 family (14B-235B)
```

#### **Workflow 4: Supervised Fine-Tuning**
```
Stages:
  1. data_transformation       - Convert to training format
  2. prepare_for_sft          - Format for NeMo
  3. train_validation_split   - Split dataset
  4. sequence_length_grouping - [Optional] Group by length
  5. training                 - Multi-node training
  6. convert_to_messages      - [Optional] Post-processing

GPU: 256 GPUs (32 nodes × 8 GPUs)
Model: Qwen3-14B
Parallelism: TP=2, PP=1, CP=2
```

#### **Workflow 5: Evaluation**
```
Stages:
  1. prepare_data              - Prepare benchmarks
  2-N. evaluate_checkpoints    - Evaluate each checkpoint
  N+1-M. evaluate_baselines    - Evaluate baseline models

Output: Accuracy, F1, category metrics
GPU: 8 GPUs per eval job
Benchmarks: Financial reasoning tasks
```

#### **Workflow 6: GRPO RL Training**
```
Stages:
  1. validate_questions        - Validate format + deduplicate
  2. data_transformation       - SDG cleanup to model-agnostic schema
  3. apply_prompt_template     - Apply prompt template + extract answer
  4. convert_to_responses_api  - Convert to NeMo-Gym Responses API format
  5. prepare_data              - Add agent routing fields
  6. prefetch_cache            - Prefetch SEC filings cache
  7. collect_rollouts          - Rollout collection + reward profiling
  8. train_validation_split    - Split into train/val sets
  9. training                  - GRPO training with NeMo-Gym
  10. eval                     - Evaluate checkpoints on benchmarks

Output: RL-trained model + eval results
GPU: 16 GPUs (2 nodes for demo), 64 GPUs (8 nodes for production)
Model: Qwen3-4B dense (demo, FSDP v2), Qwen3-30B-A3B MoE (production, Megatron)
```

### Finance Recipe Component Diagram

```mermaid
graph TB
    subgraph "Finance Recipe Structure"
        direction TB

        subgraph "Stages (42 total)"
            direction LR
            SDG[SDG Stages<br/>12 stages]
            SFT[SFT Stages<br/>4 stages]
            Eval[Eval Stages<br/>2 stages]
            RL[RL Stages<br/>10 stages]
        end

        subgraph "Workflows (6 total)"
            W1[download-sec<br/>1 stage]
            W2[template-sdg<br/>6 stages]
            W3[document-sdg<br/>7 stages]
            W4[sft<br/>6 stages]
            W5[eval<br/>9 stages]
            W6[grpo<br/>10 stages]
        end

        subgraph "Prompts"
            P1[Template Prompts]
            P2[Document Prompts]
            P3[Evaluation Prompts]
        end

        W1 -.-> SDG
        W2 -.-> SDG
        W3 -.-> SDG
        W4 -.-> SFT
        W5 -.-> Eval
        W6 -.-> RL

        SDG -.-> P1
        SDG -.-> P2
        Eval -.-> P3
    end

    style SDG fill:#c8e6c9
    style SFT fill:#bbdefb
    style Eval fill:#f8bbd0
    style RL fill:#fff3e0
    style W1 fill:#e1f5ff
    style W2 fill:#e1f5ff
    style W3 fill:#e1f5ff
    style W4 fill:#e1f5ff
    style W5 fill:#e1f5ff
    style W6 fill:#e1f5ff
```

---

## 7. Deployment Architecture

### Cluster Deployment

```mermaid
graph TB
    subgraph "Local Development Machine"
        Dev[Developer Workstation]
        NVFlow[NVFlow CLI/API]
        Configs[Cluster Configs]
    end

    subgraph "Slurm Cluster"
        direction TB

        subgraph "Login Node"
            SSH[SSH Gateway]
            Sched[Slurm Scheduler]
        end

        subgraph "Compute Nodes"
            direction LR
            Node1[GPU Node 1<br/>8x H100]
            Node2[GPU Node 2<br/>8x H100]
            NodeN[GPU Node N<br/>8x H100]
        end

        subgraph "Storage"
            SharedFS[Shared Filesystem<br/>Lustre/NFS]
            Data[Data Storage]
            Models[Model Cache]
            Output[Job Outputs]
        end

        subgraph "Container Runtime"
            Enroot[Enroot]
            Containers[.sqsh Container Images]
        end
    end

    Dev --> NVFlow
    NVFlow --> Configs
    NVFlow -->|SSH| SSH
    SSH --> Sched

    Sched -->|Schedule Jobs| Node1
    Sched -->|Schedule Jobs| Node2
    Sched -->|Schedule Jobs| NodeN

    Node1 --> Enroot
    Node2 --> Enroot
    NodeN --> Enroot

    Enroot --> Containers

    Node1 --> SharedFS
    Node2 --> SharedFS
    NodeN --> SharedFS

    SharedFS --> Data
    SharedFS --> Models
    SharedFS --> Output

    style Dev fill:#e1f5ff
    style NVFlow fill:#fff4e6
    style Sched fill:#f3e5f5
    style Node1 fill:#e8f5e9
    style Node2 fill:#e8f5e9
    style NodeN fill:#e8f5e9
    style Enroot fill:#fff3e0
```

### Container Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│          Air-gapped container images (.sqsh format)             │
│   (built locally from dockerfiles/Dockerfile.* — see INSTALL.md)│
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────┐  ┌──────────────────────┐             │
│  │ nvflow-nemo-skills   │  │ nvflow-vllm          │             │
│  │   (786d8c58)         │  │   (v0.18.1)          │             │
│  │  SDG/eval/data prep  │  │  SDG/eval inference  │             │
│  └──────────────────────┘  └──────────────────────┘             │
│                                                                 │
│  ┌──────────────────────┐  ┌──────────────────────┐             │
│  │ sglang (pulled)      │  │ nvflow-nemo-rl       │             │
│  │   (v0.5.10.post1)    │  │   (v0.6.0)           │             │
│  │  SDG inference       │  │  SFT/GRPO + Gym venv │             │
│  └──────────────────────┘  └──────────────────────┘             │
│                                                                 │
│  ┌──────────────────────┐                                       │
│  │ nvflow-vllm-grpo     │                                       │
│  │   (v0.17.1)          │                                       │
│  │ GRPO rollout/judge   │                                       │
│  └──────────────────────┘                                       │
└─────────────────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│              Shared Filesystem Mounts               │
├─────────────────────────────────────────────────────┤
│  /workspace → /lustre/.../workspace                 │
│  /hf_models → /lustre/.../models/hf_models          │
│  /outputs   → /lustre/.../outputs                   │
└─────────────────────────────────────────────────────┘
```

### Job Submission Flow

```mermaid
sequenceDiagram
    participant User
    participant NVFlow
    participant NemoSkills
    participant Slurm
    participant ComputeNode
    participant Storage

    User->>NVFlow: nflow run-all --config workflow.yaml
    NVFlow->>NVFlow: Load config & resolve context

    loop For each stage
        NVFlow->>NemoSkills: pipeline.run(stage_config)
        NemoSkills->>NemoSkills: Build sbatch script
        NemoSkills->>Slurm: sbatch (with dependencies)
        Slurm-->>NemoSkills: Job ID
        NemoSkills-->>NVFlow: Job submitted

        Slurm->>Slurm: Wait for dependencies
        Slurm->>ComputeNode: Allocate resources
        ComputeNode->>Storage: Load container image
        ComputeNode->>Storage: Mount shared filesystem
        ComputeNode->>ComputeNode: Execute stage in container
        ComputeNode->>Storage: Write outputs
        ComputeNode-->>Slurm: Job complete
    end

    NVFlow-->>User: ✅ All stages submitted
```

---

## 8. Technology Stack

### Framework Dependencies

```yaml
Core Framework:
  - Python: 3.12+
  - OmegaConf: YAML config management
  - Typer: CLI interface
  - Rich: Terminal UI

Execution:
  - NeMo-Skills: Pipeline execution & SDG
  - NeMo-RL: Reinforcement learning integration
  - Slurm: Cluster workload manager
  - Enroot: Container runtime

Infrastructure:
  - uv: Package manager
  - pre-commit: Code quality
  - pytest: Testing
  - ruff: Linting & formatting
  - mypy: Type checking

Models:
  - vLLM: Inference server
  - SGLang: Structured generation
  - HuggingFace Transformers: Model loading
```

### External Integrations

```mermaid
graph LR
    subgraph "NVFlow"
        Core[Core Framework]
    end

    subgraph "NeMo Ecosystem"
        NemoSkills[NeMo-Skills<br/>SDG & Pipelines]
        NemoRL[NeMo-RL<br/>RL Training]
        NemoFW[NeMo Framework<br/>Model Training]
    end

    subgraph "Inference"
        vLLM[vLLM<br/>Fast inference]
        SGLang[SGLang<br/>Structured gen]
    end

    subgraph "Cluster"
        Slurm[Slurm<br/>Job scheduling]
        Enroot[Enroot<br/>Containers]
    end

    subgraph "Storage"
        Lustre[Lustre FS<br/>High-perf storage]
        HF[HuggingFace Hub<br/>Model download]
    end

    Core --> NemoSkills
    Core --> NemoRL
    Core --> Slurm

    NemoSkills --> vLLM
    NemoSkills --> SGLang
    NemoSkills --> NemoFW

    Slurm --> Enroot
    Enroot --> Lustre

    NemoFW --> HF

    style Core fill:#fff4e6
    style NemoSkills fill:#e8f5e9
    style Slurm fill:#f3e5f5
```

---

## 9. Data Flow

### Complete Data Pipeline (Finance Recipe)

```mermaid
graph TB
    subgraph "Stage 1: Data Acquisition"
        SEC_API[SEC EDGAR API]
        Filings[(SEC Filings<br/>JSON format<br/>~100GB)]
    end

    subgraph "Stage 2: SDG - Template-Based"
        Seeds[Seed Questions]
        Questions[Generated Questions]
        Context[Mapped Context]
        Answers[Generated Answers]
        Filtered[Filtered Q&A<br/>~300K pairs]
    end

    subgraph "Stage 3: Data Preparation"
        Transformed[Transformed Data]
        SFTFormat[SFT Format]
        TrainVal[Train/Val Split]
    end

    subgraph "Stage 4: Training"
        Checkpoints[(Model Checkpoints)]
    end

    subgraph "Stage 5: Evaluation"
        Benchmarks[Benchmark Data]
        Results[Evaluation Results]
    end

    SEC_API --> Filings
    Filings --> Seeds
    Seeds --> Questions
    Questions --> Context
    Filings --> Context
    Context --> Answers
    Answers --> Filtered

    Filtered --> Transformed
    Transformed --> SFTFormat
    SFTFormat --> TrainVal

    TrainVal --> Checkpoints

    Benchmarks --> Results
    Checkpoints --> Results

    style Filings fill:#e3f2fd
    style Filtered fill:#f1f8e9
    style TrainVal fill:#fce4ec
    style Checkpoints fill:#e8eaf6
    style Results fill:#f3e5f5
```

### Storage Layout

```
/lustre/fsw/.../workspace/nvflow/
│
├── cluster_configs/                  # Cluster configuration
│   ├── containers.yaml              # Container definitions
│   ├── my_cluster.yaml              # User cluster config
│   └── template-slurm.yaml          # Template config
│
├── nvflow/                        # Framework code
│   ├── core/                        # Core components
│   ├── cli/                         # CLI interface
│   ├── lib/                         # Shared libraries (RL utilities)
│   └── recipes/                     # Recipe implementations
│       ├── finance/
│       └── example/
│
├── jobs/                             # Job execution artifacts
│   └── experiments/
│       ├── download-sec-demo/
│       ├── template_based_sdg-*/
│       ├── sft-training-*/
│       └── eval-*/
│           └── nemo-run/
│               ├── code/            # Packaged code snapshot
│               ├── logs/            # Execution logs
│               └── results/         # Stage outputs
│
├── outputs/                          # Output data
│   ├── finance/
│   │   ├── sec_filings/            # Downloaded filings (~100GB)
│   │   ├── sdg/                    # Generated Q&A datasets
│   │   ├── sft/                    # Training data & checkpoints
│   │   └── eval/                   # Evaluation results
│   └── logs/                        # Setup logs
│
└── models/                           # Model storage
    └── hf_models/                   # HuggingFace models
        ├── Qwen/
        ├── GPT-OSS-120B/
        └── Nemotron/
```

---

## Summary

### Key Architectural Highlights

1. **Modular Design**: Clear separation between framework, recipes, and infrastructure
2. **Hierarchical Organization**: Recipe → Workflow → Stage provides natural organization
3. **Declarative Configuration**: YAML-based configs with inheritance support
4. **Flexible Execution**: CLI, Python API, and programmatic interfaces
5. **Cluster Native**: First-class Slurm integration with dependency management
6. **Extensible**: Easy to add new recipes, workflows, and stages
7. **Built on NeMo**: Leverages NVIDIA's NeMo ecosystem (Skills, RL, Framework)

### Design Benefits

- **Reproducibility**: Version-controlled configs and deterministic execution
- **Reusability**: Stages can be shared across workflows and recipes
- **Scalability**: Seamless scaling from local development to multi-node clusters
- **Maintainability**: Clear structure and separation of concerns
- **Discoverability**: Registry pattern enables stage discovery and documentation

---

**Document Version:** 1.0
**Generated:** January 21, 2026
**Repository:** nvflow
