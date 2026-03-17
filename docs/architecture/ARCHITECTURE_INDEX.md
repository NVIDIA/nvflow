# NVFlow - Architecture Documentation Index

> **Navigation guide for all architecture documentation**
> **Start here to find the right documentation for your needs**

---

## 📚 Documentation Overview

The NVFlow architecture is documented across multiple files, each serving a specific purpose. This index helps you find the right documentation quickly.

---

## 🎯 Quick Navigation

### I want to...

| Goal | Document | Time |
|------|----------|------|
| **Get a quick overview** | [ARCHITECTURE_QUICK_REFERENCE.md](#quick-reference) | 5 min |
| **Understand the system deeply** | [ARCHITECTURE.md](#comprehensive-architecture) | 30 min |
| **View visual diagrams** | [diagrams/](#visual-diagrams) | 10 min |
| **Learn about diagrams** | [DIAGRAMS_SUMMARY.md](#diagrams-summary) | 10 min |
| **Get started with NVFlow** | [README.md](#main-readme) | 15 min |
| **Set up the cluster** | [INSTALL.md](#installation-guide) | 30 min |
| **Learn the finance recipe** | [docs/recipes/finance/](#finance-recipe-docs) | 45 min |

---

## 📖 Document Descriptions

### Quick Reference
**File:** [ARCHITECTURE_QUICK_REFERENCE.md](./ARCHITECTURE_QUICK_REFERENCE.md)
**Size:** ~5 KB
**Reading Time:** 5 minutes
**Best For:** Quick onboarding, cheat sheet, reference card

**Contents:**
- One-page architecture overview
- Core components table
- Common commands
- Quick stage creation guide
- Key features checklist
- Documentation map

**When to Use:**
- First time learning about NVFlow
- Need a quick reminder of concepts
- Looking for specific commands
- Want a printable reference

---

### Comprehensive Architecture
**File:** [ARCHITECTURE.md](./ARCHITECTURE.md)
**Size:** ~26 KB
**Reading Time:** 30 minutes
**Best For:** Deep understanding, system design, contribution

**Contents:**
1. High-level architecture with diagrams
2. System overview and design principles
3. Core framework components (detailed)
4. Hierarchical organization (Recipe → Workflow → Stage)
5. Complete execution flow with sequence diagrams
6. Finance recipe architecture (all 27 stages)
7. Deployment architecture and topology
8. Technology stack and integrations
9. Data flow diagrams

**When to Use:**
- Need comprehensive system understanding
- Planning to contribute to the codebase
- Designing new recipes or workflows
- Troubleshooting complex issues
- Presenting architecture to stakeholders

---

### Diagrams Summary
**File:** [DIAGRAMS_SUMMARY.md](./DIAGRAMS_SUMMARY.md)
**Size:** ~12 KB
**Reading Time:** 10 minutes
**Best For:** Understanding available diagrams, diagram usage guide

**Contents:**
- Overview of all 5 diagrams
- Diagram details and use cases
- Audience-specific recommendations
- Question-to-diagram mapping
- Diagram statistics
- Rendering examples
- Update guidelines

**When to Use:**
- Want to know what diagrams are available
- Need to choose the right diagram
- Want to render diagrams in different formats
- Planning to create new diagrams

---

### Visual Diagrams
**Location:** [diagrams/](../diagrams/)
**Format:** Mermaid (.mmd files)
**Count:** 5 diagrams + README
**Best For:** Visual learners, presentations, documentation

**Available Diagrams:**

1. **[architecture-overview.mmd](../diagrams/architecture-overview.mmd)**
   - High-level system architecture
   - All major components and relationships
   - 5 layers: UI, Core, Recipes, Infrastructure, Storage

2. **[finance-pipeline.mmd](../diagrams/finance-pipeline.mmd)**
   - Complete finance recipe pipeline
   - All 6 workflows with 27 stages
   - Data flow from SEC filings to evaluation

3. **[execution-flow.mmd](../diagrams/execution-flow.mmd)**
   - Runtime execution sequence diagram
   - User command to job completion
   - 4 phases: Init, Validate, Execute, Monitor

4. **[component-architecture.mmd](../diagrams/component-architecture.mmd)**
   - Class diagram of core framework
   - BaseStage, StageRegistry, WorkflowRunner
   - Relationships and dependencies

5. **[deployment-architecture.mmd](../diagrams/deployment-architecture.mmd)**
   - Infrastructure and deployment topology
   - Local machine to Slurm cluster
   - Compute nodes, storage, containers

**Viewing Options:**
- Online: https://mermaid.live/
- VS Code: Mermaid Preview extension
- CLI: `mmdc -i diagram.mmd -o diagram.png`
- GitHub: Native rendering

**When to Use:**
- Need visual understanding
- Creating presentations
- Onboarding new team members
- Documentation in other systems

---

### Main README
**File:** [README.md](../../README.md)
**Size:** ~10 KB
**Reading Time:** 15 minutes
**Best For:** Getting started, understanding concepts, running workflows

**Contents:**
- Project overview and key features
- Core concepts (Recipe, Workflow, Stage)
- Folder structure explanation
- Installation instructions
- Quick start examples
- CLI commands reference
- Development guide

**When to Use:**
- First time using NVFlow
- Need to understand basic concepts
- Want to run your first workflow
- Looking for CLI command syntax

---

### Installation Guide
**File:** [INSTALL.md](../../INSTALL.md)
**Size:** ~8 KB
**Reading Time:** 30 minutes (including setup)
**Best For:** Cluster setup, container configuration, troubleshooting

**Contents:**
1. Prerequisites (uv, yq, enroot)
2. Container setup (automated script)
3. Model download instructions
4. Cluster configuration
5. Verification steps
6. Troubleshooting guide

**When to Use:**
- Setting up NVFlow for the first time
- Configuring a new cluster
- Troubleshooting installation issues
- Understanding container requirements

---

### Finance Recipe Docs
**Location:** [docs/recipes/finance/](../recipes/finance/)
**Size:** Multiple files (~20 KB total)
**Reading Time:** 45 minutes
**Best For:** Understanding finance recipe, running production pipelines

**Main Files:**

1. **[README.md](../recipes/finance/README.md)** - Recipe overview
   - 6 workflows, 27 stages
   - Pipeline architecture
   - Getting started guide
   - Command reference

2. **[quick-start.md](../recipes/finance/quick-start.md)** - 30-min demo
   - Hands-on tutorial with 7 companies
   - Step-by-step instructions
   - Expected outputs

3. **Workflow Guides** (in `workflows/`)
   - 01-download-sec.md
   - 02-template-based-sdg.md
   - 03-document-grounded-sdg.md
   - 04-sft.md
   - 05-eval.md
   - 06-grpo.md
   - 06-finance-agent-eval.md

4. **Stage Reference** (in `stages/`)
   - Technical specifications for all 27 stages
   - Input/output formats
   - Configuration options

5. **[troubleshooting.md](../recipes/finance/troubleshooting.md)**
   - Common issues and solutions
   - Debugging tips

**When to Use:**
- Running the finance recipe
- Understanding SDG approaches
- Training financial reasoning models
- Troubleshooting finance-specific issues

---

## 🗺️ Documentation Map (Visual)

```
NVFlow Documentation
│
├─ 📘 Getting Started
│  ├─ README.md ...................... Project overview & quick start
│  ├─ INSTALL.md ..................... Cluster setup guide
│  └─ ARCHITECTURE_QUICK_REFERENCE.md  One-page cheat sheet
│
├─ 🏗️ Architecture
│  ├─ ARCHITECTURE.md ................ Comprehensive architecture (26 KB)
│  ├─ DIAGRAMS_SUMMARY.md ............ Diagram usage guide
│  ├─ ARCHITECTURE_INDEX.md .......... This file
│  └─ diagrams/ ...................... Visual diagrams (5 files)
│     ├─ architecture-overview.mmd
│     ├─ finance-pipeline.mmd
│     ├─ execution-flow.mmd
│     ├─ component-architecture.mmd
│     ├─ deployment-architecture.mmd
│     └─ README.md
│
├─ 🍴 Recipes
│  ├─ docs/recipes/finance/ .......... Finance recipe (production)
   │  │  ├─ README.md ................... Recipe overview
   │  │  ├─ quick-start.md .............. 30-min demo
   │  │  ├─ workflows/ .................. 7 workflow guides
   │  │  ├─ stages/ ..................... 27 stage specifications
│  │  └─ troubleshooting.md .......... Common issues
│  │
│  └─ nvflow/recipes/example/ ..... Example recipe (learning)
│
├─ 💻 Code & development docs
│  ├─ docs/development/console-ui.md .. Console UI guide (stage terminal output)
│  └─ tests/README.md ................ Testing guide
│
└─ 🔧 Configuration
   ├─ cluster_configs/ ............... Cluster configuration files
   └─ pyproject.toml ................. Project dependencies
```

---

## 👥 Audience-Specific Paths

### For New Users
1. Start: [README.md](../../README.md) - Understand what NVFlow is
2. Quick ref: [ARCHITECTURE_QUICK_REFERENCE.md](./ARCHITECTURE_QUICK_REFERENCE.md) - Key concepts
3. Visual: [diagrams/architecture-overview.mmd](../diagrams/architecture-overview.mmd) - See the big picture
4. Try it: [docs/recipes/finance/quick-start.md](../recipes/finance/quick-start.md) - Run first workflow

### For Data Scientists
1. Overview: [README.md](../../README.md) - Core concepts
2. Pipeline: [diagrams/finance-pipeline.mmd](../diagrams/finance-pipeline.mmd) - See data flow
3. Recipe: [docs/recipes/finance/README.md](../recipes/finance/README.md) - Finance pipeline
4. Run: [docs/recipes/finance/quick-start.md](../recipes/finance/quick-start.md) - Hands-on demo

### For ML Engineers
1. Setup: [INSTALL.md](../../INSTALL.md) - Cluster configuration
2. Architecture: [ARCHITECTURE.md](./ARCHITECTURE.md) - System design
3. Execution: [diagrams/execution-flow.mmd](../diagrams/execution-flow.mmd) - Runtime behavior
4. Troubleshoot: [docs/recipes/finance/troubleshooting.md](../recipes/finance/troubleshooting.md)

### For Software Engineers / Stage Authors
1. Components: [diagrams/component-architecture.mmd](../diagrams/component-architecture.mmd) - Class structure
2. Deep dive: [ARCHITECTURE.md](./ARCHITECTURE.md) - Design patterns
3. Code: Browse `nvflow/core/` - Framework implementation
4. Extend: [README.md](../../README.md#-creating-a-stage) - Create new stages
5. Console UI: [docs/development/console-ui.md](../development/console-ui.md) - Terminal output in stage `execute()` methods

### For DevOps/Infrastructure
1. Setup: [INSTALL.md](../../INSTALL.md) - Installation guide
2. Deployment: [diagrams/deployment-architecture.mmd](../diagrams/deployment-architecture.mmd) - Topology
3. Cluster: [ARCHITECTURE.md](./ARCHITECTURE.md#7-deployment-architecture) - Infrastructure details
4. Config: `cluster_configs/` - Configuration files

### For System Architects
1. Overview: [ARCHITECTURE_QUICK_REFERENCE.md](./ARCHITECTURE_QUICK_REFERENCE.md) - Quick scan
2. Complete: [ARCHITECTURE.md](./ARCHITECTURE.md) - Full architecture
3. All diagrams: [diagrams/](../diagrams/) - Visual representations
4. Design: [ARCHITECTURE.md](./ARCHITECTURE.md#2-system-overview) - Design principles

---

## 🔍 Finding Specific Information

### Concepts & Terminology
- **Recipe, Workflow, Stage:** [README.md](../../README.md#-core-concepts)
- **Hierarchical organization:** [ARCHITECTURE.md](./ARCHITECTURE.md#4-hierarchical-organization)
- **Design patterns:** [ARCHITECTURE.md](./ARCHITECTURE.md#key-design-patterns)

### How-To Guides
- **Create a stage:** [README.md](../../README.md#-creating-a-stage)
- **Console output in stages:** [docs/development/console-ui.md](../development/console-ui.md) - Use `console.status()`, `console.detail()`, etc.
- **Run a workflow:** [README.md](../../README.md#-quick-start)
- **Set up cluster:** [INSTALL.md](../../INSTALL.md)
- **Run finance recipe:** [docs/recipes/finance/quick-start.md](../recipes/finance/quick-start.md)

### Technical Reference
- **CLI commands:** [README.md](../../README.md#-cli-commands)
- **Core components:** [ARCHITECTURE.md](./ARCHITECTURE.md#3-core-framework-components)
- **Finance stages:** [docs/recipes/finance/stages/](../recipes/finance/stages/)
- **API reference:** Code docstrings in `nvflow/core/`

### Visual Diagrams
- **System overview:** [diagrams/architecture-overview.mmd](../diagrams/architecture-overview.mmd)
- **Data pipeline:** [diagrams/finance-pipeline.mmd](../diagrams/finance-pipeline.mmd)
- **Execution flow:** [diagrams/execution-flow.mmd](../diagrams/execution-flow.mmd)
- **Class structure:** [diagrams/component-architecture.mmd](../diagrams/component-architecture.mmd)
- **Infrastructure:** [diagrams/deployment-architecture.mmd](../diagrams/deployment-architecture.mmd)

---

## 📊 Documentation Statistics

| Metric | Count |
|--------|-------|
| Total documentation files | 20+ |
| Architecture documents | 4 |
| Visual diagrams | 5 |
| Recipe guides | 10+ |
| Total pages (estimated) | 100+ |
| Total size | ~100 KB |

---

## 🔄 Documentation Maintenance

### When to Update

| Change Type | Documents to Update |
|-------------|-------------------|
| New recipe | Architecture overview, diagrams |
| New stage | Recipe docs, pipeline diagram |
| Core framework change | ARCHITECTURE.md, component diagram |
| Infrastructure change | INSTALL.md, deployment diagram |
| New workflow | Recipe README, workflow guide |

### Update Checklist

- [ ] Update relevant markdown files
- [ ] Update diagrams if visual changes
- [ ] Test diagram rendering
- [ ] Update this index if new docs added
- [ ] Update README if major changes
- [ ] Verify all links still work

---

## 📞 Getting Help

- **Documentation issues:** Check this index for the right document
- **Architecture questions:** See [ARCHITECTURE.md](./ARCHITECTURE.md)
- **Setup problems:** See [INSTALL.md](../../INSTALL.md) troubleshooting
- **Recipe issues:** See recipe-specific troubleshooting guides
- **Code questions:** Check code docstrings and comments

---

## 🤝 Contributing to Documentation

1. **For typos/small fixes:** Edit the relevant file directly
2. **For new diagrams:** Add to `diagrams/` and update `DIAGRAMS_SUMMARY.md`
3. **For new sections:** Update relevant docs and this index
4. **For new recipes:** Create recipe docs following finance recipe structure

**Style Guide:**
- Use clear, concise language
- Include code examples where helpful
- Add diagrams for complex concepts
- Keep this index updated
- Test all commands before documenting

---

## 📄 License

All documentation is part of the NVFlow project and follows the Apache-2.0 license.

---

**Version:** 1.0
**Last Updated:** January 21, 2026
**Maintained by:** NVFlow Team

---

## 🚀 Next Steps

1. **New to NVFlow?** → Start with [README.md](../../README.md)
2. **Need quick reference?** → See [ARCHITECTURE_QUICK_REFERENCE.md](./ARCHITECTURE_QUICK_REFERENCE.md)
3. **Want deep understanding?** → Read [ARCHITECTURE.md](./ARCHITECTURE.md)
4. **Visual learner?** → Browse [diagrams/](../diagrams/)
5. **Ready to run?** → Follow [docs/recipes/finance/quick-start.md](../recipes/finance/quick-start.md)

**Happy learning! 🎉**
