# NVFlow Architecture Diagrams

This directory contains architectural diagrams for the NVFlow orchestration framework.

## 📋 Available Diagrams

### 1. **architecture-overview.mmd**
High-level system architecture showing the main components and their relationships:
- User interfaces (CLI, Python API, Scripts)
- Core framework (WorkflowRunner, StageRegistry, BaseStage)
- Recipe layer (Finance, Example, Custom)
- Execution infrastructure (NeMo-Skills, Slurm, Containers)
- Storage layer (Data, Models, Outputs)

**Best for:** Understanding the overall system design and component interaction.

### 2. **finance-pipeline.mmd**
Complete data flow pipeline for the Finance recipe:
- Stage 1: SEC filings download
- Stage 2: Synthetic data generation (Template-based & Document-grounded SDG)
- Stage 3: Data preparation for SFT
- Stage 4: Supervised fine-tuning
- Stage 5: Model evaluation

**Best for:** Understanding the end-to-end ML pipeline flow in the finance domain.

### 3. **execution-flow.mmd**
Sequence diagram showing how workflows are executed:
- User command to job submission
- Config loading and validation
- Stage execution and dependency management
- Integration with NeMo-Skills and Slurm
- Background job execution on compute nodes

**Best for:** Understanding the runtime behavior and execution sequence.

### 4. **component-architecture.mmd**
Class diagram showing core framework components:
- BaseStage abstract class
- StageRegistry singleton
- WorkflowRunner orchestrator
- Concrete stage implementations
- External dependencies

**Best for:** Understanding the code structure and class relationships.

### 5. **deployment-architecture.mmd**
Infrastructure and deployment view:
- Local development environment
- SSH connection to cluster
- Slurm cluster components (login node, compute nodes, scheduler)
- Shared storage (Lustre/NFS)
- Container runtime (Enroot)
- Job execution flow

**Best for:** Understanding the deployment topology and infrastructure requirements.

## 🎨 Viewing the Diagrams

### Option 1: Online Mermaid Live Editor
1. Visit [Mermaid Live Editor](https://mermaid.live/)
2. Copy the contents of any `.mmd` file
3. Paste into the editor
4. View and export as PNG/SVG

### Option 2: VS Code with Mermaid Extension
1. Install [Mermaid Preview](https://marketplace.visualstudio.com/items?itemName=vstirbu.vscode-mermaid-preview) extension
2. Open any `.mmd` file
3. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
4. Select "Mermaid: Preview"

### Option 3: Mermaid CLI (Generate Images)
```bash
# Install Mermaid CLI
npm install -g @mermaid-js/mermaid-cli

# Generate PNG images
mmdc -i architecture-overview.mmd -o architecture-overview.png
mmdc -i finance-pipeline.mmd -o finance-pipeline.png
mmdc -i execution-flow.mmd -o execution-flow.png
mmdc -i component-architecture.mmd -o component-architecture.png
mmdc -i deployment-architecture.mmd -o deployment-architecture.png

# Generate SVG images (better quality)
mmdc -i architecture-overview.mmd -o architecture-overview.svg
```

### Option 4: GitHub Rendering
GitHub supports Mermaid diagrams natively in Markdown files. You can include them in documentation using:

````markdown
```mermaid
graph TB
    A --> B
```
````

### Option 5: Cursor/IDE Preview
Many modern IDEs (including Cursor) have built-in Mermaid preview support. Simply open the `.mmd` file and look for a preview option.

## 📚 Additional Documentation

For detailed architectural descriptions and explanations, see:
- **[ARCHITECTURE.md](../architecture/ARCHITECTURE.md)** - Comprehensive architecture documentation with embedded diagrams
- **[README.md](../../README.md)** - Main project documentation
- **[docs/recipes/finance/README.md](../recipes/finance/README.md)** - Finance recipe documentation

## 🔧 Editing the Diagrams

### Mermaid Syntax Reference
- [Official Mermaid Documentation](https://mermaid.js.org/)
- [Flowchart Syntax](https://mermaid.js.org/syntax/flowchart.html)
- [Sequence Diagram Syntax](https://mermaid.js.org/syntax/sequenceDiagram.html)
- [Class Diagram Syntax](https://mermaid.js.org/syntax/classDiagram.html)

### Style Guide
When editing diagrams, follow these conventions:

**Color Scheme:**
- User Interface: Light blue (`#e1f5ff`)
- Core Framework: Light orange (`#fff4e6`)
- Recipe Layer: Light green (`#e8f5e9`)
- Infrastructure: Light purple (`#f3e5f5`)
- Storage: Light pink (`#fce4ec`)
- Notes: Light yellow (`#fff9c4`)

**Node Naming:**
- Use descriptive, concise labels
- Include key details in subtitle (below node name)
- Add emojis for visual categorization (optional)

**Layout:**
- Top-to-bottom flow for sequential processes
- Left-to-right for parallel components
- Use subgraphs to group related components

## 📝 Contributing

When adding new diagrams:

1. Create a new `.mmd` file in this directory
2. Follow the existing naming convention (kebab-case)
3. Add a header comment explaining the diagram purpose
4. Update this README with a description
5. Test rendering in at least one viewer before committing
6. Consider updating [ARCHITECTURE.md](../architecture/ARCHITECTURE.md) if adding significant architectural information

## 📄 License

These diagrams are part of the NVFlow project and follow the same Apache-2.0 license.

---

**Last Updated:** January 21, 2026
**Maintainer:** NVFlow Team
