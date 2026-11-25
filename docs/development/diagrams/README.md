# Architecture diagram sources

These Mermaid sources accompany the
[platform architecture](../../platform/architecture.md):

- `architecture-overview.mmd`: platform, selected project, and operator boundary
- `component-architecture.mmd`: public components and relationships
- `execution-flow.mmd`: load, plan, execute, and receipt sequence
- `deployment-architecture.mmd`: launcher package and Slurm worker topology

Concrete project and recipe diagrams belong inside the project. The finance pipeline
source lives at
[`projects/finance/docs/finance-pipeline.mmd`](../../../projects/finance/docs/finance-pipeline.mmd).

## Preview

Use an IDE Mermaid extension, the
[Mermaid Live Editor](https://mermaid.live/), or Mermaid CLI:

```bash
mmdc \
  -i docs/development/diagrams/architecture-overview.mmd \
  -o architecture-overview.svg
```

Keep diagrams backend-neutral unless the page explicitly documents an execution
adapter. Do not encode concrete project stages, model counts, cluster sizes, or
quality claims in platform diagrams.
