# NVFlow concepts

NVFlow is the workflow orchestration platform. It separates orchestration from
the applications it runs.

## Ownership

### Platform

The NVFlow platform owns:

- Project, recipe, and workflow loading
- Project discovery and path resolution
- Recipe graph validation and target-independent graph planning
- Structural and semantic validation
- Shared stages that any project can reuse, such as the NeMo Gym and
  document-grounded SDG suites
- Shared and project-local stage resolution
- Target binding and launch planning
- Code packaging and scheduler integration
- CLI and Python APIs

Platform source lives under `nvflow/`. It must not import concrete project
packages.

### Project

A project is a self-contained application bundle that owns:

- Workflow YAML
- Optional named recipe DAGs
- Domain stages and utilities
- Prompts, assets, and configuration
- Python dependencies
- Tests, documentation, and quality expectations

Maintained projects and the copyable starter live under `projects/`. The same
contract supports a project in another Git repository or as a submodule.
The root NVFlow installation does not absorb project dependencies. A copied
project can `uv sync` a lightweight local bootstrap, while commands that import
project Python transparently prepare one full cached environment for that
project; all of its recipes and workflows share it.

### Operator

The operator owns:

- Cluster configuration
- Slurm accounts/partitions or supplied Ray Jobs endpoints/routes
- Container or running cluster image identities
- Host/container mounts or a shared Ray artifact root
- Models, writable storage, and secrets

Every workflow records a default cluster for programmatic use, but CLI operators
must explicitly pass `--cluster` to `plan` and `run`. Selecting a
target does not edit project content. Discovery, `validate`, and `show-config`
remain target-independent.

## Project

`project.yaml` defines project identity, compatibility, and optional recipe
DAGs. This excerpt from the finance project's manifest shows its identity and
its `quickstart` recipe:

```yaml
manifest_version: 1
project: finance
description: "End-to-end pipeline for financial model training and evaluation"
requires_nvflow: ">=1.1.3.dev0,<2"
python_source_root: .

recipes:
  quickstart:
    description: "Template-based SDG quickstart"
    workflows:
      eval/demo: {}
      download/demo: {}
      sdg/template-based-sdg-demo:
        depends_on: [download/demo]
      sft/qwen3_4b:
        depends_on: [eval/demo, sdg/template-based-sdg-demo]
      grpo/qwen3_4b:
        depends_on: [eval/demo, sdg/template-based-sdg-demo]
```

The nearest selected manifest establishes the project root. Duplicate
identities, nested manifests, unsupported versions, and paths escaping the
project fail before submission. Workflow definitions are discovered and shown
alphabetically by their path-derived selector.

## Recipe

A recipe is a project-scoped DAG of complete workflow references. Mapping keys
are extension-free paths beneath the project's `workflows/` directory.
`depends_on` repeats those selector keys to create edges; YAML order only
stabilizes display. A node may be marked `optional`. Recipes do not add aliases
or select stage subsets. A one-workflow recipe uses the same mapping shape, for
example `workflows: {generate: {}}`.

`nvflow validate --recipe finance:quickstart` validates the graph and every reference.
`nvflow plan-recipe finance:quickstart` renders a target-independent graph plan.
NVFlow does not submit a multi-workflow recipe DAG yet. `plan`, `run`, and
`run --all` continue to operate on one workflow definition.

## Workflow

A workflow is an executable DAG of stages. Each workflow YAML is one concrete
definition/configuration containing:

- `workflow.name`: local metadata used to group related definitions
- `workflow.type`: descriptive category such as `sdg`, `training`, or `evaluation`
- `cluster`: required default target name for programmatic callers; the CLI
  always takes the target from `--cluster`
- `pipeline_stages`: ordered stage selection
- `stages`: literal `uses` implementations, configuration, and dependencies

`_base_` composes YAML files relative to the current file. A child overrides its
base. Multiple files may share `workflow.name` while selecting different
models, scale, or stage configuration. Their stable CLI selectors remain
file-derived, for example `finance:sft/qwen3_4b`. The nearest `project.yaml`
owns every workflow beneath `workflows/`; authored workflow YAML does not repeat
the project identity.

## Stage

A stage is a selected Python class that validates configuration and contributes
exact work to a workflow.

- `validate_config()` checks project-owned fields.
- `execution()` returns exact commands, dependencies, resources, runtime roles,
  services, and outputs without scheduler submission.
- `LaunchPlan` is derived from that same exact contract; execute-only stages are
  rejected.

A workflow's `uses` field selects each implementation from one of two sources:

- **Project stages** are owned by the project and selected with a
  `project:python.module:ClassName` coordinate.
- **Shared stages** are owned by NVFlow and selected by catalog name, such as
  `nemo_gym.collect`, `nemo_rl.sft`, or `data.train_validation_split`. Run
  `nvflow list-stage-types` to list them.

NVFlow registers stages beneath `project.workflow.stage`; source folder names do
not change that identity.

## Run, task, and step

A **run** or **invocation** is one `nvflow plan` or `nvflow run` operation against
a workflow and target. `run` accepts selected stages; `run --all` selects the
complete `pipeline_stages` list.

A stage can contribute one or more **tasks** to a launch plan. A task represents
a scheduler command or service with its own resources and dependencies. Most
users work with projects, recipes, workflows, and stages; tasks are primarily
useful when inspecting a workflow launch plan.

**Step** is not a separate NVFlow construct. Project documentation and output
directories may number stages as steps for readability, but stage identity and
execution order come from `pipeline_stages`.

## Project-relative path

Use `project://` for small, version-controlled files owned by the project:

```yaml
prompt_config: project://prompts/generate.yaml
```

NVFlow validates the path against the launcher project root, then renders it
against the packaged worker root. Absolute cluster paths, model paths, URLs, and
output paths are not rewritten.

`project://` is for resource values inside a selected workflow. The CLI
`--config` option always receives a launcher filesystem path, such as
`projects/starter/workflows/generate.yaml`.

## Workflow definition and plan

`load_workflow()` produces a target-independent `WorkflowDefinition`. It can be
inspected or validated without scheduler submission. Planning binds one target
and produces an immutable `LaunchPlan` with stable task IDs, declared resources
and outputs, provenance, and a digest. Static planning avoids storage access;
`--live` adds read-only filesystem discovery, through the target on Slurm and on
the launcher on Ray. Neither mode submits work.

See [Workflow loading, planning, and execution](../platform/workflow-and-planning.md)
for the detailed execution contract.

CLI workflow selectors use `project:path-under-workflows`, for example
`finance:grpo/qwen3_4b`. Recipe selectors use the same shape in recipe command
context, for example `finance:quickstart`. `--workflow` and `--recipe`
disambiguate them.

## Submission receipt

The Python execution API returns a non-durable `SubmissionReceipt` containing:

- Plan digest
- Effective target
- Planned task identities
- Project source provenance
- Best-effort experiment handles NVFlow can discover

Durable history, artifact lineage, and long-term run records are separate
provenance concerns.

## Next steps

- [Run the starter workflow](first-workflow.md)
- [Create an NVFlow project](creating-a-project.md)
- [Read the project and recipe contract](../platform/recipe-contract.md)
- [Understand planning and execution](../platform/workflow-and-planning.md)
