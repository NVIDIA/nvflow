# CLI and Python API

The CLI and Python API share the same project resolution, workflow definition,
validation, and execution behavior; Python planning differs only as noted under
the Python API's Plan section.

## CLI

Run commands through the project environment:

```bash
uv run nvflow --help
uv run nvflow <command> --help
```

`-h` is available for the root CLI and every command. Common options keep the
same short form wherever they appear: `-p/--project`, `-w/--workflow`,
`-c/--config`, `-e/--environment`, `-r/--recipe`, and `-f/--format`.
Command-specific options use `plan -v/--verbose` and
`show-config -V/--view`, avoiding a short-flag collision.
Command-specific help is the authoritative option reference.

### Select a workflow

Commands that load one workflow definition accept exactly one of:

```bash
--workflow finance:download/demo
--config projects/finance/workflows/download/demo.yaml
```

The selector format is `<project>:<path-under-workflows-without-.yaml>`. NVFlow
derives selectors from existing project manifests and folders; project authors do
not maintain aliases or indexes. The nearest `project.yaml` owns every workflow
beneath its `workflows/` directory, so workflow YAML does not repeat `project:`.
`workflow.name` is local grouping metadata, not a selector or a
second project association. Use `--config` for arbitrary YAML paths.

From inside an external project, its selectors resolve automatically from the
nearest `project.yaml`. From elsewhere, use repeatable `--project-path`,
`NVFLOW_PROJECT_PATH`, or an explicit `--project-dir`. Direct config paths also
discover their owning manifest.

Discovery, `validate`, and `show-config` are target-independent. `plan` and
`run` require an explicit `--cluster <name>`; the CLI does not infer that
selection from a `cluster` value in workflow YAML.

### Discover workflows

```bash
nvflow list-projects
nvflow list-recipes
nvflow list-recipes --project finance
nvflow list-workflows
nvflow list-workflows --project finance
nvflow list-workflows --recipe finance:quickstart
nvflow list-workflows --project-path /team/projects
nvflow list-workflows --all  # include non-runnable templates
```

Discovery reads manifests and YAML structure only. It does not import project
stages or NeMo libraries and does not contact a cluster.

- `list-projects` shows independently owned project packages.
- `list-recipes` shows named workflow DAGs from every project;
  `--project` narrows the result.
- `list-workflows --project` shows reusable workflow definitions alphabetically
  by selector.
- `list-workflows --recipe` uses the same table for the recipe's complete
  workflows, ordered by dependencies and marked when optional.
- `plan-recipe` shows dependency edges and provenance when graph details are
  needed.

Each listing shows descriptions: `list-projects` from `project.yaml`,
`list-recipes` from each recipe, `list-workflows` from `workflow.description`,
and `list-stages` from each stage's `description`.

Default project workflow output contains only runnable definitions; `--all`
adds templates. Structural consistency checks include templates in either mode.

Demo, smoke, model, and production choices are separate workflow definitions.
For example, finance download provides `finance:download/smoke`,
`finance:download/demo`, and `finance:download/sp500`. Demo and sp500 run
`download`, `build_index` and `prefetch_cache`; smoke runs the first two. Use
`list-stages --workflow <selector>` to inspect
the configured stages in any definition.

### Prepare project environments

The root NVFlow checkout and a copied project's lightweight `.venv` do not
absorb that project's full launcher stack. Semantic commands for a selected
project with a `pyproject.toml` prepare a fingerprinted execution environment
in the user cache and continue there with the same arguments and output
streams. Workflow `uses` values import selected project classes after handoff.
The behavior applies equally to maintained, external, and submodule projects;
recipes within one project share the environment.

First semantic use may pause and require network access. Prewarm one project—or
all discoverable projects—for CI or offline operation:

```bash
nvflow sync-project --project finance
nvflow sync-project --project-dir /team/project
nvflow sync-project --all
nvflow sync-project --project finance --check    # read-only readiness
nvflow sync-project --project finance --upgrade  # publish a newly resolved cache generation
```

The cache owns its synthetic `pyproject.toml`, `uv.lock`, and venv. A copied
project separately owns the lightweight `.venv` and bootstrap lock created by
`uv sync`; it never copies NVFlow override policy. Normal project, recipe,
workflow, and structural stage discovery does not prepare the full cache. Set
`NVFLOW_PROJECT_ENV_CACHE` to relocate the cache, for example to a path baked
into an offline launcher image.

### Discover stages

```bash
nvflow list-stages --workflow finance:download/demo
nvflow list-stages --config projects/finance/workflows/download/demo.yaml
```

`list-stages` reads unresolved YAML structure:
it does not require runtime secrets, import stage classes, or prepare a project
environment. It shows each workflow alias, literal implementation, dependency,
and path in pipeline order. A bare `list-stages` prints scoped usage guidance.

`--workflow` always requires an exact `project:path` selector.
YAML `workflow.type` remains a separate descriptive category.

### Discover environment selectors

```bash
nvflow list-environments --workflow finance:grpo/qwen3_4b
```

`list-environments` prints the workflow-level names accepted by
`--environment`/`-e` and whether each entry is enabled. It does not inspect or
import any integration, including Gym. A workflow without an environment
catalog reports that `-e` is not applicable and exits successfully.

### Discover shared stage types

```bash
nvflow list-stage-types
nvflow list-stage-types --namespace nemo_gym
nvflow stage-type-info nemo_gym.collect
```

`list-stage-types` lists the curated platform catalog. `--namespace` (`-n`)
limits the table to names beginning with `<namespace>.`. `stage-type-info`
accepts one exact type name and prints its implementation class, supported
contract versions, documentation path, description, and class documentation.
An unknown type exits nonzero and points back to `list-stage-types`.

These commands inspect the platform catalog. They do not select a workflow,
import project Python, prepare a project environment, or contact a cluster.
This differs from `list-stages`, which inspects stage aliases in a workflow,
and `stage-info`, which resolves one configured alias.

Select a shared implementation beside a project-owned stage alias:

```yaml
stages:
  collect_rollouts:
    uses: nemo_gym.collect
    contract_version: 1
  generate_answers:
    uses: project:my_project.stages.generate_answers:GenerateAnswersStage
```

`uses` needs NVFlow 1.2.0 or later.
`uses` must be a literal shared alias or
`project:python.module:BaseStageSubclass` coordinate. NVFlow removes this
implementation selector before validating or executing runtime configuration.
Put repeated project coordinates in a base workflow; no project registration
file or decorator is involved.

### Inspect one stage

```bash
nvflow stage-info download \
  --workflow finance:download/demo
```

### Validate

```bash
nvflow validate \
  --workflow finance:download/demo

nvflow validate download \
  --workflow finance:download/demo

nvflow validate \
  --workflow finance:eval/demo \
  -e financebench

nvflow validate --recipe finance:quickstart
nvflow plan-recipe finance:quickstart
```

Workflow validation checks the project manifest, local project resources, stage
implementation and configuration, and the dependency graph. It does not contact
the cluster or check images, models, mounts, or cluster-only data.
Optional positional stage names scope configuration validation to those active
stages and their transitive dependencies. With no stage names, validation checks
the complete workflow. Repeatable `--environment`, `-e` options validate one
fan-out subset without contacting the cluster.

Recipe validation is target-independent and does not import stage code. It
checks every complete-workflow selector, optional flag, dependency reference,
and cycle. `plan-recipe` renders the same graph with
project provenance and a deterministic digest. It does not submit workflows.

Options:

- `--workflow`, `-w`: stable project workflow selector
- `--config`, `-c`: launcher path to an arbitrary workflow YAML
- `--environment`, `-e`: environment/fan-out values to validate
- `--project-dir`: explicit external project root
- `--project-path`: ordered, repeatable project search path

### Inspect configuration

```bash
nvflow show-config \
  --workflow finance:download/demo

nvflow show-config download \
  --workflow finance:download/demo

nvflow show-config collect_rollouts \
  --workflow finance:grpo/qwen3_4b \
  -e finance_sec_search
```

The default `resolved` view shows effective values with secrets redacted.
Optional positional stage names return only those effective stage
configurations. Stage-scoped output supports the `resolved` view only because
`source`, `merged`, and `diff` describe workflow composition rather than
expanded stage execution. Repeatable `--environment`, `-e` options narrow a
stage-scoped resolved view to the requested fan-out values. When a project
catalog and `environments` contain the same entries, the internal alias is
omitted from this display. Advanced workflow-wide views are `source`, `merged`,
and `diff`. `source`, `merged`, and `diff` remain unresolved and do not require
runtime environment variables; only `resolved` may require them.

Use `--project-dir` or repeatable `--project-path` for external projects.

### Plan

```bash
nvflow plan \
  --workflow finance:download/demo \
  --cluster my_cluster

nvflow plan download \
  --workflow finance:download/demo \
  --cluster my_cluster
```

The default static plan loads the selected target YAML but does not inspect
workflow storage. `--live` adds read-only input/resume discovery and declared
output presence checks through one cached target filesystem. It can be slower
and requires valid filesystem transport fields, but it does not query Slurm or
submit work.

Add `--verbose` for the full plan digest, project provenance, every task, command,
dependency, resource request, and output path. Add `--format json` for the
complete machine-readable redacted plan. `--debug` exposes captured stage-builder
diagnostics on stderr. See
[Workflow loading and planning](workflow-and-planning.md#launch-planning) for
output-status, resource-accounting, and digest semantics.

Planning never submits work. Required `--cluster` selects the operator target
without changing project workflow YAML. Planning resolves commands, resources,
services, dependencies, outputs, and project provenance for that target.

Optional positional stage names plan only those active stages. Dependencies
outside the requested subset are treated as already satisfied, matching
stage-scoped `run` behavior. With no stage names, planning covers the complete
workflow.

Use repeatable `--environment`, `-e` options with `validate`, `plan`, and `run`
to keep the same environment subset through preflight and submission.

### Run selected stages

```bash
nvflow run download \
  --workflow finance:download/demo \
  --cluster my_cluster
```

Multiple stage names may be supplied. `--environment` narrows workflows that
define environment catalogs. `--cluster` selects the operator target;
`--project-dir` and `--project-path` select external projects.

On Slurm, `run` performs live-equivalent task-selection planning before
submission, including stage-specific filesystem discovery and resume logic. It
skips the final informational declared-output batch used only by human-readable
`plan --live`. On Ray, `run` plans statically and the workload scripts apply
their resume guards on the clusters. There is no separate `--live` switch for
execution.

### Run the configured workflow

```bash
nvflow run --all \
  --workflow finance:download/demo \
  --cluster my_cluster
```

`run --all` submits the configured `pipeline_stages` in order. It accepts the same
`--workflow`/`--config`, `--cluster`, `--environment`, `--project-dir`, and
`--project-path` selection options as `run`, and plans the same way.

### Check job status

```bash
nvflow status --cluster my_cluster
nvflow status --cluster my_cluster --workflow finance:download/demo
nvflow status --cluster my_cluster --recipe finance:quickstart
```

`status` lists your queued and running jobs on the target in one table for
both backends: `squeue --me` on Slurm, and the NVFlow Jobs you submitted on
every cluster of a Ray target. Rows are in submission order, by job ID on Slurm
and by submission time on Ray. It reaches the cluster through the target's
`ssh_tunnel` when one is configured, so it works wherever `run` does. Like
`run`, it continues in a project environment: the project you are in, or the
Starter project when a checkout without the execution stack runs it.

`run` labels each job it submits with its workflow selector, stage, and
experiment, as the Slurm job comment or as Ray Job metadata, and `status`
shows them. Select a workflow with `--workflow` or `--config`, optionally with
stage names, or a recipe with `--recipe`, to list only those jobs. Unlabeled
jobs, such as ones submitted before labeling, appear only when nothing is
selected. On Slurm, a comment the target's `extra_sbatch_args` or a stage's
`sbatch_kwargs` sets is kept instead of the label. `--verbose` adds nodes, GPUs
per node, CPUs, start time, account, and priority; Ray reports only the planned
nodes and GPUs. Ray's controller submits a step once its dependencies finish,
so `status` lists the steps still waiting, from the run records, as `PENDING`
with reason `Dependency`, as Slurm shows them.

### Output streams

CLI stdout is reserved for the command payload. For example,
`plan --format json` writes one machine-readable document to stdout. Progress,
warnings, diagnostics, and errors go to stderr so redirecting or parsing stdout
does not mix them into the payload.

`run` intentionally writes no stdout payload. Its submission progress and
warnings remain on stderr; use the Python API when an in-process
`SubmissionReceipt` is required.

### Version

```bash
nvflow version
```

## Python workflow API

Transparent project-environment handoff is a CLI feature. The Python API stays
in-process and cannot replace its caller's interpreter; callers that validate,
plan, or run project stages must already execute in a prepared project
environment. Structural loading and configuration inspection do not import the
project stage class. The lightweight external-project `.venv` is therefore a CLI
bootstrap, not an in-process execution environment. Checkout users developing
against the Python execution API should install `uv sync --extra execution`.

For script-oriented debugging, `scripts/run_flow.py` is a compatibility shim
over the packaged module. Both stay in the caller's interpreter, so PuDB can
attach without being replaced. Unlike normal `nvflow` commands, this debugging
path intentionally skips automatic handoff. It loads the selected project
source directly but requires that project's launcher dependencies in the
current interpreter:

```bash
uv run python -m pudb.run -m nvflow.cli.run_workflow \
  training \
  --workflow finance:sft/qwen3_4b \
  --cluster my_cluster
```

### Load

```python
from nvflow import load_workflow

workflow = load_workflow("projects/finance/workflows/download/demo.yaml")
```

For an external project:

```python
workflow = load_workflow(
    "workflows/generate.yaml",
    project_dir="/path/to/my-project",
)
```

Search by configured roots:

```python
workflow = load_workflow(
    "workflows/generate.yaml",
    project_search_paths=("/path/to/my-project", "/shared/projects"),
)
```

### Inspect

```python
print(workflow.project)
print(workflow.workflow_name)
print(workflow.pipeline_stages)
print(workflow.config_view())
print(workflow.config_view(stages=["download"]))
```

### Validate

```python
workflow.validate()
workflow.validate(stages=["download"])
```

### Plan

```python
from nvflow.execution import TargetBinding, WorkflowPlanner
from nvflow.integrations.nemo_skills import target_filesystem

target = TargetBinding.resolve(workflow.cluster, "my_cluster")
plan = WorkflowPlanner(workflow, target).plan()
stage_plan = WorkflowPlanner(workflow, target).plan(stages=["download"])
# Opt in to target-aware selection and declared-output presence:
live_planner = WorkflowPlanner(
    workflow,
    target,
    filesystem=target_filesystem(target.cluster),
)
live_plan = live_planner.plan(live=True)
print(live_planner.output_inspections)

print(plan.digest)
print(plan.to_dict())
```

`WorkflowPlanner.plan()` plans without a resolved target, so stages that size
jobs from the target, such as its `gpus_per_node`, use their defaults.
`nvflow plan` and `nvflow run` plan against the resolved target.

Planning fails immediately when a selected stage lacks its mandatory exact
execution contract; there is no opaque-stage compatibility mode.

`to_dict()` is the in-process representation. Use CLI JSON output when printing
or sharing plans because the CLI applies workflow-aware secret redaction.

### Run

```python
from nvflow.execution import WorkflowRunner

runner = WorkflowRunner(
    "projects/finance/workflows/download/demo.yaml",
    cluster="my_cluster",
)
receipt = runner.run()

print(receipt.plan_digest)
print(receipt.target)
print(receipt.tasks)
print(receipt.experiments)
print(receipt.provenance)
```

Run selected stages:

```python
receipt = runner.run(stages=["download"])
```

## Python recipe graph API

```python
from nvflow.project import ProjectResolver
from nvflow.recipe.planning import plan_recipe
from nvflow.recipe.resolver import resolve_recipe

resolver = ProjectResolver()
recipe = resolve_recipe("finance:quickstart", resolver)
graph_plan = plan_recipe("finance:quickstart", resolver)

print(recipe.topological_workflows())
print(graph_plan.digest)
print(graph_plan.to_dict())
```

These APIs are target-independent and do not submit the referenced workflows.

## Stable versus internal surfaces

Project authors should use the exported APIs shown above. Modules beneath
`nvflow.integrations.nemo_gym` are project-facing only where listed in the
[NeMo ecosystem integration guide](integrations/nemo-ecosystem.md). Capture internals, integration
script builders and workers, raw NeMo-Skills helper objects, and private runner
methods are implementation details.

## Related documentation

- [Concepts](../getting-started/concepts.md)
- [Workflow loading and planning](workflow-and-planning.md)
- [Project and recipe contract](recipe-contract.md)
