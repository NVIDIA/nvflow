# Workflow loading, planning, and execution

NVFlow separates target-independent workflow definition from target-resolved
planning and scheduler submission.

## Workflow YAML

A runnable workflow contains:

```yaml
workflow:
  name: training
  type: training
  description: Prepare the data and train a model

cluster: my_cluster

pipeline_stages:
  - prepare_data
  - training

stages:
  prepare_data:
    uses: project:my_project.stages.prepare:PrepareDataStage
    description: Clean and tokenize the raw records
    output_dir: /workspace/outputs/prepared
    dependencies: []

  training:
    uses: project:my_project.stages.training:TrainingStage
    input_dir: /workspace/outputs/prepared
    output_dir: /workspace/outputs/training
    dependencies: [prepare_data]
```

Both descriptions are optional. `list-workflows` shows `workflow.description`;
`list-stages` and `stage-info` show each stage's `description`, which NVFlow
removes before handing the stage its configuration.

Each YAML file is one concrete `WorkflowDefinition`. Files may share
`workflow.name` while selecting different workflow-local implementations,
models, scale, or configuration. `nvflow list-workflows` groups those
definitions for discovery, while file-derived selectors continue to identify
the exact definition loaded by every other command. The nearest `project.yaml`
owns all definitions beneath its `workflows/` directory; authored workflow YAML
does not repeat project identity.

`cluster` is a required key. The CLI overrides it with `--cluster`, which every
`plan` and `run` invocation must pass, and the Python `WorkflowRunner` falls
back to it only when the caller passes no `cluster`. Target-independent
commands such as `validate` and `show-config` do not need `--cluster`.

## Inheritance

`_base_` composes YAML files child-over-base:

```yaml
_base_: base.yaml

stages:
  training:
    total_gpus: 8
```

Base paths are relative to the child file, must remain inside the owning
project, and cannot form a cycle. NVFlow preserves source and merged views for
inspection and resolves OmegaConf interpolations for execution.

## Substages

A stage can run ordinary stages in order as its `substages`:

```yaml
stages:
  check_questions:
    description: Ask a model to check each question
    container: nemo-gym
    installation_command: export PATH=/opt/nemo_gym_venv/bin:$PATH
    dependencies: [prepare_data]
    substages:
      requests:
        uses: nemo_gym.render
        contract_version: 1
        output_dir: /workspace/outputs/checks/requests
        renders:
          questions:
            input_file: /workspace/outputs/prepared/questions.jsonl
            prompt: project://prompts/check.yaml
      verdicts:
        uses: nemo_gym.collect
        contract_version: 1
        output_dir: /workspace/outputs/checks/gym
        environments:
          checks:
            input_data: /workspace/outputs/checks/requests/questions.jsonl
            config_paths: ["{{NVFLOW_CODE_ROOT}}/nvflow/integrations/nemo_gym/configs/format_verification.yaml"]
            agent_name: sdg_format_verification_simple_agent
        rollout:
          num_repeats: 1
          models:
            policy:
              deployment: external
              model_type: openai_model
              model: my-model
              base_url: https://api.example.com/v1
              api_key_env: MODEL_API_KEY
```

Each substage is configured as it would be on its own, without `dependencies`:
it runs after the one above it. It inherits the stage's `container`, which it
cannot change, and `installation_command`, and keeps its own outputs and
completion records, so a rerun skips the work its records show is finished. On
Slurm, consecutive substage tasks that run one command in the same container on
the same hardware, each depending only on the task before it, run as one job,
each in its own shell, and the job stops at the first failure. A step with its
own container or hardware, such as a GPU embedding pass in `vllm`, and each
chunk of a chunked collection run as their own jobs. Ray runs each task as its
own Jobs submission. `list-stages` and `stage-info` name each stage's
substages.

## WorkflowDefinition

```python
from nvflow import load_workflow

workflow = load_workflow("workflows/training.yaml")
print(workflow.project)
print(workflow.workflow_name)
print(workflow.pipeline_stages)
```

Loading:

- Resolves and validates the owning project
- Merges `_base_` configuration
- Expands configured model/checkpoint stages
- Validates `project://` resources
- Does not bind a concrete operator target
- Does not submit scheduler work

## Configuration inspection

```bash
nvflow show-config --config workflows/training.yaml --view source
nvflow show-config --config workflows/training.yaml --view merged
nvflow show-config --config workflows/training.yaml --view resolved
nvflow show-config --config workflows/training.yaml --view diff

# Effective configuration for selected stages.
nvflow show-config prepare_data training --config workflows/training.yaml
```

Displayed environment-backed and conventionally named credentials are redacted.
Execution retains the resolved values internally.
Stage-scoped inspection uses the effective, dynamically expanded `resolved`
view. Composition views (`source`, `merged`, and `diff`) remain workflow-wide.

## Validation

```bash
nvflow validate --config workflows/training.yaml
nvflow validate prepare_data training --config workflows/training.yaml
```

Semantic validation imports the explicitly selected trusted project and checks:

- Required workflow fields
- Literal shared or project-local stage implementation
- Stage configuration
- Transitive dependencies
- Project-owned paths

Before that import, the CLI transparently prepares or reuses the project's
isolated launcher environment. Structural recipe planning does not require it.

Optional positional stages validate only those active stages and their
transitive dependencies. Omitting them validates the complete workflow.
Validation does not submit work.

## Recipe graph validation and planning

A recipe references one or more complete reusable workflow definitions from
`project.yaml`.
Its graph is intentionally separate from a workflow's task-level `LaunchPlan`:

```bash
nvflow validate --recipe finance:quickstart
nvflow plan-recipe finance:quickstart
nvflow plan-recipe finance:quickstart --format json
```

Both commands are target-independent. They resolve selector-keyed workflow
references, reject dangling dependencies and cycles, and record project
provenance. They do not import stage code, inspect cluster
storage, merge child launch plans, wire scheduler dependencies, or submit
workflows. Run a referenced workflow with the existing `--workflow` commands
when execution is required.

## Target binding

```python
from nvflow.execution import TargetBinding

target = TargetBinding.resolve(
    workflow_default=workflow.cluster,
    override="my_cluster",
)
```

NVFlow currently binds one workflow invocation to one concrete target.
Multi-target placement is not part of this contract. On the CLI, target binding
is always explicit through required `--cluster`; selecting it does not modify
workflow YAML.

## Launch planning

```python
from nvflow.execution import WorkflowPlanner

plan = WorkflowPlanner(workflow, target).plan()
assert plan.complete
print(plan.digest)
```

The plan reports:

- Stable plan-local task IDs
- Static/live planning mode and selected environment filters
- Commands and container roles
- Resources
- Internal and external dependencies
- Service/readiness relationships where declared
- Expected outputs
- Stage/task inclusion and whether each selected stage is inspectable
- Structured stage implementation identity (kind, catalog name, contract
  version, provider class, and active NVFlow version)
- Project commit, source subpath, materialized content digest, and submission eligibility

Serialized workflow and recipe graph plans use `plan_version: 2`.
Their `kind` field distinguishes `workflow` from `recipe` documents.
`manifest_version` is reserved for `project.yaml`.

Static planning loads and checks the explicitly selected target YAML locally,
then describes configured work without inspecting workflow input or output
filesystem state. Live planning enables target-aware, read-only input/resume
discovery plus uniform presence checks for declared outputs. It may be slower
because stages can reach storage through the target's filesystem transport;
the choice of transport is an implementation detail.

The static target check verifies the selected engine schema. Slurm-specific
runtime fields remain lazy; supplied Ray targets must declare their control
cluster, artifact root, endpoint references, qualified runtime versions, and
role routes. Neither static nor live planning contacts those endpoints.

Neither mode queries Slurm, submits jobs, or mutates project outputs, scheduler
state, mounts, or packaging state. Internal task references are validated and
dependency cycles fail planning.

Human-readable live planning reports `exists`, `missing`, `not declared`, or
`unavailable` from explicit `ArtifactDeclaration` paths. Only the outputs of the
tasks that finish a stage count, so data shared with other workflows does not.
A cyan marker identifies stages with such an existing output, without claiming
that the stage was launched or completed. One cached target filesystem is
shared by the plan; stage-specific resume checks and the final batched
declaration check reuse its observations. The plan digest excludes inspection mode and remains
available in verbose, JSON, and Python output for machine correlation rather
than in the compact human view.

The CLI equivalent is:

```bash
nvflow plan \
  --config workflows/training.yaml \
  --cluster my_cluster

nvflow plan prepare_data training \
  --config workflows/training.yaml \
  --cluster my_cluster

# Enable stage-specific filesystem discovery and resume logic.
nvflow plan \
  --config workflows/training.yaml \
  --cluster my_cluster \
  --live
```

The default renderer shows submission eligibility, selected environments,
workflow-level resource bounds, and an aligned task/resource summary by stage.
On a Slurm target it also counts the jobs each stage submits; substage steps
that share a job make that count lower than the task count. Use `--verbose` for full provenance, task details, and output paths, or
`--format json` for one complete machine-readable, redacted stdout document.
Stage-builder output is suppressed; structured warnings and optional `--debug`
diagnostics are written to stderr. More generally, stdout is reserved for the
command payload, while progress, warnings, and errors use stderr.

For NeMo-Skills generation tasks, the listed service and client commands are
the supplied-Ray lowering. Direct Slurm submits the same invocation through
NeMo-Skills `generate` with its captured arguments. Both engines run the listed
merge task as its own job after the chunks.

Optional positional stages restrict task generation to that subset.
Dependencies outside the subset are assumed to be satisfied by an earlier run;
they are validated structurally but omitted from the plan.

## Stage planning

Every runnable stage produces one exact `StageExecution`; there is no
execute-only compatibility path. Public plans are redacted projections of that
same execution object.

Common project-owned stages can use:

```python
from nvflow.execution import StageExecution, plan_run_cmd


def execution(self, context):
    return StageExecution.from_plan(
        plan_run_cmd(
            context,
            "python -m my_recipe.prepare",
            cluster=context.target.cluster,
            output_dir=context.config["output_dir"],
            expname=context.expname,
            run_after=list(context.run_after) or None,
        )
    )
```

`plan_generate` provides exact generation client/service commands. Custom
builders attach `ArtifactDeclaration` and typed service/dependency contracts
directly.

## Execution

```bash
nvflow run --all \
  --config workflows/training.yaml \
  --cluster my_cluster
```

On Slurm, `run` performs live-equivalent planning before submission,
including each stage's filesystem discovery and resume logic. This can add the
same storage latency as `plan --live`. On Ray, `run` plans statically and the
workload scripts apply their resume guards on the clusters. NVFlow dispatches
the exact graph once: to the isolated direct-Slurm engine or to the supplied-Ray
Jobs engine.

The CLI keeps stdout empty for `run`; submission progress,
warnings, and errors are written to stderr.

Programmatic execution returns a provider-neutral receipt with plan and
execution digests, engine, target fingerprint, project provenance, tasks, and
available scheduler/Jobs handles. Ray runs additionally persist controller
state and append-only events under the shared artifact root.

## Related documentation

- [Project and recipe contract](recipe-contract.md)
- [CLI and Python API](cli-and-python-api.md)
- [Cluster configuration](../operations/cluster-configuration.md)
