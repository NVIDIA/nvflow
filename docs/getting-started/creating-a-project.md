# Create an NVFlow project

An NVFlow project is a user-owned folder containing workflow definitions,
assets, optional recipe DAGs, and optional Python stages. It can live in this
repository, in another repository, or as a Git submodule.

## Quick setup

Start from the runnable
[project starter](../../projects/starter/README.md). Choose its repository
location first; rename and customize it after running the baseline:

```bash
NVFLOW_ROOT="/path/to/nvflow"
NEW_PROJECT_ROOT="/path/to/your-chosen-project-directory"

cp -R "$NVFLOW_ROOT/projects/starter" "$NEW_PROJECT_ROOT"
cd "$NEW_PROJECT_ROOT"
git init
```

Create the external project's lightweight bootstrap environment:

```bash
uv sync
uv run nvflow --help
```

`uv sync` installs the project and lightweight `nvflow` bootstrap using the
tested NVFlow Git revision in `[tool.uv.sources]`. Update its `rev` when
intentionally upgrading NVFlow; no platform resolver policy is copied into the
project.

Commit the copy, including the `uv.lock` that `uv sync` creates. `nvflow run`
submits only committed files and refuses a project with uncommitted changes:

```bash
git add -A
git commit -m "Start from the NVFlow starter"
```

The first semantic command may pause and require network access while
automatically preparing the project's full execution environment in the NVFlow
cache. To prewarm it for CI or offline use:

```bash
uv run nvflow sync-project
```

Worker dependencies remain in worker images. See the
[project and recipe contract](../platform/recipe-contract.md#dependencies) for
the complete dependency boundaries.

The [project starter](../../projects/starter/README.md#repository-contents)
shows the resulting file layout.

## Run the starter before customizing it

Keep the copied identity as `starter` and inspect the working baseline:

```bash
uv run nvflow list-workflows --project starter
uv run nvflow list-stages --workflow starter:generate
uv run nvflow validate --workflow starter:generate
uv run nvflow show-config --workflow starter:generate
```

These commands are local and do not submit work. Next, follow
[Run your first workflow](first-workflow.md) to plan and submit the
complete starter. Keep the starter unchanged until its output is correct.

After the baseline behaves as expected, adapt the copy:

1. Set the project identity in `project.yaml`.
2. Set the distribution name in `pyproject.toml`.
3. Choose a unique `base_output_dir` and model in
   `workflows/generate.yaml`.
4. Replace the starter README, input, prompt, and stage behavior.
5. Add a recipe only when a named scenario should connect complete workflows.

The `my_project` package may keep its starter name. If you rename it, also
update `packages` under `[tool.hatch.build.targets.wheel]` in `pyproject.toml`
and the `project:` stage coordinates in workflow YAML.

## Understand project identity

`project.yaml` gives the distributable project its identity:

```yaml
manifest_version: 1
project: your-project-name
description: "What this project provides"
requires_nvflow: ">=1.2,<2"
python_source_root: .
```

See the [project and recipe contract](../platform/recipe-contract.md) for field definitions,
validation rules, and stage-resolution behavior.

The manifest owns every YAML beneath its `workflows/` directory. Workflow YAML
therefore does not repeat project identity. Each active stage selects either a
shared alias or a project-owned class directly in that workflow. The project
description appears in project and workflow discovery.

Recipes are optional curated entry points. The starter intentionally defines
none. A recipe's `workflows` mapping is keyed directly by extension-free paths
beneath `workflows/`; `depends_on` repeats those keys to create graph edges:

```yaml
recipes:
  quickstart:
    workflows:
      generate: {}
      evaluate/demo:
        depends_on: [generate]
```

Recipes reference complete workflows and cannot add aliases or select stages.
Finance and HopChain provide multi-workflow examples.

## Organize a growing project by workflow family

Keep runnable workflow definitions, reusable configuration, and Python code
separate:

```text
project/
├── project.yaml
├── pyproject.toml
├── workflows/
│   ├── data-preparation/
│   ├── reward-profiling/
│   ├── sft/
│   └── rl/
├── configs/
├── your_project/
│   ├── shared/
│   ├── reward_profiling/
│   ├── sft/
│   └── rl/
├── tests/
└── docs/
```

`workflows/` is the discovery root. Discovery lists each YAML mapping below it
that defines `workflow`, `cluster`, `pipeline_stages`, and `stages`, directly or
through `_base_`, and skips the rest. Put indexed catalogs, reusable defaults
that are not workflows, and other declarative assets under `configs/`. Set
`workflow.runnable: false` in a shared `_base_` template that should not run on
its own: `nvflow list-workflows` hides it unless you pass `--all`, and recipes
cannot reference it. The flag is not inherited, so workflows that extend the
template stay listed.

Mirror workflow-family names in the Python package, tests, and documentation;
move code into `shared/` only after a second family consumes it. Add empty
future-family directories only when that functionality exists.

This is the preferred layout for new multi-workflow projects. Existing
maintained projects can migrate separately after their `uses` coordinates,
project resources, and downstream consumers are reviewed.

## Separate project assets from mounted data

Use `project://` only for small, version-controlled assets that belong to the
project, such as prompts, configuration, scripts, and test fixtures:

```yaml
prompt_config: project://prompts/generate_answer.yaml
stage_config: project://configs/generation.yaml
```

NVFlow resolves these paths locally during validation and to the packaged project
location on workers.

Real datasets, checkpoints, and generated outputs should remain outside the Git
repository. Reference their worker-visible mount destinations with absolute
paths:

```yaml
input_file: /data/team/datasets/train.jsonl
output_dir: /workspace/outputs/your-project-name
```

These paths must match mounts provided by the selected cluster target. Do not
commit large, generated, or sensitive data to the project repository. The
starter's `data/input.jsonl` is intentionally only a small runnable
fixture. See [cluster configuration](../operations/cluster-configuration.md)
for mount setup.

## Validate, inspect, plan, and run

From the project root, inspect and validate the workflow:

```bash
# doc-check: execute
uv run nvflow list-projects
uv run nvflow list-workflows --project your-project-name
uv run nvflow list-stages --workflow your-project-name:generate
uv run nvflow validate --workflow your-project-name:generate
uv run nvflow show-config --workflow your-project-name:generate
```

These commands are target-independent and do not connect to deployment
infrastructure.

Commit the working project before remote execution:

```bash
git add .
git commit -m "Initialize NVFlow project"
```

## Configure deployment infrastructure

Point NVFlow at the cluster configurations maintained by your team:

```bash
export NVFLOW_CONFIG_DIR="/path/to/team/cluster_configs"
```

The directory can serve multiple projects and contain one YAML file per Slurm
or Ray target, such as `my_cluster.yaml`. Keep credentials and site-specific
secrets outside the copied project. Cluster targets reference staged worker images
compatible with the active NVFlow and NeMo versions. Projects may
share compatible images; custom worker dependencies require a compatible
derived image.

Review the static plan without connecting to the cluster:

```bash
uv run nvflow plan \
  --config workflows/generate.yaml \
  --cluster my_cluster
```

Add `--live` only when input/resume discovery and declared-output presence are
needed. On Slurm it connects for read-only filesystem inspection and requires
valid target runtime fields; on Ray it reads the launcher's local filesystem.

After the local checks and static plan pass, submit the workflow:

```bash
uv run nvflow run --all \
  --config workflows/generate.yaml \
  --cluster my_cluster
```

The config path is enough for NVFlow to discover `project.yaml`; `--project-dir`
is only needed when discovery is ambiguous. `--cluster` selects operator-owned
infrastructure without requiring users to edit the copied project.

On Slurm, `run --all` performs that live-equivalent planning before submission;
on Ray, it plans statically.
Execution progress, warnings, and errors use stderr, and the CLI leaves stdout
empty.

## Select shared and project-owned stages

Select every implementation beside its workflow configuration:

`uses` bindings need NVFlow 1.2.0 or later.

```yaml
stages:
  prepare_data:
    uses: nemo_gym.collate
    # ...
  collect_rollouts:
    uses: nemo_gym.collect
    # ...
  generate_answers:
    uses: project:my_project.stages.generate_answers:GenerateAnswersStage
    # ...
```

During normal development, edit only the stage class and workflow YAML. Return
to `project.yaml` when project identity or compatibility changes, or when a
stable workflow becomes part of a curated recipe.

The workflow YAML supplies implementations, environments, mounted
dataset/model paths, endpoint policy, and resources. Put a repeated project
coordinate in a base workflow and inherit it with `_base_`; project authors do
not maintain registration files or decorators. Discover shared types with
`nvflow list-stage-types` and `nvflow stage-type-info <type>`. Specialized
orchestration can use the typed
`nvflow.integrations.nemo_gym.operations` facade directly.

## Authoring constraints

- Implement `execution(context) -> StageExecution`. Declare exact commands,
  runtime roles, resources, dependencies, services, and
  `ArtifactDeclaration`s; NVFlow derives the public plan from that contract.
  Do not submit through NeMo-Skills, NeMo-Run, Ray, or Slurm from project code.
- Keep launcher-only imports inside `execution()` so
  worker imports stay lightweight.
- Keep only semantic NVFlow and project-specific dependencies in
  `pyproject.toml`. The launcher reproduces its active NVFlow source; do not
  commit a parent-repository path or copied platform resolver policy.
- Remote execution requires clean committed project content. Launcher
  environments are not copied to workers, so worker dependencies belong in the
  selected image.

Keep project-specific setup, outputs, quality expectations, and limitations in
the project's README or `docs/`. The
[project and recipe contract](../platform/recipe-contract.md) covers stage
resolution, packaging, and compatibility.

## Further reading

- [Project ownership and recipe contract](../platform/recipe-contract.md)
- [Workflow loading and planning](../platform/workflow-and-planning.md)
- [Project starter](../../projects/starter/README.md)
- [Stage execution migration guide](../development/migrating-stages-to-execution.md)
