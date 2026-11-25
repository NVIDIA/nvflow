# Project ownership and recipe graph contract

NVFlow supports projects as independently owned application bundles. Maintained
projects live under `projects/`; a project may also live in another Git
repository or be attached as a Git submodule.

## Ownership boundary

NVFlow supports project, recipe, workflow, and stage contracts. It does not own
or guarantee a project's domain behavior, models, prompts, data, outputs, or
quality.

The project owner maintains:

```text
my-project/
├── project.yaml
├── pyproject.toml
├── README.md
├── workflows/
├── prompts/
├── configs/
├── assets/
├── my_project/
│   ├── __init__.py
│   ├── stages/
│   └── utils/
├── tests/
└── docs/
```

A project owns reusable workflow definitions and may define multiple recipes.
A workflow is an executable stage DAG stored in one YAML file. A recipe is a
named DAG of references to those workflow files. Recipes organize scenarios;
they do not duplicate workflow configuration or create a second package.

## Manifest version 1

`manifest_version: 1` is required and is the only supported project-manifest
schema. NVFlow does not carry a legacy project-manifest parser.

```yaml
manifest_version: 1
project: my-project
description: "My workflow application collection"
requires_nvflow: ">=1.2,<2"
python_source_root: .

recipes:
  quickstart:
    description: "Prepare data, then train and evaluate"
    workflows:
      data/prepare: {}
      training/small:
        depends_on: [data/prepare]
      evaluation/small:
        depends_on: [training/small]
        optional: true
```

### Fields

- `manifest_version` selects the manifest schema.
- `project` is the stable package and workflow identity.
- `description` appears in project and workflow discovery.
- `requires_nvflow` declares compatible platform versions.
- `python_source_root` is a project-relative import root.
- `recipes` optionally defines project-scoped recipe DAGs inline.

A recipe contains an optional `description` and a non-empty `workflows`
mapping, even when it references only one workflow. Each mapping key is an
extension-free, project-relative workflow selector beneath `workflows/`. A node
may declare only `depends_on` and `optional`; dependencies repeat those same
selector keys. Recipes always reference complete workflows and do not introduce
aliases, invocation names, stage subsets, or environment subsets. A distinct
invocation needs a distinct workflow definition, and duplicate selector keys
are rejected.

Dependency order comes only from `depends_on`; YAML order stabilizes display.
NVFlow rejects dangling dependencies, self-dependencies, cycles, unknown
workflow references, and selectors outside the project.

Unknown fields, incompatible versions, invalid identities, and source roots that
escape the project fail before stage import.

## Resolution

For an explicit workflow path, NVFlow resolves the project in this order:

1. Explicit `--project-dir` or Python `project_dir`
2. Nearest `project.yaml` while walking upward within the traversal boundary
3. Ordered `--project-path`, Python `project_search_paths`, or
   `NVFLOW_PROJECT_PATH` entries
4. A curated top-level project in an NVFlow source checkout

The nearest or explicitly selected manifest owns the workflow. Workflow YAML
does not declare a second `project` identity. Its `workflow.name` remains local
grouping metadata; the path beneath `workflows/` is the
selector. Ambiguous or nested manifests and silent stage shadowing are rejected.
Ordered external search paths may intentionally override a curated project
identity.

Resolution performs no network fetch and does not install project dependencies.

## Trust

Manifest discovery, compatibility checks, and project-path validation do not
import project Python.

Semantic workflow validation, planning, and execution import only the classes
selected by active workflow `uses` values. An operator explicitly selecting a
project is the trust decision; project code has the same privileges as other
application code in that process. Recipe graph discovery and `plan-recipe` do
not import stage code.

## Stage implementations

Each active stage declares one literal implementation in workflow YAML:

```yaml
stages:
  prepare_data:
    uses: nemo_gym.collate
    contract_version: 1
  generate_answers:
    uses: project:my_project.stages.generate_answers:GenerateAnswersStage
```

Shared aliases resolve through NVFlow's curated catalog. `project:` coordinates
resolve one `BaseStage` subclass beneath the selected project's
`python_source_root`. NVFlow rejects interpolated coordinates, modules outside
that root, non-stage classes, and active stages without `uses`. It imports no
unrelated project stage modules. A stage that groups
[`substages`](workflow-and-planning.md#substages) is the exception: it must not
declare `uses`, and each of its substages declares its own.

## Recipe discovery and graph planning

Project and recipe selectors are qualified:

```text
project:recipe
```

Workflow selectors have the same shape but are interpreted by workflow
commands:

```text
project:path-under-workflows
```

The option provides the context: `--recipe finance:quickstart` selects a
recipe DAG, while `--workflow finance:sft/qwen3_4b` selects one workflow
definition.

`nvflow validate --recipe` resolves every complete workflow selector and checks
the recipe DAG without a target.
`nvflow plan-recipe` emits a deterministic `RecipeGraphPlan` with project
provenance. Each graph node contains one project-qualified `selector`,
`workflow_type`, `depends_on`, and `optional` marker—no alias or stage subset.
This phase does not create child `LaunchPlan` objects, query a cluster, package
code, or submit the recipe. Workflow `plan` and `run` remain single-workflow
operations.

## Planning outputs

Project stages declare outputs through `TaskPlan.outputs`; they do not implement
SSH or target-filesystem checks. See
[Workflow loading and planning](workflow-and-planning.md#stage-planning) for the
planning-helper and `ArtifactDeclaration` contracts.

## Project-relative paths

`_base_` paths remain relative to the YAML file that declares them and must
stay inside the owning project. Cyclic inheritance is rejected.

Project-owned resources use `project://`:

```yaml
prompt: project://prompts/judge.yaml
config: project://configs/domain.yaml
```

Use project URIs only for assets intended to be versioned and packaged with the
project. Large or sensitive datasets, checkpoints, and generated outputs remain
deployment-owned and use absolute worker-visible paths supplied by cluster
mounts.

NVFlow:

1. Validates the path and symlinks against the launcher project root.
2. Preserves absolute paths, URLs, model paths, and output paths.
3. Renders project URIs against the packaged worker root when building stage
   execution configuration.
4. Materializes nested project URIs in packaged UTF-8 resources, so a YAML
   loaded later by a worker does not receive an unresolved URI.

These URIs are resource values, not workflow locators. CLI `--config` and the
Python workflow-loading APIs receive launcher filesystem paths.

## Dependencies

NVFlow keeps three explicit dependency boundaries:

1. A copied external project's `.venv` and `uv.lock` contain its package,
   lightweight NVFlow CLI/bootstrap, and minimal structural dependencies.
   Its sole project-level UV configuration is a tested
   `[tool.uv.sources].nvflow` Git revision, allowing `uv sync` without a package
   registry or copied NVFlow resolver policy.
2. Every selected project—maintained, external, or a submodule—gets a full
   isolated execution environment in the NVFlow user cache. It contains the
   exact active NVFlow `execution` extra, centrally pinned NeMo stack, project
   launcher dependencies, and project source.
3. Worker-only dependencies remain in selected worker images. Neither local
   environment is copied into those images or remote jobs.

Project `pyproject.toml` files contain package metadata, a semantic NVFlow
requirement, static `project.dependencies`, and optionally the project-owned
NVFlow Git source pin above; dynamically generated dependency metadata and
copied platform policy are unsupported. For the full cache, the launcher reproduces its exact
active checkout, wheel, VCS commit, or indexed version together with centralized
overrides and constraints. It records source, policy, and resolved package
details in a cache-owned synthetic `pyproject.toml` and `uv.lock` without
invoking the project's build backend. A cache-owned path file exposes
`python_source_root` directly.

First semantic use may require network access and is reported on stderr.
`nvflow sync-project` optionally prewarms the full cache, `--check` verifies it
without mutation, and `--upgrade` explicitly refreshes its lock. The Python API
remains in-process, so callers that need project stage code must already run
inside the prepared full environment.
Maintained projects declare worker-only Python packages in a `worker` optional
extra. Image-family CI verifies those imports; the launcher does not install
the extra and no image is required to carry unrelated projects' worker stacks.

## Deployment configuration

Project owners point `NVFLOW_CONFIG_DIR` at a team-owned directory containing
the cluster targets they can use. The directory may be shared by multiple
projects and does not need to live in either the project or NVFlow repository.
Keep credentials and site-specific secrets out of copyable project content.

Cluster targets reference the worker images and mounts available at that site.
Those images must be compatible with the active NVFlow and NeMo versions.
Compatible projects may share staged images; a project that needs extra worker
dependencies owns a compatible derived image.

## Packaging

Remote execution requires a clean committed Git subtree. NVFlow packages:

- Verified NeMo-Skills source from the installed distribution's `RECORD`
- Verified platform source from a clean editable Git installation or an
  installed wheel's `RECORD`
- The selected project subtree
- No unrelated file from either repository

The packaged project receives a stable project-specific worker root and Python
source path. Dirty, untracked, non-Git, nested-manifest, escaping, or colliding
sources fail before submission. A submodule is valid when its clean HEAD
matches the parent gitlink.

Repository projects and packaged worker snapshots consistently use
`projects/<project>`; the worker root is `$NVFLOW_CODE_ROOT/projects/<project>`,
which is `/nemo_run/code` on Slurm and the Ray Job's working directory on Ray.
Recipe DAGs remain declarations inside `project.yaml` and do not create
filesystem roots or environments.
CI checkouts that use project submodules must fetch them recursively (for
example, `git clone --recurse-submodules`); an absent checkout or advanced
submodule HEAD fails project resolution or provenance validation.

Planning remains available for dirty or non-Git development trees but reports
them as not submittable. A plan and its submission receipt record the project
commit, source subpath, packaged destination, and deterministic materialized
content digest. Project provenance participates in the plan digest.
Submission verifies those values again and fails if the project revision changed
after planning. A process that already loaded one revision of a project identity
must be restarted before loading another revision from the same path.

## Compatibility

Public project- and recipe-facing surfaces include:

- `nvflow` workflow behavior
- `ProjectManifest`, `ProjectContext`, and `ProjectResolver`
- `RecipeDefinition`, `RecipeSelector`, and `RecipeGraphPlan`
- `BaseStage` and workflow-local `uses`
- `load_workflow`, planning contracts, and common planning helpers
- Curated NeMo integration modules listed in the
  [NeMo ecosystem integration guide](integrations/nemo-ecosystem.md)
- Curated shared-stage bindings and project-local class coordinates
- `nvflow.stages.sdg.document_grounded` shared stage authoring APIs

Breaking these surfaces requires a compatibility path or an NVFlow
major-version change.

## Related documentation

- [Create an NVFlow project](../getting-started/creating-a-project.md)
- [Workflow loading and planning](workflow-and-planning.md)
- [Architecture](architecture.md)
