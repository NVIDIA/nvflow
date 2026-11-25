---
rfc: "0002"
title: "NVFlow Defines Recipe Contracts; Recipe Authors Own Recipe Content"
status: accepted
authors: ["@NVIDIA/nvflow-dev"]
created: 2026-08-16
discussion: null
implementation: implemented
supersedes: []
superseded-by: []
---

# RFC-0002: NVFlow Defines Recipe Contracts; Recipe Authors Own Recipe Content

## Summary

NVFlow currently stores framework code and concrete recipes inside the same Python package. This makes it unclear whether recipe content is part of the platform or an application built with the platform.

This RFC establishes two owners:

- The **platform owner** maintains NVFlow's orchestration engine, recipe contract, shared capabilities, and extension APIs.
- The **recipe owner** maintains a recipe's workflows, prompts, configs, assets, domain Python, and tests.

NVFlow may distribute reference recipes as working examples. After installing NVFlow, a user can run those recipes, copy and modify them, or create a new recipe in their own repository. Sharing a recipe back to NVFlow is optional.

## Decision requested

1. NVFlow framework code and recipe source MUST be separate ownership domains.
2. NVFlow MUST define a stable recipe manifest, resolver, root-relative path behavior, and Python registration contract.
3. Recipe authors MUST be able to keep complete recipe bundles outside the NVFlow repository.
4. Running or developing an external recipe MUST NOT require an upstream NVFlow pull request.
5. A recipe bundle MAY contain recipe-specific Python; recipes are not required to be YAML-only.
6. Existing `BaseStage` and `StageRegistry` behavior MUST remain compatible during migration.
7. Shared NVFlow packages MUST NOT import concrete recipes.
8. New shared functionality MUST NOT be added to the catch-all `nvflow.lib` namespace.
9. Existing `nvflow/recipes` paths MUST remain available until migrated recipes pass parity checks.
10. NVFlow-distributed recipe source SHOULD move to top-level `recipes/` after resolver, packaging, and path parity are proven.
11. NVFlow-distributed reference recipes MUST initially ship as data/code in the same NVFlow distribution rather than introducing a companion package.

This RFC does not introduce a hosted recipe registry, automatic remote-code download, or a global stage-type registry.

## Context

### Behavior before this RFC (historical)

This section records NVFlow as it was when this RFC was proposed. The [project contract](../platform/recipe-contract.md) describes current behavior.

Users can already point `nflow` at arbitrary workflow YAML. This works when the config continues to use an existing recipe identity such as `finance` and its built-in registered stages.

The current implementation still assumes:

- `recipe: finance` maps to `nvflow.recipes.finance`.
- Recipe metadata lives at `nvflow/recipes/<name>/recipe.yaml`.
- `_base_` paths are relative to the current YAML file.
- Prompts and configs often contain repository- or package-relative paths.
- New stages are imported by the built-in recipe package.
- Cluster code snapshots contain the expected `nvflow/recipes` paths.

Therefore a user can modify values outside the repository, but a fully independent recipe—with its own identity, assets, and stage code—still requires modifying or forking NVFlow.

### Desired behavior

A researcher should be able to:

1. Install NVFlow.
2. Run a distributed reference recipe unchanged.
3. Copy and modify a recipe in another repository.
4. Create a new recipe bundle with its own YAML, assets, and optional Python stages.
5. Run it using NVFlow's public contracts.
6. Decide independently whether to contribute it back.

## Goals

- Make the platform/recipe ownership boundary explicit.
- Let recipe authors work independently of the NVFlow repository.
- Keep reference recipes available as working examples.
- Make recipe paths portable between launcher and compute nodes.
- Support both configuration-only recipes and recipes with domain Python.
- Preserve existing stage identities and workflow behavior during migration.
- Keep reusable framework code under `nvflow`.

## Non-goals

- Making recipes YAML-only.
- Guaranteeing the behavior or quality of externally owned recipes.
- Loading untrusted code automatically.
- Building a remote recipe marketplace.
- Defining recipe-facing training or environment request interfaces.
- Inheriting prompts, configs, or assets from another recipe bundle.
- Supporting dirty or non-Git external recipe source in the initial implementation.
- Supporting recipe identity or stage shadowing.
- Replacing `StageRegistry` with a global reusable-stage registry.
- Moving all existing recipes in one change.

## Ownership contract

### Platform owner

The platform owner maintains:

```text
nvflow/
  core/
  recipe/          # RecipeManifest, RecipeContext, RecipeResolver
  execution/
  stages/          # Reusable domain-neutral stages
  lib/             # Legacy compatibility only
```

The platform owner is responsible for:

- CLI and workflow execution semantics.
- Recipe discovery and path resolution.
- `BaseStage` and `StageRegistry`.
- Shared stages and adapters.
- Compatibility and deprecation of public extension contracts.

### Recipe owner

The recipe owner maintains one self-contained application bundle:

```text
recipes/my-recipe/
  recipe.yaml
  workflows/
  prompts/
  configs/
  assets/
  stages/           # Optional recipe-specific Python
  utils/            # Optional recipe-specific Python
  tests/
```

The recipe owner is responsible for:

- Workflow composition.
- Domain-specific transformations and validation.
- Prompts, configs, and assets.
- Recipe-specific Python dependencies.
- Recipe tests and compatibility with the declared NVFlow version.

NVFlow supports the framework contracts used by a recipe, not the recipe's domain behavior or outputs.

### Operator

The operator owns cluster config, provider images, mounts, partitions/accounts, and secrets. Moving the same recipe between supported clusters should not require changes to recipe Python.

## Recipe contract

### Recipe manifest

Illustrative manifest:

```yaml
manifest_version: 1
recipe: my-recipe
requires_nvflow: "..."
python_source_root: .                    # optional; defaults to bundle root
entrypoint: my_recipe.register:register  # optional
stage_namespace: nvflow:finance          # optional
```

The version-one manifest recognizes the existing `recipe`, `description`, and `workflow_order` fields plus:

- `manifest_version`, which selects the manifest schema.
- `requires_nvflow`, which constrains compatible NVFlow versions.
- `python_source_root`, a bundle-relative Python source directory that defaults to `.`.
- `entrypoint`, an optional recipe-registration callable.
- `stage_namespace`, an optional qualified NVFlow-distributed stage namespace.

Unsupported manifest versions and unknown top-level fields fail validation with an actionable error. Existing `recipe.yaml` files without `manifest_version` remain valid during migration: NVFlow maps `recipe`, `description`, and `workflow_order` into the version-one model and applies defaults for new fields without rewriting the source file.

### Recipe context

`RecipeResolver` returns a `RecipeContext` containing:

- Bundle identity.
- Resolved stage-registration namespace.
- Launcher recipe root and packaged recipe root.
- Manifest location.
- Workflow/config search paths.
- Optional registration entrypoint.
- Bundle-relative Python source root.
- Source provenance and launcher-to-worker path mapping.

For a standalone recipe, bundle identity and stage namespace are the same. For a configuration-only derivative using `stage_namespace: nvflow:finance`, the bundle identity remains `my-finance-experiment` while stage lookup uses the installed `finance` reference recipe's namespace.

`WorkflowRunner` uses the stage namespace only for `StageRegistry` lookup. Bundle identity and roots are used for provenance, resource resolution, and diagnostics. Built-in recipes retain their existing experiment and dependency names. For an external recipe, the resolved bundle identity prefixes those names, and dependency references use the same rule.

### Recipe-relative paths

`_base_` remains file-relative and unchanged.

Other recipe assets use an explicit recipe URI:

```yaml
prompt: recipe://prompts/judge.yaml
config: recipe://configs/domain.yaml
```

Only `recipe://` values resolve relative to the appropriate root in `RecipeContext`. Absolute runtime paths such as `/workspace/...` and `/hf_models/...`, URLs, and cluster paths are never rewritten.

Launcher-side inspection and validation resolve `recipe://` against the launcher root. Worker-command rendering resolves it against the packaged root. The exact Python representation is an implementation detail.

Resolution canonicalizes the requested path and all symlinks. A path that escapes the corresponding recipe root fails before import or submission.

### Recipe registration

Recipe-specific Python registers stages through the existing `StageRegistry`.

The entrypoint receives the resolved recipe identity so copied recipes do not need hard-coded decorators for the original recipe name:

```python
def register(recipe_context) -> None:
    ...
```

The exact callable signature is illustrative. Existing decorator-based registrations remain supported through compatibility adapters.

External recipe identities must be unique and must not shadow NVFlow-distributed or already loaded recipe identities. Registration collisions are hard errors. This RFC does not add stage replacement, reload, or scoped-registry behavior.

### Configuration-only recipes

A recipe without Python may use a qualified NVFlow-distributed stage namespace:

```yaml
recipe: my-finance-experiment
stage_namespace: nvflow:finance
```

`stage_namespace` affects only `StageRegistry` lookup. `_base_` remains file-relative, and `recipe://` resolves only within the external bundle; prompts, configs, and assets are not inherited from the reference recipe.

The `nvflow:` qualifier bypasses user search paths and resolves directly to a reference recipe distributed with the installed NVFlow version. `requires_nvflow` constrains that version, and run metadata records the resolved reference-recipe content digest. Arbitrary external stage namespaces are deferred.

This RFC does not require a global `uses:` mechanism. If configuration-only recipes eventually need arbitrary composition across unrelated stage libraries, that capability requires a separate decision.

### Recipe resolution

The resolver considers:

1. An explicitly supplied recipe directory.
2. The nearest `recipe.yaml` found by walking upward from an explicit workflow/config path, bounded by its Git root or configured resolver root.
3. User-configured recipe search paths, in declared order.
4. NVFlow-distributed reference recipes.
5. Legacy `nvflow.recipes` compatibility lookup.

The nearest manifest establishes the bundle root. A bundle may not contain another `recipe.yaml`: finding a second manifest on the ancestor path before the traversal boundary, or anywhere below the selected root, is an error. Sibling bundles in a larger repository remain valid.

An explicit directory wins bundle resolution; it does not override registered stage code. Duplicate or shadowing recipe identities fail with an actionable error, and the resolver never silently selects one. Qualified `nvflow:` stage namespaces resolve directly from the installed NVFlow distribution. No network fetch occurs during resolution.

### Trust boundary

Recipe Python is trusted application code.

- NVFlow does not download or install it automatically.
- The recipe owner/operator installs its dependencies.
- The launcher includes the bundle/code in the job snapshot or mounts.
- NVFlow-distributed recipe code is trusted as part of the installed distribution.
- Selecting an external bundle for an operation is the operator's trust action.
- Secrets remain operator configuration, not recipe content.

Manifest inspection, discovery, compatibility checks, path checks, and structural validation do not import an external entrypoint. Commands that require registered stages, including semantic stage validation and execution, may import it only after those checks and explicit bundle selection. This RFC does not require a trust store or a separate approval workflow.

### Public extension compatibility

Initial supported recipe-extension surfaces are:

- `nflow` CLI and workflow-config behavior.
- `RecipeManifest`, `RecipeContext`, and `RecipeResolver`.
- `BaseStage`.
- `StageRegistry`.
- The recipe registration entrypoint.

`requires_nvflow` is the recipe's compatibility declaration. Within an NVFlow major version, breaking changes to these surfaces require a compatibility shim or documented deprecation path. Incompatible removal requires a major-version change or a superseding RFC.

`nvflow.lib`, concrete adapter internals, recipe utilities from another recipe, and private execution helpers are not supported extension APIs.

This RFC does not define recipe-facing training or environment request interfaces. Any platform capability offered to recipes must be exposed through a separately documented and versioned public extension surface.

## Source arrangement

Target repository layout:

```text
nvflow/                  # Platform source
  core/
  recipe/
  execution/
  stages/
  lib/

recipes/                 # NVFlow-distributed reference recipe applications
  example/
  finance/
  multimodal/
```

Users may use the same bundle structure in any repository:

```text
research-project/
  recipes/
    experiment-a/
      recipe.yaml
      workflows/
      prompts/
      configs/
      stages/
      tests/
```

The source location does not make recipe content part of the framework API. NVFlow-distributed recipes are examples/living proofs with whatever tests they declare; they are not the platform's main contribution.

Every NVFlow-distributed reference recipe must pass manifest/discovery and workflow-config validation in regular CI. A recipe may declare additional CPU, GPU, or multi-node canaries; only those declared paths are guaranteed. Distribution does not imply blanket production or output-quality support.

The installed NVFlow distribution includes NVFlow-distributed reference recipes as explicit package data/code. The build configuration must list non-Python assets rather than relying on implicit VCS inclusion. A companion recipe package is deferred until an independent release cadence is demonstrably needed.

### Job code snapshot

The selected external recipe bundle and its extension code must be available on compute nodes.

The initial external-source contract is deliberately narrow:

- The bundle must be inside a clean Git worktree.
- Packaging uses an exact commit and only the selected bundle subtree.
- No working-tree glob, including `*.jsonl`, is added implicitly.
- The bundle is placed under a destination derived from its validated identity rather than overlaid at the archive root.
- The run records the Git commit, source subpath, packaged destination, and bundle content digest.

Staged, modified, or untracked files inside the selected subtree fail preflight. Dirty and non-Git development modes are deferred. Generated datasets use explicit data, artifact, or mount configuration rather than code packaging.

`RecipeResolver` returns `RecipeContext` with the source metadata needed by submission. It does not register repositories with NeMo-Skills, mutate process-global packaging state, change `sys.path`, or import the recipe entrypoint. The trusted launcher loader applies the launcher Python source root when registration is required; the submission packaging step maps the same relative root into the worker snapshot.

The entrypoint and its import-time dependencies run on the launcher. A worker needs recipe or NVFlow source only when its rendered command imports that source, and its runtime dependencies must already exist in the selected worker environment. NVFlow does not install recipe dependencies at runtime.

The pinned NeMo-Run packagers may be used as internal implementation primitives, but their whole-repository and root-overlay behavior is not the public recipe transport contract.

## Shared-code boundary

Code belongs under shared `nvflow` packages only when it has a domain-neutral contract or an approved cross-recipe/platform role.

Rules:

- Shared code never imports concrete recipes.
- Recipes may import public shared capabilities.
- Platform launch and capability-integration code remain platform-owned rather than recipe-owned.
- New functionality is not added to `nvflow.lib`.
- Recipe-specific code remains with its recipe until a real reusable contract exists.

## Alternatives considered

### Keep all recipes under `nvflow`

Easy to package, but conflates platform and application ownership and forces complete external recipes to fork platform source.

Rejected as the target; retained for compatibility.

### Put every recipe in a separate repository

Maximizes separation but complicates installation, examples, and end-to-end testing.

Rejected as a requirement. Users may use separate repositories; NVFlow may keep reference recipes in the main repository.

### Make recipes YAML-only

Prevents domain Python but pushes behavior into shell commands or premature shared abstractions.

Rejected.

### Add a global reusable-stage registry

Would enable arbitrary YAML composition but expands the public API and discovery model.

Deferred. A qualified NVFlow stage namespace plus recipe-owned registration is sufficient for this RFC's ownership goal.

## Consequences

### Benefits

- Platform and recipe owners can evolve independently.
- Researchers can create and run recipes without upstream pull requests.
- Reference recipes remain available as concrete examples.
- Domain code does not accumulate in platform packages.
- Recipe-relative assets become portable.

### Costs

- Recipe resolution and root-relative paths become supported contracts.
- External recipe Python introduces a trust boundary.
- Reusing a distributed stage namespace requires version and drift diagnostics.
- First-party recipe packaging requires build work.
- Existing hard-coded package paths must be migrated.

## Compatibility and migration

1. Inventory repository-, package-, and CWD-relative recipe paths.
2. Add version-one manifest validation, legacy translation, and recipe-local validation tests.
3. Add `RecipeContext` and `RecipeResolver` while retaining legacy loading and paths.
4. Support a configuration-only external recipe using `stage_namespace: nvflow:finance`.
5. Support an explicitly selected external recipe with its own registration entrypoint.
6. Package a clean committed external bundle subtree at a bundle-specific destination.
7. Prefix external execution and dependency names with bundle identity.
8. Move `example` to top-level `recipes/` as the first source-layout canary.
9. Move finance and multimodal only after parity.
10. Retain `nvflow/recipes` compatibility until the migration window ends.

Built-in recipes preserve stage classes, config resolution, experiment names, dependency names, and output paths. External recipe names use the bundle prefix defined by this RFC; that new behavior does not alter built-in compatibility.

## Rollout and rollback

Go/no-go gates:

- Existing recipes resolve identically through `RecipeResolver`.
- Existing manifests map to the version-one model without source changes.
- External configuration-only and Python recipes run from outside the NVFlow repository; inspection does not import Python, and registration occurs only after explicit bundle selection.
- Two external bundles with matching workflow, stage, and run names produce distinct execution and dependency names.
- Assets resolve against the corresponding launcher and packaged roots, and remote snapshots contain only the selected clean Git bundle subtree.
- Missing, ambiguous, incompatible, or untrusted recipes fail before submission.

Rollback restores legacy package lookup as the primary resolver. Recipe files move only after both paths pass parity tests.

## Operational and security impact

- External recipe entrypoints execute only after the operator explicitly selects the bundle.
- Recipe resolution never fetches code from unapproved remote sources.
- Bundle roots, source commit/subpath, packaged destination, digest, and entrypoint are recorded in run metadata.
- Canonical path and symlink resolution prevents escape when reading, importing, or packaging files.
- Recipe resolution does not mutate process-global packaging or import state.
- Logs identify recipe identity and revision without exposing recipe content or secrets.

## Open and deferred questions

Review questions:

1. What compatibility duration should apply to recipe manifests and registration entrypoints within a major NVFlow version?
2. Is moving NVFlow-distributed recipe source to top-level `recipes/` part of the initial implementation or a later parity-preserving cleanup?

Deferred:

- Dirty and non-Git external recipe packaging.
- Arbitrary external stage namespaces and cross-bundle asset/config inheritance.
- Remote recipe registries and signed recipe packages.
- Third-party discovery through Python entry points.
- Global stage IDs and `uses:`.
- Hosted community recipe catalogs.

## Required validation canaries

### 1. Installed reference recipe

- Build and install an NVFlow wheel in a clean environment.
- Resolve and validate the distributed `example` recipe.
- Confirm its existing manifest maps to the version-one model without source changes.
- Confirm YAML, prompts, configs, and recipe metadata are present.

This validates explicit package-data inclusion in the NVFlow distribution.

### 2. External configuration-only recipe

- Copy the example/finance workflow into a separate repository.
- Give the bundle a new identity and `stage_namespace: nvflow:finance`.
- Copy its required prompts, configs, and assets into the external bundle.
- Resolve its `recipe://` assets against both launcher and packaged roots.
- Confirm stage lookup uses the distributed namespace while execution and dependency names use the new bundle identity.

This validates `RecipeContext`, qualified stage lookup, path handling, and external naming without Python extensions or cross-bundle asset inheritance.

### 3. External recipe with Python

- Add a recipe-owned registration entrypoint and a minimal worker module under a recipe subdirectory in a clean Git repository.
- Confirm structural inspection does not import the entrypoint.
- Explicitly select the bundle, then validate and register the recipe on the launcher.
- Submit a minimal job locally and through the remote/Slurm packaging path.
- Verify the worker imports the recipe-owned worker module without importing the registration entrypoint.
- Verify only the selected subtree appears at its bundle-specific destination and unrelated repository files are absent.
- Verify staged, modified, or untracked bundle files fail preflight.
- Verify run metadata records the commit, source subpath, packaged destination, and content digest.

This validates the launcher/worker boundary, explicit trust, scoped clean-Git packaging, and recipe Python-path handling.

## Resolution

Accepted on 2026-08-28 after maintainer review.

Implementation changed four decisions. The [project contract](../platform/recipe-contract.md) is now authoritative:

- Recipe bundles are called projects: `projects/<name>/project.yaml`, `project://` paths, and `ProjectManifest`, `ProjectContext`, and `ProjectResolver`. "Recipe" now names a DAG of workflows inside a project.
- Each active stage declares its implementation with `uses:`, either a curated shared alias or a `project:<module>:<Class>` coordinate. `StageRegistry`, registration entrypoints, and `stage_namespace` were removed (decisions 2 and 6).
- The NVFlow wheel contains platform code only; curated projects ship in the source repository (decision 11).
- Ordered external search paths may intentionally override a curated project identity.

The contract also settles the compatibility-duration question: breaking changes to its public surfaces need a compatibility path or a major NVFlow version.

## Implementation success criteria

1. Existing recipes run unchanged through `RecipeResolver`.
2. A user runs a copied/modified recipe from outside the NVFlow repository.
3. A user creates a uniquely named recipe without modifying NVFlow, and its execution names do not collide with another bundle.
4. A configuration-only recipe can use a qualified NVFlow-distributed stage namespace while owning its configs and assets.
5. A selected external recipe can register its own stages through an explicit entrypoint; inspection alone does not import it.
6. Recipe-relative assets resolve on launcher and compute nodes.
7. External recipe execution requires no upstream NVFlow pull request.
8. Shared packages do not import recipe packages.
9. Existing stage identities, names, dependencies, and output behavior remain compatible during migration.
10. NVFlow-distributed reference recipes and assets are present in the installed NVFlow distribution and pass manifest/config validation in CI.
11. External recipe transport includes only the selected clean Git subtree, records its source metadata, and does not accumulate process-global packaging state.

## Appendix: Source evidence (historical)

These are the sources reviewed when this RFC was proposed.

- NVFlow lock: NeMo-Run `0.9.0rc0.dev0` at commit `17ae86b64d7f75653351664f5d8c9e466faede00`.
- [NeMo-Run HybridPackager documentation](https://docs.nvidia.com/nemo/run/latest/guides/execution.html)
- [HybridPackager at the pinned NeMo-Run commit](https://github.com/NVIDIA-NeMo/Run/blob/17ae86b64d7f75653351664f5d8c9e466faede00/nemo_run/core/packaging/hybrid.py)
- [NeMo-Skills external repository packaging](https://github.com/NVIDIA-NeMo/Skills/blob/e06c9b900177be3f60d6a3f99135bb5de9af9bed/nemo_skills/pipeline/utils/packager.py) at `e06c9b90`, the commit NVFlow pinned when this RFC was proposed; NVFlow now pins `bcf059af`
- [NeMo-Skills code packaging behavior](https://nvidia-nemo.github.io/Skills/basics/code-packaging/)

These pinned implementations provide useful packaging primitives, but their whole-Git-root, implicit-JSONL, process-global registration, and root-overlay behavior is not adopted as the external recipe contract.
