# NVFlow

NVFlow is the workflow orchestration platform for NeMo-based data generation,
training, and evaluation. Projects define workflows from their own stages and
from NVFlow's shared stages, such as the NeMo Gym and document-grounded
synthetic data generation (SDG) suites. NVFlow validates, plans, packages, and
runs those workflows through its direct Slurm engine or on operator-supplied
Ray Jobs clusters.

Domain teams package their application logic as projects, optionally define
scenario DAGs as recipes, and reuse complete workflow definitions across them.
Projects retain ownership of their data, models, prompts, stages, dependencies,
documentation, and quality.

## How NVFlow fits together

- **Project** — A self-contained, distributable application bundle.
- **Recipe** — An optional named project-scoped DAG of reusable workflows.
- **Workflow** — A runnable YAML definition that selects, orders, and configures
  stages for one objective.
- **Stage** — A reusable Python unit of work, such as preparing data, training a
  model, or evaluating checkpoints. A stage is either owned by a project or
  provided by NVFlow as a shared stage.
- **Task** — A scheduler-level command or service contributed by a stage.

```text
Project
├── Workflow
│   └── Stage
│       └── Task
└── Recipe (optional)
    └── Workflow references and dependencies
```

“Step” is an informal documentation or output label. The formal NVFlow
execution unit is a stage selected through `pipeline_stages`. See
[Core concepts](docs/getting-started/concepts.md) for configuration inheritance,
planning, targets, and receipts.

## Why NVFlow

- Compose ordered, dependency-aware stages with declarative YAML.
- Keep application code and dependencies in self-contained projects.
- Validate configuration and inspect the launch plan before submitting work.
- Use the same workflow definition through the CLI or Python API.

Platform code, including the shared stages under `nvflow/stages/`, lives under
`nvflow/`. In-repo projects—including the copyable starter—live under
`projects/`. See
[Core concepts](docs/getting-started/concepts.md) for ownership boundaries.

## Get started

### 1. Install the local CLI

Prerequisites:

- Git
- [uv](https://docs.astral.sh/uv/)
- Python 3.13.14 (NVFlow supports `>=3.13.14,<3.14`)
- A C++ compiler: gcc and g++ on Linux, or the Xcode Command Line Tools on
  macOS. A project's first workflow command builds its environment, which
  compiles `editdistance`, a NeMo-Skills dependency that publishes no Python
  3.13 wheels.
- Linux or macOS

On macOS, Apple silicon is required. Intel macOS is not supported because
compatible PyTorch wheels are unavailable.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/NVIDIA/nvflow.git
cd nvflow
uv sync
```

> **Can't clone or install where you run `nvflow`?** Use the `nvflow-client`
> container instead. It bundles the CLI and every project's environment, so
> that host needs only the image, an SSH key, and a cluster config. Build it
> once on a connected host with `scripts/build_images.sh nvflow-client`; see
> [Remote and air-gapped launch](docs/operations/remote-launch.md).

The checkout installs the lightweight NVFlow bootstrap. Each selected project
gets a full isolated cached environment containing the execution stack and that
project's launcher dependencies; project and workflow discovery do not create it.

Repository contributors should also install the development tools and commit
hooks:

```bash
uv sync --all-extras --group ci
uv run pre-commit install
```

`--all-extras` installs the `ray`, `execution`, and `dev` extras: Ray Jobs
support, the execution stack, and the contributor tools. The `ci` group adds
project-test dependencies used by the full suite. See
[CONTRIBUTING.md](CONTRIBUTING.md) for contribution requirements.

### 2. Configure execution — required before submitting

> Before using `nvflow plan` or `nvflow run`, choose an explicit target with
> `--cluster`. Complete [INSTALL.md](INSTALL.md), then set up a target with
> [Slurm execution](docs/slurm/README.md) or
> [Ray execution](docs/ray/README.md), and return here. Either target needs:
>
> - the worker images: `.sqsh` files on Slurm, or the clusters' images on Ray;
> - the models required by the selected workflow; and
> - writable data and output mounts.
>
> Both target types can launch from a workstation through an `ssh_tunnel`
> block; see [Remote and air-gapped launch](docs/operations/remote-launch.md).

### 3. Inspect and run the starter

For detailed prerequisites and expected output, follow
[Run your first workflow](docs/getting-started/first-workflow.md).

The starter contains the smallest complete workflow:

```bash
uv run nvflow list-projects
uv run nvflow list-workflows -p starter
uv run nvflow list-stages \
  --workflow starter:generate
```

The minimal starter intentionally defines no recipe. Recipes remain optional;
projects can expose direct workflows, selector-keyed recipe DAGs, or both.

`list-workflows` prints selectors accepted by every command that loads a workflow.
`list-stages` shows the active stages for one selected workflow. The first
`validate`, `plan`, or `run` may pause and require network access while NVFlow
prepares the full project execution environment; later commands reuse it.

Use `uv run nvflow list-stage-types` to discover NVFlow's shared stages, including
NeMo Gym rollouts, NeMo-RL SFT and GRPO training, and a generic train/validation
split. See [Discover shared stage types](docs/platform/cli-and-python-api.md#discover-shared-stage-types)
for namespace filtering, detailed inspection, and workflow `uses` examples, and
[Train with NVFlow's shared stages](docs/guides/training-stages.md) for training.

Inspect its configuration locally:

```bash
uv run nvflow validate \
  --workflow starter:generate

uv run nvflow show-config \
  --workflow starter:generate
```

These commands do not require scheduler access and do not submit work.
Learn what they check and see their options under
[Validate](docs/platform/cli-and-python-api.md#validate) and
[Inspect configuration](docs/platform/cli-and-python-api.md#inspect-configuration).

After completing the required execution setup, inspect the target-resolved plan
and submit, passing your Slurm or Ray target as `--cluster`:

```bash
uv run nvflow plan \
  --workflow starter:generate \
  --cluster my_cluster

uv run nvflow run --all \
  --workflow starter:generate \
  --cluster my_cluster
```

On Ray, `run` returns once its controller is submitted; follow the run with
[Monitor and verify](docs/ray/README.md#monitor-and-verify).

Static `plan` does not inspect workflow storage; `--live` adds read-only
filesystem discovery, through the target on Slurm and on the launcher on Ray.
Neither mode submits work. See
[Plan](docs/platform/cli-and-python-api.md#plan) for details.

Use `--config projects/starter/workflows/generate.yaml` instead of `--workflow`
when you need to select an arbitrary YAML path.
See
[Run the configured workflow](docs/platform/cli-and-python-api.md#run-the-configured-workflow)
for target, environment, external-project, and output-format options.

## Choose a project

- **Starter project:** [overview](projects/starter/README.md) and
  [first-workflow guide](docs/getting-started/first-workflow.md)
- **Finance project:** [quick start](projects/finance/docs/quick-start.md) and
  [workflow guides](projects/finance/docs/workflows/) for the validated
  quick-start path
- **HopChain project:** [quick start](projects/hopchain/docs/quick-start.md)
  and [project overview](projects/hopchain/README.md)

Use `nvflow list-recipes` to discover connected scenarios across projects, or
add `--project <name>` to narrow the list. Use
`nvflow list-workflows --project <name>` to discover reusable single-workflow definitions.

## Create an NVFlow project

Start from the project starter:

```bash
cp -R projects/starter ~/my-project
cd ~/my-project
```

Then follow
[Create an NVFlow project](docs/getting-started/creating-a-project.md)
and the [project and recipe contract](docs/platform/recipe-contract.md).

In the copied project, run `uv sync` once to install its lightweight `nvflow`
bootstrap, then use `uv run nvflow ...`. The first semantic command may pause and
require network access while preparing the full cache-owned execution
environment; `uv run nvflow sync-project` is only optional prewarming for CI or
offline use. The project does not copy
NVFlow's resolver policy. Its small `[tool.uv.sources].nvflow` entry pins only
the tested NVFlow Git revision used to bootstrap without a package registry.

## Documentation

Use the [documentation hub](docs/README.md) to navigate by role:

- [Core concepts](docs/getting-started/concepts.md)
- [Platform architecture](docs/platform/architecture.md)
- [Project and recipe contract](docs/platform/recipe-contract.md)
- [Workflow loading and planning](docs/platform/workflow-and-planning.md)
- [CLI and Python API](docs/platform/cli-and-python-api.md)
- [Cluster configuration](docs/operations/cluster-configuration.md)
- [Slurm execution](docs/slurm/README.md)
- [Ray execution](docs/ray/README.md)
- [Remote launch](docs/operations/remote-launch.md)

## Repository layout

```text
nvflow/              # Platform source and public contracts
projects/            # Maintained projects and copyable starter
docs/                # Platform, operator, contributor, and maintainer docs
cluster_configs/     # Cluster templates
dockerfiles/         # Worker and optional launcher images
scripts/             # Image build, container staging, and CI scripts
tests/               # Platform contract and integration tests
```

## Contribute

- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)
- [Code of conduct](CODE_OF_CONDUCT.md)

Project contributions should include project-owned tests and documentation.
Platform changes must keep concrete project dependencies out of the platform
package.
