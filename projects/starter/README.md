# NVFlow project starter

This is both the minimal in-repo example and a project that can be copied into
another repository. Follow
[Create an NVFlow project](../../docs/getting-started/creating-a-project.md).

## What it demonstrates

- `project.yaml` gives the project its identity and optional recipe DAGs.
- `pyproject.toml` declares only project metadata and semantic dependencies.
- `workflows/generate.yaml` defines one runnable workflow and its project stage class.
- `project://` references keep the bundled input and prompt portable.
- `my_project/` provides one exact `execution()` contract used by both engines.
- The plan declares its output without implementing cluster filesystem logic.

## What it runs

The `generate_answer` stage reads `data/input.jsonl`, applies
`prompts/generate_answer.yaml`, and declares NeMo-Skills generation with the
configured Qwen model. The bundled input is intentionally a tiny runnable
fixture; real datasets should remain outside the project repository and be
referenced through worker-visible cluster mounts.

The declared output directory is:

```text
/workspace/outputs/starter/generate-answer/
```

Cluster configuration, worker images, model availability, mounts, and
credentials remain deployment-owned and are not bundled here.

## Repository contents

```text
project/
├── .gitignore
├── README.md
├── project.yaml
├── pyproject.toml
├── workflows/generate.yaml
├── data/input.jsonl
├── prompts/generate_answer.yaml
└── my_project/
    ├── __init__.py
    └── stage.py
```

## Use it as a starter

Before the first commit of a copied project:

1. Choose a project identity and description, then update `project.yaml`.
2. Choose the distribution name in `pyproject.toml`.
3. Select a model and unique `base_output_dir` in `workflows/generate.yaml`.
4. Replace the sample input, prompt, and stage behavior.
5. Optionally add a selector-keyed recipe when a named scenario should connect
   complete workflows.
6. Replace this README with the project's purpose, setup, outputs, and
   limitations; add project-owned tests and documentation as needed.

The Python package may remain `my_project` initially. Rename it only when the
project needs a domain-specific package name, and at the same time update
`packages` under `[tool.hatch.build.targets.wheel]` in `pyproject.toml` and the
project-class `uses` coordinates in workflow YAML. The
`[tool.uv.sources].nvflow` entry is the project-owned tested NVFlow revision;
update only its `rev` when upgrading NVFlow, without copying platform override
policy.

## Inspect the starter

Run these commands from the root of your copied project once you have committed
it and completed the setup and deployment steps in the creation guide. `run`
refuses a project with untracked files, so keep `my_cluster.yaml` outside the
project and point `NVFLOW_CONFIG_DIR` at the directory that holds it:

```bash
uv sync
uv run nvflow list-projects
uv run nvflow list-workflows --project starter
uv run nvflow list-stages --config workflows/generate.yaml
uv run nvflow validate --config workflows/generate.yaml
uv run nvflow show-config --config workflows/generate.yaml
export NVFLOW_CONFIG_DIR=<directory holding my_cluster.yaml>
uv run nvflow plan --config workflows/generate.yaml --cluster my_cluster
uv run nvflow run --all --config workflows/generate.yaml --cluster my_cluster
```

To run the starter inside the NVFlow checkout instead, use the
`--workflow starter:generate` commands in the
[root README](../../README.md#3-inspect-and-run-the-starter) from the
repository root. Running `uv sync` in `projects/starter` would leave an
untracked `uv.lock` there, which `run` refuses.

Discovery, validation, and configuration inspection are local. Static workflow
`plan` reads the selected target configuration without connecting to the
cluster. `plan --live` adds read-only filesystem inspection, through the target
on Slurm and on the launcher on Ray, and `run --all` connects and submits the
complete workflow.
The minimal starter intentionally defines no recipe. `uv sync` creates only the
lightweight project bootstrap; the first command that imports project Python
prepares the full isolated environment in the NVFlow cache.
