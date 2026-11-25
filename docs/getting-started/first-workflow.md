# Run your first workflow

This guide discovers the project starter, then validates, plans, and runs its
workflow on a direct Slurm or supplied Ray target.

## Prerequisites

- Complete [INSTALL.md](../../INSTALL.md), then set up a
  [Slurm](../slurm/README.md) or [Ray](../ray/README.md#set-up-a-ray-target)
  target. The starter needs only the `nemo-skills` and `vllm` images.
- Configure a target such as `my_cluster`; pass it explicitly to planning and
  execution commands.
- Mount `/hf_models/Qwen/Qwen3-4B` in the selected worker target.
- Provide writable `/workspace/outputs`.

The starter is documented in
[`projects/starter/README.md`](../../projects/starter/README.md).

## 1. Discover

```bash
uv run nvflow list-projects
uv run nvflow list-recipes --project starter
uv run nvflow list-workflows --project starter
uv run nvflow list-stages \
  --workflow starter:generate
```

The starter intentionally has no recipe, so `list-recipes` reports that
directly. The remaining commands show its reusable workflow and drill down to
configured stages.

## 2. Validate

```bash
uv run nvflow validate \
  --workflow starter:generate
```

Workflow validation checks project-owned resources, stage implementation bindings,
configuration, and stage dependencies without submitting work.
On first use, NVFlow prepares the starter's isolated launcher environment in
the user cache and transparently continues this command.

## 3. Inspect the effective configuration

```bash
uv run nvflow show-config \
  --workflow starter:generate
```

The default `resolved` view shows effective values. Use `--view source`,
`merged`, or `diff` when debugging composition. Environment-backed and
conventionally named credentials are redacted from display output.

## 4. Build a plan

```bash
uv run nvflow plan \
  --workflow starter:generate \
  --cluster my_cluster
```

The static plan loads `my_cluster.yaml` from `NVFLOW_CONFIG_DIR`, or from
`cluster_configs/` in the current directory when that variable is unset, and
summarizes the target, stages, tasks and resources. Add `--verbose` for each
task's command, dependencies and outputs, or `--format json` for the complete
plan. It does not inspect workflow filesystem state. In a copied project, keep
`my_cluster.yaml` outside the project and point `NVFLOW_CONFIG_DIR` at its
directory, because `run` refuses a project with untracked files.

Add `--live` to enable input/resume discovery and presence checks for explicitly
declared outputs. It submits nothing. On Slurm it reads storage through the
target, which needs valid target-filesystem runtime fields; on Ray it reads the
launcher's local filesystem, so the workload paths must be mounted there.

Resolve any uninspectable stage or target configuration error before running.

## 5. Submit

```bash
uv run nvflow run --all \
  --workflow starter:generate \
  --cluster my_cluster
```

On Slurm, `run --all` first performs live-equivalent planning; on Ray, it plans
statically and the workload scripts skip completed work on the clusters. It then
packages the platform and selected project and submits it: Slurm jobs through
NeMo-Skills and NeMo-Run, or one controller Job on Ray. Progress, warnings, and
errors are written to stderr; the command intentionally leaves stdout empty. The
Python API exposes the in-process submission receipt. On Ray, `run` returns once the
controller is submitted; follow the run with
[Monitor and verify](../ray/README.md#monitor-and-verify).

Expected output directory:

```text
/workspace/outputs/starter/generate-answer/
```

The starter contains one input row, so a successful fresh run produces one
JSONL output row.

## Run one stage

```bash
uv run nvflow run generate_answer \
  --workflow starter:generate \
  --cluster my_cluster
```

## Next steps

- [Run the finance quick start](../../projects/finance/docs/quick-start.md)
- [Create your own NVFlow project](creating-a-project.md)
- [CLI and Python API](../platform/cli-and-python-api.md)
- [Cluster configuration](../operations/cluster-configuration.md)
