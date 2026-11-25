# Slurm execution

NVFlow submits each workflow step to a Slurm cluster as a job that runs in the
project's worker images. Complete [INSTALL.md](../../INSTALL.md) first: it
installs the client and prepares the images, models, and data every target
needs.

For Ray clusters you already operate, use [Ray execution](../ray/README.md)
instead.

## Set up a Slurm target

Set up once per cluster, then [run workflows](#run-workflows) with the target as
`--cluster`.

### 1. Check cluster access

**Required:**
- **Slurm cluster** access with SSH keys (or run directly from a login node)
- **enroot** and **pyxis** - on cluster nodes (`enroot version`; pyxis adds `--container-image` to `srun --help`)

**Get your cluster info:**
- Slurm account: `sacctmgr show associations user=$USER` (look for the Account column)
- Available partitions: `sinfo`
- Slurm version: `scontrol show config | grep SLURM_VERSION` (25.x needs the enroot Ray template fix - see [Troubleshooting](#troubleshooting))
- Storage paths for data/models/containers

**✅ Done when:** `enroot version` works on a cluster node and you know your Slurm account, a partition, and your storage paths.

### 2. Stage the images

Slurm runs each job in a `.sqsh` copy of a [worker image](../../INSTALL.md#worker-images).
If the `.sqsh` files already exist, note their paths and continue. Otherwise,
convert them from a login node, in a clone of this repo there (the commands run
from the checkout). If your registry needs a login, add it to
`~/.config/enroot/.credentials` first.

```bash
cp cluster_configs/containers.yaml cluster_configs/my_containers.yaml
# In my_containers.yaml, replace <YOUR_REGISTRY> and delete the entries for
# images you did not build or cannot pull; a missing image fails the job.
mkdir -p outputs/logs
sbatch --account=<YOUR_SLURM_ACCOUNT> --partition=<CPU_PARTITION> \
  scripts/setup_containers.sh --config cluster_configs/my_containers.yaml \
  <PATH_TO_CONTAINERS> --platform amd64  # arm64 for Arm compute nodes
```

The job imports each image with enroot and writes `<role>-<tag>.sqsh` files,
skipping any that already exist. The credentials format and the path for sites
without a registry are in the
[container guide](../maintainers/containers.md#step-4-convert-to-sqsh-format).

**✅ Done when:** the `.sqsh` files exist on the cluster and you have their absolute paths for the target.

### 3. Create the target

```bash
cp cluster_configs/template-slurm.yaml cluster_configs/my_cluster.yaml
```

Edit `cluster_configs/my_cluster.yaml` and replace all `<PLACEHOLDER>` values:

1. **SSH settings** - Your cluster login node, username, SSH key path (ONLY for remote job submission from local machine)
2. **Slurm account/partition** - Run `sacctmgr show associations user=$USER` and `sinfo`
3. **Container paths** - Copy from `outputs/logs/slurm-containers-<jobid>.out` after running setup_containers.sh, and delete the lines for images you did not stage
4. **Mount points** - Map your cluster paths to container paths (at minimum `<hf_models>:/hf_models` and a writable data dir `<workspace_data>:/workspace` for outputs + caches). Project code and assets ship via the nemo-run packaged snapshot (`/nemo_run/code`), so the repo is not mounted — see [Mount Points](../operations/cluster-configuration.md#mounts).
5. **Environment variables** - Set `HF_HOME` to a path visible inside the container (see [env_vars docs](../operations/cluster-configuration.md#environment-variables)) and any API keys. Campaigns that grade with a hosted judge need its key forwarded — see [Hosted Model Endpoints](../hosted-endpoints.md). Forwarded values, bare names included, are written into each job's sbatch script under `job_dir` and printed in the step's `*_sbatch.log`, so keep both private. NeMo-Skills also forwards `HF_TOKEN`, `WANDB_API_KEY`, `NVIDIA_API_KEY`, `OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY` and `GEMINI_API_KEY` from your shell whenever they are set, even if `env_vars` does not list them

`template-slurm.yaml` ships with the offline flags pre-populated - leave them on:

```yaml
env_vars:
  # --- Offline enforcement (recommended) ---
  - HF_HUB_OFFLINE=1
  - HF_DATASETS_OFFLINE=1
  - TRANSFORMERS_OFFLINE=1
  # UV_OFFLINE: the nemo-rl and nemo-gym images set it themselves. Every image
  # ships its environments built in, so jobs install nothing at runtime.

  # Pre-baked tiktoken / openai_harmony cache; the vLLM and SGLang images
  # already set all three, and nemo-skills sets TIKTOKEN_CACHE_DIR.
  - TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache
  - TIKTOKEN_RS_CACHE_DIR=/opt/tiktoken_cache
  - TIKTOKEN_ENCODINGS_BASE=/opt/tiktoken_cache
```

For the [one-time connected-node stages](../../INSTALL.md#one-time-connected-node-stages-datasets)
`create_seed_data` and `prepare_data`, comment out the three `HF_*_OFFLINE` flags
just for that submission, then re-enable; `eval_data` clears them itself.

> **Note:** The template includes detailed comments for each section. Your personal config (`my_cluster.yaml`) is gitignored to protect secrets.
>
> For detailed documentation of all configuration fields, see the [Cluster Configuration Guide](../operations/cluster-configuration.md).

**✅ Done when:** `my_cluster.yaml` has no remaining `<PLACEHOLDER>` values,
uses the v1.2 image paths, and leaves `UV_OFFLINE` unset.

### 4. Verify the target

```bash
# 1. No placeholders left (expect no output)
grep -nE '^[^#]*<[A-Z][A-Z0-9_]*>' cluster_configs/my_cluster.yaml

# 2. Check that every image under `containers:` exists, on the cluster
ls -lh <PATH_TO_CONTAINERS>/*.sqsh

# 3. Test SSH to cluster (only if submitting from a local machine)
ssh -i <PATH_TO_SSH_KEY> <YOUR_USERNAME>@<YOUR_CLUSTER_LOGIN_NODE> "echo '✅ SSH OK'"

# 4. Plan the starter against the cluster's storage; --live submits nothing
uv run nvflow plan --live --workflow starter:generate --cluster my_cluster
```

A plan without `--live` only checks that the target file loads, so use `--live`
here.

**✅ Done when:** all four checks above pass.

## Run workflows

Return to the main flow, [Inspect and run the starter](../../README.md#3-inspect-and-run-the-starter)
or a project's quick start, and pass the Slurm target as `--cluster`, for
example `--cluster my_cluster`. List your queued and running jobs with
`uv run nvflow status --cluster my_cluster`, and add `--workflow <selector>` to
list only one workflow's jobs.

## Where settings live

| Setting | On Slurm |
|---|---|
| Worker images | `.sqsh` paths under the target's `containers` |
| Mounts | The target's `mounts`, from host path to container path |
| Environment variables and API keys | The target's `env_vars`; a bare `NAME` forwards the value from the shell that runs `nvflow` |
| Offline flags (`HF_*_OFFLINE`) | The target's `env_vars` |
| Account and partitions | The target; keep them out of recipes so the same recipe also runs on Ray |
| Logs | Each step's log directory next to its output, such as `generation-logs/` |
| Queued and running jobs | `uv run nvflow status --cluster <target>` |

## Troubleshooting

### enroot not available
```bash
# On cluster
module load enroot  # if available
# Or contact your cluster admin
```

### Image import fails with 401 Unauthorized
A 401 from your registry means its `machine` entry in `~/.config/enroot/.credentials` is wrong or expired. enroot sends that entry for every image on the registry host, so it blocks public images there too. Replace the token and resubmit the staging job.

### SSH connection failed
```bash
chmod 600 <PATH_TO_SSH_KEY>
ssh -i <PATH_TO_SSH_KEY> <YOUR_USERNAME>@<YOUR_CLUSTER_LOGIN_NODE>
```

### Container paths wrong
Use **absolute paths** in the target and verify each `.sqsh` exists (`ls -l <PATH_TO_CONTAINERS>/*.sqsh`). Re-run container setup if needed.

### HF_HOME / cache "No such file or directory"
`HF_HOME` (and every other path-valued env var) must resolve **inside the container** -- use a mount destination (e.g. `/workspace/cache/huggingface`) or a transparently-mounted host path (e.g. `/shared/...` when `- /shared:/shared` is in `mounts`). `$HOME` and `~/.cache` will not resolve.

### Slurm jobs won't submit
- Verify account: `sacctmgr show associations user=$USER`
- Check partition: `sinfo -p <YOUR_GPU_PARTITION>`
- `QOSMaxSubmitJobPerUserLimit` from `sbatch`, or jobs pending with `QOSMaxNodePerUserLimit`, means the partition's QOS caps how many jobs or nodes you may hold at once. NVFlow submits every stage of a workflow when you run it, so choose a partition whose caps fit the workflow, or run workflows one after another. `scontrol show partition <YOUR_GPU_PARTITION>` names the QOS, and `sacctmgr show qos <QOS_NAME> format=Name,MaxSubmitPU,MaxTRESPU` shows its caps. A submission rejected partway leaves its first jobs queued; cancel them with `scancel` before you rerun
- Look at the step's log directory next to its output (see [Where settings live](#where-settings-live)) and, for SSH launches, `ssh_tunnel.job_dir`

### A stage failed and the stages after it stay pending
NVFlow submits every stage at once, each waiting for the one before it to succeed, so the stages after a failed stage stay pending (the first shows the reason `DependencyNeverSatisfied`) unless your cluster is configured to cancel them. Find the failed job with `sacct -u $USER -S now-2days -s FAILED,TIMEOUT,OUT_OF_MEMORY,NODE_FAIL`, cancel the pending ones with `scancel`, fix the cause, and rerun the same `nvflow run` command; finished work is skipped.

A chunked generation step ends with a merge job that fails with `incomplete generation chunk: <file>` when a chunk lacks its file or its `.done` marker. If `<file>.done` exists without `<file>`, delete `<file>.done` and `<file>-async`, and the step's merged `output.jsonl` if it exists, since NeMo-Skills skips every chunk while it does; then rerun to regenerate that chunk.

### Training hangs at "Starting Ray cluster"

If training jobs hang after "Starting Ray cluster" (or you see `execve(): bad interpreter: No such file or directory`), your Slurm likely needs the enroot Ray template: set `ray_template: "ray_enroot.sub.j2"` in `my_cluster.yaml`. Full symptoms, cause, and SLURM-version notes are in [Cluster Configuration → Slurm in-allocation Ray template](../operations/cluster-configuration.md#slurm-in-allocation-ray-template).

## Reference

- **Cluster Configuration Guide**: [`docs/operations/cluster-configuration.md`](../operations/cluster-configuration.md)
- **Official Container Config (NeMo-Skills)**: https://github.com/NVIDIA-NeMo/Skills/blob/main/cluster_configs/example-slurm.yaml
- **Slurm Docs**: https://slurm.schedmd.com/
- **Enroot**: https://github.com/NVIDIA/enroot
