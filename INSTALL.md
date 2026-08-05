# Installation & Setup Guide

Operator guide for getting NVFlow running on a Slurm cluster: install the client, stage containers and models, configure your cluster, and verify. NVFlow's containers are self-sufficient — all dependencies are pre-installed, so once images and models are staged, the pipeline runs fully offline.

> This guide sets up the **cluster side**. How you run the `nflow` **client** —
> local install or the airgapped `nvflow-client` container, and whether it
> submits directly or over an SSH tunnel — is summarized in
> [Choose your client setup](#choose-your-client-setup) just below.

> **Building artifacts?** Producing the container images is a maintainer task and
> lives under [`docs/maintainers/`](docs/maintainers/), not on this page.

## Choose your client setup

`nflow` only submits Slurm jobs — the heavy work runs on the cluster. Two
independent choices decide how you run it: **how you provision the client**, and
**how it reaches Slurm**.

| You run `nflow` on… | Provision the client | Reach Slurm | Guide |
|---|---|---|---|
| Cluster login/dev node (internet) | `uv sync` | direct | [README](README.md#-installation) |
| Laptop / dev box (internet) | `uv sync` | SSH tunnel | [remote-launch](docs/remote-launch.md) |
| Anywhere, airgapped / no install | `nvflow-client` container | direct or SSH tunnel | [remote-launch](docs/remote-launch.md) |

Steps 1–6 below are the **cluster side** (stage images/models, write
`my_cluster.yaml`, verify) and apply to every row above.

## 📋 What you'll do

Six steps, top to bottom. Each step below opens with a **Goal** and ends with a **✅ Done when** check so you always know where you are.

| Step | What it does | Who needs it |
|------|--------------|--------------|
| 1. [Prerequisites](#prerequisites) | Confirm cluster access + required tools | Everyone |
| 2. [Setup Containers](#setup-containers) | Stage the five `.sqsh` images on the cluster | Everyone |
| 3. [Download Models](#download-models) | Pre-stage the HF models your workflows use | Everyone |
| 4. [GRPO Prerequisites](#grpo-prerequisites) | SEC cache prefetch | **GRPO only — else skip** |
| 5. [Configure Your Cluster](#configure-your-cluster) | Write `cluster_configs/my_cluster.yaml` | Everyone |
| 6. [Verify Installation](#verify-installation) | Sanity-check the whole setup | Everyone |

> **Shortcut:** if a maintainer already staged the `.sqsh` images and models for you, you only need Steps 1, 5, and 6.

---

## Prerequisites

**Step 1 of 6 · Goal:** confirm you can reach the cluster and have the tools the setup needs.

> **Note:** This guide assumes you've already installed the client (see the [README](README.md#-installation): install `uv`, clone the repo, run `uv sync`).
>
> 🔌 **Airgapped / no internet on the install host?** Skip the local install and drive `nflow` from the prebuilt `nvflow-client` container (CLI + venv baked in, no `uv sync`, no client internet) — see **[docs/remote-launch.md](docs/remote-launch.md)**. You still stage the worker images and models on the cluster (Steps 2–3 below); only the install differs.

### Cluster Setup Requirements

**Required:**
- **Slurm cluster** access with SSH keys (or run directly from a login node)
- **enroot** - on cluster nodes (`enroot version`)

**Only needed for the parallel container conversion script:**
- **yq** - YAML parser ([install guide](https://github.com/mikefarah/yq))
- **curl** - for downloading configs

> **Note:** If you already have `.sqsh` container images staged on the cluster, skip to [Configure Your Cluster](#configure-your-cluster).

<details>
<summary><strong>Install yq</strong> (only needed for the parallel conversion script)</summary>

```bash
# Check if installed
yq --version

# If not installed:
# macOS
brew install yq

# Linux (auto-detects architecture)
# Supported platforms: linux_amd64, linux_arm64, linux_arm, linux_386, etc.
mkdir -p $HOME/bin
ARCH=$(uname -m); case "$ARCH" in x86_64) ARCH=amd64 ;; aarch64) ARCH=arm64 ;; armv7l) ARCH=arm ;; i686) ARCH=386 ;; esac
wget "https://github.com/mikefarah/yq/releases/latest/download/yq_linux_${ARCH}" -O $HOME/bin/yq
chmod +x $HOME/bin/yq

# Add to PATH (if $HOME/bin not already in PATH)
echo 'export PATH="$HOME/bin:$PATH"' >> $HOME/.bashrc
source $HOME/.bashrc
```
</details>

**curl & enroot:**
```bash
curl --version     # Usually pre-installed
enroot version     # Run on cluster node
```

**Get your cluster info:**
- Slurm account: `sacctmgr show associations user=$USER` (look for the Account column)
- Available partitions: `sinfo`
- Slurm version: `scontrol show config | grep SLURM_VERSION` (25.x needs the enroot Ray template fix - see [Troubleshooting](#troubleshooting))
- Storage paths for data/models/containers

**✅ Done when:** `enroot version` works on a cluster node and you know your Slurm account, a partition, and your storage paths.

---

## Setup Containers

**Step 2 of 6 · Goal:** have the five `.sqsh` container images staged on your cluster, with their paths in hand.

NVFlow runs its cluster jobs inside five `.sqsh` container images. As an operator you only need the `.sqsh` files **staged on your cluster** and their paths recorded in your cluster config.

- **Already have `.sqsh` files staged** (by a maintainer or a previous setup)? Note their paths and skip to [Configure Your Cluster](#configure-your-cluster).
- **Need to build / convert them yourself?** See **[docs/maintainers/containers.md](docs/maintainers/containers.md)** — build host requirements, `docker build`, push/save, and `enroot import` to `.sqsh`.

Your cluster config references the images by fixed keys — `nemo-rl`, `nemo-skills`, `vllm`, `vllm-grpo`, and `sglang`. The build guide's "Update Container Config" step explains how to set them; [Configure Your Cluster](#configure-your-cluster) ties them into your run config.

> **`nemo-gym` (needed for GRPO & DG-SDG):** the Gym-only stages — GRPO `prepare_data` / `prefetch_cache` and the DG-SDG gym stages — run in a dedicated **CPU-only** `nemo-gym` image ([`dockerfiles/Dockerfile.nemo-gym`](docs/maintainers/containers.md#gym-worker-cpu-only)). Stage this sixth image if you run **GRPO or DG-SDG**; **SFT-only** and **eval-only** runs don't need it.

**✅ Done when:** five `.sqsh` files exist on the cluster and you have their absolute paths for the config.

---

## Download Models

**Step 3 of 6 · Goal:** pre-download the models your chosen workflows need to the cluster's HF models directory.

> ⚠️ **Important:** Pre-download models to your cluster storage before running workflows. The runtime sets `HF_HUB_OFFLINE=1`, so any model not already on disk will fail at job time.
>
> **Why this matters:**
> - Avoids wasting expensive GPU time on downloads
> - Prevents race conditions when multiple jobs start simultaneously
> - Large models (10-100+ GB) can take hours to download
> - Network failures during jobs cause workflow failures

### Using hf download

**Note:** `hf` CLI is included with nemo-skills (via `huggingface-hub`). Some models are gated and require authentication -- export your HuggingFace token before downloading:

```bash
export HF_TOKEN=<YOUR_HF_TOKEN>
```

Download models to your cluster's HuggingFace models directory. The examples below show the models used by the finance recipe workflows -- download only the ones you need:

```bash
# GRPO policy model (Qwen3-30B-A3B, MoE — used in grpo/qwen3_30b_a3b.yaml)
uv run hf download Qwen/Qwen3-30B-A3B \
  --local-dir /path/to/models/hf_models/Qwen/Qwen3-30B-A3B

# GRPO / eval judge model (GPT-OSS-120B — used for rollout judging and eval)
uv run hf download openai/gpt-oss-120b \
  --local-dir /path/to/models/hf_models/openai/gpt-oss-120b
```

For the **quick-start demo** (see [quick-start.md](docs/recipes/finance/quick-start.md)), download these additional models:

```bash
# Demo policy model (Qwen3-4B — used in sft/qwen3_4b.yaml and grpo/qwen3_4b.yaml)
uv run hf download Qwen/Qwen3-4B \
  --local-dir /path/to/models/hf_models/Qwen/Qwen3-4B

# Demo SDG generation + eval baseline (GPT-OSS-20B)
uv run hf download openai/gpt-oss-20b \
  --local-dir /path/to/models/hf_models/openai/gpt-oss-20b

# Eval baseline (Gemma 3 4B IT)
uv run hf download google/gemma-3-4b-it \
  --local-dir /path/to/models/hf_models/google/gemma-3-4b-it
```

**Storage location:** Models should go in your mounted HuggingFace models directory (see cluster config `mounts` section).

### Mount Path in Cluster Config

Ensure your cluster config has the models directory mounted (cluster config creation is explained in the [Configure Your Cluster](#configure-your-cluster) section below):

```yaml
mounts:
  - /cluster/path/to/models/hf_models:/hf_models  # Maps to /hf_models inside containers
```

### Using in Workflows

Reference models using the **container mount path** (`/hf_models`):

```yaml
stage_kwargs:
  model: /hf_models/Qwen/Qwen3-4B  # Path inside container
  server_type: sglang
```

<details>
<summary><strong>Which models does each workflow need?</strong></summary>

| Model | Demo SDG | Demo SFT | Demo GRPO | Demo Eval | Production GRPO |
|-------|:--------:|:--------:|:---------:|:---------:|:---------------:|
| `Qwen/Qwen3-4B` | | ✓ | ✓ | ✓ | |
| `openai/gpt-oss-20b` | ✓ | | | ✓ | |
| `google/gemma-3-4b-it` | | | | ✓ | |
| `openai/gpt-oss-120b` | | | ✓ | | ✓ |
| `Qwen/Qwen3-30B-A3B` | | | | | ✓ |

**Tip:** Download commonly used models once and reuse across all workflows.
</details>

### One-Time Connected-Node Stages (Datasets)

A handful of stages legitimately need internet on **first** run to pull benchmark / seed datasets from HuggingFace or SEC EDGAR. Run them on a connected node (or off-cluster) and ship the resulting artifacts to the cluster - they're reused by every subsequent run.

| Stage | Pulls from | Why |
|---|---|---|
| `workflow-2 download_sec_filings` | SEC EDGAR | Filings aren't on HF |
| `workflow-3 step-0 create_seed_data` | HF `nogabenyoash/SecQue` | Seed dataset |
| `workflow-1 step-0 prepare_data` (eval) | HF `secque`, `financebench` | Benchmark data |
| `workflow-5 step-4 prepare_data` (GRPO) | HF | Only if `should_download: true` |

For these stages, **temporarily clear** the three HF offline flags (`HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE`, `TRANSFORMERS_OFFLINE`) in your cluster config. `UV_OFFLINE` is unrelated — leave it at its default (unset); none of these stages invoke `uv`.

> **Note:** `huggingface_hub` interprets `TRANSFORMERS_OFFLINE=1` as `HF_HUB_OFFLINE=1`, so all three need to be off (or unset) for HF dataset pulls to succeed.

**✅ Done when:** the models for your workflow are on disk under your mounted `hf_models` directory.

---

## GRPO Prerequisites

**Step 4 of 6 · GRPO only — skip this entire step if you're not running GRPO.**

**There are no sources to clone.** The `nvflow-nemo-rl` trainer image bakes NeMo-RL together with one prebuilt NeMo-Gym venv per component, so GRPO `training` resolves no packages at job runtime and needs no bind-mount. The Gym-only stages (`collect_rollouts`, `compute_rewards`, `prefetch_cache`, `prepare_data`) run on the equally self-contained `nvflow-nemo-gym` image.

> How that image is built, and its internals, are covered in **[docs/development/nemo-rl-gym.md](docs/development/nemo-rl-gym.md)**.

### Prefetch SEC Filings Cache (for `finance_sec_search`)

If using the `finance_sec_search` NeMo-Gym environment, you must prefetch the SEC filings cache to a shared mounted path. The default `~/.cache` does **not** work inside Slurm containers.

The GRPO workflow includes a dedicated `prefetch_cache` stage that runs on a connected node and populates the cache under your `workflow-5-grpo/` output directory. See [`docs/recipes/finance/workflows/06-grpo.md`](docs/recipes/finance/workflows/06-grpo.md) for the full prefetch flow.

**✅ Done when:** (GRPO users) the SEC filings cache is prefetched if you use `finance_sec_search`. Everyone else: nothing to do — move on.

---

## Configure Your Cluster

**Step 5 of 6 · Goal:** create and fill in `cluster_configs/my_cluster.yaml`.

### Create Your Cluster Config

```bash
# Copy template
cp cluster_configs/template-slurm.yaml cluster_configs/my_cluster.yaml
```

### Edit Your Config

Edit `cluster_configs/my_cluster.yaml` and replace all `<PLACEHOLDER>` values:

1. **SSH settings** - Your cluster login node, username, SSH key path (ONLY for remote job submission from local machine)
2. **Slurm account/partition** - Run `sacctmgr show associations user=$USER` and `sinfo`
3. **Container paths** - Copy from `outputs/logs/slurm-containers-<jobid>.out` after running setup_containers.sh
4. **Mount points** - Map your cluster paths to container paths (at minimum `<hf_models>:/hf_models` and a writable data dir `<workspace_data>:/workspace` for outputs + caches). Recipe code and assets ship via the nemo-run packaged snapshot (`/nemo_run/code`), so the repo is not mounted — see [Mount Points](docs/cluster-configuration.md#mounts).
5. **Environment variables** - Set `HF_HOME` to a path visible inside the container (see [env_vars docs](docs/cluster-configuration.md#environment-variables)) and any API keys

### Keep the Air-Gap Enforcement Block

`template-slurm.yaml` ships with the offline flags pre-populated - leave them on:

```yaml
env_vars:
  # --- AIR-GAPPED ENFORCEMENT (recommended) ---
  - HF_HUB_OFFLINE=1
  - HF_DATASETS_OFFLINE=1
  - TRANSFORMERS_OFFLINE=1
  # UV_OFFLINE: left UNSET (global flag). The images bake every venv they need,
  # so no stage resolves packages at runtime either way.
  # - UV_OFFLINE=true

  # Pre-baked tiktoken / openai_harmony cache (set as ENV in vllm/vllm-grpo
  # already; setting here applies them uniformly to nemo-skills and nemo-rl)
  - TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache
  - TIKTOKEN_RS_CACHE_DIR=/opt/tiktoken_cache
  - TIKTOKEN_ENCODINGS_BASE=/opt/tiktoken_cache
```

For the one-time connected-node stages listed in [Download Models](#one-time-connected-node-stages-datasets) above, comment out the three `HF_*_OFFLINE` flags just for that submission, then re-enable.

> **Note:** The template includes detailed comments for each section. Your personal config (`my_cluster.yaml`) is gitignored to protect secrets.
>
> 📖 **For detailed documentation of all configuration fields, see the [Cluster Configuration Guide](docs/cluster-configuration.md)**.

**✅ Done when:** `my_cluster.yaml` has no remaining `<PLACEHOLDER>` values and keeps the air-gap enforcement block.

---

## Verify Installation

**Step 6 of 6 · Goal:** confirm the whole setup before running a real workflow.

```bash
# 1. Test NeMo-Skills import
uv run python -c "from nemo_skills.pipeline.cli import generate; print('✅ OK')"

# 2. Check containers exist
ls -lh <PATH_TO_CONTAINERS>/*.sqsh

# 3. Test SSH to cluster (only if submitting from a local machine)
ssh -i <PATH_TO_SSH_KEY> <YOUR_USERNAME>@<YOUR_CLUSTER_LOGIN_NODE> "echo '✅ SSH OK'"

# 4. Test cluster config loads
uv run python -c "from omegaconf import OmegaConf; OmegaConf.load('cluster_configs/my_cluster.yaml'); print('✅ Config OK')"

# 5. List available stages
uv run nflow list-stages
```

**✅ Done when:** all five checks above pass. You're ready to run a workflow — see [Next Steps](#next-steps).

---

## Troubleshooting

### Python 3.12 not found
```bash
# UV can install it for you
uv python install 3.12
uv sync
```

### NeMo-Skills not found
```bash
uv sync --reinstall
```

### yq not found
`yq` is only needed for the parallel `.sqsh` conversion script. Install it per the [Prerequisites → Install yq](#prerequisites) block above.

### enroot not available
```bash
# On cluster
module load enroot  # if available
# Or contact your cluster admin
```

### Container build / conversion issues
Building images, `docker login nvcr.io`, and `enroot import` quirks (filename colon, `#` separator for `nvcr.io`) are covered in **[docs/maintainers/containers.md](docs/maintainers/containers.md)**.

### SSH connection failed
```bash
chmod 600 <PATH_TO_SSH_KEY>
ssh -i <PATH_TO_SSH_KEY> <YOUR_USERNAME>@<YOUR_CLUSTER_LOGIN_NODE>
```

### Container paths wrong
Use **absolute paths** in cluster config and verify each `.sqsh` exists (`ls -l <PATH_TO_CONTAINERS>/*.sqsh`). Re-run container setup if needed.

### HF_HOME / cache "No such file or directory"
`HF_HOME` (and every other path-valued env var) must resolve **inside the container** -- use a mount destination (e.g. `/workspace/cache/huggingface`) or a transparently-mounted host path (e.g. `/shared/...` when `- /shared:/shared` is in `mounts`). `$HOME` and `~/.cache` will not resolve.

### Air-gapped runtime errors at job time
For symptoms specific to the self-sufficient runtime - GRPO `installation_command` failing with "No such file or directory", `OfflineModeIsEnabled`, `uv` trying to reach PyPI, `tiktoken` / `openai_harmony` failing offline - see the [Offline Runtime](docs/recipes/finance/troubleshooting.md#offline-runtime) section in the finance troubleshooting guide.

### Slurm jobs won't submit
- Verify account: `sacctmgr show associations user=$USER`
- Check partition: `sinfo -p <YOUR_GPU_PARTITION>`
- Look at job logs in `ssh_tunnel.job_dir`

### Ray Cluster Initialization Hangs

If training jobs hang after "Starting Ray cluster" (or you see `execve(): bad interpreter: No such file or directory`), your Slurm likely needs the enroot Ray template: set `ray_template: "ray_enroot.sub.j2"` in `my_cluster.yaml`. Full symptoms, cause, and SLURM-version notes are in [cluster-configuration.md → Ray Cluster Configuration](docs/cluster-configuration.md#ray-cluster-configuration).

---

## Next Steps

✅ **Cluster setup complete!**

You can now run workflows on your cluster. Set the config directory:

```bash
export NEMO_SKILLS_CONFIG_DIR=/path/to/nvflow/cluster_configs
```

Then head back to the [README.md](README.md#-quick-start) Quick Start section to run your first workflow.

---

## Reference

- **Build the containers**: [`docs/maintainers/containers.md`](docs/maintainers/containers.md)
- **NeMo-RL / NeMo-Gym trainer image & Gym venvs**: [`docs/development/nemo-rl-gym.md`](docs/development/nemo-rl-gym.md)
- **NVFlow Dockerfiles**: [`dockerfiles/README.md`](dockerfiles/README.md)
- **NVFlow Self-Sufficient Build / Deploy Guide**: [`dockerfiles/docker_instructions.md`](dockerfiles/docker_instructions.md)
- **Cluster Configuration Guide**: [`docs/cluster-configuration.md`](docs/cluster-configuration.md)
- **NeMo-Skills**: https://github.com/NVIDIA-NeMo/Skills
- **NeMo-RL**: https://github.com/NVIDIA-NeMo/RL
- **NeMo-Gym**: https://github.com/NVIDIA-NeMo/Gym
- **Official Container Config (NeMo-Skills)**: https://github.com/NVIDIA-NeMo/Skills/blob/main/cluster_configs/example-slurm.yaml
- **Slurm Docs**: https://slurm.schedmd.com/
- **Enroot**: https://github.com/NVIDIA/enroot

**Need help?** Check the [Cluster Configuration Guide](docs/cluster-configuration.md) or ask your team/cluster admin.
