# Installation & Setup Guide

Quick setup guide for NVFlow - a lightweight orchestration tool for Slurm clusters. NVFlow's containers are self-sufficient — all dependencies are pre-installed, so no runtime downloads are needed. Once images and models are staged, the pipeline runs fully offline.

## 📋 Steps

1. [Prerequisites](#prerequisites)
2. [Setup Containers](#setup-containers)
3. [Download Models](#download-models)
4. [Setup NeMo-RL & NeMo-Gym Sources (for GRPO)](#setup-nemo-rl--nemo-gym-sources-for-grpo)
5. [Configure Your Cluster](#configure-your-cluster)
6. [Verify Installation](#verify-installation)

---

## Prerequisites

> **Note:** This guide assumes you've already completed the [README.md](README.md) setup (installed `uv`, cloned the repo, ran `uv sync`).

### Build Host Requirements (for building container images)

The `docker build` step needs **internet access** to pull base layers, source from GitHub, and packages from PyPI / NGC / Docker Hub. The resulting `.sqsh` files then run fully offline on the cluster.

- **Docker Engine** or **Docker Desktop** (any OS - Linux, macOS, Windows/WSL2)
- **`docker login nvcr.io`** - required once for the NeMo-RL base image
- **`docker buildx`** - only needed for multi-arch / cross-arch builds (ships with Docker Desktop; on Linux: `docker buildx version`)

> **Note:** If your destination cluster is `linux/amd64` (the common case) and your build host is amd64 Linux / Intel macOS / Windows, the default `docker build` works without `buildx`.

### Cluster Setup Requirements

**Required:**
- **Slurm cluster** access with SSH keys (or run directly from a login node)
- **enroot** - on cluster nodes (`enroot version`)

**Only needed for the parallel container conversion script:**
- **yq** - YAML parser ([install guide](https://github.com/mikefarah/yq))
- **curl** - for downloading configs

> **Note:** If you already have `.sqsh` container images staged on the cluster, skip to [Configure Your Cluster](#configure-your-cluster).

**yq (YAML parser):**
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

---

## Setup Containers

NVFlow uses five containers converted to `.sqsh` format for running on Slurm clusters. **Four are built locally** from self-contained Dockerfiles in [`dockerfiles/`](dockerfiles/); the fifth (`sglang`) is pulled as-is.

**Required containers (5):**

| Container | Source | Tested Version | Action |
|-----------|--------|----------------|--------|
| `nvflow-nemo-rl` | [`dockerfiles/Dockerfile.nemo-rl`](dockerfiles/Dockerfile.nemo-rl) | base `nvcr.io/nvidia/nemo-rl:v0.6.0` | **Build** (see Step 1) |
| `nvflow-nemo-skills` | [`dockerfiles/Dockerfile.nemo-skills`](dockerfiles/Dockerfile.nemo-skills) | NeMo-Skills @ `786d8c58` | **Build** (see Step 1) |
| `nvflow-vllm` | [`dockerfiles/Dockerfile.vllm`](dockerfiles/Dockerfile.vllm) | base `vllm/vllm-openai:v0.18.1` | **Build** (SDG/eval) |
| `nvflow-vllm-grpo` | [`dockerfiles/Dockerfile.vllm-grpo`](dockerfiles/Dockerfile.vllm-grpo) | base `vllm/vllm-openai:v0.17.1` | **Build** (GRPO rollouts/judge) |
| `sglang` | Docker Hub | `lmsysorg/sglang:v0.5.10.post1` | **Pull** (no custom Dockerfile) |

> **Note:** All four custom images are **built**, not pulled. The four Dockerfiles bake in NeMo-Skills source, NeMo-Gym source, pre-built virtual environments, `tiktoken` / `openai_harmony` encoding caches, and `/root/.local → /opt/uv-python` path relocation so the images run cleanly under `enroot`/`pyxis` on Slurm with no outbound network access.

**Optional containers** (not currently used by any NVFlow recipes):

| Container | Source | Action |
|-----------|--------|--------|
| `megatron` | NeMo-Skills Dockerfiles | Build |
| `sandbox` | NeMo-Skills Dockerfiles | Build |
| `verl` | NeMo-Skills Dockerfiles | Build |
| `trtllm` | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc8` | Pull from NGC |

### Step 1: Build Docker Images

NVFlow ships self-contained Dockerfiles in [`dockerfiles/`](dockerfiles/) that pre-install all Python packages, pre-cache tokenizer encodings, and pre-build virtual environments. Run `docker build` on a connected build host:

```bash
cd /path/to/nvflow

# Build all four custom images (amd64, the common case)
docker build -f dockerfiles/Dockerfile.nemo-rl     -t nvflow-nemo-rl:v0.6.0      .
docker build -f dockerfiles/Dockerfile.nemo-skills -t nvflow-nemo-skills:786d8c58 .
docker build -f dockerfiles/Dockerfile.vllm        -t nvflow-vllm:v0.18.1        .
docker build -f dockerfiles/Dockerfile.vllm-grpo   -t nvflow-vllm-grpo:v0.17.1   .

# sglang is pulled as-is, no custom Dockerfile
docker pull lmsysorg/sglang:v0.5.10.post1
```

> **Tip:** The Dockerfiles expose `ARG`s for version pins (`NEMO_SKILLS_COMMIT`, `NEMO_GYM_BRANCH`, `VLLM_VERSION`, `BASE_IMAGE`). Defaults are listed in [`dockerfiles/README.md`](dockerfiles/README.md#version-pins). Keep `Dockerfile.nemo-skills`'s `NEMO_SKILLS_COMMIT` in sync with the host `[skills]` pin in `pyproject.toml`; `Dockerfile.nemo-rl` carries its own **independent** `NEMO_SKILLS_COMMIT` (nemo-rl's data-prep checkout) that need not match.

For **cross-arch builds** (e.g. building an `amd64` image on Apple Silicon, or a multi-arch manifest list pushed directly to a registry), see [`dockerfiles/docker_instructions.md`](dockerfiles/docker_instructions.md#1-build). Cross-arch builds use `docker buildx` with QEMU emulation and are significantly slower than native.

For optional containers (`megatron`, `sandbox`, `verl`), build them from the upstream [NeMo-Skills Dockerfiles](https://github.com/NVIDIA-NeMo/Skills/tree/9c631057c17ff37cb7f5c5470d76dc5b6c85c52f/dockerfiles).

### Step 1b: Sanity-Check Images Before Conversion

Before the time-consuming `enroot import` step, run the smoke checks in [`dockerfiles/docker_instructions.md` §2](dockerfiles/docker_instructions.md#2-sanity-checks-blockers). Each check is a **hard blocker** - if it fails locally, the image will not work in production. They verify the offline-critical pieces: `uv` works offline, the 6 Gym `.venv` symlinks are intact, `tiktoken` / `openai_harmony` caches load with `--network=none`, and `tzdata` is populated.

### Step 2: Push Images to a Registry (Option A) or Save as Tarball (Option B)

`enroot` runs on Slurm compute/login nodes (Linux only). There are two paths from a Docker image to a `.sqsh` file - pick whichever fits your offline workflow.

#### Option A: Via a private container registry (recommended)

Push the built images to a registry accessible from your cluster (Docker Hub, NGC, or a private registry):

```bash
REGISTRY=<your-registry>

docker tag nvflow-nemo-rl:v0.6.0      $REGISTRY/nvflow-nemo-rl:v0.6.0
docker tag nvflow-nemo-skills:786d8c58 $REGISTRY/nvflow-nemo-skills:786d8c58
docker tag nvflow-vllm:v0.18.1        $REGISTRY/nvflow-vllm:v0.18.1
docker tag nvflow-vllm-grpo:v0.17.1   $REGISTRY/nvflow-vllm-grpo:v0.17.1

docker push $REGISTRY/nvflow-nemo-rl:v0.6.0
docker push $REGISTRY/nvflow-nemo-skills:786d8c58
docker push $REGISTRY/nvflow-vllm:v0.18.1
docker push $REGISTRY/nvflow-vllm-grpo:v0.17.1

# sglang can be pulled directly by enroot (no push needed unless your
# cluster cannot reach Docker Hub).
```

> **Why push?** Slurm cluster nodes typically don't have Docker installed, so `enroot` needs to pull images from a registry (or load them from a Docker daemon - see Option B).

#### Option B: Via a saved tarball (no registry required)

For offline sites without a private registry, save the Docker image to a tarball, transfer it to a Linux host that has both Docker and `enroot`, load the tarball into the local Docker daemon, then import via `dockerd://`:

```bash
# On the build host
docker save nvflow-nemo-rl:v0.6.0      | gzip > nvflow-nemo-rl-v0.6.0.tar.gz
docker save nvflow-nemo-skills:786d8c58 | gzip > nvflow-nemo-skills-786d8c58.tar.gz
docker save nvflow-vllm:v0.18.1        | gzip > nvflow-vllm-v0.18.1.tar.gz
docker save nvflow-vllm-grpo:v0.17.1   | gzip > nvflow-vllm-grpo-v0.17.1.tar.gz

# Transfer the .tar.gz files to the cluster (scp / rsync / sneakernet)
```

`enroot import` natively supports only `docker://` (remote registry), `dockerd://` (local Docker daemon), and `podman://` URIs. If the cluster has neither a private registry nor a Docker daemon, run a transient local registry container, push to it, and import via `docker://localhost:5000/...`.

### Step 3: Update Container Config

Copy the template to a personal file that records the registry / tag references the cluster should pull from:

```bash
cp cluster_configs/containers.yaml cluster_configs/my_containers.yaml
```

Edit `cluster_configs/my_containers.yaml` with your registry paths. The YAML **keys** (`nemo-skills`, `nemo-rl`, `vllm`, `vllm-grpo`, `sglang`) match what the workflow code references and must not be renamed; only the registry / tag values change:

```yaml
containers:
  nemo-rl:     your-registry/nvflow-nemo-rl:v0.6.0
  nemo-skills: your-registry/nvflow-nemo-skills:786d8c58
  vllm:        your-registry/nvflow-vllm:v0.18.1          # v0.18.1 for SDG/eval
  vllm-grpo:   your-registry/nvflow-vllm-grpo:v0.17.1     # v0.17.1 for GRPO rollouts/judge
  sglang:      lmsysorg/sglang:v0.5.10.post1
```

> **Note:** `my_containers.yaml` is gitignored (`cluster_configs/*.yaml` pattern), so your registry paths stay local and won't be committed.

### Step 4: Convert to .sqsh Format

#### Option A: Automated Setup (Recommended, for Option A registries)

Use the setup script to download from your registry and convert all containers in parallel. Pass your personal config with `--config`:

```bash
# Run from a cluster login node (sbatch requires Slurm access)
sbatch --account=YOUR_ACCOUNT scripts/setup_containers.sh --config cluster_configs/my_containers.yaml ./containers
```

The `--config` flag is required - the script reads image references from the specified YAML file, pulls them via `enroot`, and converts to `.sqsh` format. See [the script](scripts/setup_containers.sh) for additional options (`--platform`, `--force`).

**Check progress:**
```bash
tail -f outputs/logs/slurm-containers-<jobid>.out
```

#### Option B: Manual Conversion

Convert images one at a time using `enroot` on a cluster node. From a registry, use `docker://$REGISTRY/...`; from a loaded tarball, use `dockerd://...` after `docker load`:

```bash
CONTAINER_DIR=<absolute path on cluster where .sqsh files should live>

enroot import --output $CONTAINER_DIR/nvflow-nemo-rl-v0.6.0.sqsh \
  "docker://$REGISTRY/nvflow-nemo-rl:v0.6.0"          # from a registry
# -- or --
gunzip -c nvflow-nemo-rl-v0.6.0.tar.gz | docker load
enroot import --output $CONTAINER_DIR/nvflow-nemo-rl-v0.6.0.sqsh \
  dockerd://nvflow-nemo-rl:v0.6.0                     # from a tarball
```

Repeat for `nemo-skills`, `vllm`, `vllm-grpo`, and (if needed) `sglang`.

**Two things to watch for:**

- **Registries with a path component need `#` instead of `/`.** `enroot` parses `docker://<host>/<path>` such that everything after the first `/` is image path, which breaks for registries where the host itself contains a path (e.g. `nvcr.io/<org>`). Use `#` to separate host from image path:
  ```bash
  enroot import --output nvflow-vllm-v0.18.1.sqsh \
    "docker://nvcr.io#<org>/nvflow-vllm:v0.18.1"
  ```
- **Filename colon.** `enroot` writes the Docker tag separator (`:`) literally into the output filename. Either pass `--output` with a shell-safe name (as above) or rename after import:
  ```bash
  mv "nvflow-nemo-rl:v0.6.0.sqsh" nvflow-nemo-rl-v0.6.0.sqsh
  ```

If the cluster authenticates to your registry, drop credentials into `~/.config/enroot/.credentials`:

```
machine <your-registry-host> login <user> password <token>
```

Move the resulting `.sqsh` files to your cluster's container storage path.

---

## Download Models

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

**Models needed per workflow:**

| Model | Demo SDG | Demo SFT | Demo GRPO | Demo Eval | Production GRPO |
|-------|:--------:|:--------:|:---------:|:---------:|:---------------:|
| `Qwen/Qwen3-4B` | | ✓ | ✓ | ✓ | |
| `openai/gpt-oss-20b` | ✓ | | | ✓ | |
| `google/gemma-3-4b-it` | | | | ✓ | |
| `openai/gpt-oss-120b` | | | ✓ | | ✓ |
| `Qwen/Qwen3-30B-A3B` | | | | | ✓ |

**Tip:** Download commonly used models once and reuse across all workflows.

### One-Time Connected-Node Stages (Datasets)

A handful of stages legitimately need internet on **first** run to pull benchmark / seed datasets from HuggingFace or SEC EDGAR. Run them on a connected node (or off-cluster) and ship the resulting artifacts to the cluster - they're reused by every subsequent run.

| Stage | Pulls from | Why |
|---|---|---|
| `workflow-2 download_sec_filings` | SEC EDGAR | Filings aren't on HF |
| `workflow-3 step-0 create_seed_data` | HF `nogabenyoash/SecQue` | Seed dataset |
| `workflow-1 step-0 prepare_data` (eval) | HF `secque`, `financebench` | Benchmark data |
| `workflow-5 step-4 prepare_data` (GRPO) | HF | Only if `should_download: true` |

For these stages, **temporarily clear** the three HF offline flags (`HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE`, `TRANSFORMERS_OFFLINE`) in your cluster config. Keep `UV_OFFLINE=true` set - `uv` should never need to resolve packages at runtime.

> **Note:** `huggingface_hub` interprets `TRANSFORMERS_OFFLINE=1` as `HF_HUB_OFFLINE=1`, so all three need to be off (or unset) for HF dataset pulls to succeed.

---

## Setup NeMo-RL & NeMo-Gym Sources (for GRPO)

> **Skip this section** if you're using the self-sufficient `nvflow-nemo-rl` image as-is (the recommended path). The image already contains the NeMo-RL source, a pinned NeMo-Gym branch, and a pre-built `.venv` symlinked across all 6 Gym components. No host clones or overlay mounts are required for SDG, SFT, GRPO, or eval workflows.

This section is **dev mode only** - read it only if you're actively iterating on NeMo-RL or NeMo-Gym source against the self-sufficient image.

### What the self-sufficient image already contains

`nvflow-nemo-rl` is built from [`dockerfiles/Dockerfile.nemo-rl`](dockerfiles/Dockerfile.nemo-rl) on top of `nvcr.io/nvidia/nemo-rl:v0.6.0` and bakes in:

- NeMo-Skills @ `786d8c58` installed into the frozen `/opt/nemo_rl_venv`
- NeMo-Gym source at `/opt/NeMo-RL/3rdparty/Gym-workspace/Gym`, checked out at the `ude/finance-sec-search-v2` branch (override via `NEMO_GYM_BRANCH` build arg)
- A pre-built Gym `.venv` symlinked across all 6 components (`equivalence_llm_judge`, `finance_sec_search`, `simple_agent`, `finance_agent`, `openai_model`, `vllm_model`)
- `/root/.local/share/uv/python` relocated to `/opt/uv-python` and `/root/.local/bin` to `/opt/uv-bin` so the venvs survive enroot/pyxis mounting `$HOME` over `/root`
- `tiktoken` / `openai_harmony` encoding caches at `/opt/tiktoken_cache`

GRPO stages call `installation_command: source /opt/NeMo-RL/3rdparty/Gym-workspace/Gym/.venv/bin/activate` and find everything they need inside the image.

### Do NOT overlay-mount source over the image paths

Bind-mounting a host clone at `/opt/NeMo-RL` or `/opt/NeMo-RL/3rdparty/Gym-workspace/Gym` **shadows the baked `.venv`**, and `installation_command` fails with `No such file or directory` - breaking `prepare_data`, `collect_rollouts`, `compute_rewards`, and `training` for GRPO.

The older dev-mode overlay snippets in `template-slurm.yaml` are commented out for exactly this reason:

```yaml
mounts:
  # DO NOT use these with the self-sufficient image — they shadow the baked .venv
  # - <PATH_TO_NEMO_RL_CLONE>:/opt/NeMo-RL
  # - <PATH_TO_GYM_CLONE>:/opt/NeMo-RL/3rdparty/Gym-workspace/Gym
```

### Dev mode: iterating on NeMo-RL / NeMo-Gym source

If you really need to iterate on NeMo-RL or NeMo-Gym source against this image, clone the source trees (NeMo-RL at `v0.6.0` with submodules, NeMo-Gym at `ude/finance-sec-search-v2`), uncomment the two overlay mounts in `cluster_configs/my_cluster.yaml`, and set `NRL_FORCE_REBUILD_VENVS=true` in `env_vars` so Ray workers rebuild their venvs against your source. Your host clone must contain a `.venv` ABI-compatible with the image, and `NRL_FORCE_REBUILD_VENVS=true` **requires internet** at job time — only use it on a connected node, never in production.

### Prefetch SEC Filings Cache (for `finance_sec_search`)

If using the `finance_sec_search` NeMo-Gym environment, you must prefetch the SEC filings cache to a shared mounted path. The default `~/.cache` does **not** work inside Slurm containers.

The GRPO workflow includes a dedicated `prefetch_cache` stage that runs on a connected node and populates the cache under your `workflow-5-grpo/` output directory. See [`docs/recipes/finance/workflows/06-grpo.md`](docs/recipes/finance/workflows/06-grpo.md) for the full prefetch flow.

---

## Configure Your Cluster

### Step 1: Create Your Cluster Config

```bash
# Copy template
cp cluster_configs/template-slurm.yaml cluster_configs/my_cluster.yaml
```

### Step 2: Edit Your Config

Edit `cluster_configs/my_cluster.yaml` and replace all `<PLACEHOLDER>` values:

1. **SSH settings** - Your cluster login node, username, SSH key path (ONLY for remote job submission from local machine)
2. **Slurm account/partition** - Run `sacctmgr show associations user=$USER` and `sinfo`
3. **Container paths** - Copy from `outputs/logs/slurm-containers-<jobid>.out` after running setup_containers.sh
4. **Mount points** - Map your cluster paths to container paths (at minimum `<hf_models>:/hf_models` and `<workspace>:/workspace`)
5. **Environment variables** - Set `HF_HOME` to a path visible inside the container (see [env_vars docs](docs/cluster-configuration.md#environment-variables)) and any API keys

### Step 3: Keep the Air-Gap Enforcement Block

`template-slurm.yaml` ships with the offline flags pre-populated - leave them on:

```yaml
env_vars:
  # --- AIR-GAPPED ENFORCEMENT (recommended) ---
  - HF_HUB_OFFLINE=1
  - HF_DATASETS_OFFLINE=1
  - TRANSFORMERS_OFFLINE=1
  - UV_OFFLINE=true

  # Pre-baked tiktoken / openai_harmony cache (set as ENV in vllm/vllm-grpo
  # already; setting here applies them uniformly to nemo-skills and nemo-rl)
  - TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache
  - TIKTOKEN_RS_CACHE_DIR=/opt/tiktoken_cache
  - TIKTOKEN_ENCODINGS_BASE=/opt/tiktoken_cache

  # Do NOT set in self-sufficient mode — forces Ray workers to re-resolve via uv
  # - NRL_FORCE_REBUILD_VENVS=true
```

For the one-time connected-node stages listed in [Download Models](#one-time-connected-node-stages-datasets) above, comment out the three `HF_*_OFFLINE` flags just for that submission, then re-enable.

> **Note:** The template includes detailed comments for each section. Your personal config (`my_cluster.yaml`) is gitignored to protect secrets.
>
> 📖 **For detailed documentation of all configuration fields, see the [Cluster Configuration Guide](docs/cluster-configuration.md)**.

---

## Verify Installation

```bash
# 1. Test NeMo-Skills import (requires `uv sync --extra skills`, or run inside the
#    nemo-skills container; the default `uv sync` does not install nemo_skills)
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
```bash
# macOS
brew install yq

# Linux (auto-detects architecture)
mkdir -p $HOME/bin
ARCH=$(uname -m); case "$ARCH" in x86_64) ARCH=amd64 ;; aarch64) ARCH=arm64 ;; armv7l) ARCH=arm ;; i686) ARCH=386 ;; esac
wget "https://github.com/mikefarah/yq/releases/latest/download/yq_linux_${ARCH}" -O $HOME/bin/yq
chmod +x $HOME/bin/yq

# Add to PATH if needed
echo 'export PATH="$HOME/bin:$PATH"' >> $HOME/.bashrc
source $HOME/.bashrc
```

### enroot not available
```bash
# On cluster
module load enroot  # if available
# Or contact your cluster admin
```

### Docker build fails on `docker login` for NeMo-RL base image
The `nvflow-nemo-rl` build pulls from `nvcr.io/nvidia/nemo-rl:v0.6.0` (NGC). Run `docker login nvcr.io` once (username `$oauthtoken`, password = your [NGC API key](https://ngc.nvidia.com/setup/api-key)).

### `enroot import` quirks (filename colon, `#` separator for `nvcr.io`)
See [Two things to watch for](#step-4-convert-to-sqsh-format) in Step 4.

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

**Problem:** Ray cluster hangs during initialization, workers fail to connect

**Symptoms:**
- Training jobs hang after "Starting Ray cluster"
- Error: `execve(): bad interpreter: No such file or directory`
- Ray workers show connection failures in logs

**Cause:** SLURM container reattachment issues (SLURM 25.x may be affected)

**Solution:**
Add `ray_template` to your cluster config:

```yaml
# In cluster_configs/my_cluster.yaml
executor: slurm
ray_template: "ray_enroot.sub.j2"  # Fixes Ray cluster initialization
```

**Check your SLURM version:**
```bash
scontrol show config | grep SLURM_VERSION
# Confirmed: SLURM 25.11.2 needs this fix, SLURM 24.x works without it
```

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

- **NVFlow Dockerfiles**: [`dockerfiles/README.md`](dockerfiles/README.md)
- **NVFlow Self-Sufficient Build / Deploy Guide**: [`dockerfiles/docker_instructions.md`](dockerfiles/docker_instructions.md)
- **Cluster Configuration Guide**: [`docs/cluster-configuration.md`](docs/cluster-configuration.md)
- **NeMo-Skills**: https://github.com/NVIDIA-NeMo/Skills
- **NeMo-Skills Dockerfiles (upstream reference)**: https://github.com/NVIDIA-NeMo/Skills/tree/9c631057c17ff37cb7f5c5470d76dc5b6c85c52f/dockerfiles
- **NeMo-RL**: https://github.com/NVIDIA-NeMo/RL
- **NeMo-Gym**: https://github.com/NVIDIA-NeMo/Gym
- **Official Container Config (NeMo-Skills)**: https://github.com/NVIDIA-NeMo/Skills/blob/main/cluster_configs/example-slurm.yaml
- **Slurm Docs**: https://slurm.schedmd.com/
- **Enroot**: https://github.com/NVIDIA/enroot

**Need help?** Check the [Cluster Configuration Guide](docs/cluster-configuration.md) or ask your team/cluster admin.
