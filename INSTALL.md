# Installation & Setup Guide

Quick setup guide for NVFlow - a lightweight orchestration tool for Slurm clusters.

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

### Cluster Setup Requirements

**Required:**
- **Slurm cluster** access with SSH keys

**Only needed for converting Docker images to .sqsh format:**
- **yq** - YAML parser ([install guide](https://github.com/mikefarah/yq))
- **curl** - for downloading configs
- **enroot** - on cluster nodes (for container conversion)

> **Note:** If you already have `.sqsh` container images, skip to [Configure Your Cluster](#configure-your-cluster).

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
- Storage paths for data/models/containers

---

## Setup Containers

NeMo-Skills requires Docker containers converted to `.sqsh` format for running on Slurm clusters.

**Required containers (4):**

| Container | Source | Tested Version | Action |
|-----------|--------|----------------|--------|
| `nemo-skills` | NeMo-Skills Dockerfiles | NeMo-Skills @ `0229040` | **Build** (see Step 1a) |
| `vllm` | Docker Hub | `vllm/vllm-openai:v0.18.1` | **Pull** (standalone SDG/eval) |
| `vllm-grpo` | Docker Hub | `vllm/vllm-openai:v0.17.1` | **Pull** (standalone GRPO rollouts/judge) |
| `sglang` | Docker Hub | `lmsysorg/sglang:v0.5.10.post1` | **Pull** (no build needed) |
| `nemo-rl` | NGC | `nvcr.io/nvidia/nemo-rl:v0.6.0` | **Pull** from NGC (no build needed) |

**Optional containers** (not currently used by any NVFlow recipes):

| Container | Source | Action |
|-----------|--------|--------|
| `megatron` | NeMo-Skills Dockerfiles | Build |
| `sandbox` | NeMo-Skills Dockerfiles | Build |
| `verl` | NeMo-Skills Dockerfiles | Build |
| `trtllm` | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc8` | Pull from NGC |

### Step 1a: Build NeMo-Skills Containers

Clone the NeMo-Skills repo at the **exact commit pinned by NVFlow** to ensure compatibility. The pinned commit is defined in [`pyproject.toml`](pyproject.toml):

```bash
# Clone NeMo-Skills and check out the pinned commit
git clone https://github.com/NVIDIA/NeMo-Skills.git
cd NeMo-Skills
git checkout 022904023ad7a83a87662a313cf72e7df5891d55
```

> **Tip:** Always use the commit hash from `pyproject.toml` (search for `nemo-skills @`). Building from a different version may cause incompatibilities.

Build the `nemo-skills` container using the [NeMo-Skills Dockerfiles](https://github.com/NVIDIA/NeMo-Skills/tree/022904023ad7a83a87662a313cf72e7df5891d55/dockerfiles):

```bash
# Build with the helper script
./dockerfiles/build.sh dockerfiles/Dockerfile.nemo-skills

# Or build directly with docker
docker build -t nemo-skills:latest -f dockerfiles/Dockerfile.nemo-skills .
```

For `vllm`, `vllm-grpo`, and `sglang`, pull pre-built images directly from Docker Hub (no build needed):

```bash
docker pull vllm/vllm-openai:v0.18.1       # standalone for SDG/eval
docker pull vllm/vllm-openai:v0.17.1       # standalone for GRPO rollouts/judge
docker pull lmsysorg/sglang:v0.5.10.post1
```

For optional containers (`megatron`, `sandbox`, `verl`), build them the same way using their respective Dockerfiles. For arm64 builds, see the [multi-platform instructions](https://github.com/NVIDIA/NeMo-Skills/tree/022904023ad7a83a87662a313cf72e7df5891d55/dockerfiles#building-for-arm64aarch64).

### Step 1b: Pull NeMo-RL Container (for SFT and GRPO)

The `nemo-rl` container is available as a pre-built image on NGC:

```bash
docker pull nvcr.io/nvidia/nemo-rl:v0.6.0
```

Alternatively, build from source using the [NeMo-RL repository](https://github.com/NVIDIA-NeMo/RL):

```bash
git clone https://github.com/NVIDIA-NeMo/RL.git
cd RL
git checkout v0.6.0
git submodule update --init --recursive
```

Follow the [NeMo-RL Docker build instructions](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/docker.md#building-the-release-image) to build the release image, then tag and push it to your registry alongside the NeMo-Skills containers.

### Step 2: Push Images to a Registry

After building, push the images to a container registry accessible from your cluster (Docker Hub, NGC, or a private registry):

```bash
# Tag and push the NeMo-Skills container
docker tag nemo-skills:latest your-registry/nemo-skills:latest
docker push your-registry/nemo-skills:latest

# Tag and push vllm (pulled from Docker Hub)
docker tag vllm/vllm-openai:v0.18.1 your-registry/nemo-skills-vllm:latest
docker push your-registry/nemo-skills-vllm:latest

# Tag and push NeMo-RL (pulled from NGC)
docker tag nvcr.io/nvidia/nemo-rl:v0.6.0 your-registry/nemo-skills-nemo-rl:latest
docker push your-registry/nemo-skills-nemo-rl:latest

# sglang can be pulled directly by enroot (no push needed unless your
# cluster cannot reach Docker Hub)

# Repeat for any optional images you built (e.g., megatron, sandbox, verl)
```

> **Why push?** Slurm cluster nodes typically don't have Docker installed, so `enroot` needs to pull images from a registry. Pushing to a registry also lets the automated setup script work.

### Step 3: Create Your Container Config

`containers.yaml` is a **template** with placeholder values -- do not edit it directly. Instead, copy it to a personal file and fill in your registry paths:

```bash
cp cluster_configs/containers.yaml cluster_configs/my_containers.yaml
```

Edit `my_containers.yaml` with your actual registry paths:

```yaml
containers:
  nemo-skills: your-registry/nemo-skills:latest
  vllm: your-registry/nemo-skills-vllm:latest          # v0.18.1 for SDG/eval
  vllm-grpo: vllm/vllm-openai:v0.17.1                  # v0.17.1 for GRPO rollouts/judge
  nemo-rl: nvcr.io/nvidia/nemo-rl:v0.6.0               # or your-registry/nemo-skills-nemo-rl:latest
  sglang: lmsysorg/sglang:v0.5.10.post1
```

> **Note:** `my_containers.yaml` is gitignored (`cluster_configs/*.yaml` pattern), so your registry paths stay local and won't be committed.

### Step 4: Convert to .sqsh Format

Choose one of the following methods to convert your container images to `.sqsh` format for Slurm.

#### Option A: Automated Setup (Recommended)

Use the setup script to download from your registry and convert all containers in parallel. Pass your personal config with `--config`:

```bash
# Run from a cluster login node (sbatch requires Slurm access)
sbatch --account=YOUR_ACCOUNT scripts/setup_containers.sh --config cluster_configs/my_containers.yaml ./containers
```

The `--config` flag is required -- the script reads image references from the specified YAML file, pulls them via `enroot`, and converts to `.sqsh` format. See [the script](scripts/setup_containers.sh) for additional options (`--platform`, `--force`).

**Check progress:**
```bash
tail -f outputs/logs/slurm-containers-<jobid>.out
```

#### Option B: Manual Conversion

Convert images one at a time using `enroot` on a cluster node:

```bash
# Import from your registry
enroot import docker://your-registry/nemo-skills:latest
enroot import docker://your-registry/nemo-skills-vllm:latest
enroot import docker://nvcr.io/nvidia/nemo-rl:v0.6.0

# Import from official registries
enroot import docker://vllm/vllm-openai:v0.18.1
enroot import docker://vllm/vllm-openai:v0.17.1
enroot import docker://lmsysorg/sglang:v0.5.10.post1
```

Move the resulting `.sqsh` files to your cluster's container storage path.

---

## Download Models

> ⚠️ **Important:** Pre-download models to your cluster storage before running workflows.
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

---

## Setup NeMo-RL & NeMo-Gym Sources (for GRPO)

> **Skip this section** if you're only running SDG/eval workflows. This setup is needed for GRPO RL training and recommended for multi-node SFT.

Both NeMo-RL and NeMo-Gym source trees are overlay-mounted into the NeMo-RL container via cluster config mounts. This ensures the container uses the exact tested source code.

### NeMo-RL Source Clone

Mount the NeMo-RL source into the container at `/opt/NeMo-RL`. If you already cloned it in [Step 1b](#step-1b-build-nemo-rl-container-for-sft-and-grpo), reuse that clone:

```bash
# Reuse the clone from Step 1b, or:
git clone https://github.com/NVIDIA-NeMo/RL.git
cd RL
git checkout v0.6.0
git submodule update --init --recursive
```

### NeMo-Gym Clone

Clone NeMo-Gym and mount it inside the NeMo-RL source tree. The Gym overlay is independent from the Gym submodule inside RL -- this lets RL and Gym evolve on separate branches. Check your workflow config (e.g., `grpo/base.yaml`) for the tested Gym branch or commit:

```bash
git clone https://github.com/NVIDIA-NeMo/Gym.git
cd Gym
git checkout ude/finance-sec-search  # finance_agent environment (until merged to main)
```

### Cluster Config Mounts

Add both overlay mounts to your cluster config (see [Configure Your Cluster](#configure-your-cluster)):

```yaml
mounts:
  - /path/to/RL:/opt/NeMo-RL
  - /path/to/Gym:/opt/NeMo-RL/3rdparty/Gym-workspace/Gym
```

No local `uv sync` is needed for either -- the container's `installation_command` handles dependency installation at runtime.

### Prefetch SEC Filings Cache (for `finance_sec_search`)

If using the `finance_sec_search` NeMo-Gym environment, you must prefetch the SEC filings cache to a shared mounted path. The default `~/.cache` does **not** work inside Slurm containers.

1. Follow the prefetch instructions in `Gym/resources_servers/finance_sec_search/README.md`
2. Set `cache_dir` in the environment config overlay to point to the shared mounted path

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
4. **Mount points** - Map your cluster paths to container paths
5. **Environment variables** - Set `HF_HOME` to a path visible inside the container (see [env_vars docs](docs/cluster-configuration.md#environment-variables)) and any API keys

> **Note:** The template includes detailed comments for each section. Your personal config (`my_cluster.yaml`) is gitignored to protect secrets.
>
> 📖 **For detailed documentation of all configuration fields, see the [Cluster Configuration Guide](docs/cluster-configuration.md)**.

---

## Verify Installation

```bash
# 1. Test NeMo-Skills import
uv run python -c "from nemo_skills.pipeline.cli import generate; print('✅ OK')"

# 2. Check containers exist
ls -lh <PATH_TO_CONTAINERS>/*.sqsh

# 3. Test SSH to cluster
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

### SSH connection failed
```bash
# Check key permissions
chmod 600 <PATH_TO_SSH_KEY>

# Test manual connection
ssh -i <PATH_TO_SSH_KEY> <YOUR_USERNAME>@<YOUR_CLUSTER_LOGIN_NODE>
```

### Container paths wrong
- Use **absolute paths** in cluster config
- Check file exists: `ls -l <PATH_TO_CONTAINERS>/<container>.sqsh`
- Re-run container setup if needed

### HF_HOME / cache "No such file or directory"
- `HF_HOME` (and other path-valued env vars) must resolve to a path **visible inside the container**
- Use a mount destination (e.g., `/workspace/cache/huggingface`) or a host path that is transparently mounted (e.g., `/shared/.../cache` when `- /shared:/shared` is in `mounts`)
- Paths that exist only on the host and have no corresponding mount will fail with `No such file or directory`
- Common mistake: using `$HOME` or `~/.cache` -- these do not resolve inside containers unless explicitly mounted

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

- **NeMo-Skills**: https://github.com/NVIDIA/NeMo-Skills
- **NeMo-Skills Dockerfiles**: https://github.com/NVIDIA/NeMo-Skills/tree/022904023ad7a83a87662a313cf72e7df5891d55/dockerfiles
- **NeMo-RL**: https://github.com/NVIDIA-NeMo/RL
- **NeMo-RL Docker Build**: https://github.com/NVIDIA-NeMo/RL/blob/main/docs/docker.md#building-the-release-image
- **Official Container Config**: https://github.com/NVIDIA/NeMo-Skills/blob/main/cluster_configs/example-slurm.yaml
- **Slurm Docs**: https://slurm.schedmd.com/
- **Enroot**: https://github.com/NVIDIA/enroot

**Need help?** Check the [Cluster Configuration Guide](docs/cluster-configuration.md) or ask your team/cluster admin.
