# Building & Staging the Cluster Containers (maintainers)

> Audience: **maintainers / builders** who produce the `.sqsh` container images for a cluster. If a maintainer has already staged the `.sqsh` files on your cluster, you don't need this page — just set the container paths in your cluster config (see [INSTALL.md → Setup Containers](../../INSTALL.md#setup-containers)) and continue.

NVFlow uses five core containers converted to `.sqsh` format for running on Slurm clusters, plus a CPU-only `nemo-gym` worker needed only for GRPO / DG-SDG (see [Gym worker](#gym-worker-cpu-only) below). Of the five core, **four are built locally** from self-contained Dockerfiles in [`dockerfiles/`](../../dockerfiles/) (`nemo-rl`, `nemo-skills`, `vllm`, `vllm-grpo`); only `sglang` is **pulled as-is**.

## Build host requirements

The `docker build` step needs **internet access** to pull base layers, source from GitHub, and packages from PyPI / NGC / Docker Hub. The resulting `.sqsh` files then run fully offline on the cluster.

- **Docker Engine** or **Docker Desktop** (any OS - Linux, macOS, Windows/WSL2)
- **`docker login nvcr.io`** - required once, so `docker build` can pull the NeMo-RL base image
- **`docker buildx`** - only needed for multi-arch / cross-arch builds (ships with Docker Desktop; on Linux: `docker buildx version`)

> **Note:** If your destination cluster is `linux/amd64` (the common case) and your build host is amd64 Linux / Intel macOS / Windows, the default `docker build` works without `buildx`.

## Required containers (5)

| Container | Source | Tested Version | Action |
|-----------|--------|----------------|--------|
| `nvflow-nemo-rl` | [`dockerfiles/Dockerfile.nemo-rl`](../../dockerfiles/Dockerfile.nemo-rl) | base `nvcr.io/nvidia/nemo-rl:v0.7.0`, Gym @ `33ef60369` | **Build** (Gym venvs baked) |
| `nvflow-nemo-skills` | [`dockerfiles/Dockerfile.nemo-skills`](../../dockerfiles/Dockerfile.nemo-skills) | NeMo-Skills @ `e06c9b90` (tag `v1.1.2`) | **Build** (see Step 1) |
| `nvflow-vllm` | [`dockerfiles/Dockerfile.vllm`](../../dockerfiles/Dockerfile.vllm) | base `vllm/vllm-openai:v0.22.0` | **Build** (SDG/eval) |
| `nvflow-vllm` (`v0.20.0*` tag) | [`dockerfiles/Dockerfile.vllm`](../../dockerfiles/Dockerfile.vllm) `--build-arg VLLM_VERSION=v0.20.0` | base `vllm/vllm-openai:v0.20.0` | **Build** (GRPO rollouts/judge) |
| `sglang` | Docker Hub | `lmsysorg/sglang:v0.5.10.post1` | **Pull** (no custom Dockerfile) |

> **Note:** The four custom worker images (`nemo-rl`, `nemo-skills`, `vllm`, `vllm-grpo`) are **built**; only `sglang` is **pulled as-is**. The custom Dockerfiles bake in their source, pre-built venvs, and `tiktoken` / `openai_harmony` caches so they run offline under `enroot`/`pyxis` with no outbound network.

**Optional containers** (not currently used by any NVFlow recipes):

| Container | Source | Action |
|-----------|--------|--------|
| `megatron` | NeMo-Skills Dockerfiles | Build |
| `sandbox` | NeMo-Skills Dockerfiles | Build |
| `verl` | NeMo-Skills Dockerfiles | Build |
| `trtllm` | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc8` | Pull from NGC |

### Gym worker (CPU-only)

The **Gym-only stages** — GRPO `prepare_data` / `prefetch_cache` and the DG-SDG gym stages — run in a dedicated **CPU-only** worker, **`nvflow-nemo-gym`** ([`dockerfiles/Dockerfile.nemo-gym`](../../dockerfiles/Dockerfile.nemo-gym), base `python:3.12-slim`, upstream Gym main `33ef60369`). It bakes one venv **per Gym component** (`gym env start … +dry_run`; `equivalence_llm_judge` + `finance_sec_search` + `format_verification` prebuilt, others build on demand) into `/opt/gym-venvs`, with Gym source at `/opt/Gym`. Stage it if you run **GRPO or DG-SDG** (SFT-only / eval-only runs don't need it). It is referenced by `my_cluster.yaml` `containers:` as **`nemo-gym`** and listed in [`cluster_configs/containers.yaml`](../../cluster_configs/containers.yaml). Build multi-arch (amd64 + arm64); `GYM_REF` is a pinned SHA, so layer caching is safe.

### Launcher image (optional, airgap-only)

Separate from the five **worker** containers above, the **`nvflow-client`** launcher image ([`dockerfiles/Dockerfile.nvflow`](../../dockerfiles/Dockerfile.nvflow), pinned Ubuntu 24.04 base with Python 3.12) bundles the `nflow` CLI + baked venv so **users in an airgapped environment who cannot `uv sync`** can drive NVFlow over an `ssh_tunnel`. It is a **launcher, not a worker**: it is *not* referenced by `my_cluster.yaml` `containers:` and is *not* required for a normal (`uv sync`) install. Build it **multi-arch (amd64 + arm64)** and match the client/cluster architecture. See [`docs/remote-launch.md`](../remote-launch.md) for usage. It is listed in [`cluster_configs/containers.yaml`](../../cluster_configs/containers.yaml) as `nvflow-client` (release-tag placeholder).

## Step 1: Build Docker Images

NVFlow ships self-contained Dockerfiles in [`dockerfiles/`](../../dockerfiles/) that pre-install all Python packages, pre-cache tokenizer encodings, and pre-build virtual environments. The full build commands — single-arch, cross-arch / multi-arch (`docker buildx` + QEMU), and the `sglang` pull — are in **[`dockerfiles/docker_instructions.md` §1](../../dockerfiles/docker_instructions.md#1-build)** (the authoritative build reference); per-image `ARG` version pins are in [`dockerfiles/README.md`](../../dockerfiles/README.md#version-pins).

> **Tip:** Keep `NEMO_SKILLS_COMMIT` consistent between `Dockerfile.nemo-skills` and `pyproject.toml`. For optional containers (`megatron`, `sandbox`, `verl`), build them from the upstream [NeMo-Skills Dockerfiles](https://github.com/NVIDIA-NeMo/Skills/tree/e06c9b90/dockerfiles).

### Step 1b: Sanity-Check Images Before Conversion

Before the time-consuming `enroot import` step, run the smoke checks in [`dockerfiles/docker_instructions.md` §2](../../dockerfiles/docker_instructions.md#2-sanity-checks-blockers). Each check is a **hard blocker** - if it fails locally, the image will not work in production. They verify the offline-critical pieces: `uv` works offline, the trainer's 7 baked Gym component venvs are present, `tiktoken` / `openai_harmony` caches load with `--network=none`, and `tzdata` is populated.

## Step 2: Get Images onto the Cluster

Slurm nodes usually have no Docker, so `enroot` pulls each image from a **registry** (`docker://`, recommended) or loads it from a **saved tarball** (`dockerd://`, for sites with no registry). Tag/push and `docker save` commands for both paths are in [`dockerfiles/docker_instructions.md` §3](../../dockerfiles/docker_instructions.md#3-convert-to-sqsh-for-the-slurm-cluster). `sglang` can be pulled directly by `enroot` — no push needed unless your cluster cannot reach Docker Hub.

## Step 3: Update Container Config

Copy the template to a personal file that records the registry / tag references the cluster should pull from:

```bash
cp cluster_configs/containers.yaml cluster_configs/my_containers.yaml
```

Edit `cluster_configs/my_containers.yaml` with your registry paths. The YAML **keys** (`nemo-skills`, `nemo-rl`, `vllm`, `vllm-grpo`, `sglang`) match what the workflow code references and must not be renamed; only the registry / tag values change:

```yaml
containers:
  nemo-rl:     your-registry/nvflow-nemo-rl:v0.7.0    # built locally; Gym venvs baked
  nemo-skills: your-registry/nvflow-nemo-skills:v1.1.2
  vllm:        your-registry/nvflow-vllm:v0.22.0      # v0.22.0 for SDG/eval
  vllm-grpo:   your-registry/nvflow-vllm:v0.20.0      # same repo as vllm, v0.20.0 tag for GRPO rollouts/judge
  sglang:      lmsysorg/sglang:v0.5.10.post1
```

> **Note:** `my_containers.yaml` is gitignored (`cluster_configs/*.yaml` pattern), so your registry paths stay local and won't be committed.

## Step 4: Convert to .sqsh Format

### Option A: Automated Setup (Recommended, for Option A registries)

Use the setup script to download from your registry and convert all containers in parallel. Pass your personal config with `--config`:

```bash
# Run from a cluster login node (sbatch requires Slurm access)
sbatch --account=YOUR_ACCOUNT scripts/setup_containers.sh --config cluster_configs/my_containers.yaml ./containers
```

The `--config` flag is required - the script reads image references from the specified YAML file, pulls them via `enroot`, and converts to `.sqsh` format. See [the script](../../scripts/setup_containers.sh) for additional options (`--platform`, `--force`).

> Output filenames are derived as `<key>-<tag>.sqsh` from the YAML key and tag (not the registry path), and any image whose file already exists is skipped — pass `--force` to re-download.

**Check progress:**
```bash
tail -f outputs/logs/slurm-containers-<jobid>.out
```

### Option B: Manual Conversion

Convert images one at a time using `enroot` on a cluster node. From a registry, use `docker://$REGISTRY/...`; from a loaded tarball, use `dockerd://...` after `docker load`:

```bash
CONTAINER_DIR=<absolute path on cluster where .sqsh files should live>

# Use <key>-<tag>.sqsh so manual imports and setup_containers.sh agree.
enroot import --output $CONTAINER_DIR/nemo-skills-v1.1.2.sqsh \
  "docker://$REGISTRY/nvflow-nemo-skills:v1.1.2"      # from a registry
# -- or --
gunzip -c nvflow-nemo-skills-v1.1.2.tar.gz | docker load
enroot import --output $CONTAINER_DIR/nemo-skills-v1.1.2.sqsh \
  dockerd://nvflow-nemo-skills:v1.1.2                 # from a tarball
```

Repeat for `vllm`, `vllm-grpo`, `nemo-gym`, and `nemo-rl`. `sglang` imports directly from its upstream registry (`docker://lmsysorg/sglang:v0.5.10.post1`).

**Two things to watch for:**

- **Registries with a path component need `#` instead of `/`.** `enroot` parses `docker://<host>/<path>` such that everything after the first `/` is image path, which breaks for registries where the host itself contains a path (e.g. `nvcr.io/<org>`). Use `#` to separate host from image path:
  ```bash
  enroot import --output vllm-v0.22.0.sqsh \
    "docker://nvcr.io#<org>/nvflow-vllm:v0.22.0"
  ```
- **Filename colon.** `enroot` writes the Docker tag separator (`:`) literally into the output filename. Either pass `--output` with a shell-safe name (as above) or rename after import:
  ```bash
    mv "nvflow-nemo-skills:v1.1.2.sqsh" nemo-skills-v1.1.2.sqsh
  ```

If the cluster authenticates to your registry, drop credentials into `~/.config/enroot/.credentials`:

```
machine <your-registry-host> login <user> password <token>
```

Move the resulting `.sqsh` files to your cluster's container storage path, then record those paths in your cluster config.
