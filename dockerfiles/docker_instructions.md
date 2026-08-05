# NVFlow Air-Gapped Docker Images

Build, validate, and deploy the five NVFlow container images for use on
air-gapped Slurm clusters.  Four of them (`nemo-rl`, `nemo-skills`, `vllm`,
`vllm-grpo`) are produced by running `docker build` against the self-contained
Dockerfiles in this directory; only `sglang` is pulled as-is. All custom images
are built on a connected host (the only step that needs internet) and then run
fully offline on the cluster.

## Images

| Image | Base | Purpose |
|---|---|---|
| `nvflow-nemo-rl` | `nvcr.io/nvidia/nemo-rl:v0.7.0` | SFT and GRPO `training`; bakes one Gym venv per component so the trainer needs no network |
| `nvflow-nemo-skills` | `ubuntu:22.04` | SDG pipeline, evaluation, data preparation, SEC data prep |
| `nvflow-vllm` | `vllm/vllm-openai:v0.22.0` | Standalone vLLM (SDG, eval) — multi-arch (amd64 + arm64) |
| `nvflow-vllm` (`v0.20.0*` tag) | `vllm/vllm-openai:v0.20.0` | Standalone vLLM (GRPO rollouts, judge) — multi-arch. Same `Dockerfile.vllm` and repo as above; `--build-arg VLLM_VERSION=v0.20.0` |
| `sglang` | `lmsysorg/sglang:v0.5.10.post1` | SGLang inference server (pulled as-is, no custom Dockerfile) |

Two more images build the same way but are only needed for specific paths:
`nvflow-nemo-gym` (CPU Gym worker, `Dockerfile.nemo-gym`, pinned by `GYM_REF`) for GRPO / DG-SDG, and `nvflow-client` (optional airgap-only launcher, `Dockerfile.nvflow`, multi-arch) for driving NVFlow over an `ssh_tunnel`. Both are covered in [containers.md](../docs/maintainers/containers.md).

## Version pins

| Build arg | Default | Where to find the right value |
|---|---|---|
| `NEMO_SKILLS_COMMIT` | `e06c9b90…` (image tag `v1.1.2`) | Must match `Dockerfile.nemo-skills` and `pyproject.toml` |
| `GYM_REF` (nemo-gym, nemo-rl) | `33ef60369…` | A commit on upstream NeMo-Gym `main`; keep both images on the same one |
| `VLLM_VERSION` (vllm) | `v0.22.0` | [vLLM releases](https://github.com/vllm-project/vllm/releases) |
| `VLLM_VERSION` (vllm-grpo) | `v0.20.0` | Pinned to match NeMo-RL v0.7.0 colocated vLLM |

## 1. Build

**Build host requirements:** any OS with Docker Engine or Docker Desktop and
internet access (Linux, macOS, Windows/WSL2 all work).  The destination Slurm
cluster is the constraint — it's almost always `linux/amd64`, so all examples
below produce amd64 images.

> **Default platform = host architecture.**  `docker build` produces an image
> for the build host's arch.  On amd64 Linux / Intel macOS / Windows that's
> `linux/amd64`.  On Apple Silicon, Graviton, or other arm64 hosts it's
> `linux/arm64` — to get amd64 from those hosts, add `--platform linux/amd64`
> via `buildx` (see arm64 section below).

### amd64 (default on amd64 build hosts)

```bash
cd /path/to/nvflow

docker build --no-cache \
    -f dockerfiles/Dockerfile.nemo-skills -t nvflow-nemo-skills:v1.1.2 .

# Airgapped trainer. Tag tracks the base version it extends. `docker build` pulls
# that base from nvcr.io, so `docker login nvcr.io` must have run first.
docker build -f dockerfiles/Dockerfile.nemo-rl -t nvflow-nemo-rl:v0.7.0 .

# CPU-only Gym worker. Tag tracks the baked GYM_REF; bump it when GYM_REF moves.
docker build -f dockerfiles/Dockerfile.nemo-gym -t nvflow-nemo-gym:0.4.0 .

# One Dockerfile builds both vLLM images; VLLM_VERSION picks the base tag, and
# both ship in the nvflow-vllm repo.
docker build -f dockerfiles/Dockerfile.vllm -t nvflow-vllm:v0.22.0 .
docker build -f dockerfiles/Dockerfile.vllm \
    --build-arg VLLM_VERSION=v0.20.0 -t nvflow-vllm:v0.20.0 .

# sglang — pulled directly, no custom Dockerfile
docker pull lmsysorg/sglang:v0.5.10.post1
```

### Cross-arch builds (e.g. amd64 image on Apple Silicon)

Cross-architecture builds need `buildx` plus QEMU emulation registered on
the host.  Register QEMU once per build host (Linux only — Docker Desktop
ships QEMU pre-registered):

```bash
docker run --privileged --rm tonistiigi/binfmt --install all
```

Then build with an explicit `--platform`:

```bash
# amd64 image from an arm64 host (most common cross-arch case for Slurm)
docker buildx build --platform linux/amd64 \
  -f dockerfiles/Dockerfile.vllm \
  -t nvflow-vllm:v0.22.0 \
  --load .
```

QEMU-emulated builds are significantly slower than native and can take
several hours, especially for `Dockerfile.nemo-rl`.  Prefer a native build
host when possible.

### linux/arm64 single-arch build

The custom images are all built multi-arch (see below); this single-platform
recipe is for testing one arch in isolation, and works for any of them by
swapping `-f`.  From an arm64 host plain `docker build` works; from an amd64
host, use `buildx` with QEMU:

```bash
docker buildx build --platform linux/arm64 \
  -f dockerfiles/Dockerfile.vllm \
  -t nvflow-vllm:v0.22.0-arm64 \
  --load .
```

`--load` only supports a single platform at a time; for multi-arch see below.

### Multi-arch build (amd64 + arm64) — push to registry

Build for both architectures and push the manifest list in one shot.  Multi-arch
builds **must** push to a registry — the local Docker image store can't hold a
manifest list, so `--load` is not an option.

Needs QEMU (above) and a `docker-container` builder — the default `docker`
driver cannot build multiple platforms:

```bash
docker buildx create --name nvflow --driver docker-container --use
docker buildx inspect --bootstrap
```

```bash
REGISTRY=<your-registry>

# Airgapped trainer. Tag tracks the base version it extends.
docker buildx build --platform linux/amd64,linux/arm64 \
  -f dockerfiles/Dockerfile.nemo-rl \
  -t $REGISTRY/nvflow-nemo-rl:v0.7.0 \
  --provenance=false --sbom=false --push .

# CPU-only Gym worker. Tag tracks the baked GYM_REF; bump it when GYM_REF moves.
docker buildx build --platform linux/amd64,linux/arm64 \
  -f dockerfiles/Dockerfile.nemo-gym \
  -t $REGISTRY/nvflow-nemo-gym:0.4.0 \
  --provenance=false --sbom=false --push .

docker buildx build --platform linux/amd64,linux/arm64 \
  -f dockerfiles/Dockerfile.vllm \
  -t $REGISTRY/nvflow-vllm:v0.22.0 \
  --provenance=false --sbom=false --push .

# Same Dockerfile and repo; VLLM_VERSION picks the base tag.
docker buildx build --platform linux/amd64,linux/arm64 \
  -f dockerfiles/Dockerfile.vllm \
  --build-arg VLLM_VERSION=v0.20.0 \
  -t $REGISTRY/nvflow-vllm:v0.20.0 \
  --provenance=false --sbom=false --push .

# --no-cache is required: ARG CACHEBUST gates the dependency-override layer, so
# a warm cache reuses stale resolutions and skips the security floors.
docker buildx build --platform linux/amd64,linux/arm64 --no-cache \
  -f dockerfiles/Dockerfile.nemo-skills \
  -t $REGISTRY/nvflow-nemo-skills:v1.1.2 \
  --provenance=false --sbom=false --push .

# Launcher, not a worker. Build from a committed tree: .baked_commit records
# `git rev-parse HEAD`, so uncommitted changes ship under the wrong provenance.
docker buildx build --platform linux/amd64,linux/arm64 \
  -f dockerfiles/Dockerfile.nvflow \
  -t $REGISTRY/nvflow-client:v1.1.2 \
  --provenance=false --sbom=false --push .

# Verify both architectures are in the manifest list
docker buildx imagetools inspect $REGISTRY/nvflow-vllm:v0.22.0
```

`--provenance=false --sbom=false` keeps the manifest list compatible with
older registries / consumers that don't understand attestation manifests.

## 2. Sanity checks (blockers)

Run these against the locally-built Docker images before the time-consuming
enroot import step.  Each check below is a hard blocker — if it fails, the
image will not work in production.

> If your build host arch differs from the image arch (e.g. running checks
> against an amd64 image on Apple Silicon), the checks will run under QEMU
> emulation as long as QEMU is registered (Docker Desktop ships it; on Linux
> see the `tonistiigi/binfmt` step above).  Without QEMU, `docker run` will
> fail with `exec format error` — defer the checks to the cluster after
> enroot import in that case.

### nemo-rl

> Required for the default release: `nvflow-nemo-rl` is built from
> `Dockerfile.nemo-rl`, and these checks are what prove its baked Gym venvs are
> usable offline. See
> [`docs/development/nemo-rl-gym.md`](../docs/development/nemo-rl-gym.md) for the
> image internals.

```bash
IMAGE=nvflow-nemo-rl:v0.7.0

# A. every Gym component venv is baked
docker run --rm $IMAGE bash -c \
  'find /opt/gym_venvs -maxdepth 3 -name .venv | sort'
# Expect: 7 paths — resources_servers/{equivalence_llm_judge,finance_sec_search,
#   format_verification}, responses_api_agents/{finance_agent,simple_agent},
#   responses_api_models/{openai_model,vllm_model}

# B. Gym imports with no network and no uv resolve
docker run --rm --network=none -e UV_OFFLINE=true $IMAGE bash -c \
  '/opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/bin/python -c \
     "import nemo_gym; print(\"nemo_gym OK\")"'
# Expect: nemo_gym OK

# C. /opt/NeMo-RL symlink (scripts/convert_checkpoint_to_hf.sh cd's to it)
docker run --rm $IMAGE bash -c \
  'cd /opt/NeMo-RL && ls examples/converters/convert_dcp_to_hf.py'
# Expect: examples/converters/convert_dcp_to_hf.py

# D. uvicorn pin (timeout_worker_healthcheck kwarg required by Gym servers)
docker run --rm $IMAGE bash -c '
  /opt/gym_venvs/resources_servers/finance_sec_search/.venv/bin/python -c "
import uvicorn, inspect
assert \"timeout_worker_healthcheck\" in inspect.signature(uvicorn.run).parameters, uvicorn.__version__
print(\"uvicorn\", uvicorn.__version__, \"OK\")"'
# Expect: uvicorn 0.52.x OK
```

### nemo-skills

```bash
IMAGE=nvflow-nemo-skills:v1.1.2

# A. tiktoken pre-cache loads offline
docker run --rm --network=none -e HF_HUB_OFFLINE=1 $IMAGE bash -c '
  python3 -c "import tiktoken; tiktoken.get_encoding(\"cl100k_base\"); print(\"OK\")"'
# Expect: OK

# B. tzdata populated (otherwise pyarrow timestamps blow up at job time:
#    ArrowInvalid: timezone "UTC" not found in timezone database)
docker run --rm $IMAGE bash -c '
  python3 -c "import pyarrow as pa; pa.array([], type=pa.timestamp(\"ns\", tz=\"UTC\")); print(\"OK\")"'
# Expect: OK

# C. SDG + SEC data-prep deps importable
#    (used by workflow-2 download_sec and workflow-3 step-0 create_seed_data;
#    all baked into Dockerfile.nemo-skills, no runtime pip install)
docker run --rm $IMAGE bash -c '
  python3 -c "import jsonlines, tiktoken, markdownify, backoff, func_timeout, tavily, edgartools, sec_parser, pandas, pyarrow, requests, bs4; print(\"OK\")"'
# Expect: OK
```

### vllm and vllm-grpo

Run the same set against both images:

```bash
for IMAGE in nvflow-vllm:v0.22.0 nvflow-vllm:v0.20.0; do
  echo "=== $IMAGE ==="

  # A. tiktoken encoding files present
  docker run --rm $IMAGE bash -c "ls /opt/tiktoken_cache/*.tiktoken"
  # Expect: o200k_base.tiktoken and cl100k_base.tiktoken

  # B. openai_harmony loads offline (no network)
  docker run --rm --network=none $IMAGE bash -c '
    python3 -c "from openai_harmony import load_harmony_encoding, HarmonyEncodingName; \
                load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS); print(\"OK\")"'
  # Expect: OK
done
```

## 3. Convert to `.sqsh` for the Slurm cluster

`enroot` runs on the Slurm compute/login nodes (Linux only).  There are two
paths from a Docker image to a `.sqsh` file — pick whichever fits your air-gap
workflow.

### Option A: via a private container registry (recommended)

Push each image, then `enroot import` from the registry on the cluster:

```bash
REGISTRY=<your-registry>

docker tag nvflow-nemo-rl:v0.7.0     $REGISTRY/nvflow-nemo-rl:v0.7.0
docker tag nvflow-nemo-gym:0.4.0     $REGISTRY/nvflow-nemo-gym:0.4.0
docker tag nvflow-nemo-skills:v1.1.2 $REGISTRY/nvflow-nemo-skills:v1.1.2
docker tag nvflow-vllm:v0.22.0       $REGISTRY/nvflow-vllm:v0.22.0
docker tag nvflow-vllm:v0.20.0       $REGISTRY/nvflow-vllm:v0.20.0

docker push $REGISTRY/nvflow-nemo-rl:v0.7.0
docker push $REGISTRY/nvflow-nemo-gym:0.4.0
docker push $REGISTRY/nvflow-nemo-skills:v1.1.2
docker push $REGISTRY/nvflow-vllm:v0.22.0
docker push $REGISTRY/nvflow-vllm:v0.20.0
```

Then on the cluster (typically a CPU partition):

```bash
CONTAINER_DIR=<absolute path to where .sqsh files should live>
REGISTRY=<your-registry>

# Name the output <yaml-key>-<tag>.sqsh to match what scripts/setup_containers.sh
# produces, so either staging method drops in to the same my_cluster.yaml.
enroot import \
  --output $CONTAINER_DIR/nemo-skills-v1.1.2.sqsh \
  "docker://$REGISTRY/nvflow-nemo-skills:v1.1.2"

# Repeat for vllm, vllm-grpo, nemo-gym, nemo-rl. sglang imports directly:
# docker://lmsysorg/sglang:v0.5.10.post1
```

If the cluster authenticates to your registry, drop credentials into
`~/.config/enroot/.credentials`:

```
machine <your-registry-host> login <user> password <token>
```

### Option B: via a saved tarball (no registry required)

For fully air-gapped sites without a private registry, save the Docker image
to a tarball, transfer it to a Linux host that has both Docker and `enroot`,
load the tarball into the local Docker daemon, then import via `dockerd://`:

```bash
# On the build host
docker save nvflow-nemo-skills:v1.1.2 | gzip > nvflow-nemo-skills-v1.1.2.tar.gz
# Transfer the .tar.gz to the cluster (scp / rsync / sneakernet)

# On the cluster (requires a Docker daemon accessible to your user)
gunzip -c nvflow-nemo-skills-v1.1.2.tar.gz | docker load
enroot import \
  --output $CONTAINER_DIR/nemo-skills-v1.1.2.sqsh \
  dockerd://nvflow-nemo-skills:v1.1.2
```

> `enroot import` natively supports only `docker://` (remote registry),
> `dockerd://` (local Docker daemon), and `podman://` URIs.  If the cluster
> has neither a private registry nor a Docker daemon, run a transient local
> registry container, push to it, and import via `docker://localhost:5000/...`.

### Two things to watch for in either option

- **Registries with a path component need `#` instead of `/`.** `enroot` parses
  `docker://<host>/<path>` such that everything after the first `/` is image
  path, which breaks for registries where the host itself contains a path
  (e.g. `nvcr.io/<org>`).  Use `#` to separate host from image path:
  ```bash
  enroot import --output vllm-v0.22.0.sqsh \
    "docker://nvcr.io#<org>/nvflow-vllm:v0.22.0"
  ```
- **Filename colon.** `enroot` writes the Docker tag separator (`:`) literally
  into the output filename.  Either pass `--output` with a shell-safe name (as
  above) or rename after import:
  ```bash
  mv "nvflow-nemo-skills:v1.1.2.sqsh" nemo-skills-v1.1.2.sqsh
  ```

## 4. Cluster config (`my_cluster.yaml`)

`cluster_configs/my_cluster.yaml` is **not tracked in git** — create it from
the template below, replacing the `<...>` placeholders with site-specific
values.  The two blocks below are the minimum required for air-gapped
operation.

### Container paths (point at the `.sqsh` files from step 3)

```yaml
containers:
  # Filenames are <yaml-key>-<tag>.sqsh, as produced by setup_containers.sh.
  nemo-rl:     <CONTAINER_DIR>/nemo-rl-v0.7.0.sqsh
  nemo-skills: <CONTAINER_DIR>/nemo-skills-v1.1.2.sqsh
  vllm:        <CONTAINER_DIR>/vllm-v0.22.0.sqsh
  vllm-grpo:   <CONTAINER_DIR>/vllm-grpo-v0.20.0.sqsh
  nemo-gym:    <CONTAINER_DIR>/nemo-gym-0.4.0.sqsh
  # sglang:    <CONTAINER_DIR>/sglang-v0.5.10.post1.sqsh
```

### Air-gap enforcement

These environment variables turn off all outbound package and model fetches:

```yaml
env_vars:
  # Disable HuggingFace network access (Hub, datasets, transformers).
  - HF_HUB_OFFLINE=1
  - HF_DATASETS_OFFLINE=1
  - TRANSFORMERS_OFFLINE=1

  # Disable uv package and Python interpreter downloads. Left UNSET: every image
  # bakes the venvs it needs, so no stage resolves at runtime either way, and
  # unset keeps a dev-mode escape hatch.
  # - UV_OFFLINE=true

  # Point tiktoken / openai_harmony at the cache baked into the images.
  # Required for nemo-skills and nemo-rl (vllm/vllm-grpo set them as ENV).
  - TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache
  - TIKTOKEN_RS_CACHE_DIR=/opt/tiktoken_cache
  - TIKTOKEN_ENCODINGS_BASE=/opt/tiktoken_cache
```

### NeMo-RL / NeMo-Gym: trainer image and Gym source

The `training` stage runs on `nvflow-nemo-rl`, built here from
`Dockerfile.nemo-rl`. It extends the NeMo-RL base with the Gym source at
`GYM_REF` and one prebuilt venv per Gym component under `/opt/gym_venvs`, so
nothing resolves at runtime and **no Gym source mount is required**.

Do not bind-mount a host Gym or NeMo-RL clone over
`/opt/nemo-rl/3rdparty/Gym-workspace/Gym` in production — it shadows the baked
source and venvs and breaks the GRPO stages. That mount is a dev-mode-only tool,
and it is the one case where `uv` resolves at runtime, so it needs `UV_OFFLINE`
unset plus a reachable pypi mirror.

The Gym-only GRPO stages (`prepare_data`, `prefetch_cache`, `collect_rollouts`,
`compute_rewards`) run on the separate `nvflow-nemo-gym` image, also with baked
per-component venvs and no mount.

For a fully-airgapped trainer with no runtime `uv` resolve, build the custom
image from [`Dockerfile.nemo-rl`](Dockerfile.nemo-rl) (bakes the Gym venvs) and
drop the Gym mount. That image needs no network at job time on its own, so
`UV_OFFLINE` still stays **unset** by policy: leaving it unset is what lets a
developer mount local Gym source and have `uv` resolve it. See
[`docs/development/nemo-rl-gym.md`](../docs/development/nemo-rl-gym.md).

## Notes for one-time / connected-node operations

A few stages legitimately need internet on first run.  Run them on a
connected node (or off-cluster) and ship the resulting artifacts onto the
air-gapped cluster:

| Stage | Why it needs internet |
|---|---|
| `workflow-2 download_sec_filings` | Downloads filings from SEC EDGAR (not HF, but still external). |
| `workflow-3 step-0 create_seed_data` | Pulls `nogabenyoash/SecQue` from HuggingFace. |
| `workflow-1 step-0 prepare_data` (eval) | Pulls finance benchmark datasets (`secque`, `financebench`) from HuggingFace. |
| `workflow-5 step-4 prepare_data` (GRPO) | Only if `should_download: true`; default `should_download: false` requires no internet. |

For these stages, temporarily clear the three HF flags
(`HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE`, `TRANSFORMERS_OFFLINE`). `UV_OFFLINE`
stays unset as always; none of these stages invoke `uv`.

Note: `huggingface_hub` interprets `TRANSFORMERS_OFFLINE=1` as
`HF_HUB_OFFLINE=1`, so all three need to be off (or unset) for HF dataset
pulls to succeed.
