# NVFlow Air-Gapped Docker Images

Build, validate, and deploy the five NVFlow container images for use on
air-gapped Slurm clusters.  Four of them are produced by running
`docker build` against the self-contained Dockerfiles in this directory;
the fifth (`sglang`) is pulled as-is from Docker Hub.  All images are built
on a connected host (the only step that needs internet) and then run fully
offline on the cluster.

## Images

| Image | Base | Purpose |
|---|---|---|
| `nvflow-nemo-rl` | `nvcr.io/nvidia/nemo-rl:v0.6.0` | SFT, GRPO training, collect_rollouts, compute_rewards |
| `nvflow-nemo-skills` | `ubuntu:22.04` | SDG pipeline, evaluation, data preparation, SEC data prep |
| `nvflow-vllm` | `vllm/vllm-openai:v0.18.1` | Standalone vLLM (SDG, eval) — multi-arch (amd64 + arm64) |
| `nvflow-vllm-grpo` | `vllm/vllm-openai:v0.17.1` | Standalone vLLM (GRPO rollouts, judge) — multi-arch |
| `sglang` | `lmsysorg/sglang:v0.5.10.post1` | SGLang inference server (pulled as-is, no custom Dockerfile) |

## Version pins

| Build arg | Default | Where to find the right value |
|---|---|---|
| `BASE_IMAGE` (nemo-rl) | `nvcr.io/nvidia/nemo-rl:v0.6.0` | [NGC NeMo-RL tags](https://catalog.ngc.nvidia.com) |
| `NEMO_SKILLS_COMMIT` | `022904023ad7a83a87662a313cf72e7df5891d55` (`0229040`) | Must match across `Dockerfile.nemo-skills` and `Dockerfile.nemo-rl` |
| `NEMO_GYM_BRANCH` | `ude/finance-sec-search-v2` | NeMo-Gym branch with finance agent |
| `VLLM_VERSION` (vllm) | `v0.18.1` | [vLLM releases](https://github.com/vllm-project/vllm/releases) |
| `VLLM_VERSION` (vllm-grpo) | `v0.17.1` | Pinned to match NeMo-RL v0.6.0 colocated vLLM |

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

docker build -f dockerfiles/Dockerfile.nemo-rl     -t nvflow-nemo-rl:v0.6.0      .
docker build -f dockerfiles/Dockerfile.nemo-skills -t nvflow-nemo-skills:0229040 .
docker build -f dockerfiles/Dockerfile.vllm        -t nvflow-vllm:v0.18.1        .
docker build -f dockerfiles/Dockerfile.vllm-grpo   -t nvflow-vllm-grpo:v0.17.1   .

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
  -t nvflow-vllm:v0.18.1 \
  --load .
```

QEMU-emulated builds are significantly slower than native and can take
several hours, especially for `Dockerfile.nemo-rl`.  Prefer a native build
host when possible.

### linux/arm64 single-arch build

Only the two vLLM images are arm64-friendly today.  From an arm64 host the
plain `docker build` works; from an amd64 host, use `buildx` with QEMU:

```bash
docker buildx build --platform linux/arm64 \
  -f dockerfiles/Dockerfile.vllm \
  -t nvflow-vllm:v0.18.1-arm64 \
  --load .
```

`--load` only supports a single platform at a time; for multi-arch see below.

### Multi-arch build (amd64 + arm64) — push to registry

For `vllm` and `vllm-grpo`, build for both architectures and push the manifest
list in one shot.  Multi-arch builds **must** push to a registry — the local
Docker image store can't hold a manifest list, so `--load` is not an option:

```bash
REGISTRY=<your-registry>

docker buildx build --platform linux/amd64,linux/arm64 \
  -f dockerfiles/Dockerfile.vllm \
  -t $REGISTRY/nvflow-vllm:v0.18.1 \
  --provenance=false --sbom=false --push .

docker buildx build --platform linux/amd64,linux/arm64 \
  -f dockerfiles/Dockerfile.vllm-grpo \
  -t $REGISTRY/nvflow-vllm-grpo:v0.17.1 \
  --provenance=false --sbom=false --push .

# Verify both architectures are in the manifest list
docker buildx imagetools inspect $REGISTRY/nvflow-vllm:v0.18.1
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

```bash
IMAGE=nvflow-nemo-rl:v0.6.0

# A. uv works offline (paths relocated out of /root)
docker run --rm -e UV_OFFLINE=true $IMAGE bash -c \
  "uv python list --only-installed | grep 3.12"
# Expect: cpython-3.12.x at /opt/uv-python/...

# B. main venv has no stale /root/.local references
docker run --rm $IMAGE bash -c '
  grep -rl "/root/.local" \
    /opt/nemo_rl_venv/pyvenv.cfg \
    /opt/ray_venvs/*/pyvenv.cfg \
    /opt/nemo-rl/3rdparty/Gym-workspace/Gym/.venv/pyvenv.cfg \
    2>/dev/null || echo "All clean"'
# Expect: All clean

# C. all 6 Gym component venvs are symlinked
docker run --rm $IMAGE bash -c '
  GYM=/opt/nemo-rl/3rdparty/Gym-workspace/Gym
  for c in \
    resources_servers/equivalence_llm_judge \
    resources_servers/finance_sec_search \
    responses_api_agents/simple_agent \
    responses_api_agents/finance_agent \
    responses_api_models/openai_model \
    responses_api_models/vllm_model; do
      [ -L "$GYM/$c/.venv" ] && echo "OK: $c" || echo "MISSING: $c"
  done'
# Expect: 6x "OK: ..."

# D. uvicorn pin (timeout_worker_healthcheck kwarg required by Gym servers)
docker run --rm $IMAGE bash -c '
  /opt/nemo-rl/3rdparty/Gym-workspace/Gym/.venv/bin/python -c "
import uvicorn, inspect
assert \"timeout_worker_healthcheck\" in inspect.signature(uvicorn.run).parameters, uvicorn.__version__
print(\"uvicorn\", uvicorn.__version__, \"OK\")"'
# Expect: uvicorn 0.37.x OK
```

### nemo-skills

```bash
IMAGE=nvflow-nemo-skills:0229040

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
for IMAGE in nvflow-vllm:v0.18.1 nvflow-vllm-grpo:v0.17.1; do
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

docker tag nvflow-nemo-rl:v0.6.0      $REGISTRY/nvflow-nemo-rl:v0.6.0
docker tag nvflow-nemo-skills:0229040 $REGISTRY/nvflow-nemo-skills:0229040
docker tag nvflow-vllm:v0.18.1        $REGISTRY/nvflow-vllm:v0.18.1
docker tag nvflow-vllm-grpo:v0.17.1   $REGISTRY/nvflow-vllm-grpo:v0.17.1

docker push $REGISTRY/nvflow-nemo-rl:v0.6.0
docker push $REGISTRY/nvflow-nemo-skills:0229040
docker push $REGISTRY/nvflow-vllm:v0.18.1
docker push $REGISTRY/nvflow-vllm-grpo:v0.17.1
```

Then on the cluster (typically a CPU partition):

```bash
CONTAINER_DIR=<absolute path to where .sqsh files should live>
REGISTRY=<your-registry>

enroot import \
  --output $CONTAINER_DIR/nvflow-nemo-rl-v0.6.0.sqsh \
  "docker://$REGISTRY/nvflow-nemo-rl:v0.6.0"

# Repeat for nemo-skills, vllm, vllm-grpo, and (optionally) sglang.
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
docker save nvflow-nemo-rl:v0.6.0 | gzip > nvflow-nemo-rl-v0.6.0.tar.gz
# Transfer the .tar.gz to the cluster (scp / rsync / sneakernet)

# On the cluster (requires a Docker daemon accessible to your user)
gunzip -c nvflow-nemo-rl-v0.6.0.tar.gz | docker load
enroot import \
  --output $CONTAINER_DIR/nvflow-nemo-rl-v0.6.0.sqsh \
  dockerd://nvflow-nemo-rl:v0.6.0
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
  enroot import --output nvflow-vllm-v0.18.1.sqsh \
    "docker://nvcr.io#<org>/nvflow-vllm:v0.18.1"
  ```
- **Filename colon.** `enroot` writes the Docker tag separator (`:`) literally
  into the output filename.  Either pass `--output` with a shell-safe name (as
  above) or rename after import:
  ```bash
  mv "nvflow-nemo-rl:v0.6.0.sqsh" nvflow-nemo-rl-v0.6.0.sqsh
  ```

## 4. Cluster config (`my_cluster.yaml`)

`cluster_configs/my_cluster.yaml` is **not tracked in git** — create it from
the template below, replacing the `<...>` placeholders with site-specific
values.  The two blocks below are the minimum required for air-gapped
operation.

### Container paths (point at the `.sqsh` files from step 3)

```yaml
containers:
  nemo-rl:     <CONTAINER_DIR>/nvflow-nemo-rl-v0.6.0.sqsh
  nemo-skills: <CONTAINER_DIR>/nvflow-nemo-skills-0229040.sqsh
  vllm:        <CONTAINER_DIR>/nvflow-vllm-v0.18.1.sqsh
  vllm-grpo:   <CONTAINER_DIR>/nvflow-vllm-grpo-v0.17.1.sqsh
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

  # Disable uv package and Python interpreter downloads.
  - UV_OFFLINE=true

  # Point tiktoken / openai_harmony at the cache baked into the images.
  # Required for nemo-skills and nemo-rl (vllm/vllm-grpo set them as ENV).
  - TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache
  - TIKTOKEN_RS_CACHE_DIR=/opt/tiktoken_cache
  - TIKTOKEN_ENCODINGS_BASE=/opt/tiktoken_cache
```

### Don't bind-mount NeMo-RL or NeMo-Gym source over the image paths

The air-gapped `nvflow-nemo-rl` image already contains NeMo-Gym venv
at `/opt/NeMo-RL/3rdparty/Gym-workspace/Gym/.venv` (sanity check **C** in
section 2 verifies this).  GRPO stages source that venv via
`installation_command: source .../Gym/.venv/bin/activate` before running.

Older dev-mode `my_cluster.yaml` templates often include host source overlays
like:

```yaml
mounts:
  # DO NOT use these with the air-gapped image — they shadow the baked .venv
  # - <host_path>/RL:/opt/NeMo-RL
  # - <host_path>/Gym:/opt/NeMo-RL/3rdparty/Gym-workspace/Gym
```

These bind-mounts hide the baked `.venv` symlink and the `installation_command`
fails with `No such file or directory` — breaking `prepare_data`,
`collect_rollouts`, `compute_rewards`, and `training` for GRPO.  Only add
these mounts if you are deliberately iterating on NeMo-RL/Gym source against a
host `.venv` you've built to be ABI-compatible with the image.

### Don't enable this in offline mode

```yaml
# - NRL_FORCE_REBUILD_VENVS=true   # forces Ray workers to re-resolve via uv
                                   # (requires internet; will fail under air-gap)
```

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
(`HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE`, `TRANSFORMERS_OFFLINE`).  Keep
`UV_OFFLINE=true` set — `uv` should never need to resolve packages at runtime.

Note: `huggingface_hub` interprets `TRANSFORMERS_OFFLINE=1` as
`HF_HUB_OFFLINE=1`, so all three need to be off (or unset) for HF dataset
pulls to succeed.
