# Build and Stage NVFlow's Container Images

NVFlow builds images for NeMo-RL, NeMo-Gym, NeMo-Skills, vLLM, SGLang, and the
optional `nvflow-client` launcher. Slurm and Ray targets use the same images,
and each ships its environments built in, so jobs install nothing at runtime.
Each image's contents and version pins are in [`README.md`](README.md).

## 1. Build

Use a Linux host with internet access, Docker Engine, Buildx `0.28.0` or newer
(the build script checks it), push access to a container registry, and enough
disk for CUDA images. The `nemo-rl` and `nemo-gym` builds are the largest: each
leaves about 90 GB of build cache per architecture. When disk is short, build
them in separate runs and clear the cache after each push with
`docker buildx prune --builder nvflow-multiarch -af`.

### Choose a registry

Set `REGISTRY` to a registry host and namespace your clusters can pull from,
and log in if it requires authentication:

```bash
export REGISTRY=<registry-host>/<namespace>
docker login <registry-host>
```

Never put registry credentials in repository files. Every image builds from
public base images, so the build host needs only this registry; logging in to
Docker Hub raises its limit on anonymous pulls.

To push to a registry on the build host instead, start one there:

```bash
docker run -d --restart=always -p 5000:5000 --name registry registry:2
export REGISTRY=localhost:5000/nvflow
```

A cluster cannot pull from a registry on the build host, so stage those images
[from a tarball](#without-a-registry).

### Prepare the builder

`scripts/build_images.sh` creates a `nvflow-multiarch` Buildx builder if it is
missing, and stops before building unless that builder lists both
`linux/amd64` and `linux/arm64`. Docker Desktop already emulates arm64. On an
amd64 Linux host, register QEMU, and again after every reboot, then restart the
builder so it detects the new platform:

```bash
docker run --privileged --rm tonistiigi/binfmt --install all
docker buildx stop nvflow-multiarch
```

Emulated arm64 builds are slow. A native arm64 node avoids emulation:
`docker buildx create --append --name nvflow-multiarch --platform linux/arm64 <arm64-docker-endpoint>`.

For a registry on `localhost` or `127.*`, the script creates the builder with
host networking and plain HTTP. If a builder of that name already exists
without those settings, remove it with `docker buildx rm nvflow-multiarch` and
rerun.

### Build and push

Build from a clean, normal clone: `nvflow-client` records the commit, so the
script refuses a dirty tree or a linked worktree. Run it inside `tmux` or
`screen`, because the builds take hours:

```bash
scripts/build_images.sh
```

The script builds `nemo-skills`, `nemo-rl`, `nemo-gym`, `vllm-0-29-0`,
`sglang-0-5-17`, and `nvflow-client` one at a time, each for `linux/amd64` and
`linux/arm64`. It
pushes each with `--provenance=false --sbom=false`, so registries that do not
understand attestation manifests can read it, checks that every pushed
manifest lists both platforms, and stops at the first failure with the end of
that target's log. Full logs and pushed digests go to
`~/nvflow-build-logs/<timestamp>/`. It publishes:

- `nvflow-nemo-skills:v1.2.0`
- `nvflow-nemo-rl:v0.8.0`
- `nvflow-nemo-gym:v0.7.0`
- `nvflow-vllm:v0.29.0`
- `nvflow-sglang:v0.5.17-ray2.58.0`
- `nvflow-client:v1.2.0`

Options:

- Name targets to build only those, in that order:
  `scripts/build_images.sh nemo-skills nemo-rl nemo-gym vllm-0-29-0`. Only
  workflows that select SGLang need `sglang-0-5-17`.
- `--log-dir DIR` writes the logs to `DIR`, which must be outside the clone.
- `--serial` builds one platform at a time and then joins them into the
  multi-architecture tag, for hosts with limited CPU or memory. It also leaves
  `<tag>-amd64` and `<tag>-arm64` tags.
- `--platform linux/amd64` (or `linux/arm64`) builds one platform and pushes
  only `<tag>-amd64` (or `<tag>-arm64`), so it never replaces a
  dual-architecture tag. Use that tag in `my_containers.yaml` on Slurm, or in
  your Ray clusters' image settings.

Run one copy of the script per builder at a time. If an image fails, rerun it
and the images after it, for example
`scripts/build_images.sh vllm-0-29-0 sglang-0-5-17 nvflow-client`; completed steps come from
the build cache, except in `nemo-skills`, which always rebuilds from scratch.
Two arm64 failures come from emulation rather than from the image:

- `Exec format error`: QEMU is no longer registered, for example after a
  reboot. Register it again and restart the builder, as in
  [Prepare the builder](#prepare-the-builder).
- `qemu: uncaught target signal 11` while `ldconfig` runs: a
  [known QEMU bug](https://gitlab.com/qemu-project/qemu/-/issues/1913) with
  Ubuntu 22.04 arm64 images. Rerun.

## 2. Verify

The `nemo-skills`, `vllm`, and `sglang` builds fail unless Python, Ray, and the
serving engine match their pins, `pip check` passes, and the tokenizer files
load. The `nemo-rl` and `nemo-gym` builds check the Gym environments they add,
and the client build checks Python and Ray in every environment it bakes. The
script also checks that every pushed manifest lists both platforms. Two checks
remain. On a host of the other architecture they run under QEMU.

The tokenizer files load without network access:

```bash
docker run --rm --network=none --entrypoint python3 \
  "$REGISTRY/nvflow-nemo-skills:v1.2.0" \
  -c 'import tiktoken; tiktoken.get_encoding("cl100k_base"); print("OK")'
for image in nvflow-vllm:v0.29.0 nvflow-sglang:v0.5.17-ray2.58.0; do
  docker run --rm --network=none --entrypoint python3 "$REGISTRY/$image" -c '
from openai_harmony import load_harmony_encoding, HarmonyEncodingName
load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
print("OK")'
done
```

The launcher image was built from your commit and validates a workflow without
network access:

```bash
docker run --rm --network=none --entrypoint bash "$REGISTRY/nvflow-client:v1.2.0" -lc "
  set -e
  test \"\$(cat /opt/nvflow/.baked_commit)\" = $(git rev-parse HEAD)
  nvflow list-projects >/dev/null
  nvflow validate --workflow starter:generate >/dev/null
  echo OK"
```

## 3. Stage the images on Slurm

Slurm runs each job in a `.sqsh` copy of an image; Ray clusters run the images
directly. Convert them as in
[Stage the images](../docs/slurm/README.md#2-stage-the-images): copy
`cluster_configs/containers.yaml` to `cluster_configs/my_containers.yaml`, set
your registry, and submit `scripts/setup_containers.sh`. It writes
`<role>-<tag>.sqsh` files and skips any that already exist; add `--force` after
rebuilding an image under the same tag.

### Without a registry

A cluster that cannot pull from your registry can import an image from a
tarball on a Linux host with Docker and Enroot. If your Docker Engine uses the
containerd image store, add `--platform linux/amd64` to `docker save`:

```bash
# On the build host: pull the cluster's platform and save it.
docker pull --platform linux/amd64 "$REGISTRY/nvflow-nemo-skills:v1.2.0"
docker tag "$REGISTRY/nvflow-nemo-skills:v1.2.0" nvflow-nemo-skills:v1.2.0
docker save nvflow-nemo-skills:v1.2.0 | gzip > nvflow-nemo-skills-v1.2.0.tar.gz

# On the cluster host, after copying the tarball there.
gunzip -c nvflow-nemo-skills-v1.2.0.tar.gz | docker load
enroot import --output <PATH_TO_CONTAINERS>/nemo-skills-v1.2.0.sqsh \
  dockerd://nvflow-nemo-skills:v1.2.0
```

`enroot import` reads only `docker://`, `dockerd://`, and `podman://` sources.
Without a Docker daemon on the cluster host, load the tarball with Podman and
import it from there:

```bash
gunzip -c nvflow-nemo-skills-v1.2.0.tar.gz | podman load
enroot import --output <PATH_TO_CONTAINERS>/nemo-skills-v1.2.0.sqsh \
  podman://nvflow-nemo-skills:v1.2.0
```

Without Podman either, convert the image on a host that has Docker and Enroot,
then copy the `.sqsh` file to the cluster.

### Manual `enroot import`

- Separate the registry host with `#`: `enroot` reads `docker://<host>/<path>`
  as a Docker Hub path. `scripts/setup_containers.sh` does this for you.

  ```bash
  enroot import --output vllm-v0.29.0.sqsh \
    "docker://<registry-host>#<namespace>/nvflow-vllm:v0.29.0"
  ```

- Pass `--output`: otherwise `enroot` writes the tag's `:` into the file name.

## 4. Updating an image

### vLLM image

[`Dockerfile.vllm`](Dockerfile.vllm) installs vLLM's official CUDA 13 wheel for
the commit pinned in [`docker-bake.hcl`](../docker-bake.hcl), checked against
its SHA-256, on uv-managed Python `3.13.14`; vLLM's own `docker/Dockerfile`
installs Python from the deadsnakes PPA, which no longer publishes `3.13.14`.
The image follows the final stage of vLLM's image for the same release: its
`nvidia/cuda` base, the CUDA compiler and headers, vLLM's requirements with its
NCCL override, `flashinfer-cubin` and `flashinfer-jit-cache`, and the same
extra packages. It leaves out DeepEP, GDRCopy, the KV connectors, and the
non-root user. NVFlow adds Ray, SAM 3.1, and the offline tiktoken cache. Ray's
`cgraph` extra hardcodes CUDA 12 CuPy, so the image pairs `ray[default]` with
`cupy-cuda13x`.

For a new vLLM release, take the commit, wheel SHA-256 values, CUDA base,
`FLASHINFER_VERSION`, and NCCL override from that release, then regenerate the
[lock files](#lock-files).

### SGLang image

SGLang's own `docker/Dockerfile` hard-codes Python 3.12, so
[`Dockerfile.sglang`](Dockerfile.sglang) follows its install recipe instead:
the official `0.5.17` wheels on SGLang's
`nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04` base, Torch from the PyTorch cu130
index, `flashinfer-cubin`, and the checksum-pinned TRT-LLM MoE cubin pool, on
uv-managed Python `3.13.14` and Ray `2.58.0`.

- The extras are `http2` and `tracing`: the official `all` extra without
  `diffusion`, whose `PyYAML==6.0.1` pin has no Python 3.13 build.
- `outlines-core==0.1.26` has no Python 3.13 wheel, so a throwaway stage
  compiles it from its sdist with a pinned Rust toolchain. It is the only
  compile step in any image.
- DeepEP, MSCCL++, Mooncake, NIXL, GDRCopy, HPC-Ops, `sgl-model-gateway`, the
  FlashInfer DCP patch, `runai-model-streamer`, and the dev tools are left
  out. They serve expert-parallel MoE, prefill/decode disaggregation, decode
  context parallelism, multi-engine routing, and object-store loading, which
  NVFlow's server launchers do not use.

### Lock files

The vLLM and SGLang images install their Python packages with `--no-deps` from
per-architecture lock files, so both architectures and later rebuilds get
identical versions: `requirements/<image>-<version>-<arch>.txt`, compiled from
`requirements/<image>-<version>.in` and any upstream version overrides in
`<image>-<version>.overrides.txt`. Each lock file's header records the
`uv pip compile` command that regenerates it; rerun it after changing an `.in`
file or a pin. The command passes `--no-config` so the locks do not pick up
NVFlow's own uv overrides from `pyproject.toml`.

Each image then runs `pip check`, ignoring known reports: NVIDIA's arm64
`nvidia-cusparselt-cu13` wheel records the platform tag `manylinux2014_sbsa`,
which `pip check` calls unsupported, and in the vLLM image, vLLM's NCCL
override replaces torch's exact NCCL pin.

### A new NeMo-RL or NeMo-Gym base image

This is a maintainer check of a candidate base image. The stock
`nvcr.io/nvidia/nemo-rl` and `nvcr.io/nvidia/nemo-gym` images are only the
build bases of `nvflow-nemo-rl` and `nvflow-nemo-gym`, not worker images. Every
image that joins a Ray cluster must have NVFlow's Python and Ray versions
(`nvflow/execution/runtime_profile.py`), so check a candidate before building
on it. These commands pull the candidates, about 45 GB for NeMo-RL and 14 GB
for NeMo-Gym:

```bash
docker run --rm --entrypoint bash nvcr.io/nvidia/nemo-rl:<candidate-tag> -c '
  set -e
  /opt/nemo_rl_venv/bin/python -c "import platform, ray; assert (platform.python_version(), ray.__version__) == (\"3.13.14\", \"2.58.0\")"
  test -f /opt/nemo-rl/examples/converters/convert_dcp_to_hf.py
  for actor in models.policy.workers.automodel_policy_worker.AutomodelPolicyWorker \
      models.policy.workers.megatron_policy_worker.MegatronPolicyWorker environments.nemo_gym.NemoGym; do
    test -x "/opt/ray_venvs/nemo_rl.$actor/bin/python"
  done
  git -C /opt/nemo-rl/3rdparty/Gym-workspace/Gym rev-parse HEAD'

docker run --rm --entrypoint bash nvcr.io/nvidia/nemo-gym:<candidate-tag> -c '
  set -e
  /opt/nemo_gym_venv/bin/python -c "import platform, ray; from importlib.metadata import version; assert (platform.python_version(), ray.__version__, version(\"vllm\")) == (\"3.13.14\", \"2.58.0\", \"0.29.0\")"
  for command in run aggregate profile health-check reverify compare; do
    gym eval "$command" --help >/dev/null
  done
  echo OK'
```

NeMo-RL trains with the Gym commit it vendors, which the first command prints,
while Gym stages run the standalone release. In v0.8.0 both are Gym v0.7.0
(`1f8afc71`). When they differ, diff the resource servers between them before
attributing a training-versus-evaluation difference to the model:

```bash
git -C <gym-checkout> diff --stat <vendored-commit> <gym-release-tag> -- resources_servers/
```

Then update the bases of `nvflow-nemo-rl` and `nvflow-nemo-gym` (the
`NEMO_RL_IMAGE` and `NEMO_GYM_IMAGE` defaults in `docker-bake.hcl` and the
matching `ARG` defaults in `Dockerfile.nemo-rl` and `Dockerfile.nemo-gym`),
their tags in `docker-bake.hcl`, the expected versions in
`Dockerfile.nemo-gym`, and the tags in `cluster_configs/`,
`ci/trivy-images.txt`, and the docs.
