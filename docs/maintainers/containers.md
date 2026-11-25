# Building & Staging the Worker Images

> Audience: anyone who builds the NVFlow images or converts them to `.sqsh` for a Slurm cluster. If the `.sqsh` files are already on your cluster, you don't need this page — set the container paths in your target ([Slurm execution → Create the target](../slurm/README.md#3-create-the-target)) and continue.

NVFlow uses four worker containers, plus SGLang for workflows that select it,
converted to `.sqsh` for Slurm and run unchanged by supplied Ray. All of them
build from this repository's Dockerfiles, with their environments built in, so
jobs install nothing at runtime. See [Native Gym data plane](#native-gym-data-plane).

## Build host requirements

Building needs internet access and push access to your registry. Every NVFlow
image builds from public sources, so building needs no NGC account.

- Docker Engine or Docker Desktop.
- Buildx with an inspected-or-created `docker-container` builder.
- binfmt/QEMU when the builder does not already advertise both amd64 and arm64.
- Push access to the destination registry.

## Required containers

| Container | Source | Tested Version | Action |
|-----------|--------|----------------|--------|
| `nvflow-nemo-rl:v0.8.0` | [`dockerfiles/Dockerfile.nemo-rl`](../../dockerfiles/Dockerfile.nemo-rl) | NeMo-RL 0.8.0 + the Gym environments GRPO trains with | **Build** |
| `nvflow-nemo-gym:v0.7.0` | [`dockerfiles/Dockerfile.nemo-gym`](../../dockerfiles/Dockerfile.nemo-gym) | NeMo-Gym 0.7.0 + the Gym environments NVFlow's workflows use | **Build** |
| `nvflow-nemo-skills:v1.2.0` | [`dockerfiles/Dockerfile.nemo-skills`](../../dockerfiles/Dockerfile.nemo-skills) | NeMo-Skills @ `bcf059af`, Python 3.13.14, Ray 2.58.0 | **Build** |
| `nvflow-vllm:v0.29.0` | official CUDA 13 wheel + [`dockerfiles/Dockerfile.vllm`](../../dockerfiles/Dockerfile.vllm) | Python 3.13.14, Ray 2.58.0 | **Build** (policy/judge serving) |
| `nvflow-sglang:v0.5.17-ray2.58.0` | official wheels + [`dockerfiles/Dockerfile.sglang`](../../dockerfiles/Dockerfile.sglang) | Python 3.13.14, Ray 2.58.0 | **Build** (only for workflows that select SGLang) |

SGLang's own Dockerfile hard-codes Python 3.12, so `nvflow-sglang` installs the
official `0.5.17` wheels on NVFlow's Python 3.13.14 and Ray 2.58.0 instead; see
the [SGLang image](../../dockerfiles/docker_instructions.md#sglang-image).
NeMo-RL's own SGLang pin (`0.5.17.dev30+g3003d70f6`) stays inside the NeMo-RL
image.

`scripts/build_images.sh nemo-gym` builds `nvflow-nemo-gym:v0.7.0`: the
NeMo-Gym release with the Gym component environments NVFlow's workflows use.
`scripts/build_images.sh nemo-rl` builds `nvflow-nemo-rl:v0.8.0`, with the
environments NVFlow's GRPO workflows train with.

**Optional containers** (not currently used by any NVFlow projects):

| Container | Source | Action |
|-----------|--------|--------|
| `megatron` | NeMo-Skills Dockerfiles | Build |
| `sandbox` | NeMo-Skills Dockerfiles | Build |
| `verl` | NeMo-Skills Dockerfiles | Build |
| `trtllm` | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc8` | Pull from NGC |

### Native Gym data plane

Gym stages run on `nvflow-nemo-gym:v0.7.0`. It contains Python 3.13.14,
Ray 2.58.0, vLLM 0.29.0, Enroot, Gym at `/opt/nemo-gym`, and the environment of
every Gym component NVFlow's workflows use, so Gym jobs install nothing at
runtime.

### Launcher image (optional, airgap-only)

Separate from the **worker** containers above, the **`nvflow-client`** launcher image ([`dockerfiles/Dockerfile.nvflow`](../../dockerfiles/Dockerfile.nvflow), pinned Ubuntu 24.04 base with uv-managed Python 3.13.14) bundles the `nvflow` CLI and project launcher environments so **users in an airgapped environment who cannot run `uv sync`** can drive NVFlow over an `ssh_tunnel`. It is a **launcher, not a worker**: it is *not* referenced by `my_cluster.yaml` `containers:` and is *not* required for a normal root install. Release builds discover, prewarm, and verify every project environment by default; `NVFLOW_PREWARM_PROJECTS` subsets are development-only. Build it **multi-arch (amd64 + arm64)** from a clean, committed tree; `scripts/build_images.sh` tags it `nvflow-client:v1.2.0` (set `NVFLOW_TAG` to build under another tag). See the [remote launch guide](../operations/remote-launch.md) for usage.

## Step 1: Build Docker Images

NVFlow ships version-controlled build definitions that pre-install all Python
packages, pre-cache tokenizer encodings, and pre-build virtual environments.
The Buildx setup, dual-architecture builds, checks, and staging commands are in
**[`dockerfiles/docker_instructions.md` §1](../../dockerfiles/docker_instructions.md#1-build)**
(the authoritative build reference); per-image version pins are in
[`dockerfiles/README.md`](../../dockerfiles/README.md#version-pins).

[`docker-bake.hcl`](../../docker-bake.hcl) builds the vLLM image from vLLM's
official CUDA 13 wheel for the pinned commit, on uv-managed Python 3.13.14 and
following the final stage of vLLM's own image, because vLLM's Dockerfile
installs Python from the deadsnakes PPA, which no longer publishes 3.13.14.
Do not replace that build with a published Python 3.12 image.

> **Tip:** `pyproject.toml` is the NeMo-Skills version source of truth. Run `uv run python scripts/check_project_environment_policy.py --write` after changing it to regenerate the managed-environment requirements and `Dockerfile.nemo-skills` pin. For optional containers (`megatron`, `sandbox`, `verl`), build them from the matching upstream [NeMo-Skills Dockerfiles](https://github.com/NVIDIA-NeMo/Skills/tree/bcf059af55c20a89f797724598f9908d126153e6/dockerfiles).

### Audit pins against sibling clones

When auditing pins, clone or refresh [NeMo-RL](https://github.com/NVIDIA-NeMo/RL),
[NeMo-Gym](https://github.com/NVIDIA-NeMo/Gym), and
[NeMo-Skills](https://github.com/NVIDIA-NeMo/Skills) next to this repository as
`../nemo-rl`, `../nemo-gym`, and `../nemo-skills`, and check that they hold the
pinned commits. These checks read local Git objects; they do not fetch source
or build images:

```bash
git -C ../nemo-rl cat-file -e 'v0.8.0^{commit}'
test "$(git -C ../nemo-gym rev-parse 'v0.7.0^{commit}')" = \
  1f8afc71d45fac14872d86fe300374afa2001e7a
git -C ../nemo-skills cat-file -e 'bcf059af55c20a89f797724598f9908d126153e6^{commit}'
```

### Step 1b: Sanity-Check Images Before Conversion

Before workflow testing, run the checks in
[`dockerfiles/docker_instructions.md` §2](../../dockerfiles/docker_instructions.md#2-verify):
offline tokenizer files and the launcher's commit. The checks each build runs
itself are listed at the start of that section.

### Vulnerability and SBOM scanning

Repository/package scans and container-image scans cover different risks.
Scanning the repository dependency manifests finds vulnerable declared
dependencies before an image is built. Scanning each built image also covers
the operating-system packages and language packages that were actually shipped.

The commands below are CI-provider-neutral and can also be reproduced locally.
Automated workflows use the same Trivy operations with deployment-specific
registry credentials and image references.

#### Scan repository dependencies

From the repository root:

```bash
# Archive all detected severities.
trivy fs --scanners vuln --include-dev-deps \
  --format json --output trivy-repository.json --exit-code 0 .

# Fail for fixable HIGH or CRITICAL findings.
trivy fs --scanners vuln --include-dev-deps \
  --severity HIGH,CRITICAL --ignore-unfixed --exit-code 1 .
```

#### Scan one container image

Authenticate using the registry's recommended credential mechanism, then select
the target platform explicitly. A multi-architecture tag resolves to a distinct
child manifest and filesystem for each platform.

```bash
IMG=registry.example.com/team/image:tag
PLATFORM=linux/amd64  # Repeat with linux/arm64 when both are deployed.
ARCH=${PLATFORM#linux/}

# Archive all detected severities.
trivy image --image-src remote --scanners vuln --timeout 90m --platform "$PLATFORM" \
  --format json --output "trivy-image-${ARCH}.json" "$IMG"

# Fail for fixable HIGH or CRITICAL findings.
trivy image --image-src remote --scanners vuln --timeout 90m --platform "$PLATFORM" \
  --table-mode detailed \
  --severity HIGH,CRITICAL --ignore-unfixed --exit-code 1 "$IMG"
```

Trivy pulls directly from the registry and does not require a Docker daemon.

#### Promotion scan

The promotion scan fails on any fixable HIGH or CRITICAL finding. Its CI job
runs with `TRIVY_IMAGE_PROMOTION=true` and the two platform child digests of
one repository, `TRIVY_IMAGE_AMD64=<repository>@sha256:<amd64-child-digest>`
and `TRIVY_IMAGE_ARM64=<repository>@sha256:<arm64-child-digest>`, and scans
each platform independently. Both platform scans must pass before a digest is
promoted or a release tag advances; a mutable multi-architecture tag alone is
not sufficient evidence.

#### Generate and rescan an SBOM

Generate one CycloneDX SBOM from each immutable platform child digest rather
than from a mutable tag alone:

```bash
IMAGE_DIGEST='sha256:<replace-with-platform-manifest-digest>'
IMAGE="registry.example.com/team/image@$IMAGE_DIGEST"
PLATFORM=linux/amd64
ARCH=${PLATFORM#linux/}
DIGEST_HEX=${IMAGE_DIGEST#sha256:}
SBOM_FILE="container-${ARCH}-${DIGEST_HEX}.cdx.json"
SCAN_FILE="trivy-sbom-scan-${ARCH}-${DIGEST_HEX}.json"

trivy image --image-src remote --platform "$PLATFORM" \
  --format cyclonedx --output "$SBOM_FILE" "$IMAGE"

# Rescan the small inventory against the current vulnerability database.
trivy sbom --scanners vuln \
  --format json --output "$SCAN_FILE" "$SBOM_FILE"
trivy sbom --scanners vuln \
  --severity HIGH,CRITICAL --ignore-unfixed --exit-code 1 \
  "$SBOM_FILE"
```

For provenance, retain the image reference, platform, child-manifest digest,
Trivy version, and SBOM SHA-256 alongside the SBOM. A checksum establishes file
integrity but not authorship. Calling an SBOM *verified* additionally requires a
trusted signature or attestation that binds that SBOM hash to the exact image
digest; verification proves that binding and signer identity, not that every
possible component was detected.

If the registry already publishes an SBOM for each child-manifest digest,
retrieve and rescan those immutable artifacts instead of generating duplicate
inventories. Confirm that amd64 and arm64 resolve to distinct SBOMs, retain the
registry manifest and blob digests, and verify any available signature or
attestation. Keep a direct image scan at build or promotion time because an
SBOM rescan can evaluate only the components recorded by its generator.

References:

- [Install Trivy](https://trivy.dev/docs/latest/getting-started/installation/)
- [Trivy container-image scanning](https://trivy.dev/docs/latest/target/container_image/)
- [Trivy SBOM generation and scanning](https://trivy.dev/docs/latest/guide/target/sbom/)
- [Sigstore attestation verification](https://docs.sigstore.dev/cosign/verifying/verify/)

## Step 2: Get Images onto the Cluster

Slurm nodes usually have no Docker, so Enroot pulls each image from a registry
(`docker://`, recommended) or loads a custom image from a saved tarball
(`dockerd://`). Commands for both paths are in
[`dockerfiles/docker_instructions.md` §3](../../dockerfiles/docker_instructions.md#3-stage-the-images-on-slurm).

## Step 3: Update Container Config

Copy the template to a personal file that records the registry / tag references the cluster should pull from:

```bash
cp cluster_configs/containers.yaml cluster_configs/my_containers.yaml
```

Edit `cluster_configs/my_containers.yaml` with your registry paths, and delete
the entries for images you did not build or cannot pull; a missing image fails
the conversion job. The YAML keys match workflow runtime roles and are shared by
Slurm and Ray:

```yaml
containers:
  nemo-rl:     your-registry/nvflow-nemo-rl:v0.8.0
  nemo-skills: your-registry/nvflow-nemo-skills:v1.2.0
  nemo-gym:    your-registry/nvflow-nemo-gym:v0.7.0   # eval, DG-SDG, GRPO prepare_data, SEC metadata prefetch
  vllm:        your-registry/nvflow-vllm:v0.29.0      # shared policy/judge server
  sglang:      your-registry/nvflow-sglang:v0.5.17-ray2.58.0    # workflows that select SGLang
```

> **Note:** `my_containers.yaml` is gitignored (`cluster_configs/*.yaml` pattern), so your registry paths stay local and won't be committed.

## Step 4: Convert to .sqsh Format

If the cluster authenticates to your registry, drop credentials into `~/.config/enroot/.credentials` first:

```
machine <your-registry-host> login <user> password <token>
```

### Option A: Automated Setup (Recommended)

Use the setup script to download from your registry and convert all containers in parallel. Pass your personal config with `--config`:

```bash
# Run from a cluster login node (sbatch requires Slurm access)
mkdir -p outputs/logs
sbatch --account=<YOUR_SLURM_ACCOUNT> --partition=<CPU_PARTITION> \
  scripts/setup_containers.sh \
  --config cluster_configs/my_containers.yaml \
  <CONTAINER_DIR>/v1.2.0 \
  --platform amd64
```

The `--config` flag is required - the script reads image references from the specified YAML file, pulls them via `enroot`, and converts to `.sqsh` format. See [the script](../../scripts/setup_containers.sh) for additional options (`--platform`, `--force`).

> Output filenames are derived as `<key>-<tag>.sqsh` from the YAML key and tag (not the registry path), and any image whose file already exists is skipped — pass `--force` to re-download.

**Check progress:**
```bash
tail -f outputs/logs/slurm-containers-<jobid>.out
```

### Option B: Manual Conversion

Convert images one at a time using `enroot` on a cluster node. From a registry, use `docker://<registry-host>#<namespace>/...`; from a loaded tarball, use `dockerd://...` after `docker load`:

```bash
CONTAINER_DIR=<absolute path on cluster where .sqsh files should live>

# Use <key>-<tag>.sqsh so manual imports and setup_containers.sh agree.
enroot import --output $CONTAINER_DIR/nemo-skills-v1.2.0.sqsh \
  "docker://<registry-host>#<namespace>/nvflow-nemo-skills:v1.2.0"  # from a registry
# -- or --
gunzip -c nvflow-nemo-skills-v1.2.0.tar.gz | docker load
enroot import --output $CONTAINER_DIR/nemo-skills-v1.2.0.sqsh \
  dockerd://nvflow-nemo-skills:v1.2.0                 # from a tarball
```

Repeat for `vllm`, `nemo-gym`, `nemo-rl`, and, for workflows that select SGLang, `sglang`.

**Two things to watch for:**

- **Separate the registry host with `#`.** `enroot` reads `docker://<host>/<path>` as a Docker Hub path, so a registry host must be followed by `#`, not `/`. `scripts/setup_containers.sh` does this for you; manual imports need it:
  ```bash
  enroot import --output vllm-v0.29.0.sqsh \
    "docker://<registry-host>#<namespace>/nvflow-vllm:v0.29.0"
  ```
- **Filename colon.** `enroot` writes the Docker tag separator (`:`) literally into the output filename. Either pass `--output` with a shell-safe name (as above) or rename after import:
  ```bash
    mv "nvflow-nemo-skills:v1.2.0.sqsh" nemo-skills-v1.2.0.sqsh
  ```

Move the resulting `.sqsh` files to your cluster's container storage path, then record those paths in your cluster config.
