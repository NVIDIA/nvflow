# NVFlow Container Images

NVFlow builds multi-architecture images for NeMo-RL, NeMo-Gym, NeMo-Skills,
vLLM, SGLang (only for workflows that select it), and the optional
`nvflow-client` launcher. Slurm and Ray use the same images, and each ships its
environments built in, so jobs install nothing at runtime. NVFlow publishes no
container images; build, verify, and stage them as described in
**[docker_instructions.md](docker_instructions.md)**.

NVFlow does not replace transitive packages solely to remediate an upstream
image; vulnerability fixes belong in the owning upstream image. The NeMo-Skills,
vLLM, and SGLang images turn off Ray's usage reporting with
`RAY_USAGE_STATS_ENABLED=0`.

## Images

| Image | Base | Purpose |
|-------|------|---------|
| `nvflow-nemo-rl:v0.8.0` | NeMo-RL `v0.8.0` release + the Gym component environments NVFlow's GRPO workflows train with | SFT and GRPO training |
| `nvflow-nemo-gym:v0.7.0` | NeMo-Gym `v0.7.0` release + the Gym component environments NVFlow's workflows use | Native Gym data plane and Gym-local vLLM |
| `nvflow-nemo-skills:v1.2.0` | `ubuntu:22.04` + NeMo-Skills `bcf059af` + Python 3.13.14/Ray 2.58.0 | SDG, evaluation, data preparation, and supplied-Ray Jobs |
| `nvflow-vllm:v0.29.0` | official vLLM `v0.29.0` CUDA 13 wheel on `nvidia/cuda:13.0.3-base-ubuntu24.04` + Python 3.13.14/Ray 2.58.0 | Standalone policy and judge inference |
| `nvflow-sglang:v0.5.17-ray2.58.0` | official SGLang `0.5.17` wheels on `nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04` + Python 3.13.14/Ray 2.58.0 | SGLang serving for workflows that select it |
| `nvflow-client:v1.2.0` | pinned `ubuntu:24.04` | Optional launcher for hosts that cannot clone or install NVFlow, built from your commit |

## vLLM

[`Dockerfile.vllm`](Dockerfile.vllm) installs vLLM's official CUDA 13 wheel for
the pinned commit, checked against the SHA-256 values in
[`docker-bake.hcl`](../docker-bake.hcl), on uv-managed Python 3.13.14,
following the final stage of vLLM's own image for the same release; vLLM's
Dockerfile installs Python from the deadsnakes PPA, which no longer publishes
3.13.14. The image adds Ray, SAM 3.1, and the offline tiktoken cache, and its
build fails unless Python, Ray, CUDA CuPy, FlashInfer, and vLLM match their
pins exactly. Python packages install from the lock files in
[`requirements/`](requirements/); see
[lock files](docker_instructions.md#lock-files).

The `vllm` runtime role uses `0.29.0`, the vLLM version NeMo-RL 0.8.0
generates with. Ray's `cgraph` extra selects `cupy-cuda12x`, so the image
installs the same Ray wheel with its Jobs dependencies plus
`cupy-cuda13x==14.0.1`, which has Python 3.13 wheels for amd64 and arm64.

## SGLang

[`Dockerfile.sglang`](Dockerfile.sglang) installs SGLang's official `0.5.17`
release wheels on Python 3.13.14 and Ray 2.58.0, on the CUDA 13.0.1 base of
SGLang's own image. SGLang's Dockerfile hard-codes Python 3.12, so the image
follows its install recipe instead of building it; see
[SGLang image](docker_instructions.md#sglang-image). NeMo-RL's own SGLang pin
(`0.5.17.dev30+g3003d70f6`) stays inside the NeMo-RL image.

## NeMo-RL and NeMo-Gym

NeMo-RL 0.8.0 generates with vLLM `0.29.0`, the version NVFlow's external
policy/judge image uses. Gym's local vLLM model server runs in its own
environment, which is not merged with NeMo-RL's; `nvflow-nemo-gym` builds it
on vLLM `0.25.1`.

[`Dockerfile.nemo-gym`](Dockerfile.nemo-gym) builds `nvflow-nemo-gym` from the
NeMo-Gym 0.7.0 release and adds the environments of the Gym components NVFlow's
workflows use, so Gym jobs install nothing at runtime.
[`Dockerfile.nemo-rl`](Dockerfile.nemo-rl) does the same for `nvflow-nemo-rl`
with the Gym components NVFlow's GRPO workflows train with, so training
installs nothing at runtime.

## Version Pins

| Build Arg | Default | Where to find the right value |
|-----------|---------|-------------------------------|
| `NEMO_SKILLS_COMMIT` | `bcf059af55c20a89f797724598f9908d126153e6` | Generated in `Dockerfile.nemo-skills` from the root `pyproject.toml` execution pin |
| `NVFLOW_PREWARM_PROJECTS` (nvflow-client) | `all` | Release images discover and bake every project; use a space-separated subset only for development images |
| `VLLM_0290_COMMIT` | `98dff2a81d747d1dba01a47f939f48c3526d4206` | [`docker-bake.hcl`](../docker-bake.hcl) |
| `VLLM_WHEEL_SHA256_AMD64` / `_ARM64` (vllm) | per vLLM version | SHA-256 of `https://wheels.vllm.ai/<commit>/vllm-<version>-cp38-abi3-manylinux_2_28_<arch>.whl`; set in [`docker-bake.hcl`](../docker-bake.hcl) |
| `FLASHINFER_VERSION` (vllm) | `0.6.18` | The `FLASHINFER_VERSION` in vLLM's own `docker/Dockerfile` for that release; must match the lock's `flashinfer-python` |
| NCCL (vllm) | `nvidia-nccl-cu13==2.30.7` | The NCCL override in vLLM's own `docker/Dockerfile`; set in [`requirements/vllm-0.29.0.overrides.txt`](requirements/vllm-0.29.0.overrides.txt) |
| `CUPY_VERSION` (vllm) | `14.0.1` (`cupy-cuda13x`) | CUDA 13 CuPy for Ray's compiled DAGs |
| `SGLANG_VERSION` (sglang) | `0.5.17` | Official SGLang release on PyPI; set in [`docker-bake.hcl`](../docker-bake.hcl) |
| `OUTLINES_CORE_VERSION`, `RUST_TOOLCHAIN` (sglang) | `0.1.26`, `1.98.1` | SGLang 0.5.17's outlines pin, compiled from its sdist because it has no Python 3.13 wheel |
| `NEMO_GYM_IMAGE` (nemo-gym) | `nvcr.io/nvidia/nemo-gym:v0.7.0` | The NeMo-Gym release `nvflow-nemo-gym` builds on; set in [`docker-bake.hcl`](../docker-bake.hcl) |
| `NEMO_RL_IMAGE` (nemo-rl) | `nvcr.io/nvidia/nemo-rl:v0.8.0` | The NeMo-RL release `nvflow-nemo-rl` builds on; set in [`docker-bake.hcl`](../docker-bake.hcl) |
