# NVFlow Container Images

NVFlow uses six worker images plus an optional launcher. Five
(`nemo-rl`, `nemo-gym`, `nemo-skills`, `vllm`, `vllm-grpo`) are **built locally**
from the self-contained Dockerfiles in this directory, as is the optional
`nvflow-client` launcher; only `sglang` is **pulled as-is**. The Dockerfiles are
build recipes — running `docker build` against each one on a connected host
produces the actual images.

For complete documentation — build instructions, sanity checks, deployment
steps, air-gapped design rationale, and rebuild guidance — see
**[docker_instructions.md](docker_instructions.md)**.

> **Air-gap.** All six images run fully offline, including the `training` stage:
> `nemo-rl` bakes one NeMo-Gym venv per component at build time, so nothing is
> resolved or downloaded at job runtime and no Gym source mount is needed.
> `UV_OFFLINE` is left **unset** by policy — it keeps a dev-mode escape hatch, not
> because any stage needs the network. See
> [`docs/development/nemo-rl-gym.md`](../docs/development/nemo-rl-gym.md) for the
> trainer/Gym details.

## Building

The full build commands — single-arch, multi-arch (`buildx` + QEMU), the
`sglang` pull, sanity checks, and `.sqsh` conversion — are in
**[docker_instructions.md](docker_instructions.md)** (the authoritative
build/deploy reference). The images and their version pins are below.

## Images

| Image | Base | Purpose |
|-------|------|---------|
| `nvflow-nemo-rl` | `nvcr.io/nvidia/nemo-rl:v0.7.0` | SFT and GRPO `training`; NeMo-Gym venvs baked per component |
| `nvflow-nemo-skills` | `ubuntu:22.04` | SDG pipeline, evaluation, data preparation |
| `nvflow-vllm` | `vllm/vllm-openai:v0.22.0` | Standalone vLLM inference (SDG, eval) — multi-arch (amd64 + arm64) |
| `nvflow-vllm` (`v0.20.0*` tag) | `vllm/vllm-openai:v0.20.0` | Standalone vLLM inference (GRPO rollouts, judge) — multi-arch. Same `Dockerfile.vllm`, built with `--build-arg VLLM_VERSION=v0.20.0` |
| `nvflow-nemo-gym` | `python:3.12-slim` | CPU-only Gym-only GRPO stages (prepare_data, prefetch_cache, collect_rollouts, compute_rewards); finance Gym venvs baked |
| `nvflow-client` | pinned `ubuntu:24.04` | Optional launcher: drive `nflow` over an SSH tunnel (airgap/off-cluster); Python 3.12 + CLI + venv baked |
| `sglang` | `lmsysorg/sglang:v0.5.10.post1` | SGLang inference server (pulled as-is) |

## Version Pins

| Build Arg | Default | Where to find the right value |
|-----------|---------|-------------------------------|
| `NEMO_SKILLS_COMMIT` | `e06c9b90…` (image tag `v1.1.2`) | Should match `Dockerfile.nemo-skills` and `pyproject.toml` |
| `GYM_REF` (nemo-gym, nemo-rl) | `33ef60369…` | A commit on upstream NeMo-Gym `main`; keep both images on the same one |
| `VLLM_VERSION` (vllm) | `v0.22.0` | [vLLM releases](https://github.com/vllm-project/vllm/releases) |
| `VLLM_VERSION` (vllm-grpo) | `v0.20.0` | Pinned to match NeMo-RL v0.7.0 colocated vLLM |
