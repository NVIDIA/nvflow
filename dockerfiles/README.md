# NVFlow Container Images

NVFlow uses five container images, all designed to run fully offline on
air-gapped Slurm clusters.  Four are **built locally** from the
self-contained Dockerfiles in this directory; the fifth (`sglang`) is pulled
as-is from Docker Hub.  The Dockerfiles are build recipes — running
`docker build` against each one on a connected host produces the actual
images.

For complete documentation — build instructions, sanity checks, deployment
steps, air-gapped design rationale, and rebuild guidance — see
**[docker_instructions.md](docker_instructions.md)**.

## Quick Start

```bash
# Requires `docker login nvcr.io` for the NGC registry (nemo-rl base image)
docker build -f dockerfiles/Dockerfile.nemo-rl     -t nvflow-nemo-rl:v0.6.0 .
docker build -f dockerfiles/Dockerfile.nemo-skills -t nvflow-nemo-skills:0229040 .
docker build -f dockerfiles/Dockerfile.vllm        -t nvflow-vllm:v0.18.1 .
docker build -f dockerfiles/Dockerfile.vllm-grpo   -t nvflow-vllm-grpo:v0.17.1 .

# Multi-arch builds (amd64 + arm64) — push directly to a registry
REGISTRY=<your-registry>
docker buildx build --platform linux/amd64,linux/arm64 \
  -f dockerfiles/Dockerfile.vllm      -t $REGISTRY/nvflow-vllm:v0.18.1      --push .
docker buildx build --platform linux/amd64,linux/arm64 \
  -f dockerfiles/Dockerfile.vllm-grpo -t $REGISTRY/nvflow-vllm-grpo:v0.17.1 --push .

# sglang — pull directly, no custom Dockerfile needed
docker pull lmsysorg/sglang:v0.5.10.post1
```

## Images

| Image | Base | Purpose |
|-------|------|---------|
| `nvflow-nemo-rl` | `nvcr.io/nvidia/nemo-rl:v0.6.0` | SFT, GRPO training, collect_rollouts, compute_rewards |
| `nvflow-nemo-skills` | `ubuntu:22.04` | SDG pipeline, evaluation, data preparation |
| `nvflow-vllm` | `vllm/vllm-openai:v0.18.1` | Standalone vLLM inference (SDG, eval) — multi-arch (amd64 + arm64) |
| `nvflow-vllm-grpo` | `vllm/vllm-openai:v0.17.1` | Standalone vLLM inference (GRPO rollouts, judge) — multi-arch |
| `sglang` | `lmsysorg/sglang:v0.5.10.post1` | SGLang inference server (pulled as-is) |

## Version Pins

| Build Arg | Default | Where to find the right value |
|-----------|---------|-------------------------------|
| `BASE_IMAGE` (nemo-rl) | `nvcr.io/nvidia/nemo-rl:v0.6.0` | [NGC NeMo-RL tags](https://catalog.ngc.nvidia.com) |
| `NEMO_SKILLS_COMMIT` | `022904023ad7a83a87662a313cf72e7df5891d55` (`0229040`) | Should match across `Dockerfile.nemo-skills` and `Dockerfile.nemo-rl` |
| `NEMO_GYM_BRANCH` | `ude/finance-sec-search-v2` | NeMo-Gym branch with finance agent |
| `VLLM_VERSION` (vllm) | `v0.18.1` | [vLLM releases](https://github.com/vllm-project/vllm/releases) |
| `VLLM_VERSION` (vllm-grpo) | `v0.17.1` | Pinned to match NeMo-RL v0.6.0 colocated vLLM |
