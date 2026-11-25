# Installation & Setup Guide

Setup shared by every NVFlow target: install the client, build the worker
images, and stage the models and datasets your workflows use. Once the models
and datasets are staged, the worker images run offline; a workflow needs
network access only for external services it calls, such as
[hosted model endpoints](docs/hosted-endpoints.md). The
[image checks](dockerfiles/docker_instructions.md#2-verify) confirm that the
nemo-skills, vLLM, and SGLang images load their tokenizer files with networking
disabled.

Then set up the target that runs your workflows:

- **Slurm:** NVFlow submits each step as a Slurm job. See
  [Slurm execution](docs/slurm/README.md).
- **Ray:** NVFlow runs each step as a Ray Job on clusters you already operate,
  such as KubeRay. See [Ray execution](docs/ray/README.md). NVFlow does not
  provision Ray clusters.

## Choose your client setup

`nvflow run` orchestrates submission; the heavy work runs on the target. It and
`nvflow plan` require an explicit `--cluster <target>`. Discovery, validation,
and configuration inspection remain target-independent. Two independent choices
decide how you run the client: **how you provision it**, and **how it reaches
the target**.

| You run `nvflow` on… | Provision the client | Reach the target | Guide |
|---|---|---|---|
| Cluster login/dev node (internet) | `uv sync` | direct | [README](README.md#get-started) |
| Laptop / dev box (internet) | `uv sync` | SSH tunnel | [remote launch](docs/operations/remote-launch.md) |
| Anywhere, airgapped / no install | `nvflow-client` container | SSH tunnel (Slurm); direct or SSH tunnel (Ray) | [remote launch](docs/operations/remote-launch.md) |

## 📋 What you'll do

Five steps, top to bottom. Each step below opens with a **Goal** and ends with a **✅ Done when** check so you always know where you are.

| Step | What it does | Who needs it |
|------|--------------|--------------|
| 1. [Prerequisites](#prerequisites) | Install the client | Everyone |
| 2. [Worker Images](#worker-images) | Build the worker images | Everyone |
| 3. [Download Models](#download-models) | Pre-stage the HF models your workflows use | Everyone |
| 4. [GRPO Prerequisites](#grpo-prerequisites) | Confirm what GRPO needs staged | **GRPO only — else skip** |
| 5. [Set Up Your Target](#set-up-your-target) | Configure and verify Slurm or Ray | Everyone |

> **Shortcut:** if your target can already reach the images and models, you only need Steps 1 and 5.

---

## Prerequisites

**Step 1 of 5 · Goal:** have the `nvflow` client installed.

> **Note:** This guide assumes you've already installed the client (see the [README](README.md#get-started): install `uv`, clone the repo, run `uv sync`).
>
> **Air-gapped / no internet on the install host?** Skip the local install and drive `nvflow` from the `nvflow-client` container, which you build on a connected host with `scripts/build_images.sh nvflow-client` (CLI + all project launcher environments baked in, no `uv sync`, no client internet) — see the **[remote launch guide](docs/operations/remote-launch.md)**. You still stage the worker images and models (Steps 2–3 below); only the install differs.

**✅ Done when:** `uv run nvflow --help` prints the CLI usage, or you drive `nvflow` from the `nvflow-client` container.

---

## Worker Images

**Step 2 of 5 · Goal:** be able to pull every image your workflows use.

Slurm and Ray run the same worker images, and targets refer to each by its
role. The Starter workflow needs only `nemo-skills` and `vllm`. NVFlow
publishes no container images: you build them yourself and push them to a
registry your targets can pull from.

**Build the NVFlow images.** Every worker image builds from this repository's
Dockerfiles: `nemo-rl` (SFT and GRPO training), `nemo-gym` (evaluation, rollout
collection, and Gym data preparation), `nemo-skills` (SDG, evaluation, data
preparation), `vllm` (model serving), and `sglang` (only for workflows that
select SGLang). Their environments are built in, so jobs install nothing at
runtime. On a connected Linux host with Docker and Buildx 0.28 or newer, from a
clean clone:

```bash
export REGISTRY=<registry-host>/<namespace>  # a registry your target can pull from
scripts/build_images.sh nemo-skills nemo-rl nemo-gym vllm-0-29-0  # add sglang-0-5-17 for SGLang workflows
```

The builds take hours. Each image builds for both amd64 and arm64, so an amd64
host needs arm64 emulation; the script prints the setup command if it is
missing. To build amd64 only, pass `--platform linux/amd64`: the script then
pushes each tag with an `-amd64` suffix, so use those tags when you stage the
images: in `my_containers.yaml` on Slurm, or in your Ray clusters' image
settings. Registry login, builder setup, disk space, and the image checks are
in the [image build guide](dockerfiles/docker_instructions.md).

Every Ray cluster must run Ray `2.58.0` and Python `3.13.14`, which these
images provide.

**✅ Done when:** your images are in your registry. Slurm targets then convert them to `.sqsh`
([Stage the images](docs/slurm/README.md#2-stage-the-images)); Ray clusters
run them directly ([Supply compatible clusters](docs/ray/README.md#2-supply-compatible-clusters)).

---

## Download Models

**Step 3 of 5 · Goal:** pre-download the models your chosen workflows need to your HF models directory.

> ⚠️ **Important:** Pre-download models to your cluster storage before running workflows. The Slurm target template sets `HF_HUB_OFFLINE=1`, which is also recommended for Ray clusters, so any model not already on disk will fail at job time.
>
> **Why this matters:**
> - Avoids wasting expensive GPU time on downloads
> - Prevents race conditions when multiple jobs start simultaneously
> - Large models (10-100+ GB) can take hours to download
> - Network failures during jobs cause workflow failures

### Using hf download

The commands run the Hugging Face CLI with `uvx`, so it isn't installed into your environment. Some models are gated and require authentication -- export your HuggingFace token before downloading:

```bash
export HF_TOKEN=<YOUR_HF_TOKEN>
```

Download models to your cluster's HuggingFace models directory -- only the ones your workflows use. The Starter needs one:

```bash
# Starter, and the finance quick-start policy (Qwen3-4B)
uvx --from huggingface_hub hf download Qwen/Qwen3-4B \
  --local-dir /path/to/models/hf_models/Qwen/Qwen3-4B
```

For the **finance quick start** (see [quick-start.md](projects/finance/docs/quick-start.md)), also download:

```bash
# SDG generation and eval baseline (GPT-OSS-20B)
uvx --from huggingface_hub hf download openai/gpt-oss-20b \
  --local-dir /path/to/models/hf_models/openai/gpt-oss-20b

# Eval baseline (Gemma 3 4B IT)
uvx --from huggingface_hub hf download google/gemma-3-4b-it \
  --local-dir /path/to/models/hf_models/google/gemma-3-4b-it

# Judge for evaluation and GRPO rewards (GPT-OSS-120B)
uvx --from huggingface_hub hf download openai/gpt-oss-120b \
  --local-dir /path/to/models/hf_models/openai/gpt-oss-120b
```

For **production GRPO**, also download:

```bash
# GRPO policy (Nemotron 3 Nano 30B-A3B — used in grpo/nemotron_nano_30b_a3b.yaml)
uvx --from huggingface_hub hf download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --local-dir /path/to/models/hf_models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
```

The smoke-scale MoE example `grpo/qwen3_30b_a3b.yaml` uses `Qwen/Qwen3-30B-A3B`;
download it the same way only if you run that example.

### Mount Path

Workflows read models at `/hf_models` inside the containers. A Slurm target
maps your models directory there in its `mounts`
([Create the target](docs/slurm/README.md#3-create-the-target)); Ray clusters
must mount it there themselves
([Supply compatible clusters](docs/ray/README.md#2-supply-compatible-clusters)).

### Using in Workflows

Reference models using the **container mount path** (`/hf_models`):

```yaml
stage_kwargs:
  model: /hf_models/Qwen/Qwen3-4B  # Path inside container
  server_type: vllm
```

<details>
<summary><strong>Which models does each workflow need?</strong></summary>

The demo columns are the finance quick-start workflows:
`finance:sdg/template-based-sdg-demo` (SDG), `finance:sft/qwen3_4b` (SFT),
`finance:grpo/qwen3_4b` (GRPO), and `finance:eval/demo` (Eval).

| Model | Starter | Demo SDG | Demo SFT | Demo GRPO | Demo Eval | Production GRPO |
|-------|:-------:|:--------:|:--------:|:---------:|:---------:|:---------------:|
| `Qwen/Qwen3-4B` | ✓ | | ✓ | ✓ | ✓ | |
| `openai/gpt-oss-20b` | | ✓ | | | ✓ | |
| `google/gemma-3-4b-it` | | | | | ✓ | |
| `openai/gpt-oss-120b` | | | ✓ | ✓ | ✓ | ✓ |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | | | | | | ✓ |

**Tip:** Download commonly used models once and reuse across all workflows.
</details>

### One-Time Connected-Node Stages (Datasets)

A handful of stages legitimately need internet on **first** run to pull benchmark / seed datasets from HuggingFace or SEC EDGAR. Run them on a connected node (or off-cluster) and ship the resulting artifacts to the cluster - they're reused by every subsequent run.

| Stage | Pulls from | Why |
|---|---|---|
| `download` in Finance `download/{smoke,demo,sp500}` | SEC EDGAR | Filings aren't on HF |
| `create_seed_data` in `sdg/template-based-sdg*` | HF `nogabenyoash/SecQue` | Seed dataset |
| `eval_data` in `eval/*`, and the SFT/GRPO `eval` stages if it hasn't run | HF `secque`, `financebench`; GitHub (Vals agentic sets) | Benchmark data |
| `prepare_data` in `grpo/*` | HF | Only if `should_download: true` |

For `create_seed_data` and `prepare_data`, **temporarily clear** the three HF offline flags (`HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE`, `TRANSFORMERS_OFFLINE`): in a Slurm target's `env_vars`, or in the Ray clusters' environment. Benchmark preparation clears them itself. `UV_OFFLINE` is unrelated: none of these stages invoke `uv`.

> **Note:** `huggingface_hub` interprets `TRANSFORMERS_OFFLINE=1` as `HF_HUB_OFFLINE=1`, so all three need to be off (or unset) for HF dataset pulls to succeed.

**✅ Done when:** the models for your workflow are on disk under your `hf_models` directory.

---

## GRPO Prerequisites

**Step 4 of 5 · GRPO only — skip this entire step if you're not running GRPO.**

**There are no runtime sources to clone.** Training runs in the `nemo-rl`
image and the Gym-only stages (`prepare_data`, `collect_rollouts`, `rejudge`)
in the `nemo-gym` image from [Step 2](#worker-images).

> Runtime details are covered in the [NeMo ecosystem integration guide](docs/platform/integrations/nemo-ecosystem.md).

### SEC Filing Data (for `finance_sec_search`)

The `finance_sec_search` environment answers from SEC filings, which it reads from a shared path rather than fetching live — the default `~/.cache` does **not** work inside the containers.

Both artifacts it needs are built by the finance download workflow, not by GRPO: run `finance:download/<scope>` first. See the [download workflow guide](projects/finance/docs/workflows/01-download-sec.md).

**✅ Done when:** (GRPO users) the download workflow has run for the scope you plan to train on, if you use `finance_sec_search`. Everyone else: nothing to do — move on.

---

## Set Up Your Target

**Step 5 of 5 · Goal:** a verified target to pass as `--cluster`.

| Backend | Guide | Template |
|---|---|---|
| Slurm | [Slurm execution](docs/slurm/README.md) | [`template-slurm.yaml`](cluster_configs/template-slurm.yaml) |
| Ray | [Ray execution](docs/ray/README.md) | [`template-ray.yaml`](cluster_configs/template-ray.yaml) |

`--cluster <target>` loads `cluster_configs/<target>.yaml` from the current
directory; to keep targets elsewhere, set `NVFLOW_CONFIG_DIR`.

**✅ Done when:** your backend guide's checks pass. Then head back to
[Get started](README.md#get-started) to run your first workflow.

---

## Troubleshooting

### Python 3.13.14 not found
```bash
# UV can install it for you
uv python install 3.13.14
uv sync
```

### NeMo-Skills not found
```bash
uv run nvflow sync-project --project <name>   # rebuild a project's environment
uv sync --extra execution                     # add NeMo-Skills to the checkout's own environment
```

### Container build / conversion issues
Building images, registry login, and `enroot import` quirks (the output filename and the `#` registry separator) are covered in **[docs/maintainers/containers.md](docs/maintainers/containers.md)**.

### Air-gapped runtime errors at job time
For symptoms specific to the self-sufficient runtime - `OfflineModeIsEnabled` or `LocalEntryNotFoundError`, GRPO `training` where `uv` tries to resolve or `ng_run` / `nemo_gym` is not found, `InterpolationKeyError` after mounting a Gym branch, `tiktoken` / `openai_harmony` failing offline, and `ArrowInvalid: timezone "UTC" not found` - see the [Connected and Offline Runtime](projects/finance/docs/troubleshooting.md#connected-and-offline-runtime) section in the finance troubleshooting guide.

### Slurm problems
enroot, SSH, container paths, and job submission are covered in
[Slurm execution → Troubleshooting](docs/slurm/README.md#troubleshooting).

---

## Reference

- **Build the containers**: [`docs/maintainers/containers.md`](docs/maintainers/containers.md)
- **NeMo-RL / NeMo-Gym integration boundaries**: [`docs/platform/integrations/nemo-ecosystem.md`](docs/platform/integrations/nemo-ecosystem.md)
- **NVFlow Dockerfiles**: [`dockerfiles/README.md`](dockerfiles/README.md)
- **NVFlow Self-Sufficient Build / Deploy Guide**: [`dockerfiles/docker_instructions.md`](dockerfiles/docker_instructions.md)
- **Cluster Configuration Guide**: [`docs/operations/cluster-configuration.md`](docs/operations/cluster-configuration.md)
- **Hosted Model Endpoints (judge base URL, model id, API key)**: [`docs/hosted-endpoints.md`](docs/hosted-endpoints.md)
- **NeMo-Skills**: https://github.com/NVIDIA-NeMo/Skills
- **NeMo-RL**: https://github.com/NVIDIA-NeMo/RL
- **NeMo-Gym**: https://github.com/NVIDIA-NeMo/Gym

**Need help?** Check the [Cluster Configuration Guide](docs/operations/cluster-configuration.md) or ask your team/cluster admin.
