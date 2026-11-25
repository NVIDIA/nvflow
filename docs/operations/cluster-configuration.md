# Cluster Configuration Guide

This guide provides detailed documentation for all fields in the cluster configuration file (`cluster_configs/my_cluster.yaml`).

## Table of Contents

- [Overview](#overview)
- [Executor Settings](#executor-settings)
- [Slurm in-allocation Ray template](#slurm-in-allocation-ray-template)
- [SSH Tunnel](#ssh-tunnel)
- [Job Directory](#job-directory)
- [Slurm Account & Partition](#slurm-account--partition)
- [Slurm Job Options](#slurm-job-options)
- [Container Paths](#container-paths)
- [Mount Points](#mount-points)
- [Timeouts](#timeouts)
- [Environment Variables](#environment-variables)
- [Email Notifications](#email-notifications)


---

## Overview

The target configuration file selects an execution engine. This document covers
direct Slurm targets: execution settings, container images, filesystem mounts,
and environment variables needed for distributed workloads. For
`executor: ray` and operator-supplied Ray Jobs endpoints, see the
[Ray execution guide](../ray/README.md).

**Location:** `cluster_configs/my_cluster.yaml` (gitignored for security)
**Template:** `cluster_configs/template-slurm.yaml`

---

## Executor Settings

### `executor`

This guide documents direct Slurm targets, so the value is `slurm`.

**Example:**
```yaml
executor: slurm
```

**Details:**
- `slurm` - Submit jobs to a Slurm workload manager (most common for HPC clusters)

---

## Slurm in-allocation Ray template

### `ray_template`

Specifies NeMo-Run's Ray initialization template inside an already submitted
Slurm job. It is unrelated to supplied Ray Jobs targets.

**Example:**
```yaml
ray_template: "ray_enroot.sub.j2"
```

**Details:**
- **Default:** `"ray.sub.j2"` (works with SLURM 24.x)
- **If Ray hangs:** Use `"ray_enroot.sub.j2"` to fix container reattachment issues
- **Symptoms:** Ray cluster hangs, workers fail to connect, error "execve(): bad interpreter"
- **Known affected:** SLURM 25.x (confirmed on 25.11.2)
- **Known working:** SLURM 24.x works without this fix

**Check if you need this:**
```bash
scontrol show config | grep SLURM_VERSION
# If Ray cluster hangs, add ray_template: "ray_enroot.sub.j2"
```

**Why this fixes the issue:**
- Old template (`ray.sub.j2`) uses `srun --container-name` for reattachment
- Some SLURM versions require `enroot exec` for container reattachment
- `ray_enroot.sub.j2` template uses the correct `enroot exec` approach

---

## SSH Tunnel

### `ssh_tunnel`

Lets a launcher outside the cluster network submit jobs and read the cluster
filesystem over SSH, so it needs no Slurm client and no cluster mounts. Leave
it out when you run `nvflow` on the cluster.

**Example:**
```yaml
ssh_tunnel:
  host: <YOUR_CLUSTER_LOGIN_NODE>
  user: <YOUR_USERNAME>
  job_dir: <PATH_TO_JOB_DIR>
  identity: <PATH_TO_SSH_KEY>
```

**Details:**
- `job_dir` must match the top-level [`job_dir`](#job_dir).
- `identity` is the private key on the launching machine.
- See the [remote launch guide](remote-launch.md) for setup and the client image.

---

## Job Directory

### `job_dir`

The directory where NeMo-Run stores job artifacts, logs, and metadata.

**Example:**
```yaml
job_dir: /homefolder/nemo-run
```

**Details:**
- Must be accessible from compute nodes

---

## Slurm Account & Partition

### `account`

Your Slurm account/project for billing and resource tracking.

**How to find:**
```bash
sacctmgr show associations user=$USER
```

**Details:**
- Controls resource allocation quotas
- May be associated with specific partitions

---

### `partition`

The default Slurm partition (queue) for GPU jobs.

**Example:**
```yaml
partition: batch
```

**How to find:**
```bash
sinfo                    # List all partitions
sinfo -p batch           # Check specific partition
```

**Details:**
- Determines available node types and time limits
- Common partition names: `batch`, `gpu`
- Jobs that request no GPUs run on `cpu_partition` instead when it is set

---

### `cpu_partition`

Dedicated partition for NVFlow-submitted CPU-only jobs such as data
preprocessing and rollout finalization.

**Example:**
```yaml
cpu_partition: cpu
```

**Details:**
- Slurm accepts a comma-separated list such as
  `cpu_short,cpu,cpu_interactive` and chooses an eligible partition.
- `setup_containers.sh` is submitted independently; override its embedded
  default with `sbatch --partition=<partition-or-list> ...`.

---

### `gpus_per_node`

GPUs on each node of `partition`.

**Example:**
```yaml
gpus_per_node: 8
```

**Details:**
- **Default:** `8`
- A stage that sets `total_gpus` runs on `total_gpus / gpus_per_node` nodes, and
  NVFlow rejects a `total_gpus` that is not a multiple of `gpus_per_node`.
- Check the value with `sinfo -p <partition> -o "%G"`.

---

### `dependency_type`

NVFlow defaults to `afterok`, so ordinary downstream jobs are not launched
after an upstream failure. No setting is required for that policy. To opt out
globally, set:

```yaml
dependency_type: afterany
```

NVFlow overrides only continuation edges to `afterany` when a stage sets
`dependent_jobs > 0`. This lets a resume attempt or its analyze job run after a
timeout while retaining fail-closed dependencies everywhere else.

---

### `job_name_prefix`

Prefix added to all SLURM job names for easier identification.

**Details:**
- Helps filter jobs in `squeue` output
- Useful when multiple users share the same account
- Job names are `<job_name_prefix><job name>`. NVFlow adds no separator and no
  job ID, so end the prefix with a separator, as in `nvflow-`.

**View your jobs:**
```bash
squeue -u $USER -o "%.18i %.60j %.8T" | grep "nvflow-"
```

---

## Slurm Job Options

### `extra_sbatch_args`

Additional arguments passed to `sbatch` for all jobs.

**Example:**
```yaml
extra_sbatch_args:
  - --exclusive           # Request exclusive node access
  - --mem=0               # Use all available memory
```

**Common options:**
| Option | Description |
|--------|-------------|
| `--exclusive` | Exclusive node access (no sharing with other jobs) |
| `--mem=0` | Request all available memory on the node |
| `--constraint=<feature>` | Request nodes with a specific feature |
| `--qos=high` | Set quality of service level |

**Details:**
- Arguments apply to ALL jobs submitted via this config, including CPU-only
  jobs on `cpu_partition`. Leave out GPU requests such as `--gres`, because
  each job sets its own GPU count, and use only constraints that the CPU
  partition's nodes also satisfy.

---

### `extra_sandbox_args`

Additional arguments for sandbox/container jobs.

**Example:**
```yaml
extra_sandbox_args:
  - --overlap
```

**Details:**
- `--overlap` - Allow job steps to overlap (useful for concurrent operations)
- Typically used for advanced scheduling scenarios

---

## Container Paths

### `containers`

Maps container names to their `.sqsh` file paths.

**Example:**
```yaml
containers:
  # The same images serve Slurm and Ray targets.
  nemo-skills: /path/to/containers/v1.2.0/nemo-skills-v1.2.0.sqsh
  vllm: /path/to/containers/v1.2.0/vllm-v0.29.0.sqsh
  nemo-rl: /path/to/containers/v1.2.0/nemo-rl-v0.8.0.sqsh
  nemo-gym: /path/to/containers/v1.2.0/nemo-gym-v0.7.0.sqsh
  # Only for workflows whose model profiles select SGLang.
  sglang: /path/to/containers/v1.2.0/sglang-v0.5.17-ray2.58.0.sqsh
```

**Details:**
- Paths generated by [Stage the images](../slurm/README.md#2-stage-the-images)
- Required containers must exist before running workflows
- Container selection is automatic based on the stage type

---

## Mount Points

### `mounts`

Maps host file system paths to container paths.

**Example:**
```yaml
mounts:
  - <CLUSTER_PATH_TO_HF_MODELS>:/hf_models   # HuggingFace models
  - <CLUSTER_PATH_TO_WORKSPACE_DATA>:/workspace   # Writable data dir (outputs + cache)
  # Add more mounts as needed:
  # - /shared/data:/data
```

**Format:** `<host_path>:<container_path>`

> **`/workspace` holds writable data, not source code.** Project code and
> checked-in assets (prompts, dataset descriptors, Gym overlays) ship to workers
> via the nemo-run packaged snapshot at `/nemo_run/code` (also the job's working
> directory), so the nvflow repo is **not** mounted. Point `/workspace` at a
> dedicated writable data directory holding `/workspace/outputs/**` (stage
> outputs, checkpoints, SEC cache, eval-data) and `/workspace/cache/**`
> (`HF_HOME`) — not your repo checkout. On an on-cluster launcher (no
> `ssh_tunnel`), keep the launcher's cwd at the repo root so resume/skip
> detection can map `/workspace/outputs/...` back to the host outputs dir.

**Common mounts:**

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| Writable data directory | `/workspace` | Outputs, checkpoints, caches (code ships via `/nemo_run/code`) |
| Shared model storage | `/hf_models` | Pre-trained models |
| Shared filesystem root | `/shared` | Access the entire shared filesystem |
| Dataset directory | `/data` | Training/evaluation datasets |

### NeMo-RL / NeMo-Gym: trainer image and Gym source

SFT and GRPO training run on `nvflow-nemo-rl:v0.8.0`, built on NeMo-RL 0.8.0.
Evaluation, DG-SDG, GRPO preparation, rollout collection, and reward
recomputation run on `nvflow-nemo-gym:v0.7.0`, built on NeMo-Gym 0.7.0. Both
images build in the Gym component environments NVFlow's workflows use and set
`UV_OFFLINE`, so jobs install nothing at runtime.

The quickstart keeps policy and judge on NVFlow-managed vLLM services;
Gym-local vLLM remains an opt-in mode.

Do not bind-mount Gym or NeMo-RL source over these images: it changes source
without changing the locked actor environments. See the
[NeMo ecosystem integration guide](../platform/integrations/nemo-ecosystem.md).

### Model-Specific Cluster Configs

Use one project-wide target unless a site has a genuine infrastructure
difference such as another partition, mount, or timeout:

| Cluster Config | Used By | Notes |
|----------------|---------|-------|
| `my_cluster.yaml` | Qwen3, Gemma3, Nemotron-3-Nano | Standard operator target |

**How it works:**
- Workflow YAML may record `cluster` metadata/defaults for programmatic callers
- CLI `plan` and `run` do not infer that value; pass
  `--cluster my_cluster` explicitly
- Create another target only for real site-specific infrastructure differences

> **Note:** No NeMo-RL or Gym source overlay is used. Nemotron-3-Nano MoE
> support is included in NeMo-RL v0.8.0.

---

## Timeouts

### `timeouts`

Sets maximum wall-time for jobs per partition.

**Example:**
```yaml
timeouts:
  batch: "04:00:00"
  cpu: "04:00:00"
  cpu_short,cpu,cpu_interactive: "04:00:00"
```

**Format:** `"HH:MM:SS"` or `"DD-HH:MM:SS"`

**Details:**
- Prevents jobs from exceeding partition limits
- Jobs are killed if they exceed timeout
- Set based on partition policies and workload requirements
- For a comma-separated partition expression, add a timeout under that exact
  combined key. NeMo-Skills performs an exact lookup.

**Check partition limits:**
```bash
sinfo -o "%P %l" | grep batch
```

---

## Environment Variables

### `env_vars`

Environment variables injected into all job containers.

> **Important:** Every path in `env_vars` must be **visible inside the container**.
> It must be either a mount destination (e.g., `/workspace`, `/hf_models`) or a
> host path that is transparently mounted (e.g., `/shared/data` when
> `- /shared:/shared` is in your `mounts` section).
>
> | Status | Example | Why |
> |--------|---------|-----|
> | Works | `HF_HOME=/workspace/cache/huggingface` | `/workspace` is a mount destination |
> | Works | `HF_HOME=/shared/cache/huggingface` | `/shared` is transparently mounted via `- /shared:/shared` |
> | Fails | `HF_HOME=/home/user/.cache/huggingface` | `/home/user` has no corresponding mount |
>
> If you see `No such file or directory` for HF cache paths, check that the path
> falls under a mount from your `mounts:` section.

**Example:**
```yaml
env_vars:
  # Infrastructure (match template-slurm.yaml order)
  - HF_HOME=<CONTAINER_PATH>/cache/huggingface
  - NCCL_DEBUG=INFO
  - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
  - CUDA_DEVICE_MAX_CONNECTIONS=1
  - TOKENIZERS_PARALLELISM=false
  - VIRTUAL_ENV=
  - VIRTUAL_ENV_PROMPT=
  # --- Mounted model/dataset offline controls ---
  - HF_HUB_OFFLINE=1
  - HF_DATASETS_OFFLINE=1
  - TRANSFORMERS_OFFLINE=1
  - TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache
  - TIKTOKEN_RS_CACHE_DIR=/opt/tiktoken_cache
  - TIKTOKEN_ENCODINGS_BASE=/opt/tiktoken_cache
  - MIN_WORKER_PORT=0              # Ray worker ports: let the kernel assign them
  - MAX_WORKER_PORT=0
  # API keys (keep secret, don't commit to git!)
  - HF_TOKEN         # bare name: forwarded from the submitting shell
  - NVIDIA_API_KEY
  - OPENAI_API_KEY
```

#### Recommended Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `HF_HOME` | `<CONTAINER_PATH>/cache/huggingface` | **Required.** HuggingFace cache directory. Path must be visible inside the container (a mount destination or a transparently-mounted host path) |
| `NCCL_DEBUG` | `INFO` | NCCL debugging output (useful for diagnosing multi-node issues) |
| `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:512` | Reduces CUDA memory fragmentation |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1` | Required for sequence parallelism in Megatron |
| `TOKENIZERS_PARALLELISM` | `false` | Disables Rayon multi-threading in the HuggingFace tokenizer Rust backend. Prevents `RuntimeError: Already borrowed` in vLLM when concurrent requests trigger simultaneous mutable borrows on the tokenizer's `RefCell`. Zero performance impact (tokenization is microseconds vs. seconds for GPU inference). Standard practice across Megatron-LM, Megatron-Bridge, and NeMo-Gym |
| `VIRTUAL_ENV` | *(empty)* | Unset to prevent host virtualenv from leaking into containers |
| `VIRTUAL_ENV_PROMPT` | *(empty)* | Unset to prevent host venv prompt from leaking into containers |
| `MIN_WORKER_PORT`, `MAX_WORKER_PORT` | `0` | Lets the kernel assign Ray's worker ports on worker nodes. A fixed range such as 7000-8000 takes ports NeMo-RL gives vLLM and fails GRPO with `EADDRINUSE`; unset falls back to NeMo-Run's 54001-54257, inside the kernel's ephemeral range |

#### Offline Asset Variables

These variables prevent model and dataset downloads when assets are already
mounted. On their own they do not let workflows run without network access.

| Variable | Value | Purpose |
|----------|-------|---------|
| `HF_HUB_OFFLINE` | `1` | Disables HuggingFace Hub network access (model + tokenizer downloads) |
| `HF_DATASETS_OFFLINE` | `1` | Disables `datasets` network access |
| `TRANSFORMERS_OFFLINE` | `1` | Disables `transformers` network access. `huggingface_hub` treats this as equivalent to `HF_HUB_OFFLINE=1` |
| `UV_OFFLINE` | *unset* | Not needed: the `nemo-rl` and `nemo-gym` images set it, and every image ships its environments built in |
| `TIKTOKEN_CACHE_DIR` | `/opt/tiktoken_cache` | Points `tiktoken` at the cache baked into the images |
| `TIKTOKEN_RS_CACHE_DIR` | `/opt/tiktoken_cache` | Points the Rust `tiktoken-rs` client at the cache (used by `openai_harmony`) |
| `TIKTOKEN_ENCODINGS_BASE` | `/opt/tiktoken_cache` | Required for `openai_harmony` to load `HARMONY_GPT_OSS` offline |

> **One-time connected-node stages:** A few stages (finance SEC `download`, `create_seed_data`, eval `eval_data`, GRPO `prepare_data` with `should_download: true`) need internet on first run to pull benchmark/seed datasets. For `create_seed_data` and `prepare_data`, **temporarily comment out** `HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE`, and `TRANSFORMERS_OFFLINE`; `eval_data` clears them itself. See [INSTALL.md → One-Time Connected-Node Stages](../../INSTALL.md#one-time-connected-node-stages-datasets).

#### NeMo-RL / GRPO training venv

Training runs on `nvflow-nemo-rl:v0.8.0`, and Gym stages run on
`nvflow-nemo-gym:v0.7.0`. Both images build in the Gym component environments
NVFlow's workflows use and set `UV_OFFLINE`, so jobs install nothing at
runtime. Do not bind-mount Gym or NeMo-RL source over the images: that changes
source without rebuilding those environments.

See [trainer image and Gym source](#nemo-rl--nemo-gym-trainer-image-and-gym-source) and the [NeMo ecosystem integration guide](../platform/integrations/nemo-ecosystem.md).

#### API Keys (Secrets)

| Variable | Purpose | How to Get |
|----------|---------|-----------|
| `NVIDIA_API_KEY` | Hosted judge endpoint used by the shipped configs — see [Hosted Model Endpoints](../hosted-endpoints.md) | Your [NVIDIA inference hub](../hosted-endpoints.md#using-the-nvidia-inference-hub) access; without it, [use OpenAI](../hosted-endpoints.md#using-openai) |
| `OPENAI_API_KEY` | Same, when the judge points at OpenAI instead | [OpenAI Dashboard](https://platform.openai.com/api-keys) |
| `HF_TOKEN` | Hugging Face model downloads | [HF Settings](https://huggingface.co/settings/tokens) |
| `WANDB_API_KEY` | Weights & Biases logging | [W&B Settings](https://wandb.ai/authorize) |

**Security notes:**
- ⚠️ **Never commit API keys to git** (config is gitignored)
- Store secrets in a secure location
- Use read-only tokens when possible
- Rotate tokens periodically

---

## Email Notifications

### `mail_type`

When to send email notifications for jobs. Valid values: `NONE`, `BEGIN`, `END`, `FAIL`, `ALL`

**Example:**
```yaml
mail_type: FAIL
```

**Options:**
- `NONE` - No emails (default)
- `BEGIN` - When job starts
- `END` - When job completes successfully
- `FAIL` - When job fails
- `ALL` - All events

---

### `mail_user`

Email address for job notifications.

**Example:**
```yaml
mail_user: user@example.com
```

**Details:**
- Optional; Slurm defaults to the submitting user
- Can use institutional email or external email
- Multiple addresses: `user1@example.com,user2@example.com`
