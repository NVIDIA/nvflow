# Cluster Configuration Guide

This guide provides detailed documentation for all fields in the cluster configuration file (`cluster_configs/my_cluster.yaml`).

## Table of Contents

- [Overview](#overview)
- [Executor Settings](#executor-settings)
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

The cluster configuration file defines how NVFlow submits and runs jobs on your Slurm cluster. It specifies execution settings, container images, file system mounts, and environment variables needed for distributed training and inference workloads.

**Location:** `cluster_configs/my_cluster.yaml` (gitignored for security)
**Template:** `cluster_configs/template-slurm.yaml`

---

## Executor Settings

### `executor`

Specifies the execution backend for running jobs. Valid values: `slurm`, `local`

**Example:**
```yaml
executor: slurm
```

**Details:**
- `slurm` - Submit jobs to a Slurm workload manager (most common for HPC clusters)
- `local` - Run jobs locally on your machine (for testing/development)

---

## Ray Cluster Configuration

### `ray_template`

Specifies the Ray cluster initialization template. Use if Ray cluster hangs during initialization.

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
- Can be overridden per job with `extra_sbatch_args`

---

### `cpu_partition`

Dedicated partition for CPU-only jobs (e.g., container setup, data preprocessing).

**Example:**
```yaml
cpu_partition: cpu
```

**Details:**
- Used by `setup_containers.sh` to build `.sqsh` images

---

### `job_name_prefix`

Prefix added to all SLURM job names for easier identification.

**Details:**
- Helps filter jobs in `squeue` output
- Useful when multiple users share the same account
- Format: `{prefix}-{stage_name}-{jobid}`

**View your jobs:**
```bash
squeue -u $USER | grep "prefix-stage_name-jobid"
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
  - --gres=gpu:8          # Request 8 GPUs per node
```

**Common options:**
| Option | Description |
|--------|-------------|
| `--exclusive` | Exclusive node access (no sharing with other jobs) |
| `--mem=0` | Request all available memory on the node |
| `--gres=gpu:N` | Request N GPUs per node |
| `--constraint=a100` | Request specific GPU/node type |
| `--qos=high` | Set quality of service level |

**Details:**
- Arguments apply to ALL jobs submitted via this config

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
  # Required
  nemo-skills: /path/to/containers/nemo-skills.sqsh
  vllm: /path/to/containers/vllm.sqsh
  vllm-grpo: /path/to/containers/vllm-grpo.sqsh   # GRPO rollouts / judge
  sglang: /path/to/containers/sglang.sqsh
  nemo-rl: /path/to/containers/nemo-rl.sqsh       # SFT/GRPO training
  nemo-gym: /path/to/containers/nemo-gym.sqsh     # CPU Gym-only GRPO stages
```

**Details:**
- Paths generated by instructions in [INSTALL.md](../INSTALL.md#setup-containers)
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
  # - /lustre/data:/data
```

**Format:** `<host_path>:<container_path>`

> **`/workspace` holds writable data, not source code.** Recipe code and
> checked-in assets (prompts, dataset descriptors, Gym overlays) ship to workers
> via the nemo-run packaged snapshot at `/nemo_run/code` (also the job's working
> directory), so the nvflow repo is **not** mounted. Point `/workspace` at a
> dedicated writable data directory holding `/workspace/outputs/**` (stage
> outputs, checkpoints, SEC cache, eval-datasets) and `/workspace/cache/**`
> (`HF_HOME`) — not your repo checkout. On an on-cluster launcher (no
> `ssh_tunnel`), keep the launcher's cwd at the repo root so resume/skip
> detection can map `/workspace/outputs/...` back to the host outputs dir.

**Common mounts:**

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| Writable data directory | `/workspace` | Outputs, checkpoints, caches (code ships via `/nemo_run/code`) |
| Shared model storage | `/hf_models` | Pre-trained models |
| Root Lustre | `/lustre` | Access entire shared filesystem |
| Dataset directory | `/data` | Training/evaluation datasets |

### NeMo-RL / NeMo-Gym: trainer image and Gym source

SFT and GRPO `training` run on the `nvflow-nemo-rl` image, built from [`dockerfiles/Dockerfile.nemo-rl`](../dockerfiles/Dockerfile.nemo-rl). It bakes the Gym source and one venv per Gym component, so nothing is resolved at job runtime and **no Gym mount is required**.

The Gym-only GRPO stages (`prepare_data`, `prefetch_cache`, `collect_rollouts`, `compute_rewards`) run on the CPU-only `nvflow-nemo-gym` image, also with baked venvs (`&gym_install_cpu` in `base.yaml`).

Do not bind-mount Gym or NeMo-RL source over the image in production — it shadows the baked tree and invalidates the container fingerprint, forcing a runtime rebuild. To iterate on Gym source in dev mode, mount your clone at `/opt/nemo-rl/3rdparty/Gym-workspace/Gym` and leave `UV_OFFLINE` unset so the editable install can resolve. See [`docs/development/nemo-rl-gym.md`](development/nemo-rl-gym.md).

### Model-Specific Cluster Configs

Some models require additional cluster-level differences (e.g. different timeouts, partitions, or env vars). Rather than cluttering a single config with conditional logic, use separate cluster config files:

| Cluster Config | Used By | Notes |
|----------------|---------|-------|
| `my_cluster.yaml` | Qwen3, Gemma3 (dense models) | Default for all standard models |
| `my_cluster_nemotron.yaml` | Nemotron-3-Nano (MoE) | Use only if Nemotron needs different mounts/env -- the `nemo-rl` image handles MoE without a NeMo-RL source overlay |

**How it works:**
- `base.yaml` (SFT workflow) sets `cluster: my_cluster` as the default
- A model config can override with `cluster: my_cluster_nemotron`
- Keep both configs in sync when making infrastructure changes

> **Note:** No NeMo-RL or Gym source overlay is mounted by default. Nemotron-3-Nano MoE support needs no host overlay, and both `nemo-rl` and `nemo-gym` ship with Gym baked in. See the [SFT Workflow Guide](recipes/finance/workflows/04-sft.md) for the current setup.

---

## Timeouts

### `timeouts`

Sets maximum wall-time for jobs per partition.

**Example:**
```yaml
timeouts:
  batch: "04:00:00"
  cpu: "04:00:00"
```

**Format:** `"HH:MM:SS"` or `"DD-HH:MM:SS"`

**Details:**
- Prevents jobs from exceeding partition limits
- Jobs are killed if they exceed timeout
- Set based on partition policies and workload requirements

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
  # --- Air-gap enforcement (recommended; on by default in template-slurm.yaml) ---
  - HF_HUB_OFFLINE=1
  - HF_DATASETS_OFFLINE=1
  - TRANSFORMERS_OFFLINE=1
  # - UV_OFFLINE=true              # keep unset to allow runtime uv builds; set only for strict airgap
  - TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache
  - TIKTOKEN_RS_CACHE_DIR=/opt/tiktoken_cache
  - TIKTOKEN_ENCODINGS_BASE=/opt/tiktoken_cache
  # API keys (keep secret, don't commit to git!)
  - HF_TOKEN=hf_...
  - OPENAI_API_KEY=sk-...
```

#### Recommended Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `HF_HOME` | `<CONTAINER_PATH>/cache/huggingface` | **Required.** HuggingFace cache directory. Path must be visible inside the container (a mount destination or a transparently-mounted host path) |
| `NCCL_DEBUG` | `INFO` | NCCL debugging output (useful for diagnosing multi-node issues) |
| `PYTORCH_CUDA_ALLOC_CONF` | `max_split_size_mb:512` | Reduces CUDA memory fragmentation |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1` | Required for sequence parallelism in Megatron |
| `TOKENIZERS_PARALLELISM` | `false` | Disables Rayon multi-threading in the HuggingFace tokenizer Rust backend. Prevents `RuntimeError: Already borrowed` in vLLM 0.17.0 when concurrent requests trigger simultaneous mutable borrows on the tokenizer's `RefCell`. Zero performance impact (tokenization is microseconds vs. seconds for GPU inference). Standard practice across Megatron-LM, Megatron-Bridge, and NeMo-Gym |
| `VIRTUAL_ENV` | *(empty)* | Unset to prevent host virtualenv from leaking into containers |
| `VIRTUAL_ENV_PROMPT` | *(empty)* | Unset to prevent host venv prompt from leaking into containers |

#### Air-Gap Enforcement Variables

These variables prevent the runtime from making outbound network calls and from missing baked-in tokenizer encodings. `template-slurm.yaml` ships them pre-populated; leave them set in normal production.

| Variable | Value | Purpose |
|----------|-------|---------|
| `HF_HUB_OFFLINE` | `1` | Disables HuggingFace Hub network access (model + tokenizer downloads) |
| `HF_DATASETS_OFFLINE` | `1` | Disables `datasets` network access |
| `TRANSFORMERS_OFFLINE` | `1` | Disables `transformers` network access. `huggingface_hub` treats this as equivalent to `HF_HUB_OFFLINE=1` |
| `UV_OFFLINE` | *unset* | Global flag; **left unset** so components beyond the baked set can be built on demand (see [trainer image and Gym source](#nemo-rl--nemo-gym-trainer-image-and-gym-source)). All GRPO/SFT venvs are baked, so nothing is built at runtime in practice. eval / SDG / SFT never invoke `uv` |
| `TIKTOKEN_CACHE_DIR` | `/opt/tiktoken_cache` | Points `tiktoken` at the cache baked into the images |
| `TIKTOKEN_RS_CACHE_DIR` | `/opt/tiktoken_cache` | Points the Rust `tiktoken-rs` client at the cache (used by `openai_harmony`) |
| `TIKTOKEN_ENCODINGS_BASE` | `/opt/tiktoken_cache` | Required for `openai_harmony` to load `HARMONY_GPT_OSS` offline |

> **One-time connected-node stages:** A few stages (`download_sec_filings`, `create_seed_data`, eval `prepare_data`, GRPO `prepare_data` with `should_download: true`) need internet on first run to pull benchmark/seed datasets. For those submissions, **temporarily comment out** `HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE`, and `TRANSFORMERS_OFFLINE`. See [INSTALL.md → One-Time Connected-Node Stages](../INSTALL.md#one-time-connected-node-stages-datasets).

#### NeMo-RL / GRPO training venv

GRPO `training` runs on the `nemo-rl` image, which bakes Gym and one venv per Gym component. Nothing is built at runtime: NeMo-RL matches `/opt/nemo_rl_container_fingerprint` and reuses the baked venvs. Do not bind-mount Gym or NeMo-RL source over the image -- that shadows the baked tree, invalidates the fingerprint, and forces a rebuild. The Gym-only stages run on the self-contained `nvflow-nemo-gym` image, also with baked venvs.

`UV_OFFLINE` is left unset so components outside the baked set can still be built on demand. Note the consequence: a fingerprint miss will silently rebuild over the cluster proxy rather than fail, so verify airgap behaviour by checking training logs for venv-build activity, not by the job succeeding. eval / SDG / SFT never invoke `uv`.

See [trainer image and Gym source](#nemo-rl--nemo-gym-trainer-image-and-gym-source) and [`docs/development/nemo-rl-gym.md`](development/nemo-rl-gym.md).

#### API Keys (Secrets)

| Variable | Purpose | How to Get |
|----------|---------|-----------|
| `OPENAI_API_KEY` | OpenAI API access | [OpenAI Dashboard](https://platform.openai.com/api-keys) |
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
- Required if `mail_type` is set
- Can use institutional email or external email
- Multiple addresses: `user1@example.com,user2@example.com`
