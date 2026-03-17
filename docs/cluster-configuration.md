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
  sglang: /path/to/containers/sglang.sqsh
  nemo-rl: /path/to/containers/nemo-rl.sqsh
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
  - <CLUSTER_PATH_TO_WORKSPACE>:/workspace   # Your workspace
  # Add more mounts as needed:
  # - /lustre/data:/data
```

**Format:** `<host_path>:<container_path>`

**Common mounts:**

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| Your workspace directory | `/workspace` | Code, configs, outputs |
| Shared model storage | `/hf_models` | Pre-trained models |
| Root Lustre | `/lustre` | Access entire shared filesystem |
| Dataset directory | `/data` | Training/evaluation datasets |

### Model-Specific Cluster Configs

Some models require additional mounts not needed by others. Rather than cluttering a single config with conditional mounts, use separate cluster config files:

| Cluster Config | Used By | Extra Mounts | Notes |
|----------------|---------|-------------|-------|
| `my_cluster.yaml` | Qwen3, Gemma3 (dense models) | None | Default for all standard models |
| `my_cluster_nemotron.yaml` | Nemotron-3-Nano (MoE) | `/path/to/RL:/opt/NeMo-RL` | NeMo-RL overlay for MoE support |

**How it works:**
- `base.yaml` (SFT workflow) sets `cluster: my_cluster` as the default
- `nemotron-3-nano.yaml` overrides with `cluster: my_cluster_nemotron`
- Both configs are identical except for the NeMo-RL overlay mount
- Keep both configs in sync when making infrastructure changes

> **Tip:** The NeMo-RL overlay mount is temporary. Once MoE support is merged into the main NeMo-RL branch, `my_cluster_nemotron.yaml` can be retired. See the [SFT Workflow Guide](recipes/finance/workflows/04-sft.md#nemotron-3-nano-special-requirements) for details.

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

**Example:**
```yaml
env_vars:
  - HF_HOME=/hf_models/cache
  - NCCL_DEBUG=INFO
  - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
  - OPENAI_API_KEY=sk-...
  - HF_TOKEN=hf_...
```

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
