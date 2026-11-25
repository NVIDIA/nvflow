# Finance Recipe Troubleshooting

Comprehensive troubleshooting guide for common issues across all finance project's workflows.
Commands read outputs through `$WORKSPACE`, the host directory your target mounts at `/workspace`.

## Quick Navigation

- [Cluster & Infrastructure](#cluster--infrastructure)
- [Connected and Offline Runtime](#connected-and-offline-runtime)
- [Resource Issues](#resource-issues)
- [Data Issues](#data-issues)
- [Training Issues](#training-issues)
- [Workflow-Specific Issues](#workflow-specific-issues)
- [Resuming Interrupted Workflows](#resuming-interrupted-workflows)
- [Frequently Asked Questions](#frequently-asked-questions)
- [Getting Additional Help](#getting-additional-help)

---

## Cluster & Infrastructure

### Slurm Job Failures

**Problem:** Jobs fail to submit or don't execute

**Solutions:**
```bash
# 1. Find the target directory: cluster_configs/ unless NVFLOW_CONFIG_DIR is set
echo "${NVFLOW_CONFIG_DIR:-cluster_configs}"

# 2. Check cluster config file
cat "${NVFLOW_CONFIG_DIR:-cluster_configs}/my_cluster.yaml"

# 3. Verify partition names match your cluster
sinfo  # List available partitions

# 4. Check the container image paths (see Container Image Issues below)
grep -A8 '^containers:' "${NVFLOW_CONFIG_DIR:-cluster_configs}/my_cluster.yaml"
```

**Common causes:**
- `my_cluster.yaml` is not in `cluster_configs/`, or `NVFLOW_CONFIG_DIR` points elsewhere
- Partition names don't match cluster
- A `containers:` entry that points at a missing `.sqsh` image
- Resource limits exceeded

### Container Image Issues

**Problem:** A job cannot start its container, or reports that an image path does not exist

**Solutions:** NVFlow's worker images are built from `dockerfiles/`. On Slurm
they are staged as `.sqsh` files that the cluster config's `containers:` entries
point at, and jobs do not pull them from a registry.

```bash
# 1. List the image paths the target uses
grep -A8 '^containers:' "${NVFLOW_CONFIG_DIR:-cluster_configs}/my_cluster.yaml"

# 2. Confirm every listed .sqsh file exists on the cluster filesystem
ls -l <PATH_TO_CONTAINERS>/*.sqsh
```

Each role needs the image built for this release: `nvflow-nemo-skills:v1.2.0`,
`nvflow-vllm:v0.29.0`, `nvflow-nemo-rl:v0.8.0`, `nvflow-nemo-gym:v0.7.0`, and
`nvflow-sglang:v0.5.17-ray2.58.0` for workflows that select SGLang. To build
and stage them, see [`dockerfiles/docker_instructions.md`](../../../dockerfiles/docker_instructions.md).

### Network/Connectivity Issues

**Problem:** Jobs can't reach external services

**Solutions:**
- Verify cluster has internet access
- Check firewall rules for SEC EDGAR (sec.gov)
- Ensure model repositories are accessible

### Ray Cluster Initialization Issues (SLURM 25.11.2+)

**Problem:** Training jobs hang during Ray cluster initialization

**Symptoms:**
```
Starting Ray cluster...
[Workers not connecting]
Error: execve(): bad interpreter(/opt/nemo_rl_venv/bin/python3): No such file or directory
```

**Root cause:** SLURM container reattachment issues (SLURM 25.x may be affected)

**Solution:**
```yaml
# Add to cluster_configs/my_cluster.yaml
executor: slurm
ray_template: "ray_enroot.sub.j2"
```

**Verify your SLURM version:**
```bash
scontrol show config | grep SLURM_VERSION
# Confirmed: SLURM 25.11.2 needs this fix, SLURM 24.x works without it
```

This changes how containers are launched, using `enroot exec` instead of `--container-name`.

---

## Connected and Offline Runtime

NVFlow's `nemo-rl` and `nemo-gym` images build in every Gym component
environment NVFlow's workflows use and set `UV_OFFLINE`, so Gym and training
jobs install nothing at runtime. The NeMo-Skills, vLLM, and SGLang images load
their tokenizer files offline.

For the full build / deploy / verify flow, see [INSTALL.md](../../../INSTALL.md) and [`dockerfiles/docker_instructions.md`](../../../dockerfiles/docker_instructions.md).

### `huggingface_hub.errors.OfflineModeIsEnabled` / `LocalEntryNotFoundError`

**Problem:** A stage fails trying to pull a model or dataset from HuggingFace Hub.

**Cause:** Air-gap mode is on (`HF_HUB_OFFLINE=1`, etc.) but the asset isn't pre-staged on disk.

**Solution:**
- **Models:** Pre-download to your mounted `hf_models` directory with `hf download` -- see [INSTALL.md → Download Models](../../../INSTALL.md#download-models).
- **Datasets / SEC filings:** Some stages (finance SEC `download`, `create_seed_data`, eval `eval_data`, GRPO `prepare_data` with `should_download: true`) need internet on first run. Run them on a connected node. For `create_seed_data` and `prepare_data`, also **temporarily clear** the three `HF_*_OFFLINE` flags where your backend sets them ([Slurm](../../../docs/slurm/README.md#where-settings-live), [Ray](../../../docs/ray/README.md#where-settings-live)); `eval_data` clears them itself. The artifacts persist under `/workspace` and are reused by every subsequent run.

### GRPO `training` fails: `uv` tries to resolve, or `ng_run` / `nemo_gym` not found

**Problem:** The `training` stage fails soon after start with `uv` trying to download packages, a hung resolution, or a missing Gym module.

**Cause:** `nvflow-nemo-rl` bakes its actor environments and the environment
of every Gym component NVFlow's GRPO workflows train with, and sets
`UV_OFFLINE`. A download attempt means the job is not running that image, a
source mount has shadowed the image checkout, or the workflow trains with a Gym
component the image does not include.

**Solution:**
1. Confirm `containers.nemo-rl` points at the `.sqsh` staged from
   `nvflow-nemo-rl:v0.8.0`.
2. Remove any Gym or NeMo-RL source mount from the `mounts:` block.
3. For a Gym component the image does not include, add it to the components
   `dockerfiles/Dockerfile.nemo-rl` builds in, and rebuild the image.

The Gym stages (`collect_rollouts`, `rejudge`, `prefetch_cache`,
`prepare_data`, and evaluation) run on `nvflow-nemo-gym:v0.7.0`. If `gym` is
missing, verify the image path and `/opt/nemo_gym_venv/bin` in the
[container maintainer guide](../../../docs/maintainers/containers.md).

### `omegaconf.errors.InterpolationKeyError: Interpolation key '<name>' not found` after mounting a Gym branch

**Problem:** A `training` or `ng_run`-driven job fails at NeMo-Gym config-load time with, e.g.:

```
omegaconf.errors.InterpolationKeyError: Interpolation key 'tavily_api_key' not found
    full_key: tavily_api_key
    object_type=dict
```

**Cause:** The mounted Gym source introduced a new `${<name>}` interpolation
that the workflow's inline Gym configuration does not define. This is source
drift, not necessarily a new runtime requirement; for example, an empty
`tavily_api_key` disables Tavily `web_search` gracefully.

**Solution (clean, no upstream change):** Add the missing value under the
relevant `environments.<name>.gym` mapping (or `benchmarks.<name>.gym` for
evaluation):

```yaml
# Required since upstream Gym introduced ${tavily_api_key} in finance_sec_search.yaml.
# Empty string disables tavily gracefully -- finance_sec_search uses SEC tools only.
tavily_api_key: ""
```

Restart the job; OmegaConf will resolve the interpolation against the overlay value and the resource server will log `No tavily_api_key configured — web_search will be unavailable` and continue.

For another key, identify the Gym component YAML that introduced the
interpolation and add the smallest matching inline override. Finance eval and
GRPO endpoint settings do not use separate maintained overlay YAMLs.

### `tiktoken` / `openai_harmony` fails to load offline

**Problem:** An SDG, eval, or GRPO stage crashes with `tiktoken` / `openai_harmony` trying to fetch encoding files from `openaipublic.blob.core.windows.net`.

**Cause:** The variables that point at the pre-cached encodings are not set inside the container. The `vllm` and `sglang` images set all three in their image `ENV`; `nemo-skills` sets only `TIKTOKEN_CACHE_DIR`, so the cluster config supplies the other two.

**Solution:** Confirm `cluster_configs/my_cluster.yaml` has all three set:

```yaml
env_vars:
  - TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache
  - TIKTOKEN_RS_CACHE_DIR=/opt/tiktoken_cache
  - TIKTOKEN_ENCODINGS_BASE=/opt/tiktoken_cache
```

Verify the cache exists inside the image:
```bash
docker run --rm --entrypoint ls nvflow-vllm:v0.29.0 /opt/tiktoken_cache
# Expect: cl100k_base.tiktoken  o200k_base.tiktoken
```

`nvflow-nemo-skills` caches only `cl100k_base`, under tiktoken's hashed file
name (`9b5ad71b2ce5302211f9c61530b329a4922fc6a4`).

### `pyarrow` / `pandas` blow up with `ArrowInvalid: timezone "UTC" not found`

**Problem:** SEC data prep fails on first record with a timezone-related error.

**Cause:** `tzdata` is not populated in the image.

**Solution:** Already fixed in `Dockerfile.nemo-skills` (apt `tzdata`). If you see this in a custom-built image, confirm `tzdata` is installed:
```bash
docker run --rm nvflow-nemo-skills:v1.2.0 bash -c \
  'python3 -c "import pyarrow as pa; pa.array([], type=pa.timestamp(\"ns\", tz=\"UTC\")); print(\"OK\")"'
```

---

## Resource Issues

### Out of Memory (OOM) Errors

**Problem:** GPU runs out of memory during training or inference

**Symptoms:**
```
RuntimeError: CUDA out of memory
torch.cuda.OutOfMemoryError
```

**Solutions:**

**For Training (SFT):** NeMo-RL settings go under the stage's `overrides`. The
global batch stays the effective batch when the micro batch shrinks:
```yaml
stages:
  training:
    overrides:
      policy:
        train_micro_batch_size: 1
        megatron_cfg:                    # Megatron backend, as in sft/qwen3_14b.yaml
          activation_checkpointing: true
          context_parallel_size: 8       # split long sequences across more GPUs
```

**For Inference (SDG workflows):** lower the generation length or the number of
requests in flight in the failing stage's `inline_args`. The value replaces the
whole string, so keep the stage's other arguments:
```yaml
stages:
  <stage>:
    inline_args: "++inference.tokens_to_generate=8192 ++max_concurrent_requests=64"
```

**Advanced solutions:**
- Reduce model size (14B instead of 32B)

### Disk Space Issues

**Problem:** Running out of storage

**Solutions:**
```bash
# Check disk usage
df -h /workspace

# Clean up old outputs
rm -rf /workspace/outputs/old_runs/

# Remove intermediate checkpoints
# Keep only final checkpoint
```

**Prevention:**
- Keep fewer checkpoints with `checkpointing.keep_top_k` (see [Checkpoint Saving Failures](#checkpoint-saving-failures))
- Monitor disk space before long runs

### Insufficient GPUs

**Problem:** Requested GPUs not available

**Solutions:**
```bash
# Check available GPUs
sinfo -p your_partition

# Adjust total_gpus in config
# Example: Use 16 GPUs instead of 256
total_gpus: 16
```

---

## Data Issues

### Missing sec_metadata.parquet

**Problem:** Template-based SDG fails with missing metadata

**Error message:**
```
FileNotFoundError: sec_metadata.parquet not found
```

**Solutions:**
```bash
# 1. Verify the download completed
ls $WORKSPACE/outputs/finance/demo/workflow-2-download-sec/step-0-download/sec_metadata.parquet

# 2. Check filings directory structure
ls $WORKSPACE/outputs/finance/demo/workflow-2-download-sec/step-0-download/data/
# Should show company directories (NVDA/, AAPL/, etc.)

# 3. Re-run the demo download if needed
uv run nvflow run --all --workflow finance:download/demo --cluster my_cluster
```

### SEC Identity Not Set

**Problem:** A `finance:download/*` command exits before submitting, with
`Environment variable 'SEC_IDENTITY_EMAIL' not found` (from `run`) or
`Workflow planning failed before configuration could be loaded` (from `plan`).
Only the first unset variable is named, so fixing one can surface the other.

**Solution:** Export both in the launching shell, then re-run:

```bash
export SEC_IDENTITY_EMAIL=your.email@company.com SEC_IDENTITY_COMPANY="Your Company"
```

### SEC Download Rate Limits

**Problem:** SEC EDGAR rate limiting (403 errors)

**Solutions:**
- Wait 10 minutes before retrying
- Verify `SEC_IDENTITY_EMAIL` and `SEC_IDENTITY_COMPANY` are valid
- Start with `demo.yaml` (7 companies) to test
- Check SEC's rate limit policy (10 requests per second)

**Launcher environment:**
```bash
export SEC_IDENTITY_EMAIL=your-email@company.com
export SEC_IDENTITY_COMPANY="Your Company Name"
```

### Data Format Issues

**Problem:** Invalid JSON or corrupted data files

**Solutions:**
```bash
# Validate JSONL format
uv run python -c "import jsonlines; list(jsonlines.open('file.jsonl'))"

# Check for empty files
wc -l output.jsonl

# Inspect sample records
head -1 output.jsonl | uv run python -m json.tool
```

---

## Training Issues

Check for training logs in any file of the form `ray-<jobid>-job.log` in the output folder set in your config

### Training failed due to `wandb.errors.errors.UsageError: api_key not configured` error

**Problem:** In your `ray-<jobid>-job.log` file you see the above error

**Solution:**
- You have a wandb api key but it is not configured correctly, set the value to env var `WANDB_API_KEY`
- You do not have a wandb api key - run with wandb mode `disabled`, see [Weights & Biases (wandb)](workflows/04-sft.md#weights--biases-wandb)


### Training Loss Not Decreasing

**Problem:** Model not learning, loss plateaus

**Symptoms:**
- Loss stays constant across steps
- Validation loss doesn't improve
- Model outputs are repetitive

**Solutions:**

**Check data quality:**
```bash
# Verify training data size
wc -l $WORKSPACE/outputs/finance/sp500/workflow-4-sft/qwen3_14b/step-2-train-validation-split/train.jsonl

# Inspect sample
head -1 $WORKSPACE/outputs/finance/sp500/workflow-4-sft/qwen3_14b/step-2-train-validation-split/train.jsonl | uv run python -m json.tool
```

**Adjust hyperparameters** (Megatron backend, as in `sft/qwen3_14b.yaml`):
```yaml
stages:
  training:
    overrides:
      sft:
        max_num_epochs: 5          # Train longer
      policy:
        megatron_cfg:
          optimizer:
            lr: 2e-6               # Try lower (was 5e-6)
          scheduler:
            lr_warmup_iters: 4000  # Increase warmup
```

**Check for issues:**
- Data contains only duplicates
- Learning rate too high or too low
- Batch size too small
- Data format incorrect

### Checkpoint Saving Failures

**Problem:** Checkpoints fail to save

**Solutions:**
```bash
# 1. Check disk space
df -h /workspace

# 2. Verify output directory permissions
ls -ld /workspace/outputs/

# 3. Reduce checkpoint frequency if space limited
```

**Config adjustment:**
```yaml
stages:
  training:
    overrides:
      checkpointing:
        save_period: 500  # Increase from 100
        keep_top_k: 3     # Keep fewer checkpoints
```

`keep_top_k` keeps the checkpoints with the best validation metric plus the
latest one, so it can delete a step that `eval_steps` evaluates after training.
To save often without that risk, set `keep_top_k: null` with `save_period` on
the eval steps, and use `ft_save_period` with `ft_keep_latest_k: 1` for
crash-recovery saves (see `grpo/nemotron_nano_30b_a3b.yaml`). Every
`save_period` checkpoint then stays on disk, so size `save_period` for the
whole run.

### Model Loading Errors

**Problem:** Cannot load model or checkpoint

**Solutions:**
```bash
# Verify model path
ls /hf_models/Qwen/Qwen3-14B/

# Check the Hugging Face export of a checkpoint
ls $WORKSPACE/outputs/.../hf_models/step_<N>/
# Should contain: config.json, model weights, tokenizer files
```

---

## Workflow-Specific Issues

### Download-SEC Workflow

**Common issues:**
- Rate limiting: See [SEC Download Rate Limits](#sec-download-rate-limits)
- Invalid tickers: Verify company symbols in config

**Quick check:**
```bash
# Test with the seven-company demo workflow
uv run nvflow run --all --workflow finance:download/demo --cluster my_cluster
```

### Template-Based SDG

**Common issues:**
- Missing metadata: See [Missing sec_metadata.parquet](#missing-sec_metadataparquet)
- OOM errors: See [Out of Memory (OOM) Errors](#out-of-memory-oom-errors)
- Slow generation: Check GPU utilization

**Tips:**
- Use `skip_filled: true` to resume interrupted runs
- Reduce `num_random_seeds` if too slow
- Check `server_gpus` and `server_nodes` in config

### Document-Grounded SDG

**Common issues:**
- OOM errors: See [Out of Memory (OOM) Errors](#out-of-memory-oom-errors)
- Long runtime: Expected ~30 hours for S&P 500, depending on the resources available
- Verification failures: Check verification model

**Note:** The S&P 500 run produces ~800K pairs in `final_result.jsonl`, which
the production GRPO workflow trains on.

### SFT Workflow

**Common issues:**
- OOM errors: See [Out of Memory (OOM) Errors](#out-of-memory-oom-errors)
- Loss not decreasing: See [Training Loss Not Decreasing](#training-loss-not-decreasing)
- Checkpoint failures: See [Checkpoint Saving Failures](#checkpoint-saving-failures)

**Quick checks:**
```bash
# Verify input data
wc -l $WORKSPACE/outputs/finance/sp500/workflow-3-template-based-sdg/step-5-filter-answers/final_result.jsonl

# Monitor training
nvidia-smi -l 1  # Watch GPU utilization
```

### Evaluation Workflow

**Common issues:**
- Model loading errors: Verify checkpoint paths
- Benchmark data missing: Check eval data preparation
- OOM during inference: Reduce batch size

---

## Resuming Interrupted Workflows

Most workflows support resuming from where they stopped:

### Download-SEC
```bash
# Automatically skips already downloaded filings
# Just re-run the same command
uv run nvflow run --all --workflow finance:download/sp500 --cluster my_cluster
```

### SDG Workflows
```yaml
# Enable skip_filled in config
stages:
  your_stage:
    stage_kwargs:
      skip_filled: true  # Skip already processed files
```

### Training (SFT)
```yaml
# Resume from latest checkpoint
training:
  dependent_jobs: 1  # Will continue from last checkpoint
```

### Rollouts or evaluation refuse to reuse earlier output
```
refusing to reuse <path>: shard identity differs in rollout_config_sha256
<path>: rollout configuration changed
```
Rollout collection and evaluation reuse completed chunks only when their input
and rollout configuration are unchanged. After changing the input data or its
path, or any rollout setting such as models, repeats, or sampling parameters,
write to a fresh output directory or delete that stage's output, then re-run.

---

## Frequently Asked Questions

**Q: How long does training take?**

A: For production qwen3-14B training (32 nodes, 256 GPUs, ~330K samples):
- **12-16 hours** total (4 sequential jobs, ~4 hours each)
- Final job may finish early, hence the range
- Depends on cluster availability and job scheduling

**Q: Can I reduce training time?**

A: Options:
- Use fewer samples (faster but may impact quality)
- Use smaller model (less capable but trains faster)
- Adjust `dependent_jobs` config (fewer, longer jobs vs more, shorter jobs)

---

## Getting Additional Help

If issues persist after trying these solutions:

1. **Check workflow-specific docs:**
   - [Download-SEC](workflows/01-download-sec.md)
   - [Template-Based SDG](workflows/02-template-based-sdg.md)
   - [Document-Grounded SDG](workflows/03-document-grounded-sdg.md)
   - [SFT](workflows/04-sft.md)
   - [Evaluation](workflows/05-eval.md)
   - [GRPO RL Training](workflows/06-grpo.md)

2. **Review the detailed stage sections in the workflow guides:**
   - [Workflow documentation](workflows/)

3. **Report issues:**
   - Check existing [GitHub issues](https://github.com/NVIDIA/nvflow/issues)
   - Create new issue with:
     - Workflow and stage name
     - Error message (full traceback)
     - Config files used
     - Environment details

4. **Feature requests:**
   - Submit [GitHub issue](https://github.com/NVIDIA/nvflow/issues) with `[Feature Request]` tag
   - Describe use case and expected behavior
