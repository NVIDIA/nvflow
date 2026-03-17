# Finance Recipe Troubleshooting

Comprehensive troubleshooting guide for common issues across all finance recipe workflows.

## Quick Navigation

- [Cluster & Infrastructure](#cluster--infrastructure)
- [Resource Issues](#resource-issues)
- [Data Issues](#data-issues)
- [Training Issues](#training-issues)
- [Workflow-Specific Issues](#workflow-specific-issues)

---

## Cluster & Infrastructure

### Slurm Job Failures

**Problem:** Jobs fail to submit or don't execute

**Solutions:**
```bash
# 1. Verify cluster configuration
echo $NEMO_SKILLS_CONFIG_DIR
# Should point to your cluster_configs directory

# 2. Check cluster config file
cat $NEMO_SKILLS_CONFIG_DIR/my_cluster.yaml

# 3. Verify partition names match your cluster
sinfo  # List available partitions

# 4. Check container access
# Ensure NGC/container credentials are configured
```

**Common causes:**
- Missing or incorrect `NEMO_SKILLS_CONFIG_DIR`
- Partition names don't match cluster
- Container authentication issues
- Resource limits exceeded

### Container/Authentication Issues

**Problem:** Cannot pull containers or authentication fails

**Solutions:**
```bash
# 1. Check NGC API key
echo $NGC_API_KEY

# 2. Verify container access
# Test pulling container manually

# 3. Update credentials in cluster config
vim $NEMO_SKILLS_CONFIG_DIR/my_cluster.yaml
```

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

**Additional notes:**
- If Ray cluster hangs during initialization, apply this fix
- The fix changes how containers are executed (uses `enroot exec` instead of `--container-name`)
- Test on your cluster - symptom is Ray cluster initialization hang

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

**For Training (SFT):**
```yaml
# Reduce batch size
training:
  per_device_train_batch_size: 2  # Lower from 4
  gradient_accumulation_steps: 16  # Increase to maintain effective batch size
```

**For Inference (SDG workflows):**
```yaml
# Reduce inference parameters
inference:
  max_tokens: 2048  # Lower from 4096
  batch_size: 1
```

**Advanced solutions:**
- Enable gradient checkpointing
- Use LoRA instead of full fine-tuning
- Reduce model size (14B instead of 32B)
- Increase context parallel size in SFT

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
- Set `save_total_limit: 3` in training config
- Use `skip_filled: true` to avoid regenerating data
- Monitor disk space before long runs

### Insufficient GPUs

**Problem:** Requested GPUs not available

**Solutions:**
```bash
# Check available GPUs
sinfo -p your_partition

# Adjust num_nodes in config
# Example: Use 16 GPUs instead of 32
num_nodes: 2  # 16 GPUs (2 nodes × 8 GPUs)
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
# 1. Verify download-sec completed successfully (from nvflow directory)
ls outputs/finance/demo/workflow-2-download-sec/step-0-download/sec_metadata.parquet

# 2. Check filings directory structure
ls outputs/finance/demo/workflow-2-download-sec/step-0-download/data/
# Should show company directories (NVDA/, AAPL/, etc.)

# 3. Re-run download-sec if needed
uv run nflow run sap-500 --config nvflow/recipes/finance/workflows/download_sec_filings.yaml
```

### SEC Download Rate Limits

**Problem:** SEC EDGAR rate limiting (403 errors)

**Solutions:**
- Wait 10 minutes before retrying
- Verify `sec_identity_email` and `sec_identity_company` are valid
- Start with `demo.yaml` (7 companies) to test
- Check SEC's rate limit policy (10 requests per second)

**Config example:**
```yaml
sec_identity_email: "your-email@company.com"
sec_identity_company: "Your Company Name"
```

### Data Format Issues

**Problem:** Invalid JSON or corrupted data files

**Solutions:**
```bash
# Validate JSONL format
python -c "import jsonlines; list(jsonlines.open('file.jsonl'))"

# Check for empty files
wc -l output.jsonl

# Inspect sample records
head -1 output.jsonl | python -m json.tool
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
# Verify training data size (from nvflow directory)
wc -l outputs/finance/sap-500/workflow-4-sft/qwen3_14b/step-2-train-validation-split/train.jsonl

# Inspect sample
head -1 outputs/finance/sap-500/workflow-4-sft/qwen3_14b/step-2-train-validation-split/train.jsonl | python -m json.tool
```

**Adjust hyperparameters:**
```yaml
training:
  learning_rate: 1e-5  # Try lower (was 2e-5)
  warmup_steps: 2000   # Increase warmup
  max_num_epochs: 5    # Train longer
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
checkpointing:
  save_period: 500  # Increase from 100
  save_total_limit: 3  # Keep fewer checkpoints
```

### Model Loading Errors

**Problem:** Cannot load model or checkpoint

**Solutions:**
```bash
# Verify model path
ls /hf_models/Qwen/Qwen3-14B/

# Check checkpoint structure
ls outputs/.../checkpoints/final/
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
# Test with demo stage (7 companies)
uv run nflow run demo --config nvflow/recipes/finance/workflows/download_sec_filings.yaml
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
- Long runtime: Expected ~10 hours for S&P 500
- Verification failures: Check verification model

**Note:** SFT integration in progress. Currently produces ~800K pairs.

### SFT Workflow

**Common issues:**
- OOM errors: See [Out of Memory (OOM) Errors](#out-of-memory-oom-errors)
- Loss not decreasing: See [Training Loss Not Decreasing](#training-loss-not-decreasing)
- Checkpoint failures: See [Checkpoint Saving Failures](#checkpoint-saving-failures)

**Quick checks:**
```bash
# Verify input data (from nvflow directory)
wc -l outputs/finance/sap-500/workflow-3-template-based-sdg/step-5-filter-answers/final_result.jsonl

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
uv run nflow run sap-500 --config nvflow/recipes/finance/workflows/download_sec_filings.yaml
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

2. **Review stage technical references:**
   - [Stage documentation](stages/)

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
