# Console Utilities Usage Guide

Simple utilities for consistent, beautiful terminal output in NVFlow orchestration.

## Quick Start

```python
from nvflow.core import console

# In your stage execute() method:
def execute(self, config, cluster, expname, run_after=None):
    console.status("Preparing to submit job...")

    # Show configuration
    console.detail("Model", config['model'])
    console.detail("GPUs", str(config['total_gpus']))
    console.detail("Cluster", cluster)

    # Submit job (via NeMo-RL)
    job_id = submit_to_cluster(...)

    console.success(f"Job submitted: {job_id}")
    console.info("View logs: ssh cluster 'tail -f /path/to/log'")
```

## Available Functions

### Main Functions

```python
from nvflow.core import console

# Headers and sections
console.header("Starting Workflow")      # Big header with lines
console.section("Running Stage: train")  # Section with lighter lines

# Status messages
console.success("Job completed")     # ✓ Green checkmark
console.info("Found 3 files")        # ℹ Blue info
console.warning("Job may be slow")   # ⚠ Yellow warning
console.error("Job failed")          # ✗ Red error
console.status("Submitting...")      # ▶ Cyan status

# Details (indented key-value)
console.detail("Model", "/path/to/model")
console.detail("GPUs", "8")

# Spacing
console.blank()                      # Empty line
console.rule("Configuration")        # Horizontal rule
```

## Example: Complete Stage

```python
from pathlib import Path
from typing import Any, Dict, List, Optional
from nvflow.core import BaseStage, StageRegistry, console

@StageRegistry.register(recipe="finance", workflow="sft", stage="sft")
class SFTStage(BaseStage):
    """Supervised fine-tuning stage."""

    workflow = "sft"

    def execute(
        self,
        config: Dict[str, Any],
        cluster: str,
        expname: str,
        run_after: Optional[List[str]] = None,
    ) -> None:
        """Execute SFT training."""

        # Show what we're doing
        console.status("Preparing SFT training job")
        console.blank()

        # Show configuration
        console.detail("Model", config['model'])
        console.detail("Data", config['data_path'])
        console.detail("GPUs", str(config.get('total_gpus', 8)))
        console.detail("Cluster", cluster)
        console.blank()

        # Validate input data exists
        data_path = Path(config['data_path'])
        if not data_path.exists():
            console.error(f"Data not found: {data_path}")
            raise FileNotFoundError(f"Data not found: {data_path}")

        console.info(f"Found training data: {data_path}")

        # Submit job (real stages use the workflow runner and cluster backend; this is illustrative)
        console.status("Submitting job to cluster...")

        try:
            job_id = "12345"  # placeholder; actual submission is handled by the workflow runner

            console.success(f"Job submitted successfully: {job_id}")
            console.info(f"Experiment: {expname}")

            # Tell user how to monitor
            console.blank()
            console.rule("Next Steps")
            console.info("Monitor job:")
            console.detail("SSH", f"ssh {cluster}")
            console.detail("Logs", f"tail -f /path/to/slurm-{job_id}.out")

        except Exception as e:
            console.error(f"Job submission failed: {e}")
            raise
```

## Output Example

```
▶ Preparing SFT training job

  Model: /models/llama-3-8b
  Data: /data/finance/train.jsonl
  GPUs: 8
  Cluster: nrt

ℹ Found training data: /data/finance/train.jsonl
▶ Submitting job to cluster...
✓ Job submitted successfully: 12345
ℹ Experiment: finance-sft-run1

────────────────── Next Steps ──────────────────
ℹ Monitor job:
  SSH: ssh nrt
  Logs: tail -f /path/to/slurm-12345.out
```

## When to Use Each Function

| Function | Use Case | Example |
|----------|----------|---------|
| `header()` | Workflow start/end | "Starting SDG Pipeline" |
| `section()` | Stage execution | "Running Stage: generate_qas" |
| `success()` | Successful operations | "Job submitted successfully" |
| `info()` | General information | "Found 10 input files" |
| `warning()` | Non-critical issues | "Job may take 30+ minutes" |
| `error()` | Errors and failures | "Job submission failed" |
| `status()` | Ongoing operations | "Submitting job..." |
| `detail()` | Configuration details | Key-value pairs |

## Remember

**These are for LOCAL orchestration logs!**

- ✅ Use in `stage.execute()` methods (runs on your machine)
- ✅ Shows what's being submitted to cluster
- ❌ Not for code running inside cluster jobs
- ❌ Not for training/inference logs (those are in Slurm files)

The actual job execution logs are captured by Slurm on the cluster and don't need (or benefit from) Rich formatting.
