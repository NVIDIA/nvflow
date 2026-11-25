# Console Utilities Usage Guide

Simple utilities for consistent, beautiful terminal output in NVFlow orchestration.

## Stream contract

`nvflow.core.console` writes orchestration messages to stderr. CLI stdout is
reserved for payloads such as plan text or `plan --format json`; progress,
warnings, diagnostics, and errors belong on stderr. `nvflow run` intentionally
leaves stdout empty in every selection mode.

Stage and orchestration code should therefore use these helpers for operator
messages instead of printing to stdout. Report concrete events such as a task
submission or a discovered file; planning alone is not evidence that a stage
finished.

## Quick Start

```python
from nvflow.core import console

# In launcher/controller code around exact execution:
console.status("Submitting exact execution graph...")

console.detail("Engine", resolved_target.engine)
console.detail("Target", resolved_target.binding.cluster)
console.success(f"Controller submitted: {receipt.controller_handle}")
```

## Available Functions

### Main Functions

```python
from nvflow.core import console

# Headers and sections
console.header("Starting Workflow")      # Big header with lines
console.section("Running Stage: train")  # Section with lighter lines

# Status messages
console.success("Job submitted")     # ✓ Green checkmark
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
from nvflow.core import BaseStage
from nvflow.execution import (
    CommandPlan,
    DependencyPlan,
    PlanningContext,
    ResourceRequest,
    StageExecution,
    TaskPlan,
)

class SFTStage(BaseStage):
    """Supervised fine-tuning stage."""

    def execution(self, context: PlanningContext) -> StageExecution:
        config = dict(context.config)
        return StageExecution(
            stage=context.stage,
            tasks=(
                TaskPlan(
                    task_id=f"{context.stage}.0",
                    stage=context.stage,
                    name=context.expname,
                    operation="shell",
                    commands=(
                        CommandPlan(
                            command=build_training_command(config),
                            container="nemo-rl",
                            runtime_role="nemo-rl",
                        ),
                    ),
                    resources=(
                        ResourceRequest(
                            nodes=config["num_nodes"],
                            gpus_per_node=config["num_gpus"],
                        ),
                    ),
                    dependencies=tuple(
                        DependencyPlan(name, external=True) for name in context.run_after
                    ),
                    parameters={"ray_application": True},
                ),
            ),
        )
```

Select the class from workflow YAML:

```yaml
stages:
  sft:
    uses: project:my_project.stages.sft:SFTStage
```

## Stderr example

```
▶ Preparing SFT training job

  Model: /models/llama-3-8b
  Data: /data/finance/train.jsonl
  GPUs: 8
  Cluster: my_cluster

ℹ Found training data: /data/finance/train.jsonl
▶ Submitting job to cluster...
✓ Job submitted successfully: 12345
ℹ Experiment: finance-sft-run1

────────────────── Next Steps ──────────────────
ℹ Monitor job:
  SSH: ssh <login-host>
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

**These are for launcher/controller orchestration logs.**

- ✅ Use around planning, engine dispatch, submission, and status transitions
- ✅ Show what exact graph and target are being used
- ✅ Writes progress and diagnostics to stderr
- ❌ Do not use for machine-readable stdout payloads
- ❌ Do not print from pure `StageExecution` builders
- ❌ Not for training/inference logs (those belong to Slurm or Ray Jobs logs)

Actual job logs are captured by the selected execution engine and do not need
or benefit from Rich formatting.
