# Migrate project stages to exact execution

NVFlow v1.2 removes imperative stage submission as a supported project API.
Every runnable stage must produce one exact `StageExecution`; the platform
derives public plans and lowers the same graph to direct Slurm or supplied Ray.

## Before

```python
class TransformStage(BaseStage):
    def execute(self, config, cluster, expname, run_after=None):
        run_cmd(
            cluster=cluster,
            expname=expname,
            command=build_command(config),
            run_after=run_after,
        )
```

This mixes project logic with one submission backend and permits planning to
drift from execution. Instead of writing `execution()` directly, a stage can
subclass `PlannedExecutionStage` and return a `StagePlan` from `plan(context)`,
or `CapturedExecutionStage`, which records the NeMo-Skills calls an existing
`build_tasks` makes. Both are exported from `nvflow.execution`.

## After

```python
from nvflow.core import BaseStage
from nvflow.execution import (
    CommandPlan,
    DependencyPlan,
    ResourceRequest,
    StageExecution,
    TaskPlan,
)


class TransformStage(BaseStage):
    def execution(self, context):
        command = build_command(dict(context.config))
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
                            command=command,
                            container="project-runtime",
                            runtime_role="project-runtime",
                        ),
                    ),
                    resources=(ResourceRequest(),),
                    dependencies=tuple(
                        DependencyPlan(name, external=True) for name in context.run_after
                    ),
                ),
            ),
        )
```

Use explicit, open `runtime_role` names that an operator target can route.
Commands built by NVFlow's planning helpers route by their `container` alias;
only a hand-built `CommandPlan` can set a different `runtime_role`. Do not inspect
`executor`, call `sbatch`, import Ray, submit through NeMo-Skills, add
`target_cluster`, or place endpoint URLs in workflow YAML.

Declare:

- Exact commands, with execution-only payloads separated from redacted display
  commands when necessary.
- Typed dependencies (`after_success`, `after_start`, `after_ready`,
  `after_any`).
- Resource requirements and Ray-native application intent. Leave scheduler
  controls (`ResourceRequest.partition`, `account`, `exclusive`, and the
  task's `slurm.sbatch_kwargs`) unset: the target supplies them, and planning
  for Ray rejects them. Planning for Ray also rejects an `installation_command`
  task parameter and any command or parameter that installs packages at
  runtime. An `installation_command` passed to a captured NeMo-Skills helper
  becomes part of the task command, so Ray rejects it only when it installs
  packages.
- Service readiness and cleanup.
- Expected artifacts.
- Secret references, never values.

`nvflow plan` fails when any selected stage lacks an exact contract. Use the
Starter project and its external-project tests as the minimum authoring
reference.
