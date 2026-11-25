# NeMo ecosystem integration boundaries

NVFlow uses components from the NeMo ecosystem without making their internal
APIs part of the project contract.

## Component roles

### NeMo-Skills

NeMo-Skills provides command and pipeline builders used by the first execution
backend. NVFlow stages use
`nvflow.integrations.nemo_skills.commands` for narrow `generate`, `run_cmd`, and
argument-context entrypoints; importing the aggregate NeMo-Skills CLI is
unsupported because it eagerly loads unrelated commands. Projects should not
depend on private NeMo-Skills experiment objects.

NVFlow's NeMo-Skills integration converts declarative jobs into backend-neutral plan
tasks for inspection.

### NeMo-Run

NeMo-Run supplies experiment, executor, tunnel, packaging, and live scheduler
handle primitives.

NVFlow wraps packaging so a submission contains platform source plus only the
selected project subtree. NeMo-Run experiment metadata remains operational state,
not durable NVFlow provenance.

### NeMo-RL

NeMo-RL is a training implementation executed inside an operator-selected
provider image. Native NeMo-RL configuration remains authoritative for trainer
behavior.

NVFlow owns orchestration fields such as target, dependencies, resources,
container role, output locations, and the launch plan. It does not translate a
generic hyperparameter schema into NeMo-RL settings.

The shared `nemo_rl.sft` and `nemo_rl.grpo` stages merge a named preset from
the project's `presets_file` with workflow overrides; NVFlow ships no presets.
See [Train with NVFlow's shared stages](../../guides/training-stages.md).

### NeMo-Gym

NeMo-Gym provides environment components and protocols used by compatible
training and evaluation workflows.

For Gym-backed execution, Gym owns environment/model lifecycle, repeats,
resume, collection, aggregation, profiling, health checks, and canonical
artifacts. NVFlow owns deterministic outer chunks, Slurm resources,
dependencies, endpoint placement, and content-validated completion evidence.

Benchmark evaluation and training-data collection use this same Gym sequence.
Their NVFlow wrappers differ only in source preparation and downstream work:
benchmark reports/comparison for eval, versus shared profiled-dataset selection
and training for GRPO. Rejudge and optional BLADE analysis are available to both.

See [NeMo Gym as NVFlow's Data Plane](nemo-gym-data-plane.md) for deployment
modes, task topology, resume, observability, BLADE, and output contracts.

Policy and judge roles share one endpoint schema:

- `deployment: gym` lets Gym own `local_vllm_model`;
- `deployment: nvflow` schedules a separate vLLM or SGLang service through NVFlow;
- `deployment: external` connects to `base_url` without launching a service.

Policy-as-judge is represented exactly as it is in Gym: omit the judge endpoint
and set the environment's `judge_model_server.name` to `policy_model`.

Each endpoint's `gym_model` mapping uses Gym's native model-component field
names (`uses_reasoning_parser`, `chat_template_kwargs`,
`vllm_serve_kwargs`, and related fields). The same mapping is retained when
switching between Gym and NVFlow deployment ownership.

Native re-verification is limited to resources servers declaring `STATELESS`;
the retained verifier handles audited finance servers whose upstream safety is
still `UNKNOWN`. BLADE remains an optional post-profile, pre-filter analysis
hook.

#### Environment isolation

NVFlow keeps four dependency boundaries separate:

- The lightweight launcher bootstrap—an NVFlow checkout, copied-project
  `.venv`, or client image—runs the platform catalog and prepares selected
  projects.
- Each maintained, external, or submodule project has one isolated cached
  launcher environment shared by all its recipes and workflows.
- `/opt/nemo_gym_venv` in the `nemo-gym` image runs the Gym CLI.
- `/opt/nemo-gym` contains Gym source and any prefetched component environments.
  Workflow `config_paths` plus inline `environments.*.gym` values select and
  configure components.

Recipes compose workflow DAGs; they do not create Python environments, and
`project.yaml` has no venv configuration. NVFlow's `nemo-rl` and `nemo-gym`
images (`dockerfiles/Dockerfile.nemo-rl`, `dockerfiles/Dockerfile.nemo-gym`)
build NVFlow's components in ahead of time, so jobs install nothing at runtime.
Larger catalogs can use multiple worker-image families rather than one
unbounded image.

## Import and ownership rules

- `nvflow.core` remains independent of concrete training and environment
  implementations.
- Provider-specific translation belongs under `nvflow.integrations` or a
  separately versioned integration package.
- Concrete project packages may call supported platform contracts but platform
  code must not import them.
- Provider runtime packages are imported in worker-side or lazy launcher paths,
  not during minimal NVFlow module import.

Project-facing NeMo integration modules are intentionally narrow:

- `nvflow.integrations.nemo_gym.contracts`: dependency-light typed operation contracts
- `nvflow.integrations.nemo_gym.environments`: environment selection and config paths
- `nvflow.integrations.nemo_gym.native_commands`: native Gym command rendering
- `nvflow.integrations.nemo_gym.native_rollout`: rollout, chunking and postprocessing
- `nvflow.integrations.nemo_gym.native_rewards`: safe existing-rollout rejudging
- `nvflow.integrations.nemo_gym.native_sharding`: deterministic completion evidence
- `nvflow.integrations.nemo_gym.operations`: typed planning/submission facade
- `nvflow.integrations.nemo_gym.reward_profiles`: standardized dataset publication
- `nvflow.integrations.nemo_gym.runtime_binding`: worker-side Gym composition and
  generated binding evidence; it adds no authored YAML fields
- `nvflow.integrations.nemo_gym.vllm_judge`: managed-service argument construction
- `nvflow.integrations.nemo_rl.checkpoints`: checkpoint resolution and HuggingFace
  conversion tasks
- `nvflow.integrations.nemo_skills`: planned-job translation and target path helpers

Ordinary projects should prefer declarative `uses: nemo_gym.<type>` and
`uses: nemo_rl.<type>` bindings to the concrete stages in `nvflow.stages`.
Specialized project composition may call the typed integration operations
directly.

The platform-facing `nvflow.integrations.nemo_skills.filesystem` module owns
target path handling. `nvflow.integrations.nemo_gym.launcher_fs` remains a
compatibility import for existing projects. Project stages declare outputs
through `ArtifactDeclaration`; they do not use the target filesystem to
implement platform-level presence checks.

Modules for shell rendering, merge/postprocess commands, capture translation,
Slurm monkeypatching, and worker entrypoints are implementation details.

## Planning versus execution

Planning is non-submitting:

```text
project config
    |
    v
NVFlow ExecutionGraph
    |
    v
redacted LaunchPlan
```

Execution dispatches once from the selected target:

```text
exact ExecutionGraph
    |
    +--> DirectSlurmEngine --> NeMo-Run --> Slurm
    |
    +--> SuppliedRayJobsEngine --> NVFlow controller --> Ray Jobs endpoints
```

Generated worker commands and packaged NVFlow source must come from the same
revision. Plans or cached job scripts created before an integration module move
are not forward-compatible; regenerate the plan and submission after upgrading
NVFlow rather than replaying stale command text.

The public plan is an inspection projection. Private engine objects are not
exposed as project APIs.

## Images and environment compatibility

Operators own image references and mounts. Images must contain the exact runtime
dependencies required by the selected project:

- The supported profile is Python `>=3.13.14,<3.14` and Ray `2.58.0` exactly.
- Prebuilt virtual environments are image-specific and are not interchangeable
  across Python or Ray versions.
- Air-gapped production images must not silently resolve dependencies at job
  runtime. On Slurm, a stage's `installation_command` may install packages when
  a job starts, to try a library before it goes into an image; Ray rejects
  runtime installs.
- Development source overlays are explicit operator choices and invalidate the
  tested image fingerprint.

Build and release details live in the
[container maintainer guide](../../maintainers/containers.md). Cluster image
selection lives in the
[cluster configuration guide](../../operations/cluster-configuration.md).

## Version pins

NVFlow pins tested NeMo-Skills and NeMo-Run versions in project dependency
metadata and container definitions. Updating a pin requires:

1. Command and plan parity tests
2. Packaging tests
3. Reference workflow plan completeness
4. Relevant project quick-start validation

## Deferred deployment evidence

Local CI validates imports, configuration, plans, packaging, and documented safe
examples. It does not claim successful Slurm submission, worker image or mount
compatibility, GPU behavior, model quality, or live external-service behavior.
OpenAI, SEC EDGAR, Tavily, and similar credentialed network paths remain
explicitly deferred until their deployment-specific quick-start canaries run.

## Concrete project usage

Provider-specific project behavior belongs with the project:

- [Finance SFT](../../../projects/finance/docs/workflows/04-sft.md)
- [Finance GRPO](../../../projects/finance/docs/workflows/06-grpo.md)
- [Finance evaluation](../../../projects/finance/docs/workflows/05-eval.md)
- [HopChain](../../../projects/hopchain/README.md)

## Related documentation

- [Platform architecture](../architecture.md)
- [Workflow loading and planning](../workflow-and-planning.md)
- [Container maintenance](../../maintainers/containers.md)
