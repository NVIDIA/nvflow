# NeMo Gym as NVFlow's Data Plane

NVFlow uses NeMo Gym as the shared rollout data plane for benchmark evaluation,
training-data collection, and post-hoc rejudging. NVFlow remains the control
plane: it owns workflow composition, Slurm resources, dependencies, outer data
chunks, endpoint placement, provenance, and completion evidence.

This boundary deliberately keeps benchmark eval and GRPO as different
user-facing workflows while avoiding separate rollout implementations.

## Terminology

Gym's `gym eval` command namespace is broader than NVFlow's finance eval
workflow. `gym eval run` is Gym's generic agent/environment rollout command and
works with benchmark, training, or custom datasets.

NVFlow exposes an opt-in shared stage suite under `nvflow.stages.nemo_gym`:

- `prepare_benchmark` invokes native benchmark preparation and rendering;
- `collate` attaches Gym agent routing to project datasets;
- `render` renders project prompt templates and their tools into Gym rows
  ([Render prompts](#render-prompts));
- `collect` produces canonical rollout/profile/health bundles;
- `build_training_dataset` publishes profiled and selected training records;
- `rejudge` applies another verifier or judge without rerunning the policy;
- `compare` compares two validated completed bundles;
- `train_validation_split` splits each environment's training data for
  [NeMo-RL GRPO](../../guides/training-stages.md).

Projects select only the shared stages they need under project-owned names:

```yaml
stages:
  prepare_data:
    uses: nemo_gym.collate
    # ...
  collect_rollouts:
    uses: nemo_gym.collect
    # ...
  rejudge:
    uses: nemo_gym.rejudge
    # ...
```

Shared use is opt-in and visible next to configuration. A custom project stage
is an explicit alternative: replace the shared alias with
`project:python.module:StageClass` in that workflow or child override. No
global stage identity is involved.
Use `nvflow list-stage-types` and `nvflow stage-type-info nemo_gym.collect` to
discover the curated catalog.

The shared collector consumes explicit project configuration:

```yaml
contract_version: 1
output_dir: /workspace/outputs/my-project/rollouts
container: nemo-gym
vllm_container: vllm
rollout:
  num_repeats: 4
  num_chunks: 2
  dependent_jobs: 0
  models:
    policy:
      deployment: nvflow
      model: /hf_models/my-model
      num_gpus: 4
environments:
  my_environment:
    enabled: true
    input_data: /workspace/outputs/my-project/prepared/train.jsonl
    output_name: my-environment
    config_paths: [resources_servers/my_environment/configs/my_environment.yaml]
    agent_name: my_environment_agent
dataset_selection:
  mode: all
  unprofiled_samples: separate
```

`vllm_container` is the cluster-config container key for an NVFlow-managed
vLLM endpoint. External and Gym-local endpoints do not require it.

An empty `input_data` file fails the collection unless
`rollout.allow_empty_input` is `true`. Then the collection completes empty: it
skips Gym's analysis, writes the analysis files empty, and still runs a
`report` extension. The option can't be combined with dataset selection or
blade.

Projects override supported fields in YAML. When a project needs different
task topology or semantics, it replaces the shared alias with
`uses: project:python.module:StageClass` and implements its own stage through
the typed operation facade.

Projects may configure a versioned, packaged Python report worker:

```yaml
extensions:
  report:
    contract_version: 1
    module: my_project.build_rollout_report
```

`report` runs after native profile and health analysis for collect and rejudge.
The module travels with packaged project worker code and executes inside the
selected container; it is not an arbitrary platform callback. An optional
`options` mapping reaches the module as `--options` JSON and is part of the
analysis identity, so changing it reruns the report.

## Deployment ownership

Policy and judge roles use the same endpoint contract.

### `deployment: gym`

Gym starts `local_vllm_model` in the rollout allocation and owns its lifecycle.
NVFlow still declares the GPUs and validates that TP × PP × DP matches
`num_gpus`.

```yaml
models:
  policy:
    deployment: gym
    model: /hf_models/Qwen/Qwen3-4B
    num_gpus: 4
    gym_model:
      uses_reasoning_parser: true
      vllm_serve_kwargs:
        tensor_parallel_size: 4
        reasoning_parser: qwen3
```

### `deployment: nvflow`

NVFlow starts the server in the same Slurm job as the Gym client, or as a
separate Ray Job, and passes its dynamic URL to Gym.
`server_type` selects `vllm` or `sglang`; `container` may override the matching
cluster alias. `gym_model.vllm_serve_env_vars`, strings or numbers, are
exported as literal values on every node of that server.

```yaml
models:
  judge:
    deployment: nvflow
    server_type: vllm
    model: /hf_models/openai/gpt-oss-120b
    num_gpus: 4
```

With `server_nodes` above 1, a vLLM server spans its nodes through Ray, so its
image needs Ray. For an image without Ray, `distributed_executor_backend: mp`
in `vllm_serve_kwargs` uses vLLM's own multi-node mode: every node runs `vllm
serve` with its rank and the first node's address, and every node but the first
also passes `--headless`. On a supplied Ray cluster, NVFlow starts one process
per node in a placement group and runs the same commands, each with the server's
`vllm_serve_env_vars`.

```yaml
models:
  policy:
    deployment: nvflow
    model: /hf_models/large-model
    num_gpus: 4
    server_nodes: 2
    gym_model:
      vllm_serve_kwargs:
        distributed_executor_backend: mp
```

### `deployment: external`

Neither Gym nor NVFlow launches the endpoint. Gym uses `base_url` and the
optional key named by `api_key_env`.

```yaml
models:
  judge:
    deployment: external
    model_type: openai_model
    model: gpt-5
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
```

Policy-as-judge is not a deployment mode. Omit `models.judge` and use Gym's
native environment route:

```yaml
judge_model_server:
  name: policy_model
```

Model request parameters and environment-specific Gym overrides remain in the
workflow YAML for every deployment mode, and NVFlow passes them to Gym
directly. Each Gym job writes the small config that binds its own datasets and
judge; those files are generated runtime artifacts, not source configuration.

## Native lifecycle

Benchmark and training datasets use the same native sequence:

1. `gym env validate` checks component composition.
2. `gym env prefetch` builds component environments, normally while building
   the `nemo-gym` and `nemo-rl` images.
3. `gym eval prepare` prepares a registered benchmark when applicable.
4. `gym dataset render` materializes a prompt template when raw rows are not
   already in Responses API form.
5. `gym dataset collate` validates datasets and stamps `task_source`; Gym
   resolves the runtime agent from that source and the active config.
6. `gym eval run --resume` starts the configured fleet, expands repeats,
   dispatches missing rows, persists failures, and shuts the fleet down.
7. `gym eval aggregate` merges chunk outputs and asks each environment for its
   benchmark-specific aggregate metrics.
8. `gym eval profile` produces per-task, per-agent, and per-repeat reward
   summaries when multiple repeats exist.
9. `gym eval health-check` audits canonical trajectory and model-call evidence.
10. `gym eval reverify` optionally applies another stateless verifier to
    existing rollouts.
11. `gym eval compare` compares source and output after every rejudge and
    remains available independently for other compatible completed runs.

Registered Gym benchmark folders are convenient for preparation and discovery,
not a requirement for rollout execution. A custom environment can run when its
model, agent, resource-server, dataset, and optional sandbox configs are
available through `config_paths`.

## Add a project environment

1. Reference an existing provider-owned Gym config directly through
   `config_paths`. A genuinely new environment owns one complete Gym config
   beside its component implementation; it does not require an NVFlow
   registration or a second descriptor.
2. Add an `environments.<name>` entry with `input_data`, `config_paths`, and
   `agent_name`. Dataset collation uses `source_data` instead of `input_data`.
   `enabled` defaults to `true`, and `output_name` defaults to the map key; omit
   both unless changing those defaults.
   NVFlow asks the pinned Gym runtime to resolve the selected agent's
   implementation and resources-server route. Do not duplicate those
   provider-owned identities in project config.
   The generated `gym_binding.json` records the resolved identities for
   diagnostics; it is an output artifact, never an authored input.
3. Put environment-specific policy/judge, request, Gym, timeout, sandbox, and
   BLADE settings in that entry. Keep shared defaults under `rollout`. Treat the
   provider config as authoritative for agent, resource-server, and model-server
   inventory; inline `gym` values should tune leaves, not delete or reconstruct
   that topology.
4. Validate locally with `nvflow validate -e <name>`, inspect the effective
   configuration with `nvflow show-config`, run a static
   `nvflow plan -e <name>`, and compose
   changed maintained environments with `gym env validate` in the pinned Gym
   image.
5. Start with a small `max_num_samples`, two repeats, and one chunk. Validate
   canonical completion, profile/health output, and live resume before enabling
   the environment in a larger workflow.

Large projects may keep one environment mapping per YAML file behind an
explicit index instead of growing one workflow mapping:

```yaml
environments:
  finance-sec-search: finance-sec-search.yaml
  swebench-verified: swebench-verified.yaml
```

Each indexed file declares a matching top-level `environment` identity.
`load_environment_catalog(index_path, boundary=...)` rejects duplicate names,
missing or escaped files, identity mismatches, malformed YAML, and non-finite
data, then returns the ordered mappings and a semantic SHA-256 digest. It never
scans directories or imports plugins.

The singular `environment:` field is only the self-identity check for this
opt-in sharded catalog. Inline workflow mappings do not use it. The plural
`environments:` map and `--environment`/`-e` selector remain NVFlow's public
execution and fan-out axis; `_environment` is internal selection transport and
must not be authored.

Credentials and cluster mounts remain operator-owned. A project-specific cache
warmer or report can be placed before or after the shared stages without
changing them.

The same contract can describe coding environments such as SWEbench Verified,
SWEbench Pro, and DeepSWE: their benchmark and sandbox-provider files are
`config_paths`, their OpenCode instance is `agent_name` (Gym resolves its
implementation), and
DeepSWE's host-routable endpoint requirement is a leaf
`gym.use_absolute_ip: true` override. Runtime sandbox deployment and image
prefetch remain operator concerns, not new NVFlow stage types.

## Scheduler tasks inside a rollout stage

One user-facing eval or `collect_rollouts` stage expands into the following
tasks.

### Rollout chunk tasks

Each chunk:

- validates a deterministic source/config fingerprint;
- validates completion evidence and exits before Gym work when it is reusable;
- starts Gym-, NVFlow-, or externally owned endpoints as configured;
- starts Gym agent, resource-server, tool, and sandbox components;
- runs every `num_repeats` value through `gym eval run --resume`;
- records canonical successes and failure sidecars;
- validates counts and writes task-local `completion.json`.

`num_chunks` is NVFlow's outer Slurm parallelism. `num_repeats` remains inside
Gym, so adding repeats does not multiply endpoint deployments.

The fingerprint covers what can change an answer: the input and its split,
repeats, prompts, tools, containers, sampling, Gym settings, and each model, how
it loads its weights, and how it generates. A resumed run keeps its answers when
only how they are produced changes: `num_samples_in_parallel`, `dependent_jobs`,
a server's GPUs and nodes, the vLLM settings that size and schedule it
(parallelism, batching and memory, compilation and CUDA graphs, prefix caching
and chunked prefill, autotuning, and logging), `vllm_serve_env_vars`, and
`model_endpoint_readiness_timeout_seconds`. A rejudge follows the same rule.

A run recorded by an earlier NVFlow version resumes while its settings are
unchanged. If it is resized later, its finished chunks start their servers once
more before Gym finds nothing left to answer. An earlier version treats a
rejudge that this version resumed as stale and starts it over.

NVFlow defaults an omitted Slurm dependency type to `afterok`. When
`dependent_jobs > 0`, it overrides only continuation attempts and their
analyze job to `afterany`, allowing resume after a timeout while keeping the
first attempt fail-closed against prior stages.

### Job 1: analyze rollouts

After every chunk succeeds, one CPU task:

- revalidates source/config identity and every chunk marker;
- atomically joins the chunks' rollouts, materialized inputs and failure
  sidecars;
- runs native aggregate on the joined rollouts for metrics only
  (`merge_shards=false`), then profile and health-check;
- runs an optional project-packaged report worker;
- runs optional BLADE analysis under the same job;
- validates all analyzed artifacts and writes task-local `completion.json`.

A one-chunk run aggregates before its original Gym fleet shuts down. A
multi-chunk run starts only lightweight CPU Gym head/agent/resource proxies for
global aggregation; it does not relaunch policy or judge inference models.

The pinned Gym profiler still reads the canonical run files in one process.
NVFlow's chunking, completion validation, shuffle, and dataset-selection joins
are disk-backed or bounded, but campaigns must size a single environment output
for the profiler's available CPU memory until Gym provides streaming profiling.

### Optional BLADE analysis

When `blade.enabled` and `blade.command` are configured, the analyze job
exports:

- `NVFLOW_ROLLOUTS`
- `NVFLOW_MATERIALIZED_INPUTS`
- `NVFLOW_REWARD_PROFILE`
- `NVFLOW_OUTPUT_DIR`
- `NVFLOW_BLADE_OUTPUT_DIR`

The command writes under `jobs/job-1-analyze-rollouts/analysis/blade`. A BLADE-style exploratory report can use
one completed run. Formal calibrated BLADE scoring additionally requires a
benchmark-specific analysis skill, comparable rollout sets, curated golden
reports, metrics, and anchor facts.

### Job 2: build training dataset

After native profiling, an optional CPU task validates native task/agent
identity, flattens each source record with ordered reward/token evidence, and
atomically writes standardized `profiled.jsonl` and selected `train.jsonl`
plus completion evidence.

Selection is explicit. `mode: all` preserves every profiled row for benchmark
or sharing workflows. `mode: scalar_variance` keeps rows whose sample standard
deviation of scalar reward is greater than or equal to `min_std`; component
metrics are unsupported in contract v1. Reward standard deviation uses the
sample convention, while token-length standard deviations use the population
convention over observed non-null token values.

Pass metrics require an explicit success threshold; NVFlow never infers
correctness from a dataset-wide maximum. `unprofiled_samples: separate` excludes
unmeasured rows from the public datasets and records their identities and reason
under job-local diagnostics. `exclude`, `include`, and `error` are explicit
alternatives. Omitting `dataset_selection` disables dataset publication;
`mode: all` enables it without value-based filtering.

Selected records come from the canonical materialized sample; a changed source
row with the same UUID or task index is rejected. Before comparison, NVFlow
applies the same shallow, top-level `responses_create_params` overrides that
Gym used during materialization and removes Gym's synthetic per-repeat seed.
Integrated collect/rejudge stages forward those effective overrides; standalone
`nemo_gym.build_training_dataset` callers set
`responses_create_params_overrides`.

Publication writes root `profiled.jsonl` and `train.jsonl`, a job-local
`selection_report.json`, optional `diagnostics/unprofiled_samples.jsonl`, and
finally job-local `completion.json` under a POSIX lock. Consumers call
`validate_dataset_completion(...)` to verify both dataset hashes, row counts,
the report, and optional diagnostics. Native Gym profile artifacts remain
available for audit and resume; no intermediate `difficulty.jsonl` is required.

## Render prompts

`nemo_gym.render` turns JSONL records into Gym rows with project prompt
templates:

```yaml
stages:
  render:
    uses: nemo_gym.render
    contract_version: 1
    output_dir: ${directories.rendered}
    container: nemo-gym
    installation_command: export PATH=/opt/nemo_gym_venv/bin:$PATH
    renders:
      pool_a:
        input_file: ${directories.split}/pool_a.jsonl
        prompt: project://prompts/agent_template.yaml
```

The template's `user` and optional `system` keys are the prompt. Every other
key, such as `tools` and `parallel_tool_calls`, is first written into each
row's `responses_create_params`, replacing any rendered `input`; a render's
`row_fields` adds fixed fields, such as a `format_verification` verifier, to
every row. Then `gym dataset render` fills the prompt from the row's fields and
fails on a field the row lacks. Each render writes `NAME.jsonl` and `NAME.done`,
which records the digests of the input, template, row fields, and output, so a
rerun with the same inputs skips it. All renders run in one CPU task.

## Split rows by reward

NVFlow ships Gym's `format_verification` environment as
`nvflow/integrations/nemo_gym/configs/format_verification.yaml`. A collection
uses it with that file in `config_paths` (under `{{NVFLOW_CODE_ROOT}}/`) and
`agent_name: sdg_format_verification_simple_agent`. Each row's `verifier`,
added by the render's `row_fields`, sets its reward: `regex` (`verify_regex`,
`verify_min_matches`), or `string_match` (`expected_markers`, `patterns`), which
rewards a reply that holds every marker and no other match of its patterns.

The `split_by_reward` report extension turns a collection into pass/fail
lists, for example when a model checks each record on the
`format_verification` environment:

<!-- doc-check: yaml-excerpt -->
```yaml
    extensions:
      report:
        contract_version: 1
        module: nvflow.integrations.nemo_gym.workers.split_by_reward
        options: {pass_rate: 1.0}
```

The analyze job runs it after merging the rollouts. A rollout succeeds when its
reward is 1.0, and a row passes when at least `pass_rate` of its rollouts
succeed; the default, 1.0, requires all of them. A rollout without a finite
reward, or whose response didn't complete, has no reply, or is masked, counts as
a failure, and `reward_split.json` counts those rows as unscored. It writes `passed.jsonl`, `failed.jsonl`, and `reward_split.json` to
`jobs/job-1-analyze-rollouts/analysis/evaluation/`. Each row is the input row
without the request fields that rendering and Gym added, plus `_reward`, the
mean reward, `_success_rate`, and `_replies`, the model's replies. The job
fails unless every row has one rollout per materialized row, so a partial
collection never reads as verdicts.

## Rejudge

`rejudge` is a separate first-class stage because it may run later with a
different judge, resources, output root, and provenance.

Its task topology mirrors rollout collection:

1. `job-0-rejudge-rollouts` writes a task-local canonical bundle, behavior
   manifest, logs, and completion record.
2. `job-1-analyze-rollouts` runs profile, health-check, evaluation or reward
   analysis, optional BLADE, and native comparison before publishing its
   completion record.
3. A workflow may append `job-2-build-training-dataset`.

This prevents GPU-backed judges from sitting idle during profile/report work.
Live planning prunes the judge task only after validating artifact hashes,
behavior identity, and source-completion linkage on the target filesystem. If that
validation is unavailable or fails, a guarded judge task remains; it removes
invalid completion records before resuming same-behavior partial output. If analysis
is retried after the canonical marker exists, it validates and reuses the bundle
before rerunning enabled analysis and comparison.

For a verifier Gym declares `STATELESS`, NVFlow delegates to
`gym eval reverify`. The maintained finance verifiers are currently marked
`UNKNOWN`, so NVFlow uses its audited fallback while preserving the same
canonical outputs, resume, aggregate, profile, health, BLADE, and
dataset-selection contracts. With `verification.unique_prompts: true`, the
fallback starts each answer's last user message with a unique line, so a judge
prompt built from that message differs from any earlier request's after the
template's fixed text. An answer whose verification Gym reports as failed is
verified again with a new line; a reply Gym scores without a failure, such as
one with no rating, is kept. Saved records keep the question unchanged; the
setting is part of the rejudge's fingerprint.

Eval rejudge normally preserves all rows. GRPO rejudge may apply the same
explicit shared selection policy and become the source consumed by the final
train/validation split.

## Resume and completion

Gym owns row-level resume. `gym eval run --resume` skips completed
`(_ng_task_index, _ng_rollout_index)` pairs and retries missing or failed rows.
NVFlow preserves Gym's three-attempt floor and raises it only when a longer
`dependent_jobs` continuation chain requires more attempts.

NVFlow owns scheduler-level resume:

- submitted chunk and rejudge tasks validate their own completion evidence
  before starting Gym or model endpoints;
- partial same-behavior work delegates row selection to Gym resume;
- a collection whose Gym fleet exits before writing a rollout, for example when
  another process takes a server's port during startup, starts again on a new
  head port, up to three attempts; so does the analysis job's aggregation
  environment when it exits before its servers are ready;
- manifests fingerprint source content, model and judge settings, sampling,
  environment overrides, repeats, and chunking, but not how servers are sized
  or scheduled;
- rejudge behavior state also binds the validated source completion digest;
- changed semantics reject stale output instead of silently reusing it;
- final and chunk `completion.json` records contain content hashes and counts, and those
  counts must equal the manifests' selected rows multiplied by repeats.

Static plans retain the worst-case topology. Live planning hashes the current
artifacts and prunes only content-validated chunk or judge work; marker
existence alone is never sufficient. Invalid markers remain scheduled and are
repaired only inside submitted jobs.

## Observability and trajectories

Gym's persisted trajectory capture is enabled with native config:

```yaml
observability_enabled: true
model_call_capture_dir: /absolute/output/path/jobs/job-0-collect-rollouts/outputs/captures/model-calls
```

`ng_trajectory` is more than a reasoning trace. It can include ordered agent
turns, exposed reasoning, model requests/responses, tool observations,
judge/verifier calls, token counts, latency, retries, and capture gaps.

Capture is useful for health checks and BLADE analysis but increases storage
and may retain sensitive prompt/tool content. Enable it by default for smoke and
diagnostic runs; make an explicit retention decision before large production
runs. Token-ID capture is a separate training feature and is not required for
ordinary evaluation.

## Output layout

Gym-owned sibling basenames remain unchanged, but every native artifact lives
under the logical job that produced it:

```text
<environment-output>/
├── profiled.jsonl                    # when dataset_selection is enabled
├── train.jsonl                       # selected training dataset
└── jobs/
    ├── job-0-collect-rollouts/
    │   └── chunks/chunk-N/
    │       ├── outputs/
    │       ├── attempts/attempt-N/logs/
    │       ├── manifest.json
    │       └── completion.json
    ├── job-1-analyze-rollouts/
    │   ├── outputs/
    │   ├── analysis/{evaluation,blade,compare}/
    │   ├── logs/
    │   └── completion.json
    └── job-2-build-training-dataset/
        ├── diagnostics/unprofiled_samples.jsonl
        ├── selection_report.json
        ├── logs/
        └── completion.json
```

Rejudge replaces job 0 with `job-0-rejudge-rollouts`. Only applicable jobs and
directories are created. Eval and SDG ordinarily stop after job 1; GRPO and
post-training dataset publication add job 2. Old flat output trees are rejected
and must be replaced with fresh output roots.

## What NVFlow intentionally retains

NVFlow keeps only behavior not owned by Gym:

- deterministic outer chunking and Slurm dependency graphs;
- GPU/node/container accounting and endpoint placement;
- guarded resume and content-validated completion markers;
- project path/provenance packaging;
- optional project-specific source enrichment and reporting;
- standardized, explicitly configured profile normalization/selection.

NVFlow should not implement a second row-level resume engine, generic rollout
collector, generic aggregate/profile/health implementation, or judge lifecycle
when the pinned Gym CLI already provides it.
