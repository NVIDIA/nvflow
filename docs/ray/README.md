# Ray execution

NVFlow runs each workflow step as a Ray Job on operator-supplied Ray clusters,
through the same project, workflow, and CLI contracts as Slurm. NVFlow does not
create, resize, autoscale, or destroy Ray clusters. Complete
[INSTALL.md](../../INSTALL.md) first: it installs the client and prepares the
images, models, and data every target needs.

For a Slurm cluster, use [Slurm execution](../slurm/README.md) instead.

## Runtime profile

This release follows the NeMo-RL 0.8.0 and NeMo-Gym 0.7.0 runtime:

- Python `>=3.13.14,<3.14` (`3.13.14` in CI and release images)
- Ray `2.58.0` exactly on the launcher and every supplied cluster

Install the Ray launcher client with:

```bash
uv sync --extra ray
```

Every cluster head and worker must use the same Ray version. The operator must
also export `NVFLOW_IMAGE_ID` on every head and worker, set to exactly the
target's `expected_image` string for that cluster. NVFlow compares that label
with `expected_image` as an exact string and does not inspect the image itself.

## Set up a Ray target

Set up once per site, then [run workflows](#run-workflows) from the main flow
with the target as `--cluster`. The examples use the Starter workflow, the
smallest complete Ray workload.

### 1. Prepare the launcher

The launcher is the host where you run `nvflow`. It must reach every Jobs
endpoint and the artifact root:

- **From a workstation:** give the target an `ssh_tunnel` block (step 3).
  NVFlow forwards each `jobs_url` through the tunnel host and checks the
  artifact root there. The tunnel carries plain `http` only, so it suits
  development endpoints such as the harness's; reach production `https`
  endpoints directly.
- **On the cluster network:** use a host, such as a login node, that mounts the
  artifact root at the same path as the clusters.

From the NVFlow repository on the launcher:

```bash
uv sync --extra ray
uv run python -c \
  'import platform, ray; print(platform.python_version(), ray.__version__)'
```

**✅ Done when:** the command prints `3.13.14 2.58.0`, and the launcher reaches
every Jobs endpoint, directly or through `ssh_tunnel`.

### 2. Supply compatible clusters

Starter requires two runtime roles:

- `nemo-skills`: controller and generation client.
- `vllm`: one-GPU model service for `/hf_models/Qwen/Qwen3-4B`.

Each cluster runs one of the [worker images](../../INSTALL.md#worker-images).
The roles may share one cluster only if its qualified image
contains both runtimes. The normal test topology uses a CPU Skills cluster and
a GPU vLLM cluster, whose model server uses one GPU.

Before NVFlow starts, the operator must ensure:

- Both Jobs API endpoints are running and the launcher can reach them,
  directly or through `ssh_tunnel`.
- Required mounts and workload environment variables are provisioned on every
  participating head/worker before Ray starts. NVFlow does not mutate a
  supplied cluster's container mounts or inject Jobs `runtime_env.env_vars`.
- The Skills/control cluster can reach both controller-facing Jobs endpoints
  and the service address advertised by the vLLM cluster.
- `/shared/nvflow` is mounted read/write at the same absolute path on both
  clusters and on the launcher or its `ssh_tunnel` host.
- `/workspace/outputs` is shared between workload clusters.
- `/hf_models/Qwen/Qwen3-4B` is visible on the vLLM cluster.
- Every head and worker exports `NVFLOW_IMAGE_ID`, set to exactly the
  cluster's `expected_image` string in the target (step 3).
- Each head sets `RAY_JOB_START_TIMEOUT_SECONDS` to cover the longest wait for
  free resources. NVFlow submits every ready step and lets the cluster queue
  them, and Ray fails a Job that cannot start in time (15 minutes by default).

For development only, the
[Ray-on-Slurm harness](../../scripts/ray-test/README.md) starts these clusters
from the target in step 3. It is test infrastructure, not an NVFlow
provisioning feature.

**✅ Done when:** every role your workflows use runs on a cluster that meets
the list above.

### 3. Create the target

Copy [`cluster_configs/template-ray.yaml`](../../cluster_configs/template-ray.yaml)
to `cluster_configs/starter_ray.yaml` and keep only the clusters and routes your
workflows use. For Starter:

```yaml
executor: ray

ray:
  control_cluster: skills
  artifact_root: /shared/nvflow

  clusters:
    skills:
      jobs_url: ${oc.env:RAY_SKILLS_JOBS_URL}
      credential_env: RAY_SKILLS_TOKEN
      expected_image: nvflow-nemo-skills:v1.2.0
      expected_python: "3.13.14"
      expected_ray: "2.58.0"

    serving:
      jobs_url: ${oc.env:RAY_VLLM_JOBS_URL}
      credential_env: RAY_VLLM_TOKEN
      expected_image: nvflow-vllm:v0.29.0
      expected_python: "3.13.14"
      expected_ray: "2.58.0"
      gpus_per_node: 8

  routes:
    nemo-skills: skills
    vllm: serving
```

`gpus_per_node` declares the size of each node in the cluster, 8 GPUs here;
Starter's vLLM server uses one of them.

When the controller reaches a cluster through a different address or secret
than the launcher, add `controller_jobs_url` or `controller_credential_env` as
shown in the template. Unauthenticated development endpoints, such as the
harness's, drop `credential_env` and set `allow_insecure: true`.

To launch from a workstation, uncomment the template's top-level `ssh_tunnel`
block and set your login host, user, and key. Each `jobs_url` is then an `http`
address that the tunnel host can reach, and each cluster needs
`allow_insecure: true`; reach `https` endpoints directly instead.

**✅ Done when:** the target names only the clusters and routes your workflows
use.

### 4. Set endpoint and credential references

Use HTTPS production endpoints:

```bash
export RAY_SKILLS_JOBS_URL=https://skills-head.example
export RAY_VLLM_JOBS_URL=https://vllm-head.example

# Every variable named by a credential_env; NVFlow refuses to connect without it:
export RAY_SKILLS_TOKEN=...
export RAY_VLLM_TOKEN=...
```

Endpoints without bearer authentication drop `credential_env` from the target.
The harness writes these URL exports to an `endpoints.env` file for you.
Credential values stay in operator-managed environments or secret mounts.
They never belong in target YAML.
Launcher exports satisfy only launcher probes. Every
`controller_credential_env` and `controller_ca_file` must independently exist
inside the Skills/control cluster; NVFlow deliberately does not forward their
values through Ray `runtime_env`.

**✅ Done when:** every `jobs_url` and `credential_env` variable is exported on
the launcher, and every controller-side credential exists inside the control
cluster.

### 5. Verify the target

```bash
# 1. Load the target and check its routes; a static plan contacts no cluster
uv run nvflow plan --workflow starter:generate --cluster starter_ray

# 2. Reach every Jobs endpoint
uv run nvflow status --cluster starter_ray
```

**✅ Done when:** the plan loads and `status` names no unreachable cluster.

## Run workflows

Return to the main flow, [Inspect and run the starter](../../README.md#3-inspect-and-run-the-starter)
or a project's quick start, and pass the Ray target as `--cluster`, for example
`--cluster starter_ray`. Only the target changes. Do not create `_ray.yaml`
workflows or add endpoint, cluster, or scheduler fields to project workflow
YAML.

On Ray, `validate` stays structural, and `plan` checks the target and routes
without contacting the clusters. `plan --live` inspects the launcher's local
filesystem, so the workload paths must be mounted there. `run` checks the
Jobs API, Python, Ray, `NVFLOW_IMAGE_ID`, and resources of each cluster the
run uses (its Jobs' clusters and the control cluster) and the shared artifact
root, then submits one controller Job and prints a
`rayjobs://<cluster>/<submission-id>` handle. Clusters the run doesn't use may be stopped or have their endpoint
variable unset. The launcher may exit after
submission: the controller owns the child Jobs, and Ray workload scripts apply
their idempotency and resume guards on the clusters.

### Monitor and verify

List your queued and running Jobs on every cluster of the target, through its
`ssh_tunnel` when it has one:

```bash
uv run nvflow status --cluster starter_ray
```

A cluster that doesn't answer, for example one that is restarting, or whose
endpoint variable is unset is named after the table with the reason; the other
clusters' Jobs are still listed. Add `--workflow <selector>` to list only one
workflow's Jobs. Steps still waiting for their dependencies have no Job yet;
`status` lists them as `PENDING` with reason `Dependency`.

For Job details and logs, use the Ray CLI from a host that reaches the Jobs
endpoints:

```bash
RAY_API_SERVER_ADDRESS="$RAY_SKILLS_JOBS_URL" \
  uv run ray job list

RAY_API_SERVER_ADDRESS="$RAY_SKILLS_JOBS_URL" \
  uv run ray job logs <controller-submission-id>

RAY_API_SERVER_ADDRESS="$RAY_VLLM_JOBS_URL" \
  uv run ray job list
```

For endpoints with bearer authentication, add
`--headers '{"Authorization": "Bearer <token>"}'` to each command, and
`--verify <ca-file>` for a private CA.

Wait until the controller and its child Jobs are terminal before verifying
artifacts or stopping disposable clusters. Durable state is written under:

```text
/shared/nvflow/.nvflow/runs/<run-id>/
```

Each child Job also writes its output to the step's log directory, such as
`generation-logs/`, so logs outlive the cluster. Model
servers write `server_<task>_<job-id>_ray.log`, NeMo-RL drivers write
`ray-<job-id>-job.log`, and other steps write `main_<task>_<job-id>_ray.log`.
Steps without a log directory write under `<run-id>/logs/` above.

A successful Starter run writes one generated row and its completion marker.
Check them from a host that mounts `/workspace`:

```bash
test -f /workspace/outputs/starter/generate-answer/output-rs0.jsonl.done
test "$(wc -l < /workspace/outputs/starter/generate-answer/output-rs0.jsonl)" -eq 1
```

### Clean up

NVFlow stops only Jobs and services owned by its run. It never stops supplied
clusters; stop harness clusters with the harness `stop` command.

## Where settings live

| Setting | On Ray |
|---|---|
| Worker images | Each cluster's image; its `NVFLOW_IMAGE_ID` must match the target's `expected_image` |
| Mounts | Provisioned on every cluster before Ray starts, at the same absolute paths |
| Environment variables and API keys | The clusters' environment, for example KubeRay `env` or `envFrom`; never the target YAML or Ray Jobs `runtime_env` |
| Offline flags (`HF_*_OFFLINE`) | The clusters' environment |
| Account and partitions | The clusters; recipes that set them are rejected |
| Logs | Each step's log directory, such as `generation-logs/`, plus run records under `<artifact_root>/.nvflow/runs/<run-id>/` |
| Queued and running jobs | `uv run nvflow status --cluster <target>`; Job details with `ray job logs` |

## Troubleshooting

### Image, Python, Ray, or storage mismatch

`run` checks every cluster it uses before submitting anything:

- `Ray cluster '<name>' image mismatch: expected …, found …`: the cluster's
  `NVFLOW_IMAGE_ID` is not exactly the target's `expected_image` string;
  `found None` means the variable is not exported. Export it on every head and
  worker with exactly the `expected_image` value. NVFlow compares only these
  strings, not the running image, so keep the label in step with the image
  each cluster runs.
- `Python mismatch`: the image's Python differs from `expected_python`; see the
  [runtime profile](#runtime-profile).
- `requires Ray 2.58.0, found …`: the named cluster, or the NVFlow Ray
  launcher, runs another Ray version. The launcher and every head and worker
  must run Ray `2.58.0` exactly; see the [runtime profile](#runtime-profile).
- `cannot read shared artifact probe <path>`: the cluster does not mount the
  artifact root at the same absolute path as the launcher or its `ssh_tunnel`
  host.

### A Job fails to start or never becomes ready

- `Job supervisor actor failed to start within … seconds`: the cluster had no
  free resources for the Job before `RAY_JOB_START_TIMEOUT_SECONDS` (15 minutes
  by default). Raise it on the cluster head to cover the longest queue wait, or
  add capacity.
- `Service task '<task>' exceeded readiness timeout`: a model server did not
  pass its readiness check within 30 minutes. Its
  `server_<task>_<job-id>_ray.log` usually shows a model path missing under
  `/hf_models` or too few GPUs on the routed cluster.
- `Runtime probe timed out for supplied cluster '<name>'`: the two-minute check
  before submission did not finish; confirm the cluster is up and not
  saturated.

### The Jobs API rejects the launcher or the controller

- `Ray cluster '<name>' requires environment variable <VAR>`: export the
  variable that the cluster's `credential_env` names.
- `Ray Jobs … failed for supplied cluster '<name>'` with HTTP 401 or 403: the
  token is wrong or expired. The controller reads its own copy, named by
  `controller_credential_env` or else `credential_env`, inside the control
  cluster; NVFlow never forwards it.
- `requires https or allow_insecure: true`: use the HTTPS endpoint, or set
  `allow_insecure: true` for a development HTTP endpoint such as a tunneled
  one.

### Find the logs of a failed step

`run` prints the controller handle, `rayjobs://<cluster>/<submission-id>`. Its
log, from `ray job logs <submission-id>` as in
[Monitor and verify](#monitor-and-verify), names the failed child Job; that
step's log directory then holds its server and main logs.

## General target configuration

Copy [`cluster_configs/template-ray.yaml`](../../cluster_configs/template-ray.yaml)
to the operator target directory and set endpoint environment variables.

The target declares:

- A supplied control cluster that runs the durable NVFlow controller.
- One or more supplied Jobs API endpoints.
- Optional controller-facing Jobs URLs when the launcher uses a tunnel or
  different DNS. `jobs_url` is launcher-facing; `controller_jobs_url` defaults
  to it and must remain reachable after the launcher exits.
- Runtime-role routes for the project-wide v1.2 images: `nemo-skills`,
  `nemo-gym`, `nemo-rl`, `vllm`, and, for workflows that select SGLang,
  `sglang`.
- Cluster-provisioned mounts and environment. These are infrastructure
  properties, not fields that Ray Jobs can apply after the cluster is running.
- One absolute POSIX artifact root mounted at the same path on all
  participating clusters and on the launcher or its `ssh_tunnel` host.
- An optional `ssh_tunnel` block for launching from outside the cluster
  network.
- Launcher- and controller-side credential environment-variable names and
  optional TLS CA paths. Controller-specific fields default to the launcher
  fields, whose names and paths must then also exist inside the control
  cluster. Secret values never belong in target YAML or Jobs runtime
  environments.

Multiple roles may route to one cluster when its image, dependencies,
resources, and networking satisfy every role. A workflow does not need a
cluster for roles it does not select.

## Execution and ownership

The launcher uploads a deterministic bundle containing NVFlow, the installed
NeMo-Skills package, and the selected project, then submits one controller Ray
Job to the declared control cluster.
Final Ray Jobs upload archives are capped at 95 MiB, below Ray's 100 MiB
dashboard HTTP request limit. Models, datasets, checkpoints, and installed
dependencies therefore remain image/shared-storage concerns.
The controller:

1. Submits child Jobs to the cluster routed for each exact command.
2. Enforces `after_success`, `after_start`, `after_ready`, and `after_any`.
3. Records run/task/attempt transitions under
   `<artifact-root>/.nvflow/runs/<run-id>/`.
4. Stops only child jobs and services created for that run.

NVFlow never stops a supplied Ray cluster.

NeMo-RL SFT/GRPO drivers attach to the supplied training cluster and schedule
their own actors. Embedded NeMo-Gym uses that same GCS. Standalone Gym rollout
and re-verification jobs attach to the supplied Gym-capable cluster; they must
not silently start another local Ray cluster.

## Networking and security

- The launcher must reach each Jobs API endpoint for preflight.
- The control cluster must reach every child Jobs API and cross-cluster service
  endpoint.
- Use HTTPS for non-loopback endpoints. Plain HTTP requires the target's
  explicit development-only `allow_insecure: true`.
- Operator-provision workload and child-cluster credentials in the control
  cluster environment or mounted identity files. NVFlow serializes references,
  never values.

## Explicit non-goals

- Ray Client
- Dynamic Ray-on-Slurm
- Dynamic KubeRay provisioning or Kubernetes API access
- Runtime package installation
- Live source-checkout mounts
- Object-store artifact transfer
- Silent fallback to Slurm or another endpoint

Ray-on-Slurm scripts may be used to create development test clusters, but those
clusters enter NVFlow only as ordinary supplied Jobs endpoints.
