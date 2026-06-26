# Running on Ray — Install & Setup (Experimental)

> **Experimental feature.** Running the finance pipeline on **pre-provisioned Ray clusters** is
> experimental. **Slurm is the only officially supported executor.** Provisioning the Ray clusters is
> the **customer's responsibility**. The supported, worked path below is **Ray-on-Slurm**.

The finance pipeline uses the **same recipe YAML, data, and output layout** on Slurm (default) and on
pre-provisioned Ray clusters — but the **setup and run commands differ** (install extra, container
acquisition, absolute paths, and Ray-native monitoring). Do **not** assume "identical to Slurm": this
doc is the Ray setup, and the per-stage run walkthrough is
[`quick-start-ray.md`](docs/recipes/finance/quick-start-ray.md) (the Ray counterpart of the Slurm
`quick-start.md`).

## The Ray path at a glance

Step 1 is the shared install; the Ray delta starts at the images and the real divergence is the
cluster (Steps 3–4). Follow this doc top-to-bottom — it says when to hop to a shared doc and back.

1. **Install nvflow** — clone + `uv sync` → [`README.md`](README.md)
2. **Build & stage the images** → [`INSTALL.md`](INSTALL.md) — plus the Ray-specific image notes in Step 2 below
3. **Provision your Ray cluster(s)** — Step 3 below
4. **Write the cluster config** (`backend.name: ray`) — Step 4 below
5. **Run the pipeline (SDG → SFT → GRPO), per stage** → [`quick-start-ray.md`](docs/recipes/finance/quick-start-ray.md) (set `cluster: my_cluster` in your recipe)

Steps 2–4 below are the Ray delta; steps 1 and 5 hop to the shared docs.

## What changes vs Slurm (the whole delta)

| | Slurm (default) | Ray (experimental) |
|---|---|---|
| Cluster config | `executor: slurm` (+ `cpu_partition`) | `backend: { name: ray, gpu_nemo_rl_dashboard_url[, cpu_nemo_skills_dashboard_url] }` |
| Who provisions compute | nvflow submits `sbatch` | **you** pre-provision Ray; nvflow is a Jobs-API client |
| Generation / judge endpoints | can self-host in-job | **external / BYO** OpenAI-compatible HTTP |
| Container format | **`.sqsh`** (enroot/pyxis) | **`.sqsh`** (same as Slurm) |
| Recipe YAML / data / output layout | — | **same** (recipe sets `cluster:`) |
| Run walkthrough | `quick-start.md` | **`quick-start-ray.md`** (Ray-native fire + verify) |
| Install extra | `uv sync` | **`uv sync --extra skills`** (Ray backend lives in nemo_skills) |
| Output / data paths | relative OK | **absolute** on the shared mount (e.g. `/lustre/...`) |
| Monitoring | `squeue` / `sinfo` | `ray job list --address <dashboard>` / dashboard Jobs tab |

## Topology (2 clusters)

```
   orchestrator (nvflow + nemo_skills, a Ray Jobs API client; runs anywhere with HTTP to the dashboards)
        │  Ray Jobs API (HTTP)
        ├──────────────► GPU cluster: nemo-rl image     — SFT/GRPO training, checkpoint convert
        └──────────────► CPU cluster: nemo-skills image — data-prep, eval client / judge / score

   generation + judge = external BYO OpenAI-compatible HTTP endpoints (NOT Ray members)
```
One image (one Python minor + Ray version) per cluster — Ray checks the Python minor at node join.

---

## Step 1 — Install nvflow (shared with Slurm)

Clone the repo and install — see [`README.md`](README.md). **Ray difference:** run
`uv sync --extra skills`, **not** the bare `uv sync`. The Ray Jobs backend lives in `nemo_skills`,
which the default `uv sync` deliberately omits (to keep the base airgap-CI-safe); without the `skills`
extra the orchestrator can't import the backend and `ray` isn't on PATH (use `uv run ray ...` /
`uv run nflow ...` so the project venv is active). See Step 5 for the in-container alternative.

## Step 2 — Container images (the Ray delta)

Build and stage the images per [`INSTALL.md`](INSTALL.md) — the build is identical to Slurm. Two
Ray-specific points:

> **`<YOUR_REGISTRY>` is a placeholder.** The image refs below are not customer-pullable as written
> — substitute your own registry. Get the images one of two ways: **build** them per `INSTALL.md`
> (your build tags, e.g. `nvflow-nemo-rl:v0.6.0`), or **pull the NVIDIA-delivered** images
> (`nvflow-nemo-skills:v1.1.1-ray`, `nvflow-nemo-rl:v0.6.0-airgap`) and push them into your own
> registry. Use whichever tags you actually pushed — the example configs use placeholder tags, so
> reconcile them with your build/pull step (see `cluster_configs/containers.yaml`).

- **The nemo-skills image MUST *install* the Ray Jobs backend so it is importable** — nemo_skills must
  be installed in the image's Python, not merely present as a source checkout. Verify inside the image
  (both imports must succeed):
  ```bash
  docker run --rm <YOUR_REGISTRY>/nvflow-nemo-skills:v1.1.1-ray \
    python -c "import nemo_skills; from nemo_skills.pipeline.utils.backends import get_execution_backend; print('ok')"
  ```
  If `import nemo_skills` fails — e.g. the source sits at `/opt/NeMo-Skills` but was never
  `pip install`-ed — the Ray backend will not load even though the files are on disk.
- For **Ray-on-Slurm**, you can pass the **registry ref** directly to the starter (the compute node
  pulls + builds it), or pre-build a `.sqsh` **on a compute node** for instant boots. Do **not**
  `enroot import` on the login node with the temp dir on Lustre — it fails (overlayfs on a network FS).
  See Step 3's image-staging note for both paths.

## Step 3 — Provision the Ray cluster(s)

nvflow needs only a **reachable Ray dashboard URL** per cluster. Per-cluster contract:
- **CPU head** (`--num-gpus 0`, control plane) + workers; dashboard bound to `0.0.0.0:<port>`.
  It must land on a **CPU partition** (e.g. `cpu_interactive`): a GPU partition like `interactive`
  commonly **rejects** a 0-GPU job (`--gpus-per-node=0`) outright, so it never schedules.
- One image / one runtime per cluster (Python-minor + Ray version checked at join).
- Shared storage mounted at the same path on every node (models / data / outputs / logs).

### Ray-on-Slurm (supported)

Use the provided starter — run it **once per cluster**. The GPU cluster uses the **nemo-rl** image
(Ray lives in its `/opt/nemo_rl_venv`); the CPU cluster uses the **nemo-skills** image (system Python):

```bash
# IMAGE is what the starter passes to pyxis --container-image. It can be EITHER a registry
# URI (docker://nvcr.io#<ORG>/<image>:<tag>) — pulled+built on the compute node — OR a prebuilt
# .sqsh path. See the image-staging note below for which to use (the registry URI is the simple path).
#
# MOUNTS (comma-separated host:container) is what the head container sees — it defaults to ONLY
# /lustre:/lustre, so pass MOUNTS= to add anything a recipe references. Every path a recipe uses
# (the /workspace repo mount and whatever `hf_model_path` points at) MUST resolve INSIDE the head
# container, because a pre-provisioned cluster's mounts come from THIS bring-up, NOT from the
# recipe / my_cluster.yaml. In particular the GRPO smoke sets hf_model_path=/hf_models/Qwen/Qwen3-4B,
# so the head needs an /hf_models mount (else vLLM can't find the model locally and HuggingFace
# rejects the absolute path as a repo id — see troubleshooting). Alternatively, leave /hf_models out
# and set the recipe's hf_model_path to an absolute path under the already-mounted /lustre.
# GPU cluster (nemo-rl): training + rollout
IMAGE=docker://nvcr.io#<YOUR_REGISTRY>/nvflow-nemo-rl:v0.6.0-airgap DASHBOARD_PORT=8265 RAY_VENV=/opt/nemo_rl_venv \
  MOUNTS=/lustre:/lustre,<CLUSTER_PATH_TO_NVFLOW_REPO>:/workspace,<CLUSTER_PATH_TO_HF_MODELS>:/hf_models \
  sbatch --account=<acct> --partition=<gpu_partition> --nodes=1 --gpus-per-node=8 \
         scripts/start_ray_on_slurm.sb

# CPU cluster (nemo-skills): data-prep + eval
IMAGE=docker://nvcr.io#<YOUR_REGISTRY>/nvflow-nemo-skills:v1.1.1-ray DASHBOARD_PORT=8266 \
  sbatch --account=<acct> --partition=<cpu_partition> --nodes=1 --gpus-per-node=0 \
         scripts/start_ray_on_slurm.sb
```
**The head IP changes on every bring-up — re-read it and update `dashboard_url` each time.** After
`sbatch`, the job logs to `nvflow-ray-head-<jobid>.out`; once the head is up it prints one line:
```bash
grep 'RAY HEAD READY' nvflow-ray-head-<jobid>.out
# RAY HEAD READY dashboard_url=http://<head-ip>:8265
```
Copy that exact `http://<head-ip>:<port>` into the matching `dashboard_url` field of
`cluster_configs/my_cluster.yaml` (Step 4) **before** running anything. The loop, every time: bring up
head → `grep 'RAY HEAD READY'` → set `dashboard_url` → run. A stale value (a previous bring-up's IP)
fails the next command with `Error: Failed to connect to Ray at address: http://<old-ip>:8265` — the
fix is always to re-read `RAY HEAD READY` and update `dashboard_url` to the **current** head. `scancel`
the job to tear a cluster down.

> **Image staging — two paths that work; pick one. Do NOT `enroot import` on the login node.**
> A login-node `enroot import` to a `.sqsh` on Lustre **fails**: enroot's squashfs build uses
> overlayfs, which **cannot mount with a network FS (Lustre) as its layer dirs**
> (`enroot-mksquashovlfs: failed to mount overlay: ... Invalid argument`); redirecting enroot's temp
> to login-node `/tmp` then fails the other way — too small (`zstd: No space left on device`). So:
>
> - **(a) Boot from the registry ref — no `.sqsh`, recommended simple path.** Set
>   `IMAGE=docker://nvcr.io#<YOUR_REGISTRY>/nvflow-nemo-rl:v0.6.0-airgap` (pyxis/enroot accepts a
>   registry URI for `--container-image`). The **compute node** pulls + builds on its **local NVMe
>   scratch** — which supports overlayfs and has space — sidestepping the login-node trap. Trade-off:
>   a ~10–12 min image pull at job start on each fresh node (vs instant boot from a `.sqsh`). On an
>   idle-exempt / interactive partition the head survives the pull (no reaper).
> - **(b) Reusable instant-boot `.sqsh` — build it on a COMPUTE node, not the login node.** Run the
>   import inside an `srun`/batch job on a GPU/CPU compute node (local scratch supports overlayfs),
>   then move the resulting `.sqsh` to Lustre and point `IMAGE=` at it for instant boots thereafter:
>   ```bash
>   srun --account=<acct> --partition=<gpu_partition> --nodes=1 --gpus-per-node=8 \
>     enroot import -o $TMPDIR/nvflow-nemo-rl.sqsh \
>       docker://nvcr.io#<YOUR_REGISTRY>/nvflow-nemo-rl:v0.6.0-airgap
>   #   then copy $TMPDIR/nvflow-nemo-rl.sqsh -> /lustre/<you>/containers/ and use that path as IMAGE=
>   ```

**Confirm each cluster is reachable before running anything:**
```bash
curl -sf http://<head>:8265/api/version                        # -> {"ray_version": "...", ...}
uv run ray job submit --address http://<head>:8265 -- echo ok  # -> SUCCEEDED (use `uv run`; see Step 1)
```

### Slurm idle-GPU reaping (read before you bring up the GPU head)

A Ray-on-Slurm head is a **long-lived** sbatch job that holds its GPUs even while idle,
waiting for the orchestrator to submit work. Many Slurm clusters run an **idle-GPU
reaper** that cancels jobs whose GPUs stay below a utilization threshold for too long on
ordinary batch partitions — which would tear your Ray head down mid-run.

Place the long-idle component on a partition that is **exempt from idle reaping** (often
`interactive` or a debug/dev QOS — confirm with your admin / `scontrol show partition`):

- **GPU Ray head → idle-exempt partition.** In a 2-cluster setup with *external*
  judge/policy servers, the GPU head's GPUs stay idle through data-prep and rollout
  collection and only light up at the training stage — so the head is the job most at
  risk. This is the one you protect.
- **CPU Ray head → any *CPU* partition.** It holds no GPUs, so the GPU reaper doesn't apply — but
  it must still be a CPU partition (a GPU partition rejects the 0-GPU job; see the contract above).
- **Busy self-hosted GPU servers (judge / policy vLLM) → a normal GPU partition.** They
  run hot during rollouts/training, so they aren't reaping targets, and the idle-exempt
  partition usually has tight caps (below) you don't want to spend on them.

> **Idle-exempt partitions usually cap concurrency and walltime.** A `MaxJobsPU` /
> `MaxJobs` limit (sometimes 1) and a shorter max walltime are common. Keep only the
> *one* long-idle component (the GPU head) there; run everything else elsewhere, and size
> the sbatch `--time` below the partition's walltime cap.

If your cluster has no idle-exempt partition, use whatever idle-exemption mechanism it
provides (a reservation, a `no-reap` QOS, or an admin exception) for the GPU head. The
symptom of getting this wrong is the GPU head being `scancel`-ed by a service account a
few minutes into an otherwise-healthy run.

## Step 4 — Write the cluster config

A cluster config describes a **cluster**, never the workflow — no stage list, reusable across recipes.
Copy the template (the filled file is git-ignored, exactly like the Slurm `my_cluster.yaml`):
```bash
cp cluster_configs/template-ray.yaml cluster_configs/my_cluster.yaml
```
```yaml
# cluster_configs/my_cluster.yaml   (one config, both clusters)
executor: none   # REQUIRED even in Mode 3 — nemo-skills reads it outside the Ray path; NOT "ray". The Ray backend submits.
backend:
  name: ray
  gpu_nemo_rl_dashboard_url:     http://<gpu-cluster-head>:8265   # GPU stages: training, rollout
  cpu_nemo_skills_dashboard_url: http://<cpu-cluster-head>:8266   # CPU stages: data-prep, eval
containers:        # keys resolve each stage's `container:`; paths unused on a precreated cluster
  nemo-rl:     <YOUR_REGISTRY>/nvflow-nemo-rl:v0.6.0-airgap
  nemo-skills: <YOUR_REGISTRY>/nvflow-nemo-skills:v1.1.1-ray
mounts:
  - <CLUSTER_PATH_TO_NVFLOW_REPO>:/workspace   # MUST be the nvflow repo ROOT — the dir containing both scripts/ and the nvflow/ package, NOT the nvflow/ package subdir (driver imports nvflow + recipes from here, writes outputs under it)
  - <CLUSTER_PATH_TO_HF_MODELS>:/hf_models
env_vars:
  - PYTHONPATH=/workspace  # REQUIRED: every stage runs `python -m nvflow…` in the job; the image omits nvflow, so this puts the repo mount on the job PYTHONPATH (else `ModuleNotFoundError: No module named 'nvflow'`)
  - OPENAI_API_KEY         # judge key forwarded into each Ray job; GRPO/gym judge reads this by default
  # - NVIDIA_API_KEY       # use instead for the nemo-skills eval judge against api.nvidia.com
```
> **`PYTHONPATH=/workspace` is not optional.** Every recipe stage (data-prep, sdg, sft, rl, eval) submits a
> `python -m nvflow.…` job to the cluster. The deliverable image bakes `nemo_skills` but **not** `nvflow`
> (by design — see Step 5), and the Ray Jobs backend ships no working-dir, so the job can only find
> `nvflow` on the mounted `/workspace` checkout. Forwarding `PYTHONPATH=/workspace` via `env_vars` is what
> makes that import resolve for the submitted job (the driver-side `PYTHONPATH` in Step 5 does not reach it).
**`run-all` auto-routes each stage by its `num_gpus`** — exactly as Slurm splits `partition` /
`cpu_partition`: GPU stages → the GPU cluster, CPU stages → the CPU cluster. A stage may set
`target_cluster: cpu|gpu` to override. The **post-training** eval stage (in `sft/`, `grpo/`) ships
with `target_cluster: cpu` because its generation server is an external endpoint. The **standalone
baseline eval** (`eval/demo.yaml`) does **not** — its model stages carry `gpus: 4` (the Slurm prehost
count), so on a two-dashboard config they route to the **GPU** cluster. To run baseline eval against an
external server on the CPU cluster, set `target_cluster: cpu` + `gpus: 0` + `server_address` on those
model stages. On the **host-driver** path (`uv run nflow` on a login node), the eval also imports the
benchmark module, checks the data file, and writes the job manifest **driver-side** (a consequence of
`executor: none`), so `datasets_dir`, `base_output_dir`, and `stages.prepare_data.output_dir` must be
absolute paths visible on the driver too — use `/lustre/...`, never `/workspace/...` (in-container only;
on a login node it fails with `No module named '<benchmark>'` and `Permission denied: '/workspace'`). The
recipe defaults ship `/workspace`; override them, and point `prepare_data.output_dir` at the same
`datasets_dir`. **Single Ray cluster instead?** Set
`dashboard_url: http://<head>:8265` — the field the Ray Jobs backend reads directly, so it works for
both single-stage `nflow run <stage>` and `run-all`. (A lone `gpu_nemo_rl_dashboard_url` only takes
effect through `run-all`'s per-stage routing; a single-stage `nflow run` would not populate the
dashboard and fails with `use_with_ray_cluster is only supported for SlurmExecutor`.)

**GRPO lane (experimental, single-node smoke).** To run GRPO on one 8-GPU nemo-rl Ray cluster, see
[`quick-start-ray.md` → Step 5](docs/recipes/finance/quick-start-ray.md) and the ready-to-edit
[`grpo/qwen3_4b_smoke.yaml`](nvflow/recipes/finance/workflows/grpo/qwen3_4b_smoke.yaml). Five
GRPO-specific edits beyond the shared `/lustre`-paths + `PYTHONPATH=/workspace` contract: (1) scope to
ONE environment with the `-e <name>` CLI flag — **required**, and the only mechanism that scopes a run;
a top-level `_environment:` recipe key is not honored (else `grpo/base.yaml`'s three envs all fan out and
`finance_sec_search` fails with `Missing local datasets`); (2) point `judge_vllm` at an
**external/API** judge (`num_gpus: 0` + `openai_base_url` + `openai_model` +
`openai_api_key: ${oc.env:NVIDIA_API_KEY}`), not the in-job vLLM judge; (3) set `training.total_gpus: 8`
for one node (shipped configs use 16 = 2 nodes); (4) skip SDG by dropping a pre-made `final_result.jsonl`
under `step-3-convert-to-responses-api/<env>/`; (5) the rollout policy-vLLM serve uses a patched wrapper
whose default `/nemo_run/code/scripts/serve_vllm_patched.py` path is empty on Ray Mode-3 — set
`NVFLOW_SERVE_VLLM_ENTRYPOINT=/workspace/scripts/serve_vllm_patched.py` in the cluster `env_vars` (or
`collect_rollouts.rollout.policy_vllm.server_entrypoint`), else it fails on `//./scripts/serve_vllm_patched.py`.
That wrapper then runs `python3 -m vllm` under the head's `RAY_VENV` (`/opt/nemo_rl_venv`), which has **no
vLLM** — the nemo-rl image keeps vLLM in a per-actor venv under `/opt/ray_venvs/<hash>/`. The wrapper
auto-discovers a vLLM-capable interpreter by scanning `/opt/ray_venvs/*/bin/python*`; if your layout differs,
set `NVFLOW_VLLM_PYTHON=/opt/ray_venvs/<hash>/bin/python` (or `NVFLOW_VLLM_VENV=<dir>`, or override the scan
with `NVFLOW_VLLM_VENV_GLOB`) in `env_vars`. On Slurm / in-container (vLLM already importable) no setting is
needed — behavior is unchanged. The GPU head must sit on an idle-exempt / non-preemptible partition to
survive training.

### Generation + judge endpoints (BYO / self-host)

*Only for stages that call a judge or generation server — eval, GRPO, compute_rewards. Skip it for the hello-world smoke and SFT-only runs.*

External, OpenAI-compatible HTTP endpoints **you** supply — never an NVIDIA-internal endpoint.

| Judge mode | `judge_vllm` config | Spawns a server? |
|---|---|---|
| Self-host pre-launched vLLM | `base_url: http://<serve>/v1` + `model: <name>` | no (external) |
| BYO OpenAI-compatible API | `openai_base_url: https://<api>/v1` + `openai_model: <id>` | no (external) |

Worked example (key reaches the Ray job via the cluster config `env_vars` above):
```yaml
# in the eval / grpo override
judge_vllm:
  openai_base_url: https://<your-api>/v1
  openai_model: <judge-model-id>
  openai_api_key: ${oc.env:OPENAI_API_KEY}  # ${oc.env:...}, NOT a literal "$OPENAI_API_KEY"
```
For **generation**, serve your checkpoint with vLLM and pass `--server_address http://<serve>/v1` to
eval. Use a **dedicated** judge endpoint for sustained batch judging (a shared endpoint rate-limits/429s);
in GRPO `training` the per-step judge burst can `429` and stall the run — see
[`quick-start-ray.md` → Troubleshooting](docs/recipes/finance/quick-start-ray.md).

The **GRPO / gym judge** defaults to `$OPENAI_API_KEY` (omit `openai_api_key` and it reads the env var).
The **eval judge is configured separately** and defaults to `gpt-5-chat-latest` over
`https://api.openai.com/v1` (see `workflows/eval/base.yaml`) — so out of the box it needs
`OPENAI_API_KEY` and outbound access. To judge with a **self-hosted / BYO** endpoint instead, override
the eval `judge:` block (`server_type: openai` + `server_address: http://<serve>/v1` + `model: <id>`).
For an NVIDIA-hosted eval judge against `api.nvidia.com`, set `NVIDIA_API_KEY` (auto-picked — omit the key).

---

## Step 5 — Smoke, then rejoin the Quick Start

`nflow` is the orchestrator — it submits jobs to the Ray dashboard(s) over HTTP and needs **both
nvflow and nemo_skills importable**. Two supported ways to run it:

- **Source checkout (dev / internet) — the Quick Start path, no mount:** `uv sync --extra skills`,
  then `uv run nflow …` from the checkout. The `skills` extra pulls nemo_skills (the default
  `uv sync` omits it, so the Ray backend won't load without it). `nflow` is only an HTTP client to
  the dashboard — it needs **no container** of its own.
- **Inside the nemo-skills container (airgap / prod):** the image bakes nemo_skills + all heavy deps
  but **not nvflow** (intentionally — nvflow is small and pure-Python). Mount your nvflow checkout
  into the container and put it on `PYTHONPATH`; its deps are already in the image, so no install is
  needed. There is no `nflow` console-script in the image, so invoke the module directly:
  ```bash
  # e.g. srun --container-image=<nemo-skills>.sqsh --container-mounts=<nvflow_checkout>:/nvflow,... \
  PYTHONPATH=/nvflow python -m nvflow.cli.main run-all --config <recipe>
  ```

Validate the Jobs-API path first (`--config` is **cwd-relative** — pass the full path from the repo
root, e.g. `nvflow/recipes/...`, not a short `grpo/...` form, which fails with `No such file or directory`):
```bash
uv run nflow run-all --config nvflow/recipes/finance/workflows/ray_hello_world.yaml
# -> "Ray job ... finished with status SUCCEEDED"
```
The target cluster is selected by the recipe's **`cluster:` field** (e.g. `ray_hello_world.yaml`
sets `cluster: my_cluster`), resolved to `cluster_configs/<name>.yaml` via `NEMO_SKILLS_CONFIG_DIR`
or `./cluster_configs/` — there is **no `--cluster` flag**.

Then run the pipeline per stage following
[`quick-start-ray.md`](docs/recipes/finance/quick-start-ray.md) — the Ray counterpart of the Slurm
`quick-start.md`. It uses the **same recipes**, but with Ray-native fire commands (absolute output
paths on the shared mount) and Ray-native verification (`ray job list` / dashboard Jobs tab) in place
of `squeue`/`sinfo`. With a two-dashboard config, **one `run-all` fans each stage to the right cluster
automatically**:
```bash
uv run nflow run-all --config nvflow/recipes/finance/workflows/sft/qwen3_4b.yaml   # the recipe's cluster: field selects the cluster
```

## Parallel execution & GRPO on Ray

- **Parallelism is the default.** The Ray backend submits independent jobs **concurrently**; eval's
  seeds/benchmarks fan out as parallel Ray jobs (watch them on the Ray dashboard *Jobs* tab or
  `ray job list --address http://<head>:8265`). Throughput is bounded by cluster GPUs + the judge
  endpoint's rate limit — use a dedicated judge to run the fan-out at full width.
- **GRPO (experimental)** submits its stages on Ray the same way as the rest of the pipeline (each
  stage a Ray job). Env-scoping + the data-prep path are validated; the full rollouts + training run on
  Ray is still being validated, so run it per stage from the single-node smoke
  ([`quick-start-ray.md` → Step 5](docs/recipes/finance/quick-start-ray.md)) rather than as one
  unattended `run-all`. Keep the judge **external** (`base_url`/`openai_base_url`); a self-hosted
  *in-job* judge is not supported on Ray.

## Troubleshooting

See [`docs/recipes/finance/troubleshooting.md`](docs/recipes/finance/troubleshooting.md). Quick hits:
- **`submit_job() got an unexpected keyword argument 'entrypoint_label_selector'`** — the orchestrator's
  Ray is too old. The Jobs-API client must be **Ray ≥ 2.54.0**; a bare resolve can pin 2.53.0. The repo
  now floors `ray[default]>=2.54.0` and pins nemo_skills to a build that passes the kwarg conditionally,
  so a fresh `uv sync --extra skills` is fixed. If you still hit it, your env is stale —
  `uv sync --extra skills` (or `uv pip install "ray[default]>=2.54.0"`). Exact-version match across the
  two cluster images is **not** required (both speak Jobs REST `version: 4`); only the ≥2.54.0 floor matters.
- **Dashboard unreachable** — the orchestrator needs HTTP to each cluster's dashboard; check
  host/port + network policy.
- **`Error: Failed to connect to Ray at address: http://<ip>:8265`** — the `dashboard_url` in
  `my_cluster.yaml` points at a **stale head IP** (the head IP changes on every bring-up). Re-read the
  current head from `grep 'RAY HEAD READY' nvflow-ray-head-<jobid>.out` and update `dashboard_url`
  (Step 3 → Step 4) to that IP, then re-run.
- **Python-minor mismatch at join** — a worker whose image Python minor differs from the head won't
  join. One image per cluster.
- **GPU Ray head `scancel`-ed mid-run by a service account** — your cluster's idle-GPU
  reaper killed the head for holding idle GPUs on a non-exempt partition. Bring the GPU
  head up on an idle-exempt partition (e.g. `interactive`), or use the cluster's
  idle-exemption mechanism; keep busy self-hosted servers (judge / policy vLLM) on a
  normal GPU partition. See **Step 3 → Slurm idle-GPU reaping**.
- **Judge auth fails in a Ray job** — Ray jobs run with the **cluster head's** environment. Put the
  key in the cluster config `env_vars:` (Step 4) **and** ensure your cluster bring-up exports it
  before `ray start`. Local-only / recipe-only keys won't reach the job.
- **`Failed to merge the Job's runtime env ... because of a conflict`** (GRPO / rollout stages) —
  the NeMo-RL training driver calls `ray.init()` again with its own runtime_env, which collides with
  the job-level runtime_env nvflow forwards whenever a key (`OPENAI_API_KEY`, `HF_HOME`, ...) is in
  both. nvflow sets `RAY_OVERRIDE_JOB_RUNTIME_ENV=1` on every submitted job so Ray merges them
  automatically (handled in the container) — you should not see this. If you submit NeMo-RL jobs
  outside nvflow, set that env var on the cluster head yourself.
- **`HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/hf_models/...'`**
  (GRPO / rollout / training) — the model path the recipe references (`hf_model_path`, e.g.
  `/hf_models/Qwen/Qwen3-4B`) isn't mounted **inside the head container**, so vLLM can't find it
  locally and HuggingFace then tries (and refuses) to treat the absolute path as a Hub repo id. A
  pre-provisioned cluster's mounts come from the **head bring-up** `MOUNTS`, not the recipe /
  `my_cluster.yaml`. Add the HF-models dir to the bring-up — `MOUNTS=…,<host>/hf_models:/hf_models`
  (Step 3) — so it resolves at the path the recipe expects, **or** set `hf_model_path` to an absolute
  path under the already-mounted `/lustre` (e.g. `/lustre/<you>/hf_models/Qwen/Qwen3-4B`).
- **`ValueError: No available memory for the cache blocks. Try increasing gpu_memory_utilization ...`**
  (GRPO rollout, single node) — two concurrent rollout vLLM resource servers collided on GPU memory.
  `num_random_seeds: N` spawns N concurrent rollout servers (`rs0`, `rs1`, …), each TP=2 and each
  grabbing vLLM's default `gpu_memory_utilization` (~0.9); on one 8-GPU node two of them land on the
  same GPUs and the second has no room for its KV cache. For a single-node smoke set
  `num_random_seeds: 1` (the shipped `grpo/qwen3_4b_smoke.yaml` now does — per-prompt reward variance
  comes from `num_generations_per_prompt`, not seeds), or give the co-located servers room with a
  lower `gpu_memory_utilization` / smaller `max_model_len`.
- **Policy serve OOMs at engine init on a RETRY** (`ValueError: Free ...` at vllm
  `multiproc_executor.py:800` / `EngineCore failed to start` / `RuntimeError: Engine core
  initialization failed`; `nvidia-smi` shows ~76 GB held by `VLLM::Worker` with no active rollout) — a
  **prior failed `collect_rollouts` attempt orphaned the policy serve's vLLM tensor-parallel workers**
  (`VLLM::Worker_TP0/TP1`) on the reused head, leaving GPU memory pinned so the next serve can't init.
  Not a code bug — stale GPU state. Clear the leaked workers by PID, then re-run (do **not**
  `pkill -f 'vllm|EngineCore'` — it matches the kill command's own process and kills your `srun` step):
  ```bash
  srun --jobid=<head-jobid> --overlap bash -c \
    'nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | xargs -r kill -9 2>/dev/null; \
     sleep 6; nvidia-smi --query-gpu=index,memory.used --format=csv'   # expect all GPUs ~0 MiB
  ```
  If the GPUs won't clear, `scancel` + re-bring-up the head for guaranteed-clean GPUs. See
  [`quick-start-ray.md` → Troubleshooting](docs/recipes/finance/quick-start-ray.md).
- **Ray placement group not released after a driver Ctrl-C** (next training attempt times out in
  `_create_placement_groups_internal` with `TimeoutError: ... resources may be busy ...` after ~3 min) —
  pressing Ctrl-C on the local `nflow` driver stops the *local* client, **not** the submitted Ray *job*.
  The job stays `RUNNING` on the cluster and keeps holding its 8-GPU placement group, so the next attempt
  can't reserve GPUs. Clearing GPU *memory* (the leaked-`VLLM::Worker` fix above) does not help — Ray's
  *logical* resource reservation is still held by the live job. Stop the stale job explicitly:
  ```bash
  uv run ray job list --address http://<gpu-head>:8265        # find the still-RUNNING <submission_id>
  uv run ray job stop --address http://<gpu-head>:8265 <submission_id>
  ```
  (Or use the dashboard **Jobs** tab.) Prefer the version-tolerant `ray job ...` commands over
  `ray status`, which does a **strict Ray-version check** and fails when the cluster's Ray (e.g. 2.54)
  differs from your local venv's Ray (e.g. 2.55) — or address the GCS endpoint directly at
  `<gpu-head>:6379`. On a login node without the `ray` CLI, drive the same operations through the
  dashboard REST API at `http://<gpu-head>:8265/api/jobs/`. See
  [`quick-start-ray.md` → Troubleshooting](docs/recipes/finance/quick-start-ray.md).
