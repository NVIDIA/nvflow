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

1. **Choose the driver** — `nvflow-client` for airgapped/prod; clone + `uv sync` for source/dev
2. **Build & stage the images** → [`INSTALL.md`](INSTALL.md) — plus the Ray-specific image notes in Step 2 below
3. **Provision your Ray cluster(s)** — Step 3 below
4. **Write the cluster config** (`backend.name: ray`) — Step 4 below
5. **Run the pipeline (SDG → SFT → GRPO), per stage** → [`quick-start-ray.md`](docs/recipes/finance/quick-start-ray.md) (set `cluster: my_cluster` in your recipe)

Steps 2–4 below are the Ray delta; steps 1 and 5 hop to the shared docs.

## What changes vs Slurm (the whole delta)

| | Slurm (default) | Ray (experimental) |
|---|---|---|
| Cluster config | `executor: slurm` (+ `cpu_partition`) | `backend: { name: ray, gpu_nemo_rl_dashboard_url[, cpu_nemo_skills_dashboard_url][, cpu_nemo_gym_dashboard_url (GRPO)] }` |
| Who provisions compute | nvflow submits `sbatch` | **you** pre-provision Ray; nvflow is a Jobs-API client |
| Generation / judge endpoints | can self-host in-job | **external / BYO** OpenAI-compatible HTTP |
| Container format | **`.sqsh`** (enroot/pyxis) | **`.sqsh`** (same as Slurm) |
| Recipe YAML / data / output layout | — | **same** (recipe sets `cluster:`) |
| Run walkthrough | `quick-start.md` | **`quick-start-ray.md`** (Ray-native fire + verify) |
| Driver | source checkout or `nvflow-client` | **`nvflow-client` for airgapped runs** (baked code archive); source + `uv sync` for development |
| Output / data paths | relative OK | **absolute** on the shared mount (e.g. `/lustre/...`) |
| Monitoring | `squeue` / `sinfo` | `ray job list --address <dashboard>` / dashboard Jobs tab |

## Topology (2 clusters; **3 for the GRPO rollout path**)

```
   orchestrator (nvflow-client: nvflow + nemo_skills + immutable code archive; HTTP to dashboards)
        │  Ray Jobs API (HTTP)
        ├──────────────► GPU cluster: nemo-rl image     — SFT/GRPO training, checkpoint convert
        ├──────────────► CPU cluster: nemo-skills image — data-prep, eval client / judge / score
        └──────────────► CPU cluster: nemo-gym image    — GRPO rollout stages (gym CLI; NO vLLM)   ← GRPO only

   generation + judge + BYO policy serve = external OpenAI-compatible HTTP endpoints (NOT Ray members)
```
One image (one Python minor + Ray version) per cluster — Ray checks the Python minor at node join.

### The 3-cluster GRPO path (recommended for GRPO on Ray)

SFT and eval need only the two clusters above. **GRPO rollouts add a third Ray cluster** running the
**nemo-gym** image, because the GRPO rollout stages (`collect_rollouts`, `prepare_data`,
`prefetch_cache`, `compute_rewards`) require the baked **`gym` CLI** — which lives *only* in the
nemo-gym image, not in nemo-skills. Each cluster carries its own dashboard URL in the cluster config
`backend:` block:

| Ray cluster | Image | Dashboard key | Stages it runs |
|---|---|---|---|
| GPU / training | **nemo-rl** | `gpu_nemo_rl_dashboard_url` | SFT / GRPO training, checkpoint convert |
| CPU / data + eval | **nemo-skills** | `cpu_nemo_skills_dashboard_url` | data-prep, split, eval client / judge / score |
| CPU / rollouts | **nemo-gym** | `cpu_nemo_gym_dashboard_url` | GRPO rollouts (`collect_rollouts`, `prepare_data`, `prefetch_cache`, `compute_rewards`) |

`collect_rollouts` routes to the nemo-gym cluster via the recipe's `target_cluster: gym` (or when a
stage sets `container: nemo-gym` and requests no GPUs). **If `cpu_nemo_gym_dashboard_url` is missing,
that stage silently falls back to the nemo-skills CPU cluster — which lacks the gym CLI — and fails at
runtime.** The stock `template-ray.yaml` ships this key **commented out**, so you must uncomment and
set it for the GRPO path (see Step 4 and Troubleshooting).

### External endpoints (NOT a 4th Ray cluster)

Three OpenAI-compatible HTTP endpoints hang off the side of the clusters. **None of them is a Ray
member** — each is a plain HTTP endpoint you point a URL at, exactly like the judge:

- **Judge** — external / hosted, `num_gpus: 0`. NVIDIA-hosted (`https://integrate.api.nvidia.com/v1`
  + an `nvapi-` key) **or** OpenAI (`https://api.openai.com/v1` + an `sk-` key). See Step 4.
- **BYO policy serve (GRPO rollouts)** — a standalone OpenAI-compatible **vLLM** endpoint on a GPU
  node, served from the **vllm-grpo** container. This is the key to why the rollout works on the
  vLLM-less nemo-gym image: the gym `vllm_model` adapter is a **pure HTTP client**, so setting
  `policy_vllm.base_url` (+ `num_gpus: 0`) makes nvflow **skip** launching an in-gym vLLM
  (`need_policy_server = not policy_vllm.base_url`, verified in `nvflow/lib/rl/rollout.py`) — no vLLM
  and no GPU needed inside the gym image. Without a `base_url`, a *local* policy vLLM
  (`policy_vllm.num_gpus: 2`) would need the gym CLI **and** vLLM in one image, which nemo-gym does not
  have — **the image gap**. See Step 3 (bring-up) and Step 4 (recipe wiring).
- **SDG generation serve** — external OpenAI-compatible endpoint for the SDG LLM stages (Step 4).

---

## Step 1 — Choose the NVFlow driver

For an **airgapped or release-qualified run**, use the `nvflow-client` image built from
`dockerfiles/Dockerfile.nvflow`. It bakes the locked venv, the exact committed NVFlow snapshot, and a
compact `/opt/nvflow-ray-code.zip`. Run `uv run --no-sync nflow ...` inside that image as shown in
Step 5. The Ray Jobs client uploads only that archive; workers perform no pip/uv install and do not
mount a host checkout.

For **source development**, clone the repo and run a bare `uv sync` as described in
[`README.md`](README.md). `nemo_skills` carries the Ray Jobs backend as a base dependency. Create a
tracked-only archive with `git archive --format=zip -o /tmp/nvflow-ray-code.zip HEAD`, point
`backend.working_dir` at it, and invoke `uv run --no-sync nflow ...`. This source path is for development, not
airgap qualification.

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
- **The nvflow-client image MUST contain the immutable Ray code archive.** Verify it without network
  or a source mount:
  ```bash
  docker run --rm --network=none <YOUR_REGISTRY>/nvflow-client:<tag> \
    python3 -c "import zipfile; z=zipfile.ZipFile('/opt/nvflow-ray-code.zip'); assert 'nvflow/__init__.py' in z.namelist(); print('ok')"
  ```
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
(Ray lives in its `/opt/nemo_rl_venv`); the CPU cluster uses the **nemo-skills** image (system Python);
the optional **nemo-gym** GRPO-rollout cluster uses the **nemo-gym** image (Ray lives in its
`/opt/gym-cli-venv`).

> **Per-image `RAY_VENV` is required for every image that ships Ray inside a uv venv.** The starter
> prepends `RAY_VENV/bin` to `PATH` so `ray start` finds the right interpreter; **each image's venv path
> differs** — `nemo-rl → /opt/nemo_rl_venv`, `nemo-gym → /opt/gym-cli-venv`, `nemo-skills → omit`
> (system Python). Passing the wrong `RAY_VENV` (or omitting it for nemo-gym) leaves the head sitting
> green while `ray start` silently failed inside — the dashboard never comes up.

```bash
# IMAGE is what the starter passes to pyxis --container-image. It can be EITHER a registry
# URI (docker://nvcr.io#<ORG>/<image>:<tag>) — pulled+built on the compute node — OR a prebuilt
# .sqsh path. See the image-staging note below for which to use (the registry URI is the simple path).
#
# MOUNTS (comma-separated host:container) is what the head container sees — it defaults to ONLY
# /lustre:/lustre, so pass MOUNTS= to add model/data paths a recipe references. Do NOT mount an
# NVFlow source checkout: the Ray Jobs working_dir delivers the baked source archive. Every data or
# model path (for example `hf_model_path`) MUST resolve INSIDE the head container, because a
# pre-provisioned cluster's mounts come from THIS bring-up, NOT from the recipe / my_cluster.yaml.
# In particular the GRPO smoke sets hf_model_path=/hf_models/Qwen/Qwen3-4B,
# so the head needs an /hf_models mount (else vLLM can't find the model locally and HuggingFace
# rejects the absolute path as a repo id — see troubleshooting). Alternatively, leave /hf_models out
# and set the recipe's hf_model_path to an absolute path under the already-mounted /lustre.
# GPU cluster (nemo-rl): training + rollout
IMAGE=docker://nvcr.io#<YOUR_REGISTRY>/nvflow-nemo-rl:v0.6.0-airgap DASHBOARD_PORT=8265 RAY_VENV=/opt/nemo_rl_venv \
  MOUNTS=/lustre:/lustre,<CLUSTER_PATH_TO_HF_MODELS>:/hf_models \
  sbatch --account=<acct> --partition=<gpu_partition> --nodes=1 --gpus-per-node=8 \
         scripts/start_ray_on_slurm.sb

# CPU cluster (nemo-skills): data-prep + eval
IMAGE=docker://nvcr.io#<YOUR_REGISTRY>/nvflow-nemo-skills:v1.1.1-ray DASHBOARD_PORT=8266 \
  sbatch --account=<acct> --partition=<cpu_partition> --nodes=1 --gpus-per-node=0 \
         scripts/start_ray_on_slurm.sb

# CPU cluster (nemo-gym): GRPO rollouts — ONLY for the 3-cluster GRPO path.
# RAY_VENV=/opt/gym-cli-venv is REQUIRED (Ray lives in the gym CLI venv); --time is REQUIRED.
IMAGE=<nemo-gym.sqsh-or-registry-ref> DASHBOARD_PORT=8267 RAY_VENV=/opt/gym-cli-venv \
  MOUNTS=/lustre:/lustre \
  sbatch --account=<acct> --partition=<cpu_partition> --time=<NN:00:00> --nodes=1 --gpus-per-node=0 \
         scripts/start_ray_on_slurm.sb
```

**BYO policy serve (GRPO rollouts only) — an external endpoint, not a Ray cluster.** Bring up a
standalone OpenAI-compatible vLLM serve from the **vllm-grpo** container on a GPU node (1 GPU is ample
for a 4B model). Its `--served-model-name` **MUST byte-equal** the recipe's `policy_vllm.model_path`
(sent verbatim as the OpenAI model id — it is **not** loaded locally):
```bash
srun --account=<acct> --partition=<gpu_partition> --nodes=1 --gpus-per-node=1 \
  --container-image=<vllm-grpo image> --container-mounts=/lustre:/lustre \
  bash -lc 'vllm serve <MODEL_PATH> --served-model-name <SERVED_NAME> \
    --host 0.0.0.0 --port 5000 --tensor-parallel-size 1 \
    --reasoning-parser qwen3 --max-model-len 40960 --trust-remote-code'
# Discover the serve node IP from the srun allocation header / `scontrol show hostnames`, then verify:
curl -sf http://<serve>:5000/v1/models | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])"
# -> <SERVED_NAME>  (must byte-equal policy_vllm.model_path)
```
> **`srun` on a GPU partition requires `--gpus-per-node=<N>` (N ≥ 1).** A GPU partition (`batch`,
> `interactive`) rejects any `srun` that requests 0 GPUs with `Batch job submission failed:
> ... Requested node configuration is not available` — even for a one-off check command. Always
> include `--gpus-per-node=1` (or `--gres=gpu:1`) when calling `srun` on a GPU partition.

> **Serve job stuck PENDING (`Reason=Priority`)?** Resubmit with a shorter `--time` — a shorter
> walltime raises backfill priority and typically lands faster:
> ```bash
> scancel <serve-jobid>
> srun ... --time=01:00:00 ...   # shorter wall = higher backfill priority
> ```
**The head IP changes on every bring-up — re-read it and update `dashboard_url` each time.** After
`sbatch`, the job logs to `nvflow-ray-head-<jobid>.out`; once the head is up it prints one line:
```bash
grep 'RAY HEAD READY' nvflow-ray-head-<jobid>.out
# RAY HEAD READY dashboard_url=http://<head-ip>:8265
```
Copy that exact `http://<head-ip>:<port>` into the matching `dashboard_url` field of
`$NEMO_SKILLS_CONFIG_DIR/my_cluster.yaml` (Step 4) **before** running anything. The loop, every time: bring up
head → `grep 'RAY HEAD READY'` → set `dashboard_url` → run. A stale value (a previous bring-up's IP)
fails the next command with `Error: Failed to connect to Ray at address: http://<old-ip>:8265` — the
fix is always to re-read `RAY HEAD READY` and update `dashboard_url` to the **current** head. `scancel`
the job to tear a cluster down.

> **If `RAY HEAD READY` never appears in the log** — `ray start` failed silently inside the container
> (wrong `RAY_VENV`, or the image is still pulling). Extract the head IP directly via `scontrol`:
> ```bash
> HEAD_NODE=$(scontrol show job <jobid> | grep -oP 'NodeList=\K\S+')
> HEAD_IP=$(getent hosts $HEAD_NODE | awk '{print $1}')
> curl -sf http://$HEAD_IP:8265/api/version   # 200 = head is up; connection refused = still starting
> ```
> Then verify the dashboard and wire `$HEAD_IP` into `my_cluster.yaml` exactly as above.

> **`yq` is typically not installed on cluster login nodes.** Use `sed -i` to patch YAML values
> (dashboard URLs, `base_url`, model paths) into a `/tmp` copy of the recipe. Never commit
> cluster-specific IPs or host paths back into the repo.

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
uv run --no-sync ray job submit --address http://<head>:8265 -- echo ok  # -> SUCCEEDED
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
Copy the template into a user-owned shared config directory. Do not modify the
image's tracked template or store secret values in the YAML:
```bash
export NEMO_SKILLS_CONFIG_DIR=/lustre/<you>/nvflow-config/cluster_configs
mkdir -p "$NEMO_SKILLS_CONFIG_DIR"
cp cluster_configs/template-ray.yaml "$NEMO_SKILLS_CONFIG_DIR/my_cluster.yaml"
```
```yaml
# $NEMO_SKILLS_CONFIG_DIR/my_cluster.yaml   (one config for all clusters — 2 for SFT/eval, 3 for GRPO)
executor: none   # REQUIRED even on the Ray backend — nemo-skills reads it outside the Ray path; NOT "ray". The Ray backend submits.
backend:
  name: ray
  working_dir: /opt/nvflow-ray-code.zip  # tracked-only archive baked in nvflow-client; install-free job cwd
  gpu_nemo_rl_dashboard_url:     http://<gpu-cluster-head>:8265   # GPU stages: SFT/GRPO training, convert
  cpu_nemo_skills_dashboard_url: http://<cpu-cluster-head>:8266   # CPU stages: data-prep, eval
  cpu_nemo_gym_dashboard_url:    http://<gym-cluster-head>:8267   # GRPO ONLY: rollouts (gym CLI cluster). OMIT for SFT/eval-only. Template ships this COMMENTED OUT — uncomment for GRPO.
containers:        # keys resolve each stage's `container:`; paths unused on a precreated cluster
  nemo-rl:     <YOUR_REGISTRY>/nvflow-nemo-rl:v0.6.0-airgap
  nemo-skills: <YOUR_REGISTRY>/nvflow-nemo-skills:v1.1.1-ray
  nemo-gym:    <YOUR_REGISTRY>/nvflow-nemo-gym:v1.1.1-ray   # GRPO rollouts only (gym CLI, no vLLM)
mounts:
  - /lustre:/lustre                            # shared data/output root; NOT a source checkout
  - <CLUSTER_PATH_TO_HF_MODELS>:/hf_models
nvflow_root: /lustre/<you>/nvflow  # REQUIRED on Ray: one absolute path visible to the driver and included in every head's bring-up MOUNTS
env_vars:
  - OPENAI_API_KEY         # judge key forwarded into each Ray job; GRPO/gym judge reads this by default
  # - NVIDIA_API_KEY       # use instead for the nemo-skills eval judge against api.nvidia.com
```
> **`backend.working_dir` is the code-delivery boundary.** The configured zip is created at image
> build time from the history-free committed snapshot. Ray uploads and extracts it as the job cwd;
> NeMo-Skills captures that initial absolute directory as `$NEMO_RUN_CODE_DIR` before the command
> can `cd` elsewhere. It does not install anything. The archive path is local to the process
> submitting the Ray job, so the
> airgapped driver must run from `nvflow-client`, where `/opt/nvflow-ray-code.zip` exists. Do not
> replace this with a source-path environment overlay, host source mount, `pip install`, or `uv sync`.

> **Export the judge key on the driver too — not just in `env_vars`.** A GRPO/eval judge configured as
> `openai_api_key: ${oc.env:NVIDIA_API_KEY}` (or `${oc.env:OPENAI_API_KEY}`) is resolved by OmegaConf's
> `oc.env` resolver **driver-side, at config load** — so the referenced variable must be exported in the
> shell running `nflow`, or the run aborts before submitting anything with
> `KeyError: Environment variable 'OPENAI_API_KEY' (or 'NVIDIA_API_KEY') not found`. The `env_vars`
> entry above (which forwards the key *into* the Ray job for the in-job gym judge) is a **separate**
> requirement — you generally need both. Pair the endpoint to the key: **NVIDIA-hosted**
> (`https://integrate.api.nvidia.com/v1` + `openai/gpt-oss-120b`) uses an **`nvapi-`** key; **OpenAI**
> (`https://api.openai.com/v1` + an OpenAI model id) uses an **`sk-`** key. Crossing them (e.g.
> `api.openai.com` with an `nvapi-` key, or a model the endpoint does not serve) fails with an auth or
> model-not-found error.
**`run-all` auto-routes each stage by its `num_gpus`** — exactly as Slurm splits `partition` /
`cpu_partition`: GPU stages → the GPU cluster, CPU stages → the CPU cluster. A stage may set
`target_cluster: cpu|gpu` to override. The **post-training** eval stage (in `sft/`, `grpo/`) ships
with `target_cluster: cpu` because its generation server is an external endpoint. The **standalone
baseline eval** (`eval/demo.yaml`) does **not** — its model stages carry `gpus: 4` (the Slurm prehost
count), so on a two-dashboard config they route to the **GPU** cluster. To run baseline eval against an
external server on the CPU cluster, set `target_cluster: cpu` + `gpus: 0` + `server_address` on those
model stages. On the **host-driver** path (`uv run --no-sync nflow` on a login node), the eval also imports the
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
GRPO-specific edits beyond the shared `/lustre` paths + baked working-directory contract: (1) scope to
ONE environment with the `-e <name>` CLI flag — **required**, and the only mechanism that scopes a run;
a top-level `_environment:` recipe key is not honored (else `grpo/base.yaml`'s three envs all fan out and
`finance_sec_search` fails with `Missing local datasets`); (2) point `judge_vllm` at an
**external/API** judge (`num_gpus: 0` + `openai_base_url` + `openai_model` +
`openai_api_key: ${oc.env:NVIDIA_API_KEY}`), not the in-job vLLM judge; (3) set `training.total_gpus: 8`
for one node (shipped configs use 16 = 2 nodes); (4) skip SDG by dropping a pre-made `final_result.jsonl`
under `step-3-convert-to-responses-api/<env>/`. The GPU head must sit on an idle-exempt /
non-preemptible partition to survive training.

**GRPO rollouts on Ray — the recommended external-serve 3-cluster path.** The single-cluster in-job
serve above self-hosts the policy vLLM *inside* the gym job, which needs vLLM baked into that image.
The **validated** path instead routes `collect_rollouts` to the **nemo-gym** cluster and points the
policy at an **external BYO vLLM serve** — so no vLLM is needed in the gym image at all. Prereqs, all
covered above:
1. Bring up the **third (nemo-gym) Ray head** with `RAY_VENV=/opt/gym-cli-venv` (Step 3) and set
   `cpu_nemo_gym_dashboard_url` in the config (this Step).
2. Bring up the **BYO policy serve** from the vllm-grpo container with `--served-model-name` equal to
   the recipe's `policy_vllm.model_path` (Step 3).
3. In the recipe, set `collect_rollouts.rollout.policy_vllm`: `base_url: http://<serve>:5000/v1` +
   `num_gpus: 0` (skips the in-gym serve) and `target_cluster: gym` (routes to the nemo-gym cluster).
   The ready-to-edit overlay is
   [`grpo/qwen3_4b_smoke_extserve.yaml`](nvflow/recipes/finance/workflows/grpo/qwen3_4b_smoke_extserve.yaml).
4. Keep the judge external (`num_gpus: 0`) and its key exported driver-side (callout above). For a
   shared hosted judge keep `rollout.num_samples_in_parallel` low (**4**) — a hosted key rate-limits
   (`429`) and stalls the run; raise it only with a self-hosted / higher-limit judge.

The per-stage run walkthrough (bring-up order, fire commands, verification) lives in
[`quick-start-ray.md` → Step 5](docs/recipes/finance/quick-start-ray.md) — this doc is setup only.

### Generation + judge endpoints (BYO / self-host)

*Only for stages that call a judge or generation server — eval, **SDG generation** (all four template-based SDG LLM stages: generate/genselect/filter), GRPO, compute_rewards. Skip it for the hello-world smoke, the SEC download, and SFT training-only runs.*

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
For **GRPO rollouts**, the policy is served the same way: a BYO OpenAI-compatible vLLM endpoint (the
vllm-grpo container), wired via `collect_rollouts.rollout.policy_vllm.base_url` + `num_gpus: 0` (see the
GRPO external-serve path in Step 4). Its `--served-model-name` **MUST byte-equal** the recipe's
`policy_vllm.model_path` — the value is sent verbatim as the OpenAI model id, so a mismatch yields a
`404` / model-not-found at rollout time.

For **generation**, serve your checkpoint with vLLM and pass `--server_address http://<serve>/v1` to
eval. The same applies to **SDG**: the shipped `sdg/template-based-sdg-demo.yaml` self-hosts an in-job
vLLM (`server_type: vllm` + `server_gpus: 4`) — not supported on Ray; use the ready-made Ray variant
[`sdg/template-based-sdg-demo_ray.yaml`](nvflow/recipes/finance/workflows/sdg/template-based-sdg-demo_ray.yaml),
which points all four LLM stages at a single external `<GEN_SERVE>` serve of gpt-oss-20b (nothing
prehosted). Use a **dedicated** judge endpoint for sustained batch judging (a shared endpoint rate-limits/429s);
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

`nflow` is the orchestrator — it submits jobs to the Ray dashboard(s) over HTTP and needs both
NVFlow and NeMo-Skills. The release-qualified path runs it from **nvflow-client**, which bakes the
locked venv and the exact `/opt/nvflow-ray-code.zip` referenced by `backend.working_dir`.

On Ray-on-Slurm, keep the user cluster config (dashboard URLs and env-var names) on shared storage,
separate from the image. Mount only that config directory plus shared data/models; do not mount a
source checkout:

```bash
CONFIG_DIR=/lustre/<you>/nvflow-config/cluster_configs
RECIPE=/opt/nvflow/nvflow/recipes/finance/workflows/ray_hello_world.yaml

srun --account=<acct> --partition=<cpu_partition> --time=00:30:00 \
  --container-image=<PATH_TO>/nvflow-client.sqsh \
  --container-mounts=/lustre:/lustre \
  --no-container-mount-home \
  env NEMO_SKILLS_CONFIG_DIR="$CONFIG_DIR" \
      UV_OFFLINE=1 UV_NO_SYNC=1 \
  bash --noprofile --norc -c \
    'cd /opt/nvflow && uv run --no-sync nflow run-all --config '"$RECIPE"
```

This performs no runtime sync or install: `uv run --no-sync` selects the baked venv, while Ray
uploads the baked zip and extracts it as each worker's cwd. Secrets remain runtime environment
variables; do not write them into the image or committed config.

For source development, run `uv sync`, create `/tmp/nvflow-ray-code.zip` with `git archive HEAD`,
change `backend.working_dir` to that path, and use `uv run --no-sync nflow ...`. That development flow still
ships a committed archive—never a live checkout mount—to workers.

Validate the Jobs-API path first (`--config` is **cwd-relative** — pass the full path from the repo
root, e.g. `nvflow/recipes/...`, not a short `grpo/...` form, which fails with `No such file or directory`):
```bash
uv run --no-sync nflow run-all --config nvflow/recipes/finance/workflows/ray_hello_world.yaml
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
uv run --no-sync nflow run-all --config nvflow/recipes/finance/workflows/sft/qwen3_4b.yaml
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
  so a fresh `uv sync` is fixed. If you still hit it: on a **source/dev checkout (with internet)** your
  driver env is stale — re-run `uv sync` to pick up the floor. **Container runs** get Ray ≥ 2.54.0 baked
  in — repull the image; **never** `pip`/`uv pip install` into an air-gapped or container run. Exact-version
  match across the two cluster images is **not** required (both speak Jobs REST `version: 4`); only the
  ≥2.54.0 floor matters.
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
- **`KeyError: Environment variable 'OPENAI_API_KEY' (or 'NVIDIA_API_KEY') not found`** at config load
  (before any job is submitted) — a judge `openai_api_key: ${oc.env:...}` is resolved **driver-side** by
  OmegaConf, so the referenced var must be exported in the shell running `nflow`. `export` it on the
  driver, then re-run. Pair the endpoint to the key form: `integrate.api.nvidia.com` ↔ `nvapi-` ↔
  `openai/gpt-oss-120b`; `api.openai.com` ↔ `sk-` ↔ an OpenAI model id. Crossing them gives an auth or
  model-not-found error, not a `KeyError`. (This is *separate* from forwarding the key into the job via
  `env_vars` — see the entry above.)
- **`Error: backend.name: ray with executor: none ... requires a dashboard URL`** (GRPO rollout /
  `collect_rollouts`) — the per-stage router found no dashboard for the stage. On the 3-cluster GRPO
  path this is almost always a **missing `cpu_nemo_gym_dashboard_url`** (the template ships it
  **commented out**): a `target_cluster: gym` stage then silently falls back to the nemo-skills CPU
  cluster, which has no `gym` CLI (a gym-CLI-not-found error; the literal *requires a dashboard URL* string
  only fires when **no** dashboard resolves at all, e.g. the config failed to load). Define **all three** `*_dashboard_url` keys in the config `backend:`
  (Step 4). For a single-cluster setup use the plain `backend.dashboard_url` shorthand instead.
- **`404` / model-not-found at rollout** (GRPO, external policy serve) — the BYO policy serve's
  `--served-model-name` does **not** byte-equal the recipe's `policy_vllm.model_path`. The model_path is
  sent verbatim as the OpenAI model id (it is not loaded locally), so the two strings must match exactly.
  Fix the serve's `--served-model-name` (or the recipe's `model_path`) and confirm with
  `curl http://<serve>:5000/v1/models` → `.data[0].id`.
- **`FileNotFoundError: .../finance_openqa_judge.txt` under `/nemo_run/code`** (GRPO judge) — the judge
  prompt template asset was not packaged into the job. nemo-run packages a **git-archive of `HEAD`**, so
  the prompt file must be **committed** (untracked / working-tree-only files are not shipped). On !185
  `finance_openqa_judge_overlay.yaml` points `judge_prompt_template_fpath` at
  `/nemo_run/code/nvflow/recipes/finance/prompts/finance_openqa_judge.txt` and that `.txt` **is
  git-tracked**, so a clean checkout works. If you add or edit a custom prompt, `git add && commit` it
  before the run (or point `judge_prompt_template_fpath` at a `${nvflow_root}`-absolute path that
  resolves live on the shared mount).
- **`ModuleNotFoundError: No module named 'rich'`** in the post-rollout **merge / enrich / analyze**
  steps (GRPO) — those steps run nvflow postprocess modules
  (`nvflow.recipes.finance.utils.rl.{enrich,analyze}_rollouts`) **inside the nemo-gym image**, and their
  import chain (`nvflow.recipes.__init__` → `nvflow.core.console` → `from rich.console import Console`)
  needs `rich`. An older nemo-gym image that doesn't bake `rich` fails here. **Fix: rebuild / repull a
  nemo-gym image that bakes `rich` at build time** — this error means your image predates that fix. Do
  **not** work around it at runtime by injecting the driver venv's `site-packages` onto the job
  `PYTHONPATH`: that is an air-gap violation **and** drags an incompatible `fastapi` into the gym server,
  crashing it at startup.
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
  uv run --no-sync ray job list --address http://<gpu-head>:8265        # find the still-RUNNING <submission_id>
  uv run --no-sync ray job stop --address http://<gpu-head>:8265 <submission_id>
  ```
  (Or use the dashboard **Jobs** tab.) Prefer the version-tolerant `ray job ...` commands over
  `ray status`, which does a **strict Ray-version check** and fails when the cluster's Ray (e.g. 2.54)
  differs from your local venv's Ray (e.g. 2.55) — or address the GCS endpoint directly at
  `<gpu-head>:6379`. On a login node without the `ray` CLI, drive the same operations through the
  dashboard REST API at `http://<gpu-head>:8265/api/jobs/`. See
  [`quick-start-ray.md` → Troubleshooting](docs/recipes/finance/quick-start-ray.md).
- **Every stage runs on the login node — no Ray job is ever submitted** (stages fail on container-only
  paths like `/opt/Gym`, or `ng_prepare_data: command not found`, and nothing shows on the Ray dashboard
  *Jobs* tab) — the **driver's** `nemo_skills` is mainline, not the fork that carries the Ray Jobs backend
  (`nemo_skills/pipeline/utils/ray_backend.py`). Without it, `executor: none` resolves to a `LocalExecutor`
  and everything runs on the login node. Pinning the *image* to the fork is not enough — the **driver env**
  must be too. **Fix:** pin the driver's `nemo_skills` to the fork build, then
  `uv lock --upgrade-package nemo-skills && uv sync`. Verify:
  `python -c "import importlib.util as u; print(u.find_spec('nemo_skills.pipeline.utils.ray_backend') is not None)"`
  must print `True`.
- **Head is `RUNNING` in `squeue` but the dashboard never comes up** (downstream `Failed to connect to
  Ray`) — `ray start` failed *inside* the head container (usually a missing/wrong `RAY_VENV`, so `ray`
  isn't on `PATH`: `ray: command not found`), but the sbatch job stays `RUNNING` — the failure is silent.
  The only tell is the `.out` log, which never prints `RAY HEAD READY`. **Fix:** always
  `grep 'RAY HEAD READY' nvflow-ray-head-<jobid>.out` before trusting a head; if absent, set the correct
  per-image `RAY_VENV` (nemo-rl `/opt/nemo_rl_venv`, nemo-gym `/opt/gym-cli-venv`, nemo-skills none — see
  Step 3). Probe an unknown image's venv with `srun --overlap --container-image=<sqsh> bash -lc 'command -v ray'`.
- **`ModuleNotFoundError: No module named 'typer'` (or another dep) after `uv sync` on a cluster** — on a
  quota-limited/Lustre checkout, uv's cache (small `$HOME`) and the `.venv` (Lustre) sit on different
  filesystems, so uv's hardlink install falls back to a partial copy and silently drops packages. `typer`
  *is* in `uv.lock` — an environment issue, not a lock bug. **Fix (source/dev checkouts only; container
  images bake their deps):** `export UV_LINK_MODE=copy` and `export UV_CACHE_DIR=/lustre/<you>/.uv-cache`
  (same filesystem as the venv), then `uv sync --reinstall`. Air-gap-clean — a pure sync from the
  committed lock, no new resolution or download beyond the lock.
- **`yq: command not found` when patching YAML** — `yq` is typically not installed on cluster login
  nodes. Use `sed -i` to substitute placeholder values (dashboard URLs, `base_url`, model paths) into a
  `/tmp` copy of the recipe before running. Commit only the placeholder form; never commit
  cluster-specific IPs or host paths into the repo.
- **In-container `python` is the wrong interpreter** — running `bash -lc 'python ...'` inside a
  container (or via `srun --overlap`) uses the system Python (login-shell resets `$PATH`), not the
  nvflow or Ray venv. Use the **explicit venv path** for verification:
  `<container>:/opt/nvflow/.venv/bin/python -c "import ray; print(ray.__version__)"` (nvflow client
  image) or the matching per-image venv (`/opt/nemo_rl_venv/bin/python`, `/opt/gym-cli-venv/bin/python`).
- **`FileNotFoundError` for files under `nvflow_root` on either the driver or a Ray job** —
  `executor: none` resolves paths on the host while submitted jobs resolve the same paths inside
  containers. A one-way `<repo>:/workspace` mount does not translate `/workspace` for the host
  process, and the host's absolute path is unavailable in the container unless it is mounted there.
  **Fix:** include the `nvflow_root` path in `MOUNTS=` when starting **every** Ray head
  (for example, `/lustre:/lustre` covers a root under `/lustre`; otherwise add
  `<root>:<root>`), then restart the heads. `my_cluster.yaml` `mounts:` do not retrofit a
  pre-provisioned cluster. Do not set `nvflow_root: /workspace` unless `/workspace` genuinely
  exists on the driver too.
- **`srun: error: pyxis: --container-mounts: invalid format: :/workspace`** — `$CKOUT` (or another
  variable used in the `MOUNTS=` env-var override to `sbatch`) is empty. `MOUNTS` is expanded by the
  *current shell* before being passed to `sbatch`; a new SSH session loses all shell variables. **Fix:**
  set `CKOUT=`, `IMAGE=`, and any other variables explicitly in the same terminal session *before*
  running the `sbatch` command — new sessions start with a clean environment.
- **`convert_dcp_to_hf.py --backend dtensor` fails with `FileNotFoundError: No metadata file found`**
  even though the checkpoint directory exists — nemo-rl v0.7.0.rc0 with `save_consolidated: false`
  saves weights as HF safetensors shards in `checkpoints/step_N/policy/weights/model/` with a
  `.hf_metadata/` directory, not a DCP `.metadata` file. The DCP converter expects the latter and finds
  nothing. **Fix:** use Automodel's official
  `/opt/nemo-rl/3rdparty/Automodel-workspace/Automodel/tools/offline_hf_consolidation.py`
  for this DTensor-v2 format (the NVFlow `format: fsdp` conversion path auto-detects it); do *not*
  use the DCP converter. Naively concatenating and re-saving shards produces invalid TP-sharded
  projection shapes, so validate the consolidated tensors and load the export in vLLM before eval.
