# Quick Start — Ray (Experimental)

The **Ray counterpart of [`quick-start.md`](quick-start.md)**: run the same finance demo
(eval → SDG → SFT → GRPO) on **pre-provisioned Ray clusters** instead of Slurm.

> **Experimental.** Slurm is the only officially supported executor; running on Ray clusters is
> experimental and the customer provisions the clusters. Set up the clusters first via
> [`INSTALL-RAY.md`](../../../INSTALL-RAY.md) (Steps 1–4), then follow this guide.

The recipes, data, and output layout are the **same** as the Slurm quick-start — so for stage details,
output trees, and reference numbers this guide points back to [`quick-start.md`](quick-start.md). What
differs is **how you invoke and verify** each stage on Ray. Read the prerequisites below first; they
are the difference.

---

## Prerequisites (the Ray difference — read this)

1. **Clusters up + dashboards reachable.** Bring up the GPU (nemo-rl) and CPU (nemo-skills) Ray
   clusters and write `$NEMO_SKILLS_CONFIG_DIR/my_cluster.yaml` per [`INSTALL-RAY.md`](../../../INSTALL-RAY.md)
   Steps 3–4. **The head IP changes on every bring-up** — after each `sbatch`, read the current head
   from `grep 'RAY HEAD READY' nvflow-ray-head-<jobid>.out` and copy that `http://<ip>:<port>` into the
   matching `dashboard_url` in `my_cluster.yaml` before running. A stale value fails with `Error: Failed
   to connect to Ray at address: http://<old-ip>:8265`.

   **What to expect after `sbatch start_ray_on_slurm.sb`:** on first boot the image is pulled from the
   registry ref, which takes **~10–12 min before the head reports ready** — that is expected, not a
   hang. When ready, the head's `.out` file prints:
   ```
   RAY HEAD READY dashboard_url=http://<head-ip>:8265
   Ray runtime started.
   ```
   **Verify** the head ID and that the dashboard is reachable:
   ```bash
   grep 'RAY HEAD READY' nvflow-ray-head-<jobid>.out          # -> dashboard_url=http://<head-ip>:8265
   curl -s -o /dev/null -w "%{http_code}\n" http://<head-ip>:8265/api/version   # -> 200
   curl -sf http://<gpu-head>:8265/api/version    # -> {"ray_version": "2.5x.x", ...}
   curl -sf http://<cpu-head>:8266/api/version
   ```
   A `200` (or the `{"ray_version": ...}` JSON) means the cluster is up and accepting job submissions; a
   connection refused/timeout means the head is still pulling the image (wait) or has exited (check the
   `.out` tail).

   **After wiring `dashboard_url` into `my_cluster.yaml`** there is nothing to launch — just confirm the
   config points at the *running* head:
   ```bash
   grep dashboard_url "$NEMO_SKILLS_CONFIG_DIR/my_cluster.yaml"   # -> current head URL(s)
   ```

2. **Use the NVFlow client driver.** For airgapped/release-qualified runs, invoke
   `uv run --no-sync nflow ...` inside `nvflow-client` as shown in
   [`INSTALL-RAY.md` Step 5](../../../INSTALL-RAY.md). The image bakes the locked venv and the compact
   `/opt/nvflow-ray-code.zip`; `backend.working_dir` uploads that immutable archive as each job's cwd,
   and NeMo-Skills captures its absolute path in `$NEMO_RUN_CODE_DIR` before a command can change cwd.
   Mount shared data/config only—never a source checkout—and perform no runtime install or sync.
   For source development, run `uv sync` once and build a tracked-only zip with `git archive HEAD`;
   point `backend.working_dir` to it before using the same `uv run --no-sync` commands.

3. **Ray client ≥ 2.54.0.** A fresh `uv sync` gives you this (the repo floors
   `ray[default]>=2.54.0`). If you see `submit_job() got an unexpected keyword argument
   'entrypoint_label_selector'`, your env is stale — re-sync. Exact version match across the two
   cluster images is **not** required.

4. **Absolute data paths visible on BOTH the driver and the cluster.** On a pre-provisioned Ray cluster the
   `mounts:` in `my_cluster.yaml` are **not** applied per-job — jobs run with the cluster's existing
   mounts. And with `executor: none` (the Ray Jobs backend) nemo-skills does the benchmark-module import, the eval
   data-file existence check, and the job-manifest writes **in the nvflow-client driver**,
   not in the job. So every `base_output_dir`, `datasets_dir`, data path, and model path **must be an
   absolute path that resolves identically on the driver host AND inside the cluster container** — use
   `/lustre/<you>/...` (mounted at the same place on both), **not** `/workspace/...` (that path is not
   the mutable shared-data contract and fails driver-side checks). The shipped recipes satisfy this **automatically**: their
   data/output paths interpolate `${nvflow_root}/...`, which the runner resolves to `/workspace` on
   Slurm and to the absolute `nvflow_root` you set in `my_cluster.yaml` on Ray (per
   [`INSTALL-RAY.md` Step 4](../../../INSTALL-RAY.md)) — so make sure `nvflow_root` is set there and
   points at such a both-sides-visible path. Only paths you write **yourself** (a hand-edited recipe, or
   an explicit placeholder file like `eval/demo_ray.yaml`) still need literal `/lustre/<you>/...` values.
   Tracked code, prompts, and scripts use paths relative to the delivered job cwd; only mutable
   data/output/model paths belong on the shared mount.

5. **No `++` CLI overrides.** `nflow` does **not** accept `++key=value` on the command line — edit the
   value in the recipe YAML instead. (`++` inside a recipe's `inline_args` is fine; that's forwarded to
   the underlying tool.)

6. **The recipe's `cluster:` field selects the cluster** — there is no `--cluster` flag. Set
   `cluster: my_cluster` in each recipe; `run-all` then auto-routes GPU stages → the nemo-rl cluster
   and CPU stages → the nemo-skills cluster (override per stage with `target_cluster: cpu|gpu`).

7. **`backend.working_dir: /opt/nvflow-ray-code.zip` is set in the cluster config** (per
   [`INSTALL-RAY.md` Step 4](../../../INSTALL-RAY.md)). The path exists in nvflow-client, where the
   Jobs submission process runs. Ray uploads and extracts it without pip/uv, making `python -m
   nvflow…` resolve from the job cwd on every worker image.

8. **`--config` is cwd-relative.** Run from the repo root and pass the full path from there (e.g.
   `nvflow/recipes/finance/workflows/grpo/qwen3_4b_smoke.yaml`), exactly as the commands below show — a
   short form like `grpo/qwen3_4b_smoke.yaml` fails with `No such file or directory` unless a
   recipe-search dir is configured.

### Slurm → Ray command cheat-sheet

| Slurm (quick-start.md) | Ray (here) |
|---|---|
| `sbatch` submits compute | you pre-provision; `nflow` submits over HTTP to the dashboard |
| `squeue --me` | `uv run --no-sync ray job list --address http://<head>:<port>` or the dashboard **Jobs** tab |
| tail `…/training-logs/ray-*-job.log` | dashboard **Jobs** tab → job logs (on the shared mount) |
| relative `outputs/...` paths | **absolute** `/lustre/<you>/...` on the shared mount |
| `bare nflow …` | `uv run --no-sync nflow …` inside nvflow-client (airgap) |

---

## Step 0 — Hello-world smoke (do this first)

Validate the Jobs-API path end to end before any real stage:
```bash
uv run --no-sync nflow run-all --config nvflow/recipes/finance/workflows/ray_hello_world.yaml
# -> "Ray job ... finished with status SUCCEEDED"
```
Nothing to edit: `base_output_dir` is `${nvflow_root}/outputs/finance/ray_hello_world`, which resolves
to the absolute `nvflow_root` from your `my_cluster.yaml` (prereq #4) — the artifact lands on the shared
mount.

**What to expect:** the driver prints `✓ Workflow Complete!` once the job is *submitted*; the job then
runs on the cluster and ends `SUCCEEDED`. **Verify** the job status and the artifact it wrote to the
shared mount:
```bash
uv run --no-sync ray job list --address http://<cpu-head>:8266     # the hello_world job -> SUCCEEDED
# or, without the ray CLI on the login node:
curl -s http://<cpu-head>:8266/api/jobs/ | python3 -c \
  "import sys,json;[print(j.get('status'),'|',(j.get('metadata') or {}).get('nemo_task_name')) for j in json.load(sys.stdin)]"
cat /lustre/<you>/ray_hello_world/step-1-hello-world/hello_world.txt
```
If this fails, fix it before continuing — every stage below uses the same submission path.

---

## The 5 stages on Ray

For each stage, the **recipe and stage list are identical** to [`quick-start.md`](quick-start.md) — open
it for stage descriptions, output trees, and reference results. Below is only the Ray delta: set
`cluster:` + an absolute `base_output_dir` in the recipe, then run with `uv run --no-sync`, and verify with Ray
tools.

**What to expect when `nflow run-all` / `run` launches (every stage).** The `nflow` header echoes the
environment, the stage list, and the cluster, e.g.:
```
Environment: equivalence_llm_judge
Cluster: my_cluster
...
✓ prepare_data job submitted
✓ collect_rollouts job submitted
✓ Workflow Complete!
```
**The driver returns once the jobs are *submitted*, not finished** — this is the key difference from
Slurm. The Ray backend hands each stage to the Ray cluster as a job; the actual compute then runs on the cluster.
So "Workflow Complete!" means *submission* succeeded — you confirm the *work* via the dashboard below.

**How to watch Ray jobs (the Ray-specific verification — there is no `squeue`/`sacct`).** List job
statuses through the dashboard REST API (the `ray` CLI may not be installed on the login node):
```bash
curl -s http://<head-ip>:8265/api/jobs/ | python3 -c \
  "import sys,json;[print(j.get('status'),'|',(j.get('metadata') or {}).get('nemo_task_name')) for j in json.load(sys.stdin)]"
```
You should see each stage progress to `SUCCEEDED`, in order, e.g. for a GRPO run:
```
SUCCEEDED | prepare_data
SUCCEEDED | rs0_chunk0-...          # rollout serve job — runs, then ends on its own
SUCCEEDED | train_validation_split
SUCCEEDED | training
```
A rollout serve job stuck `RUNNING` long after its rollout has finished, or judge `429` spam in the
rollout `…/rollout/rs*/logs/ng_run_*.log`, are the **failure signatures** — see the rollout-`429` and the
leaked-policy-serve Troubleshooting entries below.

> Before any stage that evaluates or trains with a judge (Step 1 eval, Step 5 GRPO), configure an
> **external / BYO OpenAI-compatible** generation + judge endpoint per
> [`INSTALL-RAY.md` → Generation + judge endpoints](../../../INSTALL-RAY.md). A self-hosted *in-job*
> judge is not supported on Ray. The key reaches the job via the cluster config `env_vars:`.

**Which cluster runs what — and which external endpoints each step needs:**

| Step | Runs on | External endpoints required |
|---|---|---|
| 0 — hello-world smoke | CPU cluster | none |
| 1 — baseline eval | CPU cluster | generation serve (Qwen3-4B) + judge |
| 2 — download SEC filings | CPU cluster | none (outbound access to SEC EDGAR, or staged data) |
| 3 — SDG | CPU cluster | **one** generation serve (gpt-oss-20b) |
| 4 — SFT + eval | GPU cluster (training) + CPU cluster (eval) | eval only: serve of the trained checkpoint + judge |
| 5 — GRPO + eval | GPU cluster (rollout + training) + CPU cluster (data-prep, eval) | judge (rollout + training + eval); eval generation serve |

**The GPU (nemo-rl) cluster is needed only for the SFT/GRPO training lanes (Steps 4–5)** — a customer
running only eval, SDG, or the SEC download does not need to bring it up. (In Step 5,
`collect_rollouts` launches its policy vLLM serve **on** the GPU cluster — that is in-cluster, not an
external endpoint; the judge **is** external.)

### Step 1 — Baseline eval (CPU cluster, against external serves)

**Easiest path — use the ready-made `eval/demo_ray.yaml`.** It is `eval/demo.yaml` with every Ray
override already applied: CPU routing (`target_cluster: cpu` + `gpus: 0`), external generation
`server_address`, the **external-judge repoint**, and absolute `/lustre` paths. Fill its 3 placeholders
(`<GEN_SERVE>`, `<JUDGE_SERVE>`, `/lustre/<you>`) and run:
```bash
uv run --no-sync nflow run prepare_data --config nvflow/recipes/finance/workflows/eval/demo_ray.yaml
uv run --no-sync nflow run qwen3-4b     --config nvflow/recipes/finance/workflows/eval/demo_ray.yaml
```

What that file changes vs the shipped `eval/demo.yaml`, and why (apply the same if you edit `demo.yaml`
directly):
- **Routing.** The baseline eval model stages carry `gpus: 4` and no `target_cluster`, so on a
  two-dashboard config they route to the **GPU** cluster. Set `target_cluster: cpu`, `gpus: 0`, and
  `server_address: http://<your-vllm>/v1` to run on the CPU (nemo-skills) cluster against your external
  generation server. (Post-training eval in `sft/`/`grpo/` already ships `target_cluster: cpu`.)
- **Judge.** `demo.yaml` ships a self-hosted **in-job vLLM judge** (`server_type: vllm`, `server_gpus: 2`)
  — **not supported on Ray.** Repoint the `judge:` block to your external endpoint: `server_type: openai`,
  `server_address: http://<your-judge>/v1`, `model: <served-name>`, and drop `server_gpus`.
  `demo_ray.yaml` also sets `++max_concurrent_requests=1` and `++server.max_retries=10` in
  `judge.extra_args`; keep that throttle for a shared hosted judge. If the provider still returns `429`,
  wait for its quota window or use a dedicated/higher-limit endpoint, then rerun the same model stage.
  NeMo-Skills' `skip_filled=True` resume reuses completed generations and partial judgements.
- **Paths (absolute `/lustre`, per prereq #4).** `demo_ray.yaml` pins `base_output_dir`,
  `datasets_dir`, and `stages.prepare_data.output_dir` to **explicit** `/lustre/<you>/...` placeholders
  — fill them with a path visible identically on the driver host and in the container, and keep
  `stages.prepare_data.output_dir` equal to `datasets_dir` (prepare_data must write where the eval
  reads). (If you instead edit `eval/demo.yaml` directly, its paths interpolate `${nvflow_root}/...`
  and resolve from `my_cluster.yaml`'s `nvflow_root` automatically — explicit absolutes and the
  interpolated root are both valid; prereq #4 explains the contract.)

**Verify** (read from the absolute output dir you set):
```bash
uv run --no-sync ray job list --address http://<cpu-head>:8266     # prepare_data + qwen3-4b jobs -> SUCCEEDED
cat /lustre/<you>/.../baseline-eval/qwen3-4b/eval-results/secque/metrics.json
```
**Reference (baseline ballpark):** secque `pass@1` `judge_correct` ≈ 50%, `no_answer` ≈ 0% for base
Qwen3-4B (see quick-start.md Step 1 for the full reference table). A high `no_answer` means a serve/judge
or reasoning-parser misconfig, not a model-quality result.

### Step 2 — Download SEC filings (CPU cluster)
```bash
uv run --no-sync nflow run demo --config nvflow/recipes/finance/workflows/download_sec_filings.yaml
```
(SEC EDGAR is reached from the cluster — ensure the cluster has the needed outbound access, or stage
the data, in an air-gapped environment.) **Verify** the job reached `SUCCEEDED` and the expected
artifacts landed under your absolute output dir:
```bash
uv run --no-sync ray job list --address http://<cpu-head>:8266     # demo download job -> SUCCEEDED
ls /lustre/<you>/.../workflow-2-download-sec/step-0-download/data/
# Expected: AAPL/ CSCO/ GOOG/ IBM/ META/ MSFT/ NVDA/
ls /lustre/<you>/.../workflow-2-download-sec/step-0-download/sec_metadata.parquet
```

### Step 3 — Synthetic Q&A (SDG) (CPU cluster, against an external serve)

**Easiest path — use the ready-made `sdg/template-based-sdg-demo_ray.yaml`.** It is
`sdg/template-based-sdg-demo.yaml` with every Ray override already applied: all four LLM stages
repointed to an external generation `server_address`, with nothing prehosted. Fill its **1** placeholder
(`<GEN_SERVE>`) and run:
```bash
uv run --no-sync nflow run-all --config nvflow/recipes/finance/workflows/sdg/template-based-sdg-demo_ray.yaml
```
The four LLM stages (`generate_questions`, `generate_answers`, `genselect_answers`, `filter_answers`)
all use the **same** model (gpt-oss-20b) and run sequentially, so a **single** external serve covers the
whole workflow. The recipe's `model:` value must equal your serve's `--served-model-name`.

What that file changes vs the shipped `template-based-sdg-demo.yaml`, and why (apply the same if you
edit another SDG recipe):
- **Generation.** The stock demo self-hosts an **in-job vLLM** on 4 of its 6 stages
  (`server_type: vllm`, `server_gpus: 4`, gpt-oss-20b TP=4) — **not supported on Ray**: generation must
  be an external / BYO OpenAI-compatible endpoint (per
  [`INSTALL-RAY.md` → Generation + judge endpoints](../../../INSTALL-RAY.md)), and the CPU (nemo-skills)
  cluster has no GPUs to host it. The `_ray` variant sets `server_address: http://<GEN_SERVE>/v1` in
  **every** LLM stage's `stage_kwargs:` and nulls `server_gpus`/`server_nodes`/`server_args` so
  nemo-skills prehosts nothing (`model`/`server_type`/`num_chunks`/`num_random_seeds` inherit
  unchanged).
- **Routing.** Each LLM stage also gets `target_cluster: cpu`, making the CPU routing explicit — with
  `server_gpus` nulled there is no positive GPU count left to route on. The two CPU-only stages
  (`create_seed_data`, `map_questions_to_context`) inherit as-is.
- **Paths — nothing to edit** (unlike `eval/demo_ray.yaml`). The SDG recipes read/write
  `${nvflow_root}/...`, which on Ray resolves to the absolute shared root you set as `nvflow_root` in
  `my_cluster.yaml` (per [`INSTALL-RAY.md`](../../../INSTALL-RAY.md) Step 4) — so the same recipe paths
  run on Slurm and Ray unchanged, and the `_ray` variant carries no path placeholders.

**Verify** all SDG jobs reached `SUCCEEDED` and the final Q&A set is the expected size:
```bash
uv run --no-sync ray job list --address http://<cpu-head>:8266     # the SDG-stage jobs -> SUCCEEDED
wc -l /lustre/<you>/.../workflow-3-template-based-sdg/step-5-filter-answers/final_result.jsonl
# Expected: ~1000–1200 Q&A pairs (input for SFT/GRPO)
```

### Step 4 — SFT + eval (GPU train, CPU eval)
Training auto-routes to the GPU (nemo-rl) cluster; eval to the CPU cluster.
```bash
uv run --no-sync nflow run-all --config nvflow/recipes/finance/workflows/sft/qwen3_4b.yaml
```

> **Post-training eval on Ray — repoint both serves (same pattern as Step 1).** The shipped
> `sft/qwen3_4b.yaml` eval stage self-hosts generation (`server_type: vllm`, `gpus: 4`) **and** an
> in-job vLLM judge (`judge:` with `server_type: vllm`, `server_gpus: 2`) — the pattern Step 1 already
> explains is **not supported on Ray**. Endpoints per
> [`INSTALL-RAY.md` → Generation + judge endpoints](../../../INSTALL-RAY.md):
> - **Judge — override the recipe's `stages.eval.judge` block** (a full `judge:` override there wins
>   over the `eval/base.yaml` default): `server_type: openai`,
>   `server_address: http://<JUDGE_SERVE>/v1`, `model: <served-name>`, `server_gpus: null` — the same
>   judge repoint as Step 1.
> - **Generation — serve the trained checkpoint yourself and evaluate it through the Step-1 lane.**
>   The embedded eval stage always launches its own vLLM from its `gpus:` count — a `server_address:`
>   under `stages.eval` is **not** consumed (only the eval lane's `models:` entries take one). Its
>   conversion sub-job is unaffected (idempotent Megatron→HF on the GPU cluster; it writes the
>   `hf_models/step_10` shown in the Verify block below). Serve that converted checkpoint with your own
>   external vLLM, then add a `models:` entry for it in `eval/demo_ray.yaml` —
>   `path: <.../hf_models/step_10>` (must equal the serve's `--served-model-name`),
>   `server_address: http://<serve>/v1`, `gpus: 0`, `target_cluster: cpu` — and run that entry exactly
>   as in Step 1.

**Monitor on Ray instead of `squeue`:**
```bash
uv run --no-sync ray job list --address http://<gpu-head>:8265   # training job -> SUCCEEDED
# logs: dashboard Jobs tab, or under the absolute training-logs dir on the shared mount
```
The driver returns at submission; the training job runs on the GPU cluster and ends `SUCCEEDED`.

**Verify training** (checkpoints land under your absolute output dir; paths mirror quick-start.md Step 4):
```bash
ls /lustre/<you>/.../workflow-4-sft/qwen3_4b/step-4-training/model-qwen3-4b-*/checkpoints/
# Expected: step_10/ step_16/ (save_period=10 and final epoch)
ls /lustre/<you>/.../workflow-4-sft/qwen3_4b/step-4-training/model-qwen3-4b-*/hf_models/
# Expected: step_10/ (HF-format model, converted during eval)
```
**Verify evaluation:**
```bash
cat /lustre/<you>/.../workflow-4-sft/qwen3_4b/step-5-eval/step-10/eval-results/secque/metrics.json
```

### Step 5 — GRPO + eval (GPU train, CPU eval; external judge required)

> **Experimental — start with the single-node smoke.** Use
> [`grpo/qwen3_4b_smoke.yaml`](../../../nvflow/recipes/finance/workflows/grpo/qwen3_4b_smoke.yaml)
> (one 8-GPU nemo-rl cluster: prepare_data → collect_rollouts → train_validation_split → training,
> external/API judge). Env-scoping + the data-prep path are validated; the full rollouts+training run
> on Ray is still being validated (the GPU head must sit on an idle-exempt / non-preemptible partition
> to survive training). Five GRPO-specific edits beyond the shared `/lustre` paths + baked working-directory
> contract (same as Step 1 / prereq #4):
>
> 1. **Scope to ONE environment with `-e` (required).** `grpo/base.yaml` ships three environments
>    (`equivalence_llm_judge`, `mcqa`, `finance_sec_search`). Without scoping, every stage fans out to
>    **all** of them and `finance_sec_search` fails with `AssertionError: Missing local datasets`. You
>    **must** pass `-e equivalence_llm_judge` on the command — this is the only thing that scopes a run.
>    A top-level `_environment:` key in the recipe is **not** honored by `run-all` or `nflow run`
>    (the runner injects the env filter only from the `-e` flag — `workflow_runner._run_stage` →
>    `resolve_environments` in `nvflow/lib/rl/helpers.py`).
> 2. **External / API judge, not in-job vLLM.** Repoint `judge_vllm` to an external endpoint:
>    `num_gpus: 0`, `openai_base_url`, `openai_model`. With `num_gpus: 0` + `openai_base_url` no judge
>    server is hosted. The judge key reaches the gym **one of two ways — pick one and be consistent:**
>    - **Explicit (the form `grpo/qwen3_4b_smoke.yaml` ships):** `openai_api_key: ${oc.env:NVIDIA_API_KEY}`
>      (or `${oc.env:OPENAI_API_KEY}`). `${oc.env:...}` is resolved **driver-side** at config load and
>      baked into the job — so export that var **on the driver host** (where `uv run --no-sync nflow` runs). The
>      cluster `env_vars:` do **not** feed this form.
>    - **Fallback (omit `openai_api_key`):** the gym then reads the literal `$OPENAI_API_KEY` from the
>      **job's** env at runtime, so `my_cluster.yaml` `env_vars:` **must** list `OPENAI_API_KEY` (set it
>      to your NVIDIA key, `OPENAI_API_KEY=$NVIDIA_API_KEY`, for the NVIDIA API).
>
>    The trap is mixing them — a cluster `env_vars: OPENAI_API_KEY` does **not** satisfy an explicit
>    `${oc.env:NVIDIA_API_KEY}` (which is read from the driver's `NVIDIA_API_KEY`, not the job's).
> 3. **Single 8-GPU node.** The shipped `grpo/qwen3_4b.yaml` trains with `total_gpus: 16` (2 nodes). For
>    one node set `training.total_gpus: 8`; the eval `checkpoint_path` run-name changes to
>    `grpo-qwen3-4b-8g-tp2-cp1-seq32k` (vs `…-16g-…`).
> 4. **Skip SDG — feed a pre-made dataset.** Drop one `final_result.jsonl` at
>    `<base_output_dir>/step-3-convert-to-responses-api/equivalence_llm_judge/final_result.jsonl`
>    (fields: `uuid`, `question`, `problem`, `expected_answer`, `prompt`, `responses_create_params`). Use
>    **reward-varied** rows — the `collect_rollouts` variance filter (`filter.min_reward_std`) drops
>    zero-variance prompts, leaving training empty otherwise.
>
> The smoke also ships `rollout.num_samples_in_parallel: 4` (with `max_num_samples: 24`). That caps the
> rollout at **4 concurrent** reward-judge calls: against the **shared / public** API judge a wide fan-out
> `429`s the per-key rate limit and the rollout stalls partway (see the rollout-`429` Troubleshooting entry
> below). `max_num_samples` stays `24` so the variance filter still sees enough reward-varied rows. Keep `4`
> for a shared judge; raise it for throughput only with a higher-rate-limit / self-hosted judge.
>
> The smoke also ships `rollout.num_random_seeds: 1`. Each seed spawns its **own concurrent** rollout
> vLLM server (`rs0`, `rs1`, …), each TP=2 at vLLM's default `gpu_memory_utilization` (~0.9); two of
> them on one 8-GPU node collide and the second fails engine init with `ValueError: No available memory
> for the cache blocks`. One seed is enough for a single-node smoke — per-prompt reward variance comes
> from `num_generations_per_prompt`, not seeds. Keep it at `1` (or, if you need more, lower
> `gpu_memory_utilization` / `max_model_len` so co-located servers fit).
>
> **Policy vLLM: who launches it.** In `collect_rollouts` the **stage launches its own policy vLLM
> serve** (`rollout.policy_vllm.model_path` + `num_gpus: N`, one serve per random seed `rs0`, `rs1`, …)
> and the NeMo-Gym client connects to it over HTTP — the gym does **not** self-host a policy vLLM. To
> run rollouts against an **already-running** policy serve instead (and not allocate GPUs for the
> serve), set `rollout.policy_vllm.base_url: http://<your-vllm>/v1` and `num_gpus: 0`; the stage then
> skips launching and the gym clients your serve. In **training** (Step 8) the policy vLLM is
> **colocated inside NeMo-RL** (`colocated.enabled: true`, `gpu_memory_utilization: 0.7`), sharing the
> training GPUs — there is no separate serve and no external URL.
>
> **GPU budget on one 8-GPU node.** `collect_rollouts` uses `policy_vllm.num_gpus` (2) ×
> `num_random_seeds` (1) = 2 GPUs (6 idle). `training` uses all 8 (`total_gpus: 8`, FSDP TP=2 → DP=4)
> with the policy generation **colocated** at `gpu_memory_utilization: 0.7` (from the `grpo-base`
> preset) — leave that headroom; raising it OOMs the trainer, lowering it slows generation. Stages run
> sequentially, so peak demand is the training step (all 8 GPUs).
>
> ```bash
> uv run --no-sync nflow run prepare_data collect_rollouts train_validation_split training \
>   --config nvflow/recipes/finance/workflows/grpo/qwen3_4b_smoke.yaml -e equivalence_llm_judge
> # the training job is a Ray job — monitor it on Ray, NOT `squeue`:
> uv run --no-sync ray job list --address http://<gpu-head>:8265        # the training job -> SUCCEEDED
> uv run --no-sync ray job logs --address http://<gpu-head>:8265 <job-id>   # live training log (or dashboard Jobs tab)
> ```
> **What to expect while `collect_rollouts` is in flight.** The rollout client log shows the sample
> counter advancing, and the policy serve log shows vLLM serving generation requests:
> ```bash
> tail -f <base_output_dir>/.../step-5-collect-rollouts/equivalence_llm_judge/rollout/rs0/logs/ng_run_*.log
> # -> Collecting rollouts 1/24 ... 12/24 ... 24/24   (counter advancing = healthy)
> # policy serve log shows: ... "POST /v1/chat/completions HTTP/1.1" 200 OK   + generation throughput
> ```
> A counter that **stops advancing** while the serve goes idle is the rollout-`429` failure signature
> (see Troubleshooting). The serve job ends on its own once rollouts finish — it should **not** stay
> `RUNNING` after the counter hits `24/24`.
>
> **What to expect when `training` completes.** The training job log records reward metrics, performance
> metrics (TFLOPS), then stops and saves a checkpoint:
> ```
> ... reward metrics logged per step ...
> ... performance metrics (TFLOPS) ...
> Max number of steps has been reached, stopping training early
> ... saving checkpoint ...
> ```
> **Verify the run succeeded — the checkpoint lands under `step-8-training`:**
> ```bash
> RUN=<base_output_dir>/qwen3_4b/step-8-training/equivalence_llm_judge/grpo-qwen3-4b-8g-tp2-cp1-seq32k
> ls $RUN/checkpoints/                       # smoke writes step_2 (max_num_steps:2, save_period:2)
> ls -R $RUN/checkpoints/step_2/              # weights + optimizer state = training succeeded
> ```
> A trained `checkpoints/step_<N>/` is the success signal; the Ray job also ends `SUCCEEDED`. The
> checkpoint dir is multi-GB and contains:
> ```
> step_<N>/
> ├── policy/
> │   ├── weights/model/        # safetensor shards (model weights)
> │   ├── optimizer/optim/      # .distcp shards (optimizer state)
> │   └── tokenizer/
> ├── config.yaml
> └── training_info.json
> ```
> (Production runs with more steps write `step_<N>` at every `save_period` and keep the top-k by
> `val:accuracy`.)
>
> **wandb is optional.** The smoke recipe ships a `logger.wandb` block, but `wandb_mode` defaults to
> **`disabled`** (from `grpo/base.yaml`), so training runs with `wandb_enabled=false` and needs **no**
> `WANDB_API_KEY` — the dormant `logger.wandb.name` is harmless. To enable wandb, set
> `training.wandb_mode: online` and export `WANDB_API_KEY` (forward it via the cluster `env_vars:`);
> otherwise leave it as-is.
>
> Adding **eval** (Step 9) needs the CPU (nemo-skills) cluster + an external generation serve —
> configure it exactly as the eval lane (Step 1).
>
> **Keep the GPU head alive through training.** A single-node GRPO job idles its GPUs during
> data-prep/rollout and only saturates them at the training step, so an idle-GPU reaper can `scancel`
> the Ray head mid-run. Bring the GPU head up on an idle-exempt / non-preemptible partition — see
> [`INSTALL-RAY.md` → Slurm idle-GPU reaping](../../../INSTALL-RAY.md). This is the most common cause
> of a GRPO smoke dying after `collect_rollouts`.

Same two environments and per-stage commands as quick-start.md Step 5 — run them with `uv run --no-sync` and an
**external** judge (`base_url`/`openai_base_url`), e.g. for `equivalence_llm_judge`:
```bash
uv run --no-sync nflow run data_transformation apply_prompt_template convert_to_responses_api prepare_data \
  --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml -e equivalence_llm_judge
uv run --no-sync nflow run collect_rollouts        --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml -e equivalence_llm_judge
uv run --no-sync nflow run train_validation_split  --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml -e equivalence_llm_judge
uv run --no-sync nflow run training                --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml -e equivalence_llm_judge
uv run --no-sync nflow run eval                    --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml -e equivalence_llm_judge
```
(For `finance_sec_search`, also run `validate_questions` + `prefetch_cache` and train with
`grpo/qwen3_4b_finsec.yaml`, exactly as in quick-start.md Step 5.) The Ray backend submits independent
jobs **concurrently** — watch the fan-out on the dashboard Jobs tab or `ray job list`. Use a
**dedicated** judge endpoint for the rollout/eval fan-out (a shared endpoint rate-limits/429s). Verify
rollout `summary.txt` / `filter_report.json`, checkpoints, and eval `metrics.json` under your absolute
output dir (paths mirror quick-start.md Step 5).

### Step 5 (recommended) — GRPO on **three** Ray clusters with an **external policy serve**

> **This is the validated GRPO-on-Ray path** (proven end-to-end on 3-cluster Ray, `collect_rollouts`
> → merge/enrich/analyze `✓ Workflow Complete`). It replaces the single-cluster in-gym policy serve
> from the block above with an **external / BYO** policy vLLM endpoint — the same "point at an
> already-running serve" idea Step 1 uses for the *judge*, now applied to the *policy* too.
>
> **Why external-serve (the image gap).** `collect_rollouts` routes to the **nemo-gym** cluster
> (`target_cluster: gym`), whose image carries the `gym` CLI but **no vLLM**. A *local* policy serve
> (`policy_vllm.num_gpus: 2`) would need the `gym` CLI **and** vLLM in **one** image → it fails, because
> nemo-gym has no vLLM. The gym `vllm_model` adapter is a **pure HTTP client**, so setting
> `policy_vllm.base_url` + `num_gpus: 0` makes nvflow *skip* launching an in-gym serve
> (`need_policy_server = not policy_vllm.base_url`, `rollout.py`) — no vLLM needed in the gym image, no
> GPU consumed. The BYO serve is **not** a 4th Ray cluster: it is a standalone OpenAI-compatible vLLM
> endpoint on a GPU node, exactly like the judge.
>
> Use the shipped example
> [`grpo/qwen3_4b_smoke_extserve.yaml`](../../../nvflow/recipes/finance/workflows/grpo/qwen3_4b_smoke_extserve.yaml)
> — a thin overlay of `qwen3_4b_smoke.yaml` that sets exactly the four external-serve keys below.

**(a) Bring up the three Ray heads.** Each cluster is one head; **each image needs its own
`RAY_VENV`** (nemo-gym = `/opt/gym-cli-venv`, nemo-rl = `/opt/nemo_rl_venv`), and `--time` is
**required**. The GPU (nemo-rl) and CPU (nemo-skills) heads come up per
[`INSTALL-RAY.md`](../../../INSTALL-RAY.md) Steps 3–4; the **nemo-gym** head (new for GRPO) is:
```bash
IMAGE=<PATH_TO>/nvflow-nemo-gym.sqsh DASHBOARD_PORT=8267 RAY_VENV=/opt/gym-cli-venv \
  MOUNTS=/lustre:/lustre \
  sbatch --partition=<cpu_partition> --time=NN:00:00 --gpus-per-node=0 scripts/start_ray_on_slurm.sb
```
Then wire **all three** dashboard URLs into `my_cluster.yaml` `backend:` (the head IP changes every
bring-up — re-read each from its `RAY HEAD READY` line and re-set):
```yaml
backend:
  name: ray
  gpu_nemo_rl_dashboard_url:     http://<gpu-head>:8265   # nemo-rl  — training
  cpu_nemo_skills_dashboard_url: http://<cpu-head>:8266   # nemo-skills — data-prep / split / eval
  cpu_nemo_gym_dashboard_url:    http://<gym-head>:8267   # nemo-gym — collect_rollouts (REQUIRED)
```
> **`cpu_nemo_gym_dashboard_url` is required for GRPO.** If it is unset the router **silently falls
> back** to the nemo-skills CPU cluster (`gym_url or cpu_url or gpu_url`), which has **no** `gym` CLI, so
> `collect_rollouts` misroutes and fails there. See the first Troubleshooting entry below.

**Per-stage routing (the 3-cluster chain).** Each stage's `target_cluster` / container picks the head —
poll that head to watch the stage:

| Stage | Cluster (`target_cluster`) | Dashboard |
|---|---|---|
| `collect_rollouts` | nemo-gym (`gym`) | `http://<gym-head>:8267` |
| `train_validation_split` | nemo-skills (`cpu`) | `http://<cpu-head>:8266` |
| `training` | nemo-rl (`gpu`) | `http://<gpu-head>:8265` |

**(b) Stand up the BYO policy vLLM serve and get its `base_url`.** Serve your policy model from a GPU
node with the **vllm-grpo** image (1 GPU is ample for a 4B model). The `--served-model-name` **must
byte-equal** the recipe's `policy_vllm.model_path` (which is sent **verbatim** as the OpenAI model id,
`helpers.py` — the value is *not* loaded from disk in external mode):
```bash
srun --partition=<gpu_partition> --gpus-per-node=1 --time=NN:00:00 \
  --container-image=<PATH_TO>/nvflow-vllm-grpo.sqsh --container-mounts=/lustre:/lustre \
  bash -lc 'vllm serve <MODEL_PATH> --served-model-name <SERVED_NAME> \
    --host 0.0.0.0 --port 5000 --tensor-parallel-size 1 \
    --reasoning-parser qwen3 --max-model-len 40960 --trust-remote-code'
```
Discover the serve node IP (`scontrol show hostnames <alloc>`, or the `srun` allocation header) and
verify — `.data[0].id` **must** equal `<SERVED_NAME>`:
```bash
curl -s http://<serve-node>:5000/v1/models | python3 -c "import sys,json;print(json.load(sys.stdin)['data'][0]['id'])"
# -> <SERVED_NAME>   (must byte-equal policy_vllm.model_path below)
```

**(c) Fill the recipe** — the four external-serve keys (already set in `qwen3_4b_smoke_extserve.yaml`;
apply the same to any GRPO recipe):
```yaml
stages:
  collect_rollouts:
    target_cluster: gym                 # route to cpu_nemo_gym_dashboard_url (only image with the gym CLI)
    rollout:
      num_samples_in_parallel: 4        # hosted-judge 429 floor — keep low for a shared judge (see below)
      num_random_seeds: 4               # variance-filter floor — see note below
      policy_vllm:
        model_path: <SERVED_NAME>       # sent VERBATIM as the OpenAI model id; MUST byte-equal --served-model-name
        num_gpus: 0                     # 0 + base_url => nvflow skips the in-gym serve (need_policy_server=False)
        reasoning_parser: qwen3
        base_url: http://<serve-node>:5000/v1   # your BYO serve from (b); MUST include /v1
```
The **judge** stays external + `num_gpus: 0` — either NVIDIA-hosted or OpenAI (see (d) for the
endpoint↔key↔model pairing). The example ships the NVIDIA-hosted form
(`openai_base_url: https://integrate.api.nvidia.com/v1`, `openai_model: openai/gpt-oss-120b`,
`openai_api_key: ${oc.env:NVIDIA_API_KEY}`).

> **`num_random_seeds: 4` for external-serve (variance-filter floor).** `collect_rollouts` filters on
> reward variance, and `reward_std` is computed **across seeds**. With `num_random_seeds: 1` every row's
> std is `0`, so the filter drops **all** rows → an **empty** `train.jsonl`, and the downstream
> `train_validation_split` / `training` stages then fail. This inverts the single-cluster in-gym advice
> (which keeps `1` to avoid co-located rollout-vLLM GPU collisions — see the block above): external-serve
> shares **one** endpoint with no GPU collision, so `4` seeds is safe and gives the filter signal.
> `qwen3_4b_smoke_extserve.yaml` sets this.

**(d) Export the judge key on the driver.** `openai_api_key: ${oc.env:<VAR>}` is resolved
**driver-side** at config load (OmegaConf `oc.env` resolver), so the referenced var must be exported on
the host running `nflow` **before** the run — the cluster `env_vars:` do **not** feed this form. Pair
the endpoint, key, and model correctly:

| Judge provider | `openai_base_url` | key form | `openai_model` |
|---|---|---|---|
| NVIDIA-hosted | `https://integrate.api.nvidia.com/v1` | `nvapi-…` (export `NVIDIA_API_KEY`) | e.g. `openai/gpt-oss-120b` |
| OpenAI | `https://api.openai.com/v1` | `sk-…` (export `OPENAI_API_KEY`) | e.g. `gpt-4o` |

```bash
export NVIDIA_API_KEY=nvapi-...          # matches ${oc.env:NVIDIA_API_KEY} in the recipe
```

**(e) Run the chain** — `collect_rollouts` → `train_validation_split` → `training` (smoke
`collect_rollouts` first to prove the external-serve path, then the rest):
```bash
uv run --no-sync nflow run collect_rollouts \
  --config nvflow/recipes/finance/workflows/grpo/qwen3_4b_smoke_extserve.yaml -e equivalence_llm_judge
# once rollouts land (24 samples on the smoke), continue:
uv run --no-sync nflow run train_validation_split training \
  --config nvflow/recipes/finance/workflows/grpo/qwen3_4b_smoke_extserve.yaml -e equivalence_llm_judge
```
**Verify** `collect_rollouts` on the **gym** cluster, then the split/train chain:
```bash
uv run --no-sync ray job list --address http://<gym-head>:8267        # collect_rollouts + merge/enrich/analyze -> SUCCEEDED
tail -f <base_output_dir>/.../step-5-collect-rollouts/equivalence_llm_judge/rollout/rs0/logs/ng_run_*.log
# -> Collecting rollouts 1/24 ... 24/24   (counter advancing = healthy)
# the BYO serve log shows: "POST /v1/chat/completions HTTP/1.1" 200 OK
```
> **Post-rollout `rich`/deps note.** The `collect_rollouts` merge/enrich/analyze steps run nvflow
> postprocess modules **in the nemo-gym image**, which carries the `gym` CLI venv but not nvflow's full
> Python deps — you may hit `ModuleNotFoundError: No module named 'rich'`. Workaround + durable fix are
> in the Troubleshooting entry below (do not change code to work around it).

---

## Troubleshooting (Ray-specific)

- **`ray: command not found` / `nflow: command not found`** — for release runs, use the
  nvflow-client image and prefix with `uv run --no-sync`; for source development, run `uv sync`
  once before using the same command form.
- **`Got unexpected extra argument (++…=…)`** — `nflow` has no `++` CLI override; edit the YAML.
- **`Error: Failed to connect to Ray at address: http://<ip>:8265`** — `dashboard_url` in
  `my_cluster.yaml` is a **stale head IP** (it changes every bring-up). Re-read it from
  `grep 'RAY HEAD READY' nvflow-ray-head-<jobid>.out` and update `dashboard_url` to the current head.
- **`ModuleNotFoundError: No module named 'nvflow'` in a Ray job** — confirm the airgapped driver is
  running from nvflow-client and `backend.working_dir` points to its
  `/opt/nvflow-ray-code.zip`. Do not repair this with a host checkout, environment source overlay, or
  runtime install. Also use absolute shared `datasets_dir`/`output_dir` paths.
- **`ValueError: Ray backend requires an absolute 'nvflow_root' ...` at recipe load** — your
  `my_cluster.yaml` predates the `nvflow_root` key. Recipes interpolate `${nvflow_root}/...`, and on
  Ray the runner reads that root from the cluster config — add
  `nvflow_root: /lustre/<you>/nvflow` (absolute, visible identically on the driver host and in the
  container) per [`INSTALL-RAY.md` Step 4](../../../INSTALL-RAY.md) / `cluster_configs/template-ray.yaml`.
- **Output missing / "ephemeral" `/workspace`** — a hand-set `base_output_dir` wasn't an absolute
  mounted path; set it to `/lustre/<you>/...` in the recipe (or leave the shipped `${nvflow_root}/...`
  default in place and set `nvflow_root` in `my_cluster.yaml`, per prereq #4).
- **`submit_job() got an unexpected keyword argument 'entrypoint_label_selector'`** — Ray client < 2.54;
  re-`uv sync` (the repo floors `ray[default]>=2.54.0`). See INSTALL-RAY troubleshooting.
- **Judge auth fails in a Ray job** — the key must be in `my_cluster.yaml` `env_vars:` **and** exported
  by your cluster bring-up before `ray start`; recipe-only keys don't reach the job.
- **Policy serve OOMs at engine init on a RETRY** (`ValueError: Free ...` at vllm `multiproc_executor.py:800`
  / `EngineCore failed to start` / `RuntimeError: Engine core initialization failed`) — a **prior failed
  `collect_rollouts` attempt** (a judge error, a client kill) **orphaned the policy serve's vLLM
  tensor-parallel workers** (`VLLM::Worker_TP0/TP1`) on a reused head; they keep ~76 GB pinned per GPU and
  are not cleaned up, so the next attempt's serve has no room. Not a code bug — stale GPU state. Confirm and
  clear by PID (do **not** `pkill -f 'vllm|EngineCore'` — that pattern matches the kill command's own
  process and kills your `srun` step):
  ```bash
  srun --jobid=<head-jobid> --overlap nvidia-smi   # GPU0/1 ~76 GB held by VLLM::Worker, no running rollout = leaked
  srun --jobid=<head-jobid> --overlap bash -c \
    'nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | xargs -r kill -9 2>/dev/null; \
     sleep 6; nvidia-smi --query-gpu=index,memory.used --format=csv'   # expect all GPUs ~0 MiB
  ```
  Then re-run the stage. If the GPUs won't clear, `scancel` + re-bring-up the head for guaranteed-clean GPUs.
- **Next training attempt times out in `_create_placement_groups_internal`** (`TimeoutError: ... resources
  may be busy ...`, ~3 min) after you **Ctrl-C'd the local `nflow` driver** on a prior run — Ctrl-C on the
  driver stops the *local* client, **not** the submitted Ray *job*: the job stays `RUNNING` on the cluster
  and keeps holding its 8-GPU placement group, so the next attempt can't reserve GPUs. Clearing GPU
  *memory* (the leaked-`VLLM::Worker` fix above) does **not** help here — that frees memory, but Ray's
  *logical* resource reservation is still held by the live job. Find and stop the stale job explicitly:
  ```bash
  uv run --no-sync ray job list --address http://<gpu-head>:8265        # find the still-RUNNING <submission_id>
  uv run --no-sync ray job stop --address http://<gpu-head>:8265 <submission_id>
  ```
  (Or use the dashboard **Jobs** tab to find the submission and Stop it.) Then re-run. Notes:
  - **Use `ray job ...`, not `ray status`.** `ray status` does a **strict Ray-version check** and errors out
    when the cluster's Ray (e.g. 2.54) differs from your local venv's Ray (e.g. 2.55); the `ray job ...`
    commands are version-tolerant. You can also address the GCS endpoint directly at `<gpu-head>:6379`.
  - **No `ray` CLI on the login node?** Drive the same operations through the dashboard REST API at
    `http://<gpu-head>:8265/api/jobs/` (GET to list, the per-job stop endpoint to stop).
- **`collect_rollouts` fails with a NeMo-Gym `/run` 500** (`aiohttp ... ClientResponseError: 500 ...
  url='http://127.0.0.1:<port>/run'`) — the gym head's error is opaque; the *real* cause is in the gym
  server log at `…/step-5-collect-rollouts/<env>/rollout/rs<seed>/logs/ng_run_*.log`. Most often it's the
  **reward judge** rejecting the key: look for `judge_model ... 500` / `Authorization failed` /
  `ValidationError ... NeMoGymResponse`. Verify your judge key directly —
  `curl -o /dev/null -w '%{http_code}\n' <openai_base_url>/chat/completions -H "Authorization: Bearer $KEY"
  -H 'Content-Type: application/json' -d '{"model":"<judge_model>","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'`
  should return `200` (a `401/403` means a stale/unentitled key — refresh it and re-run; the head can stay up).
- **GRPO `training` stalls with judge `429` retry spam, no step / checkpoint** (`judge_model … status=429
  kind=rate_limit try=…/max_tries=…` looping in the training log) — each training step fires
  `num_prompts_per_step × num_generations_per_prompt` reward-judge calls in a **burst**, which exceeds the
  per-key rate limit on a **shared / public** judge endpoint; the step never gets its rewards so training
  never advances (no `step_<N>` written). Not an nvflow bug — a judge quota limit. Fix: lower
  `num_generations_per_prompt` / `num_prompts_per_step` in `stages.training.overrides.grpo` to shrink the
  per-step burst (`2` is the GRPO minimum; keep `train_global_batch_size` consistent) **and/or** point the
  judge at a **higher-rate-limit or self-hosted** endpoint. For a real (non-smoke) run do **both** — keep
  rollout volume high for training quality and use a higher-limit / self-hosted judge.
- **`collect_rollouts` stalls partway with judge `429` retry spam, policy serve goes idle** (progress sticks
  at e.g. `Collecting rollouts 42% (10/24)` and never finishes) — this is the **rollout-side** analog of the
  `training` 429 above. `collect_rollouts` fires `num_samples_in_parallel` **concurrent** reward-judge calls;
  against a **shared / public** judge endpoint (e.g. `integrate.api.nvidia.com`) a wide fan-out exceeds the
  per-key rate limit, the judge returns HTTP `429` on every call, the gym retries each call ~1300× and the
  rollout wedges (the policy vLLM serve then sits idle with no new requests). Confirm it by grepping the
  rollout gym log for `429`:
  ```bash
  grep -m1 429 <base_output_dir>/step-5-collect-rollouts/<env>/rollout/rs*/logs/ng_run_*.log
  # e.g. (judge_model) [model_retry ... status=429 kind=rate_limit try=.../max_tries=...]
  ```
  Not an nvflow bug — a judge quota limit. Fix: lower `num_samples_in_parallel` in
  `stages.collect_rollouts.rollout` to throttle the concurrent judge calls (the smoke recipe ships `4`;
  `max_num_samples` stays at `24` so the variance filter still sees enough reward-varied rows) **and/or**
  point the judge at a **higher-rate-limit or self-hosted** endpoint. For a real (non-smoke) run do **both** —
  raise `num_samples_in_parallel` for throughput and use a higher-limit / self-hosted judge.

**GRPO external-serve (3-cluster) Troubleshooting**

- **`Error: backend.name: ray with executor: none ... requires a dashboard URL`** (on
  `collect_rollouts`) — the per-stage router could not resolve a dashboard for the gym stage.
  `collect_rollouts` uses `target_cluster: gym`, which reads `backend.cpu_nemo_gym_dashboard_url`. If
  that key is missing the router **silently falls back** to the nemo-skills CPU cluster
  (`gym_url or cpu_url or gpu_url`) — which has **no** `gym` CLI, so the stage misroutes and fails there.
  (If a CPU dashboard *is* set but only the gym key is missing, the symptom is instead a **gym-CLI-not-found**
  error from nemo-skills; the literal *requires a dashboard URL* string means **no** dashboard resolved at
  all — e.g. the cluster config failed to load.)
  **Fix:** define **all three** `*_dashboard_url` keys in `my_cluster.yaml` `backend:` (especially
  `cpu_nemo_gym_dashboard_url`), pointing at the *current* head IPs. For a **single-cluster** setup,
  delete the three role-specific keys and set a plain `backend.dashboard_url: http://<head>:<port>`
  instead.
- **`KeyError ... Environment variable 'NVIDIA_API_KEY' (or 'OPENAI_API_KEY') not found`** at recipe
  load — the judge `openai_api_key: ${oc.env:<VAR>}` uses an OmegaConf `oc.env` resolver that is
  evaluated **driver-side** when the config loads, **not** in the job. **Fix:** `export <VAR>=...` on
  the host running `nflow` **before** the run (`export NVIDIA_API_KEY=nvapi-...` for the NVIDIA-hosted
  form, `export OPENAI_API_KEY=sk-...` for OpenAI). The cluster `env_vars:` do **not** satisfy an
  `${oc.env:...}` reference — that is a separate mechanism (see the "Judge auth fails in a Ray job"
  entry above for the `env_vars:`-only fallback form).
- **Judge `401`/`403`/`model_not_found`, or an auth error against a valid-looking key** — the judge
  **endpoint, key, and model must match** (see the Step 5-recommended (d) table). `nvapi-…` keys work
  only against `https://integrate.api.nvidia.com/v1` with an NVIDIA-hosted model (e.g.
  `openai/gpt-oss-120b`); `sk-…` keys work only against `https://api.openai.com/v1` with an OpenAI model
  (e.g. `gpt-4o`). A `nvapi-` key sent to `api.openai.com` (or an NVIDIA model id requested from OpenAI)
  fails auth / model-not-found. **Fix:** align `openai_base_url` ↔ key ↔ `openai_model` to one provider.
- **`404` / model-not-found at rollout** (`... "POST /v1/chat/completions" 404`, or the gym `/run`
  errors on an unknown model) — the BYO **policy** serve's `--served-model-name` does **not** byte-equal
  the recipe's `policy_vllm.model_path`. In external mode `model_path` is sent **verbatim** as the
  OpenAI request `model` (`helpers.py`) — it is not a filesystem path and is not loaded locally, so any
  mismatch is a 404. **Fix:** `curl -s http://<serve>:5000/v1/models` and make `.data[0].id`
  byte-identical to `policy_vllm.model_path` (whitespace, org prefix, and case all count).
- **`FileNotFoundError: .../finance_openqa_judge.txt` under `/nemo_run/code`** — the judge prompt
  template asset was not packaged. nemo-run packages a **git-archive of `HEAD`** into `/nemo_run/code`,
  so any file the job reads from there **must be committed**. The shipped
  [`finance_openqa_judge_overlay.yaml`](../../../nvflow/recipes/finance/prompts/finance_openqa_judge_overlay.yaml)
  already points `judge_prompt_template_fpath` at
  `/nemo_run/code/nvflow/recipes/finance/prompts/finance_openqa_judge.txt` (a committed file), so this
  is a **prevention** note: if you add or edit a judge prompt, **`git add` + commit it** before the run
  (an uncommitted/untracked prompt is invisible to the git-archive), or point
  `judge_prompt_template_fpath` at a `${nvflow_root}`-absolute path on the shared mount that resolves
  live in the job.
- **`ModuleNotFoundError: No module named 'rich'`** in the post-rollout merge/enrich/analyze — those
  steps run nvflow postprocess modules (`nvflow.recipes.finance.utils.rl.{enrich,analyze}_rollouts`)
  **in the nemo-gym image**. Their import chain (`nvflow.recipes.__init__` → `nvflow.core.console` →
  `from rich.console import Console`) needs `rich`, which older nemo-gym images omit. **Fix:**
  rebuild/repull a nemo-gym image that bakes `rich` at build time. **Do NOT** work around it at runtime
  by injecting the driver venv's `site-packages` onto the job `PYTHONPATH` — that violates the air-gap
  requirement and drags an incompatible `fastapi` into the gym server, crashing it at startup.
- **`collect_rollouts` stalls partway with judge `429`** (`Collecting rollouts stuck at N%`, policy
  serve goes idle) — a **shared / hosted** judge (e.g. `integrate.api.nvidia.com`) rate-limits a wide
  fan-out. Keep `rollout.num_samples_in_parallel: 4` for a hosted judge; raise it only with an `sk-` /
  self-hosted higher-limit judge. This is the same failure as the **rollout-`429`** entry above — see it
  for the confirming `grep` and full fix.

- **`training` (external-serve) submits and loads the model, but the GPUs sit IDLE and no step
  advances** — external-serve delegates **all** rollout generation to the BYO policy endpoint, so with
  `policy_vllm.base_url` unreachable the GRPO loop loads the policy onto the GPUs and then **blocks**
  waiting on generation — no error, just idle GPUs. Almost always the `<POLICY_VLLM_HOST>` placeholder
  was never replaced, or the BYO `vllm serve` is down/`PD`. **Fix:** bring the BYO policy serve up first,
  verify `curl -s http://<serve>:5000/v1/models` → `.data[0].id == policy_vllm.model_path`, `sed` the
  real `http://<ip>:5000/v1` into `base_url`, **then** resubmit. A job already started against the
  placeholder will not recover — `ray job stop <submission_id>` it (frees the 8 GPUs) before re-running.
- **`pydantic ... ValidationError` for `policy.hf_config_overrides` (Input should be a valid dictionary)**
  at training start — a GRPO recipe set `hf_config_overrides: null`; NeMo-RL's `MasterConfig.policy` is a
  pydantic model that rejects `None`. **Fix:** set `hf_config_overrides: {}` (empty dict = "no HF config
  overrides") in the training-policy block. The shipped `grpo/qwen3_4b_smoke.yaml` now uses `{}`; if you
  copied an older recipe, change `null` → `{}`.
- **`FileNotFoundError` for an nvflow config/prompt during a `nemo-gym` stage** (a `config_paths`,
  `prompt_config`, or `judge_prompt_template_fpath` that "exists") — the gym stage `cd`s into `/opt/Gym`
  before running, so any **relative** path resolves under `/opt/Gym`, not your repo. **Fix:** use absolute
  paths — either `/nemo_run/code/nvflow/...` (the git-archive package location, for committed files) or a
  `${nvflow_root}`-absolute `/lustre/...` path on the shared mount.
- **gym rejects a reused dataset with a path / `jsonl_fpath` mismatch** — a copied step-3 dataset carried
  a stale `final_result_metrics.json` (and `..._conflict.json`) with an old `jsonl_fpath` baked in.
  **Fix:** `rm` those sidecar files from the staged dataset dir and re-run `prepare_data`; gym regenerates
  them for the new path.
- **A stage prints `✅ ... completed` but produced no output** (e.g. `prepare_data` emits only
  `agent_config_overlay.yaml`, no `train.jsonl`; `train_validation_split`/`training` then fail on missing
  data) — completion detection keys off the log stream, not the job's exit/artifacts, so a stage that
  silently no-ops still reports success. The common cause is `prepare_data` routed to a cluster with no
  `gym` CLI because the **nemo-gym head is missing** (`cpu_nemo_gym_dashboard_url` unset → silent fallback
  to nemo-skills). **Fix:** don't trust the ✅ — verify the artifact
  (`test -s <base_output_dir>/.../step-*/.../train.jsonl`); if empty, set `cpu_nemo_gym_dashboard_url`,
  bring the gym head up, and re-run.

- **`yq: command not found` when patching YAML** — cluster login nodes typically do not ship `yq`.
  Use `sed -i` to substitute placeholder values into a `/tmp` copy of the recipe; never commit
  cluster-specific IPs or host paths:
  ```bash
  cp nvflow/recipes/finance/workflows/grpo/qwen3_4b_smoke_extserve.yaml /tmp/grpo_extserve.yaml
  sed -i "s|base_url:.*|base_url: http://${SERVE_IP}:5000/v1|" /tmp/grpo_extserve.yaml
  sed -i "s|gpu_nemo_rl_dashboard_url:.*|gpu_nemo_rl_dashboard_url: http://${HEAD_IP}:8265|" \
    cluster_configs/my_cluster.yaml
  ```
- **`srun` rejected on a GPU partition with `Batch job submission failed`** — GPU partitions reject any
  `srun` that specifies 0 GPUs. Always include `--gpus-per-node=1` (or `--gres=gpu:1`) for any `srun`
  on a GPU partition, even a one-shot verification command.
- **In-container `python` / `vllm` is the wrong version** — a login-shell invocation (e.g. via
  `bash -lc`) resets `$PATH` to the base system Python, bypassing the venv. For in-container
  verification use the **explicit venv path**, e.g.:
  `/opt/nvflow/.venv/bin/python -c "import ray; print(ray.__version__)"` for the nvflow client image,
  or `/opt/nemo_rl_venv/bin/python` / `/opt/gym-cli-venv/bin/python` for the respective training /
  gym images.
- **`convert_dcp_to_hf.py --backend dtensor` fails with `FileNotFoundError: No metadata file found`
  at checkpoint convert time** — the NVFlow GRPO preset sets `save_consolidated: false`, so nemo-rl
  v0.7 saves weights as HF safetensors shards under `checkpoints/step_N/policy/weights/model/` with
  a `.hf_metadata/` directory, not a DCP `.metadata` file. The `convert_dcp_to_hf.py` script expects
  the DCP format and finds nothing. Use Automodel's official
  `/opt/nemo-rl/3rdparty/Automodel-workspace/Automodel/tools/offline_hf_consolidation.py`
  instead (the NVFlow `format: fsdp` conversion path auto-detects this DTensor-v2 format). Do not
  concatenate shards manually: TP-sharded projections require metadata-aware consolidation. Validate
  the resulting tensor shapes and load the export in vLLM before starting evaluation.

See also [`INSTALL-RAY.md` → Troubleshooting](../../../INSTALL-RAY.md) and
[`troubleshooting.md`](troubleshooting.md).

---

[Slurm Quick Start](quick-start.md) | [Ray Install & Setup](../../../INSTALL-RAY.md) | [Finance Recipe README](README.md) | [Main README](../../../README.md)
