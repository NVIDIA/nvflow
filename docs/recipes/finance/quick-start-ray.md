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
   clusters and write `cluster_configs/my_cluster.yaml` per [`INSTALL-RAY.md`](../../../INSTALL-RAY.md)
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
   grep dashboard_url cluster_configs/my_cluster.yaml   # -> http://<head-ip>:8265 (matches the .out above)
   ```

2. **Orchestrator installed with the `skills` extra.** The Ray Jobs backend lives in `nemo_skills`,
   which the default `uv sync` omits. Use one of:
   - **Source / dev:** `uv sync --extra skills`, then prefix every command with `uv run`
     (`uv run nflow ...`, `uv run ray ...`) so the project venv is active. A bare `ray`/`nflow` will be
     "command not found".
   - **In-container (airgap):** mount your nvflow checkout and run the module directly (no `nflow`
     console-script in the image):
     ```bash
     PYTHONPATH=/workspace python -m nvflow.cli.main run-all --config <recipe>
     ```
     where `/workspace` is the nvflow repo **root** (the dir containing both `scripts/` and the
     `nvflow/` package — **not** the `nvflow/` package subdir), the mount your `my_cluster.yaml` maps.

3. **Ray client ≥ 2.54.0.** A fresh `uv sync --extra skills` gives you this (the repo floors
   `ray[default]>=2.54.0`). If you see `submit_job() got an unexpected keyword argument
   'entrypoint_label_selector'`, your env is stale — re-sync. Exact version match across the two
   cluster images is **not** required.

4. **Absolute paths visible on BOTH the driver and the cluster.** On a pre-provisioned Ray cluster the
   `mounts:` in `my_cluster.yaml` are **not** applied per-job — jobs run with the cluster's existing
   mounts. And with `executor: none` (Mode 3) nemo-skills does the benchmark-module import, the eval
   data-file existence check, and the job-manifest writes **on the driver** (your `uv run nflow` host),
   not in the job. So every `base_output_dir`, `datasets_dir`, data path, and model path **must be an
   absolute path that resolves identically on the driver host AND inside the cluster container** — use
   `/lustre/<you>/...` (mounted at the same place on both), **not** `/workspace/...` (that path exists
   only inside the container; on a login-node driver it gives `No module named '<benchmark>'` and
   `Permission denied: '/workspace'`). The recipe defaults ship `/workspace/...` for the in-container
   driver — **override them for the host-driver path.** Set this in the recipe (or a `_base_` override)
   before running.

5. **No `++` CLI overrides.** `nflow` does **not** accept `++key=value` on the command line — edit the
   value in the recipe YAML instead. (`++` inside a recipe's `inline_args` is fine; that's forwarded to
   the underlying tool.)

6. **The recipe's `cluster:` field selects the cluster** — there is no `--cluster` flag. Set
   `cluster: my_cluster` in each recipe; `run-all` then auto-routes GPU stages → the nemo-rl cluster
   and CPU stages → the nemo-skills cluster (override per stage with `target_cluster: cpu|gpu`).

7. **`PYTHONPATH=/workspace` is set in the cluster config `env_vars:`** (per
   [`INSTALL-RAY.md` Step 4](../../../INSTALL-RAY.md)). This is the **first-stage blocker**: every
   stage submits a `python -m nvflow…` job to the cluster, the image bakes `nemo_skills` but **not**
   `nvflow`, and the Jobs backend ships no working-dir — so without this the *very first* job fails with
   `ModuleNotFoundError: No module named 'nvflow'`. Confirm it is in `my_cluster.yaml` before Step 0.

8. **`--config` is cwd-relative.** Run from the repo root and pass the full path from there (e.g.
   `nvflow/recipes/finance/workflows/grpo/qwen3_4b_smoke.yaml`), exactly as the commands below show — a
   short form like `grpo/qwen3_4b_smoke.yaml` fails with `No such file or directory` unless a
   recipe-search dir is configured.

### Slurm → Ray command cheat-sheet

| Slurm (quick-start.md) | Ray (here) |
|---|---|
| `sbatch` submits compute | you pre-provision; `nflow` submits over HTTP to the dashboard |
| `squeue --me` | `uv run ray job list --address http://<head>:<port>` or the dashboard **Jobs** tab |
| tail `…/training-logs/ray-*-job.log` | dashboard **Jobs** tab → job logs (on the shared mount) |
| relative `outputs/...` paths | **absolute** `/lustre/<you>/...` on the shared mount |
| `bare nflow …` | `uv run nflow …` (source) or `PYTHONPATH=<mount> python -m nvflow.cli.main …` (container) |

---

## Step 0 — Hello-world smoke (do this first)

Validate the Jobs-API path end to end before any real stage:
```bash
uv run nflow run-all --config nvflow/recipes/finance/workflows/ray_hello_world.yaml
# -> "Ray job ... finished with status SUCCEEDED"
```
First edit `base_output_dir` in `ray_hello_world.yaml` to an **absolute mounted path** (the default
`/workspace/...` is ephemeral unless your mount maps it).

**What to expect:** the driver prints `✓ Workflow Complete!` once the job is *submitted*; the job then
runs on the cluster and ends `SUCCEEDED`. **Verify** the job status and the artifact it wrote to the
shared mount:
```bash
uv run ray job list --address http://<cpu-head>:8266     # the hello_world job -> SUCCEEDED
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
`cluster:` + an absolute `base_output_dir` in the recipe, then run with `uv run`, and verify with Ray
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
Slurm. Mode-3 hands each stage to the Ray cluster as a job; the actual compute then runs on the cluster.
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

### Step 1 — Baseline eval (CPU cluster, against external serves)

**Easiest path — use the ready-made `eval/demo_ray.yaml`.** It is `eval/demo.yaml` with every Ray
override already applied: CPU routing (`target_cluster: cpu` + `gpus: 0`), external generation
`server_address`, the **external-judge repoint**, and absolute `/lustre` paths. Fill its 3 placeholders
(`<GEN_SERVE>`, `<JUDGE_SERVE>`, `/lustre/<you>`) and run:
```bash
uv run nflow run prepare_data --config nvflow/recipes/finance/workflows/eval/demo_ray.yaml
uv run nflow run qwen3-4b     --config nvflow/recipes/finance/workflows/eval/demo_ray.yaml
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
- **Paths (absolute `/lustre`, per prereq #4).** The `/workspace/...` defaults in `eval/base.yaml`
  (`datasets_dir`) and `eval/demo.yaml` (`base_output_dir`, `stages.prepare_data.output_dir`) exist only
  in-container, but with `executor: none` the benchmark import, the data-file check, and the manifest
  writes run **driver-side** — on a host driver they fail with `No module named '<benchmark>'` then
  `Permission denied: '/workspace'`. Set `base_output_dir` + `datasets_dir` to `/lustre/<you>/...`, and
  point `stages.prepare_data.output_dir` at the **same** `datasets_dir` (prepare_data must write where the
  eval reads).

**Verify** (read from the absolute output dir you set):
```bash
uv run ray job list --address http://<cpu-head>:8266     # prepare_data + qwen3-4b jobs -> SUCCEEDED
cat /lustre/<you>/.../baseline-eval/qwen3-4b/eval-results/secque/metrics.json
```
**Reference (baseline ballpark):** secque `pass@1` `judge_correct` ≈ 50%, `no_answer` ≈ 0% for base
Qwen3-4B (see quick-start.md Step 1 for the full reference table). A high `no_answer` means a serve/judge
or reasoning-parser misconfig, not a model-quality result.

### Step 2 — Download SEC filings (CPU cluster)
```bash
uv run nflow run demo --config nvflow/recipes/finance/workflows/download_sec_filings.yaml
```
(SEC EDGAR is reached from the cluster — ensure the cluster has the needed outbound access, or stage
the data, in an air-gapped environment.) **Verify** the job reached `SUCCEEDED` and the expected
artifacts landed under your absolute output dir:
```bash
uv run ray job list --address http://<cpu-head>:8266     # demo download job -> SUCCEEDED
ls /lustre/<you>/.../workflow-2-download-sec/step-0-download/data/
# Expected: AAPL/ CSCO/ GOOG/ IBM/ META/ MSFT/ NVDA/
ls /lustre/<you>/.../workflow-2-download-sec/step-0-download/sec_metadata.parquet
```

### Step 3 — Synthetic Q&A (SDG) (CPU cluster)
```bash
uv run nflow run-all --config nvflow/recipes/finance/workflows/sdg/template-based-sdg-demo.yaml
```
**Verify** all SDG jobs reached `SUCCEEDED` and the final Q&A set is the expected size:
```bash
uv run ray job list --address http://<cpu-head>:8266     # the SDG-stage jobs -> SUCCEEDED
wc -l /lustre/<you>/.../workflow-3-template-based-sdg/step-5-filter-answers/final_result.jsonl
# Expected: ~1000–1200 Q&A pairs (input for SFT/GRPO)
```

### Step 4 — SFT + eval (GPU train, CPU eval)
Training auto-routes to the GPU (nemo-rl) cluster; eval to the CPU cluster.
```bash
uv run nflow run-all --config nvflow/recipes/finance/workflows/sft/qwen3_4b.yaml
```
**Monitor on Ray instead of `squeue`:**
```bash
uv run ray job list --address http://<gpu-head>:8265   # training job -> SUCCEEDED
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
> to survive training). Five GRPO-specific edits beyond the shared `/lustre`-paths + `PYTHONPATH=/workspace`
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
>      baked into the job — so export that var **on the driver host** (where `uv run nflow` runs). The
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
> uv run nflow run prepare_data collect_rollouts train_validation_split training \
>   --config nvflow/recipes/finance/workflows/grpo/qwen3_4b_smoke.yaml -e equivalence_llm_judge
> # the training job is a Ray job — monitor it on Ray, NOT `squeue`:
> uv run ray job list --address http://<gpu-head>:8265        # the training job -> SUCCEEDED
> uv run ray job logs --address http://<gpu-head>:8265 <job-id>   # live training log (or dashboard Jobs tab)
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

Same two environments and per-stage commands as quick-start.md Step 5 — run them with `uv run` and an
**external** judge (`base_url`/`openai_base_url`), e.g. for `equivalence_llm_judge`:
```bash
uv run nflow run data_transformation apply_prompt_template convert_to_responses_api prepare_data \
  --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml -e equivalence_llm_judge
uv run nflow run collect_rollouts        --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml -e equivalence_llm_judge
uv run nflow run train_validation_split  --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml -e equivalence_llm_judge
uv run nflow run training                --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml -e equivalence_llm_judge
uv run nflow run eval                    --config nvflow/recipes/finance/workflows/grpo/qwen3_4b.yaml -e equivalence_llm_judge
```
(For `finance_sec_search`, also run `validate_questions` + `prefetch_cache` and train with
`grpo/qwen3_4b_finsec.yaml`, exactly as in quick-start.md Step 5.) The Ray backend submits independent
jobs **concurrently** — watch the fan-out on the dashboard Jobs tab or `ray job list`. Use a
**dedicated** judge endpoint for the rollout/eval fan-out (a shared endpoint rate-limits/429s). Verify
rollout `summary.txt` / `filter_report.json`, checkpoints, and eval `metrics.json` under your absolute
output dir (paths mirror quick-start.md Step 5).

---

## Troubleshooting (Ray-specific)

- **`ray: command not found` / `nflow: command not found`** — run `uv sync --extra skills` and prefix
  with `uv run` (or, in-container, `PYTHONPATH=<mount> python -m nvflow.cli.main`).
- **`Got unexpected extra argument (++…=…)`** — `nflow` has no `++` CLI override; edit the YAML.
- **`Error: Failed to connect to Ray at address: http://<ip>:8265`** — `dashboard_url` in
  `my_cluster.yaml` is a **stale head IP** (it changes every bring-up). Re-read it from
  `grep 'RAY HEAD READY' nvflow-ray-head-<jobid>.out` and update `dashboard_url` to the current head.
- **`ModuleNotFoundError: No module named 'nvflow'` in a Ray job** — the cluster config `env_vars:` must
  include `PYTHONPATH=/workspace` (the nvflow repo mount) so the *submitted job* can import nvflow; the
  image bakes nemo_skills but not nvflow, and the Jobs backend ships no working-dir. (This is separate
  from the driver-side `PYTHONPATH=/workspace` used to launch the in-container CLI.) Also use absolute
  `datasets_dir`/`output_dir`.
- **Output missing / "ephemeral" `/workspace`** — your `base_output_dir` wasn't an absolute mounted path;
  set it to `/lustre/<you>/...` in the recipe.
- **`submit_job() got an unexpected keyword argument 'entrypoint_label_selector'`** — Ray client < 2.54;
  re-`uv sync --extra skills` (the repo floors `ray[default]>=2.54.0`). See INSTALL-RAY troubleshooting.
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
  uv run ray job list --address http://<gpu-head>:8265        # find the still-RUNNING <submission_id>
  uv run ray job stop --address http://<gpu-head>:8265 <submission_id>
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

See also [`INSTALL-RAY.md` → Troubleshooting](../../../INSTALL-RAY.md) and
[`troubleshooting.md`](troubleshooting.md).

---

[Slurm Quick Start](quick-start.md) | [Ray Install & Setup](../../../INSTALL-RAY.md) | [Finance Recipe README](README.md) | [Main README](../../../README.md)
