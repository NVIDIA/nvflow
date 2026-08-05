# Running `nflow` over an SSH tunnel

`nflow` is only a **submission orchestrator**: it builds Slurm jobs and submits
them — all data, GPU work, and training run in the worker containers **on the
cluster**. When you run `nflow` somewhere that can't reach Slurm directly (a
laptop, a dev box, or an isolated/airgapped environment), it submits over an
**SSH tunnel**.

> **On a cluster login/dev node?** You don't need this doc — install per the
> [README](../README.md#-installation) and run `nflow` directly. This page is for
> the **off-cluster / tunneled** case. For all client options at a glance, see
> [INSTALL.md → Choose your client setup](../INSTALL.md#choose-your-client-setup).

## Options at a glance

```text
Where does `nflow` run?
│
├─ On a cluster login/dev node ────────────▶ sbatch ─▶ Slurm worker jobs   (no tunnel)
│     install: uv sync
│
└─ Off-cluster (laptop / dev box / airgap) ──ssh_tunnel──▶ login node ─sbatch─▶ workers
      provision the launcher, pick one:
        A. host install       — uv sync (client host needs internet)
        B. nvflow-client image — no uv sync, no client internet
              ├─ enroot        (cluster node)
              ├─ docker/podman (off-cluster machine)
              └─ pyxis srun    (cluster node, via Slurm)

Worker jobs (nemo-skills · vllm · vllm-grpo · nemo-rl · nemo-gym · sglang)
always run on the cluster; the client only submits.
```

## Prerequisites

- **Cluster side is set up** ([INSTALL.md](../INSTALL.md)): worker `.sqsh` images
  and models are staged, and you have a `my_cluster.yaml`.
- **SSH key auth** to a cluster login node that can run `sbatch`:
  ```bash
  ssh -i <key> <user>@<host> 'hostname && command -v sbatch'
  ```

## Step 1 — Configure `my_cluster.yaml` (add the tunnel)

Put `my_cluster.yaml` where the launcher reads it — **container:** the mounted
`/work` dir (`NEMO_SKILLS_CONFIG_DIR=/work`); **host install:** `cluster_configs/`.
Add an `ssh_tunnel` block so the launcher reaches Slurm over SSH (no Slurm client
or Lustre needed on the client):

```yaml
ssh_tunnel:
  host: <login node you SSH into to run sbatch>
  user: <username>
  identity: <path to your SSH key>   # container: /opt/ssh/<key> (id_rsa / id_ed25519)
  job_dir: <absolute cluster path where the tunnel stages jobs>
```

> `/work` is a **bind mount** — prepare `my_cluster.yaml` before starting the
> container, or edit it live afterward; it just must be complete before
> `nflow run`. It holds secrets: keep it in `/work`, never bake it into an image.

The rest of `my_cluster.yaml` is your standard cluster config (containers,
`mounts:`, `env_vars`); `ssh_tunnel` is the only tunnel-specific addition. In
`mounts:`, keep `/hf_models` and point `/workspace` at a **writable data dir**
(outputs + HF cache) — **not** the repo checkout. Recipe code and checked-in
assets reach workers via the packaged snapshot (`/nemo_run/code`), so the repo is
never mounted. See [cluster-configuration.md → Mounts](cluster-configuration.md#mounts).

## Step 2 — Start the launcher (pick one)

### A. Host install (`uv sync`) — client host has internet

Follow the [README install](../README.md#-installation) (`git clone` + `uv sync`).
Invoke the CLI as **`uv run nflow …`**. No client internet? Use the
`nvflow-client` image (option B below) instead.

### B. Client container — airgapped / no local install (invoke as **`nflow …`**)

The `nvflow-client` image bundles the `nflow` CLI + venv (no `uv sync`, no client
internet). Start it, mounting your **SSH key** (`→ /opt/ssh`) and the **`/work`**
dir holding `my_cluster.yaml`:

```bash
# --- Cluster node (enroot) — if the .sqsh is already staged, skip the import ---
enroot import -o nvflow-client.sqsh 'docker://<registry>#<org>/nvflow-client:<tag>'  # only from a registry ref
enroot create --name nvflow-client /path/to/nvflow-client.sqsh
ENROOT_MOUNT_HOME=n enroot start --rw \
  -m ~/.ssh:/opt/ssh -m /path/to/work:/work \
  -e NEMO_SKILLS_CONFIG_DIR=/work nvflow-client bash

# --- Cluster node via Slurm (pyxis/srun) — starts from the .sqsh directly ---
srun --container-image=/path/to/nvflow-client.sqsh \
  --container-mounts=/path/to/work:/work,$HOME/.ssh:/opt/ssh \
  --container-workdir=/opt/nvflow \
  --export=ALL,NEMO_SKILLS_CONFIG_DIR=/work --pty bash

# --- Off-cluster machine (docker/podman) ---
docker run --rm -it -v ~/.ssh:/opt/ssh:ro -v /path/to/work:/work \
  -e NEMO_SKILLS_CONFIG_DIR=/work <registry>/nvflow-client:<tag> bash
```

> Prefer **enroot** (cluster) or **docker/podman** (off-cluster); the `srun` form
> burns an allocation just to host the launcher. Do **not** bind-mount over
> `/opt/nvflow` (baked source/venv/`.git` that nemo-run packages via `git archive`).
> Host keys auto-accept on first connect (baked `ssh_config` reads
> `/opt/ssh/known_hosts`; a *changed* key is still rejected). Build details:
> [containers.md](maintainers/containers.md).

## Step 3 — Launch and monitor over the tunnel

```bash
nflow list-stages --recipe finance          # verify: CLI loads + config resolves
nflow run <stage> -c <config> -e <env>      # submit (detaches when queued)
```

The client has **no Slurm client or cluster filesystem**, so monitor on the
cluster over the same SSH:

```bash
ssh -i <key> <user>@<host> 'squeue --me'          # or: sacct -j <jobid>
ssh -i <key> <user>@<host> 'ls <job_dir>/...'     # logs/artifacts land on Lustre
```

`nemo experiment status <exp-id>` (printed at submit) also works over the tunnel.

## Notes

- **Connected-node prerequisites** (benchmark datasets, SEC filings, model
  downloads) need internet and the `HF_*_OFFLINE` flags **off** for that one run —
  do them once per [INSTALL.md](../INSTALL.md), then keep the flags **on**. The
  container can stage models itself:
  `uv run hf download <repo> --local-dir /hf_models/<repo>` (mount the models dir).
- **Everything runs on the cluster; the client only submits.** GPU work, data
  I/O, and the rollout/judge servers all execute inside Slurm jobs. Recipe code
  and checked-in assets ship with each job via `/nemo_run/code` (see Step 1), so
  the client needs no repo and the repo is never mounted on workers.
- **Laptop / off-cluster specifics** (validated: a client with **no repo mount**
  ran the full matrix end-to-end — staging → SDG → SFT → eval and **both GRPO
  workflows** (`finance_sec_search` via the client, equivalence via a repo
  install) — proving all I/O is cluster-side and checked-in assets resolve from
  `/nemo_run/code`, incl. Gym `config_paths`, prefetch `ticker`, and judge
  fpaths):
  - `ssh_tunnel.host` must be an **FQDN reachable from the laptop** (VPN), and
    `ssh_tunnel.identity` your **local** key (e.g. `~/.ssh/id_rsa`).
  - `mounts:` and `job_dir` are **cluster Lustre paths**; the laptop needs none of
    them locally. Resume/chunk-skip is probed over the tunnel (`LauncherFS`), so
    **no local mount is required** — and while `ssh_tunnel` is set a local mount
    is ignored anyway. (A client running **on-cluster without** `ssh_tunnel` must
    run from the repo root so `resolve_host_path` can map `/workspace/outputs/...`
    back to the host outputs dir for skip-detection.)
  - Dev-mode source overlays (Gym / NeMo-RL) must live **on the cluster**, not the
    laptop — they bind into the worker jobs.
