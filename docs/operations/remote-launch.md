# Running `nvflow` over an SSH tunnel

`nvflow` is only a **submission orchestrator**: it builds Slurm jobs and submits
them — all data, GPU work, and training run in the worker containers **on the
cluster**. When you run `nvflow` somewhere that can't reach Slurm directly (a
laptop, a dev box, or an isolated/airgapped environment), it submits over an
**SSH tunnel**.

> **Installed with `uv sync` on a cluster login/dev node?** You don't need this
> doc; run `nvflow` directly per the [README](../../README.md#get-started). This
> page covers every other case, including the `nvflow-client` container on a
> cluster node, which also submits over the tunnel. For all client options at a
> glance, see
> [INSTALL.md → Choose your client setup](../../INSTALL.md#choose-your-client-setup).
>
> **Ray target?** Ray targets take the same `ssh_tunnel` block; see
> [Prepare the launcher](../ray/README.md#1-prepare-the-launcher).

## Options at a glance

```text
Where does `nvflow` run?
│
├─ Cluster login/dev node, uv sync install ──▶ sbatch ─▶ Slurm worker jobs   (no tunnel)
│
└─ Anywhere else ──ssh_tunnel──▶ login node ─sbatch─▶ workers
      provision the launcher, pick one:
        A. host install       — uv sync (client host needs internet)
        B. nvflow-client image — no clone, no uv sync, no client internet
              ├─ enroot        (cluster node)
              ├─ docker/podman (off-cluster machine)
              └─ pyxis srun    (cluster node, via Slurm)

Worker jobs (nemo-skills · vllm · sglang · nemo-rl · nemo-gym)
always run on the cluster; the client only submits.
```

## Prerequisites

- **Cluster side is set up** ([INSTALL.md](../../INSTALL.md) and
  [Slurm execution](../slurm/README.md)): worker `.sqsh` images
  and models are staged, and you have a `my_cluster.yaml`.
- **SSH key auth** to a cluster login node that can run `sbatch`. Run this
  check where the launcher will run, with the key it will use:
  ```bash
  ssh -i <key> <user>@<host> 'hostname && command -v sbatch'
  ```
  The client container has no Slurm client, so even on a cluster node it
  submits over SSH. If the check fails there, add that key's public key to
  `~/.ssh/authorized_keys` on the cluster.

## Step 1 — Configure `my_cluster.yaml` (add the tunnel)

Put `my_cluster.yaml` where the launcher reads it — **container:** the mounted
`/work` dir (`NVFLOW_CONFIG_DIR=/work`); **host install:** `cluster_configs/`.
Add an `ssh_tunnel` block so the launcher reaches Slurm over SSH (no Slurm client
or cluster filesystem needed on the client):

```yaml
ssh_tunnel:
  host: <login node you SSH into to run sbatch>
  user: <username>
  identity: <path to your SSH key>   # container: /opt/ssh/<key> (id_rsa / id_ed25519)
  job_dir: <absolute cluster path where the tunnel stages jobs>
```

> `/work` is a **bind mount** — prepare `my_cluster.yaml` before starting the
> container, or edit it live afterward; it just must be complete before
> `nvflow run`. It holds secrets: keep it in `/work`, never bake it into an image.

The rest of `my_cluster.yaml` is your standard cluster config (containers,
`mounts:`, `env_vars`); `ssh_tunnel` is the only tunnel-specific addition.
Without a clone, start from the template the image ships: inside the container,
run `cp /opt/nvflow/cluster_configs/template-slurm.yaml /work/my_cluster.yaml`,
then edit the copy on the host. In
`mounts:`, keep `/hf_models` and point `/workspace` at a **writable data dir**
(outputs + HF cache) — **not** the repo checkout. Project code and checked-in
assets reach workers via the packaged snapshot (`/nemo_run/code`), so the repo is
never mounted. See [cluster-configuration.md → Mounts](cluster-configuration.md#mounts).

## Step 2 — Start the launcher (pick one)

### A. Host install (`uv sync`) — client host has internet

Follow the [README install](../../README.md#get-started)
(`git clone` + `uv sync`).
Invoke the CLI as **`uv run nvflow …`**. No client internet? Use the
`nvflow-client` image (option B below) instead.
The first command that imports a project-owned stage prepares that project's
cached launcher environment; catalog discovery does not.

### B. Client container — airgapped / no local install (invoke as **`nvflow …`**)

The `nvflow-client` image bundles the `nvflow` CLI plus every discoverable
project environment by default (no `uv sync`, no client internet). A
space-separated `NVFLOW_PREWARM_PROJECTS` subset is for development images only;
release images verify all projects offline. Start it, mounting your **SSH key**
(`→ /opt/ssh`) and the **`/work`** dir holding `my_cluster.yaml`:

```bash
# --- Cluster node (enroot) — if the .sqsh is already staged, skip the import ---
enroot import -o /path/to/nvflow-client.sqsh \
  "docker://<registry-host>#<namespace>/nvflow-client:v1.2.0"
enroot create --name nvflow-client /path/to/nvflow-client.sqsh
ENROOT_MOUNT_HOME=n enroot start --rw \
  -m ~/.ssh:/opt/ssh -m /path/to/work:/work \
  -e NVFLOW_CONFIG_DIR=/work nvflow-client bash

# --- Cluster node via Slurm (pyxis/srun) — starts from the .sqsh directly ---
srun -A <account> -p <cpu-partition> -t 01:00:00 \
  --container-image=/path/to/nvflow-client.sqsh \
  --container-mounts=/path/to/work:/work,$HOME/.ssh:/opt/ssh \
  --container-workdir=/opt/nvflow \
  --export=ALL,NVFLOW_CONFIG_DIR=/work --container-env=NVFLOW_CONFIG_DIR --pty bash

# --- Off-cluster machine (docker/podman) ---
docker run --rm -it -v ~/.ssh:/opt/ssh:ro -v /path/to/work:/work \
  -e NVFLOW_CONFIG_DIR=/work "<registry-host>/<namespace>/nvflow-client:v1.2.0" bash
```

> - Prefer **enroot** (cluster) or **docker/podman** (off-cluster); the `srun`
>   form burns an allocation just to host the launcher.
> - `enroot import` produces a `.sqsh` for the importing node's CPU
>   architecture, so import on the same kind of node that will start it. If a
>   login node lacks enroot or cannot reach the registry, run the import as a
>   Slurm job: `srun -A <account> -p <cpu-partition> enroot import ...`.
> - The `srun` form needs `--container-env=NVFLOW_CONFIG_DIR`; without it,
>   pyxis keeps the image's own value and ignores `--export`.
> - Do **not** bind-mount over `/opt/nvflow` (baked source/venv/`.git` that
>   nemo-run packages via `git archive`).
> - Host keys auto-accept on first connect: the baked `ssh_config` reads
>   `/opt/ssh/known_hosts`, and a *changed* key is still rejected. Docker mounts
>   `~/.ssh` read-only, so SSH to the login node once from the host first; the
>   prerequisite check does this.
>
> Build details: [containers.md](../maintainers/containers.md).

## Step 3 — Launch and monitor over the tunnel

Start with the starter workflow:

```bash
nvflow plan --live --workflow starter:generate --cluster my_cluster  # reads the cluster over the tunnel
nvflow run --all --workflow starter:generate --cluster my_cluster    # one GPU job and a merge job
```

Run other workflows the same way, for example
`nvflow run --all --workflow finance:download/demo --cluster my_cluster`.

List, validation, and configuration-inspection commands are target-independent.
`plan` and `run` require explicit `--cluster`. Execution always
performs live-equivalent, stage-specific filesystem discovery and resume logic
before submission. Progress, warnings, and errors use stderr; `run` and
all its selection modes leave stdout empty.

Check your jobs from the client; `status` goes through the same tunnel:

```bash
nvflow status --cluster my_cluster
```

The client has **no cluster filesystem and no Slurm tools**, so inspect
finished jobs, logs, and artifacts on the cluster over the same SSH, or from a
cluster shell outside the container:

```bash
ssh -i <key> <user>@<host> 'sacct -j <jobid>'     # finished-job accounting
ssh -i <key> <user>@<host> 'ls <job_dir>/...'     # logs/artifacts stay on the cluster
```

## Notes

- **Connected-node prerequisites** (benchmark datasets, SEC filings, model
  downloads) need internet and the `HF_*_OFFLINE` flags **off** for that one run —
  do them once per [INSTALL.md](../../INSTALL.md), then keep the flags **on**.
  Download models with the `uvx --from huggingface_hub hf download` commands in
  [INSTALL.md → Download Models](../../INSTALL.md#download-models), on a host
  that mounts the models directory and has internet access. The client
  container has no `uvx`, so inside it run
  `uv run hf download <repo> --local-dir /hf_models/<repo>` instead; it needs
  the models directory mounted at `/hf_models` and internet access.
- **Everything runs on the cluster; the client only submits.** GPU work, data
  I/O, and the rollout/judge servers all execute inside Slurm jobs. Project code
  and checked-in assets ship with each job via `/nemo_run/code` (see Step 1), so
  the client needs no repo and the repo is never mounted on workers.
- **Laptop / off-cluster specifics:**
  - `ssh_tunnel.host` must be an **FQDN reachable from the laptop**, and
    `ssh_tunnel.identity` your **local** key (e.g. `~/.ssh/id_rsa`).
  - `mounts:` and `job_dir` are **cluster paths**; the laptop needs none of
    them locally. Stage-specific filesystem discovery and resume checks use the
    target's internal filesystem transport, so **no local mount is required**.
    While `ssh_tunnel` is set, a local mount is ignored.
