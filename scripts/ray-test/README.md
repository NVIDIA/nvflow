# Ray-on-Slurm test clusters

This directory is optional test infrastructure. NVFlow never imports or invokes
it and uses the clusters only as ordinary supplied Ray Jobs endpoints. CI tests
the launcher logic but cannot exercise site-specific Slurm/pyxis behavior.

`clusters.py` starts, checks, and stops the clusters declared in an NVFlow Ray
target, one Slurm allocation each. The Ray target supplies each cluster's
routed roles, `expected_image`, and `gpus_per_node` (none means a CPU cluster);
`start` exports each cluster's `expected_image` as `NVFLOW_IMAGE_ID` on every
node.
A direct Slurm target supplies the account, partitions, the `.sqsh` image for
each role, the mounts, and the `env_vars`. As in a Slurm submission,
`NAME=value` entries are set directly and bare `NAME` entries come from the
shell that runs `start`; the Ray processes, and therefore every Job, inherit
them.

## Prerequisites

- A Slurm login node with pyxis/enroot and an NVFlow checkout after
  `uv sync --extra ray`, for `clusters.py`. Run `nvflow` there, or from a
  workstation whose Ray target has your Slurm target's `ssh_tunnel` block.
- Staged `.sqsh` images under the Slurm target's `containers` for every routed
  role.
- The login node and compute nodes reach compute-node ports from 20000 up,
  where each cluster picks free ports for Ray when it starts, and the Skills
  cluster reaches the vLLM service address.
- An isolated test network: these HTTP endpoints have no authentication, so
  anyone who reaches them can run code in the clusters. Variables whose names
  end in `TOKEN`, `API_KEY`, `SECRET`, `PASSWORD`, or `CREDENTIAL` are withheld
  unless `start` is given `--pass-credentials`; pass them only on a network
  limited to people who may read them.

## Run

1. Create the Ray target as in
   [Create the target](../../docs/ray/README.md#3-create-the-target). Keep each
   `jobs_url` as `${oc.env:NAME}`, drop the credential fields, and set
   `allow_insecure: true`.

2. Start clusters. Name the ones you want, or omit names to start all:

   ```bash
   export NVFLOW_INSECURE_TEST_ENDPOINT_ACK=I_UNDERSTAND_RAY_JOBS_HAS_NO_AUTH
   uv run python scripts/ray-test/clusters.py start cluster_configs/my_ray.yaml \
     skills serving --slurm-target cluster_configs/my_cluster.yaml
   ```

   `start` waits until each cluster accepts a test job and answers from this
   node, printing each cluster's Slurm state and pending reason as it changes,
   then prints the path of `endpoints.env`. It gives up after `--timeout`
   seconds (default 7200), including time queued in Slurm. GPU clusters take whole nodes; CPU clusters take
   `--cpus` CPUs (default 16) and `--mem` memory (default 64G) on shared
   nodes; raise them for heavier CPU steps. Other options: `--nodes`,
   `--time`, `--timeout`, and `--pass-credentials` for workflows that need
   API keys or tokens.

3. Check readiness at any time:

   ```bash
   uv run python scripts/ray-test/clusters.py check cluster_configs/my_ray.yaml
   ```

4. Load the endpoints that the target's `jobs_url` variables read, where you
   run `nvflow`:

   ```bash
   source <artifact_root>/test-infra/<user>/my_ray/endpoints.env
   # From a workstation:
   source <(ssh <user>@<login-node> cat <artifact_root>/test-infra/<user>/my_ray/endpoints.env)
   ```

5. [Run workflows](../../docs/ray/README.md#run-workflows) with
   `--cluster my_ray`.

6. When the controller and its child Jobs are terminal, stop the clusters:

   ```bash
   uv run python scripts/ray-test/clusters.py stop cluster_configs/my_ray.yaml
   ```

Endpoint records, Slurm logs, and `endpoints.env` live in
`<artifact_root>/test-infra/$USER/<target>/`. To add a role such as `nemo-rl`,
declare its cluster and route in the Ray target and start it by name.
