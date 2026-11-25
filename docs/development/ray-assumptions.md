# Ray execution assumptions

This document records the claims that are allowed to shape NVFlow's Ray Jobs
engine.

## Accepted facts

- The release runtime profile follows NeMo-RL 0.8.0 and NeMo-Gym 0.7.0:
  Python `>=3.13.14,<3.14` with reference interpreter `3.13.14`, and Ray
  `2.58.0` exactly on the launcher client and every supplied cluster.
- Ray and Slurm are different layers. `executor: slurm` directly submits the
  existing Slurm workload. `executor: ray` submits to operator-supplied Ray Jobs
  endpoints and contains no Slurm provisioning behavior.
- NVFlow uses the Ray Jobs HTTP API. Ray Client is not part of the execution
  contract.
- A supplied Ray cluster has a fixed runtime environment for its Jobs
  entrypoint. Routing therefore uses explicit runtime roles and probed
  capabilities, never recursive inspection of workflow YAML.
- NeMo-RL calls `ray.init(address="auto")` before creating its actors and reuses
  an externally managed cluster. A Ray Jobs execution must prove that it
  attached to the expected GCS and must fail if it starts a second local
  cluster.
- Embedded NeMo-Gym receives NeMo-RL's current GCS address. It does not need a
  separately routed Gym cluster.
- Standalone Gym server and rollout commands call Gym's Ray initializer. Under
  the Ray engine they must attach to their routed supplied cluster. Pure
  collation, profiling, cache, and transform commands do not become
  Ray-native applications merely because Gym is installed.
- Cross-cluster stages exchange durable artifacts through a POSIX artifact root
  mounted at the same path. Object-store transfer is not supported in this
  release.
- The controller runs as a durable Ray Job on a designated supplied control
  cluster. It must be able to reach every routed Jobs endpoint and declared
  service endpoint after the launching client exits.

## Rejected assumptions

- NVFlow does not require one cluster per NeMo product name. Multiple runtime
  roles may route to one cluster when actual capabilities are compatible.
- Positive `num_gpus`, nested GPU fields, or a container alias are not reliable
  cluster-routing heuristics.
- Ray dependencies do not replace workflow dependencies. `after_success`,
  `after_start`, `after_ready`, and `after_any` remain explicit graph edges.
- Missing or unhealthy endpoints never fall back to another role or to direct
  Slurm execution.
- A successful submission is not workflow completion. Terminal Jobs state and
  declared artifact evidence are separate.

## Product scope

- Supported: direct Slurm and customer-supplied Ray Jobs clusters, including
  clusters deployed by KubeRay.
- Test infrastructure only: manually or script-created Ray-on-Slurm clusters.
- Out of scope: Ray cluster creation, resizing, autoscaling, leases, dynamic
  Ray-on-Slurm, dynamic KubeRay, Ray Client, and object-store artifact transfer.

## Required live canaries

Before production support is declared:

1. A controller Job submits, observes, and cancels deterministic child Jobs on
   its own endpoint and a second supplied endpoint after launcher disconnect.
2. A NeMo-RL SFT and GRPO driver attaches to the expected supplied GCS, places
   actors on the expected workers, and starts no second head or raylet.
3. Embedded Gym reports the same GCS as NeMo-RL.
4. Standalone Gym rollout/re-verification attaches to its expected supplied
   GCS, cleans its child services, and starts no local fallback cluster.
5. The same canaries run with no Slurm executable, configuration, or environment
   variable available to the Ray engine.
