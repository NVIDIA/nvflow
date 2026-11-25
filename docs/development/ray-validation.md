# Ray support validation matrix

Ray support is declared only when the supplied-cluster path passes the same
project contracts as direct Slurm. This document records reproducible gates;
it is not a substitute for CI artifacts or cluster run records.

## Automated repository gates

- Direct Slurm and Ray modules satisfy static import boundaries.
- Every runnable bundled stage produces an exact, template-free execution
  contract.
- Every runnable Starter, HopChain, and Finance workflow has a complete,
  acyclic graph with declared outputs and target-routable runtime roles.
- Relocated Starter and custom external-project fixtures package without
  checkout-relative assumptions.
- Python is `>=3.13.14,<3.14`; CI uses `3.13.14`.
- Ray is `2.58.0` in the launcher and supplied-cluster target contract.
- Worker/controller source passes the current Python 3.13 import and syntax
  boundary.
- Runtime dependency installation, remote package URIs, live source mounts,
  unsafe archive paths, and serialized target credentials are rejected, as are
  task scheduler controls (`partition`, `account`, `exclusive`,
  `sbatch_kwargs`, `installation_command`).
- Jobs archives stay below the 95 MiB budget and control variables are shell
  exports, never `runtime_env.env_vars`.
- Fake Jobs API tests cover submission, deterministic adoption, bounded
  transient-request retries, status/logs/cancellation, dependency conditions,
  readiness timeout, service cleanup, resource checks, and interrupted journal
  tail recovery. Controller-process restart remains a supplied-cluster
  acceptance item below.

## Local Ray Jobs canary

The dependency-light canary is:

1. Start Ray `2.58.0` with Python `3.13.14` and `NVFLOW_IMAGE_ID`.
2. Probe the Jobs API.
3. Submit the runtime identity probe and verify shared-artifact visibility.
4. Upload a local ZIP and submit a deterministic hello-world Job.
5. Require terminal `SUCCEEDED`, retrieve logs, and stop the local cluster.

This validates the API client and runtime profile, not multi-node GPU
placement.

The implementation was exercised locally with Python `3.13.14` and Ray
`2.58.0`: runtime/image/artifact probing passed and a deterministic
hello-world Job reached `SUCCEEDED`.

## Supplied-cluster acceptance

Run on release-qualified supplied clusters:

1. Controller on one cluster submits/reconciles children on itself and a second
   cluster after launcher disconnect.
2. Failure and cancellation stop only run-owned child Jobs, never clusters.
3. Standalone Gym rollout/re-verification attaches to the expected GCS and
   starts no local fallback cluster.
4. NeMo-RL SFT and GRPO attach to the expected training GCS; embedded Gym
   reports that same GCS.
5. Execute every bundled project quick-start through direct Slurm and supplied
   Ray using unchanged workflow YAML.
6. Compare declared artifacts, completion evidence, checkpoints, and
   evaluation outputs rather than scheduler IDs or timing.

The environment-specific acceptance run must persist its target fingerprint,
execution digest, bundle digest, image IDs, Jobs handles, and artifact paths.

## Explicitly separate claims

- Passing a manually created Ray-on-Slurm cluster proves supplied Ray behavior;
  it does not make dynamic Ray-on-Slurm a product feature.
- A KubeRay-deployed supplied cluster should satisfy the same Jobs contract,
  but NVFlow may claim “Kubernetes validated” only after a real KubeRay smoke
  run.
- Unit and local canaries do not substitute for Finance SFT/Gym/GRPO GPU
  acceptance.
