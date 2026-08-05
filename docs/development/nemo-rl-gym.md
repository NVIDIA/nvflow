# NeMo-RL / NeMo-Gym: trainer image & Gym venvs (advanced)

> Audience: **advanced / dev**. For a normal GRPO run you do **not** need this page — follow INSTALL.md and the quick-start. This page explains how GRPO `training` gets NeMo-RL and NeMo-Gym, and how to iterate on Gym source. (SFT `training` runs on the same image but never touches Gym.)

## How the trainer gets NeMo-RL and Gym

GRPO `training` runs on `nvflow-nemo-rl`, built from [`dockerfiles/Dockerfile.nemo-rl`](../../dockerfiles/Dockerfile.nemo-rl). The NeMo-RL base supplies Transformer Engine and the prebuilt NeMo-RL / Ray venvs but leaves the Gym venvs unbuilt, because upstream gates that prefetch behind `NEMO_GYM_PREFETCH_CONFIGS`. Our image closes exactly that gap and changes nothing else:

- The Gym submodule is advanced in place to `GYM_REF` and reinstalled editable into the Gym actor venv.
- One venv is baked **per Gym component** under `/opt/gym_venvs`, by driving `gym env start … +dry_run=true` from that actor venv. Driving it this way is what makes Gym pin each component to the container's own interpreter and Ray version.
- `training.py` sets `env.nemo_gym.skip_venv_if_present = True` (`nvflow/recipes/finance/stages/rl/training.py:157`), so NeMo-RL reuses the baked venvs rather than building.
- The nemo-skills `installation_command` for the trainer is a no-op (`"true"`) — no Gym CLI setup is needed inside the trainer container.

The result is that **nothing resolves at job runtime and no Gym mount is required.**

The Gym-only stages (`prepare_data`, `prefetch_cache`, `collect_rollouts`, `compute_rewards`) run on the CPU-only `nvflow-nemo-gym` image instead, which bakes the Gym CLI (`/opt/gym-cli-venv`) and its own per-component venvs (`/opt/gym-venvs`). They share the `&gym_install_cpu` command in `nvflow/recipes/finance/workflows/grpo/base.yaml`, which only puts the baked CLI on `PATH` — no build, no network. See [`docs/maintainers/containers.md`](../maintainers/containers.md) for both builds.

### Why two venv directories

`nvflow-nemo-gym` bakes to `/opt/gym-venvs` (hyphen; `gym_uv_venv_dir` in the workflow YAML); the trainer bakes to `/opt/gym_venvs` (underscore; `NEMO_GYM_VENV_DIR`, inherited from the base).

Aligning the paths would not make the venvs interchangeable. The images differ in Python (3.12 vs 3.13) and Ray (2.56.1 vs 2.55.1), and both are hard constraints: a Gym server joining the trainer's Ray cluster is version-checked on Ray and on Python down to the patch level, and a venv is bound to its interpreter. Each image bakes where its own runtime looks.

## Dev iteration on Gym source

To work against a modified Gym, bind-mount your clone over the trainer's Gym path:

```yaml
mounts:
  - <host_path>/Gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym
```

This is the one configuration where `uv` resolves at runtime, so it needs `UV_OFFLINE` unset and a reachable pypi mirror. `skip_venv_if_present=True` still applies, so remove the stale venv if you want a rebuild.

**Do not use this mount in production.** It shadows the baked source and venvs and invalidates the container fingerprint, which turns a fully offline run into one that silently rebuilds over the cluster proxy.

## Why `UV_OFFLINE` stays unset

The images need no resolve, so setting it would change nothing in a normal run. It is left unset deliberately, to keep the dev-iteration path above working.

The trade-off is worth stating: because it is unset, a fingerprint miss **rebuilds instead of failing loudly**. So verify airgap behaviour by checking training logs for venv-build activity, not by the job succeeding.

## Regression guard

The `&gym_install_cpu` command and the Gym env-start wiring are covered by `tests/test_grpo_gym_install.py` — run `uv run pytest tests/test_grpo_gym_install.py -v` before changing either.
