# HopChain Project

The HopChain project implements a HopChain-inspired synthetic data generation
pipeline for multi-hop vision-language reasoning. It follows the paper
[HopChain: Multi-Hop Data Synthesis for Generalizable Vision-Language Reasoning](https://arxiv.org/pdf/2603.17024)
and expresses the workflow as reusable NVFlow stages.

This project owns its workflow behavior, models, prompts, stage code, outputs,
tests, and quality expectations. NVFlow supplies the project, recipe, and execution
contracts.

Start with the [HopChain quick start](docs/quick-start.md).
When this project lives externally, `uv sync` installs its lightweight `nvflow`
bootstrap. The first semantic command may pause while preparing the full
isolated HopChain environment in the NVFlow cache; later commands reuse it.
In the NVFlow checkout, use the root launcher instead of creating a second
project bootstrap.
`pyproject.toml` keeps only Pydantic in the launcher layer; its `worker` extra
records the packages expected from compatible HopChain worker images.

## Recipes

The `hopchain:quickstart` and `hopchain:vision-reasoning` recipes connect image
filtering to the matching HopChain SDG workflow. Their node keys are the exact
project-relative workflow selectors shown below:

```bash
uv run nvflow list-recipes --project hopchain
uv run nvflow list-workflows --recipe hopchain:quickstart
uv run nvflow validate --recipe hopchain:quickstart
uv run nvflow plan-recipe hopchain:quickstart
```

These commands inspect the graph without a cluster. Recipe DAG submission is
not implemented; run the referenced workflows individually as shown below.

## Workflows

| Workflow | Demo config | Full config |
| --- | --- | --- |
| Image filter | `projects/hopchain/workflows/image_filter/demo.yaml` | `projects/hopchain/workflows/image_filter/full.yaml` |
| SDG | `projects/hopchain/workflows/sdg/demo.yaml` | `projects/hopchain/workflows/sdg/full.yaml` |

The demo SDG config stops after verified-question visualization. It has no
external API dependency. The full config additionally runs the OpenAI
judge, reconciliation, Omni difficulty filtering, and SFT trace generation.

## Configuration Contract

Configuration is split between workflow and cluster files:

- Each full workflow YAML defines its stages, model profiles, execution
  IDs, chunking, packaged-code paths, and worker data paths.
- Each demo YAML inherits its corresponding full workflow and overrides
  only the stage selection and small-run settings.
- The target defines where jobs run. On Slurm,
  [`cluster_configs/my_cluster.yaml`](../../docs/operations/cluster-configuration.md)
  holds the account, partitions, mounts, and named container image paths; on
  Ray, the [routed clusters](../../docs/ray/README.md#where-settings-live)
  supply the images and mounts.
- Optional private project changes go in git-ignored `private_*.yaml` overlays
  next to the workflow they modify.
- The full workflow's hosted-judge key, `NVIDIA_API_KEY` by default, comes from
  `env_vars` in the Slurm target or from the Ray clusters' environment — see
  [Hosted Model Endpoints](../../docs/hosted-endpoints.md).

NVFlow packages project code and checked-in assets with each job; the source
checkout is not mounted into workers. Mount a dedicated writable data directory
at `/workspace`. On the host, inspect files through the source path of that
mount.

Run the demo workflows in order:

```bash
uv run nvflow validate \
  --workflow hopchain:image_filter/demo

uv run nvflow plan \
  --workflow hopchain:image_filter/demo \
  --cluster my_cluster

uv run nvflow run --all \
  --workflow hopchain:image_filter/demo \
  --cluster my_cluster

uv run nvflow run --all \
  --workflow hopchain:sdg/demo \
  --cluster my_cluster
```

Validation is target-independent. See the
[planning contract](../../docs/platform/cli-and-python-api.md#plan) for static,
`--live`, and execution behavior.

Demo outputs use stable directories:

```text
/workspace/outputs/hopchain/image_filter/execution/demo/
/workspace/outputs/hopchain/sdg/execution/demo/
```

Run the full workflows with their full configs:

```bash
uv run nvflow run --all \
  --workflow hopchain:image_filter/full \
  --cluster my_cluster

uv run nvflow run --all \
  --workflow hopchain:sdg/full \
  --cluster my_cluster
```

Full-workflow outputs use these directories:

```text
/workspace/outputs/hopchain/image_filter/execution/full/
/workspace/outputs/hopchain/sdg/execution/full/
```

## SDG Stages

The full `hopchain_sdg` workflow runs:

1. `prepare_filtered_image_inputs`
2. `preprocess_identify_categories`
3. `identify_categories`
4. `localize_instances`
5. `sample_instance_combinations`
6. `preprocess_generate_multihop_queries`
7. `generate_multihop_queries`
8. `verify_candidate_queries`
9. `visualize_candidate_hopchain_data`
10. `judge_candidate_queries_openai`
11. `reconcile_llm_judges`
12. `visualize_reconciled_hopchain_data`
13. `preprocess_filter_easy_candidates`
14. `filter_easy_candidates`
15. `preprocess_generate_sft_reasoning_traces`
16. `generate_sft_reasoning_traces`
17. `preprocess_filter_sft_reasoning_traces`
18. `filter_sft_reasoning_traces`

## Inputs and Models

The image filter recursively scans `/workspace/data/images/` and writes
`/workspace/outputs/hopchain/image_filter/execution/demo/image-filter/kept_images.jsonl`.
The SDG demo reads that file as its input.

By default the containers must see checkpoints at:

```text
/hf_models/Qwen/Qwen3.5-397B-A17B
/hf_models/facebook/sam3.1/sam3.1_multiplex.pt
/hf_models/nvidia/omni-step70
```

Set host-to-container mappings in `my_cluster.yaml` and server behavior in an
ignored local YAML overlay. Keep machine-specific paths in those local files.

Do not commit API keys or credential files. See the
[quick start](docs/quick-start.md#run-the-full-workflow) for full-workflow credential setup
and the data-egress warning.
