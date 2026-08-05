# Multimodal HopChain Recipe

The multimodal recipe implements a HopChain-inspired synthetic data generation
pipeline for multi-hop vision-language reasoning. It follows the paper
[HopChain: Multi-Hop Data Synthesis for Generalizable Vision-Language Reasoning](https://arxiv.org/pdf/2603.17024)
and expresses the workflow as reusable NVFlow stages.

Start with the [HopChain quick start](quick-start.md).

## Workflows

| Workflow | Demo config | Full config |
| --- | --- | --- |
| Image filter | `nvflow/recipes/multimodal/workflows/image_filter/hopchain-image-filter-demo.yaml` | `nvflow/recipes/multimodal/workflows/image_filter/hopchain-image-filter.yaml` |
| SDG | `nvflow/recipes/multimodal/workflows/sdg/hopchain-sdg-demo.yaml` | `nvflow/recipes/multimodal/workflows/sdg/hopchain-sdg.yaml` |

The demo SDG config stops after verified-question visualization. It has no
external API dependency. The full config additionally runs the OpenAI
judge, reconciliation, Omni difficulty filtering, and SFT trace generation.

## Configuration Contract

Configuration is split between workflow and cluster files:

- Each full workflow YAML defines its stages, model profiles, execution
  IDs, chunking, and repository-relative input/output paths.
- Each demo YAML inherits its corresponding full workflow and overrides
  only the stage selection and small-run settings.
- [`cluster_configs/my_cluster.yaml`](../../cluster-configuration.md) defines
  the local Slurm account, partitions, mounts, and named container image paths.
- Optional private recipe changes go in git-ignored `private_*.yaml` overlays
  next to the workflow they modify.
- The full workflow's OpenAI key is supplied as `OPENAI_API_KEY` under `env_vars`
  in `cluster_configs/my_cluster.yaml`.

Run the demo workflows in order:

```bash
uv run nflow run-all \
  --config nvflow/recipes/multimodal/workflows/image_filter/hopchain-image-filter-demo.yaml

uv run nflow run-all \
  --config nvflow/recipes/multimodal/workflows/sdg/hopchain-sdg-demo.yaml
```

Outputs are deterministic:

```text
outputs/hopchain/image_filter/execution/demo/
outputs/hopchain/sdg/execution/demo/
```

Run the full workflows with their full configs:

```bash
uv run nflow run-all \
  --config nvflow/recipes/multimodal/workflows/image_filter/hopchain-image-filter.yaml

uv run nflow run-all \
  --config nvflow/recipes/multimodal/workflows/sdg/hopchain-sdg.yaml
```

Full-workflow outputs use these directories:

```text
outputs/hopchain/image_filter/execution/full/
outputs/hopchain/sdg/execution/full/
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

The image filter recursively scans `data/images/` and writes
`outputs/hopchain/image_filter/execution/demo/image-filter/kept_images.jsonl`.
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
[quick start](quick-start.md#run-the-full-workflow) for full-workflow credential setup
and the data-egress warning.
