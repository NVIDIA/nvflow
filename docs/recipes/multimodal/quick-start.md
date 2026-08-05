# HopChain Quick Start

Run HopChain from a folder of images to verified multi-hop vision-language
questions.

The demo has two commands:

1. Filter the source images with Qwen.
2. Generate and verify multi-hop questions with Qwen and SAM 3.1.

Plan for 30–60 minutes for a small demo run, plus Slurm queue time. Image
filtering typically takes 5–10 minutes, and the SDG dependency chain takes 20
minutes or more. Runtime increases with the number of images and generated
queries.

The demo ends after verified-question visualization. The full workflow
adds the OpenAI judge and Omni curation stages.

## Pipeline Overview

```text
  ┌──────────────────┐     ┌──────────────────┐     ┌───────────────────────────┐
  │ 1. Your images   │────▶│ 2. Image Filter  │────▶│ 3. SDG                    │
  │ (a folder)       │     │ keep complex     │     │ categories → localize →   │
  └──────────────────┘     └────────┬─────────┘     │ combine → generate &      │
                                    │               │ verify multi-hop queries  │
                            kept_images.jsonl       └─────────────┬─────────────┘
                                                  ┌───────────────┴───────────────┐
                                                  ▼                               ▼
                                      ┌───────────────────────┐   ┌───────────────────────┐
                                      │ 4. Judge + reconcile  │   │ 5. Difficulty filter  │
                                      │ (OpenAI API)          │   │ + SFT reasoning traces│
                                      └───────────────────────┘   └───────────────────────┘
```

The full path continues from verified questions through the external
judge, judge reconciliation, difficulty filtering, and SFT reasoning-trace
generation. Image filtering is a separate workflow so its output can be reused
by several SDG runs.

## Prerequisites

- Cluster access configured as described in [INSTALL.md](../../../INSTALL.md),
  with all commands run from the `nvflow` repository root.
- A Slurm cluster config created from `cluster_configs/template-slurm.yaml`; see
  [Step 1](#1-configure-models-and-cluster).
- GPUs for the model workers. The **core** path (Steps 3–4) needs:
  - A **VLM server** for image scoring and question generation. The reference
    config serves `Qwen/Qwen3.5-397B-A17B` with SGLang
  - A **SAM 3.1 worker** for object localization.
- The **full** path additionally needs an OpenAI API key for the LLM judge and
  the Omni reasoning VLM for difficulty filtering.

> **Heads up:** The reference models are large. For a quick try, use
> [smaller models you can serve](#local-overrides).

## 1. Configure Models and Cluster

### SAM 3.1 checkpoint

Request access to
[Meta's gated SAM 3.1 repository](https://huggingface.co/facebook/sam3.1), then
download the checkpoint once from a connected host:

```bash
uv run hf auth login
uv run hf download facebook/sam3.1 sam3.1_multiplex.pt \
  --local-dir /path/to/models/hf_models/facebook/sam3.1
```

After the checkpoint is downloaded, the compute jobs do not need `HF_TOKEN`.

### Cluster configuration

Follow [Configure Your Cluster](../../../INSTALL.md#configure-your-cluster) to
create `cluster_configs/my_cluster.yaml`. The
[Cluster Configuration Guide](../../cluster-configuration.md) documents every
available field.

In `my_cluster.yaml`, configure the named `nemo-skills`, `sglang`, and `vllm`
container entries. Mount the checkout and your host model directory so they
are visible on every compute node. The reference configs expect these paths
inside the containers:

```text
/hf_models/Qwen/Qwen3.5-397B-A17B
/hf_models/facebook/sam3.1/sam3.1_multiplex.pt
/hf_models/nvidia/omni-step70                 # full workflow only
```

The workflow resolves the repository root from the shell's standard `PWD`.
Make the checkout visible to Slurm jobs at the same absolute path. On sites
that mount a workspace at `/workspace`, launch NVFlow from the checkout under
that mount, such as `/workspace/nvflow`.

## 2. Add Images

Copy or mount images anywhere below:

```bash
mkdir -p data/images
# Copy or mount images below data/images/.
```

Subdirectories are scanned recursively. The demo selects at most 100 images and
the SDG step uses at most 25 images that pass filtering. Prefer visually rich
scenes, documents, charts, or infographics with several distinct regions.

Public datasets that fit the recipe well include:

| Dataset | Why it fits HopChain | Source |
| --- | --- | --- |
| COCO 2017 validation | Everyday multi-object scenes; a practical first run | <http://images.cocodataset.org/zips/val2017.zip> |
| Visual Genome | Dense objects and relationships | <https://homes.cs.washington.edu/~ranjay/visualgenome/> |
| InfographicVQA, DocVQA, or ChartQA | Text- and figure-rich images for OCR reasoning | Hugging Face Datasets |
| ADE20K | Complex scene-parsing images | <https://groups.csail.mit.edu/vision/datasets/ADE20K/> |
| Open Images V7 | Large and diverse multi-object collection | <https://storage.googleapis.com/openimages/web/index.html> |

Images remain path-referenced throughout the pipeline, so keep the directory
mounted and unchanged until the run completes.

## 3. Filter Images (~5–10 minutes)

Validate, preview, and submit the demo:

```bash
uv run nflow validate \
  --config nvflow/recipes/multimodal/workflows/image_filter/hopchain-image-filter-demo.yaml

uv run nflow list-stages \
  --config nvflow/recipes/multimodal/workflows/image_filter/hopchain-image-filter-demo.yaml

uv run nflow run-all \
  --config nvflow/recipes/multimodal/workflows/image_filter/hopchain-image-filter-demo.yaml
```

`run-all` submits Slurm work and returns. After the job finishes, inspect the
deterministic demo output:

```bash
python -m json.tool outputs/hopchain/image_filter/execution/demo/image-filter/summary.json
wc -l outputs/hopchain/image_filter/execution/demo/image-filter/kept_images.jsonl
```

The second command must report at least one kept image before SDG can proceed.

The image-filter output contains:

```text
image-filter/
├── image_catalog.jsonl
├── output.jsonl
├── final_output.jsonl
├── kept_images.jsonl
└── summary.json
```

`final_output.jsonl` includes every scored image; `kept_images.jsonl` contains
only images that passed the configured quality and complexity thresholds.

## 4. Generate Multi-Hop Questions (~20+ minutes)

The SDG demo reads
`outputs/hopchain/image_filter/execution/demo/image-filter/kept_images.jsonl`.

```bash
uv run nflow validate \
  --config nvflow/recipes/multimodal/workflows/sdg/hopchain-sdg-demo.yaml

uv run nflow list-stages \
  --config nvflow/recipes/multimodal/workflows/sdg/hopchain-sdg-demo.yaml

uv run nflow run-all \
  --config nvflow/recipes/multimodal/workflows/sdg/hopchain-sdg-demo.yaml
```

The demo runs the local core path:

```text
prepare images -> identify categories -> localize with SAM -> sample object
combinations -> generate questions -> verify questions -> build visualization
```

After the dependency chain completes:

```bash
python -m json.tool \
  outputs/hopchain/sdg/execution/demo/step-5-verify-candidate-queries/summary.json

wc -l \
  outputs/hopchain/sdg/execution/demo/step-5-verify-candidate-queries/final_candidates.jsonl
```

Review the generated HTML under
`outputs/hopchain/sdg/execution/demo/step-6-visualize-candidate-hopchain-data/`.

The core output layout is:

```text
sdg/execution/demo/
├── step-0-prepare-filtered-inputs/filtered_image_inputs.jsonl
├── step-1-identify-categories/final_output.jsonl
├── step-2-localize-instances/
├── step-3-sample-instance-combinations/instance_combinations.jsonl
├── step-4-generate-multihop-queries/final_output.jsonl
├── step-5-verify-candidate-queries/
│   ├── final_candidates.jsonl
│   ├── rejected_candidates.jsonl
│   └── summary.json
└── step-6-visualize-candidate-hopchain-data/
```

## Run the Full Workflow

The full configs use the `full` execution ID. They process the complete input
set, use full-run chunk counts, call the
OpenAI judge, run the Omni
difficulty filter, and create SFT reasoning traces.

Before running the full workflow, add your OpenAI key to
`cluster_configs/my_cluster.yaml`, following the existing
[environment-variable instructions](../../cluster-configuration.md#environment-variables):

```yaml
env_vars:
  # ...existing cluster environment variables...
  - OPENAI_API_KEY=<your-key>
```

The OpenAI judge sends question and image content to an external service. Only
enable the full path when that data transfer is allowed.

Then run:

```bash
uv run nflow run-all \
  --config nvflow/recipes/multimodal/workflows/image_filter/hopchain-image-filter.yaml

uv run nflow run-all \
  --config nvflow/recipes/multimodal/workflows/sdg/hopchain-sdg.yaml
```

Full-workflow outputs live under:

```text
outputs/hopchain/image_filter/execution/full/
outputs/hopchain/sdg/execution/full/
```

The full-workflow stage groups are:

| Steps | Work | Needs |
| --- | --- | --- |
| 0–6 | Prepare, identify, localize, combine, generate, verify, visualize | Qwen and SAM |
| 7–9 | OpenAI judge, reconcile, and visualize reconciled data | `OPENAI_API_KEY` |
| 10 | Filter easy candidates | Omni reasoning VLM |
| 11–12 | Generate and filter SFT reasoning traces | Qwen |

Adjust full-run chunk counts after checking the image-filter and combination
counts for your dataset.

## Local Overrides

Put deployment-specific recipe changes in a small `private_*.yaml` overlay next
to the workflow it modifies (`private_*.yaml` files are git-ignored repo-wide).
For example:

```yaml
# nvflow/recipes/multimodal/workflows/image_filter/private_hopchain-image-filter.yaml
_base_: hopchain-image-filter-demo.yaml

execution_id: my_test
model_profiles:
  qwen:
    server_gpus: 4
    server_nodes: 1
    server_args: >-
      --model-path /hf_models/Qwen/Qwen3.5-397B-A17B
      --served-model-name qwen3.5-397b-a17b
      --tp 4
      --trust-remote-code
```

Use another small overlay based on `hopchain-sdg-demo.yaml` (in
`workflows/sdg/`) when the SDG model profile also needs to change. Keep host
paths, Slurm partitions, mounts, and container image paths in
`cluster_configs/my_cluster.yaml`.

## Next Steps

- Review `final_candidates.jsonl` and the candidate HTML before enabling the
  external judge.
- Tune `min_complexity_score` or `allowed_quality_ratings` in a local
  image-filter overlay when the kept set is too broad or too small.
- Use a local SDG overlay to calibrate `sample_count`, query count, and chunk
  counts before a full run.
- Read the [multimodal HopChain guide](README.md) for the complete stage list
  and configuration contract.

## Troubleshooting

### The config validates, but the job cannot see files

`validate` runs in the launch shell; the stage itself runs in a container on a
compute node. Confirm that the checkout, images, outputs, and checkpoint paths
are covered by `my_cluster.yaml` mounts and appear at the paths documented
above.

### No images were selected

Confirm `data/images/` contains supported image files. If filtering ran
but kept zero images, inspect `final_output.jsonl` and lower
`min_complexity_score` in a local image-filter overlay.

### A job requests the wrong partition or container

Partitions and container image paths come from `cluster_configs/my_cluster.yaml`.
Check `partition`, `cpu_partition`, and the named container entries there.

### The full workflow fails at the judge stage

Confirm `OPENAI_API_KEY` is present under `env_vars` in the ignored
`cluster_configs/my_cluster.yaml`. The cluster config injects it into the
`nemo-skills` container used by the full-workflow judge.

[Multimodal HopChain Guide](README.md) | [Main README](../../../README.md)
