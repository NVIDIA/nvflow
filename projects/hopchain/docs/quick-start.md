# HopChain Quick Start

Run HopChain from a folder of images to verified multi-hop vision-language
questions.

The demo has two commands:

1. Filter the source images with Qwen.
2. Generate and verify multi-hop questions with Qwen and SAM 3.1.

Plan for 30–60 minutes for a small demo run, plus queue time. Image
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

The `hopchain:quickstart` recipe records the image-filter → SDG dependency:

```bash
uv run nvflow list-workflows --recipe hopchain:quickstart
uv run nvflow validate --recipe hopchain:quickstart
uv run nvflow plan-recipe hopchain:quickstart
```

These commands are target-independent. The graph is an inspection and
validation contract; submit its two workflow selectors individually in the
steps below.

## Prerequisites

- Shared setup from [INSTALL.md](../../../INSTALL.md),
  with commands run from the `nvflow` repository root (or `/opt/nvflow` in the
  optional launcher image).
- A target from [Slurm execution](../../../docs/slurm/README.md) or
  [Ray execution](../../../docs/ray/README.md#set-up-a-ray-target); see
  [Step 1](#1-configure-models-and-cluster). On Ray, pass your Ray target
  instead of `my_cluster`.
- For the **full** path, a key for the LLM judge, which defaults to the NVIDIA
  inference hub (`NVIDIA_API_KEY`;
  [switch to OpenAI](../../../docs/hosted-endpoints.md#hopchain-judge)).

### What each step needs

| Step | Models under `/hf_models` | Images (roles) | Largest job | Also needs |
|---|---|---|---|---|
| 3. Filter images | Qwen3.5-397B-A17B | `nemo-skills`, `sglang` | 16 GPUs on 2 nodes for the Qwen VLM | Nothing else |
| 4. Generate questions | Qwen3.5-397B-A17B, SAM 3.1 | `nemo-skills`, `sglang`, `vllm` | 16 GPUs on 2 nodes; SAM 3.1 takes 1 GPU | Nothing else |
| Full workflow | Also Omni (`nvidia/omni-step70`) | `nemo-skills`, `sglang`, `vllm` | 16 GPUs on 2 nodes; Omni takes 4 GPUs | Network to the hosted judge, and its key |

SAM 3.1 and Omni run in the `vllm` image. On Ray, route every listed role to a
cluster with that capacity, and keep the control cluster available for every
step.

> **Heads up:** The reference models are large. For a quick try, serve a
> smaller Qwen vision-language model through a [local override](#local-overrides).

## 1. Configure Models and Cluster

### SAM 3.1 checkpoint

Request access to
[Meta's gated SAM 3.1 repository](https://huggingface.co/facebook/sam3.1), then
download the checkpoint once from a connected host:

```bash
uvx --from huggingface_hub hf auth login
uvx --from huggingface_hub hf download facebook/sam3.1 sam3.1_multiplex.pt \
  --local-dir /path/to/models/hf_models/facebook/sam3.1
```

After the checkpoint is downloaded, the compute jobs do not need `HF_TOKEN`.

### Cluster configuration

On Slurm, follow [Create the target](../../../docs/slurm/README.md#3-create-the-target) to
create `cluster_configs/my_cluster.yaml`. The
[Cluster Configuration Guide](../../../docs/operations/cluster-configuration.md) documents every
available field.

In `my_cluster.yaml`, configure the named `nemo-skills`, `sglang`, and `vllm`
container entries. Mount a dedicated writable data directory at `/workspace`
and your host model directory at `/hf_models` so they are visible on every
compute node. For example:

```yaml
mounts:
  - /path/to/hopchain-data:/workspace
  - /path/to/models/hf_models:/hf_models
```

On Ray, route `nemo-skills`, `sglang`, and `vllm` to clusters that mount the
same two paths. On both backends the repository is not mounted: NVFlow packages
its code and checked-in assets with each job.

The reference configs expect these paths inside the containers:

```text
/hf_models/Qwen/Qwen3.5-397B-A17B
/hf_models/facebook/sam3.1/sam3.1_multiplex.pt
/hf_models/nvidia/omni-step70                 # full workflow only
```

The launcher still reads `my_cluster.yaml` locally. When using `nvflow-client`,
set `NVFLOW_CONFIG_DIR` to the directory containing that file, as described
in the [remote-launch guide](../../../docs/operations/remote-launch.md).

`validate`, `show-config`, and list commands are target-independent. Every
`plan` and `run` command must explicitly select a target; the
examples below use `--cluster my_cluster`.

Static planning checks the selected target YAML locally without inspecting
workflow filesystem state. `plan --live` adds input/resume discovery and
presence checks for explicitly declared outputs, and submits nothing: on Slurm
it reads storage through the target, and on Ray it reads the launcher's local
filesystem. On Slurm, every `run` performs that live-equivalent planning before
submission; on Ray, `run` plans statically and the workload scripts skip
completed work. Progress, warnings, and errors use stderr, while execution
commands leave stdout empty.

## 2. Add Images

Set `HOPCHAIN_DATA` to the host source of the `/workspace` mount, then copy or
mount images below its `data/images` directory:

```bash
HOPCHAIN_DATA=/path/to/hopchain-data
mkdir -p "$HOPCHAIN_DATA/data/images"
# Copy or mount images below "$HOPCHAIN_DATA/data/images".
```

Subdirectories are scanned recursively. The demo selects at most 100 images and
the SDG step uses at most 25 images that pass filtering. Prefer visually rich
scenes, documents, charts, or infographics with several distinct regions.

Public datasets that fit the project well include:

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
uv run nvflow validate \
  --workflow hopchain:image_filter/demo

uv run nvflow list-stages \
  --workflow hopchain:image_filter/demo

uv run nvflow run --all \
  --workflow hopchain:image_filter/demo \
  --cluster my_cluster
```

`run --all` submits the work and returns. After the job finishes, inspect the
deterministic demo output:

```bash
python -m json.tool \
  "$HOPCHAIN_DATA/outputs/hopchain/image_filter/execution/demo/image-filter/summary.json"
wc -l \
  "$HOPCHAIN_DATA/outputs/hopchain/image_filter/execution/demo/image-filter/kept_images.jsonl"
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
`/workspace/outputs/hopchain/image_filter/execution/demo/image-filter/kept_images.jsonl`
inside the worker.

```bash
uv run nvflow validate \
  --workflow hopchain:sdg/demo

uv run nvflow list-stages \
  --workflow hopchain:sdg/demo

uv run nvflow run --all \
  --workflow hopchain:sdg/demo \
  --cluster my_cluster
```

The demo runs the local core path:

```text
prepare images -> identify categories -> localize with SAM -> sample object
combinations -> generate questions -> verify questions -> build visualization
```

After the dependency chain completes:

```bash
python -m json.tool \
  "$HOPCHAIN_DATA/outputs/hopchain/sdg/execution/demo/step-5-verify-candidate-queries/summary.json"

wc -l \
  "$HOPCHAIN_DATA/outputs/hopchain/sdg/execution/demo/step-5-verify-candidate-queries/final_candidates.jsonl"
```

Review the generated HTML under
`$HOPCHAIN_DATA/outputs/hopchain/sdg/execution/demo/step-6-visualize-candidate-hopchain-data/`.

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
set, use full-run chunk counts, call the hosted LLM judge, run the Omni
difficulty filter, and create SFT reasoning traces.

Before running the full workflow on Slurm, forward your judge API key through
`cluster_configs/my_cluster.yaml`, following the existing
[environment-variable instructions](../../../docs/operations/cluster-configuration.md#environment-variables):

```yaml
env_vars:
  # ...existing cluster environment variables...
  - NVIDIA_API_KEY
```

On Ray, provision the key in the clusters' environment instead. See
[Hosted Model Endpoints](../../../docs/hosted-endpoints.md) for both backends,
and to use OpenAI or another provider.

The judge sends question and image content to an external service. Only enable
the full path when that data transfer is allowed.

Then run:

```bash
uv run nvflow run --all \
  --workflow hopchain:image_filter/full \
  --cluster my_cluster

uv run nvflow run --all \
  --workflow hopchain:sdg/full \
  --cluster my_cluster
```

Full-workflow outputs live under:

```text
$HOPCHAIN_DATA/outputs/hopchain/image_filter/execution/full/
$HOPCHAIN_DATA/outputs/hopchain/sdg/execution/full/
```

The full-workflow stage groups are:

| Steps | Work | Needs |
| --- | --- | --- |
| 0–6 | Prepare, identify, localize, combine, generate, verify, visualize | Qwen and SAM |
| 7–9 | Hosted LLM judge, reconcile, and visualize reconciled data | `NVIDIA_API_KEY` |
| 10 | Filter easy candidates | Omni reasoning VLM |
| 11–12 | Generate and filter SFT reasoning traces | Qwen |

Adjust full-run chunk counts after checking the image-filter and combination
counts for your dataset.

## Local Overrides

Put deployment-specific project changes in a small `private_*.yaml` overlay next
to the workflow it modifies (`private_*.yaml` files are git-ignored repo-wide).
For example:

```yaml
# projects/hopchain/workflows/image_filter/private_demo.yaml
_base_: demo.yaml

execution_id: my_test
model_profiles:
  qwen:
    model: <served-model-name>
    server_gpus: 4
    server_nodes: 1
    server_args: >-
      --model-path /hf_models/<org>/<smaller-vision-language-model>
      --served-model-name <served-model-name>
      --context-length 131072
      --tp 4
      --trust-remote-code
      --reasoning-parser qwen3
```

This serves a smaller Qwen vision-language model of your choice on one node
with 4 GPUs; download it under your `/hf_models` mount first. `server_args`
replaces the reference string, so keep `--reasoning-parser qwen3` and
`--context-length`, and keep `model` equal to `--served-model-name`, since
requests address the model by that name.

Use another small overlay based on `workflows/sdg/demo.yaml` when the SDG
model profile also needs to change. Keep host paths, partitions, mounts, and
images out of overlays; see where settings live on
[Slurm](../../../docs/slurm/README.md#where-settings-live) and
[Ray](../../../docs/ray/README.md#where-settings-live).

## Next Steps

- Review `final_candidates.jsonl` and the candidate HTML before enabling the
  external judge.
- Tune `min_complexity_score` or `allowed_quality_ratings` in a local
  image-filter overlay when the kept set is too broad or too small.
- Use a local SDG overlay to calibrate `sample_count`, query count, and chunk
  counts before a full run.
- Read the [HopChain guide](../README.md) for the complete stage list
  and configuration contract.

## Troubleshooting

### The config validates, but the job cannot see files

`validate` runs in the launch shell; the stage itself runs in a container on a
worker. NVFlow packages the code with each job; do not mount the checkout.
Confirm that the writable data directory, images, outputs, and checkpoint paths
are mounted (by `my_cluster.yaml` on Slurm, by the clusters on Ray) and appear
at the container paths documented above.

### No images were selected

Confirm `$HOPCHAIN_DATA/data/images/` contains supported image files. If
filtering ran but kept zero images, inspect `final_output.jsonl` and lower
`min_complexity_score` in a local image-filter overlay.

### A job requests the wrong partition or container

On Slurm, partitions and container image paths come from
`cluster_configs/my_cluster.yaml`. Check `partition`, `cpu_partition`, and the
named container entries there. On Ray, each role runs in the image of the
cluster its route names.

### The full workflow fails at the judge stage

Confirm the key named by `api_key_name` in the judge stage config reaches the
`nemo-skills` container used by the full-workflow judge: on Slurm, it must be
under `env_vars` in the ignored `cluster_configs/my_cluster.yaml`; on Ray, in
the environment of the cluster routed to `nemo-skills`. [Hosted Model Endpoints](../../../docs/hosted-endpoints.md) lists the failure
modes for a wrong key, base URL or model id.

[HopChain Guide](../README.md) | [Main README](../../../README.md)
