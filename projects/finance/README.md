# Finance project

The finance project is a self-contained NVFlow application for financial
reasoning data generation, supervised fine-tuning, reinforcement learning, and
evaluation. It owns its workflow YAML, prompts, assets, Python stages,
dependencies, tests, and operational guidance.

NVFlow provides the project and recipe contracts, workflow loading, planning, packaging, and
execution integration. The finance team owns the behavior and quality of this
project.

## Prerequisites

- NVFlow installed from the repository root.
- A configured Slurm or Ray target with the required worker images and a
  writable `/workspace/outputs`.
- Models required by the workflow mounted under `/hf_models`.
- API credentials only for workflows that explicitly use hosted services.
- SEC identity values updated before downloading EDGAR filings.

See [`INSTALL.md`](../../INSTALL.md) for shared setup, then
[Slurm execution](../../docs/slurm/README.md) or
[Ray execution](../../docs/ray/README.md) for your target.
When this project lives externally, `uv sync` installs its lightweight `nvflow`
bootstrap. The first semantic command may pause while preparing the full
isolated finance environment in the NVFlow cache; later commands reuse it. In
the NVFlow checkout, use the root launcher instead of creating a second project
bootstrap.
`pyproject.toml` keeps launcher dependencies minimal; its `worker` extra records
finance utility dependencies that belong in compatible worker images.

## Recipes

`project.yaml` defines demo and production scenarios with validated data-path
connections. Three of them are documented recipes:

- `finance:quickstart` uses template-based demo data for Qwen3-4B SFT,
  single-turn GRPO, and multi-turn GRPO.
- `finance:template-based-sft` uses template-based S&P 500 data for
  single-turn Qwen3-14B and Nemotron-3-Nano SFT.
- `finance:document-grounded-local-search-grpo` uses document-grounded S&P 500
  data for multi-turn Nemotron-3-Nano GRPO with `finance_sec_search`, answering
  from the local full-text index built by `download/sp500` rather than sec.gov.

`project.yaml` also defines the `finance:vals-no-web` and `finance:vals11`
recipes. Inspect any recipe without a cluster:

```bash
uv run nvflow list-recipes --project finance
uv run nvflow list-workflows --recipe finance:quickstart
uv run nvflow validate --recipe finance:quickstart
uv run nvflow plan-recipe finance:quickstart
```

Recipe commands validate and display the cross-workflow DAG; they do not submit
it. Execute one referenced workflow at a time with `run`.

## Workflows

These guides follow a common research journey, while each workflow definition
remains independently selectable. Workflow discovery is alphabetical; recipe
dependencies describe the curated scenario order. Choose only the workflows
required by your use case.

| Guide | Workflow | Purpose |
|---:|---|---|
| 1 | [Download SEC](docs/workflows/01-download-sec.md) | Download and extract SEC filings |
| 2 | [Template-based SDG](docs/workflows/02-template-based-sdg.md) | Generate synthetic Q&A from seed questions |
| 3 | [Document-grounded SDG](docs/workflows/03-document-grounded-sdg.md) | Generate and verify Q&A from documents |
| 4 | [SFT](docs/workflows/04-sft.md) | Fine-tune a model and evaluate checkpoints |
| 5 | [Evaluation](docs/workflows/05-eval.md) | Evaluate released models or training checkpoints |
| 6 | [GRPO](docs/workflows/06-grpo.md) | Collect rollouts, train with GRPO, and evaluate |

## Start with the quickstart recipe

Follow the [finance quick start](docs/quick-start.md) for the complete
small-scale path. Start by validating the recipe above, then inspect each
workflow before submitting it:

```bash
uv run nvflow list-stages \
  --workflow finance:sdg/template-based-sdg-demo

uv run nvflow validate \
  --workflow finance:sdg/template-based-sdg-demo

uv run nvflow show-config \
  --workflow finance:sdg/template-based-sdg-demo

uv run nvflow plan \
  --workflow finance:sdg/template-based-sdg-demo \
  --cluster my_cluster
```

The list, validation, and configuration-inspection commands are
target-independent. See the
[planning contract](../../docs/platform/cli-and-python-api.md#plan) for static
and `--live` behavior.

Submit only after the resolved configuration and plan match the intended data,
models, resources, dependencies, and output paths:

```bash
uv run nvflow run --all \
  --workflow finance:sdg/template-based-sdg-demo \
  --cluster my_cluster
```

`run` always performs live-equivalent planning before submission. It keeps
stdout empty; progress, warnings, and errors use stderr.

## Project layout

```text
projects/finance/
├── project.yaml
├── pyproject.toml
├── configs/
├── prompts/
├── workflows/
├── nvfinance/
│   ├── stages/
│   └── utils/
├── tests/
└── docs/
```

- `project.yaml` declares project identity, compatibility, and selector-keyed
  recipe DAGs.
- `workflows/` contains runnable configurations, stage implementation
  bindings, and inherited configuration fragments.
- `nvfinance/` contains finance-owned stage and utility code.
- `docs/` describes finance-specific inputs, outputs, models, quality, and
  troubleshooting.

## Outputs and validation

Workflow outputs are written beneath the configured `/workspace/outputs`
directories. Each workflow guide identifies its exact output contract and
validation checks.

Use the [troubleshooting guide](docs/troubleshooting.md) for cluster, offline,
data, training, evaluation, and resume issues.

## Further reading

- [Finance quick start](docs/quick-start.md)
- [Workflow guides](docs/workflows/)
- [GroundingVerifier](docs/grounding-verifier/README.md): opt-in grounding checks for GRPO rollouts
- [Rubric generation smoke test](docs/rubric-generation-quick-start.md): a controlled comparison of the rubric-generation methods
- [Troubleshooting](docs/troubleshooting.md)
- [NVFlow project and recipe contract](../../docs/platform/recipe-contract.md)
- [Create an NVFlow project](../../docs/getting-started/creating-a-project.md)
