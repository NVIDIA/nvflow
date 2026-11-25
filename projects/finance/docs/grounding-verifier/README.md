# GroundingVerifier for finance

GroundingVerifier evaluates completed `finance_agent` rollouts against the SEC
evidence preserved in their native tool-call history. It writes a separate
sidecar and does not modify the original rollout, reward, or training data.

The evaluator:

1. deterministically splits the submitted answer into claims;
2. routes each claim to retrieved evidence with
   `sentence-transformers/all-MiniLM-L6-v2`;
3. checks entailment with
   `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`; and
4. verifies SEC attribution, entities, metrics, dates, protected values,
   calculations, comparisons, and evidence-scoped refusals.

Both public model revisions can be pinned. Incomplete attribution or model
failure produces `unavailable`; unsupported or conflicting claims are blocked.
Evidence is limited to `retrieve_information` excerpts retained by the rollout.

## Air-gapped setup

NVFlow workers run with Hugging Face and Transformers offline flags enabled, so
the two public models must be staged before submitting `evaluate_grounding`.
From a connected host, download the pinned revisions into the cluster directory
mounted as `/hf_models`:

```bash
uvx --from huggingface_hub hf download sentence-transformers/all-MiniLM-L6-v2 \
  config.json model.safetensors special_tokens_map.json \
  tokenizer.json tokenizer_config.json vocab.txt \
  --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  --local-dir /path/to/models/hf_models/sentence-transformers/all-MiniLM-L6-v2

uvx --from huggingface_hub hf download MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli \
  added_tokens.json config.json model.safetensors special_tokens_map.json \
  spm.model tokenizer.json tokenizer_config.json \
  --revision 6f5cf0a2b59cabb106aca4c287eed12e357e90eb \
  --local-dir /path/to/models/hf_models/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
```

The workflow loads `/hf_models/sentence-transformers/all-MiniLM-L6-v2` and
`/hf_models/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`. See [INSTALL.md →
Download Models](../../../../INSTALL.md#download-models) for the mount
configuration and general model-staging procedure.

## NVFlow integration

Add `evaluate_grounding` after `collect_rollouts` in a finance GRPO workflow;
`grpo/base.yaml` lists it, commented out. For each environment, the stage reads
the rollouts `collect_rollouts` analyzed, refuses them unless their completion
record still matches, and writes `grounding-verifier.jsonl` under
`grounding-verifier-eval/<environment>/`.

```yaml
pipeline_stages:
  - collect_rollouts
  - evaluate_grounding
```

## Repeated feature acceptance

Every GroundingVerifier feature can define a native acceptance profile with a
fixed number of finance-agent runs. The first profile covers the air-gapped
model setup and runs one prepared `finance_sec_search` task three times:

```bash
uv run nvflow run --all --workflow finance:grpo/feature_gates/grounding_airgap \
  --cluster my_cluster -e finance_sec_search
```

Change `feature_runs` in the profile to change the repetition count. The
profile collects its rollouts in a dedicated `feature-gates/grounding_airgap`
directory, evaluates every row, and then fails unless the rollouts still match
their completion record, the sidecar is complete and matches every rollout
fingerprint, each run has its expected rows, the offline model contract holds,
and the unavailable rate stays within the configured limit. Its compact summary
is written beside the sidecar as `feature-gate-grounding_airgap.json`. Delete
the profile's directory to collect fresh rollouts.

This native profile requires the demo `prepare_data` output and SEC cache from
the quick-start workflow. Ordinary CI tests validate the gate logic without a
GPU; executing the finance agent itself requires the configured NVFlow Slurm
environment. `allow` and `block` are both valid execution outcomes because an
unsupported agent answer should be blocked. Detection efficacy remains covered
by the pinned public-model benchmark below.

## Public-model benchmark

The controlled benchmark uses eight FY2024 facts from Amazon, Alphabet, Meta,
and Tesla, which were held out from the earlier AAPL/MSFT/NVDA development set.
Each fixed evidence trace is evaluated once with its grounded answer and against
six mutations: numeric fabrication, entity conflation, metric fabrication,
temporal conflation, accession conflation, and an unsupported claim. This gives
eight grounded and 48 adversarial cases.

```bash
uv sync --all-extras --group ci
PYTHONPATH=projects/finance uv run python -m nvfinance.utils.rl.grounding_benchmark \
  --output-dir artifacts/grounding-verifier-benchmark \
  --routing-model-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  --nli-model-revision 6f5cf0a2b59cabb106aca4c287eed12e357e90eb
```

The command writes `cases.jsonl`, `results.jsonl`, and `summary.json`. It also
records latency, throughput, memory use, model revisions, and host details.
It succeeds only when grounded acceptance and attack rejection are each at
least 90% and the unavailable rate is at most 5%.

The same public-model gate and native rollout-to-sidecar check can be run with:

```bash
PYTHONPATH=projects/finance GROUNDING_VERIFIER_RUN_MODELS=1 uv run pytest -q \
  projects/finance/tests/test_grounding_benchmark.py -k pinned_public_models --no-cov
```

For the native held-out study, see the [frozen evaluation
protocol](evaluation-protocol.md) and [results](evaluation-results.md).

## Scope

This integration verifies whether a submitted answer is supported by the
retrieval excerpts and SEC metadata preserved in an NVFlow rollout. It does not
independently re-verify the complete SEC filing or establish unrestricted
production hallucination detection.
