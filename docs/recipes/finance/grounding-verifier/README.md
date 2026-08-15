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
uv run hf download sentence-transformers/all-MiniLM-L6-v2 \
  config.json model.safetensors special_tokens_map.json \
  tokenizer.json tokenizer_config.json vocab.txt \
  --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
  --local-dir /path/to/models/hf_models/sentence-transformers/all-MiniLM-L6-v2

uv run hf download MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli \
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

Add `evaluate_grounding` after `collect_rollouts` in a finance GRPO workflow.
The stage reads completed `output-rs*.jsonl` files and writes matching
`grounding-verifier-rs*.jsonl` sidecars.

```yaml
pipeline_stages:
  - collect_rollouts
  - evaluate_grounding
```

## Public-model benchmark

The controlled benchmark uses eight FY2024 facts from Amazon, Alphabet, Meta,
and Tesla, which were held out from the earlier AAPL/MSFT/NVDA development set.
Each fixed evidence trace is evaluated once with its grounded answer and against
six mutations: numeric fabrication, entity conflation, metric fabrication,
temporal conflation, accession conflation, and an unsupported claim. This gives
eight grounded and 48 adversarial cases.

```bash
uv sync
uv run python -m nvflow.recipes.finance.utils.rl.grounding_benchmark \
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
GROUNDING_VERIFIER_RUN_MODELS=1 uv run pytest -q \
  tests/test_grounding_benchmark.py -k pinned_public_models --no-cov
```

For the native held-out study, see the [frozen evaluation
protocol](evaluation-protocol.md) and [results](evaluation-results.md).

## Scope

This integration verifies whether a submitted answer is supported by the
retrieval excerpts and SEC metadata preserved in an NVFlow rollout. It does not
independently re-verify the complete SEC filing or establish unrestricted
production hallucination detection.
