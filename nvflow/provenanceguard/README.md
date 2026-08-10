# ProvenanceGuard Open v1

Uncalibrated, not paper-faithful open approximation of the research
ProvenanceGuard system.  Provides claim decomposition, embedding-centroid
source routing, DeBERTa NLI scoring, and protected-value checking to produce
per-rollout-row `allow` / `block` / `unavailable` verdicts.

## Algorithm

**routing-nli-v1** — pipeline: decompose, route, NLI, protected-value, decision.

## Decision Semantics

- **allow**: every claim is entailed AND every protected value found in evidence.
- **block**: any claim is contradiction, neutral, no_source, or protected_value_mismatch.
- **unavailable**: no evidence, no claims extracted, or any NLI/model error on any claim.  If any claim has an error, the row is unavailable even alongside block-worthy verdicts.

## Evidence Basis

`retrieval_model_excerpt` — evidence comes from `retrieve_information` tool-call
outputs in the rollout trace, NOT from primary SEC filing verification.

## Generic vs Finance Layers

| Layer | Location | Contents |
|---|---|---|
| Generic | `nvflow/provenanceguard/` | Types, protocols, decomposer, embedder, router, NLI, evaluator, protected_values |
| Finance sidecar + CLI | `nvflow/recipes/finance/utils/rl/provenanceguard.py` | Trace extraction, deterministic/atomic sidecar evaluator, CLI |
| Finance stage | `nvflow/recipes/finance/stages/rl/evaluate_provenance.py` | Registered as `finance/grpo/evaluate_provenance` |

The generic package contains no finance-specific code.  Trace extraction,
sidecar evaluation, and CLI live in the finance recipe layer.

## Enablement

Uncomment `# - evaluate_provenance` in `pipeline_stages` in
`nvflow/recipes/finance/workflows/grpo/base.yaml`.  The stage config
(`stages.evaluate_provenance`) is active by default; only the pipeline_stages
entry is commented out (opt-in).

## Paths

- **Input**: `${directories.step-5-collect-rollouts}/{env}/rollout/output-rs<seed>.jsonl`
- **Input marker** (required): `output-rs<seed>.jsonl.done`
- **Output**: `${directories.provenanceguard-eval}/{env}/provenanceguard-rs<seed>.jsonl`
- **Output marker** (created atomically after `os.replace`): `provenanceguard-rs<seed>.jsonl.done`
- **Directory**: `${model_output_dir}/provenanceguard-eval`

Depends only on `collect_rollouts`; no downstream dependencies.

## Sidecar Protocol

- Exactly one sidecar row for every physical input line (including blank/malformed/non-object).
- No random UUID; `evaluation_uuid` is a deterministic SHA-256 of raw-line fingerprint + seed + line number + algorithm + config digest.  Duplicate identical rows get distinguishable IDs by physical line position.
- Input gate first: both the input file and its sibling `.done` marker must exist before any output state is touched.  A missing input gate preserves prior output and output `.done` unchanged.
- Stale output `.done` marker is cleared only after the input gate passes, before mkdir/temp/evaluation.  Atomic temp write + `os.replace`; a fresh empty `.done` marker is created only after successful replace.  If evaluation fails after the valid input gate, prior output remains intact but the stale `.done` marker stays absent.
- Input file is never modified.

## Source ID Stability

Stable canonical IDs (`sec:cik=<10-digit>:accession=<...>:doc=<...>`) are only
produced when a 10-digit zero-padded CIK + accession + document are all
present.  Never falls back to storage keys or URLs for the canonical ID.
Multiple keys, any unknown key, or no canonical ID yields `attribution_state:
unavailable`; `source_ids` contains only known canonical candidates.

## Protected Values

Full normalized dates/numbers/currency/percent are required for a match.
A bare 4-digit year in the evidence does NOT satisfy a date protected value.

## NLI Label Validation

The NLI scorer validates actual model config label names
(entailment/neutral/contradiction, case-insensitive).  It never assumes
arbitrary `LABEL_0` ordering — if a label does not match one of the three
canonical classes, it raises `ValueError`.

## Model IDs and Licenses

| Model | ID | License (from public model card) |
|---|---|---|
| Routing embedder | `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 |
| NLI scorer | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | MIT |

Both are lazy-loaded (torch/transformers imports happen inside `_ensure_loaded`,
not at module import time).  Production deployments should pin revisions from
Hugging Face Hub metadata; revisions are optional and not hardcoded.

## Differences from Private/Research Harness

This open implementation differs from the private/research ProvenanceGuard
harness in several ways.  No parity is claimed.

- **Rule-based decomposition** rather than LLM/Gemma-based decomposition.
- **No token alignment/conflation head** — claims are not aligned to specific
  evidence tokens.
- **No RF calibration** — NLI scores are used directly without random forest
  calibration.
- Evidence is model-retrieved excerpts, not primary SEC verification.

## Tests

```bash
uv run --frozen --offline pytest tests/test_provenanceguard.py -q -o addopts=""
```

Tests use deterministic fakes (no model downloads, no network).  No real model
execution or download is performed.
