# Native finance evaluation results

## Verdict

The frozen confirmation passed its preregistered GO criteria.

| Endpoint | Correct | Rate | Wilson 95% interval |
|---|---:|---:|---:|
| Grounded acceptance | 119/120 | 99.17% | 95.43-99.85% |
| Adversarial rejection | 120/120 | 100% | 96.90-100% |

- Calculations: 80/80 paired decisions correct.
- Comparisons: 80/80 paired decisions correct.
- Refusals: 79/80 paired decisions correct.
- False allows: 0.
- Exact two-sided McNemar comparison with an accept-all baseline:
  `p = 9.18e-35`.

All attacks were blocked. They included numeric fabrication; entity, metric,
temporal, and source conflation; wrong or contradictory comparison winners;
and unsupported refusal detail or assertion.

## Collection and validation

- Native collection: 180 rollouts in 1:46:40.
- Frozen evaluation: 120 independent traces from 60 unseen issuers.
- Guard evaluation: 240 paired rows in 33.60 seconds.
- Native exclusions: 40 missing or incorrect source-bound tuples, nine
  unexpected retrieval counts, one unpaired submission, and ten eligible
  reserve traces. Exclusions did not use guard outcomes.
- Repository tests: 385 passed and one skipped.
- Pinned-public-model integration test, Ruff, `uv lock --check`, and
  `git diff --check`: passed.

## Residual error

The only false block was a grounded refusal for a company ending in `N.V.`.
The deterministic decomposer split the legal suffix into an incomplete clause,
which the NLI model classified as contradiction. The candidate was not changed
after observing the confirmation result.

## Public models

```text
sentence-transformers/all-MiniLM-L6-v2
revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41

MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
revision 6f5cf0a2b59cabb106aca4c287eed12e357e90eb
```

## Claim boundary

These results demonstrate source-bound detection on controlled, issuer-held-out
retrieval excerpts preserved in native NVFlow finance traces. They do not
independently validate complete SEC documents, estimate production failure
prevalence, prove unrestricted hallucination detection, or demonstrate an
NVIDIA Slurm deployment.
