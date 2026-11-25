# Native finance evaluation protocol

The confirmation protocol was frozen before task generation, rollout
collection, or GroundingVerifier evaluation.

## Separation

- All 60 issuers were excluded from development and three earlier studies.
- GroundingVerifier was not used for task generation, rollout collection,
  eligibility, labeling, or adversarial-answer creation.
- Evaluation inputs and labels were frozen before the one-shot guard run.

## Trace generation

An independent generator used official SEC ticker, submissions, filing, and
XBRL Company Facts data to select a 10-K and establish its company, metric,
year, value, and unit. Each task supplied the target filing date, canonical SEC
URL, and bounded retrieval range.

A pinned, locally hosted `Qwen3-4B-Instruct-2507` policy then executed the native
NVFlow `finance_agent` path:

```text
sec_filing_search -> parse_html_page -> retrieve_information -> submit_final_result
```

The tasks covered year-over-year calculations, cross-issuer comparisons, and
closed-world evidence-based refusals. This design tests evidence use and answer
verification, not autonomous discovery of an unknown filing.

## Eligibility and freezing

A calculation or comparison was eligible only when every retrieved
company/metric/year/value/unit tuple agreed with the independent SEC/XBRL
reference. A refusal was eligible only when its bounded evidence contained a
valid coverage fact but omitted the requested metric.

We collected 180 native rollouts and selected 120 eligible traces by task-ID
hash, balanced as 40 calculations, 40 comparisons, and 40 refusals. Selection
did not use GroundingVerifier outcomes.

Each trace contributed one grounded answer and one independently assigned
adversarial answer. Attacks covered numeric fabrication; entity, metric,
temporal, and source conflation; wrong or contradictory comparison winners;
and unsupported refusal claims.

## Success criteria

The preregistered GO gate required:

- at least 95% grounded acceptance and attack rejection;
- Wilson 95% lower bounds of at least 90%;
- at least 90% accuracy in every task category; and
- zero adversarial `allow` decisions.

Intervals used two-sided Wilson estimates and 20,000 issuer-cluster bootstrap
resamples with seed `26081432`.
