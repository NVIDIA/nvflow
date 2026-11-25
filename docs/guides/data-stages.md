# Dedup, decontaminate, and filter datasets

NVFlow ships three data-cleaning stages that any project selects by catalog
name. All read JSONL records and write JSONL outputs with a report.

| Stage type | What it does |
|---|---|
| `data.dedup` | Removes exact and near-duplicate records |
| `data.decontaminate` | Drops records that overlap evaluation sets |
| `data.filter` | Drops records whose id appears in another file |

```bash
uv run nvflow stage-type-info data.dedup
uv run nvflow stage-type-info data.decontaminate
uv run nvflow stage-type-info data.filter
```

They run in the `nemo-skills` container by default (set `container` to change
it). Dedup's near-duplicate layer needs `datasketch`, which NVFlow's
`nemo-skills` image includes.

## Remove duplicates

```yaml
stages:
  dedup:
    uses: data.dedup
    contract_version: 1
    input_files:
      - ${directories.raw}/curated.jsonl
      - ${directories.raw}/synthetic.jsonl
    output_dir: ${directories.dedup}
    text_fields: [question]
    id_field: uuid
    near:
      threshold: 0.8
    guard_fields: [source_document]
    keep: {field: origin, order: [curated, synthetic]}
    conflict_field: answer
```

- `text_fields` is required: the fields joined into each record's text. Dotted
  paths such as `messages.0.content` reach nested values. `text_after`, a
  regular expression, keeps only the text after its last match, such as the
  question at the end of a rendered prompt.
- Exact duplicates have the same text once case, punctuation, and whitespace
  are ignored. Set `exact: false` to skip them.
- `near` also matches records whose word-shingle Jaccard similarity, estimated
  with MinHash, is at least `threshold` (default `0.8`). `shingle_size`
  defaults to 5 words and `num_perm` to 128. Leave `near` unset for exact
  matches only.
- `guard_fields` must also be equal, so the same question about two different
  documents is kept twice.
- Each group keeps one record: the first by `keep.order` of the `keep.field`
  value, then by input order.
- `conflict_field` flags groups whose values differ in more than case and
  spacing, such as two answers to one question. `on_conflict` keeps the priority record (`keep_priority`, the
  default), drops the whole group (`drop_group`), or keeps every record
  (`keep_all`).

Outputs: `kept.jsonl`; `dropped.jsonl`, where each record gains `_dedup` with
its group, the id it duplicates, the reason, and the similarity;
`groups.jsonl`; `report.json`; and `dedup.done`.

## Decontaminate against evaluation sets

```yaml
stages:
  decontaminate:
    uses: data.decontaminate
    contract_version: 1
    input_file: ${directories.dedup}/kept.jsonl
    output_dir: ${directories.decontaminate}
    text_fields: [question]
    id_field: uuid
    references:
      benchmark_a: ${directories.eval}/benchmark_a.jsonl
      benchmark_b:
        path: ${directories.eval}/benchmark_b.jsonl
        text_fields: [prompt]
    ngram: {size: 8, drop: 0.5, review: 0.3}
    embedding:
      backend: vllm
      model: /models/embedding-model
      prefix: "query: "
      review: 0.85
```

Each reference set is a JSONL file, with `text_fields`, `text_after`, and
`id_field` defaulting to the stage's own. A reference record without text fails
the run, and `report.json` counts each set's records. The layers run cheapest
first:

| Layer | Score | Default |
|---|---|---|
| `exact` | 1 when the normalized text equals a reference text | on, and always drops |
| `ngram` | Share of a reference's word n-grams that the record contains | `size: 8`, `drop: 0.5`, `top_k: 5` |
| `embedding` | Cosine similarity of text embeddings | off; `top_k: 5` when on |

A score at or above a layer's `drop` threshold drops the record. A score at or
above `review` keeps it and lists the pair in `candidates.jsonl` for a judge or
a person to check. Set `ngram` or `embedding` to `null`, or `exact` to `false`,
to turn a layer off.

The `ngram` and `embedding` layers keep each record's `top_k` best matches. A
reference shorter than `size` words counts as one n-gram of all its words, so a
record that contains it still matches.

Embedding backends:

- `vllm` embeds offline with `vllm.LLM` in the `vllm` container on one GPU.
  `options` go to `vllm.LLM`, and a `pooler_config` mapping becomes vLLM's
  `PoolerConfig`; `container` and `num_gpus` change the placement. With
  `chat_template: true`, each text goes through the model's chat template with
  `prefix` as the system message, for models trained to take their instruction
  that way.
- `sentence_transformers` runs in the stage's container, on CPU by default
  (`device`, `options`).

`dimensions` keeps the first vector components, for models trained with
Matryoshka embeddings. `max_chars` splits longer text into overlapping windows
and scores the best one. Vectors are cached under `cache_dir` (default
`{output_dir}/cache`) by model settings and text. Every row is embedded, even
one a lexical layer drops, so a rerun with new references or thresholds reuses
the row vectors. Those changes mean deleting `output_dir`, so keep the vectors
by setting `cache_dir` outside it.

Outputs: `kept.jsonl`; `dropped.jsonl`, where each record gains
`_contamination` with every matching reference, layer, and score;
`candidates.jsonl`; `report.json`; and `decontaminate.done`.

### Have a model judge the review pairs

Run the judge as NeMo Gym stages: render `candidates.jsonl` with
`nemo_gym.render`, adding a `verifier` row field; collect replies with
`nemo_gym.collect` on Gym's `format_verification` environment and the
`split_by_reward` report extension (see
[NeMo Gym as NVFlow's data plane](../platform/integrations/nemo-gym-data-plane.md#split-rows-by-reward));
then drop the records whose pairs failed with `data.filter`.

## Drop records by id

```yaml
stages:
  drop_contaminated:
    uses: data.filter
    contract_version: 1
    input_file: ${directories.decontaminate}/kept.jsonl
    exclude_file: ${directories.judge}/failed.jsonl
    output_dir: ${directories.filtered}
    id_field: uuid
    exclude_id_field: id
```

A record is excluded when its `id_field` value appears as an
`exclude_id_field` value (default: the same field) in `exclude_file`. A row of
either file without its id field fails the run, and `report.json` counts the
exclude ids that matched no record. Outputs: `kept.jsonl`, `dropped.jsonl`, and
`report.json`.

## Chunks, resume, and changed inputs

- `num_chunks` (default 1) splits the rows into chunks that run as parallel
  tasks, and a final task combines them. A step starts only after the steps
  before it succeed. `dependent_jobs` adds retries, each starting however the
  attempt before it ended.
- One chunk with a CPU embedding backend runs as one task. With `vllm`, the
  exact and n-gram layers run in a CPU task, and the embeddings and final step
  in a GPU task.
- Every step writes its outputs atomically and then a completion record, and
  skips work whose record is complete. A stopped run continues where it
  stopped.
- `manifest.json` records each input's SHA-256 digest, the chunk ranges, and the
  settings. A rerun with different inputs or settings fails rather than mixing
  results: use a new `output_dir`, or delete the old one.
