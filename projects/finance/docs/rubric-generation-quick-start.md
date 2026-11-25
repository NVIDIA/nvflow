# Rubric generation smoke test

This quick start runs a controlled comparison of the standalone shared rubric
black box. Both arms use the same first 20 prepared questions, rubric model,
and execution geometry. Only `method` differs.

## Required inputs

- `RUBRIC_INPUT_FILE`: container-visible path to an input JSONL file. Each row
  must contain `question`, `context`, and `reference_answer`.
- `RUBRIC_MODEL_CHECKPOINT`: container-visible path to the rubric-generation
  model checkpoint.
- `RUBRIC_OUTPUT_ROOT` (optional): container-visible output directory. It
  defaults to `/workspace/outputs/finance/rubric-generation-smoke`.

Simple uses `question`, `context`, and one `reference_answer`; it does not rank
multiple rollouts. OpenRubric supports either of these inputs:

- `context`, `reference_answer`, `candidate_answers`, and `preferred_index`: it
  keeps the indexed candidate as the positive answer and ranks the rollouts to
  choose the weakest candidate as the negative; or
- only `context` and `reference_answer`: it automatically generates a
  controlled negative answer.

You may also provide an existing `preferred_answer` and `rejected_answer` pair.
The `context` is the evidence for that individual question; it is a JSONL
field, not a separate mount.

## Input JSONL format

Store one JSON object per line. The examples below are pretty-printed only for
readability; each complete object must occupy one line in the input file.

Simple input:

```json
{
  "id": "question-001",
  "question": "What was the company's revenue growth?",
  "context": "The filing reports revenue of $120M, up from $100M last year.",
  "reference_answer": "Revenue grew by 20%."
}
```

OpenRubric with candidate rollouts (typically eight):

```json
{
  "id": "question-001",
  "question": "What was the company's revenue growth?",
  "context": "The filing reports revenue of $120M, up from $100M last year.",
  "reference_answer": "Revenue grew by 20%.",
  "candidate_answers": [
    "Revenue grew by 20%.",
    "Revenue grew by 10%."
  ],
  "preferred_index": 0
}
```

For OpenRubric without candidate rollouts, use the Simple input shape; the
stage creates a controlled negative automatically. Alternatively, replace
`candidate_answers` and `preferred_index` with explicit `preferred_answer` and
`rejected_answer` strings.

## Resources

Each arm starts its own SGLang server for `RUBRIC_MODEL_CHECKPOINT` on 2 nodes
× 8 GPUs, so the target needs the `sglang` container, and 32 GPUs if both arms
run at once. The ranking and rubric-generation passes each allow 16,384 output
tokens, so the model must serve a context window well above 32K tokens; on a
32K window the stage fails with `ContextWindowExceededError` rather than
degrading. Agentic contexts reach about 66K tokens, so size the window to your
corpus.

## Generate rubrics

```bash
RUBRIC_INPUT_FILE=/data/rubric-input.jsonl \
RUBRIC_MODEL_CHECKPOINT=/models/rubric-generator \
RUBRIC_OUTPUT_ROOT=/workspace/outputs/finance/rubric-generation-smoke \
uv run nvflow run --all \
  --config projects/finance/workflows/sdg/rubric-smoke-simple.yaml \
  --cluster my_cluster

RUBRIC_INPUT_FILE=/data/rubric-input.jsonl \
RUBRIC_MODEL_CHECKPOINT=/models/rubric-generator \
RUBRIC_OUTPUT_ROOT=/workspace/outputs/finance/rubric-generation-smoke \
uv run nvflow run --all \
  --config projects/finance/workflows/sdg/rubric-smoke-openrubric.yaml \
  --cluster my_cluster
```

Each command writes 20 validated rows to its own
`step-2-rubrics/rubrics.jsonl` file, under `simple/` or `openrubric/` in
`RUBRIC_OUTPUT_ROOT`.

## Optional finance GRPO handoff

Finance owns the adapter that joins generated rubrics to finance GRPO records and
encodes each rubric for the finance verifier. A finance workflow can select it with:

```yaml
uses: project:nvfinance.stages.rl.prepare_grpo_data:PrepareRubricGRPODataStage
```

The adapter first joins by matching `uuid`/`id`, then falls back to exact
`problem`/`question` text when no matching identifier is found. It writes
`{output_dir}/{env_name}/train.jsonl` and `val.jsonl`. This handoff is not part
of the standalone shared `rubric.build` stage.
