# Train with NVFlow's shared stages

NVFlow ships NeMo-RL training stages and a dataset split stage that any project
selects by catalog name. The project keeps its data preparation, its NeMo-RL
presets, and its model choices.

| Stage type | What it does |
|---|---|
| `data.train_validation_split` | Splits one JSONL file into `train.jsonl`, `val.jsonl`, and `split.done` |
| `nemo_gym.train_validation_split` | Runs that split for each selected NeMo Gym environment |
| `nemo_rl.sft` | Supervised fine-tuning with NeMo-RL |
| `nemo_rl.grpo` | GRPO training with NeMo-RL on NeMo Gym environments |

Inspect any of them from the CLI:

```bash
uv run nvflow list-stage-types --namespace nemo_rl
uv run nvflow stage-type-info data.train_validation_split
```

## Split a dataset

```yaml
stages:
  train_validation_split:
    uses: data.train_validation_split
    contract_version: 1
    input_file: ${directories.prepared}/final_result.jsonl
    output_dir: ${directories.split}
    val_ratio: 0.1
    random_seed: 42
    stratify_by: question_type
    fields: [input, output, uuid, total_token_length]
    max_token_length: 32768
    token_length_field: total_token_length
```

Only `input_file` and `output_dir` are required:

- `val_ratio` defaults to `0.1` and `random_seed` to `42`.
- `stratify_by` keeps each value's share in both files. Records without the
  field form their own `<missing>` group. Leave it unset to split all records
  as one group.
- `fields` keeps only the listed fields, in that order. A missing field is
  written as `""`, or `0` for the token-length field. Leave it unset to keep
  every field.
- `max_token_length` drops records whose `token_length_field` value is larger.
  It requires `token_length_field`.
- `sort_by` orders `train.jsonl` by a numeric field, such as
  `difficulty_profile.avg_reward`, instead of shuffling it. `sort_order` is
  `desc` (the default) or `asc`, and records without the field go last.
- `stratify_by` also takes a list of fields, to stratify by their combination.
- `group_by` keeps records that share the field's value in the same file, such
  as several questions about one document.

`split.done` records the SHA-256 digest and row count of both files, so
downstream stages can tell a finished split from a partial one.

`splits` replaces the train and validation files with named splits:

```yaml
    splits: {held_out: 0.05, pool_a: 0.475, pool_b: 0.475}
    group_by: source_document
    id_field: uuid
```

The stage writes `<name>.jsonl` for each split, `manifest.jsonl` with each
record's `id_field` value (or row number) and split, and `splits.done` with the
digests of all of them. Ratios must sum to 1, and `splits` can't be combined
with `val_ratio` or `sort_by`. Without `group_by`, each split's total stays
within a row of its ratio, even across small strata. With `group_by` or
`id_field` set, the input order doesn't change which split a record lands in.

`nemo_gym.train_validation_split` accepts the same options except `splits`. It reads
`{input_dir}/{environment}/{input_filename}` (`input_filename` defaults to
`final_result.jsonl`) for each environment in `environments`, and writes to
`{output_dir}/{environment}/`. With `input_contract: dataset_selection`, it
first checks the Gym dataset-selection evidence next to each input file.

## Train with your own presets

Both training stages merge one named preset from the project's `presets_file`
with the workflow's `overrides`. NVFlow ships no presets. Point `presets_file`
at a YAML file in your project with a `project://` URI:

```yaml
stages:
  training:
    uses: nemo_rl.sft
    contract_version: 1
    presets_file: project://workflows/sft/sft_presets.yaml
    preset: sft-base
    model_name: Qwen/Qwen3-4B
    training_data: ${directories.split}/train.jsonl
    validation_data: ${directories.split}/val.jsonl
    output_dir: ${directories.training}
    total_gpus: 8
    backend: megatron
    overrides:
      sft:
        max_num_epochs: 1
    wandb_project: my-project-sft
    wandb_mode: online
    dependencies:
      - train_validation_split
```

The presets file maps each preset name to a complete NeMo-RL 0.8.0
configuration. NVFlow merges the workflow's `overrides` into the preset and
passes the result to NeMo-RL as its config file, so the preset must set every
key NeMo-RL requires, including `policy.offload_policy_before_refit` and
`policy.offload_optimizer_for_refit`. Start from Finance's
[SFT presets](../../projects/finance/workflows/sft/sft_presets.yaml); this
excerpt shows only a few of the keys a preset needs:

```yaml
presets:
  sft-base:
    policy:
      max_total_sequence_length: 32768
      offload_policy_before_refit: false
      offload_optimizer_for_refit: true
    sft:
      max_num_epochs: 3
    checkpointing:
      enabled: true
```

Both `preset` and `presets_file` are required. NVFlow sets the model, cluster
size, data paths, checkpoint and log directories, and the checkpoint save
deadline on the command line, so presets do not need them. W&B logging is off
unless `wandb_mode` is `online` or `offline`.

`nemo_rl.grpo` uses the same `presets_file` and `preset` fields with a GRPO
preset. It also needs `environments` and `data_source_dir`, the directory that
holds each environment's `train.jsonl` and `val.jsonl` from the split. With
`input_contract: train_validation_split`, training checks each `split.done`
before it starts. A judge that NVFlow launches needs `vllm_container` or
`models.judge.container`.

The Finance [SFT](../../projects/finance/workflows/sft/base.yaml) and
[GRPO](../../projects/finance/workflows/grpo/base.yaml) workflows are complete
examples.

## Evaluate checkpoints

vLLM serves HuggingFace directories, while NeMo-RL writes Megatron or FSDP
checkpoints. Project evaluation stages call
`nvflow.integrations.nemo_rl.checkpoints.plan_checkpoint_conversion` to plan
the conversion task before serving a checkpoint.

## Related documentation

- [NeMo ecosystem integration boundaries](../platform/integrations/nemo-ecosystem.md)
- [Finance SFT](../../projects/finance/docs/workflows/04-sft.md)
- [Finance GRPO](../../projects/finance/docs/workflows/06-grpo.md)
