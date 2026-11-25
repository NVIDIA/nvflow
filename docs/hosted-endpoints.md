# Hosted Model Endpoints

Some stages grade with a model over an HTTP API instead of serving one on cluster
GPUs. The client is OpenAI-compatible, so an endpoint is just three values: base
URL, model id, and the name of the environment variable holding the key.

The shipped configs point at the [NVIDIA inference hub](#using-the-nvidia-inference-hub).
If you don't have a key for it, use OpenAI.

## Using OpenAI

The jobs need the key. On Slurm, export it in the shell you submit from and list
the bare name under the target's `env_vars`; a name with no value is taken from
the submitting shell, so the secret stays out of the file:

```bash
export OPENAI_API_KEY=sk-...
```

```yaml
env_vars:
  - OPENAI_API_KEY
```

On Ray, provision the key in every cluster's environment instead; NVFlow never
forwards secret values to Ray Jobs. See where settings live for
[Slurm](slurm/README.md#where-settings-live) and
[Ray](ray/README.md#where-settings-live).

Then point the judge at OpenAI. Finance evals share one judge definition, the
`.judge_gpt52` anchor in `projects/finance/workflows/eval/base.yaml`. NVFlow
refuses to run uncommitted project files, so override it in a git-ignored
`private_*.yaml` overlay next to the workflow instead of editing it. Anchors
don't cross files, so set the judge on each benchmark that uses it:

```yaml
# projects/finance/workflows/eval/private_baselines.yaml
_base_: baselines.yaml

.openai_judge: &openai_judge
  deployment: external
  model_type: openai_model
  base_url: "https://api.openai.com/v1"
  model: "gpt-5.2"
  api_key_env: OPENAI_API_KEY

benchmarks:
  fabv1: {models: {judge: *openai_judge}}
  secque: {models: {judge: *openai_judge}}
  financebench: {models: {judge: *openai_judge}}
```

Run it as `--workflow finance:eval/private_baselines`.

`api_key_env` names the variable rather than holding the secret.
`deployment: external` reserves no GPUs because neither Gym nor NVFlow launches
the endpoint. The same block configures a GRPO `rejudge` judge, set per
environment in your model config.

For an on-cluster model, `deployment: nvflow` asks NVFlow to launch a separate
vLLM or SGLang service, while `deployment: gym` asks Gym to launch local vLLM.

## Using the NVIDIA inference hub

This is where the shipped configs point, reading `NVIDIA_API_KEY`. Setup is the
same; only the values differ. The hub requires the fully qualified model id —
`openai/openai/gpt-5.2`, not `gpt-5.2`:

```yaml
deployment: external
model_type: openai_model
base_url: "https://inference-api.nvidia.com/v1"
model: "openai/openai/gpt-5.2"
api_key_env: NVIDIA_API_KEY
```

## HopChain judge

The HopChain judge (`projects/hopchain/workflows/sdg/full.yaml`,
stage `judge_candidate_queries_openai`) reaches its endpoint through LiteLLM, so the
field names differ — `provider`, `model`, `api_base`, `api_key_name` — and LiteLLM
strips the first segment of `model` to pick the provider. Against OpenAI, set
`provider: openai`, `model: gpt-5.5`, and `api_key_name: OPENAI_API_KEY`, and
delete `api_base`, which the shipped config points at the NVIDIA hub.

## Troubleshooting

| Symptom | Cause |
|---------|-------|
| `401` with an empty key in the logs | Key not reaching the job: on Slurm, check for the bare name under `env_vars`; on Ray, check the clusters' environment |
| `403` model access denied | Key has no access to that model, or the id is not fully qualified |
| `500` from a GPT-5.x endpoint | `reasoning.effort: none` is echoed back and rejected; set it explicitly, as the QA `gym` block in `projects/finance/workflows/eval/base.yaml` does |

Judge calls go out from the compute nodes, so those need outbound network access,
not just your login shell.

## See also

- [Cluster Configuration → Environment Variables](operations/cluster-configuration.md#environment-variables)
- [Finance eval workflow → Choosing a judge](../projects/finance/docs/workflows/05-eval.md#choosing-a-judge)
