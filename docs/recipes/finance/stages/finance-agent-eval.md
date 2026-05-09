# Finance Agent Eval Stages Reference

> **Status:** finance_agent evaluation is currently disabled in `eval/base.yaml` pending further validation. The configuration below is preserved for re-enablement.

Technical reference for the finance-agent evaluation stages (vals-ai/finance-agent benchmark).

> **Note:** Finance agent evaluation is now integrated into the main eval workflow. The `finance_agent` benchmark is defined in `workflows/eval/base.yaml` and runs alongside SEC-QUE and FinanceBench. See [Eval Workflow](../workflows/05-eval.md) for usage.

## Quick Navigation

- [prepare_data](#prepare_data)
- [agent-gpt-oss-120b](#agent-gpt-oss-120b)
- [Common Agent Parameters](#common-agent-parameters)

---

## prepare_data

**File:** `nvflow/recipes/finance/stages/evaluation/prepare_data.py`
**Registry:** `recipe="finance"`, `workflow="eval"`, `stage="prepare_data"`

### Purpose

The shared `prepare_data` stage now downloads **all** benchmark datasets including `finance_agent`. The `finance_agent` dataset is configured in `workflows/eval/base.yaml` under `benchmarks`.

### Finance Agent Dataset

- **Source:** [vals-ai/finance-agent/data/public.csv](https://github.com/vals-ai/finance-agent/blob/main/data/public.csv)
- **50 public questions** with expert-authored answers
- **Fields:** Question, Answer, Question Type, Expert time (mins), Rubric

### Process

1. **Download:** Fetch public.csv from GitHub
2. **Convert:** Transform to eval.jsonl with `problem`, `expected_answer`, `question_type`, `expert_time_mins`, `rubric`
3. **Save:** Write to `${output_dir}/finance_agent/eval.jsonl` and `eval_stats.json`

### Outputs

```
${output_dir}/
└── finance_agent/
    ├── eval.jsonl          # 50 evaluation samples
    ├── eval_stats.json     # Dataset statistics
    └── public.csv          # Raw CSV (for reference)
```

### Resources

- **Compute:** CPU only
- **Runtime:** ~1 min (small dataset)

---

## agent-gpt-oss-120b

**Registry:** `recipe="finance"`, `workflow="eval"`, `stage="agent-gpt-oss-120b"`

### Purpose

Evaluate GPT-OSS-120B as a **multi-turn agent** on the finance-agent benchmark. Uses GENERATION_MODULE from the dataset (`agent_gen`) to run the agent loop with tool calls (Tavily web search, SEC EDGAR, HTML parsing).

### Key Differences from Standard Eval

| Aspect | Standard Eval (05-eval) | Finance Agent Eval |
|--------|--------------------------|---------------------|
| Benchmark | secque, financebench | finance_agent only |
| Mode | Single-turn direct inference | Multi-turn agent with tools |
| GENERATION_MODULE | nemo-skills default | `finance_agent.agent_gen` |
| Judge | `sec_judge.yaml` | `sec_judge_strict.yaml` |
| max_turns | N/A | 50 |
| max_concurrent_requests | Parallel | 1 (sequential per question) |

### Configuration (from eval/base.yaml benchmarks section)

```yaml
agent-gpt-oss-120b:
  benchmarks: [finance_agent]
  datasets_dir: /workspace/nvflow/recipes/finance/datasets
  judge: *judge_finance_strict
  installation_command: "pip install -q model-library==0.1.8 func-timeout backoff tavily compute-eval @ git+..."
  extra_args: >-
    ++max_turns=50
    ++inference.tokens_to_generate=32000
    ++inference.temperature=0.0
    ++max_concurrent_requests=1
  rollouts:
    model: /hf_models/openai/gpt-oss-120b
    server_type: vllm
    server_gpus: 8
    server_nodes: 1
    extra_args: >-
      ++prompt_format=openai
      ++max_turns=50
      ++inference.tokens_to_generate=32768
      ++inference.temperature=0.0
  stage_kwargs:
    server_args: "--max-model-len 131072"
```

### Resources

- **GPUs:** 8 (120B model)
- **Judge:** GPT-5.1 via OpenAI API (external)
- **Tools:** Tavily API (web search), compute-eval for tool execution
- **Runtime:** Longer than single-turn (multi-turn + tool calls)

---

## Common Agent Parameters

### Agent-Specific Configuration

| Parameter | Description |
|-----------|-------------|
| `installation_command` | Pip install model-library, func-timeout, tavily, compute-eval |
| `judge` | `judge_finance_strict` (GPT-5.1, sec_judge_strict.yaml) |
| `extra_args.max_turns` | Max agent turns per question (default: 50) |
| `extra_args.max_concurrent_requests` | 1 (sequential to avoid API rate limits) |
| `rollouts.extra_args.prompt_format` | `openai` (OpenAI function-calling format) |

### Judge (judge_finance_strict)

Strict finance-domain judge matching vals-ai/finance-agent's judge_new.py:
- **Model:** GPT-5.1
- **Prompt:** `sec_judge_strict.yaml` (domain tolerance rules, few-shot examples)
- **Temperature:** 0.0
- **Skip extraction:** Yes (judgement only)

---

## Output Structure

```
${base_output_dir}/baselines/gpt-oss-120b/
├── eval-results/
│   └── finance_agent/
│       ├── output-rs*.jsonl    # Predictions per seed
│       └── metrics.json         # Aggregated metrics
└── logs/
```

---

## Adding More Agent Model Stages

The finance_agent benchmark is part of the unified eval workflow. Any model evaluated through the embedded eval stage (in SFT/GRPO pipelines) or the standalone baselines workflow will automatically include finance_agent (since it's defined in `eval/base.yaml`).

To add a new baseline model for agent evaluation, add it to `eval/baselines.yaml`:

```yaml
models:
  qwen3-32b:
    path: /hf_models/Qwen/Qwen3-32B
    server_type: vllm
    gpus: 2
    nodes: 1
    inference_args: >-
      ++prompt_config=/workspace/nvflow/recipes/finance/prompts/secque_template.yaml
      ++inference.tokens_to_generate=32768
      ++inference.temperature=0.0
    server_args: "--max-model-len 65536 --async-scheduling"
```

---

## Validation

```bash
# Check finance_agent eval output (from nvflow directory)
ls outputs/finance/sap-500/workflow-1-baseline-eval/baselines/gpt-oss-120b/eval-results/finance_agent/

# View metrics
cat outputs/finance/sap-500/workflow-1-baseline-eval/baselines/gpt-oss-120b/eval-results/finance_agent/metrics.json | jq .

# Check prepared dataset
ls /workspace/nvflow/recipes/finance/datasets/finance_agent/
```

---

## Common Issues

### Tavily API errors

**Symptom:** Agent fails during web search tool calls

**Solution:** Set `TAVILY_API_KEY` in environment; verify API quota

### compute-eval import errors

**Symptom:** GENERATION_MODULE fails to load agent_gen

**Solution:** Ensure `installation_command` ran; check `compute-eval @ git+https://github.com/NVIDIA/compute-eval.git@2d14770` is installed

### OOM with 120B model

**Solution:** Use `server_gpus: 8`; ensure `--max-model-len 131072` fits in GPU memory

---

See [Finance Agent Benchmark](../workflows/06-finance-agent-eval.md) for an overview, or [Eval Workflow](../workflows/05-eval.md) for full usage examples and configuration.
