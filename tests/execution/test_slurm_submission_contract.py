# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Direct-Slurm submissions of maintained workflows against real NeMo-Skills APIs."""

from __future__ import annotations

import contextlib
import inspect
import json
import re
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.requires_nemo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Workflow -> NeMo-Skills generate invocations it submits on direct Slurm.
WORKFLOWS = {
    "projects/finance/workflows/sft/qwen3_4b.yaml": 0,
    "projects/finance/workflows/grpo/qwen3_4b.yaml": 2,
    "projects/finance/workflows/sdg/document-grounded-sdg.yaml": 0,
    "projects/hopchain/workflows/sdg/full.yaml": 5,
}
TARGET = """\
executor: slurm
job_dir: {job_dir}
account: acct
partition: batch
cpu_partition: cpu
gpus_per_node: 8
containers:
  nemo-skills: ns-image
  vllm: vllm-image
  sglang: sglang-image
  nemo-rl: rl-image
  nemo-gym: gym-image
env_vars: [HF_HOME=/hf]
mounts: ['/workspace:/workspace', '/hf:/hf']
"""


class _MissingFilesystem:
    def batch_exists(self, paths):
        return dict.fromkeys(paths, False)

    def batch_completion_matches(self, probes):
        return [False] * len(probes)

    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: False


def _plan(
    workflow: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    live: bool = False,
) -> tuple[Any, Any, Any]:
    """Plan *workflow* for a temporary direct-Slurm target."""
    import nvflow.integrations.nemo_skills.target as target_module
    from nvflow import load_workflow
    from nvflow.execution import TargetBinding, WorkflowPlanner
    from nvflow.execution.target import resolve_target

    target_path = tmp_path / "contract-slurm.yaml"
    target_path.write_text(TARGET.format(job_dir=tmp_path / "jobs"), encoding="utf-8")
    monkeypatch.setenv("NVFLOW_CONFIG_DIR", str(tmp_path))
    definition = load_workflow(PROJECT_ROOT / workflow)
    binding = TargetBinding.resolve(definition.cluster, "contract-slurm")
    resolved = resolve_target(binding, target_path)
    with target_module.bind_target_config(resolved.source_path):
        graph = WorkflowPlanner(
            definition,
            binding,
            filesystem=_MissingFilesystem(),
        ).execution_graph(resolved, None, None, live=live)
    return definition, resolved, graph


def _submit(
    workflow: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    workflow_label: str | None = None,
) -> list[dict[str, Any]]:
    """Plan and submit *workflow*, recording each scheduler-facing call and its job comment."""
    import yaml
    from nemo_skills.pipeline.utils import declarative
    from nemo_skills.pipeline.utils import exp as exp_module

    import nvflow.execution.engines.slurm as slurm_engine
    import nvflow.integrations.nemo_skills as nemo_skills_integration
    import nvflow.integrations.nemo_skills.commands as commands
    import nvflow.integrations.nemo_skills.filesystem as filesystem
    import nvflow.integrations.nemo_skills.slurm as slurm_integration
    import nvflow.integrations.nemo_skills.target as target_module
    import nvflow.project.packaging as packaging
    from nvflow.execution.engines.base import EngineContext
    from nvflow.execution.engines.slurm import DirectSlurmEngine

    definition, resolved, graph = _plan(workflow, tmp_path, monkeypatch)
    cluster_config = yaml.safe_load(resolved.source_path.read_text(encoding="utf-8"))
    if graph.provenance is not None:
        graph = replace(
            graph,
            provenance=replace(graph.provenance, submittable=True, clean=True),
        )

    calls: list[dict[str, Any]] = []

    def record(via: str, name: str, kwargs: dict[str, Any]) -> None:
        calls.append(
            {
                "via": via,
                "name": name,
                "kwargs": kwargs,
                "comment": slurm_integration.active_job_comment(),
            }
        )

    def run_cmd(**kwargs: Any) -> None:
        kwargs.pop("ctx", None)
        record("run_cmd", kwargs["expname"], kwargs)

    def generate(**kwargs: Any) -> None:
        kwargs.pop("ctx", None)
        record("generate", kwargs["expname"], kwargs)

    def pipeline_run(self: Any, *_args: Any, **_kwargs: Any) -> None:
        dependencies = [
            dependency
            for job in self.jobs
            for dependency in job.get("dependencies") or ()
            if isinstance(dependency, str)
        ]
        record(
            "Pipeline",
            self.name,
            {"run_after": [*(self.run_after or ()), *dependencies]},
        )

    @contextlib.contextmanager
    def get_exp(name: str, _cluster_config: dict[str, Any]):
        record("get_exp", name, {})
        yield types.SimpleNamespace(jobs=[])

    def add_task(exp: Any, **kwargs: Any) -> str:
        exp.jobs.append(types.SimpleNamespace(id=kwargs["task_name"], handle=""))
        record("add_task", kwargs["task_name"], kwargs)
        return str(kwargs["task_name"])

    def run_exp(exp: Any, *_args: Any, **_kwargs: Any) -> None:
        for index, job in enumerate(exp.jobs):
            job.handle = f"slurm:///{1000 + index}/master/0"

    def sbatch(_argv: Any, **kwargs: Any) -> Any:
        record("sbatch", "cleanup", {"script": kwargs["input"]})
        return types.SimpleNamespace(returncode=0, stdout="Submitted batch job 1", stderr="")

    monkeypatch.setattr(commands, "run_cmd", run_cmd)
    monkeypatch.setattr(commands, "generate", generate)
    monkeypatch.setattr(declarative.Pipeline, "run", pipeline_run)
    monkeypatch.setattr(exp_module, "get_exp", get_exp)
    monkeypatch.setattr(exp_module, "add_task", add_task)
    monkeypatch.setattr(exp_module, "run_exp", run_exp)
    monkeypatch.setattr(slurm_engine, "subprocess", types.SimpleNamespace(run=sbatch))
    monkeypatch.setattr(slurm_engine, "_clean_stale_experiments", lambda *_a, **_k: None)
    monkeypatch.setattr(slurm_engine, "discover_submission_receipt", lambda *_a, **_k: None)
    monkeypatch.setattr(
        nemo_skills_integration,
        "load_target_config",
        lambda *_a, **_k: dict(cluster_config),
    )
    monkeypatch.setattr(nemo_skills_integration, "launcher_is_remote", lambda _config: False)
    monkeypatch.setattr(slurm_integration, "apply_sbatch_args_autopatch", lambda: None)
    monkeypatch.setattr(target_module, "bind_target_config", lambda _path: contextlib.nullcontext())
    monkeypatch.setattr(filesystem, "submission_tunnel_scope", contextlib.nullcontext)
    monkeypatch.setattr(
        packaging,
        "project_submission",
        lambda *_a, **_k: contextlib.nullcontext(),
    )

    DirectSlurmEngine().submit(
        graph,
        EngineContext(
            target=resolved,
            project_context=definition.project_context,
            platform_source="contract",
            cache_dir=tmp_path / "cache",
            workflow=workflow_label,
        ),
    )
    return calls


@pytest.mark.parametrize(("workflow", "generations"), WORKFLOWS.items())
def test_submissions_bind_to_nemo_skills_and_wait_on_submitted_experiments(
    workflow: str,
    generations: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_skills.pipeline.generate import generate
    from nemo_skills.pipeline.run_cmd import run_cmd

    signatures = {
        "run_cmd": inspect.signature(run_cmd),
        "generate": inspect.signature(generate),
    }
    calls = _submit(workflow, tmp_path, monkeypatch)

    assert calls
    assert sum(call["via"] == "generate" for call in calls) == generations
    submitted: list[str] = []
    for call in calls:
        kwargs = call["kwargs"]
        if call["via"] in signatures:
            signatures[call["via"]].bind_partial(**kwargs)
        for name in kwargs.get("run_after") or ():
            assert name in submitted, f"{call['name']} waits on unsubmitted experiment {name!r}"
        if call["via"] in {"run_cmd", "generate", "Pipeline", "get_exp"}:
            submitted.append(call["name"])
        rendered = json.dumps(kwargs, default=str)
        assert "/nemo_run/code//nemo_run/code" not in rendered
        assert "checkpoint_must_save_by=None" not in rendered
        assert "{{NVFLOW_CODE_ROOT}}" not in rendered


@pytest.mark.parametrize("workflow", WORKFLOWS)
def test_every_scheduler_call_runs_under_its_workflow_and_stage_label(
    workflow: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nvflow.execution.job_labels import JobLabel

    _definition, _resolved, graph = _plan(workflow, tmp_path, monkeypatch)
    stages = {stage.stage for stage in graph.stages}
    calls = _submit(workflow, tmp_path, monkeypatch, workflow_label="demo:selected")

    assert calls
    for call in calls:
        label = JobLabel.from_comment(call["comment"] or "")
        assert label is not None, call["name"]
        assert label.workflow == "demo:selected"
        assert label.stage in stages
        assert label.experiment
    for call in calls:
        if call["via"] == "sbatch":
            assert f"#SBATCH --comment={call['comment']}" in call["kwargs"]["script"]


def test_chunked_generation_steps_end_with_their_own_merge_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _submit("projects/finance/workflows/grpo/qwen3_4b.yaml", tmp_path, monkeypatch)

    generations = [call["name"] for call in calls if call["via"] == "generate"]
    assert generations
    for name in generations:
        waiting = [
            call["name"] for call in calls if name in (call["kwargs"].get("run_after") or ())
        ]
        assert waiting == [f"{name}-merge"]
        assert any(f"{name}-merge" in (call["kwargs"].get("run_after") or ()) for call in calls)


_HET_HOST = re.compile(r'(POLICY|JUDGE)_HOST="\$\{SLURM_MASTER_NODE_HET_GROUP_(\d+):-localhost\}"')


@pytest.mark.parametrize(
    "workflow",
    (
        "projects/finance/workflows/eval/smoke.yaml",
        "projects/finance/workflows/sft/qwen3_4b.yaml",
        "projects/finance/workflows/grpo/qwen3_4b.yaml",
    ),
)
def test_heterogeneous_clients_reach_their_own_server_groups(
    workflow: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _definition, _resolved, graph = _plan(workflow, tmp_path, monkeypatch)

    references = 0
    for task in graph.tasks:
        if len(task.resources) < 2:
            continue
        commands = "\n".join(command.command for command in task.commands)
        assert "${SLURM_MASTER_NODE:-127.0.0.1}" not in commands, task.name
        roles = [resource.role for resource in task.resources]
        for server, index in _HET_HOST.findall(commands):
            assert roles[int(index)].endswith(f"-{server.lower()}"), (task.name, server, roles)
            references += 1
    assert references


def test_grpo_managed_judge_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import nvflow.integrations.nemo_skills.slurm as slurm_integration

    excluded_jobs: set[str] = set()
    monkeypatch.setattr(
        slurm_integration,
        "jobs_excluded_from_dependencies",
        lambda: contextlib.nullcontext(excluded_jobs),
    )
    calls = _submit("projects/finance/workflows/grpo/qwen3_4b.yaml", tmp_path, monkeypatch)
    tasks = {call["name"]: call["kwargs"] for call in calls if call["via"] == "add_task"}
    training = next(kwargs for name, kwargs in tasks.items() if name.endswith("-grpo-0"))
    judge = next(kwargs for name, kwargs in tasks.items() if name.endswith("-judge-0"))
    cleanup = [call["kwargs"]["script"] for call in calls if call["via"] == "sbatch"]

    assert "export NVFLOW_CHECKPOINT_SAVE_BY=99:23:45:00" in training["cmd"]
    assert judge["log_dir"].endswith("/training-logs")
    assert judge["sbatch_kwargs"] == {"time": "100-00:00:00", "dependency_type": "after"}
    assert "SERVER_HOST=$(hostname)" in judge["cmd"]
    assert excluded_jobs == {"slurm:///1001/master/0"}
    assert len(cleanup) == 1
    assert "--dependency=afterany:1000" in cleanup[0]
    assert "scancel 1001" in cleanup[0]


def test_multi_node_gym_generation_plans_live_without_scheduler_executors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_skills.pipeline.utils import declarative
    from nemo_skills.pipeline.utils import exp as exp_module

    def no_executor(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("planning built a scheduler executor")

    monkeypatch.setattr(exp_module, "get_executor", no_executor)
    monkeypatch.setattr(declarative, "get_executor", no_executor)
    _definition, _resolved, graph = _plan(
        "projects/finance/workflows/sdg/relational-sdg-hard-categories.yaml",
        tmp_path,
        monkeypatch,
        live=True,
    )

    assert any(resource.nodes == 2 for task in graph.tasks for resource in task.resources)
