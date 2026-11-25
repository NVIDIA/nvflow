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
"""Contracts for optional Ray-on-Slurm test infrastructure."""

import importlib.util
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from nvflow.execution.target import TargetBinding, resolve_target

ROOT = Path(__file__).parents[3]
HARNESS = ROOT / "scripts" / "ray-test"
START_CLUSTER = HARNESS / "start_cluster.slurm"
CLUSTERS = HARNESS / "clusters.py"

RAY_TARGET = """\
executor: ray
ray:
  control_cluster: skills
  artifact_root: {artifact_root}
  clusters:
    skills:
      jobs_url: ${{oc.env:RAY_SKILLS_JOBS_URL}}
      allow_insecure: true
      expected_image: nvflow-nemo-skills:v1.2.0
      expected_ray: "2.58.0"
    serving:
      jobs_url: ${{oc.env:RAY_VLLM_JOBS_URL}}
      controller_jobs_url: ${{oc.env:RAY_VLLM_CONTROLLER_JOBS_URL}}
      allow_insecure: true
      expected_image: nvflow-vllm:v0.29.0
      expected_ray: "2.58.0"
      gpus_per_node: 8
  routes:
    nemo-skills: skills
    vllm: serving
"""

SLURM_TARGET = """\
executor: slurm
account: finance
partition: gpu
cpu_partition: cpu_short,cpu
containers:
  nemo-skills: /images/nemo-skills.sqsh
  vllm: /images/vllm.sqsh
mounts:
  - /lustre/models:/hf_models
  - /lustre/outputs:/workspace/outputs
env_vars:
  - HF_TOKEN=secret-token-value
"""


@pytest.fixture
def clusters() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ray_test_clusters", CLUSTERS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _targets(clusters: ModuleType, tmp_path: Path):
    ray_path = tmp_path / "cw_ray.yaml"
    ray_path.write_text(RAY_TARGET.format(artifact_root=tmp_path / "artifacts"), encoding="utf-8")
    slurm_path = tmp_path / "cw.yaml"
    slurm_path.write_text(SLURM_TARGET, encoding="utf-8")
    return ray_path, clusters.load_ray_target(ray_path), clusters.load_slurm_settings(slurm_path)


def test_start_defaults_fit_the_quick_start_cpu_steps(
    clusters: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ray_path, _target, _slurm = _targets(clusters, tmp_path)
    received: dict[str, object] = {}

    def record(_target, _slurm, _selected, **options):
        received.update(options)
        return tmp_path / "endpoints.env"

    monkeypatch.setattr(clusters, "start", record)

    assert clusters.main(["start", str(ray_path), "--slurm-target", str(tmp_path / "cw.yaml")]) == 0
    assert (received["cpus"], received["memory"], received["timeout"]) == (16, "64G", 7200)


def _publish(clusters: ModuleType, target, name: str, host: str, job_id: str) -> None:
    target.state_dir.mkdir(parents=True, exist_ok=True)
    clusters.endpoint_path(target, name).write_text(
        f"ROLE={name}\nJOBS_URL=http://{host}:8265\nCONTROLLER_JOBS_URL=http://{host}:8265\n"
        f"IMAGE_ID=test\nSLURM_JOB_ID={job_id}\n",
        encoding="utf-8",
    )


def test_start_cluster_script_is_executable_valid_shell() -> None:
    assert START_CLUSTER.stat().st_mode & 0o111
    subprocess.run(["bash", "-n", str(START_CLUSTER)], check=True)


def test_ray_steps_never_combine_overlap_with_exclusive() -> None:
    # Slurm rejects an srun step that passes both flags.
    script = START_CLUSTER.read_text(encoding="utf-8").replace("\\\n", " ")
    steps = [line for line in script.splitlines() if line.strip().startswith("srun ")]

    assert steps
    assert not [step for step in steps if "--overlap" in step and "--exclusive" in step]


def test_node_ip_skips_link_local_and_loopback_addresses(tmp_path: Path) -> None:
    script = START_CLUSTER.read_text(encoding="utf-8")
    function = re.search(r"^node_ip\(\) \{\n.*?^\}\n", script, re.MULTILINE | re.DOTALL)
    assert function is not None
    assert 'head_ip="$(node_ip ' in script and 'worker_ip="$(node_ip ' in script
    getent = tmp_path / "getent"
    getent.write_text(
        "#!/bin/sh\n"
        "printf '169.254.3.1 STREAM gpu-node\\n127.0.1.1 STREAM gpu-node\\n"
        "10.0.0.7 STREAM gpu-node\\n'\n",
        encoding="utf-8",
    )
    getent.chmod(0o755)

    resolved = subprocess.run(
        ["bash", "-c", f"{function.group(0)}node_ip gpu-node"],
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=True,
    )

    assert resolved.stdout.strip() == "10.0.0.7"


def test_free_port_skips_ports_already_listening(tmp_path: Path) -> None:
    script = START_CLUSTER.read_text(encoding="utf-8")
    function = re.search(r"^free_port\(\) \{\n.*?^\}\n", script, re.MULTILINE | re.DOTALL)
    assert function is not None
    ss = tmp_path / "ss"
    ss.write_text(
        '#!/bin/sh\ncase "$2" in *:20000|*:20001) echo "LISTEN 0 128 0.0.0.0:busy";; esac\n',
        encoding="utf-8",
    )
    ss.chmod(0o755)

    picked = subprocess.run(
        ["bash", "-c", f"{function.group(0)}free_port 20000"],
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=True,
    )

    assert picked.stdout.strip() == "20002"


def test_shared_nodes_cannot_collide_on_ray_default_ports() -> None:
    script = START_CLUSTER.read_text(encoding="utf-8")

    assert not re.search(r"\b(6379|8265|10001|52365)\b", script)
    assert "--port='${RAY_PORT}'" in script and "--dashboard-port='${DASHBOARD_PORT}'" in script
    assert "--ray-client-server-port='${CLIENT_PORT}'" in script
    # Queued Jobs must not fail after Ray's default 15-minute start timeout.
    assert "export RAY_JOB_START_TIMEOUT_SECONDS='${RAY_JOB_START_TIMEOUT_SECONDS}'" in script
    assert script.count('${os_ports}"') == 2
    # Ray answers /api/version even when its job agent failed to bind.
    assert '"http://${head_ip}:${DASHBOARD_PORT}/api/jobs/"' in script


def test_ray_accepts_the_port_flags_the_harness_passes() -> None:
    parameter = pytest.importorskip("ray._private.parameter")
    script = START_CLUSTER.read_text(encoding="utf-8")
    head = re.search(r'^head_command="(.*?)"$', script, re.MULTILINE | re.DOTALL)
    os_ports = re.search(r'^os_ports="(.*)"$', script, re.MULTILINE)
    assert head is not None and os_ports is not None
    command = head.group(1).replace("${os_ports}", os_ports.group(1))
    picked = iter(range(20000, 20010))
    flags = {
        name.replace("-", "_"): next(picked) if value.startswith("'") else int(value)
        for name, value in re.findall(r"--([a-z-]+-port)=('\$\{\w+\}'|\d+)", command)
    }

    assert {"ray_client_server_port", "dashboard_agent_listen_port"} <= set(flags)
    # `ray start` runs this validation before it starts anything.
    parameter.RayParams(**flags)


def test_ray_starts_through_python_because_images_may_not_link_the_ray_script() -> None:
    script = START_CLUSTER.read_text(encoding="utf-8")

    assert "exec ray " not in script
    assert script.count("exec python -m ray.scripts.scripts start") == 2


def test_harness_is_explicitly_test_only_and_external_to_nvflow() -> None:
    start = START_CLUSTER.read_text(encoding="utf-8")
    launcher = CLUSTERS.read_text(encoding="utf-8")

    assert "TEST INFRASTRUCTURE ONLY" in start
    assert "TEST INFRASTRUCTURE ONLY" in launcher
    assert "nvflow run" not in start
    assert "pip install" not in start
    assert "git clone" not in start
    assert "I_UNDERSTAND_RAY_JOBS_HAS_NO_AUTH" in start
    assert "secret_unset" in start
    assert "import nvflow" not in launcher and "from nvflow" not in launcher


def test_each_cluster_starts_from_its_routed_image_and_slurm_settings(
    clusters: ModuleType, tmp_path: Path
) -> None:
    _, target, slurm = _targets(clusters, tmp_path)

    shape = {"nodes": 1, "time_limit": "01:00:00", "cpus": 16, "memory": "64G"}
    skills, skills_env = clusters.allocation(target.clusters["skills"], target, slurm, **shape)
    serving, serving_env = clusters.allocation(target.clusters["serving"], target, slurm, **shape)

    assert "--partition=cpu_short,cpu" in skills
    assert "--cpus-per-task=16" in skills and "--mem=64G" in skills
    assert "--exclusive" not in skills
    assert not any(argument.startswith("--gpus-per-node") for argument in skills)
    assert "--partition=gpu" in serving and "--gpus-per-node=8" in serving
    assert "--exclusive" in serving
    assert not any(argument.startswith(("--cpus-per-task", "--mem")) for argument in serving)
    assert skills_env["IMAGE"] == "/images/nemo-skills.sqsh"
    assert skills_env["IMAGE_ID"] == "nvflow-nemo-skills:v1.2.0"
    assert serving_env["IMAGE"] == "/images/vllm.sqsh"
    root = target.artifact_root
    assert serving_env["MOUNTS"] == (
        f"{root}:{root},/lustre/models:/hf_models,/lustre/outputs:/workspace/outputs"
    )
    assert "secret-token-value" not in repr((skills, skills_env, serving, serving_env))


def test_target_env_vars_reach_clusters_like_a_slurm_submission(
    clusters: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, target, _ = _targets(clusters, tmp_path)
    slurm_path = tmp_path / "with-env.yaml"
    slurm_path.write_text(
        SLURM_TARGET.split("env_vars:")[0]
        + "env_vars:\n"
        + "  - HF_HOME=/lustre/hf\n  - TIKTOKEN_CACHE_DIR\n  - UNSET_IN_SHELL\n"
        + "  - HF_TOKEN\n  - NVIDIA_API_KEY=listed-key\n  - ROLE=not-the-cluster\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "/lustre/tiktoken")
    monkeypatch.setenv("HF_TOKEN", "shell-token")
    monkeypatch.delenv("UNSET_IN_SHELL", raising=False)
    slurm = clusters.load_slurm_settings(slurm_path)
    shape = {"nodes": 1, "time_limit": "01:00:00", "cpus": 4, "memory": "16G"}

    _, withheld = clusters.allocation(target.clusters["skills"], target, slurm, **shape)
    _, passed = clusters.allocation(
        target.clusters["skills"], target, slurm, **shape, pass_credentials=True
    )

    assert withheld["HF_HOME"] == "/lustre/hf"
    assert withheld["TIKTOKEN_CACHE_DIR"] == "/lustre/tiktoken"
    assert withheld["ROLE"] == "skills"
    assert "UNSET_IN_SHELL" not in withheld
    assert not {"HF_TOKEN", "NVIDIA_API_KEY"} & set(withheld)
    assert withheld["NVFLOW_TEST_PASS_CREDENTIALS"] == ""
    assert (passed["HF_TOKEN"], passed["NVIDIA_API_KEY"]) == ("shell-token", "listed-key")
    assert passed["NVFLOW_TEST_PASS_CREDENTIALS"] == "1"

    # NeMo-Skills reads env_vars only as a list of NAME=value or NAME entries.
    slurm_path.write_text(
        SLURM_TARGET.split("env_vars:")[0] + "env_vars:\n  HF_HOME: /lustre/hf\n",
        encoding="utf-8",
    )
    with pytest.raises(clusters.HarnessError, match="env_vars must be a list"):
        clusters.load_slurm_settings(slurm_path)


@pytest.mark.parametrize(("pass_credentials", "expected"), [("", "unset HF_TOKEN ;"), ("1", "")])
def test_start_script_withholds_credentials_unless_passed(
    pass_credentials: str, expected: str
) -> None:
    script = START_CLUSTER.read_text(encoding="utf-8")
    function = re.search(r"^credential_unset\(\) \{\n.*?^\}\n", script, re.MULTILINE | re.DOTALL)
    assert function is not None

    result = subprocess.run(
        ["bash", "-c", f"{function.group(0)}credential_unset"],
        env={
            "PATH": os.environ["PATH"],
            "HF_HOME": "/lustre/hf",
            "HF_TOKEN": "secret",
            "NVFLOW_TEST_PASS_CREDENTIALS": pass_credentials,
        },
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout == expected


def test_endpoints_file_satisfies_nvflow_target_resolution(
    clusters: ModuleType, tmp_path: Path
) -> None:
    ray_path, target, _ = _targets(clusters, tmp_path)
    _publish(clusters, target, "skills", "10.0.0.1", "101")
    _publish(clusters, target, "serving", "10.0.0.2", "102")

    endpoints = clusters.write_endpoints(target)
    environment = {
        name: value
        for name, _, value in (
            token.partition("=") for token in shlex.split(endpoints.read_text(encoding="utf-8"))
        )
        if value
    }

    resolved = resolve_target(TargetBinding.resolve("cw_ray"), ray_path, environ=environment)
    assert resolved.ray is not None
    assert resolved.ray.clusters["skills"].jobs_url == "http://10.0.0.1:8265"
    assert resolved.ray.clusters["serving"].controller_jobs_url == "http://10.0.0.2:8265"


def test_start_cancels_every_submitted_cluster_when_one_fails(
    clusters: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, target, slurm = _targets(clusters, tmp_path)
    monkeypatch.setenv("NVFLOW_INSECURE_TEST_ENDPOINT_ACK", clusters.ACKNOWLEDGEMENT)
    job_ids = iter(["101", "102"])
    monkeypatch.setattr(clusters, "_submit", lambda command, environment: next(job_ids))
    monkeypatch.setattr(clusters, "_queue_status", lambda job_id: ("", ""))
    cancelled: list[str] = []
    monkeypatch.setattr(clusters, "_cancel", cancelled.append)

    with pytest.raises(clusters.HarnessError, match="ended before Ray started"):
        clusters.start(
            target,
            slurm,
            list(target.clusters.values()),
            nodes=1,
            time_limit="01:00:00",
            cpus=16,
            memory="64G",
            timeout=60,
        )

    assert cancelled == ["101", "102"]
    assert not list(target.state_dir.glob("*.job"))


def test_start_reports_a_cluster_that_dies_while_another_is_still_queued(
    clusters: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, target, slurm = _targets(clusters, tmp_path)
    monkeypatch.setenv("NVFLOW_INSECURE_TEST_ENDPOINT_ACK", clusters.ACKNOWLEDGEMENT)
    job_ids = iter(["101", "102"])
    monkeypatch.setattr(clusters, "_submit", lambda command, environment: next(job_ids))
    monkeypatch.setattr(
        clusters, "_queue_status", {"101": ("PENDING", "Priority"), "102": ("", "")}.get
    )
    cancelled: list[str] = []
    monkeypatch.setattr(clusters, "_cancel", cancelled.append)

    with pytest.raises(clusters.HarnessError, match="serving: Slurm job 102 ended"):
        clusters.start(
            target,
            slurm,
            list(target.clusters.values()),
            nodes=1,
            time_limit="01:00:00",
            cpus=16,
            memory="64G",
            timeout=60,
        )

    assert cancelled == ["101", "102"]


def test_start_prints_each_queue_state_change_while_it_waits(
    clusters: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, target, _ = _targets(clusters, tmp_path)
    polls = iter([("PENDING", "Priority"), ("PENDING", "Priority"), ("RUNNING", "None")])

    def status(_job_id: str) -> tuple[str, str]:
        state = next(polls)
        if state[0] == "RUNNING":
            _publish(clusters, target, "skills", "10.0.0.1", "101")
        return state

    monkeypatch.setattr(clusters, "_queue_status", status)
    monkeypatch.setattr(clusters.time, "sleep", lambda _seconds: None)

    clusters._wait_for_endpoints(target, [(target.clusters["skills"], "101")], float("inf"))

    report = capsys.readouterr().out
    assert report.count("skills     PENDING (Priority)") == 1
    assert "skills     RUNNING, starting Ray" in report


def test_start_requires_the_insecure_endpoint_acknowledgement(
    clusters: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, target, slurm = _targets(clusters, tmp_path)
    monkeypatch.delenv("NVFLOW_INSECURE_TEST_ENDPOINT_ACK", raising=False)

    with pytest.raises(clusters.HarnessError, match="no authentication"):
        clusters.start(
            target, slurm, [], nodes=1, time_limit="01:00:00", cpus=16, memory="64G", timeout=60
        )


def test_check_reports_each_cluster_readiness_from_this_host(
    clusters: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, target, _ = _targets(clusters, tmp_path)
    _publish(clusters, target, "skills", "10.0.0.1", "101")
    _publish(clusters, target, "serving", "10.0.0.2", "102")
    monkeypatch.setattr(clusters, "_queue_state", lambda job_id: "RUNNING")
    versions = {"http://10.0.0.1:8265": "2.58.0", "http://10.0.0.2:8265": "2.56.1"}
    monkeypatch.setattr(clusters, "_ray_version", versions.__getitem__)

    assert not clusters.check(target, list(target.clusters.values()))
    report = capsys.readouterr().out
    assert "skills     job 101" in report and "ready" in report
    assert "Ray 2.56.1, expected 2.58.0" in report


def test_selected_clusters_must_be_declared_and_publishable(
    clusters: ModuleType, tmp_path: Path
) -> None:
    ray_path, target, _ = _targets(clusters, tmp_path)

    with pytest.raises(clusters.HarnessError, match="unknown cluster training"):
        clusters.select(target, ["training"])
    ray_path.write_text(
        ray_path.read_text(encoding="utf-8").replace(
            "${oc.env:RAY_SKILLS_JOBS_URL}", "http://10.0.0.1:8265"
        ),
        encoding="utf-8",
    )
    with pytest.raises(clusters.HarnessError, match="oc.env:NAME"):
        clusters.load_ray_target(ray_path)
