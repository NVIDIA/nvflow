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
"""GRPO training consumes the shared judge endpoint contract."""

from __future__ import annotations

import copy
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.requires_nemo
pytest.importorskip("nemo_skills")

from nvflow.integrations.nemo_gym import runtime_binding  # noqa: E402
from nvflow.stages.nemo_rl.grpo import GRPOStage, PreparedGRPOConfig  # noqa: E402


def test_policy_as_judge_route_is_resolved_without_authored_topology() -> None:
    config = {
        "environments": {
            "finance": {
                "config_paths": [
                    "resources_servers/finance_sec_search/configs/finance_sec_search.yaml"
                ],
                "agent_name": "finance_agent",
                "gym": {"finance_service": {"api_key": "configured"}},
            }
        }
    }
    nemo_rl_config: dict[str, Any] = {"env": {"nemo_gym": {}}}

    stage = GRPOStage()
    mode, job = stage._inject_judge_config(
        config,
        nemo_rl_config,
        "/outputs/training",
        {"containers": {}},
    )
    binding_commands = stage._configure_runtime_bindings(
        config,
        nemo_rl_config,
        "/outputs/training",
        mode,
    )

    assert mode == "policy_as_judge"
    assert job is None
    assert nemo_rl_config["env"]["nemo_gym"]["config_paths"] == [
        "/outputs/training/gym-bindings/finance.yaml"
    ]
    assert len(binding_commands) == 1
    request = runtime_binding._parse_request(shlex.split(binding_commands[0])[-1])
    assert request["config_overrides"] == {"finance_service": {"api_key": "configured"}}


def test_native_rollout_judge_maps_to_managed_training_service() -> None:
    config = {
        "vllm_container": "vllm",
        "environments": {
            "finance_sec_search": {
                "config_paths": [
                    "resources_servers/finance_sec_search/configs/finance_sec_search.yaml"
                ],
                "agent_name": "finance_agent",
                "models": {
                    "judge": {
                        "deployment": "gym",
                        "model": "/models/judge",
                        "num_gpus": 2,
                        "gym_model": {
                            "uses_reasoning_parser": True,
                            "chat_template_kwargs": {"reasoning_effort": "high"},
                            "vllm_serve_env_vars": {"VLLM_RAY_DP_PACK_STRATEGY": "strict"},
                            "vllm_serve_kwargs": {
                                "tensor_parallel_size": 2,
                                "max_model_len": 32768,
                                "hf_config_overrides": {"rope_scaling": {"factor": 2.0}},
                            },
                        },
                    }
                },
            }
        },
    }
    nemo_rl_config: dict[str, Any] = {"env": {"nemo_gym": {}}}

    stage = GRPOStage()
    mode, job = stage._inject_judge_config(
        config,
        nemo_rl_config,
        "/outputs/training",
        {"containers": {"vllm": "/images/vllm.sqsh"}},
        judge_port=12345,
    )
    binding_commands = stage._configure_runtime_bindings(
        config,
        nemo_rl_config,
        "/outputs/training",
        mode,
    )

    assert mode == "local_vllm"
    assert job is not None
    assert job["num_gpus"] == 2
    assert job["container"] == "/images/vllm.sqsh"
    assert "--model /models/judge" in job["server_cmd"]
    assert "--max-model-len 32768" in job["server_cmd"]
    assert """--hf-overrides '{"rope_scaling":{"factor":2.0}}'""" in job["server_cmd"]
    assert (
        "env NVFLOW_SERVER_ENV=VLLM_RAY_DP_PACK_STRATEGY VLLM_RAY_DP_PACK_STRATEGY=strict python3"
        in job["server_cmd"]
    )
    gym = nemo_rl_config["env"]["nemo_gym"]
    assert gym["judge_model"]["responses_api_models"]["vllm_model"]["model"] == "/models/judge"
    assert gym["judge_model"]["responses_api_models"]["vllm_model"]["chat_template_kwargs"] == {
        "reasoning_effort": "high"
    }
    assert gym["config_paths"][-1] == "/outputs/training/gym-bindings/finance_sec_search.yaml"
    assert len(binding_commands) == 1
    assert "nvflow.integrations.nemo_gym.runtime_binding" in binding_commands[0]


def test_managed_judge_allocates_and_publishes_a_runtime_port(tmp_path: Path) -> None:
    config = {
        "vllm_container": "vllm",
        "environments": {
            "finance": {
                "config_paths": ["resources_servers/finance/configs/finance.yaml"],
                "agent_name": "finance_agent",
                "models": {
                    "judge": {
                        "deployment": "nvflow",
                        "model": "/models/judge",
                        "num_gpus": 1,
                        "gym_model": {"vllm_serve_kwargs": {}},
                    }
                },
            }
        },
    }
    _mode, job = GRPOStage()._inject_judge_config(
        config,
        {"env": {"nemo_gym": {}}},
        str(tmp_path),
        {"containers": {"vllm": "/images/vllm.sqsh"}},
        judge_port=None,
    )
    assert job is not None
    command = job["server_cmd"]
    assert "${runtime:judge_port}" not in command
    assert '--port "$JUDGE_PORT"' in command

    package = tmp_path / "nvflow" / "integrations" / "model_servers"
    package.mkdir(parents=True)
    for parent in (tmp_path / "nvflow", tmp_path / "nvflow" / "integrations", package):
        (parent / "__init__.py").touch()
    (package / "serve.py").write_text(
        """import os
import sys
from pathlib import Path

port = sys.argv[sys.argv.index("--port") + 1]
Path(os.environ["CAPTURED_JUDGE_PORT"]).write_text(port, encoding="utf-8")
""",
        encoding="utf-8",
    )
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    nvidia_smi = binary_dir / "nvidia-smi"
    nvidia_smi.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    nvidia_smi.chmod(0o755)
    captured_port = tmp_path / "captured-port"
    environment = {
        **os.environ,
        "CAPTURED_JUDGE_PORT": str(captured_port),
        "NVFLOW_CODE_ROOT": str(tmp_path),
        "PATH": f"{binary_dir}:{os.environ['PATH']}",
        "PYTHONPATH": "",
    }

    subprocess.run(["bash", "-c", command], env=environment, check=True)

    published = Path(job["host_file"]).read_text(encoding="utf-8").strip()
    host, port = published.rsplit(":", 1)
    assert host
    assert port.isdigit()
    assert int(port) > 0
    assert captured_port.read_text(encoding="utf-8") == port


def _managed_judge(tmp_path: Path, nodes: int, serve_kwargs: dict[str, Any]) -> dict[str, Any]:
    judge = {
        "deployment": "nvflow",
        "model": "/models/judge",
        "num_gpus": 4,
        "server_nodes": nodes,
        "gym_model": {"vllm_serve_kwargs": serve_kwargs},
    }
    config = {
        "vllm_container": "vllm",
        "environments": {
            "finance": {
                "config_paths": ["resources_servers/finance/configs/finance.yaml"],
                "agent_name": "finance_agent",
                "models": {"judge": judge},
            }
        },
    }
    _mode, job = GRPOStage()._inject_judge_config(
        config,
        {"env": {"nemo_gym": {}}},
        str(tmp_path),
        {"containers": {"vllm": "/images/vllm.sqsh"}},
        judge_port=None,
    )
    assert job is not None
    return job


@pytest.mark.parametrize(("backend", "without_ray"), [(None, False), ("mp", True)])
def test_a_multi_node_managed_judge_runs_on_its_own_backend(
    tmp_path: Path,
    backend: str | None,
    without_ray: bool,
) -> None:
    serve_kwargs: dict[str, Any] = {"tensor_parallel_size": 8}
    if backend is not None:
        serve_kwargs["distributed_executor_backend"] = backend

    command = _managed_judge(tmp_path, 2, serve_kwargs)["server_cmd"]

    assert "--num_nodes 2" in command
    assert ("--node_rank ${SLURM_PROCID:-0}" in command) is without_ray


@pytest.mark.parametrize("nodes", [1, 2])
def test_only_a_judges_first_node_publishes_its_address(tmp_path: Path, nodes: int) -> None:
    job = _managed_judge(tmp_path, nodes, {"distributed_executor_backend": "mp"})
    prepared = PreparedGRPOConfig(
        nemo_rl_config={},
        run_name="run",
        output_dir=str(tmp_path),
        expname="exp",
        hf_model_name="model",
        num_nodes=1,
        num_gpus=4,
        backend="megatron",
        judge_job_info=job,
    )

    _train, judge_command, host_file = GRPOStage._judge_pair_commands(prepared, "train", 0, 0)

    first_node = 'if [ "${SLURM_PROCID:-0}" = 0 ]; then '
    assert judge_command.startswith(
        f"{first_node}rm -f {host_file}; fi && " if nodes > 1 else "rm -f"
    )
    assert (
        f'{first_node}printf "%s:%s\\n" "$SERVER_HOST" "$JUDGE_PORT" > {host_file}; fi'
        in judge_command
    ) is (nodes > 1)


def test_external_judge_maps_without_scheduling_a_service() -> None:
    config = {
        "environments": {
            "equivalence_llm_judge": {
                "config_paths": [
                    "resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml"
                ],
                "agent_name": "equivalence_llm_judge_simple_agent",
                "models": {
                    "judge": {
                        "deployment": "external",
                        "model_type": "openai_model",
                        "model": "provider/judge",
                        "base_url": "https://judge.example/v1",
                        "api_key_env": "JUDGE_API_KEY",
                    }
                },
            }
        }
    }
    nemo_rl_config: dict[str, Any] = {"env": {"nemo_gym": {}}}

    mode, job = GRPOStage()._inject_judge_config(
        config,
        nemo_rl_config,
        "/outputs/training",
        {"containers": {"vllm": "/images/vllm.sqsh"}},
    )

    assert mode == "openai"
    assert job is None
    hosted = nemo_rl_config["env"]["nemo_gym"]["judge_model"]["responses_api_models"][
        "openai_model"
    ]
    assert hosted["openai_model"] == "provider/judge"
    assert hosted["openai_base_url"] == "https://judge.example/v1"


def test_inference_provider_judge_keeps_native_gym_adapter() -> None:
    config = {
        "environments": {
            "equivalence_llm_judge": {
                "config_paths": [
                    "resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml"
                ],
                "agent_name": "equivalence_llm_judge_simple_agent",
                "models": {
                    "judge": {
                        "deployment": "external",
                        "model_type": "inference_provider",
                        "model": "provider/judge",
                        "base_url": "https://judge.example/v1",
                        "api_key_env": "JUDGE_API_KEY",
                    }
                },
            }
        }
    }
    nemo_rl_config: dict[str, Any] = {"env": {"nemo_gym": {}}}

    mode, job = GRPOStage()._inject_judge_config(
        config,
        nemo_rl_config,
        "/outputs/training",
        {"containers": {"vllm": "/images/vllm.sqsh"}},
    )

    assert mode == "inference_provider"
    assert job is None
    hosted = nemo_rl_config["env"]["nemo_gym"]["judge_model"]["responses_api_models"][
        "inference_provider"
    ]
    assert hosted == {
        "entrypoint": "app.py",
        "base_url": "https://judge.example/v1",
        "api_key": "$JUDGE_API_KEY",
        "model": "provider/judge",
    }


def test_combined_training_routes_every_environment_to_shared_judge() -> None:
    judge = {
        "deployment": "external",
        "model_type": "openai_model",
        "model": "provider/judge",
        "base_url": "https://judge.example/v1",
    }
    config = {
        "environments": {
            "first": {
                "config_paths": ["resources_servers/first/configs/first.yaml"],
                "agent_name": "first_agent",
                "models": {"judge": copy.deepcopy(judge)},
            },
            "second": {
                "config_paths": ["resources_servers/second/configs/second.yaml"],
                "agent_name": "second_agent",
                "models": {"judge": copy.deepcopy(judge)},
            },
        }
    }
    nemo_rl_config: dict[str, Any] = {"env": {"nemo_gym": {}}}

    stage = GRPOStage()
    mode, job = stage._inject_judge_config(
        config,
        nemo_rl_config,
        "/outputs/training",
        {"containers": {"vllm": "/images/vllm.sqsh"}},
    )
    binding_commands = stage._configure_runtime_bindings(
        config,
        nemo_rl_config,
        "/outputs/training",
        mode,
    )

    assert mode == "openai"
    assert job is None
    gym = nemo_rl_config["env"]["nemo_gym"]
    assert gym["config_paths"][-2:] == [
        "/outputs/training/gym-bindings/first.yaml",
        "/outputs/training/gym-bindings/second.yaml",
    ]
    assert len(binding_commands) == 2


def test_combined_training_rejects_conflicting_judges() -> None:
    config = {
        "environments": {
            "first": {
                "models": {
                    "judge": {
                        "deployment": "external",
                        "model_type": "openai_model",
                        "model": "judge-a",
                        "base_url": "https://judge.example/v1",
                    }
                }
            },
            "second": {
                "models": {
                    "judge": {
                        "deployment": "external",
                        "model_type": "openai_model",
                        "model": "judge-b",
                        "base_url": "https://judge.example/v1",
                    }
                }
            },
        }
    }

    with pytest.raises(ValueError, match="judge settings differ across environments"):
        GRPOStage()._inject_judge_config(
            config,
            {"env": {"nemo_gym": {}}},
            "/outputs/training",
            {"containers": {"vllm": "/images/vllm.sqsh"}},
        )


def test_managed_training_judge_honors_container_and_rejects_other_engines() -> None:
    endpoint = {
        "deployment": "nvflow",
        "server_type": "vllm",
        "container": "judge-vllm",
        "model": "/models/judge",
        "num_gpus": 2,
        "gym_model": {"vllm_serve_kwargs": {"tensor_parallel_size": 2}},
    }
    config = {
        "environments": {
            "finance": {
                "config_paths": ["resources_servers/finance/configs/finance.yaml"],
                "agent_name": "finance_agent",
                "models": {"judge": endpoint},
            }
        }
    }

    _mode, job = GRPOStage()._inject_judge_config(
        config,
        {"env": {"nemo_gym": {}}},
        "/outputs/training",
        {"containers": {"judge-vllm": "/images/judge.sqsh"}},
        judge_port=12345,
    )

    assert job is not None
    assert job["container"] == "/images/judge.sqsh"

    endpoint["server_type"] = "sglang"
    with pytest.raises(ValueError, match="supports only vLLM"):
        GRPOStage()._inject_judge_config(
            config,
            {"env": {"nemo_gym": {}}},
            "/outputs/training",
            {"containers": {}},
        )
