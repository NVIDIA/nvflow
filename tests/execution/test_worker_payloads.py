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
"""Files written on workers from embedded payloads."""

from __future__ import annotations

import subprocess
from pathlib import Path

from nvflow.core.code_root import CODE_ROOT_TOKEN
from nvflow.execution.runtime import write_payload_command


def test_payload_paths_follow_the_worker_code_root(tmp_path: Path) -> None:
    content = (
        f"judge: {CODE_ROOT_TOKEN}/projects/finance/prompts/judge.txt\n"
        "data: /nemo_run/code/projects/finance/data.jsonl\n"
        "limit: ${policy.max_total_sequence_length}\n"
    )
    target = tmp_path / "config.yaml"
    command = write_payload_command(content, str(target))

    # Engines rewrite the token in command text before the worker could.
    assert CODE_ROOT_TOKEN not in command
    subprocess.run(
        ["bash", "-c", command],
        env={"PATH": "/usr/bin:/bin", "NVFLOW_CODE_ROOT": "/code"},
        check=True,
    )

    assert target.read_text() == (
        "judge: /code/projects/finance/prompts/judge.txt\n"
        "data: /code/projects/finance/data.jsonl\n"
        "limit: ${policy.max_total_sequence_length}\n"
    )
