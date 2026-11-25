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
import os
import subprocess
from pathlib import Path

import pytest

from nvflow.core.node_address import NODE_ADDRESS_COMMAND


def _fake_command(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


@pytest.mark.parametrize(
    ("resolved", "interfaces", "expected"),
    [
        (
            "169.254.3.1 STREAM gpu-node\n10.0.0.7 STREAM gpu-node\n",
            "169.254.3.1 10.0.0.7",
            "10.0.0.7",
        ),
        ("127.0.1.1 STREAM gpu-node\n", "169.254.3.1 10.0.0.5 fe80::1", "10.0.0.5"),
        (None, None, ""),
    ],
    ids=["link-local-first", "hostname-is-loopback", "no-usable-address"],
)
def test_node_address_skips_addresses_other_machines_cannot_reach(
    tmp_path: Path,
    resolved: str | None,
    interfaces: str | None,
    expected: str,
) -> None:
    _fake_command(
        tmp_path / "getent",
        "exit 2" if resolved is None else f"printf '{resolved}'",
    )
    _fake_command(
        tmp_path / "hostname",
        'if [ "${1:-}" != "-I" ]; then echo gpu-node; exit 0; fi\n'
        + ("exit 1" if interfaces is None else f"echo '{interfaces} '"),
    )

    result = subprocess.run(
        ["bash", "-c", f'set -euo pipefail; address=$({NODE_ADDRESS_COMMAND}); echo "$address"'],
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == expected
