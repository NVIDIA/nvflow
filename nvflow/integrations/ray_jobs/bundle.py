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
"""Build a run envelope while preserving an immutable worker bundle."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nvflow.project.packaging import WorkerBundle

RUN_SPEC_PATH = ".nvflow/run.json"
WORKER_BUNDLE_PATH = ".nvflow/worker-bundle.zip"


@dataclass(frozen=True, slots=True)
class ControllerBundle:
    """Per-run envelope submitted to the durable control cluster."""

    path: Path
    digest: str
    compressed_size: int


def build_controller_bundle(
    worker: WorkerBundle,
    run_spec: dict[str, Any],
    destination_dir: Path,
    *,
    max_compressed_bytes: int = 95 * 1024 * 1024,
) -> ControllerBundle:
    """Embed immutable code plus a redacted run spec and reusable child ZIP."""
    spec_bytes = json.dumps(
        run_spec,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    digest = hashlib.sha256()
    digest.update(worker.digest.encode())
    digest.update(b"\0")
    digest.update(spec_bytes)
    identity = digest.hexdigest()
    destination_dir.mkdir(parents=True, exist_ok=True)
    output = destination_dir / f"nvflow-controller-{identity[:20]}.zip"
    temporary = output.with_suffix(".zip.tmp")
    if not output.exists():
        try:
            with (
                zipfile.ZipFile(worker.path) as source,
                zipfile.ZipFile(
                    temporary,
                    mode="w",
                    compression=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                ) as target,
            ):
                for name in sorted(source.namelist()):
                    info = source.getinfo(name)
                    copied = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    copied.compress_type = zipfile.ZIP_DEFLATED
                    copied.create_system = info.create_system
                    copied.external_attr = info.external_attr
                    target.writestr(copied, source.read(name))
                _write(target, RUN_SPEC_PATH, spec_bytes)
                _write(target, WORKER_BUNDLE_PATH, worker.path.read_bytes())
            if temporary.stat().st_size > max_compressed_bytes:
                raise ValueError(
                    f"Ray controller bundle is too large: {temporary.stat().st_size} > "
                    f"{max_compressed_bytes} bytes"
                )
            temporary.replace(output)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    return ControllerBundle(
        path=output,
        digest=f"sha256:{identity}",
        compressed_size=output.stat().st_size,
    )


def _write(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o600 << 16
    archive.writestr(info, data)


__all__ = [
    "ControllerBundle",
    "RUN_SPEC_PATH",
    "WORKER_BUNDLE_PATH",
    "build_controller_bundle",
]
