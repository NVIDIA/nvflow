#!/usr/bin/env python3
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
"""Write the materialized tracked-file list used by Docker snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath

from nvflow.project.snapshot import materialized_files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        type=PurePosixPath,
        help="Tracked file or directory to omit; may be repeated.",
    )
    args = parser.parse_args()

    paths = (
        path
        for path in materialized_files(args.root)
        if not any(path == excluded or path.is_relative_to(excluded) for excluded in args.exclude)
    )
    args.output.write_bytes(b"".join(f"{path.as_posix()}\0".encode() for path in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
