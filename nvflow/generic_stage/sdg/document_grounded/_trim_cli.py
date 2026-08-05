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
"""Trim DG-SDG JSONL files in place to a per-stage allowlist.

Invoked at every DG-SDG stage boundary (either as part of the producing job's
``postprocess_cmd`` for Gym stages, or chained with ``&&`` to the CPU command
for non-Gym stages). Drops every JSON key not present in ``--keep_fields``,
including the cross-stage scratch listed in ``_schemas.ALWAYS_DROP``.

The trim is in-place via a ``<path>.trim_tmp`` rename, so partial failures
don't leave a half-written file at the canonical path.

Usage::

    python -m nvflow.generic_stage.sdg.document_grounded._trim_cli \\
        --paths /abs/path/to/file.jsonl /abs/path/to/dir \\
        --keep_fields context problem generation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def trim_file(path: Path, keep: set[str]) -> tuple[int, int]:
    """Rewrite ``path`` in place keeping only top-level keys in ``keep``.

    Returns ``(records_processed, field_instances_dropped)``.
    """
    tmp = path.with_suffix(path.suffix + ".trim_tmp")
    rec_count = 0
    drop_count = 0
    with path.open() as fin, tmp.open("w") as fout:
        for line in fin:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            slim = {k: v for k, v in record.items() if k in keep}
            drop_count += len(record) - len(slim)
            fout.write(json.dumps(slim) + "\n")
            rec_count += 1
    tmp.replace(path)
    return rec_count, drop_count


def _resolve_paths(args_paths: list[str]) -> list[Path]:
    """Expand globs and directories into a flat list of JSONL files."""
    matched: list[Path] = []
    for raw in args_paths:
        candidate = Path(raw)
        if "*" in raw or "?" in raw:
            matched.extend(sorted(candidate.parent.glob(candidate.name)))
        elif candidate.is_dir():
            # Unlike shell globs, pathlib's glob("*.jsonl") also matches
            # dotfiles (e.g. ``.responses_api_input.jsonl``, the internal
            # render/join cache written by responses_api.render_and_convert).
            # That file is never a stage *output* -- trimming it strips
            # ``responses_create_params`` (ALWAYS_DROP), which enrich_rollouts'
            # join key is computed from, silently poisoning the cache for any
            # future re-merge. Exclude dotfiles to match intended shell-glob
            # semantics and keep internal caches out of stage-boundary trims.
            matched.extend(
                sorted(p for p in candidate.glob("*.jsonl") if not p.name.startswith("."))
            )
        else:
            matched.append(candidate)
    return matched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="JSONL files, directories (globbed as *.jsonl), or glob patterns.",
    )
    parser.add_argument(
        "--keep_fields",
        nargs="+",
        required=True,
        help="Top-level JSON keys to keep. Everything else is dropped.",
    )
    args = parser.parse_args(argv)

    keep = set(args.keep_fields)
    files = _resolve_paths(args.paths)
    if not files:
        print(
            f"[trim] no files matched from {args.paths!r}; nothing to do",
            file=sys.stderr,
        )
        return 0

    for f in files:
        if not f.exists():
            print(f"[trim] {f}: missing, skipping", file=sys.stderr)
            continue
        n, d = trim_file(f, keep)
        print(f"[trim] {f}: {n} records, dropped {d} field-instances")
    return 0


if __name__ == "__main__":
    sys.exit(main())
