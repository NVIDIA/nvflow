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
"""Render a prompt template into NeMo Gym rows with ``gym dataset render``.

The template's ``user`` and ``system`` keys are the prompt. Every other key,
such as ``tools`` and ``parallel_tool_calls``, is first stamped into each row's
``responses_create_params``, replacing any rendered ``input``; ``--row_fields``
sets fixed fields, such as a verifier, on every row, replacing the row's own.
Gym's ``_ng_*`` fields from an earlier collection are dropped, since a new
collection numbers its rows itself. ``NAME.done`` records the digests of the
input, template, row fields, and output, so a rerun with the same inputs skips
the work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

PROMPT_KEYS = ("user", "system")


def _evidence(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            rows += bool(line.strip())
    return {"sha256": digest.hexdigest(), "rows": rows}


def completion_path(output: Path) -> Path:
    return output.with_suffix(".done")


def _expected(
    source: Path, template: Path, row_fields: Mapping[str, Any], output: Path
) -> dict[str, Any]:
    fields = json.dumps(row_fields, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return {
        "schema_version": 1,
        "status": "complete",
        "input": _evidence(source),
        "template": _evidence(template)["sha256"],
        "row_fields": hashlib.sha256(fields).hexdigest(),
        "output": _evidence(output),
    }


def is_complete(source: Path, template: Path, row_fields: Mapping[str, Any], output: Path) -> bool:
    try:
        record = json.loads(completion_path(output).read_text(encoding="utf-8"))
        return bool(record == _expected(source, template, row_fields, output))
    except (OSError, json.JSONDecodeError):
        return False


def stamp(
    source: Path,
    template: Mapping[str, Any],
    row_fields: Mapping[str, Any],
    destination: Path,
) -> int:
    """Write rows with the row fields, the template's request parameters, and no input."""
    parameters = {key: value for key, value in template.items() if key not in PROMPT_KEYS}
    rows = 0
    with source.open(encoding="utf-8") as reader, destination.open("w", encoding="utf-8") as writer:
        for line in reader:
            if not line.strip():
                continue
            row = {
                key: value for key, value in json.loads(line).items() if not key.startswith("_ng_")
            }
            row.update(row_fields)
            request = {
                key: value
                for key, value in (row.get("responses_create_params") or {}).items()
                if key != "input"
            }
            row["responses_create_params"] = {**request, **parameters}
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1
    return rows


def render(
    source: Path,
    template_path: Path,
    output: Path,
    *,
    row_fields: Mapping[str, Any] | None = None,
    gym: str = "gym",
) -> int:
    """Stamp, render with Gym, check the row count, and publish ``NAME.done``."""
    row_fields = dict(row_fields or {})
    if is_complete(source, template_path, row_fields, output):
        print(f"{output} is complete; skipping")
        return _evidence(output)["rows"]
    if shutil.which(gym) is None:
        raise RuntimeError(f"{gym!r} is not on PATH; set the stage's installation_command")
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    if not isinstance(template, dict) or not template.get("user"):
        raise ValueError(f"{template_path} needs a 'user' prompt template")
    completion_path(output).unlink(missing_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    stamped = output.with_name(f".{output.name}.stamped.{token}")
    rendered = output.with_name(f".{output.name}.rendered.{token}")
    try:
        rows = stamp(source, template, row_fields, stamped)
        subprocess.run(
            [
                gym,
                "dataset",
                "render",
                f"+input_jsonl_fpath={stamped}",
                f"+prompt_config={template_path}",
                f"+output_jsonl_fpath={rendered}",
            ],
            check=True,
        )
        produced = _evidence(rendered)["rows"]
        if produced != rows:
            raise RuntimeError(f"gym dataset render wrote {produced} rows for {rows} inputs")
        os.replace(rendered, output)
    finally:
        stamped.unlink(missing_ok=True)
        rendered.unlink(missing_ok=True)
    marker = completion_path(output)
    temporary = marker.with_name(f".{marker.name}.{token}.tmp")
    try:
        temporary.write_text(
            json.dumps(_expected(source, template_path, row_fields, output), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, marker)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Rendered {rows:,} rows into {output}")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row_fields", default="{}", help="JSON fields added to every row")
    parser.add_argument("--gym", default="gym", help="Gym CLI executable")
    args = parser.parse_args(argv)
    row_fields = json.loads(args.row_fields)
    render(args.input, args.prompt, args.output, row_fields=row_fields, gym=args.gym)
    return 0


if __name__ == "__main__":
    sys.exit(main())
