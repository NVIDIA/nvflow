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
"""Deterministically and atomically shuffle one JSONL file in place."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import os
import sys
import tempfile
from pathlib import Path

from nvflow.utils import setup_logger

logger = setup_logger(__name__)


def shuffle_file(
    input_file: str | Path,
    random_seed: int,
    *,
    chunk_bytes: int = 64 * 1024 * 1024,
) -> int:
    """Shuffle JSONL deterministically with bounded memory and atomic replacement."""
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be greater than zero")

    count = 0
    prefix_length = 32 + 16 + 1  # BLAKE2 hex key + row index + tab.
    with tempfile.TemporaryDirectory(prefix=".shuffle-jsonl.", dir=path.parent) as directory:
        temporary_dir = Path(directory)
        chunks: list[Path] = []
        entries: list[tuple[bytes, int, bytes]] = []
        buffered = 0

        def flush_chunk() -> None:
            nonlocal entries, buffered
            if not entries:
                return
            entries.sort(key=lambda item: (item[0], item[1]))
            chunk = temporary_dir / f"chunk-{len(chunks):06d}"
            with chunk.open("wb") as stream:
                for key, index, line in entries:
                    stream.write(key.hex().encode())
                    stream.write(f"{index:016x}\t".encode())
                    stream.write(line)
            chunks.append(chunk)
            entries = []
            buffered = 0

        seed = str(random_seed).encode()
        with path.open("rb") as source:
            for index, line in enumerate(source):
                normalized = line if line.endswith(b"\n") else line + b"\n"
                key = hashlib.blake2b(
                    seed + b":" + str(index).encode(),
                    digest_size=16,
                ).digest()
                entries.append((key, index, normalized))
                buffered += len(normalized) + prefix_length
                count += 1
                if buffered >= chunk_bytes:
                    flush_chunk()
        flush_chunk()

        descriptor, raw_temporary = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        os.close(descriptor)
        temporary = Path(raw_temporary)
        streams = [chunk.open("rb") for chunk in chunks]
        try:
            with temporary.open("wb") as destination:
                for encoded in heapq.merge(*streams):
                    destination.write(encoded[prefix_length:])
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, path)
        finally:
            for stream in streams:
                stream.close()
            temporary.unlink(missing_ok=True)
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()
    try:
        count = shuffle_file(args.input_file, args.random_seed)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    logger.info(
        "Shuffled %d rows with seed=%d -> %s",
        count,
        args.random_seed,
        args.input_file,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
