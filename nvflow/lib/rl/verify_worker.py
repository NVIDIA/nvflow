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
"""Re-judge rollouts via NeMo-Gym ServerClient.

Uses the same ServerClient API that every NeMo-Gym component uses
internally, so port discovery, URL routing, and retries are handled
automatically.

Must run inside the Gym venv (``nemo_gym`` must be importable).

Usage:
    python -m nvflow.lib.rl.verify_worker \\
        <input.jsonl> <output.jsonl> \\
        <head_host> <head_port> <environment_name> <num_parallel>
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from nemo_gym.config_types import BaseServerConfig
from nemo_gym.server_utils import ServerClient
from tqdm.asyncio import tqdm


def _wait_for_server_client(
    head_host: str,
    head_port: int,
    timeout: int = 600,
    poll_interval: int = 5,
) -> ServerClient:
    """Poll until the head server responds and returns a ServerClient."""
    head_config = BaseServerConfig(host=head_host, port=head_port)
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            return ServerClient.load_from_global_config(head_config)
        except Exception as e:
            last_err = e
            time.sleep(poll_interval)
    raise RuntimeError(
        f"Could not connect to head server at {head_host}:{head_port} after {timeout}s: {last_err}"
    )


async def _wait_for_verify_endpoint(
    client: ServerClient,
    environment_name: str,
    timeout: int = 600,
    poll_interval: int = 5,
) -> None:
    """Poll the /verify endpoint until it responds (non-404)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = await client.post(
                server_name=environment_name,
                url_path="/verify",
                json={},
            )
            if resp.status != 404:
                print(f"  /verify endpoint ready (HTTP {resp.status})")
                return
        except Exception:
            pass
        await asyncio.sleep(poll_interval)
    raise RuntimeError(f"/verify endpoint on {environment_name} not available after {timeout}s")


async def verify_rollouts(
    input_file: str,
    output_file: str,
    head_host: str,
    head_port: int,
    environment_name: str,
    num_parallel: int,
) -> None:
    with open(input_file) as f:
        rollouts = [json.loads(line) for line in f if line.strip()]

    if not rollouts:
        print("WARNING: No rollouts found in input file.")
        Path(output_file).write_text("")
        return

    print(f"Connecting to NeMo-Gym head server at {head_host}:{head_port} ...")
    client = _wait_for_server_client(head_host, head_port)
    print(f"  Connected. Waiting for {environment_name} /verify endpoint ...")
    await _wait_for_verify_endpoint(client, environment_name)

    print(f"Re-judging {len(rollouts)} rollouts via {environment_name} /verify")

    max_retries = 3
    retry_base_delay = 2.0
    semaphore = asyncio.Semaphore(num_parallel)
    results: list[dict | None] = [None] * len(rollouts)
    error_count = 0

    async def _verify(idx: int, rollout: dict) -> None:
        nonlocal error_count
        verify_request = {
            "responses_create_params": rollout["responses_create_params"],
            "response": rollout["response"],
        }
        for key in (
            "expected_answer",
            "uuid",
            "options",
            "metadata",
            "template_metadata",
        ):
            if key in rollout:
                verify_request[key] = rollout[key]

        last_status = 0
        last_body = ""
        for attempt in range(max_retries + 1):
            try:
                async with semaphore:
                    resp = await client.post(
                        server_name=environment_name,
                        url_path="/verify",
                        json=verify_request,
                    )
                    last_status = resp.status
                    if resp.status == 200:
                        result = await resp.json()
                        updated = dict(rollout)
                        updated["reward"] = result.get("reward", rollout.get("reward", 0.0))
                        updated["judge_evaluations"] = result.get(
                            "judge_evaluations",
                            rollout.get("judge_evaluations", []),
                        )
                        results[idx] = updated
                        return
                    last_body = await resp.text()
            except Exception as e:
                last_status = 0
                last_body = str(e)

            if attempt < max_retries:
                delay = retry_base_delay * (2**attempt)
                await asyncio.sleep(delay)

        error_count += 1
        if error_count <= 10:
            print(
                f"ERROR: /verify returned {last_status} for idx={idx} "
                f"after {max_retries + 1} attempts: {last_body[:200]}"
            )
        elif error_count == 11:
            print("ERROR: suppressing further per-record error messages ...")
        results[idx] = None

    tasks = [_verify(i, r) for i, r in enumerate(rollouts)]
    await tqdm.gather(*tasks, desc="Verifying rollouts", miniters=10)

    succeeded = sum(1 for r in results if r is not None)
    with open(output_file, "w") as f:
        for r in results:
            if r is not None:
                f.write(json.dumps(r) + "\n")

    if error_count > 0:
        print(
            f"\nFATAL: {error_count}/{len(rollouts)} requests failed. "
            f"Only {succeeded} results written."
        )
        sys.exit(1)

    rewards = [r.get("reward", 0.0) for r in results if r is not None]
    if rewards:
        avg = sum(rewards) / len(rewards)
        print(f"  Average reward: {avg:.4f} ({len(rewards)} samples)")


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print(
            "Usage: python -m nvflow.lib.rl.verify_worker "
            "<input.jsonl> <output.jsonl> "
            "<head_host> <head_port> <environment_name> <num_parallel>"
        )
        sys.exit(1)

    _args = sys.argv[1:]
    # NeMo-Gym's aiohttp client lazily calls get_global_config_dict() which
    # invokes @hydra.main and parses sys.argv.  Wipe argv so Hydra sees no
    # overrides (our positional args are not Hydra overrides).
    sys.argv = [sys.argv[0]]

    asyncio.run(
        verify_rollouts(
            input_file=_args[0],
            output_file=_args[1],
            head_host=_args[2],
            head_port=int(_args[3]),
            environment_name=_args[4],
            num_parallel=int(_args[5]),
        )
    )
