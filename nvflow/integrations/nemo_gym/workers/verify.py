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

Input joins are indexed in a temporary SQLite database and request tasks are
created in bounded batches, so large repeated rollout corpora do not need to
fit in memory.

Must run inside the Gym venv (``nemo_gym`` must be importable).

With ``--unique-prompts``, every request starts the answer's last user message
with a unique line, so a judge prompt built from that message differs from any
earlier request's after the template's fixed text. An answer whose
verification Gym reports as failed is verified again with a new line; a reply
Gym scores without a failure, such as one with no rating, is kept. Saved
records keep the question as it was.

Usage:
    python -m nvflow.integrations.nemo_gym.workers.verify \\
        <materialized-inputs.jsonl> <rollouts.jsonl> <output.jsonl> <failures.jsonl> \\
        <head_host> <head_port> <resources_server_instance> <num_parallel> [--unique-prompts]
"""

import asyncio
import copy
import functools
import json
import math
import os
import sqlite3
import sys
import time
import uuid
from pathlib import Path

from nemo_gym.config_types import BaseServerConfig
from nemo_gym.server_utils import ServerClient
from tqdm import tqdm

from nvflow.utils import setup_logger

logger = setup_logger(__name__)


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
    resources_server_instance: str,
    timeout: int = 600,
    poll_interval: int = 5,
) -> None:
    """Poll the /verify endpoint until it responds (non-404)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = await client.post(
                server_name=resources_server_instance,
                url_path="/verify",
                json={},
            )
            if resp.status != 404:
                logger.info("  /verify endpoint ready (HTTP %d)", resp.status)
                return
        except Exception:
            pass
        await asyncio.sleep(poll_interval)
    raise RuntimeError(
        f"/verify endpoint on {resources_server_instance} not available after {timeout}s"
    )


def _last_question(request: dict) -> dict | None:
    """The last user message of *request* whose content is text, if any."""
    messages = (request.get("responses_create_params") or {}).get("input")
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if (
            isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
        ):
            return message
    return None


@functools.cache
def _warn_unmarked() -> None:
    logger.warning(
        "--unique-prompts: a request has no text user message to mark and is sent unchanged"
    )


def with_request_line(request: dict) -> tuple[dict, str]:
    """A copy of *request* whose question starts with a unique line, and that line.

    A request without a text user message comes back as it is, with no line.
    """
    if _last_question(request) is None:
        _warn_unmarked()
        return request, ""
    line = f"Request {uuid.uuid4().hex}"
    marked = {
        **request,
        "responses_create_params": copy.deepcopy(request["responses_create_params"]),
    }
    question = _last_question(marked)
    assert question is not None
    question["content"] = f"{line}\n\n{question['content']}"
    return marked, line


def without_request_line(record: dict, line: str) -> None:
    """Remove *line* from the question of *record*, which /verify echoed back."""
    if not line:
        return
    question = _last_question(record)
    prefix = f"{line}\n\n"
    if question is not None and question["content"].startswith(prefix):
        question["content"] = question["content"][len(prefix) :]


async def verify_rollouts(
    materialized_inputs_file: str,
    rollouts_file: str,
    output_file: str,
    failures_file: str,
    head_host: str,
    head_port: int,
    resources_server_instance: str,
    num_parallel: int,
    unique_prompts: bool = False,
) -> None:
    def key(row: dict) -> tuple[object, object]:
        return row["_ng_task_index"], row["_ng_rollout_index"]

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path = Path(failures_file)
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.write_text("", encoding="utf-8")
    index_path = output_path.with_name(f".{output_path.name}.join.sqlite3")
    index_path.unlink(missing_ok=True)
    database = sqlite3.connect(index_path)
    try:
        database.execute("PRAGMA journal_mode=OFF")
        database.execute("PRAGMA synchronous=OFF")
        database.execute("CREATE TABLE inputs (identity TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        database.execute("CREATE TABLE rollouts (identity TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        database.execute("CREATE TABLE completed (identity TEXT PRIMARY KEY)")

        def identity_token(row: dict) -> str:
            return json.dumps(key(row), ensure_ascii=False, separators=(",", ":"))

        def index_jsonl(table: str, path: str) -> int:
            count = 0
            with open(path) as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    try:
                        database.execute(
                            f"INSERT INTO {table} (identity, payload) VALUES (?, ?)",
                            (identity_token(row), line),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(
                            f"{path}:{line_number}: duplicate task/rollout identity"
                        ) from exc
                    count += 1
            return count

        materialized_count = index_jsonl("inputs", materialized_inputs_file)
        rollout_count = index_jsonl("rollouts", rollouts_file)
        if rollout_count == 0:
            logger.warning("No rollouts found in input file.")
            output_path.write_text("")
            return

        missing_rollouts = database.execute(
            "SELECT COUNT(*) FROM inputs LEFT JOIN rollouts USING (identity) "
            "WHERE rollouts.identity IS NULL"
        ).fetchone()[0]
        unexpected_rollouts = database.execute(
            "SELECT COUNT(*) FROM rollouts LEFT JOIN inputs USING (identity) "
            "WHERE inputs.identity IS NULL"
        ).fetchone()[0]
        if materialized_count != rollout_count or missing_rollouts or unexpected_rollouts:
            raise ValueError(
                "materialized inputs and rollouts have different task/rollout identities"
            )

        completed_count = 0
        if output_path.exists():
            with output_path.open() as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    try:
                        database.execute(
                            "INSERT INTO completed (identity) VALUES (?)",
                            (identity_token(row),),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(
                            f"{output_path}:{line_number}: duplicate completed "
                            "reverification identity"
                        ) from exc
                    completed_count += 1
        unexpected_completed = database.execute(
            "SELECT COUNT(*) FROM completed LEFT JOIN inputs USING (identity) "
            "WHERE inputs.identity IS NULL"
        ).fetchone()[0]
        if unexpected_completed:
            raise ValueError(
                f"reverification output contains {unexpected_completed} unexpected identities"
            )

        pending_count = rollout_count - completed_count
        logger.info(
            "Re-judging %d/%d rollouts via %s /verify",
            pending_count,
            rollout_count,
            resources_server_instance,
        )
        if pending_count == 0:
            return
        if num_parallel <= 0:
            raise ValueError("num_parallel must be greater than zero")

        logger.info("Connecting to NeMo-Gym head server at %s:%d ...", head_host, head_port)
        client = _wait_for_server_client(head_host, head_port)
        logger.info("  Connected. Waiting for %s /verify endpoint ...", resources_server_instance)
        await _wait_for_verify_endpoint(client, resources_server_instance)

        max_retries = 3
        retry_base_delay = 2.0
        error_count = 0

        async def _verify(
            row_key: tuple[object, object],
            materialized: dict,
            rollout: dict,
        ) -> dict | None:
            nonlocal error_count
            verify_request = {
                **materialized,
                "response": rollout["response"],
            }

            last_status = 0
            last_body = ""
            failed: dict | None = None
            for attempt in range(max_retries + 1):
                request, line = (
                    with_request_line(verify_request) if unique_prompts else (verify_request, "")
                )
                try:
                    resp = await client.post(
                        server_name=resources_server_instance,
                        url_path="/verify",
                        json=request,
                    )
                    last_status = resp.status
                    if resp.status == 200:
                        result = await resp.json()
                        if not isinstance(result, dict):
                            last_body = "response body must be a JSON object"
                            continue
                        if unique_prompts:
                            without_request_line(result, line)
                        updated = dict(materialized)
                        updated.update(result)
                        for field in ("_ng_task_index", "_ng_rollout_index", "agent_ref"):
                            if field in materialized:
                                updated[field] = materialized[field]
                        updated["response"] = rollout["response"]
                        if "ng_trajectory" in rollout:
                            updated["ng_trajectory"] = rollout["ng_trajectory"]
                        retry_failure = (
                            unique_prompts
                            and bool(updated.get("_ng_failure_class"))
                            and attempt < max_retries
                        )
                        if retry_failure:
                            failed = updated
                            last_body = f"verification failed: {updated['_ng_failure_class']}"
                        elif updated.get("_ng_no_persist") or updated.get("_ng_failure_class"):
                            return updated
                        else:
                            reward = result.get("reward")
                            if (
                                isinstance(reward, bool)
                                or not isinstance(reward, int | float)
                                or not math.isfinite(reward)
                            ):
                                last_body = (
                                    "successful /verify response requires a fresh finite reward"
                                )
                                continue
                            return updated
                    else:
                        last_body = await resp.text()
                except Exception as exc:
                    last_status = 0
                    last_body = str(exc)

                if attempt < max_retries:
                    delay = retry_base_delay * (2**attempt)
                    await asyncio.sleep(delay)

            if failed is not None:
                logger.warning(
                    "/verify still failed for identity=%r after %d attempts: %s",
                    row_key,
                    max_retries + 1,
                    last_body[:200],
                )
                return failed
            error_count += 1
            if error_count <= 10:
                logger.error(
                    "/verify returned %d for identity=%r after %d attempts: %s",
                    last_status,
                    row_key,
                    max_retries + 1,
                    last_body[:200],
                )
            elif error_count == 11:
                logger.error("Suppressing further per-record error messages ...")
            return None

        pending_rows = database.execute(
            "SELECT inputs.identity, inputs.payload, rollouts.payload "
            "FROM inputs JOIN rollouts USING (identity) "
            "LEFT JOIN completed USING (identity) "
            "WHERE completed.identity IS NULL ORDER BY inputs.rowid"
        )
        succeeded = completed_count
        new_rewards: list[float] = []
        verification_failure_count = 0
        progress = tqdm(total=pending_count, desc="Verifying rollouts", miniters=10)
        try:
            with output_path.open("a") as output, failures_path.open("a") as failures:
                while batch := pending_rows.fetchmany(num_parallel):
                    tasks = [
                        asyncio.create_task(
                            _verify(
                                tuple(json.loads(identity)),
                                json.loads(materialized),
                                json.loads(rollout),
                            )
                        )
                        for identity, materialized, rollout in batch
                    ]
                    for task in asyncio.as_completed(tasks):
                        result = await task
                        progress.update()
                        if result is not None:
                            if result.get("_ng_no_persist"):
                                verification_failure_count += 1
                                continue
                            if result.get("_ng_failure_class"):
                                failures.write(json.dumps(result) + "\n")
                                failures.flush()
                                os.fsync(failures.fileno())
                                verification_failure_count += 1
                                continue
                            output.write(json.dumps(result) + "\n")
                            output.flush()
                            os.fsync(output.fileno())
                            succeeded += 1
                            new_rewards.append(float(result.get("reward", 0.0)))
        finally:
            progress.close()

        if error_count > 0 or verification_failure_count > 0:
            logger.fatal(
                "%d/%d reverifications failed. Only %d results written.",
                error_count + verification_failure_count,
                rollout_count,
                succeeded,
            )
            raise RuntimeError("one or more rejudge requests failed")

        if new_rewards:
            avg = sum(new_rewards) / len(new_rewards)
            logger.info("  Average reward: %.4f (%d samples)", avg, len(new_rewards))
    finally:
        database.close()
        index_path.unlink(missing_ok=True)


if __name__ == "__main__":
    _unique_prompts = sys.argv[9:] == ["--unique-prompts"]
    if len(sys.argv) != 9 and not _unique_prompts:
        logger.error(
            "Usage: python -m nvflow.integrations.nemo_gym.workers.verify "
            "<materialized-inputs.jsonl> <rollouts.jsonl> <output.jsonl> <failures.jsonl> "
            "<head_host> <head_port> <resources_server_instance> <num_parallel> "
            "[--unique-prompts]"
        )
        sys.exit(1)

    _args = sys.argv[1:9]
    # NeMo-Gym's aiohttp client lazily calls get_global_config_dict() which
    # invokes @hydra.main and parses sys.argv.  Wipe argv so Hydra sees no
    # overrides (our positional args are not Hydra overrides).
    sys.argv = [sys.argv[0]]

    asyncio.run(
        verify_rollouts(
            materialized_inputs_file=_args[0],
            rollouts_file=_args[1],
            output_file=_args[2],
            failures_file=_args[3],
            head_host=_args[4],
            head_port=int(_args[5]),
            resources_server_instance=_args[6],
            num_parallel=int(_args[7]),
            unique_prompts=_unique_prompts,
        )
    )
