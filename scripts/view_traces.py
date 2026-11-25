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
"""Lightweight NeMo-Gym rollout-trace viewer (stdlib only, no Gradio).

Serves a small web UI to spot-check rollout traces, one at a time or two side
by side. The server reads only the requested record (seek by line with a lazy
offset cache), so it opens multi-GB ``rollouts.jsonl`` files without loading
them, and indexes each file by question and answer in the background.

Design inspired by the (removed) ``nemo_gym/dataset_viewer.py`` but with zero
external dependencies -- pure Python stdlib.

Usage:
    uv run python scripts/view_traces.py [--root <dir>] [--port 8800]

``--root`` defaults to ``$NVFLOW_TRACE_ROOT`` if set, otherwise the current
directory. Then open the forwarded http://localhost:<port> in your
browser. In Cursor / VS Code Remote the port is auto-forwarded over SSH -- just
click the "open in browser" notification (or use the Ports panel).

Rendering:
  * Rollout records (have ``responses_create_params`` + ``response``) are
    rendered as their conversation in recorded order -- prompt, reasoning,
    assistant text, tool calls, and tool outputs labeled with the call they
    answer -- under a header of the record's other fields and the response's
    metadata and usage.
  * Any other record falls back to collapsible JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

MAX_CONTENT_CHARS = 20000  # per-step display cap to keep the page responsive
MAX_FILES = 1000
MAX_SEARCH_MATCHES = 10000
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
SCAN_MAXDEPTH = 8
# Tools often return JSON packed inside a string, sometimes a few levels deep.
NESTED_JSON_DEPTH = 3
# Heavy / non-trace dirs to skip while scanning for *.jsonl (keeps the scan fast
# even when rooted at the whole grpo workflow dir -- e.g. the 14 GB SEC cache).
PRUNE_DIRS = {
    ".venv",
    "venv",
    ".git",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "cache",
    "logs",
    "filings",
    "filings_metadata",
    "preprocessed_datasets",
}
# jsonl filename substrings that are pipeline input artifacts (no traces) -- hidden
# from the dropdown to reduce noise.
SKIP_FILE_SUBSTRINGS = ("materialized_inputs", "chunk_input")
TRACE_FIELDS = ("responses_create_params", "response")
MODEL_KINDS = frozenset({"assistant", "reasoning", "tool_call"})

# Point this at a single workflow output dir, not at the parent of the SEC dump
# tree -- scanning tens of thousands of filings makes the directory listing crawl.
DEFAULT_ROOT = os.environ.get("NVFLOW_TRACE_ROOT") or "."


# ---------------------------------------------------------------------------
# Record parsing (schema-aware, dict-based -- no openai/pydantic deps)
# ---------------------------------------------------------------------------
def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                parts.append(c.get("text") or c.get("output") or json.dumps(c, indent=2))
            else:
                parts.append(str(c))
        return "\n".join(parts)
    if content is None:
        return ""
    return json.dumps(content, indent=2)


def _decode_nested(value, depth: int = 0):
    """Decode JSON objects and arrays packed inside strings; other strings stay as they are."""
    if isinstance(value, dict):
        return {k: _decode_nested(v, depth) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_nested(v, depth) for v in value]
    if isinstance(value, str) and depth < NESTED_JSON_DEPTH and value.lstrip()[:1] in ("{", "["):
        try:
            return _decode_nested(json.loads(value), depth + 1)
        except ValueError:
            return value
    return value


def _json_text(value) -> str:
    """Pretty JSON for a JSON string or value, with nested JSON decoded; plain text stays text."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return value
    value = _decode_nested(value)
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False)


def _item_kind(m: dict) -> str:
    item_type = m.get("type") or ("message" if m.get("role") else "")
    if item_type == "message":
        return str(m.get("role") or "assistant")
    return {
        "function_call": "tool_call",
        "function_call_output": "tool_output",
        "reasoning": "reasoning",
    }.get(item_type, "other")


def _item_content(m: dict, kind: str) -> str:
    if kind == "tool_call":
        return _json_text(m.get("arguments", "{}"))
    if kind == "tool_output":
        return _json_text(m.get("output", ""))
    if kind == "reasoning":
        summary = m.get("summary") or []
        text = "\n".join(s.get("text", "") for s in summary if isinstance(s, dict))
        return text or _content_to_text(m.get("content", ""))
    if kind == "other":
        return json.dumps(m, indent=2, ensure_ascii=False)
    return _content_to_text(m.get("content", ""))


def trace_steps(items: list) -> list[dict]:
    """One step per item, numbered by model turn; a tool output names the call it answers.

    Outputs are matched to calls by ``call_id``, so parallel calls whose outputs
    arrive in another order are still labeled correctly.
    """
    steps: list[dict] = []
    calls: dict[str, tuple[int, str]] = {}
    turn, in_turn = 0, False
    for m in items:
        if not isinstance(m, dict):
            continue
        kind = _item_kind(m)
        if kind in MODEL_KINDS:
            if not in_turn:
                turn, in_turn = turn + 1, True
        else:
            in_turn = False
        title, call = "", None
        if kind == "tool_call":
            call, title = len(calls) + 1, str(m.get("name") or "?")
            calls[str(m.get("call_id") or m.get("id") or call)] = (call, title)
        elif kind == "tool_output":
            call, title = calls.get(str(m.get("call_id")), (None, ""))
        elif kind == "other":
            title = str(m.get("type") or "item")
        content = _item_content(m, kind)
        if len(content) > MAX_CONTENT_CHARS:
            content = (
                content[:MAX_CONTENT_CHARS]
                + f"\n\n... [truncated {len(content) - MAX_CONTENT_CHARS} chars]"
            )
        steps.append({"kind": kind, "title": title, "content": content, "turn": turn, "call": call})
    return steps


def _flatten(prefix: str, value: dict, out: dict) -> None:
    for key, item in value.items():
        name = f"{prefix}.{key}"
        if isinstance(item, dict):
            _flatten(name, item, out)
        elif item not in (None, ""):
            out[name] = item


def record_header(rec: dict) -> dict:
    """Every top-level field except the trace, reward first, then the response's metadata,
    token usage, and the tools the model was offered."""
    header: dict = {}
    if rec.get("reward") is not None:
        header["reward"] = rec["reward"]
    for key, value in rec.items():
        if key not in TRACE_FIELDS and key not in header and value not in (None, "", [], {}):
            header[key] = value
    response = rec.get("response")
    if isinstance(response, dict):
        for name in ("metadata", "usage"):
            if isinstance(response.get(name), dict):
                _flatten(f"response.{name}", response[name], header)
    params = rec.get("responses_create_params")
    tools = params.get("tools") if isinstance(params, dict) else None
    if isinstance(tools, list):
        names = [t["name"] for t in tools if isinstance(t, dict) and t.get("name")]
        if names:
            header["tools offered"] = ", ".join(names)
    return header


def parse_record(rec) -> dict:
    """Return {schema, header, steps} for rollout records, else {schema, header, raw}."""
    if not isinstance(rec, dict):
        return {"schema": "generic", "header": {}, "raw": rec}
    rcp, resp = rec.get("responses_create_params"), rec.get("response")
    if isinstance(rcp, dict) and isinstance(resp, dict):
        raw_inp = rcp.get("input")
        if isinstance(raw_inp, str):
            inp: list = [{"role": "user", "content": raw_inp}]
        elif isinstance(raw_inp, list):
            inp = raw_inp
        else:
            inp = []
        raw_out = resp.get("output")
        out: list = raw_out if isinstance(raw_out, list) else []
        return {"schema": "rollout", "header": record_header(rec), "steps": trace_steps(inp + out)}
    header = {"reward": rec["reward"]} if rec.get("reward") is not None else {}
    return {"schema": "generic", "header": header, "raw": rec}


# ---------------------------------------------------------------------------
# JSONL reader: read one record by index via a lazy byte-offset cache.
# ---------------------------------------------------------------------------
class JsonlReader:
    """Reads a single record by index without loading the whole file.

    Maintains a lazily-grown byte-offset index so repeated/sequential access is
    cheap. A line counts once its newline is written, so a file that grows keeps
    its offsets. Thread-safe: a lock guards the shared offset cache (the HTTP
    server is threaded).
    """

    def __init__(self, path: Path):
        self.path = path
        self.offsets: list[int] = [0]  # offsets[i] = byte offset of line i
        self.eof = False
        self.count: int | None = None  # known once the end of the file is reached
        self._lock = threading.Lock()

    def grew(self) -> None:
        """Read past the old end of the file on the next request."""
        with self._lock:
            self.eof = False
            self.count = None

    def _extend_to(self, index: int):
        """Grow the offset cache through line `index`. Caller must hold the lock."""
        if self.eof or index < len(self.offsets) - 1:
            return
        with open(self.path, "rb") as f:
            f.seek(self.offsets[-1])
            i = len(self.offsets) - 1
            while i <= index:
                line = f.readline()
                if not line.endswith(b"\n"):
                    self.eof = True
                    self.count = len(self.offsets) - 1
                    break
                i += 1
                self.offsets.append(f.tell())

    def get(self, index: int):
        if index < 0:
            return None
        with self._lock:
            self._extend_to(index)
            # The line at the last offset has no newline yet.
            if index >= len(self.offsets) - 1:
                return None
            offset = self.offsets[index]
        # File I/O + parse outside the lock (independent of shared state).
        try:
            with open(self.path, "rb") as f:
                f.seek(offset)
                line = f.readline()
        except OSError:
            return None
        if not line:
            return None
        try:
            return json.loads(line)
        except Exception:
            return {
                "_parse_error": True,
                "raw_line": line.decode("utf-8", "replace")[:MAX_CONTENT_CHARS],
            }


def summarize_line(line_no: int, line: bytes) -> list[Any]:
    """[line, question, answer, reward] for one JSONL line; values a record lacks are None."""
    try:
        rec = json.loads(line)
    except ValueError:
        rec = None
    if not isinstance(rec, dict):
        return [line_no, None, None, None]
    task, answer, reward = (
        rec.get("_ng_task_index"),
        rec.get("_ng_rollout_index"),
        rec.get("reward"),
    )
    return [
        line_no,
        task if type(task) is int else None,
        answer if type(answer) is int else None,
        reward if type(reward) in (int, float) else None,
    ]


class TraceIndex:
    """Question, answer, and reward of every record in one file, ordered by question.

    Built in a background thread, so the first record shows at once; when the
    file grows, only the new records are read. A line counts once its newline is
    written. Gym rollouts carry ``_ng_task_index`` and ``_ng_rollout_index``;
    records without them are listed by line, after the others.
    """

    def __init__(self, path: Path):
        self.path = path
        self.rows: list[list[Any]] = []
        self.scanned = 0
        self.done = False
        self.error: str | None = None
        self._offset = 0
        self._wanted = 0
        self._running = False
        self._lock = threading.Lock()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if not self.done and not self._running:
                self._running = True
                threading.Thread(target=self._build, daemon=True).start()
            if not self.done:
                return {"status": "indexing", "scanned": self.scanned}
            return {"status": "ready", "rows": self.rows, "error": self.error}

    def grew(self) -> None:
        """Index the records appended since the last scan on the next snapshot."""
        with self._lock:
            self._wanted += 1
            self.done = False

    def _build(self) -> None:
        while True:
            with self._lock:
                wanted, offset, line_no = self._wanted, self._offset, self.scanned
            rows: list[list[Any]] = []
            error = None
            try:
                with open(self.path, "rb") as f:
                    f.seek(offset)
                    for line in iter(f.readline, b""):
                        if not line.endswith(b"\n"):
                            break
                        if line.strip():
                            rows.append(summarize_line(line_no, line))
                        line_no += 1
                        offset += len(line)
                        self.scanned = line_no
            except OSError as e:
                error = f"{type(e).__name__}: {e}"
            with self._lock:
                self.rows = sorted(
                    self.rows + rows, key=lambda r: (r[1] is None, r[1] or 0, r[2] or 0, r[0])
                )
                self.error, self._offset, self.scanned = error, offset, line_no
                if self._wanted == wanted:
                    self.done, self._running = True, False
                    return


def search_lines(
    path: Path, needle: str, limit: int = MAX_SEARCH_MATCHES
) -> tuple[list[int], bool]:
    """Lines whose raw JSON text contains *needle*, ignoring case; True if *limit* cut the list."""
    target = needle.lower().encode("utf-8")
    hits: list[int] = []
    with open(path, "rb") as f:
        for line_no, line in enumerate(f):
            if target in line.lower():
                hits.append(line_no)
                if len(hits) >= limit:
                    return hits, True
    return hits, False


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------
def list_jsonl_files(root: Path):
    files = []
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        if len(rel.parts) > SCAN_MAXDEPTH:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS and not d.startswith(".")]
        for fn in filenames:
            if not fn.endswith(".jsonl") or any(s in fn for s in SKIP_FILE_SUBSTRINGS):
                continue
            try:
                if os.path.getsize(os.path.join(dirpath, fn)) == 0:
                    continue
            except OSError:
                continue
            files.append(str((Path(dirpath) / fn).relative_to(root)))
            if len(files) >= MAX_FILES:
                return sorted(files)
    return sorted(files)


def _request_host(header: str) -> str:
    """The host of an HTTP ``Host`` header, without its port."""
    header = header.strip().lower()
    if header.startswith("["):
        return header[1 : header.find("]")] if "]" in header else header
    return header.rsplit(":", 1)[0] if header.count(":") == 1 else header


def make_handler(root: Path, allowed_hosts: frozenset[str] | None = LOOPBACK_HOSTS):
    """Request handler serving *root*; requests naming another host are refused.

    Checking the ``Host`` header keeps a web page from reaching a loopback-bound
    viewer by pointing its own domain at 127.0.0.1. *allowed_hosts* None skips it.
    """
    root = root.resolve()
    opened: dict[str, tuple[tuple[int, int, int], JsonlReader, TraceIndex]] = {}
    searches: dict[tuple[str, str, tuple[int, int, int]], tuple[list[int], bool]] = {}
    lock = threading.Lock()

    def open_file(rel: str):
        """Reader and index for *rel* under root; a file that only grew is read on, not again."""
        target = (root / rel).resolve()
        try:
            target.relative_to(root)  # no path traversal
            stat = target.stat()
        except (ValueError, OSError):
            return None
        if not target.is_file():
            return None
        version = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        grown = False
        with lock:
            entry = opened.get(rel)
            if entry is not None and entry[0] != version:
                inode, size, _modified = entry[0]
                if inode == stat.st_ino and stat.st_size > size:
                    grown = True
                    entry = (version, entry[1], entry[2])
                else:
                    entry = None
            if entry is None:
                entry = (version, JsonlReader(target), TraceIndex(target))
            opened[rel] = entry
        # A reader stays locked through a long scan; other files mustn't wait on it.
        if grown:
            entry[1].grew()
            entry[2].grew()
        return entry

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _send_json(self, obj, code=200):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, text):
            body = text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802  (BaseHTTPRequestHandler requires this name)
            try:
                self._route()
            except Exception as e:  # never leave the client hanging on a bare 500
                try:
                    self._send_json({"error": f"{type(e).__name__}: {e}"}, 500)
                except Exception:
                    pass

        def _route(self):
            if allowed_hosts is not None and (
                _request_host(self.headers.get("Host") or "") not in allowed_hosts
            ):
                self._send_json({"error": "requests must name this viewer's own host"}, 403)
                return
            parsed = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(parsed.query)
            path = parsed.path
            if path == "/" or path == "/index.html":
                self._send_html(PAGE)
                return
            if path == "/api/files":
                self._send_json({"root": str(root), "files": list_jsonl_files(root)})
                return
            if path not in ("/api/record", "/api/index", "/api/search"):
                self._send_json({"error": "not found"}, 404)
                return
            rel = (q.get("file") or [""])[0]
            entry = open_file(rel)
            if entry is None:
                self._send_json({"error": f"file not found under root: {rel}"}, 404)
                return
            version, reader, index = entry
            if path == "/api/index":
                self._send_json(index.snapshot())
                return
            if path == "/api/search":
                needle = (q.get("q") or [""])[0]
                key = (rel, needle.lower(), version)
                with lock:
                    found = searches.get(key)
                if found is None:
                    found = search_lines(reader.path, needle)
                    with lock:
                        if len(searches) >= 64:
                            searches.clear()
                        searches[key] = found
                self._send_json({"lines": found[0], "truncated": found[1]})
                return
            try:
                line = int((q.get("index") or ["0"])[0])
            except ValueError:
                line = 0
            rec = reader.get(line)
            if rec is None:
                self._send_json(
                    {"error": "no record at index", "index": line, "known_total": reader.count},
                    404,
                )
                return
            self._send_json(
                {
                    "index": line,
                    "known_total": reader.count,
                    "parsed": parse_record(rec),
                    "raw": rec,
                }
            )

    return Handler


# ---------------------------------------------------------------------------
# Frontend (single self-contained page)
# ---------------------------------------------------------------------------
PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Rollout Trace Viewer</title>
<style>
  :root { --bg:#0f1116; --panel:#171a21; --muted:#8b93a7; --fg:#e6e8ee; --border:#2a2f3a; }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--fg); }
  header { position:sticky; top:0; background:var(--panel); border-bottom:1px solid var(--border); padding:10px 14px; display:flex; gap:8px; align-items:center; flex-wrap:wrap; z-index:10; }
  select, input, button { background:#0f1320; color:var(--fg); border:1px solid var(--border); border-radius:6px; padding:6px 9px; font:inherit; }
  button { cursor:pointer; } button:hover { border-color:#4a5266; }
  .seg { display:inline-flex; } .seg button { border-radius:0; } .seg button:first-child { border-radius:6px 0 0 6px; } .seg button:last-child { border-radius:0 6px 6px 0; border-left:none; }
  .seg button.on { background:#24304a; border-color:#4a5c86; }
  label.chk { color:var(--muted); font-size:13px; display:inline-flex; gap:5px; align-items:center; }
  input#q { width:230px; }
  .grow { flex:1; }
  .badge { padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; }
  .r0 { background:#3a1f24; color:#ff8a9b; } .rmid { background:#3a341f; color:#ffd479; } .r1 { background:#1f3a28; color:#7ee0a0; }
  #panes { padding:14px; }
  #panes.single { max-width:1100px; margin:0 auto; }
  #panes.compare { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:14px; }
  .pbar { display:flex; gap:6px; align-items:center; flex-wrap:wrap; margin-bottom:10px; }
  .pbar select { flex:1 1 260px; min-width:0; }
  .pbar input[type=number] { width:78px; }
  .pbar .lbl { color:var(--muted); font-size:12px; }
  .pstatus { color:var(--muted); font-size:12px; }
  .ptitle { display:flex; gap:10px; align-items:center; margin:0 0 8px; font-weight:600; }
  .hdr { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:12px 14px; margin-bottom:14px; }
  .kv { display:grid; grid-template-columns:minmax(120px,max-content) 1fr; gap:4px 12px; }
  .kv .k { color:var(--muted); overflow-wrap:anywhere; } .kv > div { min-width:0; overflow-wrap:anywhere; }
  details.long summary { cursor:pointer; color:var(--muted); }
  .turn { border:1px solid var(--border); border-left:4px solid var(--muted); border-radius:8px; margin:8px 0; background:var(--panel); overflow:hidden; }
  .t-title { padding:9px 12px; cursor:pointer; display:flex; align-items:center; gap:10px; user-select:none; }
  .t-title:hover { background:rgba(255,255,255,.035); }
  .t-title .ic { font-size:15px; width:18px; text-align:center; flex:none; }
  .t-title .pill { font-size:11px; font-weight:700; letter-spacing:.02em; text-transform:uppercase; padding:2px 8px; border-radius:999px; background:#222838; color:var(--fg); flex:none; }
  .t-title .ttl { font-weight:600; flex:none; font-family:ui-monospace,monospace; color:#ffd479; }
  .t-title .prev { color:var(--muted); font-size:12px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; flex:1; }
  .t-title .meta { color:var(--muted); font-size:11px; white-space:nowrap; flex:none; }
  .t-title .chev { color:var(--muted); transition:transform .15s; flex:none; }
  .turn[data-open="1"] .chev { transform:rotate(90deg); }
  .t-body { border-top:1px solid var(--border); }
  .turn pre { margin:0; padding:10px 12px; white-space:pre-wrap; word-break:break-word; overflow-x:auto; font:12px/1.6 ui-monospace,Menlo,Consolas,monospace; color:#cdd3e0; }
  .k-user { border-left-color:#5b8def; } .k-user .pill { background:#1c2c50; color:#9cc0ff; }
  .k-system,.k-developer { border-left-color:#8b93a7; } .k-system .pill,.k-developer .pill { background:#2a2f3a; color:#c4cbdb; }
  .k-assistant { border-left-color:#3fb27f; } .k-assistant .pill { background:#16382a; color:#7ee0a0; }
  .k-reasoning { border-left-color:#a472e8; } .k-reasoning .pill { background:#2e2148; color:#c9a9f5; } .k-reasoning pre { font-style:italic; color:#b9aed0; }
  .k-tool_call { border-left-color:#e0913f; } .k-tool_call .pill { background:#3a2a16; color:#ffc98a; }
  .k-tool_output { border-left-color:#3fb0b2; } .k-tool_output .pill { background:#16383a; color:#8ee6e8; }
  .k-other { border-left-color:#666; }
  /* collapsible JSON tree */
  .jtree { font:12px/1.7 ui-monospace,Menlo,Consolas,monospace; padding:10px 12px; }
  .jline { white-space:pre-wrap; word-break:break-word; }
  .jtext { white-space:pre-wrap; word-break:break-word; color:#7ee0a0; margin:2px 0 6px 14px; }
  .jtoggle { cursor:pointer; border-radius:4px; }
  .jtoggle:hover { background:rgba(255,255,255,.04); }
  .jkey { color:#9cc0ff; }
  .jpunc { color:#8b93a7; }
  .jn-str { color:#7ee0a0; } .jn-num { color:#ffc98a; } .jn-bool { color:#c9a9f5; } .jn-null { color:#8b93a7; }
  .jchildren { margin-left:14px; border-left:1px solid var(--border); padding-left:10px; }
  .jchev { display:inline-block; width:12px; color:var(--muted); transition:transform .1s; }
  .jsum { color:var(--muted); font-style:italic; }
  .jnode.collapsed > .jchildren, .jnode.collapsed > .jclose { display:none; }
  .jnode:not(.collapsed) > .jline .jsum, .jnode:not(.collapsed) > .jline .jcb-inline { display:none; }
  .jnode.collapsed > .jline .jchev { transform:rotate(-90deg); }
  .hidden { display:none !important; }
  details.more { margin-top:8px; } details.more summary { cursor:pointer; color:var(--muted); font-size:12px; } details.more .kv { margin-top:6px; }
  .perr { color:#ff8a9b; padding:14px; }
</style></head>
<body>
<header>
  <span class="seg"><button id="m-single" class="on">Single</button><button id="m-compare">Compare</button></span>
  <label class="chk hidden" id="linkwrap"><input type="checkbox" id="link" checked> same question in both</label>
  <select id="reward" title="filter by reward">
    <option value="all">All answers</option>
    <option value="fail">Reward below 1</option>
    <option value="pass">Reward 1 or more</option>
  </select>
  <input id="q" placeholder="containing text (raw JSON)">
  <button id="apply">Filter</button>
  <span class="grow"></span>
  <button id="expand">Expand all</button>
  <button id="collapse">Collapse all</button>
  <button id="rawbtn">Raw JSON</button>
</header>
<div id="panes" class="single"></div>
<script>
const $=(s,el=document)=>el.querySelector(s);
const enc=encodeURIComponent;
const S={mode:"single",raw:false,link:true,reward:"all",q:"",files:[],panes:[]};
function esc(s){return (s==null?"":String(s)).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));}
async function api(path){ return (await fetch(path)).json(); }
function rewardBadge(r){ if(r==null) return ""; const c=r>=1?"r1":(r>0?"rmid":"r0"); return `<span class="badge ${c}">reward ${esc(r)}</span>`; }
// recursive collapsible + colorized JSON tree; long or multi-line strings show as text
function jsonNode(val,key,depth){
  const keyHtml=key!=null?`<span class="jkey">${esc(JSON.stringify(key))}</span><span class="jpunc">: </span>`:"";
  if(val===null) return `<div class="jline">${keyHtml}<span class="jn-null">null</span></div>`;
  if(typeof val==="string" && (val.includes("\n") || val.length>160))
    return `<div class="jline">${keyHtml}<div class="jtext">${esc(val)}</div></div>`;
  if(typeof val!=="object"){
    const cls=typeof val==="number"?"jn-num":(typeof val==="boolean"?"jn-bool":"jn-str");
    const disp=typeof val==="string"?JSON.stringify(val):String(val);
    return `<div class="jline">${keyHtml}<span class="${cls}">${esc(disp)}</span></div>`;
  }
  const arr=Array.isArray(val);
  const entries=arr?val.map((v,i)=>[i,v]):Object.entries(val);
  const ob=arr?"[":"{", cb=arr?"]":"}";
  const sum=arr?`${entries.length} items`:`${entries.length} keys`;
  const collapsed=depth>=2?" collapsed":"";
  const inner=entries.length?entries.map(([k,v])=>jsonNode(v,arr?null:k,depth+1)).join(""):`<div class="jline jpunc">(empty)</div>`;
  return `<div class="jnode${collapsed}">`
    +`<div class="jline jtoggle" onclick="this.parentElement.classList.toggle('collapsed')">`
    +`<span class="jchev">\u25be</span>${keyHtml}<span class="jpunc">${ob}</span><span class="jsum"> ${sum} </span><span class="jpunc jcb-inline">${cb}</span></div>`
    +`<div class="jchildren">${inner}</div>`
    +`<div class="jline jclose"><span class="jpunc">${cb}</span></div></div>`;
}
function renderJson(obj){ return `<div class="jtree">${jsonNode(obj,null,0)}</div>`; }
function fieldRows(h,keys){
  let rows="";
  for(const k of keys){
    const v=h[k];
    let cell;
    if(v!==null && typeof v==="object") cell=renderJson(v);
    else {
      const s=String(v);
      cell=(s.length>240 || s.includes("\n"))
        ? `<details class="long"><summary>${esc(s.slice(0,160).replace(/\s+/g," "))}\u2026</summary><div class="jtext">${esc(s)}</div></details>`
        : esc(s);
    }
    rows+=`<div class="k">${esc(k)}</div><div>${cell}</div>`;
  }
  return rows;
}
// Gym's own bookkeeping fields (_ng_*, ng_*) fold away under the record's fields
function headerHtml(h){
  const keys=Object.keys(h||{});
  if(!keys.length) return "";
  const gym=keys.filter(k=>k.startsWith("_ng_") || k.startsWith("ng_"));
  const main=keys.filter(k=>!gym.includes(k));
  const more=gym.length ? `<details class="more"><summary>${gym.length} Gym fields</summary><div class="kv">${fieldRows(h,gym)}</div></details>` : "";
  return `<div class="hdr"><div class="kv">${fieldRows(h,main)}</div>${more}</div>`;
}
const KIND_ICON={user:"\u{1F464}",system:"\u2699\uFE0F",developer:"\u2699\uFE0F",assistant:"\u{1F4AC}",reasoning:"\u{1F9E0}",tool_call:"\u{1F527}",tool_output:"\u{1F4C4}",other:"\u2022"};
const KIND_LABEL={tool_call:"tool call",tool_output:"tool output"};
function stepBody(t){
  let obj=null; try{obj=JSON.parse(t.content);}catch(e){}
  if(obj!==null && typeof obj==="object") return renderJson(obj);  // JSON -> collapsible tree
  return `<pre>${esc(t.content)}</pre>`;                            // plain text -> pre
}
function stepHtml(t){
  const label=KIND_LABEL[t.kind]||t.kind;
  const prev=esc((t.content||"").replace(/\s+/g," ").trim().slice(0,140));
  const meta=`turn ${t.turn}`+(t.call!=null?` \u00b7 call ${t.call}`:"")+` \u00b7 ${(t.content||"").length}c`;
  return `<div class="turn k-${esc(t.kind)}" data-open="0">
    <div class="t-title" onclick="toggleStep(this.parentElement)">
      <span class="ic">${KIND_ICON[t.kind]||"\u2022"}</span><span class="pill">${esc(label)}</span>
      <span class="ttl">${esc(t.title||"")}</span>
      <span class="prev">${prev}</span>
      <span class="meta">${meta}</span>
      <span class="chev">\u25b8</span></div>
    <div class="t-body hidden">${stepBody(t)}</div></div>`;
}
function toggleStep(el){ const open=el.getAttribute("data-open")==="1"; el.setAttribute("data-open",open?"0":"1"); el.querySelector(".t-body").classList.toggle("hidden",open); }
function setAll(open){ document.querySelectorAll(".pane .turn").forEach(el=>{ el.setAttribute("data-open",open?"1":"0"); el.querySelector(".t-body").classList.toggle("hidden",!open); }); }
function rowOf(p,line){ return p.rows ? p.rows.find(r=>r[0]===line) : null; }
function filtered(){ return S.reward!=="all" || S.q!==""; }
function status(p,msg){
  let text=msg;
  if(text==null){
    if(!p.rows) text="indexing\u2026";
    else {
      const pos=p.list.findIndex(r=>r[0]===p.line);
      const at=pos>=0?String(pos+1):"\u2013";
      const cut=p.truncated ? `, text search stopped after its first ${p.truncated} matches` : "";
      text=filtered() ? `${at} of ${p.list.length} matching (${p.rows.length} records${cut})` : `${at} of ${p.rows.length}`;
    }
  }
  $(".pstatus",p.el).textContent=text;
}
// one pane: file, position, and the rendered record
function newPane(){
  const el=document.createElement("section");
  el.className="pane";
  el.innerHTML=`<div class="pbar">
      <select class="file" title="trace file"></select>
      <span class="lbl">question</span><input class="task" type="number" min="0">
      <span class="lbl">answer</span><input class="answer" type="number" min="0">
      <span class="lbl">record</span><input class="line" type="number" min="0">
      <button class="go">Go</button><button class="prev">\u25c0</button><button class="next">\u25b6</button><button class="rand">Random</button>
      <span class="pstatus"></span></div>
    <div class="content"></div>`;
  const p={el,file:null,rows:null,list:[],line:0,data:null,edited:"line",token:0};
  const sel=$(".file",el);
  S.files.forEach(f=>{const o=document.createElement("option");o.value=f;o.textContent=f;sel.appendChild(o);});
  sel.onchange=e=>setFile(p,e.target.value);
  $(".task",el).oninput=()=>{p.edited="task";};
  $(".answer",el).oninput=()=>{p.edited="task";};
  $(".line",el).oninput=()=>{p.edited="line";};
  el.querySelectorAll("input").forEach(inp=>{inp.onkeydown=e=>{if(e.key==="Enter") go(p);};});
  $(".go",el).onclick=()=>go(p);
  $(".prev",el).onclick=()=>step(p,-1);
  $(".next",el).onclick=()=>step(p,1);
  $(".rand",el).onclick=()=>step(p,null);
  $("#panes").appendChild(el);
  S.panes.push(p);
  return p;
}
async function setFile(p,file){
  const token=++p.token;
  p.file=file; p.rows=null; p.list=[]; p.data=null;
  $(".file",p.el).value=file;
  status(p,"loading\u2026");
  await showLine(p,0,false);
  for(;;){  // the index builds in the background; the first record is already showing
    const d=await api(`/api/index?file=${enc(file)}`);
    if(token!==p.token) return;
    if(d.status==="ready"){ p.rows=d.rows; break; }
    status(p,`indexing\u2026 ${d.scanned} records read`);
    await new Promise(r=>setTimeout(r,700));
  }
  await applyFilter(p);
  if(S.mode==="compare") syncFrom(S.panes[0]);
}
async function applyFilter(p){
  if(!p.rows) return;
  let rows=p.rows;
  if(S.reward==="fail") rows=rows.filter(r=>r[3]!=null && r[3]<1);
  if(S.reward==="pass") rows=rows.filter(r=>r[3]!=null && r[3]>=1);
  p.truncated=0;
  if(S.q){
    const d=await api(`/api/search?file=${enc(p.file)}&q=${enc(S.q)}`);
    const hit=new Set(d.lines||[]);
    rows=rows.filter(r=>hit.has(r[0]));
    if(d.truncated) p.truncated=hit.size;
  }
  p.list=rows;
  if(p.list.length && !p.list.some(r=>r[0]===p.line)) await showLine(p,p.list[0][0],false);
  else renderPane(p);
}
async function showLine(p,line,sync=true){
  const d=await api(`/api/record?file=${enc(p.file)}&index=${line}`);
  if(d.error){ status(p,d.error); return; }
  p.line=d.index; p.data=d;
  renderPane(p);
  if(sync) syncFrom(p);
}
function renderPane(p){
  const d=p.data; if(!d) return;
  const row=rowOf(p,d.index);
  $(".line",p.el).value=d.index;
  $(".task",p.el).value=row && row[1]!=null ? row[1] : "";
  $(".answer",p.el).value=row && row[2]!=null ? row[2] : "";
  const reward=row && row[3]!=null ? row[3] : (d.parsed.header||{}).reward;
  const elapsed=(d.parsed.header||{})._ng_elapsed_s;
  const ident=(row && row[1]!=null ? `question ${row[1]} \u00b7 answer ${row[2]} \u00b7 record ${d.index}` : `record ${d.index}`)
    +(typeof elapsed==="number" ? ` \u00b7 ${(elapsed/60).toFixed(1)} min` : "");
  let body;
  if(S.raw) body=`<div class="hdr">${renderJson(d.raw)}</div>`;
  else if(d.parsed.schema==="rollout") body=headerHtml(d.parsed.header)+d.parsed.steps.map(stepHtml).join("");
  else body=headerHtml(d.parsed.header)+`<div class="hdr">${renderJson(d.parsed.raw)}</div>`;
  $(".content",p.el).innerHTML=`<div class="ptitle">${rewardBadge(reward)}<span>${esc(ident)}</span></div>`+body;
  status(p);
}
async function go(p){
  if(p.edited==="task"){
    if(!p.rows){ status(p,"still indexing\u2026"); return; }
    const task=parseInt($(".task",p.el).value,10), answer=parseInt($(".answer",p.el).value,10);
    const row=p.rows.find(r=>r[1]===task && r[2]===answer) || p.rows.find(r=>r[1]===task);
    p.edited="line";
    if(!row){ status(p,`question ${task} is not in this file`); return; }
    await showLine(p,row[0]);
    return;
  }
  await showLine(p,parseInt($(".line",p.el).value||"0",10));
}
async function step(p,delta){  // delta null: random
  if(!p.rows){ status(p,"still indexing\u2026"); return; }
  if(!p.list.length){ status(p,"no answers match the filter"); return; }
  let pos=p.list.findIndex(r=>r[0]===p.line);
  if(delta===null) pos=Math.floor(Math.random()*p.list.length);
  else if(pos<0) pos=delta>0?0:p.list.length-1;
  else pos=Math.min(p.list.length-1,Math.max(0,pos+delta));
  await showLine(p,p.list[pos][0]);
}
// compare mode: move the other pane to the same question -- the other answer when
// both panes show one file, the same answer number otherwise
function syncFrom(src){
  if(S.mode!=="compare" || !S.link || !src) return;
  const row=rowOf(src,src.line);
  if(!row || row[1]==null) return;
  for(const p of S.panes){
    if(p===src || !p.rows) continue;
    const same=p.rows.filter(r=>r[1]===row[1]);
    if(!same.length){ status(p,`question ${row[1]} is not in this file`); continue; }
    const pick=p.file===src.file ? (same.find(r=>r[2]!==row[2])||same[0]) : (same.find(r=>r[2]===row[2])||same[0]);
    if(pick[0]!==p.line) showLine(p,pick[0],false);
  }
}
function setMode(mode){
  S.mode=mode;
  $("#m-single").classList.toggle("on",mode==="single");
  $("#m-compare").classList.toggle("on",mode==="compare");
  $("#linkwrap").classList.toggle("hidden",mode!=="compare");
  $("#panes").className=mode;
  if(mode==="compare" && S.panes.length===1) setFile(newPane(),S.panes[0].file);
  if(mode==="single") while(S.panes.length>1) S.panes.pop().el.remove();
}
async function refilter(){
  S.reward=$("#reward").value; S.q=$("#q").value.trim();
  for(const p of S.panes) await applyFilter(p);
  syncFrom(S.panes[0]);
}
$("#m-single").onclick=()=>setMode("single");
$("#m-compare").onclick=()=>setMode("compare");
$("#link").onchange=e=>{S.link=e.target.checked; syncFrom(S.panes[0]);};
$("#reward").onchange=refilter;
$("#apply").onclick=refilter;
$("#q").onkeydown=e=>{if(e.key==="Enter") refilter();};
$("#expand").onclick=()=>setAll(true);
$("#collapse").onclick=()=>setAll(false);
$("#rawbtn").onclick=()=>{S.raw=!S.raw; S.panes.forEach(renderPane);};
(async()=>{
  const d=await api("/api/files");
  S.files=d.files;
  if(!d.files.length){ $("#panes").innerHTML=`<div class="perr">No .jsonl files found under ${esc(d.root)}.</div>`; return; }
  await setFile(newPane(),d.files[0]);
})();
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser(
        description="Lightweight NeMo-Gym rollout-trace viewer (stdlib only)."
    )
    ap.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help=(
            "Directory scanned for *.jsonl files (file dropdown). "
            "Defaults to $NVFLOW_TRACE_ROOT, or the current directory."
        ),
    )
    ap.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host (default 127.0.0.1; use SSH/Cursor port-forward).",
    )
    ap.add_argument("--port", type=int, default=8800, help="Bind port (default 8800).")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    if not root.is_dir():
        raise SystemExit(f"--root is not a directory: {root}")
    root = root.resolve()

    handler = make_handler(root, LOOPBACK_HOSTS if args.host in LOOPBACK_HOSTS else None)
    try:
        httpd = ThreadingHTTPServer((args.host, args.port), handler)
    except OSError as e:
        raise SystemExit(
            f"Could not bind {args.host}:{args.port} ({e}).\n"
            "A viewer may already be running -- stop it, or pass a different --port."
        ) from e
    print(f"Rollout trace viewer serving {root}", flush=True)
    print(
        f"  http://{args.host}:{args.port}   (Cursor/VS Code will auto-forward this port)",
        flush=True,
    )
    print("  Ctrl+C to stop.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping.")
        httpd.shutdown()


if __name__ == "__main__":
    main()
