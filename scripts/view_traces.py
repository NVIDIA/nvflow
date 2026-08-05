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

Serves a tiny web UI to spot-check rollout traces ONE record at a time. The
server reads only the requested record (seek by line with a lazy offset cache),
so it handles multi-GB ``output-rs*.jsonl`` files without loading them.

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
    rendered as a conversation: prompt -> reasoning -> tool calls -> tool
    outputs -> final answer, plus a verdict header (reward / judge / expected).
  * Any other record falls back to pretty-printed JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_CONTENT_CHARS = 20000  # per-turn display cap to keep the page responsive
MAX_FILES = 1000
SCAN_MAXDEPTH = 8
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
}
# jsonl filename substrings that are pipeline input artifacts (no traces) -- hidden
# from the dropdown to reduce noise.
SKIP_FILE_SUBSTRINGS = ("materialized_inputs", "chunk_input")

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


def _pretty_json_str(s):
    try:
        return json.dumps(json.loads(s), indent=2)
    except Exception:
        return s if isinstance(s, str) else json.dumps(s, indent=2)


def parse_item(m: dict):
    """Convert one input/output item into {kind, title, content}."""
    if not m.get("type") and m.get("role"):
        m = {**m, "type": "message"}
    t = m.get("type")
    if t == "message":
        role = m.get("role", "assistant")
        return {"kind": role, "title": "", "content": _content_to_text(m.get("content", ""))}
    if t == "function_call":
        name = m.get("name", "?")
        return {
            "kind": "tool_call",
            "title": name,
            "content": _pretty_json_str(m.get("arguments", "{}")),
        }
    if t == "function_call_output":
        return {
            "kind": "tool_output",
            "title": "",
            "content": _pretty_json_str(m.get("output", "")),
        }
    if t == "reasoning":
        txt = "\n".join(s.get("text", "") for s in (m.get("summary") or []) if isinstance(s, dict))
        return {
            "kind": "reasoning",
            "title": "",
            "content": txt or _content_to_text(m.get("content", "")),
        }
    return {"kind": "other", "title": t or "item", "content": json.dumps(m, indent=2)}


HEADER_FIELDS = (
    "question",
    "problem",
    "expected_answer",
    "question_type",
    "reward",
    "judge_rating",
    "judge_text",
    "uuid",
    "current_date",
)


def parse_record(rec: dict) -> dict:
    """Return {schema, header, turns} for rollout records, else generic fallback."""
    header = {}
    for k in HEADER_FIELDS:
        if k in rec and rec[k] not in (None, ""):
            header[k] = rec[k]
    dp = rec.get("difficulty_profile")
    if isinstance(dp, dict) and "avg_reward" in dp:
        header["difficulty_profile.avg_reward"] = dp["avg_reward"]

    rcp = rec.get("responses_create_params")
    resp = rec.get("response")
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
        turns = []
        turn, step = 0, 0
        for m in inp + out:
            if not isinstance(m, dict):
                continue
            if m.get("role") == "user":
                turn += 1
                step = 0
            if m.get("type") == "function_call":
                step += 1
            ti = parse_item(m)
            content = ti["content"] or ""
            if len(content) > MAX_CONTENT_CHARS:
                ti["content"] = (
                    content[:MAX_CONTENT_CHARS]
                    + f"\n\n... [truncated {len(content) - MAX_CONTENT_CHARS} chars]"
                )
            ti["turn"], ti["step"] = turn, step
            turns.append(ti)
        return {"schema": "rollout", "header": header, "turns": turns}

    return {"schema": "generic", "header": header, "raw": rec}


# ---------------------------------------------------------------------------
# JSONL reader: read one record by index via a lazy byte-offset cache.
# ---------------------------------------------------------------------------
class JsonlReader:
    """Reads a single record by index without loading the whole file.

    Maintains a lazily-grown byte-offset index so repeated/sequential access is
    cheap. Thread-safe: a lock guards the shared offset cache (the HTTP server is
    threaded).
    """

    def __init__(self, path: Path):
        self.path = path
        self.offsets: list[int] = [0]  # offsets[i] = byte offset of line i
        self.eof = False
        self.count: int | None = None  # filled lazily (Random / total)
        self._lock = threading.Lock()

    def _extend_to(self, index: int):
        """Grow the offset cache through line `index`. Caller must hold the lock."""
        if self.eof or index < len(self.offsets):
            return
        with open(self.path, "rb") as f:
            f.seek(self.offsets[-1])
            i = len(self.offsets) - 1
            while i <= index:
                line = f.readline()
                if not line:
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
            if index >= len(self.offsets) - 1 and self.eof:
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

    def total(self) -> int:
        """Record count via a single chunked byte scan (cached). Used by Random."""
        with self._lock:
            if self.count is not None:
                return self.count
            n, last = 0, b""
            with open(self.path, "rb") as f:
                while True:
                    chunk = f.read(1 << 20)
                    if not chunk:
                        break
                    n += chunk.count(b"\n")
                    last = chunk[-1:]
            if last and last != b"\n":  # final line without trailing newline
                n += 1
            self.count = n
            return n


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
            if fn.endswith(".jsonl") and not any(s in fn for s in SKIP_FILE_SUBSTRINGS):
                files.append(str((Path(dirpath) / fn).relative_to(root)))
                if len(files) >= MAX_FILES:
                    return sorted(files)
    return sorted(files)


def make_handler(root: Path):
    root = root.resolve()
    readers: dict[str, JsonlReader] = {}
    readers_lock = threading.Lock()

    def reader_for(rel: str):
        # Resolve safely within root (no path traversal).
        target = (root / rel).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            return None
        if not target.is_file():
            return None
        with readers_lock:
            if rel not in readers:
                readers[rel] = JsonlReader(target)
            return readers[rel]

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
            parsed = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(parsed.query)
            path = parsed.path
            if path == "/" or path == "/index.html":
                self._send_html(PAGE)
                return
            if path == "/api/files":
                self._send_json({"root": str(root), "files": list_jsonl_files(root)})
                return
            if path in ("/api/record", "/api/random"):
                rel = (q.get("file") or [""])[0]
                rd = reader_for(rel)
                if rd is None:
                    self._send_json({"error": f"file not found under root: {rel}"}, 404)
                    return
                if path == "/api/random":
                    total = rd.total()
                    index = random.randint(0, max(0, total - 1)) if total else 0
                else:
                    try:
                        index = int((q.get("index") or ["0"])[0])
                    except ValueError:
                        index = 0
                rec = rd.get(index)
                if rec is None:
                    self._send_json(
                        {"error": "no record at index", "index": index, "known_total": rd.count},
                        404,
                    )
                    return
                parsed_rec = parse_record(rec)
                self._send_json(
                    {"index": index, "known_total": rd.count, "parsed": parsed_rec, "raw": rec}
                )
                return
            self._send_json({"error": "not found"}, 404)

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
  select { max-width:46vw; }
  input#idx { width:90px; }
  .grow { flex:1; }
  .badge { padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600; }
  .r0 { background:#3a1f24; color:#ff8a9b; } .rmid { background:#3a341f; color:#ffd479; } .r1 { background:#1f3a28; color:#7ee0a0; }
  #wrap { padding:14px; max-width:1100px; margin:0 auto; }
  .hdr { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:12px 14px; margin-bottom:14px; }
  .hdr h2 { margin:0 0 8px; font-size:15px; }
  .kv { display:grid; grid-template-columns:160px 1fr; gap:4px 12px; }
  .kv .k { color:var(--muted); }
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
  .hidden { display:none; }
  #status { color:var(--muted); font-size:12px; }
  #err { color:#ff8a9b; padding:14px; }
</style></head>
<body>
<header>
  <select id="file" title="rollout jsonl file"></select>
  <button id="prev">&#9664; Prev</button>
  <input id="idx" type="number" min="0" value="0" title="record number">
  <button id="go">Go</button>
  <button id="next">Next &#9654;</button>
  <button id="rand">&#8635; Random</button>
  <span class="grow"></span>
  <button id="expand">Expand all</button>
  <button id="collapse">Collapse all</button>
  <button id="rawbtn">Raw JSON</button>
  <span id="status"></span>
</header>
<div id="wrap"><div id="content"></div><div id="err"></div></div>
<script>
const $=s=>document.querySelector(s);
let state={file:null,index:0,raw:false,lastData:null};
function esc(s){return (s==null?"":String(s)).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
function rewardBadge(r){ if(r==null) return ""; const c=r>=1?"r1":(r>0?"rmid":"r0"); return `<span class="badge ${c}">reward ${r}</span>`; }
// recursive collapsible + colorized JSON tree
function jsonNode(val,key,depth){
  const keyHtml=key!=null?`<span class="jkey">${esc(JSON.stringify(key))}</span><span class="jpunc">: </span>`:"";
  if(val===null) return `<div class="jline">${keyHtml}<span class="jn-null">null</span></div>`;
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
async function loadFiles(){
  const d=await (await fetch("/api/files")).json();
  const sel=$("#file"); sel.innerHTML="";
  d.files.forEach(f=>{const o=document.createElement("option");o.value=f;o.textContent=f;sel.appendChild(o);});
  $("#status").textContent=d.files.length+" files under "+d.root;
  if(d.files.length){ state.file=d.files[0]; sel.value=state.file; show(0); }
  else { $("#err").textContent="No .jsonl files found under root."; }
}
function header(h){
  if(!h||!Object.keys(h).length) return "";
  let rows="";
  for(const k of Object.keys(h)){
    let v=h[k];
    if(typeof v==="object") v=JSON.stringify(v,null,2);
    rows+=`<div class="k">${esc(k)}</div><div>${esc(v)}</div>`;
  }
  return `<div class="hdr"><h2>${rewardBadge(h.reward)} ${h.question_type?esc(h.question_type):""}</h2><div class="kv">${rows}</div></div>`;
}
const KIND_ICON={user:"\u{1F464}",system:"\u2699\uFE0F",developer:"\u2699\uFE0F",assistant:"\u{1F4AC}",reasoning:"\u{1F9E0}",tool_call:"\u{1F527}",tool_output:"\u{1F4C4}",other:"\u2022"};
const KIND_LABEL={tool_call:"tool call",tool_output:"tool output"};
function turnBody(t){
  let obj=null; try{obj=JSON.parse(t.content);}catch(e){}
  if(obj!==null && typeof obj==="object") return renderJson(obj);  // JSON -> collapsible tree
  return `<pre>${esc(t.content)}</pre>`;                            // plain text -> pre
}
function turn(t){
  const icon=KIND_ICON[t.kind]||"\u2022";
  const label=KIND_LABEL[t.kind]||t.kind;
  const prev=esc((t.content||"").replace(/\s+/g," ").trim().slice(0,140));
  return `<div class="turn k-${esc(t.kind)}" data-open="0">
    <div class="t-title" onclick="toggleTurn(this.parentElement)">
      <span class="ic">${icon}</span><span class="pill">${esc(label)}</span>
      <span class="ttl">${esc(t.title||"")}</span>
      <span class="prev">${prev}</span>
      <span class="meta">T${t.turn}.S${t.step} \u00b7 ${(t.content||"").length}c</span>
      <span class="chev">\u25b8</span></div>
    <div class="t-body hidden">${turnBody(t)}</div></div>`;
}
function toggleTurn(el){ const open=el.getAttribute("data-open")==="1"; el.setAttribute("data-open",open?"0":"1"); el.querySelector(".t-body").classList.toggle("hidden",open); }
function setAll(open){ document.querySelectorAll("#content .turn").forEach(el=>{ el.setAttribute("data-open",open?"1":"0"); el.querySelector(".t-body").classList.toggle("hidden",!open); }); }
function render(d){
  $("#err").textContent="";
  state.lastData=d;
  const idxlbl="record "+d.index+(d.known_total!=null?(" / "+d.known_total):"");
  $("#status").textContent=state.file+" \u00b7 "+idxlbl;
  $("#idx").value=d.index;
  if(state.raw){ $("#content").innerHTML=`<div class="hdr">`+renderJson(d.raw)+`</div>`; return; }
  const p=d.parsed;
  if(p.schema==="rollout"){
    $("#content").innerHTML=header(p.header)+p.turns.map(turn).join("");
  } else {
    $("#content").innerHTML=header(p.header)+`<div class="turn k-other" data-open="1"><div class="t-title" onclick="toggleTurn(this.parentElement)"><span class="ic">{}</span><span class="pill">record</span><span class="prev">no rollout trace \u2014 collapsible JSON</span><span class="chev">\u25b8</span></div><div class="t-body">`+renderJson(p.raw)+`</div></div>`;
  }
}
async function show(i){
  if(state.file==null) return;
  state.index=Math.max(0,i);
  const u="/api/record?file="+encodeURIComponent(state.file)+"&index="+state.index;
  const r=await fetch(u); const d=await r.json();
  if(d.error){ $("#err").textContent=d.error+(d.known_total!=null?(" (total "+d.known_total+")"):""); return; }
  render(d);
}
async function rnd(){
  if(state.file==null) return;
  $("#status").textContent="picking random (counting records once)...";
  const r=await fetch("/api/random?file="+encodeURIComponent(state.file)); const d=await r.json();
  if(d.error){ $("#err").textContent=d.error; return; } render(d);
}
$("#file").onchange=e=>{state.file=e.target.value; show(0);};
$("#prev").onclick=()=>show(state.index-1);
$("#next").onclick=()=>show(state.index+1);
$("#go").onclick=()=>show(parseInt($("#idx").value||"0",10));
$("#idx").onkeydown=e=>{if(e.key==="Enter")show(parseInt($("#idx").value||"0",10));};
$("#rand").onclick=rnd;
$("#expand").onclick=()=>setAll(true);
$("#collapse").onclick=()=>setAll(false);
$("#rawbtn").onclick=()=>{state.raw=!state.raw; if(state.lastData) render(state.lastData);};
loadFiles();
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

    handler = make_handler(root)
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
