# Rollout Trace Viewer

A lightweight, dependency-free web UI to spot-check NeMo-Gym rollout traces one
record at a time. Implemented in [`scripts/view_traces.py`](../scripts/view_traces.py)
(pure Python stdlib -- no Gradio, no extra installs).

It reads only the requested record (seek-by-line with a lazy byte-offset cache),
so it opens record 0 or record 35,000 of a multi-GB `output-rs*.jsonl` without
loading the file.

## Run

```bash
cd nvflow
uv run python scripts/view_traces.py [--root <dir>] [--port 8800]
```

- `--root` (default: `$NVFLOW_TRACE_ROOT` if set, else the current directory) --
  directory scanned for `*.jsonl` files (the file dropdown). Heavy/non-trace dirs
  (`cache/`, `logs/`, `.venv/`, ...) and input artifacts
  (`*materialized_inputs*`, `*chunk_input*`) are skipped automatically.
  Point it at a single workflow output dir. Do **not** point it at a parent that
  also holds the SEC filing dump -- scanning tens of thousands of filings makes
  the directory listing crawl.
- `--port` (default 8800), `--host` (default `127.0.0.1`).

## View it in the browser

The server binds `127.0.0.1`, so reach it through the SSH tunnel:

- In **Cursor / VS Code Remote**: the port is auto-forwarded. Open the **Ports**
  panel, find the port, click the globe ("Open in Browser"). If it isn't listed,
  "Forward a Port" -> enter the port. (Start the server in Cursor's integrated
  terminal so auto-forward triggers.)
- Manual fallback from your laptop: `ssh -L 8800:localhost:8800 <host>` then open
  `http://localhost:8800`.

## Using it

- **File dropdown**: pick a rollout file. For traces choose
  `…/rollout/output-rs*.jsonl` or the curated `…/rollout/analysis_rs*/{best,worst,intermediate}.jsonl`.
  A `train.jsonl` has no trace (just question + difficulty) and renders as a
  collapsible JSON record.
- **Navigate one record at a time**: record-number box + **Go**, **Prev/Next**,
  **Random** (Random counts the file once, then is instant).
- **Trace rendering**: the exact recorded order of `input` + `response.output` --
  each step color-coded with an icon/pill (user, reasoning, tool call, tool
  output, assistant), collapsed by default with a one-line preview. Click a step
  to expand; **Expand all / Collapse all** at the top right.
- **JSON as a tree**: tool-call args, tool outputs, and the **Raw JSON** view
  render as a colorized, collapsible tree -- click any `{}`/`[]` to fold/unfold
  nested fields.
- **Verdict header**: reward badge, judge rating/text, expected answer,
  question type, uuid.

## Notes

- Stdlib only; runs under `uv run python` (3.12) or any `python3` (3.9+).
- Responses use `Cache-Control: no-store`, so a plain refresh always shows the
  latest after a server restart (restart the server to pick up code edits).
- Single-user local tool: it serves on localhost only and reads files read-only.
