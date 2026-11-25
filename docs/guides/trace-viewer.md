# Rollout Trace Viewer

A lightweight, dependency-free web UI to spot-check NeMo-Gym rollout traces, one
at a time or two side by side. Implemented in
[`scripts/view_traces.py`](../../scripts/view_traces.py) (pure Python stdlib --
no Gradio, no extra installs).

It reads only the requested record (seek-by-line with a lazy byte-offset cache),
so it opens record 0 or record 35,000 of a multi-GB `rollouts.jsonl` without
loading the file. A background pass indexes each file by question and answer.

## Run

From the repository root:

```bash
uv run python scripts/view_traces.py [--root <dir>] [--port 8800]
```

- `--root` (default: `$NVFLOW_TRACE_ROOT` if set, else the current directory) --
  directory scanned for `*.jsonl` files (the file dropdown). Heavy/non-trace dirs
  (`cache/`, `logs/`, `.venv/`, ...), Gym's prepared inputs
  (`preprocessed_datasets/`, `*materialized_inputs*`, `*chunk_input*`), and
  empty files are skipped automatically.
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

- **File dropdown** (one per pane): pick `rollouts.jsonl`, `rollout_verdicts.jsonl`,
  or any other `*.jsonl`. A record without a trace renders as collapsible JSON.
- **Navigate**: by question and answer (Gym's `_ng_task_index` and
  `_ng_rollout_index`) or by record (line) number, then **Go**. **◀ / ▶** step
  through the file in question order, and **Random** picks a record.
- **Filter** (applies to every pane): **Reward below 1** or **Reward 1 or more**,
  and **containing text**, a case-insensitive match against the record's raw JSON
  line (for example `Time budget exhausted`). ◀ / ▶ and Random then move through
  the matching records only. Text search stops after 10,000 matches, and the
  pane's position line says so.
- **Compare**: two panes side by side, each with its own file. With **same
  question in both** on, moving one pane moves the other to the same question:
  to the other answer when both panes show one file (a question's repeats), or
  to the same answer number in another file (one run or teacher against another).
- **Trace rendering**: the exact recorded order of `input` + `response.output` --
  each step color-coded with an icon/pill (user, reasoning, tool call, tool
  output, assistant), numbered by model turn, and collapsed by default with a
  one-line preview. A tool output names the tool and call it answers (matched by
  `call_id`, so parallel calls stay paired). Click a step to expand;
  **Expand all / Collapse all** at the top right.
- **JSON as a tree**: tool-call args, tool outputs, and the **Raw JSON** view
  render as a colorized, collapsible tree -- click any `{}`/`[]` to fold/unfold.
  JSON that a tool packed inside a string is unpacked, and long text shows with
  its line breaks.
- **Header**: every top-level field of the record, reward first, then the
  response's metadata and token usage and the tools the model was offered. Long
  values fold, and Gym's bookkeeping fields (`_ng_*`, `ng_*`) fold under
  **Gym fields**.

## Notes

- Stdlib only; runs under `uv run python` (3.13) or any `python3` (3.9+).
- Responses use `Cache-Control: no-store`, so a plain refresh always shows the
  latest after a server restart (restart the server to pick up code edits).
- A file that grows, such as a running job's `rollouts.jsonl`, is indexed from
  where it stopped, and its new records show after a page reload; a record
  counts once its line ends. A file that shrinks or is replaced is read again.
- Single-user local tool: it serves on localhost only and reads files read-only.
  On `127.0.0.1` it answers only requests addressed to localhost, so a web page
  can't reach it through a domain that resolves there.
