# Download SEC Filings

## Purpose

Download SEC filings from the EDGAR database for specified companies, and build
the artifacts that downstream workflows read: extracted sections for SDG and a
full-text search index for the agent's `edgar_search` tool.

## When to Use

- **First step** before any SDG workflow
- When you need to update to newer SEC filings
- To download filings for specific companies or date ranges
- To rebuild the search index after adding filings

## Prerequisites

Before running this workflow, ensure you have:

- ✅ **Output directory writable** (workflow creates `outputs/finance/demo/workflow-2-download-sec/step-0-download`)

- ✅ **SEC EDGAR identity available** to the launcher. Both values are required and
  have no default, for every command on this workflow, including `plan` and
  single-stage runs such as `run build_index` or `run prefetch_cache`. Pick
  whichever of these suits you:

  ```bash
  # Per shell session
  export SEC_IDENTITY_EMAIL=your.email@company.com
  export SEC_IDENTITY_COMPANY="Your Company"

  # Or for a single command
  SEC_IDENTITY_EMAIL=your.email@company.com SEC_IDENTITY_COMPANY="Your Company" \
    uv run nvflow run --all --workflow finance:download/demo --cluster my_cluster

  # Or persist them in a gitignored .env that uv loads
  printf 'SEC_IDENTITY_EMAIL=your.email@company.com\nSEC_IDENTITY_COMPANY="Your Company"\n' > .env
  uv run --env-file .env nvflow run --all --workflow finance:download/demo --cluster my_cluster
  ```

  - **Why needed:** The SEC EDGAR API requires all applications to identify themselves per their [Fair Access Policy](https://www.sec.gov/os/accessing-edgar-data)
  - **Required format:** Valid email address and company/organization name
  - Quote any value containing spaces. In a `.env` file an unquoted value is
    skipped with a parse warning, which surfaces later as the variable being unset.
  - These are read when the launcher resolves the configuration, so a cluster
    config's `env_vars` cannot supply them.

## How It Works

Three stages per scope. Each is selected by name, so you can rebuild the index
or refresh the cache without redoing the download.

1. **`download`** — download and section extraction
   (`projects/finance/nvfinance/stages/download/download_sec_filings.py`).
   Fetches each filing's primary document and HTML exhibits from EDGAR, then
   extracts Items (1, 1A, 2, …) from the form types that have an extractor.
   Records one row per document in `sec_metadata.parquet`, which everything
   downstream reads.

2. **`build_index`** — full-text search index
   (`projects/finance/nvfinance/stages/download/build_index.py`).
   Converts every primary document and exhibit to text and writes a SQLite FTS5
   index, plus the `.metadata` companion NeMo-Gym reads filing metadata from.
   This is what the agent's `edgar_search` tool queries instead of calling a
   hosted SEC API.

3. **`prefetch_cache`** — filing-metadata cache
   (`projects/finance/nvfinance/stages/download/prefetch_cache.py`).
   Fetches each company's filing list and the ticker registry, which is what the
   agent's `sec_filing_search` tool looks up by ticker. Reads neither the corpus
   nor the index, so it runs alongside the download rather than after it.
   Training and eval read this cache instead of calling sec.gov themselves.
   The smoke scope omits it.

## Configuration Files

- **Smoke:** `projects/finance/workflows/download/smoke.yaml` selects `configs/smoke.yaml` — 2 companies, 1 year
- **Demo:** `projects/finance/workflows/download/demo.yaml` selects `configs/demo.yaml` — 7 companies, 4 form types, 2020-2024
- **Production:** `projects/finance/workflows/download/sp500.yaml` selects `configs/sp500.yaml` — 499 S&P 500 companies, 18 form types

`forms` and the year range are hard filters at EDGAR, so these files define the
corpus exactly. Widening either means a longer download; the S&P 500 scope as
committed is a multi-day job against SEC's rate limit.

## Workflow Overview

```
┌─────────────────────────┐
│ Input                   │
├─────────────────────────┤
│ • Ticker list           │
│ • Date range + forms    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│ download                │     │ prefetch_cache          │  Independent: no
└───────────┬─────────────┘     └───────────┬─────────────┘  corpus, no index
            │                               │
            ▼                               │
┌─────────────────────────┐                 │
│ build_index             │  Index +        │
└───────────┬─────────────┘  .metadata      │
            │                               │
            └───────────────┬───────────────┘
                            ▼
┌────────────────────────────────────────────────────────────────┐
│ Output                                                         │
├────────────────────────────────────────────────────────────────┤
│ outputs/finance/{smoke,demo,sp500}/workflow-2-download-sec/    │
│ ├── step-0-download/                                           │
│ │   ├── data/{TICKER}/{FORM}/{YEAR}/{ACC_NO}/                  │
│ │   └── sec_metadata.parquet                                   │
│ ├── step-1-index/                                              │
│ │   ├── {scope}_sec_fts.sqlite                                 │
│ │   └── {scope}_sec_fts.sqlite.metadata                        │
│ └── cache/finance_sec_search/                                  │
│     ├── filings_metadata/{CIK}.json                            │
│     └── tickers.json                                           │
└────────────────────────────────────────────────────────────────┘
```

**See [technical reference](#detailed-stage-reference) for detailed parameters and configuration.**

## Usage

Each scope is its own workflow, so select the one you want. `finance:download/sp500`
is a multi-day download; start with `finance:download/demo`.

### Demo (Recommended for First Time)

```bash
# Seven-company demo
uv run nvflow run --all --workflow finance:download/demo --cluster my_cluster
```

### Production

```bash
# 499 S&P 500 companies
uv run nvflow run --all --workflow finance:download/sp500 --cluster my_cluster
```

### Refreshing an existing corpus

Re-run the whole workflow against the same paths. Every stage does only new
work: the download skips accession numbers already in `sec_metadata.parquet`,
extraction skips filings that already have sections, and `build_index` with
`incremental: true` adds only documents the index does not hold.

Before re-running:

1. Raise `end_year` in the ticker config if the year has rolled over. EDGAR
   filters on it, so later filings are otherwise never fetched.
2. Set `force: true` on `prefetch_cache` (see
   [below](#refreshing-only-the-metadata-cache)), for example in a config that
   extends the scope:

   ```yaml
   _base_: sp500.yaml
   stages:
     prefetch_cache:
       force: true
   ```

Keep `scope` and `index_dir` unchanged between refreshes. The incremental build
seeds from `step-1-index/<scope>_sec_fts.sqlite`; if that file is not found it
logs a warning naming the path and re-parses the whole corpus. The corpus directory itself can move: recorded
filing paths are resolved relative to `output_dir`.

### Rebuilding only the index

After adding filings to an existing corpus, run the index stage. With
`incremental: true`, the default for every scope, it parses only what it does
not already hold:

```bash
uv run nvflow run build_index --workflow finance:download/sp500 --cluster my_cluster
```

Indexing runs at roughly 6 documents/second, so a full S&P 500 build is about
26 hours and will outlive most job time limits. With `incremental: true` this is
safe to interrupt: the unfinished index is kept as `*.sqlite.building` and the
next run resumes from it, so re-submit the stage until it publishes. Pick a
partition whose limit you can live with (`sinfo -o "%P %l"`).

Setting `incremental: false` discards any unfinished build and re-parses the
whole corpus; the published index keeps serving until the new one replaces it.
Use it only when the text extraction itself has changed.

### Refreshing only the metadata cache

The cache is independent of the corpus, so it can be refreshed on its own:

```bash
uv run nvflow run prefetch_cache --workflow finance:download/sp500 --cluster my_cluster
```

Companies already cached are skipped, so a plain re-run only adds companies new
to the ticker config; it does not pick up new filings from companies already
cached. Set `force: true` on the stage to re-fetch and overwrite every company's
filing list, which takes minutes for the S&P 500 scope. `tickers.json` is
rewritten on every run.

## Output Structure

```
outputs/finance/demo/workflow-2-download-sec/
└── step-0-download/
    ├── data/                                    # SEC filing documents
    │   ├── AAPL/
    │   │   ├── 10-K/
    │   │   │   ├── 2023/
    │   │   │   │   └── 0000320193-23-000106/   # Accession number directory
    │   │   │   │       ├── primary-document.html    # Full 10-K filing
    │   │   │   │       ├── 1.html                   # Item 1: Business
    │   │   │   │       ├── 1A.html                  # Item 1A: Risk Factors
    │   │   │   │       ├── 1B.html                  # Item 1B: Unresolved Staff Comments
    │   │   │   │       ├── 2.html                   # Item 2: Properties
    │   │   │   │       ├── 3.html                   # Item 3: Legal Proceedings
    │   │   │   │       ├── ...                      # Other sections (4-15)
    │   │   │   │       └── exhibits/
    │   │   │   │           ├── EX-10.16.html
    │   │   │   │           ├── EX-10.17.html
    │   │   │   │           ├── EX-21.1.html
    │   │   │   │           ├── EX-23.1.html
    │   │   │   │           ├── EX-31.1.html
    │   │   │   │           └── ...
    │   │   │   └── 2024/
    │   │   │       └── 0000320193-24-000123/
    │   │   ├── 10-Q/
    │   │   │   └── 2024/
    │   │   │       ├── 0000320193-24-000045/
    │   │   │       └── 0000320193-24-000089/
    │   │   └── 8-K/
    │   ├── NVDA/
    │   ├── GOOG/
    │   └── ...
    ├── sec_metadata.parquet      # One row per document, sections included
    └── download-logs/            # Download logs
```

## Expected Results

| Config | Companies | Documents | Corpus | Index | Download | Index build |
|--------|-----------|-----------|--------|-------|----------|-------------|
| Smoke | 2 | ~100 | ~50MB | — | 2 min | — |
| Demo | 7 | ~2K | ~1GB | ~150MB | 20 min | 5 min |
| Production | 499 | ~575K | ~100GB | ~30GB | days | ~26 h |

Downloading is bounded by SEC's 10 requests/second ceiling and indexing by
single-threaded HTML parsing, so neither gets faster with more nodes.
Re-running the download is cheap when nothing is missing: one filing-list call
per ticker and no document fetches.

## Validation

`$WORKSPACE` is the host directory your target mounts at `/workspace`, as in the
[quick start](../quick-start.md).

```bash
# Check number of companies
ls $WORKSPACE/outputs/finance/demo/workflow-2-download-sec/step-0-download/data/ | wc -l
# Demo: 7, Production: 499

# Check metadata
uv run python -c "import pandas as pd; df = pd.read_parquet('$WORKSPACE/outputs/finance/demo/workflow-2-download-sec/step-0-download/sec_metadata.parquet'); print(df.groupby('file_type').size())"

# Check storage used
du -sh $WORKSPACE/outputs/finance/demo/workflow-2-download-sec/step-0-download/
```

A clean corpus is one where re-running the download reports zero filings to
download for every ticker. With `end_year` at the current year a small non-zero
count means EDGAR has new filings, not that anything drifted; read the
magnitude, and do not chase zero.

### Index

```bash
# What the index recorded about its own build (run from the repository root)
PYTHONPATH=projects/finance uv run python -m nvfinance.utils.index.build_index preflight \
    --index_path $WORKSPACE/outputs/finance/demo/workflow-2-download-sec/step-1-index/demo_sec_fts.sqlite

# Both files must be here, or the environment will not start
ls -l $WORKSPACE/outputs/finance/demo/workflow-2-download-sec/step-1-index/
```

## Next Steps

After downloading SEC filings, you can:

- **[Template-Based SDG](02-template-based-sdg.md)** - Generate Q&A from seed questions
- **[Document-Grounded SDG](03-document-grounded-sdg.md)** - Generate verified Q&A from documents

Both SDG workflows consume the downloaded filings.

## Common Issues

### SEC Identity Not Set

**Symptom:** The command exits before submitting anything, with one of:

```text
✗ Error: KeyError raised while resolving interpolation: "Environment variable 'SEC_IDENTITY_EMAIL' not found"
✗ Error: Workflow planning failed before configuration could be loaded
```

`run` names only the first unset variable; `plan` names none.

**Solution:** Export both, then re-run:

```bash
export SEC_IDENTITY_EMAIL=your.email@company.com SEC_IDENTITY_COMPANY="Your Company"
```

### SEC Rate Limiting

**Symptom:** Download fails with rate limit errors

**Solution:**
- The downloader includes automatic throttling
- Wait 10 minutes and retry
- Ensure `SEC_IDENTITY_EMAIL` and `SEC_IDENTITY_COMPANY` are exported (required by SEC)

**Symptom:** ReadTimeout Error (httpx.ReadTimeout: The read operation timed out)

**Solution:**
- There was a network delay in downloading the file so the read timed out
- Wait a few minutes and retry

### Storage Space

**Symptom:** Disk full during download

**Solution:**
- S&P 500 filings require ~100GB, plus ~50GB for the index
- Section extraction creates far more files than filings, so check your inode
  quota as well as free space
- Use `finance:download/demo` first to test with minimal storage

### Index build refuses to publish

**Symptom:** `N of M candidate rows have no file on disk`, and the index is left
as `*.sqlite.building`

**Cause:** `sec_metadata.parquet` describes filings that are not on disk. The
builder treats an absent file as a skip, so publishing would give you an index
that silently omits them.

**Solution:** Re-run the download stage until every ticker reports zero filings
to download, then re-run the index stage with `incremental: true` so it resumes
from the unfinished file instead of re-parsing. If the gap is expected, re-run
with `allow_missing_paths: true`.

### Environment refuses to start against a copied index

**Symptom:** NeMo-Gym reports that the metadata sidecar is missing, or that it
was built from a different index

**Cause:** The companion is a separate file discovered by name beside the index,
so copying only the `.sqlite` leaves it behind, and pairing files from two
different builds is rejected: NeMo-Gym joins them on rowid, which a rebuild
reassigns.

**Solution:** Copy `*.sqlite.metadata` alongside the `.sqlite`, always from the
same build. If they have already been mixed, re-run `build_index`, which
writes both.

## Configuration Parameters

See the [download stage reference](#detailed-stage-reference) for detailed parameter documentation.

Key parameters:
- `config`: Path to ticker configuration file
- `output_dir`: Where to save filings (default: `/workspace/outputs/finance/demo/workflow-2-download-sec/step-0-download`)
- `sec_identity_email`: Read from required `SEC_IDENTITY_EMAIL`
- `sec_identity_company`: Read from required `SEC_IDENTITY_COMPANY`

## Technical Details

For comprehensive technical documentation, see:
- **[Download stage reference](#detailed-stage-reference)**

## Detailed stage reference

Technical reference for the download-sec workflow stages.

Every scope runs the download itself, then the search index built on top of it.
The demo and sp500 scopes also warm the filing-metadata cache.

| Stage | Container | Produces |
|-------|-----------|----------|
| `download` | nemo-skills | `step-0-download/` filings, `sec_metadata.parquet`, extracted sections |
| `build_index` | nemo-skills | `step-1-index/<scope>_sec_fts.sqlite` and `….sqlite.metadata` |
| `prefetch_cache` | nemo-gym | `cache/finance_sec_search/` filing metadata and `tickers.json` |

### Stage: download

**File:** `projects/finance/nvfinance/stages/download/download_sec_filings.py`
**Binding:** workflow alias `download` selects `DownloadSecFilingsStage`

> **Note:** The three workflow variants configure the same complete stage with a
> different ticker file:
> - `finance:download/smoke`: 2 companies, 1 year — for pipeline smoke tests
> - `finance:download/demo`: 7 companies (NVDA, AAPL, GOOG, MSFT, CSCO, META, IBM), 4 form types (2020-2024)
> - `finance:download/sp500`: 499 S&P 500 companies, 18 form types (2019-2026)

#### Purpose

Download SEC filings from the EDGAR database, then extract sections from those
whose form type has an extractor.

#### Inputs

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `output_dir` | path | Yes | Directory to save downloaded filings |
| `config` | path | Yes | Path to ticker configuration file |
| `sec_identity_email` | string | Yes | Your email from `SEC_IDENTITY_EMAIL` |
| `sec_identity_company` | string | Yes | Your company from `SEC_IDENTITY_COMPANY` |
| `download_only` | bool | No | Fetch filings and skip section extraction |
| `extract_only` | bool | No | Extract sections from an existing corpus and skip fetching |

#### Outputs

- `${output_dir}/data/` - Downloaded SEC filings by ticker
- `${output_dir}/sec_metadata.parquet` - One row per document, including extracted sections
- `${output_dir}/download-logs/` - Download logs

#### Form types and section extraction

`forms` and the year range reach EDGAR as hard filters, so the ticker config
defines the corpus exactly. Only 10-K, 10-Q and 8-K (including their `/A`
amendments) have section extractors; other form types are downloaded and
indexed, and the extraction step reports how many it skipped.

The search index does not read sections, so a corpus meant only for
`edgar_search` can be fetched with `download_only: true`. Extraction is driven
entirely by `sec_metadata.parquet` — it takes the primary documents of supported
forms and subtracts filings that already have section rows — so running the same
stage later with `extract_only: true` fills in the sections without refetching.

#### Resource Requirements

- **Compute:** CPU only
- **Storage:** ~250MB (demo), ~100GB (sp500)
- **Runtime:** 10 min (demo), days (sp500, at SEC's rate limit)

### Stage: build_index

**File:** `projects/finance/nvfinance/stages/download/build_index.py`
**CLI:** `projects/finance/nvfinance/utils/index/build_index.py`
**Registry:** `project="finance"`, `workflow="download-sec"`, `stage="build_index"`

#### Purpose

Build both files NeMo-Gym's `edgar_search` tool opens, from the primary
documents and exhibits recorded in `sec_metadata.parquet`. Extracted sections
are skipped, since they are fragments of documents already indexed.

- the SQLite FTS5 index searches run against
- beside it, a `.metadata` companion holding the same rows without the filing
  text. Searches need every candidate's form type and date, and reading those
  out of a table that also stores filing bodies is slow enough that NeMo-Gym
  refuses to serve without the companion on a large index.

Both come out of one job, so an index is never published without a companion
that matches it.

#### Inputs

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `artifact_root` | path | Yes | The download stage's output directory |
| `index_path` | path | Yes | Destination `.sqlite` file |
| `incremental` | bool | No | Resume an unfinished build, or add only documents the index does not already hold (default `true`) |
| `allow_missing_paths` | bool | No | Publish even when parquet rows have no file on disk |

#### Outputs

- `index_path` - the search index, published only after the build succeeds
- `${index_path}.metadata` - the companion, published just before the index.
  NeMo-Gym discovers it by name, so it must stay in the same directory as the
  index; copying an index without it makes the environment fail to start
- `${index_path}.building` - left behind when a build is rejected, for inspection

#### The missing-path gate

A parquet row whose file is absent is skipped rather than treated as an error,
so a build over an incomplete corpus would otherwise succeed while quietly
omitting filings. The stage therefore builds to a temporary name and publishes
by rename only when no row was missing its file; otherwise it fails with the
count and example paths.

The fix is to re-run the download stage until every ticker reports zero filings
to download, which is both the detection and the repair. `allow_missing_paths`
exists for the case where the gap is known and accepted.

#### Resource Requirements

- **Compute:** CPU only
- **Storage:** ~150MB (demo), ~30GB (sp500)
- **Runtime:** 5 min (demo), ~26 h (sp500)

### Stage: prefetch_cache

**File:** `projects/finance/nvfinance/stages/download/prefetch_cache.py`
**Registry:** `project="finance"`, `workflow="download-sec"`, `stage="prefetch_cache"`

#### Purpose

Warm what NeMo-Gym's `sec_filing_search` tool looks up by ticker: each company's
filing list and the ticker-to-CIK registry. Without this, every rollout and eval
question resolves tickers against sec.gov on the critical path, where the rate
limit turns into rollout timeouts and concurrent runs race on the same cache.

This is a different tool from `edgar_search`, which `build_index` serves. Both
are supported, so both artifacts are built.

Only filing **metadata** is prefetched. Filing bodies are cached on first read,
so the cache keeps growing during training and eval.

#### Inputs

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `cache_dir` | path | Yes | Destination cache directory |
| `ticker_config` | path | Yes | Ticker config to warm; mirrors the download scope |
| `script` | path | Yes | Prefetch script, relative to `gym_path` |
| `gym_path` | path | Yes | NeMo-Gym checkout inside the container |
| `supplementary_tickers` | path | No | Extra companies to warm beyond `ticker_config` |
| `force` | bool | No | Re-fetch companies already cached |

#### Outputs

- `${cache_dir}/filings_metadata/{CIK}.json` - one file per company
- `${cache_dir}/tickers.json` - the ticker-to-CIK registry
- `${cache_dir}/logs/` - prefetch logs

#### Who reads the cache

The `cache_dir` in the `finance_sec_search` overlays must match the path this
stage writes. A mismatch does not fail: the tool reports itself unavailable and
the run still completes, just without filing search.

- GRPO demo and `eval/demo`'s `fabv1` read the demo scope's cache
- GRPO sp500 and `eval/baselines`' `fabv1` read the sp500 scope's cache
- `fabv1` asks about companies beyond the demo tickers; the tool fetches those
  from sec.gov on first use and adds them to the demo cache

#### Supplementary tickers

The agentic eval set asks about companies that are deliberately outside the
S&P 500, so the sp500 and demo scopes both pass `supplementary_tickers` to warm
them too. NeMo-Gym ships that list with the resource server.

#### Resource Requirements

- **Compute:** CPU only
- **Storage:** a few hundred MB
- **Runtime:** seconds (demo), minutes (sp500, at SEC's rate limit)
