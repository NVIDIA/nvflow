# Workflow 1: Download SEC Filings

## Purpose

Download SEC 10-K, 10-Q, and 8-K filings from the EDGAR database for specified companies.

## When to Use

- **First step** before any SDG workflow
- When you need to update to newer SEC filings
- To download filings for specific companies or date ranges

## Prerequisites

Before running this workflow, ensure you have:

- ✅ **Output directory writable** (workflow creates `outputs/finance/demo/workflow-2-download-sec/step-0-download`)

- ✅ **SEC EDGAR identity configured** in your workflow YAML:
  ```yaml
  stages:
    demo:  # or sap-500
      sec_identity_email: your.email@company.com
      sec_identity_company: YourCompany
  ```
  - **Why needed:** The SEC EDGAR API requires all applications to identify themselves per their [Fair Access Policy](https://www.sec.gov/os/accessing-edgar-data)
  - **Required format:** Valid email address and company/organization name

## How It Works

This workflow has 2 steps:

1. **NVFlow Stage** (`nvflow/recipes/finance/stages/download/download_sec_filings.py`)
   - Orchestrates the download job on your Slurm cluster
   - Manages containerized execution and output paths

2. **SEC Downloader Tool** (bundled in NVFlow)
   - Interfaces directly with the SEC EDGAR API
   - Downloads filings in HTML format
   - Parses and extracts individual sections (Items 1, 1A, 2, etc.) from 10-K/10-Q filings
   - Generates metadata file (`sec_metadata.parquet`) for downstream stages

The NVFlow stage handles cluster integration and output management while the bundled SEC downloader tool handles SEC API interaction and parsing.

## Configuration Files

- **Demo:** `configs/demo.yaml` - 7 companies (NVDA, AAPL, GOOG, MSFT, CSCO, META, IBM)
- **Production:** `configs/sp500.yaml` - 500+ S&P 500 companies

## Workflow Overview

```
┌─────────────────────────┐
│ Input                   │
├─────────────────────────┤
│ • Ticker list           │
│ • Date range            │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ sap-500 / demo          │  Download: SEC filings from EDGAR
└───────────┬─────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────────────┐
│ Output                                                         │
├────────────────────────────────────────────────────────────────┤
│ outputs/finance/{demo,sap-500}/workflow-2-download-sec/       │
│ └── step-0-download/                                          │
│     ├── data/                                                  │
│     │   └── {TICKER}/                                          │
│     │       ├── 10-K/{YEAR}/{ACC_NO}/                          │
│     │       ├── 10-Q/{YEAR}/{ACC_NO}/                          │
│     │       └── 8-K/{YEAR}/{ACC_NO}/                           │
│     └── sec_metadata.parquet                                   │
└────────────────────────────────────────────────────────────────┘
```

## Stage: sap-500

Downloads SEC filings using the bundled `sec-downloader-parser` tool.

**See [technical reference](../stages/download-sec.md) for detailed parameters and configuration.**

## Usage

### Demo (Recommended for First Time)

```bash
# Use demo stage (7 companies)
uv run nflow run demo --config nvflow/recipes/finance/workflows/download_sec_filings.yaml
```

### Production

```bash
# Use sap-500 stage (500+ S&P 500 companies)
uv run nflow run sap-500 --config nvflow/recipes/finance/workflows/download_sec_filings.yaml
```

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
    ├── sec_metadata.parquet      # Metadata for all filings
    └── download-logs/            # Download logs
```

## Expected Results

| Config | Companies | Filings | Storage | Time |
|--------|-----------|---------|---------|------|
| Demo | 7 | ~50-70 | ~250MB | 10 min |
| Production | 500+ | ~10K | ~100GB | 2 hours |

## Validation

```bash
# Check number of companies (from nvflow directory)
ls outputs/finance/demo/workflow-2-download-sec/step-0-download/data/ | wc -l
# Demo: 7, Production: 500+

# Check metadata
python -c "import pandas as pd; df = pd.read_parquet('outputs/finance/demo/workflow-2-download-sec/step-0-download/sec_metadata.parquet'); print(f'Total filings: {len(df[df.file_type ==\"primary_document\"])}')"

# Check storage used
du -sh outputs/finance/demo/workflow-2-download-sec/step-0-download/
```

## Next Steps

After downloading SEC filings, you can:

- **[Template-Based SDG](02-template-based-sdg.md)** - Generate Q&A from seed questions
- **[Document-Grounded SDG](03-document-grounded-sdg.md)** - Generate verified Q&A from documents

Both SDG workflows consume the downloaded filings.

## Common Issues

### SEC Rate Limiting

**Symptom:** Download fails with rate limit errors

**Solution:**
- The downloader includes automatic throttling
- Wait 10 minutes and retry
- Ensure `sec_identity_email` and `sec_identity_company` are properly set (required by SEC)

**Symptom:** ReadTimeout Error (httpx.ReadTimeout: The read operation timed out)

**Solution:**
- There was a network delay in downloading the file so the read timed out
- Wait a few minutes and retry

### Storage Space

**Symptom:** Disk full during download

**Solution:**
- S&P 500 filings require ~100GB
- Use demo.yaml first to test with minimal storage

## Configuration Parameters

See [sap-500 stage reference](../stages/download-sec.md) for detailed parameter documentation.

Key parameters:
- `config`: Path to ticker configuration file
- `output_dir`: Where to save filings (default: `/workspace/outputs/finance/demo/workflow-2-download-sec/step-0-download`)
- `sec_identity_email`: Your email (required by SEC EDGAR)
- `sec_identity_company`: Your company name (required)

## Technical Details

For comprehensive technical documentation, see:
- **[sap-500 Stage Reference](../stages/download-sec.md)**
