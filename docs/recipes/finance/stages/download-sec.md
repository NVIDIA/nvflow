# Download-SEC Stages Reference

Technical reference for the download-sec workflow stage.

## Stage: smoke / demo / sap-500

**File:** `nvflow/recipes/finance/stages/download/download_sec_filings.py`
**Registry:** `recipe="finance"`, `workflow="download-sec"`, `stage="smoke"`, `stage="demo"` and `stage="sap-500"`

> **Note:** All three stages share one implementation and differ only in the ticker config they load:
> - `smoke`: 2 companies, 1 year — for pipeline smoke tests
> - `demo`: 7 companies (NVDA, AAPL, GOOG, MSFT, CSCO, META, IBM) with 10-K and 10-Q forms (2020-2024)
> - `sap-500`: 500+ S&P 500 companies with 10-K, 10-Q, and 8-K forms

### Purpose

Download SEC filings from EDGAR database using the sec-downloader-parser tool.

### Inputs

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `output_dir` | path | Yes | Directory to save downloaded filings |
| `config` | path | Yes | Path to ticker configuration file |
| `sec_identity_email` | string | Yes | Your email (required by SEC) |
| `sec_identity_company` | string | Yes | Your company name (required by SEC) |

### Outputs

- `${output_dir}/step-0-download/data/` - Downloaded SEC filings by ticker
- `${output_dir}/step-0-download/sec_metadata.parquet` - Filing metadata
- `${output_dir}/step-0-download/download-logs/` - Download logs

### Resource Requirements

- **Compute:** CPU only
- **Storage:** ~100MB (demo), ~100GB (sp500)
- **Runtime:** 5 min (demo), 2 hours (sp500)

See [Workflow 1](../workflows/01-download-sec.md) for usage examples.
