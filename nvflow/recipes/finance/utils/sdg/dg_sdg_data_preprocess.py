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
"""
Unified SEC Data Preprocessing Script.

This script combines the functionality of:
1. dg_sdg_chunk.py - Chunk SEC 10-K/10-Q HTML files into Markdown, Clean HTML, and Original HTML
2. dg_sdg_get_chunk_list.py - Generate CSV file lists from chunked files
3. dg_sdg_10k.py / dg_sdg_10q.py - Generate JSONL training data from CSVs

Usage:
    python dg_sdg_data_preprocess.py --input_dir /path/to/sec_filings \
        --output_dir /path/to/output \
        --distribution_dir /path/to/distributions \
        --total_samples 150000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

# External dependencies
from bs4 import BeautifulSoup, NavigableString, Tag

from nvflow.utils import setup_logger

logger = setup_logger(__name__)

try:
    import tiktoken

    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Constants
DEFAULT_MODEL = "gpt-4"
HEADING_TAGS = {"h1", "h2", "h3", "h4"}
SKIP_DIR_NAMES = {"chunked_html", "chunks", "chunked_unified"}

Block = dict[str, Any]


# =============================================================================
# Part 1: Chunking Functions (from dg_sdg_chunk.py)
# =============================================================================


def get_encoder(model_name: str = DEFAULT_MODEL):
    """Get tiktoken encoder for token counting."""
    if not HAS_TIKTOKEN:
        # Fallback: approximate tokens as words / 0.75
        class FakeEncoder:
            def encode(self, text):
                return text.split()

        return FakeEncoder()

    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def extract_head_assets(soup: BeautifulSoup) -> str:
    """Return <style> and stylesheet <link> tags for the Original HTML."""
    head = soup.find("head")
    if not head:
        return ""

    assets: list[str] = []
    for tag in head.find_all(["style", "link"]):
        if tag.name == "style":
            assets.append(str(tag))
        elif tag.name == "link":
            rel = tag.get("rel", [])
            if rel and any("stylesheet" in r.lower() for r in rel):
                assets.append(str(tag))
    return "\n".join(assets)


def clean_table_html(table_tag: Tag) -> str:
    """Returns a CLEAN HTML string for the table (removed styles/classes, preserved structure)."""
    rows_html = []
    for tr in table_tag.find_all("tr"):
        cells_html = []
        for cell in tr.find_all(["td", "th"]):
            text = cell.get_text(" ", strip=True)
            colspan = cell.get("colspan")
            rowspan = cell.get("rowspan")

            attrs = ""
            if colspan:
                attrs += f' colspan="{colspan}"'
            if rowspan:
                attrs += f' rowspan="{rowspan}"'

            tag_name = cell.name
            cells_html.append(f"<{tag_name}{attrs}>{text}</{tag_name}>")

        rows_html.append(f"<tr>{''.join(cells_html)}</tr>")

    return f"<table>{''.join(rows_html)}</table>"


def table_to_markdown(table_tag: Tag) -> str:
    """Returns a Markdown pipe table string."""
    rows = []
    for tr in table_tag.find_all("tr"):
        cells = []
        for cell in tr.find_all(["td", "th"]):
            cell_text = cell.get_text(" ", strip=True)
            cell_text = cell_text.replace("|", r"\|")
            cells.append(cell_text)
        if cells:
            rows.append(cells)

    if not rows:
        return ""

    md_lines = []
    for i, row in enumerate(rows):
        line = "| " + " | ".join(row) + " |"
        md_lines.append(line)
        if i == 0:
            col_count = len(row)
            sep = "|" + "|".join(["---"] * col_count) + "|"
            md_lines.append(sep)

    return "\n".join(md_lines)


def process_element(element: Any) -> list[tuple[str, str, str]]:
    """Recursive function to process elements. Returns list of (markdown, clean_html, original_html)."""
    results = []

    if isinstance(element, NavigableString):
        text = str(element).strip()
        if text:
            results.append((text, text, str(element)))
        return results

    if isinstance(element, Tag):
        if element.name == "table":
            md = table_to_markdown(element)
            clean_html = clean_table_html(element)
            orig_html = str(element)
            results.append((md, clean_html, orig_html))
            return results

        if element.name in ["script", "style", "meta", "link", "base", "title", "head"]:
            return results

        if element.find("table"):
            for child in element.children:
                results.extend(process_element(child))
        else:
            text = element.get_text(" ", strip=True)
            if text:
                results.append((text, text, str(element)))

    return results


def html_to_sections(soup: BeautifulSoup, encoder, min_tokens: int = 1) -> list[list[Block]]:
    """Convert HTML to logical sections with 3 formats."""
    body = soup.body or soup

    raw_items = []
    for child in body.children:
        raw_items.extend(process_element(child))

    sections: list[list[Block]] = []
    current_section: list[Block] = []

    for md, clean, orig in raw_items:
        is_table = md.strip().startswith("|")
        tag_label = "table" if is_table else "p"

        token_count = len(encoder.encode(md)) or min_tokens

        block: Block = {
            "md": md,
            "clean_html": clean,
            "orig_html": orig,
            "tag": tag_label,
            "tokens": token_count,
        }

        if is_table:
            if current_section:
                sections.append(current_section)
            sections.append([block])
            current_section = []
            continue

        current_section.append(block)

    if current_section:
        sections.append(current_section)

    return sections


def _split_large_section(section: Sequence[Block], max_tokens: int) -> list[list[Block]]:
    """Split large sections into smaller chunks."""
    split_chunks: list[list[Block]] = []
    current: list[Block] = []
    token_total = 0

    def flush_current():
        nonlocal current, token_total
        if current:
            split_chunks.append(current)
            current = []
            token_total = 0

    for block in section:
        block_tokens = block["tokens"]

        if block_tokens > max_tokens:
            flush_current()
            split_chunks.append([block])
            continue

        if token_total + block_tokens > max_tokens and current:
            flush_current()

        current.append(block)
        token_total += block_tokens

    flush_current()
    return split_chunks


def chunk_sections(sections: Sequence[Sequence[Block]], max_tokens: int) -> list[list[Block]]:
    """Chunk sections based on token limits."""
    chunks: list[list[Block]] = []
    current_blocks: list[Block] = []
    token_count = 0

    def flush_current():
        nonlocal current_blocks, token_count
        if current_blocks:
            chunks.append(current_blocks)
            current_blocks = []
            token_count = 0

    for section in sections:
        section_tokens = sum(block["tokens"] for block in section)

        if token_count + section_tokens <= max_tokens:
            current_blocks.extend(section)
            token_count += section_tokens
            continue

        if section_tokens > max_tokens:
            flush_current()
            sub_chunks = _split_large_section(section, max_tokens)
            chunks.extend(sub_chunks)
            continue

        flush_current()
        current_blocks.extend(section)
        token_count += section_tokens

    flush_current()
    return chunks


def apply_overlap(chunks: Sequence[list[Block]], overlap_tokens: int) -> list[list[Block]]:
    """Apply overlap between chunks."""
    if overlap_tokens <= 0:
        return list(chunks)

    overlapped: list[list[Block]] = []
    for idx, chunk in enumerate(chunks):
        if idx == 0:
            overlapped.append(chunk)
            continue

        carry_tokens = 0
        overlap_blocks: list[Block] = []
        prev_chunk = chunks[idx - 1]

        for block in reversed(prev_chunk):
            overlap_blocks.insert(0, block)
            carry_tokens += block["tokens"]
            if carry_tokens >= overlap_tokens:
                break

        overlapped.append(overlap_blocks + chunk)

    return overlapped


def write_chunk_outputs(
    output_dir: Path,
    base_name: str,
    chunk_index: int,
    chunk_blocks: Sequence[Block],
    base_info: str,
    head_assets: str,
    base_href: str | None,
):
    """Write chunk outputs in 3 formats."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Markdown (.txt)
    md_content = "\n\n".join(block["md"] for block in chunk_blocks)
    full_txt = f"{base_info}\n\n{md_content}" if base_info else md_content

    txt_path = output_dir / f"{base_name}_{chunk_index}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_txt)

    # 2. Original HTML (_orig.html)
    orig_body = "\n".join(block["orig_html"] for block in chunk_blocks)
    head_content = [
        '<meta charset="utf-8">',
        f'<base href="{base_href}">' if base_href else "",
        head_assets or "",
    ]
    orig_doc = f"""<html><head>{"".join(head_content)}</head><body>{orig_body}</body></html>"""

    orig_path = output_dir / f"{base_name}_{chunk_index}_orig.html"
    with open(orig_path, "w", encoding="utf-8") as f:
        f.write(orig_doc)

    # 3. Clean HTML (_clean.html)
    clean_body = "\n<br>\n".join(block["clean_html"] for block in chunk_blocks)
    clean_doc = f"""<html><body>{clean_body}</body></html>"""

    clean_path = output_dir / f"{base_name}_{chunk_index}_clean.html"
    with open(clean_path, "w", encoding="utf-8") as f:
        f.write(clean_doc)


def iter_html_files(root: Path, filing_types: Sequence[str], pattern: str) -> Iterable[Path]:
    """Iterate over HTML files in the directory structure."""
    if not root.is_dir():
        raise ValueError(f"Expected directory for input_path, got: {root}")

    normalized_types = {ftype.upper() for ftype in filing_types}

    for ticker_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for filing_type in normalized_types:
            filing_dir = ticker_dir / filing_type
            if not filing_dir.exists():
                continue

            for html_path in sorted(filing_dir.rglob(pattern)):
                if not html_path.is_file() or html_path.suffix.lower() != ".html":
                    continue

                rel_parts = html_path.relative_to(root).parts[:-1]
                if any(part in SKIP_DIR_NAMES for part in rel_parts):
                    continue
                if "exhibits" in rel_parts:
                    continue

                yield html_path


def generate_header_info(html_path: Path) -> str:
    """Generate header info from file path."""
    parts = html_path.parts
    try:
        if "10-K" in parts:
            idx = parts.index("10-K")
            ticker = parts[idx - 1]
            year = parts[idx + 1]
            form = "10-K"
        elif "10-Q" in parts:
            idx = parts.index("10-Q")
            ticker = parts[idx - 1]
            year = parts[idx + 1]
            form = "10-Q"
        else:
            return f"File: {html_path.name}"
        return f"{ticker} {form} form for fiscal year {year}, file: {html_path.name}"
    except Exception:
        return f"File: {html_path.name}"


def process_html_file(
    html_path: Path,
    encoder,
    max_tokens: int,
    overlap_tokens: int,
    output_root: Path,
    rel_parent: Path,
) -> int:
    """Process a single HTML file and create chunks."""
    try:
        with open(html_path, encoding="utf-8") as f:
            html_text = f.read()
    except Exception as e:
        logger.error("Cannot read %s: %s", html_path, e)
        return 0

    soup = BeautifulSoup(html_text, "html.parser")
    head_assets = extract_head_assets(soup)
    base_href = html_path.parent.resolve().as_uri().rstrip("/") + "/"

    sections = html_to_sections(soup, encoder)
    chunks = chunk_sections(sections, max_tokens)
    chunks = apply_overlap(chunks, overlap_tokens)

    if not chunks:
        return 0

    chunk_dir = output_root / rel_parent / html_path.stem
    header_info = generate_header_info(html_path)

    for idx, chunk in enumerate(chunks):
        write_chunk_outputs(
            chunk_dir,
            html_path.stem,
            idx,
            chunk,
            base_info=header_info,
            head_assets=head_assets,
            base_href=base_href,
        )

    return len(chunks)


def run_chunking(
    input_dir: Path,
    output_dir: Path,
    max_tokens: int = 2000,
    overlap_tokens: int = 100,
    model: str = DEFAULT_MODEL,
    filings: list[str] = None,
) -> Path:
    """Run the chunking process on all HTML files."""
    if filings is None:
        filings = ["10-K", "10-Q"]

    logger.info("Starting chunking from %s to %s", input_dir, output_dir)

    encoder = get_encoder(model)
    html_files = list(iter_html_files(input_dir, filings, "*.html"))

    if not html_files:
        logger.warning("No HTML files found under %s", input_dir)
        return output_dir

    total_chunks = 0
    for html_file in html_files:
        rel_parent = html_file.parent.relative_to(input_dir)
        chunks_created = process_html_file(
            html_file,
            encoder=encoder,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            output_root=output_dir,
            rel_parent=rel_parent,
        )
        total_chunks += chunks_created

    logger.info("Processed %d file(s), created %d chunk(s)", len(html_files), total_chunks)
    return output_dir


# =============================================================================
# Part 2: Generate Chunk Lists (from dg_sdg_get_chunk_list.py)
# =============================================================================


def get_relative_path(full_path: str, start_path: str) -> str:
    """Get relative path from start_path."""
    return os.path.relpath(full_path, start_path)


def generate_chunk_lists(chunk_dir: Path, csv_output_dir: Path) -> Path:
    """Generate CSV file lists from chunked files."""
    logger.info("Generating chunk lists from %s to %s", chunk_dir, csv_output_dir)

    os.makedirs(csv_output_dir, exist_ok=True)

    result: dict[str, dict[str, list[dict[str, str]]]] = {"10-K": {}, "10-Q": {}}
    input_root = str(chunk_dir)

    if not os.path.exists(input_root):
        logger.warning("Chunk directory not found: %s", input_root)
        return csv_output_dir

    for company in os.listdir(input_root):
        company_path = os.path.join(input_root, company)
        if not os.path.isdir(company_path):
            continue

        for form_type in ["10-K", "10-Q"]:
            form_path = os.path.join(company_path, form_type)
            if not os.path.exists(form_path):
                continue

            for year in os.listdir(form_path):
                year_path = os.path.join(form_path, year)
                if not os.path.isdir(year_path):
                    continue

                for accession in os.listdir(year_path):
                    accession_path = os.path.join(year_path, accession)
                    if not os.path.isdir(accession_path):
                        continue

                    for item in os.listdir(accession_path):
                        item_path = os.path.join(accession_path, item)

                        if not os.path.isdir(item_path):
                            continue

                        if item == "primary-document":
                            continue

                        if item not in result[form_type]:
                            result[form_type][item] = []

                        files = os.listdir(item_path)
                        chunks = {}

                        for filename in files:
                            if not (filename.endswith(".txt") or filename.endswith(".html")):
                                continue

                            try:
                                name_part = Path(filename).stem

                                file_type = "markdown"
                                if name_part.endswith("_clean"):
                                    file_type = "html_clean"
                                    core_name = name_part[:-6]
                                elif name_part.endswith("_orig"):
                                    file_type = "html_origin"
                                    core_name = name_part[:-5]
                                else:
                                    file_type = "markdown"
                                    core_name = name_part

                                base_name, idx_str = core_name.rsplit("_", 1)
                                idx = int(idx_str)

                                rel_path = get_relative_path(
                                    os.path.join(item_path, filename), input_root
                                )
                                chunk_id = f"{item_path}/{base_name}_{idx}"

                                if chunk_id not in chunks:
                                    chunks[chunk_id] = {
                                        "markdown": "",
                                        "html_clean": "",
                                        "html_origin": "",
                                    }

                                chunks[chunk_id][file_type] = rel_path

                            except Exception:
                                continue

                        for chunk_data in chunks.values():
                            result[form_type][item].append(chunk_data)

    # Write CSVs
    for form_type in ["10-K", "10-Q"]:
        for item, chunks in result[form_type].items():
            csv_filename = f"{form_type.lower().replace('-', '')}_{item}.csv"
            csv_path = os.path.join(csv_output_dir, csv_filename)

            chunks.sort(key=lambda x: x["markdown"])

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["markdown", "html_clean", "html_origin"])
                for chunk in chunks:
                    writer.writerow(
                        [
                            chunk.get("markdown", ""),
                            chunk.get("html_clean", ""),
                            chunk.get("html_origin", ""),
                        ]
                    )
            logger.info("Generated %s with %d rows", csv_path, len(chunks))

    return csv_output_dir


# =============================================================================
# Part 3: Generate JSONL Data (from dg_sdg_10k.py / dg_sdg_10q.py)
# =============================================================================


def load_distribution(distribution_path: str) -> dict[str, int]:
    """Load distribution file into a dictionary."""
    if not HAS_PANDAS:
        # Fallback to csv module
        dist = {}
        with open(distribution_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                dist[row["item_section"]] = int(row["count"])
        return dist

    df = pd.read_csv(distribution_path)
    return {row["item_section"]: row["count"] for _, row in df.iterrows()}


def weighted_random_choice(distribution_dict: dict[str, int]) -> str:
    """Choose an item based on weighted distribution."""
    items = sorted(distribution_dict.keys())
    weights = [distribution_dict[item] for item in items]
    total_weight = sum(weights)
    if total_weight == 0:
        return random.choice(items)
    return random.choices(items, weights=weights, k=1)[0]


def generate_jsonl_data(
    chunk_dir: Path,
    csv_dir: Path,
    distribution_dir: Path,
    output_dir: Path,
    form_type: str,  # "10-K" or "10-Q"
    total_samples: int = 150000,
    max_skip_count: int = 20000,
    seed: int = 42,
) -> int:
    """Generate JSONL data for a specific form type."""
    random.seed(seed)

    form_lower = form_type.lower().replace("-", "")
    distribution_1_company_path = distribution_dir / f"{form_lower}_1company_distribution.csv"
    distribution_2_company_path = distribution_dir / f"{form_lower}_2company_distribution.csv"

    if not distribution_1_company_path.exists() or not distribution_2_company_path.exists():
        logger.warning("Distribution files not found for %s, skipping", form_type)
        return 0

    output_file = output_dir / f"{form_type.lower()}-data.jsonl"
    data_path = str(chunk_dir)
    csv_path_root = str(csv_dir)

    logger.info("Generating %s data from %s", form_type, csv_path_root)

    # Build data dictionaries
    company_data_dict: dict[str, dict[str, dict[str, list[str]]]] = {}
    item_data_dict: dict[str, list[str]] = {}

    if not os.path.exists(csv_path_root):
        logger.warning("CSV directory not found: %s", csv_path_root)
        return 0

    file_list = sorted(os.listdir(csv_path_root))
    for file_name in file_list:
        if form_lower in file_name and file_name.endswith(".csv"):
            item_name = file_name.replace(".csv", "")
            if item_name not in item_data_dict:
                item_data_dict[item_name] = []

            file_path = os.path.join(csv_path_root, file_name)
            try:
                if HAS_PANDAS:
                    df = pd.read_csv(file_path)
                    rows = df.iterrows()
                else:
                    with open(file_path) as f:
                        reader = csv.DictReader(f)
                        rows = list(enumerate(list(reader)))
            except Exception as e:
                logger.warning("Error reading %s: %s", file_path, e)
                continue

            for _, row in rows:
                if HAS_PANDAS:
                    markdown_path = row["markdown"]
                    if pd.isna(markdown_path):
                        continue
                else:
                    markdown_path = row.get("markdown", "")
                    if not markdown_path:
                        continue

                parts = markdown_path.split("/")
                try:
                    if form_type in parts:
                        idx = parts.index(form_type)
                        company_name = parts[idx - 1]
                        year = parts[idx + 1]
                    else:
                        company_name = parts[0]
                        year = parts[2]
                except IndexError:
                    continue

                if company_name not in company_data_dict:
                    company_data_dict[company_name] = {}
                if item_name not in company_data_dict[company_name]:
                    company_data_dict[company_name][item_name] = {}
                if year not in company_data_dict[company_name][item_name]:
                    company_data_dict[company_name][item_name][year] = []

                company_data_dict[company_name][item_name][year].append(markdown_path)

                if company_name not in item_data_dict[item_name]:
                    item_data_dict[item_name].append(company_name)

    for item in item_data_dict:
        item_data_dict[item].sort()

    def remove_chunk_from_dict(comp, item, yr, path):
        if (
            comp in company_data_dict
            and item in company_data_dict[comp]
            and yr in company_data_dict[comp][item]
        ):
            if path in company_data_dict[comp][item][yr]:
                company_data_dict[comp][item][yr].remove(path)
                if len(company_data_dict[comp][item][yr]) == 0:
                    del company_data_dict[comp][item][yr]
                    if not company_data_dict[comp][item]:
                        del company_data_dict[comp][item]
                        if not company_data_dict[comp]:
                            del company_data_dict[comp]
                        if comp in item_data_dict.get(item, []):
                            item_data_dict[item].remove(comp)

    # Load distributions
    distribution_1_company = load_distribution(str(distribution_1_company_path))
    distribution_2_company = load_distribution(str(distribution_2_company_path))

    distribution_1_company_count = sum(distribution_1_company.values())
    distribution_2_company_count = sum(distribution_2_company.values())

    total_distribution_count = distribution_1_company_count + distribution_2_company_count
    if total_distribution_count == 0:
        total_distribution_count = 1
        distribution_1_company_count = 1

    # Generate samples
    company1_result = []
    company2_result = []
    question_count = 0
    skip_count = 0

    stats = {"1_company": 0, "2_company": 0, "item_usage": defaultdict(int)}

    while question_count < total_samples:
        one_company_probability = distribution_1_company_count / total_distribution_count
        if random.random() < one_company_probability:
            selected_company_type = "1_company"
            selected_distribution = distribution_1_company
        else:
            selected_company_type = "2_company"
            selected_distribution = distribution_2_company

        try:
            chosen_item_sections_str = weighted_random_choice(selected_distribution)
        except (IndexError, ValueError):
            break

        chosen_item_sections = chosen_item_sections_str.split(";")

        valid_choice = True
        for sec in chosen_item_sections:
            if sec not in item_data_dict:
                valid_choice = False
                break
        if not valid_choice:
            skip_count += 1
            if skip_count > max_skip_count:
                break
            continue

        success = False
        result = {}

        if selected_company_type == "1_company":
            if len(chosen_item_sections) == 1:
                chosen_item_section = chosen_item_sections[0]
                if len(item_data_dict.get(chosen_item_section, [])) == 0:
                    skip_count += 1
                    if skip_count > max_skip_count:
                        break
                    continue
                company_name = random.choice(item_data_dict[chosen_item_section])
                year_list = sorted(company_data_dict[company_name][chosen_item_section].keys())
                year = random.choice(year_list)
                if len(company_data_dict[company_name][chosen_item_section][year]) == 0:
                    skip_count += 1
                    continue
                chosen_md_path = random.choice(
                    company_data_dict[company_name][chosen_item_section][year]
                )

                remove_chunk_from_dict(company_name, chosen_item_section, year, chosen_md_path)

                result["year"] = year
                result["company_name"] = company_name
                result["item_section0"] = chosen_item_section
                result["markdown_path0"] = chosen_md_path
                company1_result.append(result)
                question_count += 1
                success = True

            elif len(chosen_item_sections) == 2:
                chosen_item_section1 = chosen_item_sections[0]
                chosen_item_section2 = chosen_item_sections[1]
                if (
                    len(item_data_dict.get(chosen_item_section1, [])) == 0
                    or len(item_data_dict.get(chosen_item_section2, [])) == 0
                ):
                    skip_count += 1
                    if skip_count > max_skip_count:
                        break
                    continue

                company_list = sorted(
                    set(item_data_dict[chosen_item_section1])
                    & set(item_data_dict[chosen_item_section2])
                )
                if not company_list:
                    skip_count += 1
                    continue
                company_name = random.choice(company_list)
                sec1_years = list(company_data_dict[company_name][chosen_item_section1].keys())
                sec2_years = list(company_data_dict[company_name][chosen_item_section2].keys())
                if len(sec1_years) == 0 or len(sec2_years) == 0:
                    skip_count += 1
                    continue
                year_intersection = sorted(set(sec1_years) & set(sec2_years))
                if len(year_intersection) == 0:
                    skip_count += 1
                    continue
                chosen_year = random.choice(year_intersection)
                chosen_md_path1 = random.choice(
                    company_data_dict[company_name][chosen_item_section1][chosen_year]
                )
                chosen_md_path2 = random.choice(
                    company_data_dict[company_name][chosen_item_section2][chosen_year]
                )

                remove_chunk_from_dict(
                    company_name, chosen_item_section1, chosen_year, chosen_md_path1
                )
                remove_chunk_from_dict(
                    company_name, chosen_item_section2, chosen_year, chosen_md_path2
                )

                result["company_name"] = company_name
                result["year"] = chosen_year
                result["item_section0"] = chosen_item_section1
                result["item_section1"] = chosen_item_section2
                result["markdown_path0"] = chosen_md_path1
                result["markdown_path1"] = chosen_md_path2
                company1_result.append(result)
                question_count += 1
                success = True
        else:
            # 2_company logic
            if len(chosen_item_sections) == 1:
                chosen_item_section = chosen_item_sections[0]
                if len(item_data_dict.get(chosen_item_section, [])) < 2:
                    skip_count += 1
                    if skip_count > max_skip_count:
                        break
                    continue

                companies = random.sample(item_data_dict[chosen_item_section], 2)
                company_name_1 = companies[0]
                company_name_2 = companies[1]

                result["company_name0"] = company_name_1
                result["company_name1"] = company_name_2

                sec1_years = list(company_data_dict[company_name_1][chosen_item_section].keys())
                sec2_years = list(company_data_dict[company_name_2][chosen_item_section].keys())

                year_intersection = sorted(set(sec1_years) & set(sec2_years))
                if len(year_intersection) == 0:
                    skip_count += 1
                    continue
                chosen_year = random.choice(year_intersection)

                chosen_md_path1 = random.choice(
                    company_data_dict[company_name_1][chosen_item_section][chosen_year]
                )
                chosen_md_path2 = random.choice(
                    company_data_dict[company_name_2][chosen_item_section][chosen_year]
                )

                remove_chunk_from_dict(
                    company_name_1, chosen_item_section, chosen_year, chosen_md_path1
                )
                remove_chunk_from_dict(
                    company_name_2, chosen_item_section, chosen_year, chosen_md_path2
                )

                result["year"] = chosen_year
                result["item_section0"] = chosen_item_section
                result["item_section1"] = chosen_item_section
                result["markdown_path0"] = chosen_md_path1
                result["markdown_path1"] = chosen_md_path2

                company2_result.append(result)
                question_count += 1
                success = True

            elif len(chosen_item_sections) == 2:
                chosen_item_section1 = chosen_item_sections[0]
                chosen_item_section2 = chosen_item_sections[1]

                if (
                    len(item_data_dict.get(chosen_item_section1, [])) == 0
                    or len(item_data_dict.get(chosen_item_section2, [])) == 0
                ):
                    skip_count += 1
                    continue

                company_name_1 = random.choice(item_data_dict[chosen_item_section1])
                candidates_2 = [
                    c for c in item_data_dict[chosen_item_section2] if c != company_name_1
                ]
                if not candidates_2:
                    skip_count += 1
                    continue
                company_name_2 = random.choice(candidates_2)

                result["company_name0"] = company_name_1
                result["company_name1"] = company_name_2

                sec1_years = list(company_data_dict[company_name_1][chosen_item_section1].keys())
                sec2_years = list(company_data_dict[company_name_2][chosen_item_section2].keys())

                year_intersection = sorted(set(sec1_years) & set(sec2_years))
                if len(year_intersection) == 0:
                    skip_count += 1
                    continue
                chosen_year = random.choice(year_intersection)

                chosen_md_path1 = random.choice(
                    company_data_dict[company_name_1][chosen_item_section1][chosen_year]
                )
                chosen_md_path2 = random.choice(
                    company_data_dict[company_name_2][chosen_item_section2][chosen_year]
                )

                remove_chunk_from_dict(
                    company_name_1, chosen_item_section1, chosen_year, chosen_md_path1
                )
                remove_chunk_from_dict(
                    company_name_2, chosen_item_section2, chosen_year, chosen_md_path2
                )

                result["year"] = chosen_year
                result["item_section0"] = chosen_item_section1
                result["item_section1"] = chosen_item_section2
                result["markdown_path0"] = chosen_md_path1
                result["markdown_path1"] = chosen_md_path2

                company2_result.append(result)
                question_count += 1
                success = True

            elif len(chosen_item_sections) >= 3:
                s1 = chosen_item_sections[0]
                s2 = chosen_item_sections[1]
                s3 = chosen_item_sections[2]

                if (
                    len(item_data_dict.get(s1, [])) == 0
                    or len(item_data_dict.get(s2, [])) == 0
                    or len(item_data_dict.get(s3, [])) == 0
                ):
                    skip_count += 1
                    continue

                company_name_1 = random.choice(item_data_dict[s1])

                candidates_2 = sorted(set(item_data_dict[s2]) & set(item_data_dict[s3]))
                if company_name_1 in candidates_2:
                    candidates_2.remove(company_name_1)
                if not candidates_2:
                    skip_count += 1
                    continue
                company_name_2 = random.choice(candidates_2)

                sec1_years = sorted(company_data_dict[company_name_1][s1].keys())
                sec2_years = sorted(company_data_dict[company_name_2][s2].keys())
                sec3_years = sorted(company_data_dict[company_name_2][s3].keys())

                year_intersection = sorted(set(sec1_years) & set(sec2_years) & set(sec3_years))
                if not year_intersection:
                    skip_count += 1
                    continue
                chosen_year = random.choice(year_intersection)

                path1 = random.choice(company_data_dict[company_name_1][s1][chosen_year])

                if s2 == s3:
                    available_chunks = company_data_dict[company_name_2][s2][chosen_year]
                    if len(available_chunks) < 2:
                        skip_count += 1
                        continue
                    path2, path3 = random.sample(available_chunks, 2)
                else:
                    path2 = random.choice(company_data_dict[company_name_2][s2][chosen_year])
                    path3 = random.choice(company_data_dict[company_name_2][s3][chosen_year])

                remove_chunk_from_dict(company_name_1, s1, chosen_year, path1)
                remove_chunk_from_dict(company_name_2, s2, chosen_year, path2)
                remove_chunk_from_dict(company_name_2, s3, chosen_year, path3)

                result["company_name0"] = company_name_1
                result["company_name1"] = company_name_2
                result["year"] = chosen_year
                result["item_section0"] = s1
                result["item_section1"] = s2
                result["item_section2"] = s3
                result["markdown_path0"] = path1
                result["markdown_path1"] = path2
                result["markdown_path2"] = path3

                company2_result.append(result)
                question_count += 1
                success = True

        if success:
            if selected_company_type == "1_company":
                stats["1_company"] += 1
            else:
                stats["2_company"] += 1

            for sec in chosen_item_sections:
                stats["item_usage"][sec] += 1

    # Print Statistics
    if question_count > 0:
        logger.info("%s Stats:", form_type)
        logger.info("  Total Generated: %d", question_count)
        logger.info(
            "  1-Company Count: %d (%.1f%%)",
            stats["1_company"],
            stats["1_company"] / question_count * 100,
        )
        logger.info(
            "  2-Company Count: %d (%.1f%%)",
            stats["2_company"],
            stats["2_company"] / question_count * 100,
        )

    # Write Output JSONL
    os.makedirs(output_dir, exist_ok=True)

    item_prefix = "Item " if form_type == "10-K" else ""

    with open(output_file, "w", encoding="utf-8") as out_f:
        for result in company1_result:
            if "item_section1" not in result:
                company_name = result["company_name"]
                year = result["year"]
                item_section = result["item_section0"].replace(f"{form_lower}_", item_prefix)
                md_path = result["markdown_path0"]
                abs_path = os.path.join(data_path, md_path)
                try:
                    with open(abs_path) as f:
                        content = f.read()
                except FileNotFoundError:
                    continue
                out_f.write(
                    json.dumps(
                        {
                            "company_name": company_name,
                            "year": year,
                            "item_section0": item_section,
                            "file_path0": md_path,
                            "content0": content,
                            "file_type": form_type,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            else:
                company_name = result["company_name"]
                year = result["year"]
                item_section0 = result["item_section0"].replace(f"{form_lower}_", item_prefix)
                item_section1 = result["item_section1"].replace(f"{form_lower}_", item_prefix)
                md_path0 = result["markdown_path0"]
                md_path1 = result["markdown_path1"]
                try:
                    with open(os.path.join(data_path, md_path0)) as f:
                        content0 = f.read()
                    with open(os.path.join(data_path, md_path1)) as f:
                        content1 = f.read()
                except FileNotFoundError:
                    continue
                out_f.write(
                    json.dumps(
                        {
                            "company_name": company_name,
                            "year": year,
                            "item_section0": item_section0,
                            "item_section1": item_section1,
                            "file_path0": md_path0,
                            "file_path1": md_path1,
                            "content0": content0,
                            "content1": content1,
                            "file_type": form_type,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        for result in company2_result:
            company_name0 = result["company_name0"]
            company_name1 = result["company_name1"]
            year = result["year"]
            item_section0 = result.get("item_section0", "").replace(f"{form_lower}_", item_prefix)
            item_section1 = result.get("item_section1", "").replace(f"{form_lower}_", item_prefix)

            md_path0 = result["markdown_path0"]
            md_path1 = result["markdown_path1"]

            if "item_section2" in result:
                item_section2 = result.get("item_section2", "").replace(
                    f"{form_lower}_", item_prefix
                )
                md_path2 = result["markdown_path2"]
                try:
                    with open(os.path.join(data_path, md_path0)) as f:
                        content0 = f.read()
                    with open(os.path.join(data_path, md_path1)) as f:
                        content1 = f.read()
                    with open(os.path.join(data_path, md_path2)) as f:
                        content2 = f.read()
                except FileNotFoundError:
                    continue
                out_f.write(
                    json.dumps(
                        {
                            "company_name0": company_name0,
                            "company_name1": company_name1,
                            "year": year,
                            "item_section0": item_section0,
                            "item_section1": item_section1,
                            "item_section2": item_section2,
                            "file_path0": md_path0,
                            "file_path1": md_path1,
                            "file_path2": md_path2,
                            "content0": content0,
                            "content1": content1,
                            "content2": content2,
                            "file_type": form_type,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            else:
                try:
                    with open(os.path.join(data_path, md_path0)) as f:
                        content0 = f.read()
                    with open(os.path.join(data_path, md_path1)) as f:
                        content1 = f.read()
                except FileNotFoundError:
                    continue
                out_f.write(
                    json.dumps(
                        {
                            "company_name0": company_name0,
                            "company_name1": company_name1,
                            "year": year,
                            "item_section0": item_section0,
                            "item_section1": item_section1,
                            "file_path0": md_path0,
                            "file_path1": md_path1,
                            "content0": content0,
                            "content1": content1,
                            "file_type": form_type,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    logger.info(
        "Written %d records to %s", len(company1_result) + len(company2_result), output_file
    )
    return len(company1_result) + len(company2_result)


# =============================================================================
# Main Entry Point
# =============================================================================


def run_full_preprocess(
    input_dir: Path,
    output_dir: Path,
    distribution_dir: Path,
    max_tokens: int = 2000,
    overlap_tokens: int = 100,
    total_samples: int = 150000,
    max_skip_count: int = 20000,
    seed: int = 42,
):
    """Run the full preprocessing pipeline."""
    logger.info("=" * 60)
    logger.info("SEC Data Preprocessing Pipeline")
    logger.info("=" * 60)

    # Step 1: Chunk HTML files
    chunk_output_dir = output_dir / "chunks"
    run_chunking(
        input_dir=input_dir,
        output_dir=chunk_output_dir,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )

    # Step 2: Generate CSV file lists
    csv_output_dir = output_dir / "csv_lists"
    generate_chunk_lists(
        chunk_dir=chunk_output_dir,
        csv_output_dir=csv_output_dir,
    )

    # Step 3: Generate JSONL data for both 10-K and 10-Q
    jsonl_output_dir = output_dir / "jsonl"

    total_10k = generate_jsonl_data(
        chunk_dir=chunk_output_dir,
        csv_dir=csv_output_dir,
        distribution_dir=distribution_dir,
        output_dir=jsonl_output_dir,
        form_type="10-K",
        total_samples=total_samples,
        max_skip_count=max_skip_count,
        seed=seed,
    )

    total_10q = generate_jsonl_data(
        chunk_dir=chunk_output_dir,
        csv_dir=csv_output_dir,
        distribution_dir=distribution_dir,
        output_dir=jsonl_output_dir,
        form_type="10-Q",
        total_samples=total_samples,
        max_skip_count=max_skip_count,
        seed=seed,
    )

    logger.info("=" * 60)
    logger.info("Preprocessing Complete!")
    logger.info("  10-K samples: %d", total_10k)
    logger.info("  10-Q samples: %d", total_10q)
    logger.info("  Output directory: %s", output_dir)
    logger.info("=" * 60)

    return jsonl_output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Unified SEC Data Preprocessing Pipeline")
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Input directory containing raw SEC filings (e.g., /data/sec_filings)",
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True, help="Output directory for processed data"
    )
    parser.add_argument(
        "--distribution_dir",
        type=Path,
        required=True,
        help="Directory containing distribution CSV files",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=2000, help="Maximum tokens per chunk (default: 2000)"
    )
    parser.add_argument(
        "--overlap_tokens",
        type=int,
        default=100,
        help="Overlap tokens between chunks (default: 100)",
    )
    parser.add_argument(
        "--total_samples",
        type=int,
        default=150000,
        help="Total samples to generate per form type (default: 150000)",
    )
    parser.add_argument(
        "--max_skip_count",
        type=int,
        default=20000,
        help="Maximum number of skipped samples before stopping (default: 20000)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_full_preprocess(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        distribution_dir=args.distribution_dir,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        total_samples=args.total_samples,
        max_skip_count=args.max_skip_count,
        seed=args.seed,
    )
