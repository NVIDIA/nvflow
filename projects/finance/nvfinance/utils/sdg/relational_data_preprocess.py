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
"""Relational (multi-document) SEC preprocessing engine.

This is the context-construction engine for relational sampling within the
``document_grounded_sdg`` workflow (``sampling_strategy: relational``). Instead of
building each generation context from a single filing (as ``sampling_strategy:
random`` does), it samples contexts from *related* filings using weighted "relation modes"
(temporal / cross-form / same-year / cross-company), then a downstream prompt asks a
question that must be answered by reasoning ACROSS the relation (e.g. guidance in
filing T vs actual in filing T+1). "Beat-or-Miss" is one prompt over this machinery.

Design: this module is ADDITIVE and does NOT modify the shipped ``document_grounded``
single-document path. It reuses the SEC chunking infrastructure (``run_chunking``,
``iter_html_files``, ``generate_header_info``) from ``dg_sdg_data_preprocess`` and the
distribution helpers from ``nvflow.stages.sdg.document_grounded.sampling`` unchanged, and
self-contains the relation-aware chunk-list / jsonl generation so the relational path
is byte-for-byte the validated engine.

Relation modes (weights; each yields a multi-filing context unless noted):
  - multi_period                 : same company, different periods, same form
  - cross_form                   : same company, different form types (8-K + 10-K)
  - cross_form_multi_period      : both axes
  - same_year_multi_filing       : multiple filings, same fiscal year
  - multi_company_multi_year     : across companies + years
  - multi_company_cross_form     : across companies + forms
  - multi_form                   : >= 3 form types, one company
  - single-doc (fallback)        : one filing (== document_grounded_sdg)

Setting all relation weights to 0.0 reduces to single-document sampling, giving a
built-in equivalence with ``document_grounded_sdg``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

# Reuse the SEC chunking infrastructure from the single-document engine unchanged.
from nvfinance.utils.sdg.dg_sdg_data_preprocess import (
    run_chunking,
)
from nvflow.stages.sdg.document_grounded.sampling import load_distribution, weighted_random_choice
from nvflow.utils import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Forward-looking language detector (used by guidance_prefilter)
# ---------------------------------------------------------------------------
_FORWARD_LOOKING_PATTERNS: list[str] = [
    # Explicit guidance / outlook
    "guidance",
    "outlook",
    "forecast",
    "forward-looking",
    "forward looking",
    # Expectation language
    "we expect",
    "the company expects",
    "management expects",
    "expected to be",
    "expected to range",
    "anticipate",
    "projected",
    "projections",
    # Target / goal language (DEF 14A compensation, strategic plans)
    "target",
    "threshold",
    "performance goal",
    "hurdle",
    "payout",
    "vesting",
    # Planning language
    "plan to",
    "plans to",
    "intend to",
    "intends to",
    "committed to",
    # Range guidance patterns
    "range of $",
    "between $",
    "approximately $",
    "estimated at $",
    # Temporal forward references
    "for the next",
    "for the coming",
    "for fiscal year",
    "full-year",
    "full year",
]


def has_forward_looking_content(text: str) -> bool:
    """Return True if *text* contains forward-looking / guidance language.

    The check is intentionally broad to minimise false negatives.  A false
    positive (a chunk that mentions "target" but has no real guidance) simply
    means one extra LLM call that returns an empty list.
    """
    text_lower = text.lower()
    return any(pat in text_lower for pat in _FORWARD_LOOKING_PATTERNS)


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


def generate_chunk_lists(
    chunk_dir: Path,
    csv_output_dir: Path,
    filings: list[str] | None = None,
) -> Path:
    """Generate CSV file lists from chunked files."""
    if filings is None:
        filings = ["10-K", "10-Q"]

    logger.info(
        "Generating chunk lists from %s to %s (forms: %s)", chunk_dir, csv_output_dir, filings
    )

    os.makedirs(csv_output_dir, exist_ok=True)

    result: dict[str, dict[str, list[dict[str, str]]]] = {ft: {} for ft in filings}
    input_root = str(chunk_dir)

    if not os.path.exists(input_root):
        logger.warning("Chunk directory not found: %s", input_root)
        return csv_output_dir

    for company in os.listdir(input_root):
        company_path = os.path.join(input_root, company)
        if not os.path.isdir(company_path):
            continue

        for form_type in filings:
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

                    all_items = [
                        d
                        for d in os.listdir(accession_path)
                        if os.path.isdir(os.path.join(accession_path, d))
                    ]
                    has_named_sections = any(d != "primary-document" for d in all_items)

                    for item in all_items:
                        item_path = os.path.join(accession_path, item)

                        if item == "primary-document" and has_named_sections:
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
    for form_type in filings:
        for item, chunks_list in result[form_type].items():
            csv_filename = f"{form_type.lower().replace('-', '')}_{item}.csv"
            csv_path = os.path.join(csv_output_dir, csv_filename)

            chunks_list.sort(key=lambda x: x["markdown"])

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["markdown", "html_clean", "html_origin"])
                for chunk in chunks_list:
                    writer.writerow(
                        [
                            chunk.get("markdown", ""),
                            chunk.get("html_clean", ""),
                            chunk.get("html_origin", ""),
                        ]
                    )
            logger.info("Generated %s with %d rows", csv_path, len(chunks_list))

    return csv_output_dir


# =============================================================================
# Part 3: Generate JSONL Data (from dg_sdg_10k.py / dg_sdg_10q.py)
# =============================================================================


def generate_jsonl_data(
    chunk_dir: Path,
    csv_dir: Path,
    distribution_dir: Path,
    output_dir: Path,
    form_type: str,
    total_samples: int = 150000,
    max_skip_count: int = 20000,
    seed: int = 42,
    guidance_prefilter: bool = False,
) -> int:
    """Generate JSONL data for a specific form type."""
    random.seed(seed)

    form_lower = form_type.lower().replace("-", "")
    distribution_1_company_path = distribution_dir / f"{form_lower}_1company_distribution.csv"
    distribution_2_company_path = distribution_dir / f"{form_lower}_2company_distribution.csv"

    if not distribution_1_company_path.exists() or not distribution_2_company_path.exists():
        if not _auto_generate_distribution(csv_dir, distribution_dir, form_type):
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
                if guidance_prefilter and not has_forward_looking_content(content):
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
                if guidance_prefilter and not (
                    has_forward_looking_content(content0) or has_forward_looking_content(content1)
                ):
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
                if guidance_prefilter and not (
                    has_forward_looking_content(content0)
                    or has_forward_looking_content(content1)
                    or has_forward_looking_content(content2)
                ):
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
                if guidance_prefilter and not (
                    has_forward_looking_content(content0) or has_forward_looking_content(content1)
                ):
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
# Part 4: Multi-Period Sampling
# =============================================================================


def generate_multi_period_jsonl_data(
    chunk_dir: Path,
    csv_dir: Path,
    distribution_dir: Path,
    output_dir: Path,
    form_type: str,
    total_samples: int = 5000,
    max_skip_count: int = 20000,
    seed: int = 42,
    min_periods: int = 2,
    max_periods: int = 4,
    section_boost: dict[str, float] | None = None,
    guidance_prefilter: bool = False,
) -> int:
    """Generate JSONL data pairing the same section across consecutive years.

    This enables temporal reasoning questions such as "Trends" (multi-year 10-K)
    and "Beat or Miss" (sequential 8-K earnings disclosures).  Each record
    contains 2-4 content fields from the same company, same item section, but
    different years.

    Args:
        section_boost: Optional mapping from item-section suffix (e.g. ``"2-2"``)
            to a multiplier (e.g. ``5.0``).  Eligible pairs whose item section
            ends with a boosted suffix receive proportionally higher sampling
            weight, biasing towards sections that are especially valuable for
            certain question categories (e.g. 8-K Item 2.02 for Beat_or_Miss).
    """
    random.seed(seed)

    form_lower = form_type.lower().replace("-", "")
    csv_path_root = str(csv_dir)

    if not os.path.exists(csv_path_root):
        logger.warning("CSV directory not found: %s", csv_path_root)
        return 0

    company_data: dict[str, dict[str, dict[str, list[str]]]] = {}
    item_data: dict[str, list[str]] = {}

    for file_name in sorted(os.listdir(csv_path_root)):
        if not (form_lower in file_name and file_name.endswith(".csv")):
            continue

        item_name = file_name.replace(".csv", "")
        if item_name not in item_data:
            item_data[item_name] = []

        file_path = os.path.join(csv_path_root, file_name)
        try:
            if HAS_PANDAS:
                df = pd.read_csv(file_path)
                rows = df.iterrows()
            else:
                with open(file_path) as f:
                    reader = csv.DictReader(f)
                    rows = list(enumerate(list(reader)))
        except Exception:
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

            if company_name not in company_data:
                company_data[company_name] = {}
            if item_name not in company_data[company_name]:
                company_data[company_name][item_name] = {}
            if year not in company_data[company_name][item_name]:
                company_data[company_name][item_name][year] = []

            company_data[company_name][item_name][year].append(markdown_path)

            if company_name not in item_data[item_name]:
                item_data[item_name].append(company_name)

    eligible: list[tuple[str, str, list[str]]] = []
    for company, items in company_data.items():
        for item, years_dict in items.items():
            years_sorted = sorted(years_dict.keys())
            if len(years_sorted) >= min_periods:
                eligible.append((company, item, years_sorted))

    if not eligible:
        logger.warning("No company/item pairs span >= %d years for %s", min_periods, form_type)
        return 0

    # Build sampling weights: apply section_boost multipliers
    eligible_weights: list[float] = []
    for _company, item, _years in eligible:
        w = 1.0
        if section_boost:
            for suffix, multiplier in section_boost.items():
                if item.endswith(suffix):
                    w = multiplier
                    break
        eligible_weights.append(w)

    boosted_sections = set(section_boost or {})
    boosted_count = sum(
        1 for _, item, _ in eligible if any(item.endswith(s) for s in boosted_sections)
    )
    logger.info(
        "Multi-period sampling for %s: %d eligible (company, item) pairs "
        "(%d boosted by section_boost)",
        form_type,
        len(eligible),
        boosted_count,
    )

    data_path = str(chunk_dir)
    item_prefix = "Item " if form_type == "10-K" else ""
    output_file = output_dir / f"{form_type.lower()}-multi-period-data.jsonl"
    os.makedirs(output_dir, exist_ok=True)

    results: list[dict] = []
    skip_count = 0
    question_count = 0

    while question_count < total_samples and skip_count < max_skip_count:
        company, item, years_sorted = random.choices(eligible, weights=eligible_weights, k=1)[0]

        num_periods = random.randint(min_periods, min(max_periods, len(years_sorted)))

        max_start = len(years_sorted) - num_periods
        start_idx = random.randint(0, max_start)
        chosen_years = years_sorted[start_idx : start_idx + num_periods]

        section_label = item.replace(f"{form_lower}_", item_prefix)

        record: dict = {
            "company_name": company,
            "item_section0": section_label,
            "file_type": form_type,
            "multi_period": True,
            "num_periods": num_periods,
        }

        success = True
        for period_idx, yr in enumerate(chosen_years):
            chunks = company_data[company][item].get(yr, [])
            if not chunks:
                success = False
                break
            md_path = random.choice(chunks)
            abs_path = os.path.join(data_path, md_path)
            try:
                with open(abs_path) as f:
                    content = f.read()
            except FileNotFoundError:
                success = False
                break

            record[f"year{period_idx}"] = yr
            record[f"content{period_idx}"] = content
            record[f"file_path{period_idx}"] = md_path

        if not success:
            skip_count += 1
            continue

        if guidance_prefilter:
            any_forward = any(
                has_forward_looking_content(record.get(f"content{i}", ""))
                for i in range(num_periods)
            )
            if not any_forward:
                skip_count += 1
                continue

        results.append(record)
        question_count += 1

    with open(output_file, "w", encoding="utf-8") as out_f:
        for record in results:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "Multi-period %s: written %d records to %s (guidance_prefilter=%s)",
        form_type,
        len(results),
        output_file,
        guidance_prefilter,
    )
    return len(results)


# =============================================================================
# Part 5: Cross-Form Sampling
# =============================================================================


def generate_cross_form_jsonl_data(
    chunk_dir: Path,
    csv_dir: Path,
    output_dir: Path,
    form_types: list[str],
    total_samples: int = 5000,
    max_skip_count: int = 20000,
    seed: int = 42,
    guidance_prefilter: bool = False,
) -> int:
    """Generate JSONL data combining chunks from different form types for the same company.

    Creates contexts where, e.g., a 10-K item section and an 8-K item section
    for the same company and year are paired together.  This enables question
    categories like "Beat or Miss" and "Complex Retrieval" that inherently
    need multiple document types.
    """
    if len(form_types) < 2:
        logger.warning("Cross-form sampling requires >= 2 form types, got %s", form_types)
        return 0

    random.seed(seed)

    # Build per-form data structures: form -> {company -> {item -> {year -> [paths]}}}
    form_company_data: dict[str, dict[str, dict[str, dict[str, list[str]]]]] = {}
    form_item_data: dict[str, dict[str, list[str]]] = {}

    csv_path_root = str(csv_dir)
    if not os.path.exists(csv_path_root):
        logger.warning("CSV directory not found: %s", csv_path_root)
        return 0

    for form_type in form_types:
        form_lower = form_type.lower().replace("-", "")
        company_data: dict[str, dict[str, dict[str, list[str]]]] = {}
        item_data: dict[str, list[str]] = {}

        for file_name in sorted(os.listdir(csv_path_root)):
            if not (form_lower in file_name and file_name.endswith(".csv")):
                continue

            item_name = file_name.replace(".csv", "")
            if item_name not in item_data:
                item_data[item_name] = []

            file_path = os.path.join(csv_path_root, file_name)
            try:
                if HAS_PANDAS:
                    df = pd.read_csv(file_path)
                    rows = df.iterrows()
                else:
                    with open(file_path) as f:
                        reader = csv.DictReader(f)
                        rows = list(enumerate(list(reader)))
            except Exception:
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

                if company_name not in company_data:
                    company_data[company_name] = {}
                if item_name not in company_data[company_name]:
                    company_data[company_name][item_name] = {}
                if year not in company_data[company_name][item_name]:
                    company_data[company_name][item_name][year] = []

                company_data[company_name][item_name][year].append(markdown_path)

                if company_name not in item_data[item_name]:
                    item_data[item_name].append(company_name)

        form_company_data[form_type] = company_data
        form_item_data[form_type] = item_data

    # Find companies that appear in at least 2 form types
    company_forms: dict[str, set[str]] = defaultdict(set)
    for form_type, company_data in form_company_data.items():
        for company in company_data:
            company_forms[company].add(form_type)

    eligible_companies = [c for c, forms in company_forms.items() if len(forms) >= 2]
    if not eligible_companies:
        logger.warning("No companies found with chunks in >= 2 form types")
        return 0

    logger.info(
        "Cross-form sampling: %d eligible companies across %s",
        len(eligible_companies),
        form_types,
    )

    output_file = output_dir / "cross-form-data.jsonl"
    os.makedirs(output_dir, exist_ok=True)
    data_path = str(chunk_dir)

    results = []
    skip_count = 0
    question_count = 0

    # Sample pairs of (form_type_a, form_type_b) uniformly
    form_pairs = []
    for i, fa in enumerate(form_types):
        for fb in form_types[i + 1 :]:
            form_pairs.append((fa, fb))

    while question_count < total_samples and skip_count < max_skip_count:
        form_a, form_b = random.choice(form_pairs)

        # Pick a company that has chunks in both forms
        candidates = [
            c
            for c in eligible_companies
            if form_a in company_forms[c] and form_b in company_forms[c]
        ]
        if not candidates:
            skip_count += 1
            continue

        company = random.choice(candidates)

        # Pick an item from each form
        items_a = list(form_company_data[form_a].get(company, {}).keys())
        items_b = list(form_company_data[form_b].get(company, {}).keys())
        if not items_a or not items_b:
            skip_count += 1
            continue

        item_a = random.choice(items_a)
        item_b = random.choice(items_b)

        # Find overlapping years
        years_a = set(form_company_data[form_a][company][item_a].keys())
        years_b = set(form_company_data[form_b][company][item_b].keys())
        common_years = sorted(years_a & years_b)
        if not common_years:
            skip_count += 1
            continue

        year = random.choice(common_years)

        paths_a = form_company_data[form_a][company][item_a][year]
        paths_b = form_company_data[form_b][company][item_b][year]
        if not paths_a or not paths_b:
            skip_count += 1
            continue

        md_path_a = random.choice(paths_a)
        md_path_b = random.choice(paths_b)

        form_lower_a = form_a.lower().replace("-", "")
        form_lower_b = form_b.lower().replace("-", "")
        item_prefix_a = "Item " if form_a == "10-K" else ""
        item_prefix_b = "Item " if form_b == "10-K" else ""
        section_a = item_a.replace(f"{form_lower_a}_", item_prefix_a)
        section_b = item_b.replace(f"{form_lower_b}_", item_prefix_b)

        try:
            with open(os.path.join(data_path, md_path_a)) as f:
                content_a = f.read()
            with open(os.path.join(data_path, md_path_b)) as f:
                content_b = f.read()
        except FileNotFoundError:
            skip_count += 1
            continue

        if guidance_prefilter:
            if not (
                has_forward_looking_content(content_a) or has_forward_looking_content(content_b)
            ):
                skip_count += 1
                continue

        results.append(
            {
                "company_name": company,
                "year": year,
                "item_section0": section_a,
                "item_section1": section_b,
                "file_path0": md_path_a,
                "file_path1": md_path_b,
                "content0": content_a,
                "content1": content_b,
                "file_type": f"{form_a}+{form_b}",
                "file_type0": form_a,
                "file_type1": form_b,
            }
        )
        question_count += 1

    with open(output_file, "w", encoding="utf-8") as out_f:
        for record in results:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "Cross-form: written %d records to %s (guidance_prefilter=%s)",
        len(results),
        output_file,
        guidance_prefilter,
    )
    return len(results)


# =============================================================================
# Part 6: Cross-Form Multi-Period Sampling
# =============================================================================


def _build_form_data(
    csv_dir: Path,
    form_type: str,
) -> tuple[dict[str, dict[str, dict[str, list[str]]]], dict[str, list[str]]]:
    """Build company_data and item_data dicts for a single form type from CSVs.

    Returns (company_data, item_data) where:
        company_data[company][item][year] = [markdown_paths]
        item_data[item] = [companies]
    """
    form_lower = form_type.lower().replace("-", "")
    csv_path_root = str(csv_dir)
    company_data: dict[str, dict[str, dict[str, list[str]]]] = {}
    item_data: dict[str, list[str]] = {}

    if not os.path.exists(csv_path_root):
        return company_data, item_data

    for file_name in sorted(os.listdir(csv_path_root)):
        if not (form_lower in file_name and file_name.endswith(".csv")):
            continue
        item_name = file_name.replace(".csv", "")
        if item_name not in item_data:
            item_data[item_name] = []

        file_path = os.path.join(csv_path_root, file_name)
        try:
            if HAS_PANDAS:
                df = pd.read_csv(file_path)
                rows = df.iterrows()
            else:
                with open(file_path) as f:
                    reader = csv.DictReader(f)
                    rows = list(enumerate(list(reader)))
        except Exception:
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

            if company_name not in company_data:
                company_data[company_name] = {}
            if item_name not in company_data[company_name]:
                company_data[company_name][item_name] = {}
            if year not in company_data[company_name][item_name]:
                company_data[company_name][item_name][year] = []
            company_data[company_name][item_name][year].append(markdown_path)
            if company_name not in item_data[item_name]:
                item_data[item_name].append(company_name)

    return company_data, item_data


def generate_cross_form_multi_period_jsonl_data(
    chunk_dir: Path,
    csv_dir: Path,
    output_dir: Path,
    form_types: list[str],
    total_samples: int = 5000,
    guidance_prefilter: bool = False,
    max_skip_count: int = 20000,
    seed: int = 42,
) -> int:
    """Generate JSONL pairing different form types from different years for the same company.

    Produces contexts like "Company A's 10-K from 2023 + Company A's 8-K from 2024",
    enabling questions that compare annual performance with quarterly earnings surprises
    or bridge guidance across different filing types over time.
    """
    if len(form_types) < 2:
        logger.warning("Cross-form-multi-period requires >= 2 form types, got %s", form_types)
        return 0

    random.seed(seed)

    form_company_data: dict[str, dict[str, dict[str, dict[str, list[str]]]]] = {}
    for ft in form_types:
        cd, _ = _build_form_data(csv_dir, ft)
        form_company_data[ft] = cd

    company_forms: dict[str, set[str]] = defaultdict(set)
    for ft, cd in form_company_data.items():
        for company in cd:
            company_forms[company].add(ft)

    eligible_companies = [c for c, forms in company_forms.items() if len(forms) >= 2]
    if not eligible_companies:
        logger.warning("Cross-form-multi-period: no companies in >= 2 form types")
        return 0

    form_pairs = []
    for i, fa in enumerate(form_types):
        for fb in form_types[i + 1 :]:
            form_pairs.append((fa, fb))

    data_path = str(chunk_dir)
    output_file = output_dir / "cross-form-multi-period-data.jsonl"
    os.makedirs(output_dir, exist_ok=True)

    results: list[dict] = []
    skip_count = 0
    question_count = 0

    while question_count < total_samples and skip_count < max_skip_count:
        form_a, form_b = random.choice(form_pairs)
        candidates = [
            c
            for c in eligible_companies
            if form_a in company_forms[c] and form_b in company_forms[c]
        ]
        if not candidates:
            skip_count += 1
            continue

        company = random.choice(candidates)
        items_a = list(form_company_data[form_a].get(company, {}).keys())
        items_b = list(form_company_data[form_b].get(company, {}).keys())
        if not items_a or not items_b:
            skip_count += 1
            continue

        item_a = random.choice(items_a)
        item_b = random.choice(items_b)

        years_a = sorted(form_company_data[form_a][company][item_a].keys())
        years_b = sorted(form_company_data[form_b][company][item_b].keys())
        if not years_a or not years_b:
            skip_count += 1
            continue

        year_a = random.choice(years_a)
        remaining_years_b = [y for y in years_b if y != year_a]
        if not remaining_years_b:
            skip_count += 1
            continue
        year_b = random.choice(remaining_years_b)

        paths_a = form_company_data[form_a][company][item_a][year_a]
        paths_b = form_company_data[form_b][company][item_b][year_b]
        if not paths_a or not paths_b:
            skip_count += 1
            continue

        md_a = random.choice(paths_a)
        md_b = random.choice(paths_b)

        try:
            with open(os.path.join(data_path, md_a)) as f:
                content_a = f.read()
            with open(os.path.join(data_path, md_b)) as f:
                content_b = f.read()
        except FileNotFoundError:
            skip_count += 1
            continue

        if guidance_prefilter:
            if not (
                has_forward_looking_content(content_a) or has_forward_looking_content(content_b)
            ):
                skip_count += 1
                continue

        form_lower_a = form_a.lower().replace("-", "")
        form_lower_b = form_b.lower().replace("-", "")
        prefix_a = "Item " if form_a == "10-K" else ""
        prefix_b = "Item " if form_b == "10-K" else ""
        section_a = item_a.replace(f"{form_lower_a}_", prefix_a)
        section_b = item_b.replace(f"{form_lower_b}_", prefix_b)

        results.append(
            {
                "company_name": company,
                "year0": year_a,
                "year1": year_b,
                "item_section0": section_a,
                "item_section1": section_b,
                "file_path0": md_a,
                "file_path1": md_b,
                "content0": content_a,
                "content1": content_b,
                "file_type": f"{form_a}+{form_b}",
                "file_type0": form_a,
                "file_type1": form_b,
                "cross_form_multi_period": True,
            }
        )
        question_count += 1

    with open(output_file, "w", encoding="utf-8") as out_f:
        for rec in results:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(
        "Cross-form-multi-period: written %d records to %s (guidance_prefilter=%s)",
        len(results),
        output_file,
        guidance_prefilter,
    )
    return len(results)


# =============================================================================
# Part 7: Multi-Company Multi-Year Same Form Sampling
# =============================================================================


def generate_multi_company_multi_year_jsonl_data(
    chunk_dir: Path,
    csv_dir: Path,
    output_dir: Path,
    form_types: list[str],
    total_samples: int = 5000,
    max_skip_count: int = 20000,
    seed: int = 42,
    max_year_gap: int = 3,
    guidance_prefilter: bool = False,
) -> int:
    """Generate JSONL pairing two companies with the same form type but different years.

    Produces contexts like "COST 10-K 2023 + BJ 10-K 2024", enabling temporal
    cross-company comparison questions (Market Analysis, Trends).
    """
    random.seed(seed)

    form_data: dict[str, tuple[dict, dict]] = {}
    for ft in form_types:
        form_data[ft] = _build_form_data(csv_dir, ft)

    data_path = str(chunk_dir)
    output_file = output_dir / "multi-company-multi-year-data.jsonl"
    os.makedirs(output_dir, exist_ok=True)

    results: list[dict] = []
    skip_count = 0
    question_count = 0

    while question_count < total_samples and skip_count < max_skip_count:
        form_type = random.choice(form_types)
        company_data, item_data = form_data[form_type]

        all_items = list(item_data.keys())
        if not all_items:
            skip_count += 1
            continue
        item = random.choice(all_items)

        companies_with_item = item_data.get(item, [])
        if len(companies_with_item) < 2:
            skip_count += 1
            continue

        comp_a, comp_b = random.sample(companies_with_item, 2)

        years_a = sorted(company_data.get(comp_a, {}).get(item, {}).keys())
        years_b = sorted(company_data.get(comp_b, {}).get(item, {}).keys())
        if not years_a or not years_b:
            skip_count += 1
            continue

        year_a = random.choice(years_a)
        diff_years_b = [
            y for y in years_b if y != year_a and abs(int(y) - int(year_a)) <= max_year_gap
        ]
        if not diff_years_b:
            skip_count += 1
            continue
        year_b = random.choice(diff_years_b)

        paths_a = company_data[comp_a][item].get(year_a, [])
        paths_b = company_data[comp_b][item].get(year_b, [])
        if not paths_a or not paths_b:
            skip_count += 1
            continue

        md_a = random.choice(paths_a)
        md_b = random.choice(paths_b)

        try:
            with open(os.path.join(data_path, md_a)) as f:
                content_a = f.read()
            with open(os.path.join(data_path, md_b)) as f:
                content_b = f.read()
        except FileNotFoundError:
            skip_count += 1
            continue

        if guidance_prefilter:
            if not (
                has_forward_looking_content(content_a) or has_forward_looking_content(content_b)
            ):
                skip_count += 1
                continue

        form_lower = form_type.lower().replace("-", "")
        prefix = "Item " if form_type == "10-K" else ""
        section = item.replace(f"{form_lower}_", prefix)

        results.append(
            {
                "company_name0": comp_a,
                "company_name1": comp_b,
                "year0": year_a,
                "year1": year_b,
                "item_section0": section,
                "item_section1": section,
                "file_path0": md_a,
                "file_path1": md_b,
                "content0": content_a,
                "content1": content_b,
                "file_type": form_type,
                "multi_company_multi_year": True,
            }
        )
        question_count += 1

    with open(output_file, "w", encoding="utf-8") as out_f:
        for rec in results:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(
        "Multi-company-multi-year: written %d records to %s (guidance_prefilter=%s)",
        len(results),
        output_file,
        guidance_prefilter,
    )
    return len(results)


# =============================================================================
# Part 8: Multi-Company Cross-Form Same Year Sampling
# =============================================================================


def generate_multi_company_cross_form_jsonl_data(
    chunk_dir: Path,
    csv_dir: Path,
    output_dir: Path,
    form_types: list[str],
    total_samples: int = 5000,
    max_skip_count: int = 20000,
    seed: int = 42,
    guidance_prefilter: bool = False,
) -> int:
    """Generate JSONL pairing two companies with different form types in the same year.

    Produces contexts like "Company A's 10-K + Company B's 8-K, both 2024",
    enabling cross-company cross-document-type comparison questions.
    """
    if len(form_types) < 2:
        logger.warning("Multi-company-cross-form requires >= 2 form types")
        return 0

    random.seed(seed)

    form_company_data: dict[str, dict[str, dict[str, dict[str, list[str]]]]] = {}
    for ft in form_types:
        cd, _ = _build_form_data(csv_dir, ft)
        form_company_data[ft] = cd

    form_pairs = []
    for i, fa in enumerate(form_types):
        for fb in form_types[i + 1 :]:
            form_pairs.append((fa, fb))

    data_path = str(chunk_dir)
    output_file = output_dir / "multi-company-cross-form-data.jsonl"
    os.makedirs(output_dir, exist_ok=True)

    results: list[dict] = []
    skip_count = 0
    question_count = 0

    while question_count < total_samples and skip_count < max_skip_count:
        form_a, form_b = random.choice(form_pairs)

        companies_a = list(form_company_data[form_a].keys())
        companies_b = list(form_company_data[form_b].keys())
        if not companies_a or not companies_b:
            skip_count += 1
            continue

        comp_a = random.choice(companies_a)
        candidates_b = [c for c in companies_b if c != comp_a]
        if not candidates_b:
            skip_count += 1
            continue
        comp_b = random.choice(candidates_b)

        items_a = list(form_company_data[form_a][comp_a].keys())
        items_b = list(form_company_data[form_b][comp_b].keys())
        if not items_a or not items_b:
            skip_count += 1
            continue

        item_a = random.choice(items_a)
        item_b = random.choice(items_b)

        years_a = set(form_company_data[form_a][comp_a][item_a].keys())
        years_b = set(form_company_data[form_b][comp_b][item_b].keys())
        common_years = sorted(years_a & years_b)
        if not common_years:
            skip_count += 1
            continue

        year = random.choice(common_years)

        paths_a = form_company_data[form_a][comp_a][item_a][year]
        paths_b = form_company_data[form_b][comp_b][item_b][year]
        if not paths_a or not paths_b:
            skip_count += 1
            continue

        md_a = random.choice(paths_a)
        md_b = random.choice(paths_b)

        try:
            with open(os.path.join(data_path, md_a)) as f:
                content_a = f.read()
            with open(os.path.join(data_path, md_b)) as f:
                content_b = f.read()
        except FileNotFoundError:
            skip_count += 1
            continue

        if guidance_prefilter:
            if not (
                has_forward_looking_content(content_a) or has_forward_looking_content(content_b)
            ):
                skip_count += 1
                continue

        fl_a = form_a.lower().replace("-", "")
        fl_b = form_b.lower().replace("-", "")
        pref_a = "Item " if form_a == "10-K" else ""
        pref_b = "Item " if form_b == "10-K" else ""
        section_a = item_a.replace(f"{fl_a}_", pref_a)
        section_b = item_b.replace(f"{fl_b}_", pref_b)

        results.append(
            {
                "company_name0": comp_a,
                "company_name1": comp_b,
                "year": year,
                "item_section0": section_a,
                "item_section1": section_b,
                "file_path0": md_a,
                "file_path1": md_b,
                "content0": content_a,
                "content1": content_b,
                "file_type": f"{form_a}+{form_b}",
                "file_type0": form_a,
                "file_type1": form_b,
                "multi_company_cross_form": True,
            }
        )
        question_count += 1

    with open(output_file, "w", encoding="utf-8") as out_f:
        for rec in results:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(
        "Multi-company-cross-form: written %d records to %s (guidance_prefilter=%s)",
        len(results),
        output_file,
        guidance_prefilter,
    )
    return len(results)


# =============================================================================
# Part 9: Same-Year Multi-Filing Sampling
# =============================================================================


def generate_same_year_multi_filing_jsonl_data(
    chunk_dir: Path,
    csv_dir: Path,
    output_dir: Path,
    form_types: list[str],
    total_samples: int = 5000,
    max_skip_count: int = 20000,
    seed: int = 42,
    min_filings: int = 3,
    max_filings: int = 5,
    guidance_prefilter: bool = False,
) -> int:
    """Generate JSONL combining multiple filings of the same form within a single year.

    Produces contexts like "4 quarterly 8-Ks from Company A in 2024",
    which is the primary data pattern for VALS Beat_or_Miss questions.
    Different filings are identified by distinct accession-number directories
    in the chunk paths.
    """
    random.seed(seed)
    data_path = str(chunk_dir)
    output_file = output_dir / "same-year-multi-filing-data.jsonl"
    os.makedirs(output_dir, exist_ok=True)

    # Build per-form data with accession-level granularity:
    # company -> item -> year -> {accession: [paths]}
    eligible: list[tuple[str, str, str, str, list[str]]] = []
    # (form_type, company, item, year, list_of_accession_representative_paths)

    for form_type in form_types:
        cd, _ = _build_form_data(csv_dir, form_type)
        for company, items in cd.items():
            for item, years_dict in items.items():
                for year, paths in years_dict.items():
                    accessions: dict[str, str] = {}
                    for p in paths:
                        parts = p.split("/")
                        acc = None
                        if form_type in parts:
                            fidx = parts.index(form_type)
                            if fidx + 2 < len(parts):
                                acc = parts[fidx + 2]
                        if acc is None:
                            acc = p
                        if acc not in accessions:
                            accessions[acc] = p

                    if len(accessions) >= min_filings:
                        eligible.append((form_type, company, item, year, list(accessions.values())))

    if not eligible:
        logger.warning(
            "Same-year-multi-filing: no (company, item, year) with >= %d accessions",
            min_filings,
        )
        return 0

    logger.info(
        "Same-year-multi-filing: %d eligible (form, company, item, year) tuples",
        len(eligible),
    )

    results: list[dict] = []
    skip_count = 0
    question_count = 0

    while question_count < total_samples and skip_count < max_skip_count:
        form_type, company, item, year, acc_paths = random.choice(eligible)

        n_filings = random.randint(min_filings, min(max_filings, len(acc_paths)))
        chosen_paths = random.sample(acc_paths, n_filings)

        form_lower = form_type.lower().replace("-", "")
        prefix = "Item " if form_type == "10-K" else ""
        section = item.replace(f"{form_lower}_", prefix)

        record: dict = {
            "company_name": company,
            "year": year,
            "item_section0": section,
            "file_type": form_type,
            "same_year_multi_filing": True,
            "num_filings": n_filings,
        }

        success = True
        for i, md_path in enumerate(chosen_paths):
            try:
                with open(os.path.join(data_path, md_path)) as f:
                    content = f.read()
            except FileNotFoundError:
                success = False
                break
            record[f"file_path{i}"] = md_path
            record[f"content{i}"] = content

        if not success:
            skip_count += 1
            continue

        if guidance_prefilter:
            any_forward = any(
                has_forward_looking_content(record.get(f"content{i}", "")) for i in range(n_filings)
            )
            if not any_forward:
                skip_count += 1
                continue

        results.append(record)
        question_count += 1

    with open(output_file, "w", encoding="utf-8") as out_f:
        for rec in results:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(
        "Same-year-multi-filing: written %d records to %s (guidance_prefilter=%s)",
        len(results),
        output_file,
        guidance_prefilter,
    )
    return len(results)


# =============================================================================
# Part 10: Multi-Form Sampling (3+ form types, 1 company, same year)
# =============================================================================


def generate_multi_form_jsonl_data(
    chunk_dir: Path,
    csv_dir: Path,
    output_dir: Path,
    form_types: list[str],
    total_samples: int = 5000,
    max_skip_count: int = 20000,
    seed: int = 42,
    min_forms: int = 3,
    max_forms: int = 5,
    guidance_prefilter: bool = False,
) -> int:
    """Generate JSONL combining 3+ different form types for the same company and year.

    Each record randomly picks between ``min_forms`` and ``max_forms`` different
    form types (e.g. 10-K + 10-Q + 8-K + DEF 14A from 2024), enabling
    comprehensive analysis questions that synthesise annual, quarterly,
    event-driven, and proxy filings.
    """
    if len(form_types) < min_forms:
        logger.warning("Multi-form requires >= %d form types, got %d", min_forms, len(form_types))
        return 0

    random.seed(seed)

    form_company_data: dict[str, dict[str, dict[str, dict[str, list[str]]]]] = {}
    for ft in form_types:
        cd, _ = _build_form_data(csv_dir, ft)
        form_company_data[ft] = cd

    company_forms: dict[str, set[str]] = defaultdict(set)
    for ft, cd in form_company_data.items():
        for company in cd:
            company_forms[company].add(ft)

    eligible_companies = [c for c, forms in company_forms.items() if len(forms) >= min_forms]
    if not eligible_companies:
        logger.warning("Multi-form: no companies in >= %d form types", min_forms)
        return 0

    from itertools import combinations

    effective_max = min(max_forms, len(form_types))
    form_combos: list[tuple[str, ...]] = []
    for n in range(min_forms, effective_max + 1):
        form_combos.extend(combinations(form_types, n))

    data_path = str(chunk_dir)
    output_file = output_dir / "multi-form-data.jsonl"
    os.makedirs(output_dir, exist_ok=True)

    results: list[dict] = []
    skip_count = 0
    question_count = 0

    while question_count < total_samples and skip_count < max_skip_count:
        combo = random.choice(form_combos)
        n_forms = len(combo)
        candidates = [c for c in eligible_companies if all(ft in company_forms[c] for ft in combo)]
        if not candidates:
            skip_count += 1
            continue

        company = random.choice(candidates)

        items_per_form = {}
        for ft in combo:
            items = list(form_company_data[ft].get(company, {}).keys())
            if not items:
                break
            items_per_form[ft] = random.choice(items)

        if len(items_per_form) < n_forms:
            skip_count += 1
            continue

        years_sets = []
        for ft in combo:
            item = items_per_form[ft]
            yrs = set(form_company_data[ft][company][item].keys())
            years_sets.append(yrs)

        common_years = sorted(set.intersection(*years_sets))
        if not common_years:
            skip_count += 1
            continue

        year = random.choice(common_years)

        record: dict = {
            "company_name": company,
            "year": year,
            "file_type": "+".join(combo),
            "multi_form": True,
            "num_forms": n_forms,
        }

        success = True
        for i, ft in enumerate(combo):
            item = items_per_form[ft]
            paths = form_company_data[ft][company][item].get(year, [])
            if not paths:
                success = False
                break
            md_path = random.choice(paths)
            try:
                with open(os.path.join(data_path, md_path)) as f:
                    content = f.read()
            except FileNotFoundError:
                success = False
                break

            fl = ft.lower().replace("-", "")
            prefix = "Item " if ft == "10-K" else ""
            section = item.replace(f"{fl}_", prefix)

            record[f"item_section{i}"] = section
            record[f"file_path{i}"] = md_path
            record[f"content{i}"] = content
            record[f"file_type{i}"] = ft

        if not success:
            skip_count += 1
            continue

        if guidance_prefilter:
            any_forward = any(
                has_forward_looking_content(record.get(f"content{i}", "")) for i in range(n_forms)
            )
            if not any_forward:
                skip_count += 1
                continue

        results.append(record)
        question_count += 1

    with open(output_file, "w", encoding="utf-8") as out_f:
        for rec in results:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(
        "Multi-form: written %d records to %s (guidance_prefilter=%s)",
        len(results),
        output_file,
        guidance_prefilter,
    )
    return len(results)


# =============================================================================
# Main Entry Point
# =============================================================================


def run_relational_preprocess(
    input_dir: Path,
    output_dir: Path,
    distribution_dir: Path,
    max_tokens: int = 2000,
    overlap_tokens: int = 100,
    total_samples: int = 150000,
    max_skip_count: int = 20000,
    seed: int = 42,
    form_weights: dict[str, float] | None = None,
    cross_form_weight: float = 0.0,
    multi_period_weight: float = 0.0,
    multi_period_section_boost: dict[str, dict[str, float]] | None = None,
    multi_period_max_periods: dict[str, int] | None = None,
    skip_chunking: bool = False,
    cross_form_multi_period_weight: float = 0.0,
    multi_company_multi_year_weight: float = 0.0,
    multi_company_cross_form_weight: float = 0.0,
    same_year_multi_filing_weight: float = 0.0,
    multi_form_weight: float = 0.0,
    multi_company_multi_year_max_gap: int = 3,
    same_year_multi_filing_min: int = 3,
    same_year_multi_filing_max: int = 5,
    multi_form_min_forms: int = 3,
    multi_form_max_forms: int = 5,
    guidance_prefilter: bool = False,
):
    """Run the full preprocessing pipeline.

    Args:
        form_weights: Optional dict mapping form types to weights, e.g.
            ``{"10-K": 1.0, "10-Q": 1.0, "8-K": 1.0}``.  When provided the
            filings list is derived from the keys and ``total_samples`` is
            allocated proportionally per form type.  When *None* the legacy
            behaviour (hardcoded 10-K + 10-Q, each getting ``total_samples``)
            is preserved for backward compatibility.
        cross_form_weight: Fraction of ``total_samples`` to allocate to
            cross-form contexts (0.0 to 1.0). Requires ``form_weights`` with
            >= 2 form types. Defaults to 0.0 (disabled).
        multi_period_weight: Fraction of ``total_samples`` to allocate to
            multi-period contexts (0.0 to 1.0).  Pairs the same item section
            across 2-4 consecutive years for the same company. Defaults to
            0.0 (disabled).
        multi_period_section_boost: Optional per-form-type section boost map.
            E.g. ``{"8-K": {"2-2": 5.0}}`` means multi-period 8-K sampling
            gives 5x weight to Item 2.02 pairs (for Beat_or_Miss questions).
        multi_period_max_periods: Optional per-form-type max periods override.
            E.g. ``{"8-K": 6, "8-K_A": 6}`` allows up to 6 consecutive
            periods for 8-K filings (vs default 4). Useful for Beat_or_Miss
            questions that need 4-5 sequential quarterly earnings releases.
        cross_form_multi_period_weight: Fraction for cross-form + multi-period
            contexts (1 company, 2 form types, different years). 0.0 = disabled.
        multi_company_multi_year_weight: Fraction for multi-company, multi-year,
            same form contexts (2 companies, same form, different years).
        multi_company_cross_form_weight: Fraction for multi-company, cross-form,
            same year contexts (2 companies, different form types, same year).
        same_year_multi_filing_weight: Fraction for same-year multi-filing
            contexts (1 company, same form, same year, 3+ distinct filings).
            Critical for Beat_or_Miss with quarterly 8-K earnings releases.
        multi_form_weight: Fraction for multi-form contexts (1 company,
            3-5 different form types, same year).
        multi_company_multi_year_max_gap: Max year gap between companies (default 3).
        same_year_multi_filing_min: Min filings per year to be eligible (default 3).
        same_year_multi_filing_max: Max filings to sample per record (default 5).
        multi_form_min_forms: Min form types to combine (default 3).
        multi_form_max_forms: Max form types to combine (default 5).
    """
    logger.info("=" * 60)
    logger.info("SEC Data Preprocessing Pipeline")
    logger.info("=" * 60)

    if form_weights is not None:
        sample_filings = list(form_weights.keys())
        weight_sum = sum(form_weights.values()) or 1.0
        logger.info("Using configurable form_weights for sampling: %s", form_weights)
    else:
        sample_filings = ["10-K", "10-Q"]

    # Discover all form types present in the input directory
    all_filings = sorted(
        {
            d.name
            for ticker_dir in input_dir.iterdir()
            if ticker_dir.is_dir()
            for d in ticker_dir.iterdir()
            if d.is_dir()
        }
    )
    if not all_filings:
        all_filings = list(_KNOWN_FORM_TYPES)
    logger.info("Discovered form types in input: %s", all_filings)

    # Step 1: Chunk ALL form types (not just the ones in form_weights)
    chunk_output_dir = output_dir / "chunks"
    if skip_chunking and chunk_output_dir.exists():
        logger.info("Skipping chunking — reusing existing chunks in %s", chunk_output_dir)
    else:
        run_chunking(
            input_dir=input_dir,
            output_dir=chunk_output_dir,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            filings=all_filings,
        )

    # Step 2: Generate CSV file lists for ALL form types
    csv_output_dir = output_dir / "csv_lists"
    generate_chunk_lists(
        chunk_dir=chunk_output_dir,
        csv_output_dir=csv_output_dir,
        filings=all_filings,
    )

    # Step 3: Generate JSONL data
    jsonl_output_dir = output_dir / "jsonl"
    totals: dict[str, int] = {}

    # Split total_samples across all sampling modes
    cross_form_samples = 0
    multi_period_samples = 0
    cfmp_samples = 0  # cross-form multi-period
    mcmy_samples = 0  # multi-company multi-year
    mccf_samples = 0  # multi-company cross-form
    symf_samples = 0  # same-year multi-filing
    tf_samples = 0  # triple-form
    single_form_total = total_samples

    if multi_period_weight > 0:
        multi_period_samples = max(1, int(total_samples * multi_period_weight))
        single_form_total -= multi_period_samples
    if cross_form_weight > 0 and form_weights is not None and len(sample_filings) >= 2:
        cross_form_samples = max(1, int(total_samples * cross_form_weight))
        single_form_total -= cross_form_samples
    if cross_form_multi_period_weight > 0 and form_weights is not None and len(sample_filings) >= 2:
        cfmp_samples = max(1, int(total_samples * cross_form_multi_period_weight))
        single_form_total -= cfmp_samples
    if multi_company_multi_year_weight > 0:
        mcmy_samples = max(1, int(total_samples * multi_company_multi_year_weight))
        single_form_total -= mcmy_samples
    if (
        multi_company_cross_form_weight > 0
        and form_weights is not None
        and len(sample_filings) >= 2
    ):
        mccf_samples = max(1, int(total_samples * multi_company_cross_form_weight))
        single_form_total -= mccf_samples
    if same_year_multi_filing_weight > 0:
        symf_samples = max(1, int(total_samples * same_year_multi_filing_weight))
        single_form_total -= symf_samples
    if multi_form_weight > 0 and form_weights is not None and len(sample_filings) >= 3:
        tf_samples = max(1, int(total_samples * multi_form_weight))
        single_form_total -= tf_samples
    single_form_total = max(0, single_form_total)

    if guidance_prefilter:
        logger.info(
            "Guidance pre-filter ENABLED — only contexts with forward-looking language will be kept"
        )

    if form_weights is not None:
        for form_type, weight in form_weights.items():
            form_samples = max(1, int(single_form_total * weight / weight_sum))
            totals[form_type] = generate_jsonl_data(
                chunk_dir=chunk_output_dir,
                csv_dir=csv_output_dir,
                distribution_dir=distribution_dir,
                output_dir=jsonl_output_dir,
                form_type=form_type,
                total_samples=form_samples,
                max_skip_count=max_skip_count,
                seed=seed,
                guidance_prefilter=guidance_prefilter,
            )
    else:
        totals["10-K"] = generate_jsonl_data(
            chunk_dir=chunk_output_dir,
            csv_dir=csv_output_dir,
            distribution_dir=distribution_dir,
            output_dir=jsonl_output_dir,
            form_type="10-K",
            total_samples=total_samples,
            max_skip_count=max_skip_count,
            seed=seed,
            guidance_prefilter=guidance_prefilter,
        )
        totals["10-Q"] = generate_jsonl_data(
            chunk_dir=chunk_output_dir,
            csv_dir=csv_output_dir,
            distribution_dir=distribution_dir,
            output_dir=jsonl_output_dir,
            form_type="10-Q",
            total_samples=total_samples,
            max_skip_count=max_skip_count,
            seed=seed,
            guidance_prefilter=guidance_prefilter,
        )

    # Step 4 (optional): Cross-form sampling
    if cross_form_samples > 0:
        totals["cross-form"] = generate_cross_form_jsonl_data(
            chunk_dir=chunk_output_dir,
            csv_dir=csv_output_dir,
            output_dir=jsonl_output_dir,
            form_types=sample_filings,
            total_samples=cross_form_samples,
            max_skip_count=max_skip_count,
            seed=seed,
            guidance_prefilter=guidance_prefilter,
        )

    # Step 5 (optional): Multi-period sampling
    if multi_period_samples > 0:
        mp_form_types = sample_filings if form_weights is not None else ["10-K", "10-Q"]
        mp_per_form = max(1, multi_period_samples // len(mp_form_types))
        for ft in mp_form_types:
            ft_boost = (multi_period_section_boost or {}).get(ft)
            ft_max_periods = (multi_period_max_periods or {}).get(ft, 4)
            totals[f"{ft}_multi_period"] = generate_multi_period_jsonl_data(
                chunk_dir=chunk_output_dir,
                csv_dir=csv_output_dir,
                distribution_dir=distribution_dir,
                output_dir=jsonl_output_dir,
                form_type=ft,
                total_samples=mp_per_form,
                max_skip_count=max_skip_count,
                seed=seed,
                section_boost=ft_boost,
                max_periods=ft_max_periods,
                guidance_prefilter=guidance_prefilter,
            )

    # Step 6 (optional): Cross-form multi-period sampling
    if cfmp_samples > 0:
        totals["cross_form_multi_period"] = generate_cross_form_multi_period_jsonl_data(
            chunk_dir=chunk_output_dir,
            csv_dir=csv_output_dir,
            output_dir=jsonl_output_dir,
            form_types=sample_filings,
            total_samples=cfmp_samples,
            max_skip_count=max_skip_count,
            seed=seed,
            guidance_prefilter=guidance_prefilter,
        )

    # Step 7 (optional): Multi-company multi-year same form
    if mcmy_samples > 0:
        totals["multi_company_multi_year"] = generate_multi_company_multi_year_jsonl_data(
            chunk_dir=chunk_output_dir,
            csv_dir=csv_output_dir,
            output_dir=jsonl_output_dir,
            form_types=sample_filings if form_weights is not None else ["10-K", "10-Q"],
            total_samples=mcmy_samples,
            max_skip_count=max_skip_count,
            seed=seed,
            max_year_gap=multi_company_multi_year_max_gap,
            guidance_prefilter=guidance_prefilter,
        )

    # Step 8 (optional): Multi-company cross-form same year
    if mccf_samples > 0:
        totals["multi_company_cross_form"] = generate_multi_company_cross_form_jsonl_data(
            chunk_dir=chunk_output_dir,
            csv_dir=csv_output_dir,
            output_dir=jsonl_output_dir,
            form_types=sample_filings,
            total_samples=mccf_samples,
            max_skip_count=max_skip_count,
            seed=seed,
            guidance_prefilter=guidance_prefilter,
        )

    # Step 9 (optional): Same-year multi-filing
    if symf_samples > 0:
        totals["same_year_multi_filing"] = generate_same_year_multi_filing_jsonl_data(
            chunk_dir=chunk_output_dir,
            csv_dir=csv_output_dir,
            output_dir=jsonl_output_dir,
            form_types=sample_filings if form_weights is not None else ["10-K", "10-Q"],
            total_samples=symf_samples,
            max_skip_count=max_skip_count,
            seed=seed,
            min_filings=same_year_multi_filing_min,
            max_filings=same_year_multi_filing_max,
            guidance_prefilter=guidance_prefilter,
        )

    # Step 10 (optional): Multi-form (3-5 form types, same company, same year)
    if tf_samples > 0:
        totals["multi_form"] = generate_multi_form_jsonl_data(
            chunk_dir=chunk_output_dir,
            csv_dir=csv_output_dir,
            output_dir=jsonl_output_dir,
            form_types=sample_filings,
            total_samples=tf_samples,
            max_skip_count=max_skip_count,
            seed=seed,
            min_forms=multi_form_min_forms,
            max_forms=multi_form_max_forms,
            guidance_prefilter=guidance_prefilter,
        )

    logger.info("=" * 60)
    logger.info("Preprocessing Complete!")
    for ft, count in totals.items():
        logger.info("  %s samples: %d", ft, count)
    logger.info("  Output directory: %s", output_dir)
    logger.info("=" * 60)

    return jsonl_output_dir


# ---------------------------------------------------------------------------
# Ported helpers (constants + small utilities used by the relational generators)
# ---------------------------------------------------------------------------
_KNOWN_FORM_TYPES = {
    "10-K",
    "10-K_A",
    "10-Q",
    "10-Q_A",
    "20-F",
    "4",
    "4_A",
    "6-K",
    "8-K",
    "8-K_A",
    "DEF 14A",
    "FWP",
    "S-1",
    "S-1_A",
}


def get_relative_path(full_path: str, start_path: str) -> str:
    """Get relative path from start_path."""
    return os.path.relpath(full_path, start_path)


def _auto_generate_distribution(
    csv_dir: Path,
    distribution_dir: Path,
    form_type: str,
) -> bool:
    """Auto-generate uniform 1-company and 2-company distribution CSVs for a form type.

    Scans the CSV directory for item files matching this form type and creates
    uniform distribution files (count=1 per item section). Returns True if files
    were created successfully.
    """
    form_lower = form_type.lower().replace("-", "")
    prefix = f"{form_lower}_"

    item_sections = []
    if csv_dir.exists():
        for f in sorted(csv_dir.iterdir()):
            if f.name.startswith(prefix) and f.name.endswith(".csv"):
                item_sections.append(f.stem)

    if not item_sections:
        logger.warning(
            "No CSV files found for %s in %s, cannot auto-generate distributions",
            form_type,
            csv_dir,
        )
        return False

    os.makedirs(distribution_dir, exist_ok=True)

    dist_1c_path = distribution_dir / f"{form_lower}_1company_distribution.csv"
    with open(dist_1c_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_section", "count"])
        for sec in item_sections:
            writer.writerow([sec, 1])
    logger.info(
        "Auto-generated 1-company distribution for %s: %s (%d items)",
        form_type,
        dist_1c_path,
        len(item_sections),
    )

    dist_2c_path = distribution_dir / f"{form_lower}_2company_distribution.csv"
    with open(dist_2c_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_section", "count"])
        for sec in item_sections:
            writer.writerow([sec, 1])
    logger.info(
        "Auto-generated 2-company distribution for %s: %s (%d items)",
        form_type,
        dist_2c_path,
        len(item_sections),
    )

    return True


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
    parser.add_argument(
        "--form_weights",
        type=str,
        default=None,
        help='JSON string mapping form types to weights, e.g. \'{"10-K": 1.0, "10-Q": 1.0, "8-K": 1.0}\'',
    )
    parser.add_argument(
        "--cross_form_weight",
        type=float,
        default=0.0,
        help="Fraction of total_samples for cross-form contexts (0.0-1.0, default: 0.0 = disabled)",
    )
    parser.add_argument(
        "--multi_period_weight",
        type=float,
        default=0.0,
        help="Fraction of total_samples for multi-period contexts (0.0-1.0, default: 0.0 = disabled)",
    )
    parser.add_argument(
        "--multi_period_section_boost",
        type=str,
        default=None,
        help='JSON string mapping form types to section boost dicts, e.g. \'{"8-K": {"2-2": 5.0}}\'',
    )
    parser.add_argument(
        "--multi_period_max_periods",
        type=str,
        default=None,
        help='JSON string mapping form types to max periods, e.g. \'{"8-K": 6, "8-K_A": 6}\'',
    )
    parser.add_argument(
        "--skip_chunking",
        action="store_true",
        default=False,
        help="Skip the chunking step and reuse existing chunks in <output_dir>/chunks",
    )
    parser.add_argument(
        "--cross_form_multi_period_weight",
        type=float,
        default=0.0,
        help="Fraction for cross-form multi-period (1 company, 2 forms, different years)",
    )
    parser.add_argument(
        "--multi_company_multi_year_weight",
        type=float,
        default=0.0,
        help="Fraction for multi-company multi-year same form",
    )
    parser.add_argument(
        "--multi_company_cross_form_weight",
        type=float,
        default=0.0,
        help="Fraction for multi-company cross-form same year",
    )
    parser.add_argument(
        "--same_year_multi_filing_weight",
        type=float,
        default=0.0,
        help="Fraction for same-year multi-filing (e.g. 4 quarterly 8-Ks)",
    )
    parser.add_argument(
        "--multi_form_weight",
        type=float,
        default=0.0,
        help="Fraction for multi-form (3-5 form types, 1 company, same year)",
    )
    parser.add_argument(
        "--multi_company_multi_year_max_gap",
        type=int,
        default=3,
        help="Max year gap for multi-company multi-year sampling (default: 3)",
    )
    parser.add_argument(
        "--same_year_multi_filing_min",
        type=int,
        default=3,
        help="Min distinct filings in same year to be eligible (default: 3)",
    )
    parser.add_argument(
        "--same_year_multi_filing_max",
        type=int,
        default=5,
        help="Max filings to sample per record (default: 5)",
    )
    parser.add_argument(
        "--multi_form_min_forms",
        type=int,
        default=3,
        help="Min form types to combine in multi-form sampling (default: 3)",
    )
    parser.add_argument(
        "--multi_form_max_forms",
        type=int,
        default=5,
        help="Max form types to combine in multi-form sampling (default: 5)",
    )
    parser.add_argument(
        "--guidance_prefilter",
        action="store_true",
        default=False,
        help="Only keep sampled contexts where at least one chunk contains forward-looking language (guidance, outlook, forecast, target, etc.)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fw = json.loads(args.form_weights) if args.form_weights else None
    mpsb = json.loads(args.multi_period_section_boost) if args.multi_period_section_boost else None
    mpmp = json.loads(args.multi_period_max_periods) if args.multi_period_max_periods else None
    run_relational_preprocess(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        distribution_dir=args.distribution_dir,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        total_samples=args.total_samples,
        max_skip_count=args.max_skip_count,
        seed=args.seed,
        form_weights=fw,
        cross_form_weight=args.cross_form_weight,
        multi_period_weight=args.multi_period_weight,
        multi_period_section_boost=mpsb,
        multi_period_max_periods=mpmp,
        skip_chunking=args.skip_chunking,
        cross_form_multi_period_weight=args.cross_form_multi_period_weight,
        multi_company_multi_year_weight=args.multi_company_multi_year_weight,
        multi_company_cross_form_weight=args.multi_company_cross_form_weight,
        same_year_multi_filing_weight=args.same_year_multi_filing_weight,
        multi_form_weight=args.multi_form_weight,
        multi_company_multi_year_max_gap=args.multi_company_multi_year_max_gap,
        same_year_multi_filing_min=args.same_year_multi_filing_min,
        same_year_multi_filing_max=args.same_year_multi_filing_max,
        multi_form_min_forms=args.multi_form_min_forms,
        multi_form_max_forms=args.multi_form_max_forms,
        guidance_prefilter=args.guidance_prefilter,
    )
