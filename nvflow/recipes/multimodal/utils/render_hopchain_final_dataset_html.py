#!/usr/bin/env python
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
"""Render generated HopChain synthetic data into paginated HTML review files."""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import logging
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from nvflow.recipes.multimodal.utils.hopchain_sdg_models import (
    GeneratedHopChainQuery,
    InstanceCombinationRecord,
    ReconciledHopChainQuery,
)
from nvflow.recipes.multimodal.utils.image_utils import encode_image_as_data_uri

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BOX_COLORS = (
    "#e11d48",
    "#2563eb",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#0891b2",
)


class HtmlPageSummary(BaseModel):
    """Metadata for one rendered HTML page."""

    page_number: int = Field(ge=1)
    file_name: str
    row_count: int = Field(ge=0)
    start_row: int = Field(ge=1)
    end_row: int = Field(ge=0)


class HopchainHtmlVisualizationSummary(BaseModel):
    """Summary metadata for a visualization render run."""

    queries_input: str
    combinations_input: str
    output_dir: str
    index_file: str
    total_queries: int = Field(ge=0)
    rendered_queries: int = Field(ge=0)
    sample_seed: int | None = None
    rows_per_file: int = Field(ge=1)
    image_max_dimension: int | None = None
    page_files: list[HtmlPageSummary] = Field(default_factory=list)


class RenderableHopchainRecord(BaseModel):
    """Joined query-generation row with its sampled instance combination."""

    query: GeneratedHopChainQuery | ReconciledHopChainQuery
    combination: InstanceCombinationRecord


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Render generated HopChain data into review HTML files"
    )
    parser.add_argument(
        "--queries-input", required=True, help="Path to step-4 generated query JSONL"
    )
    parser.add_argument(
        "--combinations-input", required=True, help="Path to step-3 instance combinations JSONL"
    )
    parser.add_argument("--output-dir", required=True, help="Directory for HTML outputs")
    parser.add_argument("--summary", required=True, help="Summary JSON path")
    parser.add_argument(
        "--rows-per-file", type=int, default=100, help="Maximum rendered rows per HTML file"
    )
    parser.add_argument(
        "--image-max-dimension",
        type=int,
        default=1024,
        help="Max image dimension for embedded previews; use 0 to disable resizing",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="Optional random sample size applied before rendering",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used when sample-count is provided",
    )
    parser.add_argument("--title", default="HopChain Synthetic Data Review", help="HTML title")
    args = parser.parse_args()
    if args.sample_count is not None and args.sample_count <= 0:
        parser.error("--sample-count must be a positive integer")
    return args


def load_queries(input_path: Path) -> list[GeneratedHopChainQuery | ReconciledHopChainQuery]:
    """Load generated HopChain query records from JSONL."""
    queries: list[GeneratedHopChainQuery | ReconciledHopChainQuery] = []
    with input_path.open("r") as input_file:
        for line_num, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if "image_fullpath" not in payload:
                    image_directory = payload.get("image_directory")
                    image_file_name = payload.get("image_file_name")
                    if image_directory and image_file_name:
                        payload["image_fullpath"] = str(Path(image_directory) / image_file_name)
                if "llm_judge_reconciliation_status" in payload:
                    queries.append(ReconciledHopChainQuery.model_validate(payload))
                else:
                    queries.append(GeneratedHopChainQuery.model_validate(payload))
            except Exception as exc:
                logger.exception(
                    "Failed to parse generated HopChain query on line %s: %s", line_num, exc
                )
    logger.info("Loaded %s generated HopChain queries from %s", len(queries), input_path)
    return queries


def load_combinations(input_path: Path) -> dict[str, InstanceCombinationRecord]:
    """Load sampled instance combinations keyed by combination_id."""
    combinations: dict[str, InstanceCombinationRecord] = {}
    with input_path.open("r") as input_file:
        for line_num, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = InstanceCombinationRecord.model_validate(json.loads(line))
                combinations[record.combination_id] = record
            except Exception as exc:
                logger.exception(
                    "Failed to parse instance combination on line %s: %s", line_num, exc
                )
    logger.info("Loaded %s instance combinations from %s", len(combinations), input_path)
    return combinations


def join_records(
    queries: list[GeneratedHopChainQuery | ReconciledHopChainQuery],
    combinations_by_id: dict[str, InstanceCombinationRecord],
) -> list[RenderableHopchainRecord]:
    """Join generated queries with their sampled instance combinations."""
    joined_records: list[RenderableHopchainRecord] = []
    for query in queries:
        combination = combinations_by_id.get(query.combination_id)
        if combination is None:
            logger.warning("Missing combination record for %s", query.combination_id)
            continue
        joined_records.append(RenderableHopchainRecord(query=query, combination=combination))
    logger.info("Joined %s query rows with combination records", len(joined_records))
    return joined_records


def load_rgb_image(image_path: str) -> Image.Image:
    """Load an image and normalize to RGB for drawing and JPEG output."""
    image = Image.open(image_path)
    if image.mode in ("RGBA", "P", "LA"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode == "P":
            image = image.convert("RGBA")
        background.paste(image, mask=image.split()[-1] if image.mode in ("RGBA", "LA") else None)
        image.close()
        return background
    if image.mode != "RGB":
        converted = image.convert("RGB")
        image.close()
        return converted
    return image


def encode_pil_image_as_data_uri(image: Image.Image) -> str:
    """Encode a PIL image as a base64 JPEG data URI."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{image_data}"


def draw_text_label(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    text: str,
    fill_color: str,
    font: ImageFont.ImageFont,
) -> None:
    """Draw a text label with a solid colored background."""
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    else:
        width = int(draw.textlength(text, font=font))
        height = 14
        left, top, right, bottom = x, y, x + width, y + height
    draw.rectangle((left - 3, top - 2, right + 3, bottom + 2), fill=fill_color)
    draw.text((x, y), text, fill="white", font=font)


def build_annotated_image_uri(
    record: RenderableHopchainRecord, image_max_dimension: int | None
) -> str:
    """Render the source image with bounding boxes for the selected instances."""
    image = load_rgb_image(record.combination.image_path)
    try:
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        instance_order = {
            instance_id: idx for idx, instance_id in enumerate(record.query.involved_instance_ids)
        }
        default_offset = len(instance_order)
        sorted_instances = sorted(
            record.combination.instances,
            key=lambda instance: instance_order.get(instance.instance_id, default_offset),
        )

        for idx, instance in enumerate(sorted_instances):
            color = BOX_COLORS[idx % len(BOX_COLORS)]
            bbox = instance.bbox_xyxy
            draw.rectangle((bbox.x1, bbox.y1, bbox.x2, bbox.y2), outline=color, width=5)
            label = f"{idx + 1}. {instance.instance_id} ({instance.category})"
            label_y = max(0, bbox.y1 - 18)
            draw_text_label(draw, x=bbox.x1 + 2, y=label_y, text=label, fill_color=color, font=font)

        if image_max_dimension:
            image.thumbnail((image_max_dimension, image_max_dimension), Image.Resampling.LANCZOS)
        return encode_pil_image_as_data_uri(image)
    finally:
        image.close()


def render_clickable_image_html(
    *,
    image_uri: str,
    alt_text: str,
    caption: str,
    css_class: str,
    show_hint: bool,
) -> str:
    """Render a clickable image that opens in a larger modal."""
    escaped_alt = html.escape(alt_text)
    escaped_caption = html.escape(caption)
    hint_html = (
        '<div class="image-hint">Click image to open a larger view.</div>' if show_hint else ""
    )
    caption_html = f'<div class="image-caption">{escaped_caption}</div>' if caption else ""
    return (
        f'<a class="clickable-image-link" href="{image_uri}" target="_blank" rel="noopener noreferrer" '
        f'data-modal-caption="{escaped_caption}">'
        f'<img class="{css_class} clickable-image" src="{image_uri}" alt="{escaped_alt}" '
        f'data-modal-caption="{escaped_caption}" loading="lazy" />'
        "</a>"
        f"{hint_html}"
        f"{caption_html}"
    )


def build_instance_crops_html(record: RenderableHopchainRecord) -> str:
    """Render selected instance crops as small base64 thumbnails."""
    ordered_instances = []
    instances_by_id = {instance.instance_id: instance for instance in record.combination.instances}
    for instance_id in record.query.involved_instance_ids:
        if instance_id in instances_by_id:
            ordered_instances.append(instances_by_id[instance_id])
    if not ordered_instances:
        ordered_instances = list(record.combination.instances)

    cards = []
    for idx, instance in enumerate(ordered_instances):
        color = BOX_COLORS[idx % len(BOX_COLORS)]
        crop_uri = encode_image_as_data_uri(instance.crop_path, max_dimension=256)
        label = html.escape(f"{idx + 1}. {instance.instance_id} ({instance.category})")
        cards.append(
            f"""
            <div class="crop-card">
              {
                render_clickable_image_html(
                    image_uri=crop_uri,
                    alt_text=label,
                    caption=label,
                    css_class="crop-image",
                    show_hint=False,
                )
            }
              <div class="crop-label" style="border-left-color: {color};">{label}</div>
            </div>
            """
        )
    return f'<div class="crop-grid">{"".join(cards)}</div>'


def build_reasoning_hops_html(record: RenderableHopchainRecord) -> str:
    """Render reasoning hops as an ordered list."""
    if not record.query.query_metadata.reasoning_hops:
        return '<div class="empty-state">No reasoning hops available.</div>'

    hop_items = []
    for hop in record.query.query_metadata.reasoning_hops:
        objects = ", ".join(hop.objects_involved) if hop.objects_involved else "n/a"
        from_instance = hop.from_instance or "n/a"
        to_instance = hop.to_instance or "n/a"
        hop_items.append(
            f"""
            <li>
              <div><strong>Hop {hop.hop_number}</strong> | type: {html.escape(hop.hop_type)}</div>
              <div>from: {html.escape(from_instance)} | to: {html.escape(to_instance)}</div>
              <div>objects: {html.escape(objects)}</div>
              <div class="hop-description">{html.escape(hop.description)}</div>
              <div class="hop-output"><strong>Output:</strong> {html.escape(hop.output)}</div>
            </li>
            """
        )
    return f'<ol class="hop-list">{"".join(hop_items)}</ol>'


def build_collapsible_text_block_html(
    *,
    label: str,
    content_html: str,
    open_by_default: bool,
) -> str:
    """Render a labeled content block inside a collapsible section."""
    open_attr = " open" if open_by_default else ""
    escaped_label = html.escape(label)
    return f"""
    <details class="text-block"{open_attr}>
      <summary>{escaped_label}</summary>
      <div class="text-block-content">
        {content_html}
      </div>
    </details>
    """


def build_llm_judge_html(query: GeneratedHopChainQuery | ReconciledHopChainQuery) -> str:
    """Render compact LLM judge reconciliation metadata when present."""
    if not isinstance(query, ReconciledHopChainQuery):
        return ""

    judge_lines = [
        html.escape(f"{judge.judge_name} ({judge.provider}/{judge.model}): {judge.answer}")
        for judge in query.llm_judge_answers
    ]
    judge_answers_html = (
        "<pre>" + "\n".join(judge_lines) + "</pre>"
        if judge_lines
        else '<div class="empty-state">No LLM judge answers available.</div>'
    )
    consensus_lines = [
        f"Reconciliation status: {html.escape(query.llm_judge_reconciliation_status)}",
        f"Consensus answer: {html.escape(query.llm_judge_consensus_answer or 'n/a')}",
        (
            "Consensus matches hypothetical answer: "
            f"{str(query.llm_judge_consensus_matches_hypothetical_answer).lower()}"
        ),
    ]
    if query.llm_judge_rejection_reasons:
        consensus_lines.append(
            "Rejection reasons: "
            + ", ".join(html.escape(reason) for reason in query.llm_judge_rejection_reasons)
        )

    return build_collapsible_text_block_html(
        label="LLM Judge Consensus",
        content_html="<pre>" + "\n".join(consensus_lines) + "</pre>",
        open_by_default=True,
    ) + build_collapsible_text_block_html(
        label="LLM Judge Answers",
        content_html=judge_answers_html,
        open_by_default=True,
    )


def build_record_row(record: RenderableHopchainRecord, image_max_dimension: int | None) -> str:
    """Render one synthetic HopChain row as a side-by-side HTML section."""
    query = record.query
    annotated_image_uri = build_annotated_image_uri(record, image_max_dimension=image_max_dimension)
    image_name = html.escape(query.image_file_name)
    question = html.escape(query.question)
    answer = html.escape(query.hypothetical_answer)
    instance_ids = html.escape(", ".join(query.involved_instance_ids))
    instance_chain = html.escape(query.query_metadata.instance_chain)
    payload_json = html.escape(json.dumps(query.model_dump(), indent=2))

    return f"""
    <section class="record-row">
      <div class="image-panel">
        {
        render_clickable_image_html(
            image_uri=annotated_image_uri,
            alt_text=image_name,
            caption=image_name,
            css_class="annotated-image",
            show_hint=True,
        )
    }
        <div class="image-meta">
          <div><strong>{image_name}</strong></div>
          <div>image_id: {html.escape(query.image_id)}</div>
          <div>query_id: {html.escape(query.query_id)}</div>
          <div>combination_id: {html.escape(query.combination_id)}</div>
          <div>involved instances: {instance_ids}</div>
        </div>
        <div class="crop-section">
          <div class="label">Selected Instance Crops</div>
          {build_instance_crops_html(record)}
        </div>
      </div>
      <div class="data-panel">
        <div class="meta-grid">
          <div><strong>Hop count:</strong> {query.hop_count}</div>
          <div><strong>Primary capability:</strong> {
        html.escape(query.query_metadata.primary_capability)
    }</div>
          <div><strong>Uses all instances:</strong> {
        str(query.query_metadata.uses_all_instances).lower()
    }</div>
          <div><strong>Generator prompt version:</strong> {
        html.escape(query.query_metadata.generator_prompt_version or "n/a")
    }</div>
        </div>

        {
        build_collapsible_text_block_html(
            label="Question",
            content_html=f"<pre>{question}</pre>",
            open_by_default=True,
        )
    }

        {
        build_collapsible_text_block_html(
            label="Hypothetical Answer",
            content_html=f"<pre>{answer}</pre>",
            open_by_default=True,
        )
    }

        {build_llm_judge_html(query)}

        {
        build_collapsible_text_block_html(
            label="Instance Chain",
            content_html=f"<pre>{instance_chain}</pre>",
            open_by_default=True,
        )
    }

        {
        build_collapsible_text_block_html(
            label="Design Rationale",
            content_html=f"<pre>{html.escape(query.query_metadata.design_rationale)}</pre>",
            open_by_default=False,
        )
    }

        {
        build_collapsible_text_block_html(
            label="Reasoning Hops",
            content_html=build_reasoning_hops_html(record),
            open_by_default=True,
        )
    }

        <details class="payload-details">
          <summary>Raw generated JSON payload</summary>
          <pre>{payload_json}</pre>
        </details>
      </div>
    </section>
    """


def build_navigation(page_number: int, total_pages: int) -> str:
    """Render page navigation links."""
    links = ['<a href="index.html">Index</a>']
    if page_number > 1:
        links.append(f'<a href="hopchain_review_{page_number - 1:04d}.html">Previous</a>')
    if page_number < total_pages:
        links.append(f'<a href="hopchain_review_{page_number + 1:04d}.html">Next</a>')
    return f'<nav class="page-nav">{"".join(links)}</nav>'


def build_page_html(
    *,
    page_records: list[RenderableHopchainRecord],
    page_number: int,
    total_pages: int,
    total_records: int,
    rows_per_file: int,
    image_max_dimension: int | None,
    title: str,
) -> str:
    """Build one HTML page."""
    rows = "\n".join(
        build_record_row(record, image_max_dimension=image_max_dimension) for record in page_records
    )
    start_row = (page_number - 1) * rows_per_file + 1
    end_row = start_row + len(page_records) - 1

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)} | Page {page_number}</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f5f6f8;
      color: #111827;
    }}
    .page {{
      max-width: 1900px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 8px 0;
    }}
    .summary {{
      color: #4b5563;
      margin-bottom: 16px;
    }}
    .page-nav {{
      display: flex;
      gap: 12px;
      margin: 16px 0 24px;
    }}
    .page-nav a {{
      color: #2563eb;
      text-decoration: none;
      font-weight: 600;
    }}
    .record-row {{
      display: grid;
      grid-template-columns: minmax(340px, 760px) minmax(0, 1fr);
      gap: 20px;
      align-items: start;
      background: #ffffff;
      border: 1px solid #dbe1e8;
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 18px;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }}
    .annotated-image {{
      width: 100%;
      max-height: 700px;
      object-fit: contain;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      background: #fff;
    }}
    .clickable-image {{
      cursor: zoom-in;
    }}
    .clickable-image-link {{
      display: block;
      text-decoration: none;
      color: inherit;
    }}
    .image-hint {{
      margin-top: 8px;
      font-size: 12px;
      color: #4b5563;
    }}
    .image-caption {{
      margin-top: 4px;
      font-size: 12px;
      color: #6b7280;
      word-break: break-word;
    }}
    .image-meta {{
      margin-top: 10px;
      font-size: 13px;
      line-height: 1.5;
    }}
    .crop-section {{
      margin-top: 14px;
    }}
    .crop-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
      gap: 10px;
    }}
    .crop-card {{
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      overflow: hidden;
    }}
    .crop-image {{
      width: 100%;
      height: 120px;
      object-fit: contain;
      background: #fff;
      display: block;
    }}
    .crop-label {{
      font-size: 12px;
      padding: 8px;
      border-left: 6px solid #2563eb;
      line-height: 1.35;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px 16px;
      margin-bottom: 16px;
      font-size: 14px;
    }}
    .text-block {{
      margin-bottom: 14px;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      background: #f9fafb;
      overflow: hidden;
    }}
    .text-block summary {{
      cursor: pointer;
      list-style: none;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: #4b5563;
      padding: 10px 12px;
      background: #f9fafb;
    }}
    .text-block summary::-webkit-details-marker {{
      display: none;
    }}
    .text-block summary::before {{
      content: "▸";
      display: inline-block;
      margin-right: 8px;
      color: #6b7280;
    }}
    .text-block[open] summary::before {{
      content: "▾";
    }}
    .text-block-content {{
      border-top: 1px solid #e5e7eb;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: #f9fafb;
      padding: 10px 12px;
      line-height: 1.45;
      font-size: 13px;
    }}
    .hop-list {{
      margin: 0;
      padding-left: 20px;
    }}
    .hop-list li {{
      margin-bottom: 10px;
      line-height: 1.45;
    }}
    .hop-description,
    .hop-output {{
      margin-top: 4px;
    }}
    .payload-details {{
      margin-top: 16px;
    }}
    .payload-details summary {{
      cursor: pointer;
      font-weight: 600;
      margin-bottom: 8px;
    }}
    .image-modal {{
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      padding: 24px;
      background: rgba(17, 24, 39, 0.88);
      z-index: 9999;
    }}
    .image-modal.open {{
      display: flex;
    }}
    .image-modal-content {{
      position: relative;
      max-width: min(1400px, 96vw);
      max-height: 92vh;
      padding: 18px;
      border-radius: 12px;
      background: #111827;
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.45);
    }}
    .image-modal-close {{
      position: absolute;
      top: 8px;
      right: 10px;
      border: 0;
      background: transparent;
      color: #f9fafb;
      font-size: 34px;
      line-height: 1;
      cursor: pointer;
    }}
    .image-modal-close:hover {{
      color: #d1d5db;
    }}
    .image-modal img {{
      display: block;
      max-width: min(1320px, 92vw);
      max-height: calc(92vh - 80px);
      object-fit: contain;
      margin: 0 auto;
      background: #fff;
      border-radius: 8px;
    }}
    .image-modal-caption {{
      margin-top: 10px;
      font-size: 12px;
      color: #d1d5db;
      word-break: break-word;
      text-align: center;
    }}
    .empty-state {{
      color: #6b7280;
      font-style: italic;
    }}
    @media (max-width: 1200px) {{
      .record-row {{
        grid-template-columns: 1fr;
      }}
      .meta-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>{html.escape(title)}</h1>
    <p class="summary">Page {page_number} of {total_pages} | Rows {start_row}-{end_row} of {total_records}</p>
    {build_navigation(page_number, total_pages)}
    {rows}
    {build_navigation(page_number, total_pages)}
  </div>
  <div class="image-modal" id="image-modal" aria-hidden="true">
    <div class="image-modal-content">
      <button class="image-modal-close" id="image-modal-close" aria-label="Close image modal">&times;</button>
      <img id="image-modal-img" src="" alt="" />
      <div class="image-modal-caption" id="image-modal-caption"></div>
    </div>
  </div>
  <script>
    (() => {{
      const modal = document.getElementById("image-modal");
      const modalImage = document.getElementById("image-modal-img");
      const modalCaption = document.getElementById("image-modal-caption");
      const closeButton = document.getElementById("image-modal-close");
      const imageLinks = document.querySelectorAll(".clickable-image-link");

      function closeModal() {{
        modal.classList.remove("open");
        modal.setAttribute("aria-hidden", "true");
        modalImage.src = "";
        modalImage.alt = "";
        modalCaption.textContent = "";
      }}

      function openModal(link) {{
        const image = link.querySelector(".clickable-image");
        if (!image) {{
          return;
        }}
        modalImage.src = image.src;
        modalImage.alt = image.alt || "";
        modalCaption.textContent = link.dataset.modalCaption || image.dataset.modalCaption || image.alt || "";
        modal.classList.add("open");
        modal.setAttribute("aria-hidden", "false");
      }}

      imageLinks.forEach((link) => {{
        link.addEventListener("click", (event) => {{
          event.preventDefault();
          openModal(link);
        }});
      }});

      closeButton.addEventListener("click", closeModal);
      modal.addEventListener("click", (event) => {{
        if (event.target === modal) {{
          closeModal();
        }}
      }});
      document.addEventListener("keydown", (event) => {{
        if (event.key === "Escape" && modal.classList.contains("open")) {{
          closeModal();
        }}
      }});
    }})();
  </script>
</body>
</html>
"""


def build_index_html(summary: HopchainHtmlVisualizationSummary, title: str) -> str:
    """Build the index page that links to all rendered HTML chunks."""
    if summary.page_files:
        rows = "\n".join(
            f"""
            <tr>
              <td>{page.page_number}</td>
              <td><a href="{html.escape(page.file_name)}">{html.escape(page.file_name)}</a></td>
              <td>{page.row_count}</td>
              <td>{page.start_row}-{page.end_row}</td>
            </tr>
            """
            for page in summary.page_files
        )
        table_html = f"""
        <table>
          <thead>
            <tr>
              <th>Page</th>
              <th>File</th>
              <th>Rows</th>
              <th>Range</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
        """
    else:
        table_html = '<p class="empty-state">No generated query rows were available to render.</p>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      background: #f9fafb;
      color: #111827;
    }}
    .summary {{
      margin-bottom: 20px;
      color: #4b5563;
      line-height: 1.5;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: #fff;
    }}
    th, td {{
      border: 1px solid #d1d5db;
      padding: 10px 12px;
      text-align: left;
    }}
    th {{
      background: #f3f4f6;
    }}
    a {{
      color: #2563eb;
      text-decoration: none;
    }}
    .empty-state {{
      color: #6b7280;
      font-style: italic;
    }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="summary">
    <div>Total queries: {summary.total_queries}</div>
    <div>Rendered queries: {summary.rendered_queries}</div>
    <div>Rows per file: {summary.rows_per_file}</div>
    <div>Pages generated: {len(summary.page_files)}</div>
    <div>Queries input: {html.escape(summary.queries_input)}</div>
    <div>Combinations input: {html.escape(summary.combinations_input)}</div>
  </div>
  {table_html}
</body>
</html>
"""


def main() -> None:
    """Entry point."""
    args = parse_args()
    if args.sample_count is not None and args.sample_count <= 0:
        raise SystemExit("error: --sample-count must be a positive integer")
    queries_input = Path(args.queries_input)
    combinations_input = Path(args.combinations_input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_max_dimension = None if args.image_max_dimension == 0 else args.image_max_dimension

    queries = load_queries(queries_input)
    combinations_by_id = load_combinations(combinations_input)
    renderable_records = join_records(queries, combinations_by_id)
    sample_seed = None
    if args.sample_count is not None and args.sample_count < len(renderable_records):
        random.Random(args.sample_seed).shuffle(renderable_records)
        renderable_records = renderable_records[: args.sample_count]
        sample_seed = args.sample_seed

    total_pages = (
        max(1, math.ceil(len(renderable_records) / args.rows_per_file)) if renderable_records else 0
    )
    page_summaries: list[HtmlPageSummary] = []

    for page_number in range(1, total_pages + 1):
        start_idx = (page_number - 1) * args.rows_per_file
        end_idx = min(start_idx + args.rows_per_file, len(renderable_records))
        page_records = renderable_records[start_idx:end_idx]
        page_file_name = f"hopchain_review_{page_number:04d}.html"
        page_path = output_dir / page_file_name
        page_path.write_text(
            build_page_html(
                page_records=page_records,
                page_number=page_number,
                total_pages=total_pages,
                total_records=len(renderable_records),
                rows_per_file=args.rows_per_file,
                image_max_dimension=image_max_dimension,
                title=args.title,
            )
        )
        page_summaries.append(
            HtmlPageSummary(
                page_number=page_number,
                file_name=page_file_name,
                row_count=len(page_records),
                start_row=start_idx + 1,
                end_row=end_idx,
            )
        )

    summary = HopchainHtmlVisualizationSummary(
        queries_input=str(queries_input),
        combinations_input=str(combinations_input),
        output_dir=str(output_dir),
        index_file=str(output_dir / "index.html"),
        total_queries=len(queries),
        rendered_queries=len(renderable_records),
        sample_seed=sample_seed,
        rows_per_file=args.rows_per_file,
        image_max_dimension=image_max_dimension,
        page_files=page_summaries,
    )

    (output_dir / "index.html").write_text(build_index_html(summary, title=args.title))
    Path(args.summary).write_text(summary.model_dump_json(indent=2))
    logger.info("Wrote %s HTML page(s) to %s", len(page_summaries), output_dir)


if __name__ == "__main__":
    main()
