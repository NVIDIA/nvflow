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
"""Generic HTML-to-Markdown chunking engine for document-grounded SDG.

Converts HTML documents into token-limited chunks in three formats:
Markdown (.txt), Clean HTML (_clean.html), and Original HTML (_orig.html).

Domain-specific file iteration and header generation are injected via
``file_iter`` and ``header_builder`` callbacks in ``run_chunking``.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag

from nvflow.utils import setup_logger

logger = setup_logger(__name__)

try:
    import tiktoken

    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

DEFAULT_MODEL = "gpt-4"
HEADING_TAGS = {"h1", "h2", "h3", "h4"}
SKIP_DIR_NAMES = {"chunked_html", "chunks", "chunked_unified"}

Block = dict[str, Any]


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def get_encoder(model_name: str = DEFAULT_MODEL):
    """Get tiktoken encoder for token counting."""
    if not HAS_TIKTOKEN:

        class FakeEncoder:
            def encode(self, text):
                return text.split()

        return FakeEncoder()

    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# HTML processing helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Section splitting and chunking
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Write chunk outputs
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Process a single file and run_chunking orchestrator
# ---------------------------------------------------------------------------


def process_html_file(
    html_path: Path,
    encoder,
    max_tokens: int,
    overlap_tokens: int,
    output_root: Path,
    rel_parent: Path,
    *,
    header_builder: Callable[[Path], str] | None = None,
) -> int:
    """Process a single HTML file and create chunks.

    Args:
        html_path: Path to the HTML file
        encoder: Tiktoken encoder instance
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Overlap tokens between chunks
        output_root: Root directory for chunk outputs
        rel_parent: Relative parent path for directory structure
        header_builder: Optional callback ``(html_path) -> str`` that
            generates a header info string for chunks.  Defaults to
            returning the filename.
    """
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

    if header_builder is not None:
        header_info = header_builder(html_path)
    else:
        header_info = f"File: {html_path.name}"

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
    file_iter: Callable[[Path], Iterable[Path]],
    *,
    max_tokens: int = 2000,
    overlap_tokens: int = 100,
    model: str = DEFAULT_MODEL,
    header_builder: Callable[[Path], str] | None = None,
) -> Path:
    """Run the chunking process on HTML files yielded by file_iter.

    Args:
        input_dir: Root input directory (used to compute relative paths)
        output_dir: Root output directory for chunks
        file_iter: ``(input_dir) -> Iterable[Path]`` that yields HTML
            file paths to process.  For SEC filings this walks
            ticker/form/year directories; other domains provide their own.
        max_tokens: Maximum tokens per chunk (default 2000)
        overlap_tokens: Overlap tokens between chunks (default 100)
        model: Tiktoken model name for tokenizer (default "gpt-4")
        header_builder: Optional ``(html_path) -> str`` for chunk headers
    """
    logger.info("Starting chunking from %s to %s", input_dir, output_dir)

    encoder = get_encoder(model)
    html_files = list(file_iter(input_dir))

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
            header_builder=header_builder,
        )
        total_chunks += chunks_created

    logger.info("Processed %d file(s), created %d chunk(s)", len(html_files), total_chunks)
    return output_dir


def _default_file_iter(input_dir: Path) -> Iterable[Path]:
    """Default file iterator: recursively find all *.html files."""
    return sorted(input_dir.rglob("*.html"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk HTML files into token-limited segments.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Input directory containing HTML files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for chunks",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2000,
        help="Maximum tokens per chunk (default: 2000)",
    )
    parser.add_argument(
        "--overlap_tokens",
        type=int,
        default=100,
        help="Overlap tokens between chunks (default: 100)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Tiktoken model name (default: {DEFAULT_MODEL})",
    )

    args = parser.parse_args()
    run_chunking(
        args.input_dir,
        args.output_dir,
        file_iter=_default_file_iter,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        model=args.model,
    )
