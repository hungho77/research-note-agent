#!/usr/bin/env python3
"""Publish concise and detailed research notes to Notion."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

MAX_BLOCK_TEXT = 1900
NOTION_PAGE_ID_RE = re.compile(r"[0-9a-fA-F]{32}")


def load_dotenv_if_available() -> None:
    """Load a local .env file when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return

    load_dotenv()


def notion_client(token: str) -> Any:
    """Create a Notion API client after dependencies are installed."""
    try:
        from notion_client import Client
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: notion-client. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    return Client(auth=token)


def normalize_notion_page_id(value: str) -> str:
    """Return a Notion page ID from either a raw ID or a Notion page URL."""
    compact = value.replace("-", "")
    match = NOTION_PAGE_ID_RE.search(compact)
    if not match:
        raise SystemExit(
            "Could not find a Notion page ID. Provide a 32-character page ID "
            "or a Notion page URL."
        )
    page_id = match.group(0).lower()
    return (
        f"{page_id[:8]}-{page_id[8:12]}-{page_id[12:16]}-"
        f"{page_id[16:20]}-{page_id[20:]}"
    )


def markdown_heading_to_title(markdown: str, fallback: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line.removeprefix("# ").strip() or fallback
    return fallback


def extract_concise_note(markdown: str) -> str:
    section_name = "## Notion concise note"
    if section_name not in markdown:
        return markdown[:MAX_BLOCK_TEXT]

    after_section = markdown.split(section_name, 1)[1].strip()
    next_section = after_section.find("\n## ")
    if next_section != -1:
        after_section = after_section[:next_section].strip()
    return after_section or markdown[:MAX_BLOCK_TEXT]


def chunk_text(text: str, chunk_size: int = MAX_BLOCK_TEXT) -> list[str]:
    return [
        text[index : index + chunk_size]
        for index in range(0, len(text), chunk_size)
    ] or [""]


def rich_text(content: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": {"content": content}}]


def paragraph_block(text: str) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": rich_text(text)},
    }


def heading_block(text: str, level: int) -> dict[str, Any]:
    heading_type = f"heading_{min(max(level, 1), 3)}"
    return {
        "object": "block",
        "type": heading_type,
        heading_type: {"rich_text": rich_text(text)},
    }


def bullet_block(text: str) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {"rich_text": rich_text(text)},
    }


def markdown_to_blocks(markdown: str) -> list[dict[str, Any]]:
    """Convert a small Markdown subset into Notion blocks."""
    blocks: list[dict[str, Any]] = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        paragraph = "\n".join(paragraph_lines).strip()
        for chunk in chunk_text(paragraph):
            blocks.append(paragraph_block(chunk))
        paragraph_lines.clear()

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            continue

        if line.startswith("### "):
            flush_paragraph()
            blocks.append(heading_block(line.removeprefix("### ").strip(), 3))
        elif line.startswith("## "):
            flush_paragraph()
            blocks.append(heading_block(line.removeprefix("## ").strip(), 2))
        elif line.startswith("# "):
            flush_paragraph()
            blocks.append(heading_block(line.removeprefix("# ").strip(), 1))
        elif line.startswith(("- ", "* ")):
            flush_paragraph()
            blocks.append(bullet_block(line[2:].strip()))
        else:
            paragraph_lines.append(line)

    flush_paragraph()
    return blocks or [paragraph_block("")]


def append_blocks(notion: Any, page_id: str, content: str) -> None:
    notion.blocks.children.append(
        block_id=page_id,
        children=markdown_to_blocks(content),
    )


def create_page(notion: Any, parent_page_id: str, title: str, content: str) -> str:
    response = notion.pages.create(
        parent={"page_id": parent_page_id},
        properties={"title": {"title": [{"text": {"content": title}}]}},
        children=markdown_to_blocks(content),
    )
    return response["id"]


def parent_page_from_args(args: argparse.Namespace) -> str:
    raw_parent = (
        args.parent_page
        or os.getenv("NOTION_PARENT_PAGE_ID")
        or os.getenv("NOTION_PARENT_PAGE_URL")
    )
    if not raw_parent:
        raise SystemExit(
            "A Notion parent page is required. Set NOTION_PARENT_PAGE_ID, "
            "NOTION_PARENT_PAGE_URL, or pass --parent-page."
        )
    return normalize_notion_page_id(raw_parent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a research note to Notion.")
    parser.add_argument("note", type=Path, help="Path to a Markdown note in notes/.")
    parser.add_argument(
        "--parent-page",
        help="Notion parent page ID or URL. Defaults to NOTION_PARENT_PAGE_ID/URL.",
    )
    parser.add_argument(
        "--append-to-parent",
        action="store_true",
        help=(
            "Append the concise note directly to the parent page before "
            "creating the investigation subpage."
        ),
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv_if_available()
    args = parse_args()

    notion_token = os.getenv("NOTION_TOKEN")
    if not notion_token:
        raise SystemExit("NOTION_TOKEN is required.")

    parent_page_id = parent_page_from_args(args)
    markdown = args.note.read_text(encoding="utf-8")
    title = markdown_heading_to_title(markdown, args.note.stem)
    concise_note = extract_concise_note(markdown)
    notion = notion_client(notion_token)

    if args.append_to_parent:
        append_blocks(notion, parent_page_id, f"## {title}\n\n{concise_note}")
        concise_page_id = parent_page_id
        print(f"Appended concise note to parent page: {parent_page_id}")
    else:
        concise_page_id = create_page(
            notion=notion,
            parent_page_id=parent_page_id,
            title=f"Research note: {title}",
            content=concise_note,
        )
        print(f"Created concise note page: {concise_page_id}")

    investigation_page_id = create_page(
        notion=notion,
        parent_page_id=concise_page_id,
        title=f"Investigation: {title}",
        content=markdown,
    )
    print(f"Created investigation subpage: {investigation_page_id}")


if __name__ == "__main__":
    main()
