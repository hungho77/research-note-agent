#!/usr/bin/env python3
"""Extract text from a PDF into a UTF-8 text file."""

from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader


def extract_pdf_text(pdf_path: Path) -> str:
    """Return text extracted from every page in a PDF."""
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []

    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(f"\n\n--- Page {index} ---\n\n{text.strip()}")

    return "".join(pages).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract text from a research PDF.")
    parser.add_argument("pdf", type=Path, help="Path to the PDF in papers/.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Destination text file. Defaults to the PDF path with a .txt suffix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path = args.pdf

    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise SystemExit(f"Expected a .pdf file, got: {pdf_path}")

    output_path = args.output or pdf_path.with_suffix(".txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(extract_pdf_text(pdf_path), encoding="utf-8")
    print(f"Extracted text to {output_path}")


if __name__ == "__main__":
    main()
