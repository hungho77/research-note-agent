#!/usr/bin/env python3
"""Generate structured Markdown research notes from extracted paper text."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_PROMPT = Path("prompts/research_note.md")


def read_text(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    return path.read_text(encoding="utf-8")


def generate_note(paper_text: str, prompt: str, model: str) -> str:
    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": paper_text,
            },
        ],
    )
    return response.output_text.strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a structured research note.")
    parser.add_argument("input", type=Path, help="Extracted paper text file.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Destination Markdown note. Defaults to notes/<input-stem>.md.",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=DEFAULT_PROMPT,
        help="Prompt template path.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        help="OpenAI model to use.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required.")

    paper_text = read_text(args.input)
    prompt = read_text(args.prompt)
    output_path = args.output or Path("notes") / f"{args.input.stem}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    note = generate_note(paper_text=paper_text, prompt=prompt, model=args.model)
    output_path.write_text(note, encoding="utf-8")
    print(f"Generated note at {output_path}")


if __name__ == "__main__":
    main()
