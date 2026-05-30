# research-note-agent

`research-note-agent` is a small workflow repository for turning research PDFs into structured Markdown notes and publishing concise notes plus detailed investigation pages to Notion.

## What this repository does

1. Stores research papers in `papers/`.
2. Extracts PDF text content with `scripts/extract_pdf.py`.
3. Generates structured research notes in `notes/` with `scripts/generate_note.py`.
4. Writes concise notes to a configured Notion parent page with `scripts/publish_to_notion.py`.
5. Creates detailed investigation subpages in Notion from the full Markdown note.

## Repository structure

```text
papers/      # local PDFs and paper assets; PDFs are ignored by default
notes/       # generated or curated Markdown research notes
scripts/     # extraction, note generation, and Notion publishing scripts
prompts/     # prompt templates used by generation scripts
notes/read-paper-template.md  # reusable manual reading-note template
README.md    # project documentation
requirements.txt
.env.example
```

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the example environment file and fill in local secrets:

```bash
cp .env.example .env
```

Required environment variables:

- `OPENAI_API_KEY`: API key used by `scripts/generate_note.py`.
- `NOTION_TOKEN`: Notion integration token used by `scripts/publish_to_notion.py`.
- `NOTION_PARENT_PAGE_ID`: parent page where notes and investigation subpages are created.
- `NOTION_PARENT_PAGE_URL`: optional alternative to `NOTION_PARENT_PAGE_ID`; useful when copying a Notion page link.

Do not commit `.env` or any real credentials.

## Usage

### 1. Add a paper locally

Place a PDF in `papers/`. PDF files are ignored by default to avoid accidentally committing copyrighted content.

### 2. Extract PDF content

```bash
python scripts/extract_pdf.py papers/example.pdf --output papers/example.txt
```

### 3. Start from the reading-note template

For manual paper reading, copy the reusable template first:

```bash
cp notes/read-paper-template.md notes/attention-is-all-you-need.md
```

Fill in the metadata, research questions, implementation track, optimization track, and `## Notion concise note` section before publishing. The template is designed for repeated paper reading and for Notion publishing.

### 4. Generate a structured note

```bash
python scripts/generate_note.py papers/example.txt --output notes/example.md
```

By default, the script uses the prompt template in `prompts/research_note.md` and the model from `OPENAI_MODEL` or `gpt-4.1-mini`. For reading notes that follow the manual template more closely, pass `--prompt prompts/read_paper_note.md`.

### 5. Publish to Notion

```bash
python scripts/publish_to_notion.py notes/example.md
```

To target a copied Notion page link directly, pass it as `--parent-page`:

```bash
python scripts/publish_to_notion.py notes/example.md --parent-page "https://app.notion.com/p/Paper-Notes-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx?source=copy_link"
```

To write the concise note directly into that existing parent page and create the detailed investigation as a subpage beneath it, add `--append-to-parent`:

```bash
python scripts/publish_to_notion.py notes/example.md --parent-page "https://app.notion.com/p/Paper-Notes-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx?source=copy_link" --append-to-parent
```

Without `--append-to-parent`, the publisher creates:

- a concise child page under the configured parent page, based on the `## Notion concise note` section when present; and
- a detailed investigation subpage containing the full Markdown note.

## Security notes

- Never commit `NOTION_TOKEN`, `NOTION_PARENT_PAGE_ID`, `OPENAI_API_KEY`, or `.env`.
- Review generated notes before publishing them to Notion.
- Confirm you have permission before storing or sharing PDFs.
