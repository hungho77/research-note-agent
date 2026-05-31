# Paper Reading Project

This workspace is for reading research papers and turning each paper into a clear technical note.

## Goal

For every paper, capture:

- The real pain point or research problem the paper addresses.
- The key idea and main contributions.
- The method, assumptions, and important design choices.
- The evidence: datasets, metrics, baselines, ablations, and results.
- The limitations, weak points, and failure cases.
- The practical value: when the paper is useful and when it is not.
- Follow-up ideas for experiments, implementation, or future research.

## Folder Workflow

- `Unread/`: papers waiting to be reviewed.
- `Read/`: papers already reviewed.
- `notes/`: one note per reviewed paper.
- `assets/figures/`: original diagrams used inside blog-style notes.
- `templates/`: reusable Markdown and LaTeX templates.

## Reading Process

1. Skim the abstract, introduction, figures, and conclusion.
2. Write the paper's pain point in one or two sentences.
3. Identify the main contribution, not just the model name.
4. Map the method: inputs, outputs, architecture, training objective, and evaluation setup.
5. Check whether the experiments actually support the claims.
6. List limitations and unclear assumptions.
7. Decide whether the paper is worth implementing, citing, or only remembering.

## Recommended Note Names

Use this pattern:

```text
notes/YYYY-MM-DD_short-paper-title.md
```

For a polished report, copy `templates/paper-review-template.tex` and fill it in.

## Illustrated Blog Notes

Use `notes/2026-05-31_transformer-illustrated.md` as an example for blog-style paper notes with embedded figures.

Recommended flow:

1. Explain the problem in plain language.
2. Add one original diagram for the core idea.
3. Explain the diagram before adding equations or implementation details.
4. Keep figures in `assets/figures/<paper-name>/` and link them from the note.
