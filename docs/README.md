# Documentation Guide

The `docs/` directory contains all written deliverables for the AI Development Workflow assignment.

- `ai_workflow_report.md` – primary narrative suitable for conversion to PDF via Markdown export (e.g., VS Code, Pandoc).
- `ai_workflow_report.tex` – LaTeX source for professional PDF compilation.
- `plp_article_post.md` – template for publishing on the PLP Academy Community.

## Exporting the PDF

### Option 1: Markdown to PDF (Pandoc)
```bash
pandoc ai_workflow_report.md -o ai_workflow_report.pdf --from markdown --template eisvogel --listings
```

### Option 2: LaTeX Compilation
```bash
pdflatex ai_workflow_report.tex
pdflatex ai_workflow_report.tex
```

The generated `ai_workflow_report.pdf` should be 5–10 pages and included in the repository prior to submission.

