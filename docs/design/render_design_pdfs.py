"""Render TerraScout design notes from Markdown to simple PDFs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import textwrap

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


SOURCE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SOURCE_DIR / "pdf"
PAGE_WIDTH_IN = 8.5
PAGE_HEIGHT_IN = 11.0
LEFT = 0.08
TOP = 0.94
LINE_HEIGHT = 0.022
CHARS_PER_LINE = 92
PDF_METADATA = {
    "Title": "TerraScout Design Note",
    "Author": "TerraScout",
    "Creator": "docs/design/render_design_pdfs.py",
    "Producer": "matplotlib",
    "CreationDate": datetime(2026, 1, 1, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 1, 1, tzinfo=timezone.utc),
}


def render_all() -> list[Path]:
    """Render every layer note to a PDF and return the written paths."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    for source in sorted(SOURCE_DIR.glob("l*.md")):
        output = OUTPUT_DIR / f"{source.stem}.pdf"
        render_markdown(source, output)
        rendered.append(output)
    return rendered


def render_markdown(source: Path, output: Path) -> None:
    """Render one Markdown file into a readable text PDF."""

    lines = _markdown_to_lines(source.read_text())
    with PdfPages(output, metadata=PDF_METADATA) as pdf:
        page_lines: list[tuple[str, str]] = []
        y = TOP
        for style, line in lines:
            needed = LINE_HEIGHT * (1.45 if style == "h1" else 1.0)
            if y - needed < 0.06:
                _write_page(pdf, page_lines)
                page_lines = []
                y = TOP
            page_lines.append((style, line))
            y -= needed
        if page_lines:
            _write_page(pdf, page_lines)


def _markdown_to_lines(markdown: str) -> list[tuple[str, str]]:
    rendered: list[tuple[str, str]] = []
    in_code = False
    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if line.startswith("```"):
            in_code = not in_code
            rendered.append(("code", ""))
            continue
        if in_code:
            rendered.append(("code", line))
            continue
        if line.startswith("# "):
            rendered.append(("h1", line[2:]))
            rendered.append(("body", ""))
            continue
        if line.startswith("## "):
            rendered.append(("h2", line[3:]))
            continue
        if not line:
            rendered.append(("body", ""))
            continue
        prefix = "- " if line.startswith("- ") else ""
        content = line[2:] if prefix else line
        wrapped = textwrap.wrap(content, width=CHARS_PER_LINE - len(prefix)) or [""]
        for idx, wrapped_line in enumerate(wrapped):
            rendered.append(("body", f"{prefix if idx == 0 else '  '}{wrapped_line}"))
    return rendered


def _write_page(pdf: PdfPages, lines: list[tuple[str, str]]) -> None:
    fig = plt.figure(figsize=(PAGE_WIDTH_IN, PAGE_HEIGHT_IN))
    fig.patch.set_facecolor("white")
    plt.axis("off")
    y = TOP
    for style, line in lines:
        if style == "h1":
            size = 14
            weight = "bold"
            family = "DejaVu Sans"
        elif style == "h2":
            size = 11
            weight = "bold"
            family = "DejaVu Sans"
        elif style == "code":
            size = 8
            weight = "normal"
            family = "DejaVu Sans Mono"
        else:
            size = 9
            weight = "normal"
            family = "DejaVu Sans"
        fig.text(LEFT, y, line, fontsize=size, fontweight=weight, family=family, va="top")
        y -= LINE_HEIGHT * (1.45 if style == "h1" else 1.0)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for output in render_all():
        print(output)


if __name__ == "__main__":
    main()
