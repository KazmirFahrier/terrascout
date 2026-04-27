"""Render the TerraScout project one-pager Markdown to a one-page PDF."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import textwrap

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


DOCS_DIR = Path(__file__).resolve().parent
SOURCE = DOCS_DIR / "PROJECT_ONE_PAGER.md"
OUTPUT = DOCS_DIR / "PROJECT_ONE_PAGER.pdf"
PDF_METADATA = {
    "Title": "TerraScout One-Pager",
    "Author": "TerraScout",
    "Creator": "docs/render_project_one_pager_pdf.py",
    "Producer": "matplotlib",
    "CreationDate": datetime(2026, 1, 1, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 1, 1, tzinfo=timezone.utc),
}


def render_one_pager(source: Path = SOURCE, output: Path = OUTPUT) -> Path:
    """Render a compact one-page PDF summary."""

    sections = _parse_sections(source.read_text())
    with PdfPages(output, metadata=PDF_METADATA) as pdf:
        fig = plt.figure(figsize=(8.5, 11.0))
        fig.patch.set_facecolor("white")
        plt.axis("off")

        fig.text(0.06, 0.965, "TerraScout", fontsize=24, fontweight="bold", va="top")
        fig.text(
            0.06,
            0.925,
            "Modular autonomy stack for a GPS-degraded orchard inspection rover",
            fontsize=10.5,
            color="#333333",
            va="top",
        )

        _draw_text_block(fig, 0.06, 0.875, "Summary", sections["summary"], width=78, lines=4)
        _draw_stack(fig, sections["stack"])
        _draw_metrics(fig, sections["metrics"])
        _draw_text_block(fig, 0.06, 0.205, "Reproduce", sections["reproduce"], width=88, lines=5)
        _draw_text_block(fig, 0.06, 0.105, "Current Roadmap", sections["roadmap"], width=88, lines=4)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    return output


def _parse_sections(markdown: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {
        "summary": [],
        "stack": [],
        "metrics": [],
        "reproduce": [],
        "roadmap": [],
    }
    current = ""
    in_code = False
    for raw in markdown.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if line.startswith("## "):
            title = line[3:].lower()
            if title.startswith("summary"):
                current = "summary"
            elif title.startswith("implemented"):
                current = "stack"
            elif title.startswith("current"):
                current = "metrics"
            elif title.startswith("reproduce"):
                current = "reproduce"
            elif title.startswith("roadmap"):
                current = "roadmap"
            else:
                current = ""
            continue
        if not current or not line or line.startswith("| ---") or line.startswith("#"):
            continue
        if in_code:
            sections[current].append(line)
        elif line.startswith("|") or line.startswith("-") or current == "summary":
            sections[current].append(line)
    return sections


def _draw_text_block(
    fig: plt.Figure,
    x: float,
    y: float,
    title: str,
    content: list[str],
    width: int,
    lines: int,
) -> None:
    fig.text(x, y, title, fontsize=12, fontweight="bold", va="top")
    text = " ".join(_clean_line(line) for line in content)
    wrapped = textwrap.wrap(text, width=width)[:lines]
    for idx, line in enumerate(wrapped):
        fig.text(x, y - 0.027 * (idx + 1), line, fontsize=8.5, va="top")


def _draw_stack(fig: plt.Figure, rows: list[str]) -> None:
    fig.text(0.06, 0.745, "Implemented Stack", fontsize=12, fontweight="bold", va="top")
    y = 0.718
    for raw in rows:
        cells = _table_cells(raw)
        if len(cells) != 3 or cells[0] == "Layer":
            continue
        fig.text(0.06, y, cells[0], fontsize=7.5, fontweight="bold", va="top")
        fig.text(0.205, y, cells[1], fontsize=7.5, va="top")
        fig.text(0.39, y, _clip(cells[2], 64), fontsize=7.5, va="top")
        y -= 0.025


def _draw_metrics(fig: plt.Figure, rows: list[str]) -> None:
    fig.text(0.06, 0.47, "Headline Metrics", fontsize=12, fontweight="bold", va="top")
    y = 0.443
    metrics = [_table_cells(row) for row in rows]
    selected = [
        cells
        for cells in metrics
        if len(cells) == 2
        and cells[0] not in {"Metric"}
        and cells[0]
        in {
            "Mission inspection success",
            "Collision events",
            "Particle-filter relocalization",
            "Resource scheduler oracle gap",
            "Hybrid A* steering effort",
            "EKF-SLAM map accuracy",
            "30-row acceptance pass",
            "30-row wall time",
        }
    ]
    for metric, value in selected:
        fig.text(0.06, y, metric, fontsize=8, va="top")
        fig.text(0.57, y, value, fontsize=8, fontweight="bold", va="top")
        y -= 0.026


def _clean_line(line: str) -> str:
    if line.startswith("- "):
        return line[2:]
    return line.replace("`", "")


def _table_cells(row: str) -> list[str]:
    return [cell.strip().replace("`", "") for cell in row.strip("|").split("|")]


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def main() -> None:
    print(render_one_pager())


if __name__ == "__main__":
    main()
