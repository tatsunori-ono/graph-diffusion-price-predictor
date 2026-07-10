"""Report rendering helpers: Markdown table formatting and Markdown -> PDF.

PDF generation strategy (in order of preference):
1. ``pandoc`` (+ a LaTeX engine such as ``pdflatex`` or ``xelatex``) if
   available on PATH -- produces the highest quality typeset PDF.
2. A ``reportlab``-based fallback that parses the same Markdown (headings,
   paragraphs, pipe tables, image references) into a reasonably formatted
   PDF, used only if pandoc/LaTeX are not available in the environment.
3. If neither works, the Markdown file alone is still produced, and the
   caller is told exactly which command to run later to get a PDF.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def fmt_pct(x, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{x * 100:.{decimals}f}%"


def fmt_num(x, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{x:.{decimals}f}"


def df_to_markdown_table(df: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    df_fmt = df.copy()
    for col in df_fmt.columns:
        if pd.api.types.is_float_dtype(df_fmt[col]):
            df_fmt[col] = df_fmt[col].map(lambda v: float_format.format(v) if pd.notna(v) else "n/a")
    header = "| " + " | ".join([str(df_fmt.index.name or "")] + [str(c) for c in df_fmt.columns]) + " |"
    sep = "| " + " | ".join(["---"] * (len(df_fmt.columns) + 1)) + " |"
    rows = []
    for idx, row in df_fmt.iterrows():
        rows.append("| " + " | ".join([str(idx)] + [str(v) for v in row.values]) + " |")
    return "\n".join([header, sep] + rows)


def write_markdown(content: str, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    logger.info("Wrote markdown report: %s", path)
    return path


def _pandoc_available() -> bool:
    return shutil.which("pandoc") is not None


def render_pdf_with_pandoc(md_path: str | Path, pdf_path: str | Path) -> bool:
    md_path, pdf_path = Path(md_path), Path(pdf_path)
    if not _pandoc_available():
        return False
    cmd = [
        "pandoc",
        str(md_path),
        "-o", str(pdf_path),
        "--pdf-engine=pdflatex",
        "-V", "geometry:margin=1in",
        "-V", "fontsize=10pt",
        "--resource-path", str(md_path.parent),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.warning("pandoc failed (%s): %s", result.returncode, result.stderr[-2000:])
            return False
        return pdf_path.exists()
    except Exception as exc:  # noqa: BLE001
        logger.warning("pandoc invocation raised %s", exc)
        return False


def render_pdf_with_reportlab(md_path: str | Path, pdf_path: str | Path) -> bool:
    """Very small Markdown -> PDF fallback covering headings, paragraphs,
    pipe tables, and image references -- enough for this report's structure."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak,
        )
    except ImportError:
        logger.warning("reportlab not installed -- cannot build fallback PDF.")
        return False

    md_path, pdf_path = Path(md_path), Path(pdf_path)
    text = md_path.read_text()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1c", fontSize=18, leading=22, spaceAfter=12, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="H2c", fontSize=14, leading=18, spaceBefore=14, spaceAfter=8, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="H3c", fontSize=12, leading=15, spaceBefore=10, spaceAfter=6, fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="Bodyc", fontSize=9.5, leading=13, spaceAfter=8))

    story = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("# "):
            story.append(Paragraph(stripped[2:], styles["H1c"]))
        elif stripped.startswith("## "):
            story.append(Paragraph(stripped[3:], styles["H2c"]))
        elif stripped.startswith("### "):
            story.append(Paragraph(stripped[4:], styles["H3c"]))
        elif stripped.startswith("!["):
            m = re.search(r"\((.*?)\)", stripped)
            if m:
                img_path = (md_path.parent / m.group(1)).resolve()
                if img_path.exists():
                    try:
                        story.append(Image(str(img_path), width=6.2 * inch, height=3.0 * inch, kind="proportional"))
                        story.append(Spacer(1, 8))
                    except Exception:
                        pass
        elif stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            i -= 1
            rows = [
                [c.strip() for c in row.strip("|").split("|")]
                for row in table_lines
                if not re.match(r"^\|?\s*-+\s*\|", row)
            ]
            if rows:
                t = Table(rows, hAlign="LEFT")
                t.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                            ("FONTSIZE", (0, 0), (-1, -1), 7),
                            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ]
                    )
                )
                story.append(t)
                story.append(Spacer(1, 10))
        elif stripped == "":
            story.append(Spacer(1, 4))
        elif stripped.startswith("---"):
            pass
        else:
            safe = stripped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            safe = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", safe)
            story.append(Paragraph(safe, styles["Bodyc"]))
        i += 1

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=LETTER,
        leftMargin=0.8 * inch, rightMargin=0.8 * inch, topMargin=0.7 * inch, bottomMargin=0.7 * inch,
    )
    doc.build(story)
    return pdf_path.exists()


def build_pdf(md_path: str | Path, pdf_path: str | Path) -> tuple[bool, str]:
    """Try pandoc first, then reportlab. Returns (success, method)."""
    if render_pdf_with_pandoc(md_path, pdf_path):
        return True, "pandoc"
    logger.warning("pandoc/LaTeX unavailable or failed -- trying reportlab fallback.")
    if render_pdf_with_reportlab(md_path, pdf_path):
        return True, "reportlab"
    logger.error(
        "Could not generate a PDF automatically. The Markdown report is still "
        "available at %s. To generate a PDF manually once pandoc + a LaTeX "
        "engine are installed, run:\n  pandoc %s -o %s --pdf-engine=pdflatex",
        md_path, md_path, pdf_path,
    )
    return False, "none"
