"""Format-specific extraction backends for non-OpenRouter paths."""

from __future__ import annotations

import re
from pathlib import Path

from coarse.types import ExtractionError


def _extract_docling(path: Path) -> str:
    """Extract via Docling (free, offline). Supports PDF, DOCX, HTML, LaTeX."""
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(path))
    return result.document.export_to_markdown(page_break_placeholder="<!-- PAGE BREAK -->")


def _extract_plaintext(path: Path) -> str:
    """Read a plain text or markdown file as-is."""
    return path.read_text(encoding="utf-8")


_LATEX_HEADING_RE = re.compile(r"\\(section|subsection|subsubsection|paragraph)\*?\{([^}]*)\}")
_LATEX_HEADING_LEVEL = {
    "section": "#",
    "subsection": "##",
    "subsubsection": "###",
    "paragraph": "####",
}
_LATEX_PREAMBLE_RE = re.compile(
    r"^\\(documentclass|usepackage|title|author|date|maketitle"
    r"|begin\{document\}|end\{document\})\b.*$",
    re.MULTILINE,
)
_LATEX_INPUT_RE = re.compile(r"\\(?:input|include)\s*\{\s*([^}]+?)\s*\}")
_MAX_LATEX_INPUT_DEPTH = 25


def _latex_directive_active(line_prefix: str) -> bool:
    """False when an unescaped % earlier on the line comments the directive out."""
    escaped = False
    for ch in line_prefix:
        if escaped:
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == "%":
            return False
    return True


def _resolve_latex_input(name: str, search_dirs: tuple[Path, ...]) -> Path | None:
    """Resolve an \\input / \\include target against the given search directories.

    LaTeX resolves these paths relative to the *main* document's directory, but
    the ``import`` package and some layouts make them relative to the including
    file. We try both, with and without an added ``.tex`` suffix.
    """
    name = name.strip()
    for base_dir in search_dirs:
        for candidate in (base_dir / name, base_dir / f"{name}.tex"):
            if candidate.is_file():
                return candidate
    return None


def _read_latex_with_inputs(
    path: Path, root_dir: Path | None = None, seen: frozenset[Path] | None = None, depth: int = 0
) -> str:
    """Read LaTeX source, recursively inlining \\input / \\include directives.

    Targets resolve against both the including file's directory and the main
    document's directory (``root_dir``), with a missing ``.tex`` suffix added.
    Commented-out directives are left alone; self-referential cycles and runaway
    depth return empty so extraction of a multi-file paper never hangs.
    """
    root_dir = path.parent if root_dir is None else root_dir
    seen = frozenset() if seen is None else seen
    resolved = path.resolve()
    if depth > _MAX_LATEX_INPUT_DEPTH or resolved in seen:
        return ""
    seen = seen | {resolved}
    text = path.read_text(encoding="utf-8")
    search_dirs = (path.parent, root_dir) if path.parent != root_dir else (root_dir,)

    def _inline(match: re.Match[str]) -> str:
        line_start = text.rfind("\n", 0, match.start()) + 1
        if not _latex_directive_active(text[line_start : match.start()]):
            return match.group(0)
        target = _resolve_latex_input(match.group(1), search_dirs)
        if target is None:
            return match.group(0)
        return _read_latex_with_inputs(target, root_dir, seen, depth + 1)

    return _LATEX_INPUT_RE.sub(_inline, text)


def _extract_latex_regex(path: Path) -> str:
    """Extract from LaTeX source with heading conversion to markdown.

    ``\\input`` / ``\\include`` directives are inlined recursively, so a
    multi-file paper (a main file that pulls in per-section sources) extracts
    in full instead of collapsing to a handful of bare directive lines.
    """
    text = _read_latex_with_inputs(path)
    text = _LATEX_PREAMBLE_RE.sub("", text)
    text = _LATEX_HEADING_RE.sub(lambda m: f"{_LATEX_HEADING_LEVEL[m.group(1)]} {m.group(2)}", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_html_markdownify(path: Path) -> str:
    """Convert HTML to markdown via markdownify (lightweight fallback)."""
    try:
        import markdownify
    except ImportError:
        raise ExtractionError(
            "HTML extraction requires markdownify: pip install coarse-ink[formats]"
        )
    html_str = path.read_text(encoding="utf-8")
    return markdownify.markdownify(html_str, heading_style="ATX")


def _extract_docx_mammoth(path: Path) -> str:
    """Convert DOCX to markdown via mammoth (lightweight fallback)."""
    try:
        import mammoth
    except ImportError:
        raise ExtractionError("DOCX extraction requires mammoth: pip install coarse-ink[formats]")
    with open(path, "rb") as f:
        result = mammoth.convert_to_markdown(f)
    return result.value


def _extract_epub(path: Path) -> str:
    """Extract EPUB chapters to markdown via ebooklib + markdownify."""
    try:
        import ebooklib
        import markdownify
        from ebooklib import epub
    except ImportError:
        raise ExtractionError(
            "EPUB extraction requires ebooklib and markdownify: pip install coarse-ink[formats]"
        )
    book = epub.read_epub(str(path))
    chapters = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        html_content = item.get_content().decode("utf-8", errors="replace")
        md = markdownify.markdownify(html_content, heading_style="ATX")
        md = md.strip()
        if md:
            chapters.append(md)
    if not chapters:
        raise ExtractionError(f"No text content found in EPUB: {path}")
    return "\n\n---\n\n".join(chapters)
