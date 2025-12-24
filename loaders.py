import os
from dataclasses import dataclass
from typing import Iterable, List

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover - environment-specific
    raise RuntimeError("PyMuPDF (fitz) is required for PDF parsing.") from exc


@dataclass
class DocElement:
    text: str
    page: int
    content_type: str
    is_heading: bool = False
    heading_level: int | None = None


def _classify_line(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "paragraph"
    if stripped.startswith(""):
        return "table"
    if stripped[:2].isdigit() and stripped[2:3] in {")",
        ".",
    }:
        return "list"
    if stripped.startswith(("- ", "* ", " ", " ", " ", " ", " ", " ", " ", " ", " ")):
        return "list"
    return "paragraph"


def _heading_level(font_size: float, median: float) -> int | None:
    if font_size >= median + 4:
        return 1
    if font_size >= median + 2:
        return 2
    if font_size >= median + 1:
        return 3
    return None


def parse_pdf(path: str) -> List[DocElement]:
    doc = fitz.open(path)
    elements: List[DocElement] = []
    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        text_dict = page.get_text("dict")
        sizes = [
            span.get("size", 0)
            for block in text_dict.get("blocks", [])
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        ]
        median_size = sorted(sizes)[len(sizes) // 2] if sizes else 10
        for block in text_dict.get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                line_text = "".join(span.get("text", "") for span in spans).strip()
                if not line_text:
                    continue
                avg_size = (
                    sum(span.get("size", 0) for span in spans) / len(spans)
                    if spans
                    else median_size
                )
                heading_level = _heading_level(avg_size, median_size)
                content_type = _classify_line(line_text)
                elements.append(
                    DocElement(
                        text=line_text,
                        page=page_index + 1,
                        content_type=content_type,
                        is_heading=heading_level is not None,
                        heading_level=heading_level,
                    )
                )
    return elements


def parse_txt(path: str) -> List[DocElement]:
    elements: List[DocElement] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            content_type = _classify_line(text)
            elements.append(DocElement(text=text, page=1, content_type=content_type))
    return elements


def parse_docx(path: str) -> List[DocElement]:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("python-docx is required for DOCX parsing.") from exc
    doc = docx.Document(path)
    elements: List[DocElement] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        is_heading = para.style.name.lower().startswith("heading")
        heading_level = None
        if is_heading:
            digits = "".join(ch for ch in para.style.name if ch.isdigit())
            heading_level = int(digits) if digits else 2
        content_type = _classify_line(text)
        elements.append(
            DocElement(
                text=text,
                page=1,
                content_type=content_type,
                is_heading=is_heading,
                heading_level=heading_level,
            )
        )
    return elements


def load_document(path: str) -> List[DocElement]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return parse_pdf(path)
    if ext == ".txt":
        return parse_txt(path)
    if ext == ".docx":
        return parse_docx(path)
    raise ValueError(f"Unsupported file type: {ext}")


def load_folder(folder: str) -> Iterable[tuple[str, List[DocElement]]]:
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isdir(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in {".pdf", ".txt", ".docx"}:
            continue
        yield name, load_document(path)
