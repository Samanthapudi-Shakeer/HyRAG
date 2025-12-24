import hashlib
from dataclasses import dataclass
from typing import Iterable, List

from loaders import DocElement


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    page: int
    section_path: str
    content_type: str


def elements_to_markdown(elements: Iterable[DocElement]) -> str:
    lines: List[str] = []
    for element in elements:
        if element.is_heading and element.heading_level:
            prefix = "#" * min(element.heading_level, 6)
            lines.append(f"{prefix} {element.text}")
        else:
            if element.content_type == "list" and not element.text.startswith(("- ", "* ")):
                lines.append(f"- {element.text}")
            else:
                lines.append(element.text)
    return "\n".join(lines)


def _docling_chunk(markdown: str, min_tokens: int, max_tokens: int):
    try:
        from docling.chunking import HybridChunker
        from docling.datamodel.document import DoclingDocument
    except ImportError as exc:
        raise RuntimeError(
            "Docling is required for hybrid chunking. Install docling and retry."
        ) from exc

    if hasattr(DoclingDocument, "from_markdown"):
        document = DoclingDocument.from_markdown(markdown)
    else:
        document = DoclingDocument(markdown)

    chunker = HybridChunker(min_tokens=min_tokens, max_tokens=max_tokens)
    return chunker.chunk(document)


def hybrid_chunk(
    source: str,
    elements: Iterable[DocElement],
    min_tokens: int,
    max_tokens: int,
) -> List[Chunk]:
    markdown = elements_to_markdown(elements)
    docling_chunks = _docling_chunk(markdown, min_tokens, max_tokens)

    chunks: List[Chunk] = []
    for idx, doc_chunk in enumerate(docling_chunks):
        metadata = getattr(doc_chunk, "metadata", {}) or {}
        text = getattr(doc_chunk, "text", None) or getattr(doc_chunk, "content", "")
        section_path = (
            getattr(doc_chunk, "section_path", "")
            or metadata.get("section_path")
            or metadata.get("headings", "")
            or ""
        )
        page = metadata.get("page") or metadata.get("page_number") or 1
        content_type = metadata.get("content_type") or "paragraph"
        seed = f"{source}-{section_path}-{page}-{idx}"
        chunk_id = hashlib.sha1(seed.encode("utf-8")).hexdigest()
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=str(text).strip(),
                source=source,
                page=int(page),
                section_path=str(section_path),
                content_type=str(content_type),
            )
        )
    return chunks
