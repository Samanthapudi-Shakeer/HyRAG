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


def build_section_paths(elements: Iterable[DocElement]) -> List[tuple[DocElement, str]]:
    stack: List[str] = []
    mapped: List[tuple[DocElement, str]] = []
    for element in elements:
        if element.is_heading and element.heading_level:
            level = min(element.heading_level, 6)
            while len(stack) >= level:
                stack.pop()
            stack.append(element.text.strip())
            mapped.append((element, " > ".join(stack)))
        else:
            mapped.append((element, " > ".join(stack)))
    return mapped


def _token_count(text: str) -> int:
    return len(text.split())


def _split_by_tokens(text: str, max_tokens: int) -> List[str]:
    if _token_count(text) <= max_tokens:
        return [text]
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    for part in parts:
        if _token_count(" ".join(current + [part])) <= max_tokens:
            current.append(part)
        else:
            if current:
                chunks.append("\n\n".join(current))
            current = [part]
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def hybrid_chunk(
    source: str,
    elements: Iterable[DocElement],
    min_tokens: int,
    max_tokens: int,
) -> List[Chunk]:
    mapped = build_section_paths(elements)
    section_groups: List[dict] = []
    current_group: dict | None = None
    for element, section_path in mapped:
        if element.is_heading:
            current_group = {
                "section_path": section_path,
                "texts": [element.text],
                "pages": [element.page],
                "content_type": "heading",
            }
            section_groups.append(current_group)
            continue
        if current_group is None:
            current_group = {
                "section_path": section_path,
                "texts": [],
                "pages": [],
                "content_type": element.content_type,
            }
            section_groups.append(current_group)
        current_group["texts"].append(element.text)
        current_group["pages"].append(element.page)

    initial_chunks: List[dict] = []
    for group in section_groups:
        text = "\n".join(group["texts"]).strip()
        if not text:
            continue
        split_texts = _split_by_tokens(text, max_tokens)
        for part in split_texts:
            initial_chunks.append(
                {
                    "text": part,
                    "section_path": group["section_path"],
                    "page": group["pages"][0] if group["pages"] else 1,
                    "content_type": group["content_type"],
                }
            )

    merged: List[dict] = []
    buffer: dict | None = None
    for chunk in initial_chunks:
        token_count = _token_count(chunk["text"])
        if token_count < min_tokens and buffer is not None:
            buffer["text"] = f"{buffer['text']}\n\n{chunk['text']}"
            buffer["page"] = min(buffer["page"], chunk["page"])
            continue
        if buffer is not None:
            merged.append(buffer)
        buffer = chunk
    if buffer is not None:
        merged.append(buffer)

    chunks: List[Chunk] = []
    for idx, chunk in enumerate(merged):
        seed = f"{source}-{chunk['section_path']}-{chunk['page']}-{idx}"
        chunk_id = hashlib.sha1(seed.encode("utf-8")).hexdigest()
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=chunk["text"],
                source=source,
                page=chunk["page"],
                section_path=chunk["section_path"],
                content_type=chunk["content_type"],
            )
        )
    return chunks
