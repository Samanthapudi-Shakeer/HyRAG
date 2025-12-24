import json
import os
import sys
from typing import List

from chunking import Chunk, elements_to_markdown, hybrid_chunk
from indexing import build_bm25, build_faiss, embed_texts, ensure_store_dir, save_bm25, save_faiss
from loaders import load_folder
from utils_qms import ensure_dir, load_config, write_jsonl


def ingest_folder(docs_folder: str, store_dir: str) -> List[Chunk]:
    ensure_dir(store_dir)
    if not os.path.isdir(docs_folder):
        raise FileNotFoundError(f"Documents folder not found: {docs_folder}")
    chunks: List[Chunk] = []
    doc_found = False
    for filename, elements in load_folder(docs_folder):
        doc_found = True
        markdown = elements_to_markdown(elements)
        _ = markdown  # placeholder for downstream validation if needed
        chunks.extend(
            hybrid_chunk(
                source=filename,
                elements=elements,
                min_tokens=CONFIG.MIN_CHUNK_TOKENS,
                max_tokens=CONFIG.MAX_CHUNK_TOKENS,
            )
        )
    if not doc_found:
        raise ValueError("No supported documents found in DOCS_FOLDER.")
    return chunks


def build_indexes(chunks: List[Chunk], store_dir: str) -> None:
    texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(texts, CONFIG.OLLAMA_BASE_URL, CONFIG.EMBED_MODEL, CONFIG.OFFLINE_GUARD)
    faiss_index = build_faiss(embeddings)
    bm25_index = build_bm25(texts)

    save_faiss(faiss_index, os.path.join(store_dir, "faiss.index"))
    save_bm25(bm25_index, os.path.join(store_dir, "bm25.json"))

    metadata = [
        {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "source": chunk.source,
            "page": chunk.page,
            "section_path": chunk.section_path,
            "content_type": chunk.content_type,
        }
        for chunk in chunks
    ]
    write_jsonl(os.path.join(store_dir, "chunks.jsonl"), metadata)


def main() -> None:
    print("Loading configuration...")
    global CONFIG
    CONFIG = load_config()
    print(f"Ingesting documents from {CONFIG.DOCS_FOLDER}...")
    chunks = ingest_folder(CONFIG.DOCS_FOLDER, CONFIG.STORE_DIR)
    print(f"Generated {len(chunks)} chunks. Building indexes...")
    build_indexes(chunks, CONFIG.STORE_DIR)
    print(f"Artifacts written to {CONFIG.STORE_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Build failed: {exc}", file=sys.stderr)
        sys.exit(1)
