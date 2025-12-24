# HyRAG Offline QMS RAG

Fully offline RAG system for QMS-style documents using local Ollama models for embeddings and generation.

## Requirements

- Python 3.10+
- Local Ollama server running on `http://localhost:11434`
- Models already pulled in Ollama:
  - Embeddings: `bge-small-en-v1.5`
  - LLM: configured in `config.yml` (default `qwen2.5:7b`)
- System packages (typical): `build-essential`, `python3-dev`

### Python dependencies

Install dependencies in your environment (example):

```bash
pip install fastapi uvicorn pyyaml requests numpy faiss-cpu pymupdf python-docx
```

> Note: `python-docx` is only needed for DOCX ingestion.

## Configuration

Edit `config.yml` to set:

- `DOCS_FOLDER`: directory with PDF/DOCX/TXT files
- `STORE_DIR`: output directory for indexes/artifacts
- Retrieval parameters (BM25/FAISS) and HyDE settings
- `OLLAMA_BASE_URL`: local Ollama endpoint

## Build the index

```bash
python db_build2.py
```

This parses documents, chunks them, builds BM25 and FAISS indexes, and writes artifacts to `STORE_DIR`.

### Troubleshooting build errors

- **500 Internal Server Error from Ollama**: ensure the embedding model configured in `config.yml` is already pulled in Ollama. Example: `OLLAMA_BASE_URL/api/tags` should list `bge-small-en-v1.5`.
- **Missing model**: update `EMBED_MODEL` in `config.yml` to a model that exists locally, or pull the model in Ollama.
- **Ollama not running**: start Ollama locally before running `db_build2.py`.

## Run the query API

```bash
uvicorn ollama_fastapi:app --host 0.0.0.0 --port 8000
```

## Query example

```bash
curl -s http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is the document control procedure?"}' | jq
```

The response includes a grounded answer with citations, or a refusal if evidence is insufficient.

## Offline enforcement

The system only calls `http://localhost` (Ollama). If `OFFLINE_GUARD` is enabled in `config.yml`, non-local URLs are blocked.
