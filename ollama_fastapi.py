import hashlib
import json
import os
from typing import Any, Dict, List

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from indexing import BM25Index, build_bm25, embed_texts, load_bm25, load_faiss, search_faiss, tokenize
from prompts import final_answer_prompt, hypothetical_answer_prompt
from utils_qms import assert_local_url, load_config, read_jsonl, read_pickle


class QueryRequest(BaseModel):
    query: str


class RAGArtifacts:
    def __init__(self, store_dir: str):
        self.store_dir = store_dir
        self.chunks = self._load_chunks(store_dir)
        self.bm25 = self._load_bm25(store_dir, self.chunks)
        self.faiss = self._load_faiss(store_dir)

    @staticmethod
    def _load_chunks(store_dir: str) -> List[Dict[str, Any]]:
        jsonl_path = os.path.join(store_dir, "chunks.jsonl")
        pkl_path = os.path.join(store_dir, "chunks_metadata.pkl")
        if os.path.exists(jsonl_path):
            return RAGArtifacts._normalize_chunks(read_jsonl(jsonl_path))
        if os.path.exists(pkl_path):
            payload = read_pickle(pkl_path)
            if isinstance(payload, dict) and "chunks" in payload:
                payload = payload["chunks"]
            if isinstance(payload, list):
                return RAGArtifacts._normalize_chunks(payload)
            raise RuntimeError(
                "chunks_metadata.pkl did not contain a list payload or chunks field."
            )
        raise FileNotFoundError("No chunk metadata found (chunks.jsonl or chunks_metadata.pkl).")

    @staticmethod
    def _normalize_chunks(raw_chunks: List[Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_chunks):
            if isinstance(item, dict):
                data = dict(item)
            else:
                data = {
                    "chunk_id": getattr(item, "chunk_id", None),
                    "text": getattr(item, "text", None),
                    "source": getattr(item, "source", None),
                    "page": getattr(item, "page", None),
                    "section_path": getattr(item, "section_path", None),
                    "content_type": getattr(item, "content_type", None),
                }

            source = data.get("source") or "unknown"
            page = data.get("page") or 1
            section_path = data.get("section_path") or ""
            text = data.get("text") or ""
            content_type = data.get("content_type") or "paragraph"
            chunk_id = data.get("chunk_id")
            if not chunk_id:
                seed = f"{source}-{section_path}-{page}-{idx}"
                chunk_id = hashlib.sha1(seed.encode("utf-8")).hexdigest()

            normalized.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "source": source,
                    "page": page,
                    "section_path": section_path,
                    "content_type": content_type,
                }
            )
        return normalized

    @staticmethod
    def _load_bm25(store_dir: str, chunks: List[Dict[str, Any]]) -> BM25Index:
        bm25_path = os.path.join(store_dir, "bm25.json")
        if os.path.exists(bm25_path):
            return load_bm25(bm25_path)
        texts = [chunk.get("text", "") for chunk in chunks]
        return build_bm25(texts)

    @staticmethod
    def _load_faiss(store_dir: str):
        faiss_path = os.path.join(store_dir, "faiss.index")
        legacy_path = os.path.join(store_dir, "faiss_index.bin")
        if os.path.exists(faiss_path):
            return load_faiss(faiss_path)
        if os.path.exists(legacy_path):
            return load_faiss(legacy_path)
        raise FileNotFoundError("No FAISS index found (faiss.index or faiss_index.bin).")


def ollama_generate(prompt: str, model: str, base_url: str, offline_guard: bool) -> str:
    if offline_guard:
        assert_local_url(base_url)
    response = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


def rrf_fusion(bm25_results: List[tuple[int, float]], dense_results: List[tuple[int, float]], k: int) -> Dict[int, float]:
    fused: Dict[int, float] = {}
    for rank, (idx, _) in enumerate(bm25_results):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, (idx, _) in enumerate(dense_results):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


def build_context(
    artifacts: RAGArtifacts,
    primary_indices: List[int],
    auxiliary_indices: List[int],
    topk: int,
    aux_slots: int,
) -> List[Dict[str, Any]]:
    context_blocks: List[Dict[str, Any]] = []
    seen = set()

    primary_needed = max(topk - aux_slots, 0)
    for idx in primary_indices:
        if len(context_blocks) >= primary_needed:
            break
        if idx in seen:
            continue
        chunk = artifacts.chunks[idx]
        context_blocks.append(chunk)
        seen.add(idx)

    for idx in auxiliary_indices:
        if len(context_blocks) >= topk:
            break
        if idx in seen:
            continue
        chunk = artifacts.chunks[idx]
        context_blocks.append(chunk)
        seen.add(idx)

    context_blocks.sort(key=lambda c: (c.get("section_path", ""), c.get("page", 0)))
    return context_blocks


def build_citations(context_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    citations = []
    for idx, block in enumerate(context_blocks):
        citations.append(
            {
                "citation": f"C{idx + 1}",
                "source": block.get("source"),
                "page": block.get("page"),
                "section_path": block.get("section_path"),
            }
        )
    return citations


def concise_hypothetical(text: str, max_sentences: int = 2, max_chars: int = 400) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    sentences = []
    start = 0
    for idx, char in enumerate(cleaned):
        if char in ".!?":
            sentence = cleaned[start : idx + 1].strip()
            if sentence:
                sentences.append(sentence)
            start = idx + 1
        if len(sentences) >= max_sentences:
            break
    if not sentences:
        sentences = [cleaned]
    concise = " ".join(sentences).strip()
    if len(concise) > max_chars:
        concise = concise[: max_chars - 1].rstrip() + "…"
    return concise


def chunk_to_prompt_blocks(context_blocks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    prompt_blocks: List[Dict[str, str]] = []
    for idx, block in enumerate(context_blocks):
        citation = f"{block.get('source')} p.{block.get('page')} | {block.get('section_path')}"
        prompt_blocks.append({"citation": citation, "text": block.get("text", "")})
    return prompt_blocks


def retrieve_primary(
    artifacts: RAGArtifacts,
    query: str,
    config,
) -> tuple[List[int], Dict[int, float], float]:
    query_tokens = tokenize(query)
    bm25_scores = artifacts.bm25.score(query_tokens)
    max_bm25 = max(bm25_scores) if bm25_scores else 0.0
    bm25_ranked = sorted(
        [(idx, score) for idx, score in enumerate(bm25_scores)],
        key=lambda item: item[1],
        reverse=True,
    )[: config.BM25_TOPN]

    dense_ranked: List[tuple[int, float]] = []
    if config.USE_DENSE:
        query_emb = embed_texts([query], config.OLLAMA_BASE_URL, config.EMBED_MODEL, config.OFFLINE_GUARD)
        if artifacts.faiss.d != query_emb.shape[1]:
            dense_ranked = []
        else:
            dense_ranked = search_faiss(artifacts.faiss, query_emb, config.DENSE_TOPN)

    fused_scores = rrf_fusion(bm25_ranked, dense_ranked, config.RRF_K)
    ranked_indices = [idx for idx, _ in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)]
    return ranked_indices, fused_scores, max_bm25


def retrieve_auxiliary(artifacts: RAGArtifacts, query: str, config) -> tuple[List[int], str]:
    prompt = hypothetical_answer_prompt(query)
    hypothetical = ollama_generate(prompt, config.LLM_MODEL, config.OLLAMA_BASE_URL, config.OFFLINE_GUARD)
    if not hypothetical:
        return [], ""
    hypo_emb = embed_texts([hypothetical], config.OLLAMA_BASE_URL, config.EMBED_MODEL, config.OFFLINE_GUARD)
    if artifacts.faiss.d != hypo_emb.shape[1]:
        return [], hypothetical
    aux_results = search_faiss(artifacts.faiss, hypo_emb, config.HYDE_TOPN)
    return [idx for idx, _ in aux_results], hypothetical


def check_evidence(scores: Dict[int, float], max_bm25: float, config) -> bool:
    if not scores:
        return False
    best = max(scores.values())
    if best >= config.MIN_SCORE_THRESHOLD:
        return True
    return max_bm25 > 0.0


config = load_config()
app = FastAPI()
artifacts: RAGArtifacts | None = None


@app.on_event("startup")
def load_artifacts() -> None:
    global artifacts
    if not os.path.isdir(config.STORE_DIR):
        raise RuntimeError(f"STORE_DIR missing: {config.STORE_DIR}")
    artifacts = RAGArtifacts(config.STORE_DIR)


@app.post("/query")
def query_endpoint(request: QueryRequest) -> Dict[str, Any]:
    if artifacts is None:
        raise HTTPException(status_code=500, detail="Artifacts not loaded")
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    primary_indices, fused_scores, max_bm25 = retrieve_primary(artifacts, query, config)
    if not primary_indices or not check_evidence(fused_scores, max_bm25, config):
        _, hypothetical = retrieve_auxiliary(artifacts, query, config)
        return {
            "answer": concise_hypothetical(hypothetical) or "Not available in the document.",
            "citations": [],
            "hypothetical_answer": hypothetical,
            "matched_sources": {
                "primary": [],
                "auxiliary": [],
            },
            "debug": {"reason": "low_confidence"} if config.DEBUG else None,
        }

    auxiliary_indices, hypothetical = retrieve_auxiliary(artifacts, query, config)
    context_blocks = build_context(
        artifacts,
        primary_indices,
        auxiliary_indices,
        topk=config.FINAL_TOPK,
        aux_slots=config.AUX_CONTEXT_SLOTS,
    )

    prompt_blocks = chunk_to_prompt_blocks(context_blocks)
    prompt = final_answer_prompt(query, prompt_blocks)
    answer = ollama_generate(prompt, config.LLM_MODEL, config.OLLAMA_BASE_URL, config.OFFLINE_GUARD)
    if not answer:
        answer = concise_hypothetical(hypothetical) or "Not available in the document."

    citations = build_citations(context_blocks)
    def _summarize(indices: List[int], source_type: str) -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        for idx in indices:
            chunk = artifacts.chunks[idx]
            summary.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "source": chunk.get("source"),
                    "page": chunk.get("page"),
                    "section_path": chunk.get("section_path"),
                    "source_type": source_type,
                }
            )
        return summary

    response: Dict[str, Any] = {
        "answer": answer,
        "citations": citations,
        "hypothetical_answer": hypothetical,
        "matched_sources": {
            "primary": _summarize(primary_indices[: config.FINAL_TOPK], "docstore"),
            "auxiliary": _summarize(auxiliary_indices, "hyde"),
        },
    }
    if config.DEBUG:
        response["debug"] = {
            "primary_indices": primary_indices[: config.FINAL_TOPK],
            "auxiliary_indices": auxiliary_indices,
            "scores": fused_scores,
        }
    return response
