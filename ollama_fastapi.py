import json
import os
from typing import Any, Dict, List

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from indexing import BM25Index, embed_texts, load_bm25, load_faiss, search_faiss, tokenize
from prompts import final_answer_prompt, hypothetical_answer_prompt
from utils_qms import assert_local_url, load_config, read_jsonl


class QueryRequest(BaseModel):
    query: str


class RAGArtifacts:
    def __init__(self, store_dir: str):
        self.store_dir = store_dir
        self.chunks = read_jsonl(os.path.join(store_dir, "chunks.jsonl"))
        self.bm25 = load_bm25(os.path.join(store_dir, "bm25.json"))
        self.faiss = load_faiss(os.path.join(store_dir, "faiss.index"))


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
) -> tuple[List[int], Dict[int, float]]:
    query_tokens = tokenize(query)
    bm25_scores = artifacts.bm25.score(query_tokens)
    bm25_ranked = sorted(
        [(idx, score) for idx, score in enumerate(bm25_scores)],
        key=lambda item: item[1],
        reverse=True,
    )[: config.BM25_TOPN]

    dense_ranked: List[tuple[int, float]] = []
    if config.USE_DENSE:
        query_emb = embed_texts([query], config.OLLAMA_BASE_URL, config.EMBED_MODEL, config.OFFLINE_GUARD)
        dense_ranked = search_faiss(artifacts.faiss, query_emb, config.DENSE_TOPN)

    fused_scores = rrf_fusion(bm25_ranked, dense_ranked, config.RRF_K)
    ranked_indices = [idx for idx, _ in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)]
    return ranked_indices, fused_scores


def retrieve_auxiliary(artifacts: RAGArtifacts, query: str, config) -> List[int]:
    prompt = hypothetical_answer_prompt(query)
    hypothetical = ollama_generate(prompt, config.LLM_MODEL, config.OLLAMA_BASE_URL, config.OFFLINE_GUARD)
    if not hypothetical:
        return []
    hypo_emb = embed_texts([hypothetical], config.OLLAMA_BASE_URL, config.EMBED_MODEL, config.OFFLINE_GUARD)
    aux_results = search_faiss(artifacts.faiss, hypo_emb, config.HYDE_TOPN)
    return [idx for idx, _ in aux_results]


def check_evidence(scores: Dict[int, float], config) -> bool:
    if not scores:
        return False
    best = max(scores.values())
    return best >= config.MIN_SCORE_THRESHOLD


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

    primary_indices, fused_scores = retrieve_primary(artifacts, query, config)
    if not check_evidence(fused_scores, config):
        return {
            "answer": "Not available in the document.",
            "citations": [],
            "debug": {"reason": "low_confidence"} if config.DEBUG else None,
        }

    auxiliary_indices = retrieve_auxiliary(artifacts, query, config)
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

    citations = build_citations(context_blocks)
    response: Dict[str, Any] = {"answer": answer, "citations": citations}
    if config.DEBUG:
        response["debug"] = {
            "primary_indices": primary_indices[: config.FINAL_TOPK],
            "auxiliary_indices": auxiliary_indices,
            "scores": fused_scores,
        }
    return response
