import json
import math
import os
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import requests

from utils_qms import assert_local_url


@dataclass
class BM25Index:
    corpus_tokens: list[list[str]]
    idf: dict[str, float]
    avgdl: float

    def score(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0 for _ in range(len(self.corpus_tokens))]
        for idx, doc_tokens in enumerate(self.corpus_tokens):
            doc_len = len(doc_tokens)
            denom = doc_len / self.avgdl if self.avgdl else 0.0
            for token in query_tokens:
                tf = doc_tokens.count(token)
                if tf == 0:
                    continue
                idf = self.idf.get(token, 0.0)
                numer = tf * 2.2
                scores[idx] += idf * (numer / (tf + 1.2 * (1.0 + 0.75 * denom)))
        return scores


def tokenize(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


def build_bm25(texts: Iterable[str]) -> BM25Index:
    corpus_tokens = [tokenize(text) for text in texts]
    doc_count = len(corpus_tokens)
    avgdl = sum(len(doc) for doc in corpus_tokens) / doc_count if doc_count else 0.0
    doc_freq: dict[str, int] = {}
    for doc in corpus_tokens:
        for token in set(doc):
            doc_freq[token] = doc_freq.get(token, 0) + 1
    idf = {
        token: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
        for token, freq in doc_freq.items()
    }
    return BM25Index(corpus_tokens=corpus_tokens, idf=idf, avgdl=avgdl)


def save_bm25(index: BM25Index, path: str) -> None:
    payload = {
        "corpus_tokens": index.corpus_tokens,
        "idf": index.idf,
        "avgdl": index.avgdl,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def load_bm25(path: str) -> BM25Index:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return BM25Index(
        corpus_tokens=payload["corpus_tokens"],
        idf=payload["idf"],
        avgdl=payload["avgdl"],
    )


def embed_texts(texts: List[str], base_url: str, model: str, offline_guard: bool) -> np.ndarray:
    if offline_guard:
        assert_local_url(base_url)
    embeddings: List[List[float]] = []
    for text in texts:
        response = requests.post(
            f"{base_url}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        embeddings.append(data["embedding"])
    return np.array(embeddings, dtype="float32")


def build_faiss(embeddings: np.ndarray):
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def save_faiss(index, path: str) -> None:
    import faiss

    faiss.write_index(index, path)


def load_faiss(path: str):
    import faiss

    return faiss.read_index(path)


def search_faiss(index, query_embedding: np.ndarray, topn: int) -> list[tuple[int, float]]:
    import faiss

    query = query_embedding.astype("float32")
    faiss.normalize_L2(query)
    scores, indices = index.search(query, topn)
    results: list[tuple[int, float]] = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        results.append((int(idx), float(score)))
    return results


def ensure_store_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
