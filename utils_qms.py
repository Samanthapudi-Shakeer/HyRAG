import json
import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml


@dataclass
class Config:
    DOCS_FOLDER: str
    STORE_DIR: str
    BM25_TOPN: int
    DENSE_TOPN: int
    USE_DENSE: bool
    FINAL_TOPK: int
    HYDE_TOPN: int
    HYDE_WEIGHT: float
    RRF_K: int
    MIN_CHUNK_TOKENS: int
    MAX_CHUNK_TOKENS: int
    MIN_SCORE_THRESHOLD: float
    AUX_CONTEXT_SLOTS: int
    EMBED_MODEL: str
    LLM_MODEL: str
    OLLAMA_BASE_URL: str
    DEBUG: bool
    OFFLINE_GUARD: bool


def load_config(path: str = "config.yml") -> Config:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return Config(**data)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def assert_local_url(url: str) -> None:
    if not url.startswith("http://localhost") and not url.startswith("http://127.0.0.1"):
        raise ValueError(f"Non-local URL blocked: {url}")


def read_jsonl(path: str) -> list[Dict[str, Any]]:
    items: list[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                items.append(json.loads(line))
    return items


def write_jsonl(path: str, items: list[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
