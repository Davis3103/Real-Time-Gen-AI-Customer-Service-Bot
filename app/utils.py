from __future__ import annotations
import json, os, re, math
from pathlib import Path
from typing import Any, Dict, List, Optional

def load_json(path: str | Path, default=None):
    path = Path(path)
    if not path.exists():
        return default if default is not None else []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def normalize_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text).strip()
    return text

def keyword_score(text: str, keywords: List[str]) -> int:
    t = text.lower()
    return sum(1 for kw in keywords if kw.lower() in t)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(text), step):
        chunks.append(text[start:start+chunk_size])
        if start + chunk_size >= len(text):
            break
    return chunks

def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")

def format_sources(items: List[dict]) -> str:
    lines = []
    for i, item in enumerate(items, 1):
        title = item.get("title") or item.get("id") or f"doc_{i}"
        lines.append(f"{i}. {title}")
    return "\n".join(lines)

def simple_extract_keywords(text: str, top_n: int = 8) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]+", text.lower())
    stop = {
        "the","and","for","with","that","this","from","are","was","were","your","you",
        "into","can","will","about","what","when","how","why","who","which","have","has",
        "had","not","but","use","used","using","also","more","than","over","under","into"
    }
    freq = {}
    for w in words:
        if w in stop or len(w) < 3:
            continue
        freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:top_n]]

def get_env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}
