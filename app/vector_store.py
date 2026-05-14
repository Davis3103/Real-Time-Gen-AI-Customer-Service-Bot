from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pickle
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .utils import normalize_text, load_json, save_json

@dataclass
class StoredDocument:
    id: str
    title: str
    content: str
    source: str = "local"
    metadata: dict | None = None

class LocalVectorStore:
    def __init__(self, persist_path: str | Path):
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self.docs: List[StoredDocument] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None

    @property
    def state_file(self) -> Path:
        return self.persist_path / "store.pkl"

    def add_documents(self, docs: List[dict], rebuild: bool = True) -> None:
        for doc in docs:
            self.docs.append(
                StoredDocument(
                    id=str(doc.get("id") or len(self.docs) + 1),
                    title=doc.get("title", f"Doc {len(self.docs) + 1}"),
                    content=normalize_text(doc.get("content") or doc.get("answer") or doc.get("abstract") or ""),
                    source=doc.get("source", "local"),
                    metadata=doc.get("metadata") or {},
                )
            )
        if rebuild:
            self.build_index()

    def build_index(self) -> None:
        corpus = [f"{d.title}. {d.content}" for d in self.docs]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(corpus) if corpus else None

    def search(self, query: str, top_k: int = 3) -> List[dict]:
        if not self.docs:
            return []
        if self.vectorizer is None or self.matrix is None:
            self.build_index()
        if self.vectorizer is None or self.matrix is None:
            return []
        qv = self.vectorizer.transform([normalize_text(query)])
        scores = cosine_similarity(qv, self.matrix)[0]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked:
            doc = self.docs[idx]
            results.append({
                "id": doc.id,
                "title": doc.title,
                "content": doc.content,
                "source": doc.source,
                "metadata": doc.metadata or {},
                "score": float(score),
            })
        return results

    def save(self) -> None:
        payload = {
            "docs": [asdict(d) for d in self.docs],
            "vectorizer": self.vectorizer,
            "matrix": self.matrix,
        }
        with self.state_file.open("wb") as f:
            pickle.dump(payload, f)

    def load(self) -> bool:
        if not self.state_file.exists():
            return False
        with self.state_file.open("rb") as f:
            payload = pickle.load(f)
        self.docs = [StoredDocument(**d) for d in payload.get("docs", [])]
        self.vectorizer = payload.get("vectorizer")
        self.matrix = payload.get("matrix")
        return True
