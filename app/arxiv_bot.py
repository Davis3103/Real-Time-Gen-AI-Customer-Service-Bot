from __future__ import annotations
from pathlib import Path
from typing import List
import re
from collections import Counter

from .vector_store import LocalVectorStore
from .utils import load_json, simple_extract_keywords, normalize_text

def extractive_summary(text: str, max_sentences: int = 3) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= max_sentences:
        return text
    words = re.findall(r"[A-Za-z]{3,}", text.lower())
    freq = Counter(words)
    scored = []
    for idx, sent in enumerate(sentences):
        score = sum(freq.get(w, 0) for w in re.findall(r"[A-Za-z]{3,}", sent.lower()))
        scored.append((idx, score, sent))
    top = sorted(scored, key=lambda x: (-x[1], x[0]))[:max_sentences]
    top = [s for _, _, s in sorted(top, key=lambda x: x[0])]
    return " ".join(top)

class ResearchExpertBot:
    def __init__(self, data_path: str | Path, persist_dir: str | Path):
        self.data_path = Path(data_path)
        self.store = LocalVectorStore(persist_dir)
        self.load_data()

    def load_data(self) -> None:
        raw = load_json(self.data_path, default=[])
        docs = []
        for i, item in enumerate(raw, 1):
            docs.append({
                "id": item.get("id", f"arxiv_{i}"),
                "title": item.get("title", f"Paper {i}"),
                "content": item.get("abstract", ""),
                "source": "arxiv_sample",
                "metadata": {"title": item.get("title", "")},
            })
        if not docs:
            docs = [{
                "id":"arxiv_fallback_1",
                "title":"Generative AI for Customer Service",
                "content":"Generative AI can automate support, summarize tickets, and improve response personalization in service systems."
            }]
        self.store.add_documents(docs)

    def answer(self, query: str) -> dict:
        results = self.store.search(query, top_k=3)
        if not results:
            return {"response": "No relevant paper found.", "sources": []}
        top = results[0]
        summary = extractive_summary(top["content"], max_sentences=2)
        keywords = simple_extract_keywords(query + " " + top["content"])
        response = (
            f"Paper: {top['title']}\n"
            f"Summary: {summary}\n"
            f"Concept keywords: {', '.join(keywords)}\n"
            f"Follow-up hint: ask about methodology, results, or limitations."
        )
        return {"response": response, "sources": results}
