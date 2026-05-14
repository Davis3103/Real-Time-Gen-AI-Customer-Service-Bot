from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import json

from .vector_store import LocalVectorStore
from .utils import load_json, save_json, normalize_text

def load_documents_from_sources(source_paths: List[str | Path]) -> List[dict]:
    docs = []
    for path in source_paths:
        p = Path(path)
        if not p.exists():
            continue
        if p.suffix.lower() in {".json"}:
            data = load_json(p, default=[])
            for item in data:
                if isinstance(item, dict):
                    docs.append({
                        "id": item.get("id") or item.get("title") or p.stem,
                        "title": item.get("title") or item.get("question") or p.stem,
                        "content": item.get("content") or item.get("answer") or item.get("abstract") or "",
                        "source": str(p),
                        "metadata": item.get("metadata") or {},
                    })
        else:
            text = p.read_text(encoding="utf-8", errors="ignore")
            docs.append({
                "id": p.stem,
                "title": p.stem.replace("_", " ").title(),
                "content": text,
                "source": str(p),
                "metadata": {},
            })
    return docs

def update_knowledge_base(store: LocalVectorStore, source_paths: List[str | Path]) -> dict:
    new_docs = load_documents_from_sources(source_paths)
    if not new_docs:
        return {"added": 0, "status": "no_sources_found"}
    existing = {(d.id, d.title, d.content) for d in store.docs}
    added = 0
    for doc in new_docs:
        key = (str(doc.get("id")), str(doc.get("title")), str(doc.get("content")))
        if key not in existing:
            store.add_documents([doc], rebuild=False)
            added += 1
    store.build_index()
    store.save()
    return {"added": added, "status": "updated"}

def periodic_update_note() -> str:
    return (
        "To make the knowledge base update periodically, call update_knowledge_base() "
        "from a scheduled script, cron job, or cloud scheduler."
    )
