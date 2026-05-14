from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import re

from .vector_store import LocalVectorStore
from .utils import load_json, normalize_text, simple_extract_keywords

MEDICAL_SYMPTOMS = {
    "fever","cough","cold","headache","nausea","vomiting","diarrhea","fatigue","pain",
    "dizziness","rash","thirst","shortness of breath","wheezing","weakness","blurred vision"
}
MEDICAL_DISEASES = {
    "diabetes","asthma","flu","influenza","dehydration","hypertension","anemia","infection"
}
MEDICAL_TREATMENTS = {
    "rest","fluids","insulin","inhaler","medication","therapy","antibiotics","rehydration"
}

class MedicalQABot:
    def __init__(self, data_path: str | Path, persist_dir: str | Path):
        self.data_path = Path(data_path)
        self.store = LocalVectorStore(persist_dir)
        self.load_data()

    def load_data(self) -> None:
        raw = load_json(self.data_path, default=[])
        docs = []
        for i, item in enumerate(raw, 1):
            docs.append({
                "id": item.get("id", f"med_{i}"),
                "title": item.get("question", f"Medical item {i}"),
                "content": item.get("answer", ""),
                "source": "medquad_sample",
                "metadata": {"question": item.get("question", "")},
            })
        if not docs:
            docs = [{
                "id":"med_fallback_1",
                "title":"What are common symptoms of diabetes?",
                "content":"Common symptoms include increased thirst, frequent urination, fatigue, blurred vision, and slow healing.",
                "source":"fallback"
            }]
        self.store.add_documents(docs)

    def extract_entities(self, query: str) -> dict:
        q = query.lower()
        symptoms = [s for s in MEDICAL_SYMPTOMS if s in q]
        diseases = [d for d in MEDICAL_DISEASES if d in q]
        treatments = [t for t in MEDICAL_TREATMENTS if t in q]
        return {"symptoms": symptoms, "diseases": diseases, "treatments": treatments}

    def answer(self, query: str) -> dict:
        results = self.store.search(query, top_k=3)
        entities = self.extract_entities(query)
        answer = results[0]["content"] if results else "I could not find a direct answer in the medical knowledge base."
        keywords = simple_extract_keywords(query, top_n=6)
        safety = (
            "Medical note: This is informational only and not a diagnosis. "
            "Please consult a licensed clinician for personal medical advice."
        )
        response = (
            f"{answer}\n\n"
            f"Detected medical terms: symptoms={entities['symptoms']}, diseases={entities['diseases']}, treatments={entities['treatments']}.\n"
            f"Query keywords: {keywords}.\n"
            f"{safety}"
        )
        return {"response": response, "sources": results, "entities": entities}
