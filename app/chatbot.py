from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any

from .config import GEMINI_API_KEY, PRIMARY_LANGUAGE, KNOWLEDGE_BASE_PATH, PERSIST_DIR, MEDICAL_SAMPLE_PATH, ARXIV_SAMPLE_PATH
from .vector_store import LocalVectorStore
from .utils import load_json, normalize_text, format_sources
from .sentiment import detect_sentiment, tone_prefix
from .multilingual import detect_language, respond_in_language, translate_text
from .multimodal import analyze_image, build_image_generation_prompt
from .dynamic_update import update_knowledge_base
from .medical_bot import MedicalQABot
from .arxiv_bot import ResearchExpertBot

class CustomerServiceBot:
    def __init__(self):
        self.customer_store = LocalVectorStore(PERSIST_DIR / "customer")
        self.customer_store.load()
        if not self.customer_store.docs:
            self._bootstrap_customer_store()
        self.medical_bot = MedicalQABot(MEDICAL_SAMPLE_PATH, PERSIST_DIR / "medical")
        self.research_bot = ResearchExpertBot(ARXIV_SAMPLE_PATH, PERSIST_DIR / "research")

    def _bootstrap_customer_store(self):
        data = load_json(KNOWLEDGE_BASE_PATH, default=[])
        if data:
            self.customer_store.add_documents(data)
            self.customer_store.save()

    def refresh_knowledge_base(self, sources: list[str | Path] | None = None) -> dict:
        sources = sources or [KNOWLEDGE_BASE_PATH]
        return update_knowledge_base(self.customer_store, sources)

    def _genai_response(self, prompt: str) -> str | None:
        if not GEMINI_API_KEY:
            return None
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(prompt)
            return resp.text
        except Exception:
            return None

    def respond(self, message: str, mode: str = "customer", image_path: str | None = None, user_language: str | None = None) -> dict:
        user_language = user_language or detect_language(message or "")
        sentiment = detect_sentiment(message)
        prefix = tone_prefix(sentiment["label"])

        if mode == "medical":
            result = self.medical_bot.answer(message)
            response = result["response"]
        elif mode == "research":
            result = self.research_bot.answer(message)
            response = result["response"]
        elif mode == "multimodal":
            if image_path:
                image_info = analyze_image(image_path, prompt=message or "Describe the image")
                response = image_info.get("response", "")
            else:
                response = "Please upload an image for multimodal analysis."
        else:
            docs = self.customer_store.search(message, top_k=3)
            context = "\n\n".join([f"[{d['title']}] {d['content']}" for d in docs])
            prompt = (
                "You are a helpful real-time customer service assistant. "
                "Use the context to answer accurately and concisely. "
                "If context is insufficient, say what is missing and propose next steps.\n\n"
                f"Context:\n{context}\n\nUser query: {message}"
            )
            response = self._genai_response(prompt)
            if not response:
                if docs:
                    response = docs[0]["content"]
                else:
                    response = "I could not find a matching answer in the knowledge base. Please contact support or provide more details."

        response = prefix + response
        if user_language and user_language != PRIMARY_LANGUAGE:
            response = respond_in_language(response, user_language, PRIMARY_LANGUAGE)

        return {
            "response": response,
            "sentiment": sentiment,
            "language": user_language,
            "mode": mode,
        }

def build_image_prompt(user_request: str) -> str:
    return build_image_generation_prompt(user_request, context="customer service bot")
