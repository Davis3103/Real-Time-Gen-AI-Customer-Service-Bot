from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"

KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base" / "base_knowledge.json"
MEDICAL_SAMPLE_PATH = DATA_DIR / "medical" / "medquad_sample.json"
ARXIV_SAMPLE_PATH = DATA_DIR / "arxiv" / "arxiv_sample.json"

PERSIST_DIR = DATA_DIR / "vector_store"
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
PRIMARY_LANGUAGE = os.getenv("PRIMARY_LANGUAGE", "en").strip().lower()
SUPPORTED_LANGUAGES = ["en", "hi", "es", "fr", "te"]
