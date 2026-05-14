from __future__ import annotations
from functools import lru_cache
from typing import Dict

LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "te": "Telugu",
}

PHRASE_MAP = {
    ("en","hi"): {"hello":"नमस्ते","thank you":"धन्यवाद","sorry":"माफ़ कीजिए","help":"मदद"},
    ("en","es"): {"hello":"hola","thank you":"gracias","sorry":"lo siento","help":"ayuda"},
    ("en","fr"): {"hello":"bonjour","thank you":"merci","sorry":"désolé","help":"aide"},
    ("en","te"): {"hello":"హలో","thank you":"ధన్యవాదాలు","sorry":"క్షమించండి","help":"సహాయం"},
}

@lru_cache(maxsize=1)
def _translator():
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator
    except Exception:
        return None

def detect_language(text: str) -> str:
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "en"

def translate_text(text: str, target: str = "en", source: str = "auto") -> str:
    if not text:
        return text
    GoogleTranslator = _translator()
    if GoogleTranslator is not None:
        try:
            if source == "auto":
                source = "auto"
            return GoogleTranslator(source=source, target=target).translate(text)
        except Exception:
            pass
    return _fallback_translate(text, target)

def _fallback_translate(text: str, target: str) -> str:
    target = target.lower()
    if target == "en":
        return text
    mapping = PHRASE_MAP.get(("en", target), {})
    out = text
    for eng, native in mapping.items():
        out = out.replace(eng, native).replace(eng.title(), native)
    return out

def respond_in_language(text_en: str, user_language: str, primary_language: str = "en") -> str:
    user_language = (user_language or "en").lower()
    if user_language == primary_language:
        return text_en
    return translate_text(text_en, target=user_language, source=primary_language)
