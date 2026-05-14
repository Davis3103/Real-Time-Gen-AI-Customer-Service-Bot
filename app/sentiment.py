from __future__ import annotations
from functools import lru_cache

POSITIVE_WORDS = {"good","great","excellent","thanks","thank","happy","love","awesome","nice","helpful","resolved"}
NEGATIVE_WORDS = {"bad","terrible","angry","hate","upset","slow","issue","problem","not working","broken","frustrated","worried","unhappy"}

@lru_cache(maxsize=1)
def _pipeline():
    try:
        from transformers import pipeline
        return pipeline("sentiment-analysis")
    except Exception:
        return None

def detect_sentiment(text: str) -> dict:
    text_l = (text or "").lower()
    pipe = _pipeline()
    if pipe is not None:
        try:
            out = pipe(text[:512])[0]
            label = out.get("label", "NEUTRAL").upper()
            score = float(out.get("score", 0.0))
            if "NEG" in label:
                label = "negative"
            elif "POS" in label:
                label = "positive"
            else:
                label = "neutral"
            return {"label": label, "score": score, "method": "transformers"}
        except Exception:
            pass
    pos = sum(word in text_l for word in POSITIVE_WORDS)
    neg = sum(word in text_l for word in NEGATIVE_WORDS)
    if neg > pos:
        return {"label": "negative", "score": min(1.0, 0.55 + 0.1 * neg), "method": "lexicon"}
    if pos > neg:
        return {"label": "positive", "score": min(1.0, 0.55 + 0.1 * pos), "method": "lexicon"}
    return {"label": "neutral", "score": 0.5, "method": "lexicon"}

def sentiment_tone(sentiment: str) -> str:
    sentiment = (sentiment or "neutral").lower()
    if sentiment == "negative":
        return "empathetic"
    if sentiment == "positive":
        return "friendly"
    return "professional"

def tone_prefix(sentiment: str) -> str:
    s = (sentiment or "neutral").lower()
    if s == "negative":
        return "I understand this is frustrating. "
    if s == "positive":
        return "Glad to help. "
    return ""
