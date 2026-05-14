from __future__ import annotations
from typing import List
import numpy as np

def tfidf_ready():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer

def train_word2vec(sentences: List[List[str]]):
    try:
        from gensim.models import Word2Vec
    except Exception:
        return None
    return Word2Vec(sentences=sentences, vector_size=64, window=4, min_count=1, workers=1)

def vectorize_texts(texts: List[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english")
    mat = vec.fit_transform(texts)
    return vec, mat

def semantic_search_score(query: str, documents: List[str]):
    vec, mat = vectorize_texts([query] + documents)
    q = mat[0]
    d = mat[1:]
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(q, d)[0]
