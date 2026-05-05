import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer

_model = None
# Module-level stores persist within a Streamlit session
_req_store = {"chunks": [], "embeddings": None}
_course_store = {}  # course_id -> {"chunks": [...], "embeddings": np.ndarray}


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def simple_chunker(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        if start + chunk_size >= text_length:
            break
        start += chunk_size - overlap
    return chunks


def embed_requirements(text):
    global _req_store
    chunks = simple_chunker(text, chunk_size=800, overlap=150)
    if not chunks:
        _req_store = {"chunks": [], "embeddings": None}
        return
    embeddings = _get_model().encode(chunks, convert_to_numpy=True)
    _req_store = {"chunks": chunks, "embeddings": embeddings}


def embed_course_content(course_id, text):
    if course_id in _course_store:
        return
    chunks = simple_chunker(text, chunk_size=1500, overlap=200)
    if not chunks:
        return
    embeddings = _get_model().encode(chunks, convert_to_numpy=True)
    _course_store[course_id] = {"chunks": chunks, "embeddings": embeddings}


def _cosine_query(query_text, chunks, embeddings, n_results):
    if embeddings is None or not chunks:
        return []
    q = _get_model().encode([query_text], convert_to_numpy=True)[0]
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q)
    norms = np.where(norms == 0, 1e-10, norms)
    scores = (embeddings @ q) / norms
    top_k = min(n_results, len(chunks))
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_idx]


def query_requirements(query_text, n_results=3):
    return _cosine_query(query_text, _req_store["chunks"], _req_store["embeddings"], n_results)


def query_course_content(course_id, query_text, n_results=5):
    if course_id not in _course_store:
        return []
    s = _course_store[course_id]
    return _cosine_query(query_text, s["chunks"], s["embeddings"], n_results)


def get_full_requirements_text():
    return "\n\n".join(_req_store["chunks"])


def get_full_course_text(course_id):
    if course_id not in _course_store:
        return ""
    return "\n\n".join(_course_store[course_id]["chunks"])


def generate_course_id(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()
