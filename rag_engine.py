import os
import chromadb
from chromadb.utils import embedding_functions
import hashlib

# Persistent DB path
CHROMA_PATH = "chroma_db"
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Using sentence-transformers' fast and small model for local execution without API keys
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def get_collection(name):
    return client.get_or_create_collection(name=name, embedding_function=emb_fn)

def delete_collection(name):
    try:
        client.delete_collection(name)
    except Exception:
        pass

def simple_chunker(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        if start + chunk_size >= text_length:
            break
        start += (chunk_size - overlap)
        
    return chunks

def embed_requirements(text):
    """Embed the requirements text. Since there's one set of requirements, we overwrite."""
    delete_collection("requirements")
    col = get_collection("requirements")
    
    chunks = simple_chunker(text, chunk_size=800, overlap=150)
    if not chunks:
        return
    
    docs = chunks
    ids = [f"req_{i}" for i in range(len(chunks))]
    col.add(documents=docs, ids=ids)

def embed_course_content(course_id, text):
    """Embed course content. Skips if already embedded (content hash = stable ID)."""
    col = get_collection(f"course_{course_id}")
    if col.count() > 0:
        return

    chunks = simple_chunker(text, chunk_size=1500, overlap=200)
    if not chunks:
        return

    ids = [f"course_{course_id}_{i}" for i in range(len(chunks))]
    col.add(documents=chunks, ids=ids)

def query_requirements(query_text, n_results=3):
    col = get_collection("requirements")
    if col.count() == 0:
        return []
        
    res = col.query(query_texts=[query_text], n_results=min(n_results, col.count()))
    if res and res["documents"] and len(res["documents"]) > 0:
        return res["documents"][0]
    return []

def query_course_content(course_id, query_text, n_results=5):
    col = get_collection(f"course_{course_id}")
    if col.count() == 0:
        return []
        
    res = col.query(query_texts=[query_text], n_results=min(n_results, col.count()))
    if res and res["documents"] and len(res["documents"]) > 0:
        return res["documents"][0]
    return []

def get_full_requirements_text():
    """Retrieve all requirements chunks."""
    col = get_collection("requirements")
    if col.count() == 0:
        return ""
    res = col.get()
    if res and res["documents"]:
        return "\n\n".join(res["documents"])
    return ""

def get_full_course_text(course_id):
    """Retrieve all course chunks."""
    col = get_collection(f"course_{course_id}")
    if col.count() == 0:
        return ""
    res = col.get()
    if res and res["documents"]:
        return "\n\n".join(res["documents"])
    return ""

def generate_course_id(name):
    return hashlib.md5(name.encode('utf-8')).hexdigest()
