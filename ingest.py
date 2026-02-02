from sentence_transformers import SentenceTransformer
import os
import json
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
embeddings_store = []

def chunk_text(text, chunk_size=40):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

doc_id = 0

for file in os.listdir("data"):
    with open(f"data/{file}", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    embeddings = model.encode(chunks)

    for chunk, emb in zip(chunks, embeddings):
        documents.append({
            "id": doc_id,
            "source": file,
            "text": chunk
        })
        embeddings_store.append(emb.tolist())
        doc_id += 1

# Save locally (acts as vector DB)
with open("vectors.json", "w") as f:
    json.dump({
        "documents": documents,
        "embeddings": embeddings_store
    }, f)

print("Documents successfully embedded and stored locally")
