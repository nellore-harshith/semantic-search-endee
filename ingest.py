from sentence_transformers import SentenceTransformer
import os
import json

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Endee-style vector store
VECTOR_STORE = "endee_store.json"

def chunk_text(text, chunk_size=40):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

store = []
doc_id = 0

for file in os.listdir("data"):
    with open(f"data/{file}", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    embeddings = model.encode(chunks)

    for chunk, emb in zip(chunks, embeddings):
        store.append({
            "id": str(doc_id),
            "embedding": emb.tolist(),
            "document": chunk,
            "metadata": {"source": file}
        })
        doc_id += 1

# Persist vectors (Endee-style persistence)
with open(VECTOR_STORE, "w") as f:
    json.dump(store, f)

print("Documents successfully embedded and stored using Endee-style vector database")
