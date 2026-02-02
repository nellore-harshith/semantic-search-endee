from sentence_transformers import SentenceTransformer
import json
import numpy as np

VECTOR_STORE = "endee_store.json"

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Endee-style vector DB
with open(VECTOR_STORE, "r") as f:
    store = json.load(f)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def semantic_search(query, top_k=3):
    query_embedding = model.encode(query).tolist()

    scored_results = []
    for item in store:
        score = cosine_similarity(query_embedding, item["embedding"])
        scored_results.append((score, item))

    scored_results.sort(reverse=True, key=lambda x: x[0])
    return scored_results[:top_k]

# Example query
query = input("Enter your search query: ")
results = semantic_search(query)

print("\nTop Results:\n")
for score, item in results:
    print(f"Score: {score:.4f}")
    print(f"Source: {item['metadata']['source']}")
    print(f"Text: {item['document']}")
    print("-" * 50)
