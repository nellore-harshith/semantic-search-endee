from sentence_transformers import SentenceTransformer
import json
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load stored vectors
with open("vectors.json", "r") as f:
    data = json.load(f)

documents = data["documents"]
embeddings = np.array(data["embeddings"])

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# User query
query = input("Enter your search query: ")

query_embedding = model.encode(query)

scores = []

for i, emb in enumerate(embeddings):
    score = cosine_similarity(query_embedding, emb)
    scores.append((score, documents[i]))

# Sort by similarity score
scores.sort(key=lambda x: x[0], reverse=True)

print("\nTop Semantic Search Results:\n")

for score, doc in scores[:3]:
    print(f"[{doc['source']}] {doc['text']}")
    print(f"Similarity Score: {score:.4f}")
    print("-" * 50)
