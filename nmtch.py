import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# -------------------------------
# 1. Sample dataset of names
# -------------------------------
names = ["Rajesh Kumar","Rajesh Kumar Sharma","Rajesh Sharma","sharma Rajesh",
    "John Smith", "Jon Smyth", "Jane Doe", "Janet Doe",
    "Michael Johnson", "Michel Jonson", "Mick Jhonson",
    "Alicia Keys", "Alisha Keyes", "Alice Key",
    "Jonson Micheal"  # Added swapped order
]

# -------------------------------
# 2. Preprocessing: normalize names (casefold + sort tokens)
# -------------------------------
def normalize_name(name: str) -> str:
    tokens = name.lower().split()
    tokens.sort()  # Sort alphabetically to remove order sensitivity
    return " ".join(tokens)

normalized_names = [normalize_name(n) for n in names]

# -------------------------------
# 3. Create character-level TF-IDF embeddings
# -------------------------------
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
name_vectors = vectorizer.fit_transform(normalized_names)

# Convert to dense float32 array for FAISS
name_vectors = name_vectors.astype(np.float32).toarray()

# Normalize vectors for cosine similarity
name_vectors = normalize(name_vectors, norm='l2')

# -------------------------------
# 4. Build FAISS index (cosine similarity via inner product)
# -------------------------------
dimension = name_vectors.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(name_vectors)

# -------------------------------
# 5. Search function
# -------------------------------
def find_similar_names(query, top_k=3):
    query_norm = normalize_name(query)
    query_vec = vectorizer.transform([query_norm]).astype(np.float32).toarray()
    query_vec = normalize(query_vec, norm='l2')

    distances, indices = index.search(query_vec, top_k)
    results = [(names[i], float(distances[0][pos])) for pos, i in enumerate(indices[0])]
    return results

# -------------------------------
# 6. Example usage
# -------------------------------
if __name__ == "__main__":
    query_name = "Rajesh Kumar"
    matches = find_similar_names(query_name, top_k=5)

    print(f"Query: {query_name}")
    print("Matches:")
    for name, score in matches:
        print(f"  {name} (similarity: {score:.4f})")
