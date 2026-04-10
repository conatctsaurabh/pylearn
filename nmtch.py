import re
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# -------------------------------
# 1. Sample dataset of names
# -------------------------------
names = [
    "John Smith Pvt Ltd", "Jon Smyth", "Jane Doe", "Janet Doe",
    "Michael Johnson", "Michel Jonson", "Mick Jhonson",
    "Alicia Keys Pte", "Alisha Keyes", "Alice Key",
    "Jonson Micheal", "Micheal A. Jonson", "Johnson Michael Inc",
    "Johnson & Johnson Pvt Ltd", "The Bank of America"
]

# -------------------------------
# 2. Noise words & stopwords
# -------------------------------
NOISE_WORDS = {
    "pte", "pvt", "ltd", "inc", "corp", "co", "company", "llc", "plc"
}
STOP_WORDS = {
    "and", "of", "&", "the", "for", "in", "at", "by"
}

# -------------------------------
# 3. Preprocessing function
# -------------------------------
def normalize_name(name: str) -> str:
    # Lowercase
    name = name.lower()
    # Replace & with 'and' for consistency
    name = name.replace("&", " and ")
    # Remove punctuation
    name = re.sub(r"[^\w\s]", " ", name)
    # Tokenize
    tokens = name.split()
    # Remove noise words and stopwords
    tokens = [t for t in tokens if t not in NOISE_WORDS and t not in STOP_WORDS]
    # Sort tokens alphabetically for order-insensitivity
    tokens.sort()
    return " ".join(tokens)

# Preprocess corpus
normalized_names = [normalize_name(n) for n in names]

# -------------------------------
# 4. Create character-level TF-IDF embeddings
# -------------------------------
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
name_vectors = vectorizer.fit_transform(normalized_names).astype(np.float32).toarray()
name_vectors = normalize(name_vectors, norm='l2')

# -------------------------------
# 5. Build FAISS CPU index
# -------------------------------
dimension = name_vectors.shape[1]
index = faiss.IndexFlatIP(dimension)  # Cosine similarity via inner product
index.add(name_vectors)

# -------------------------------
# 6. Matching function with tagging
# -------------------------------
def find_similar_names(query, top_k=5, exact_threshold=0.95, partial_threshold=0.75):
    query_norm = normalize_name(query)
    query_vec = vectorizer.transform([query_norm]).astype(np.float32).toarray()
    query_vec = normalize(query_vec, norm='l2')

    distances, indices = index.search(query_vec, top_k)
    results = []

    for pos, idx in enumerate(indices[0]):
        score = float(distances[0][pos])
        matched_name = names[idx]

        # Tagging criteria
        if score >= exact_threshold:
            tag = "Exact"
        elif score >= partial_threshold:
            tag = "Partial"
        else:
            tag = "Fuzzy"

        results.append({
            "name": matched_name,
            "similarity": round(score, 4),
            "match_type": tag
        })

    return results

# -------------------------------
# 7. Example usage
# -------------------------------
if __name__ == "__main__":
    query_name = "The Pvt Micheal Jonson & Co Ltd"
    matches = find_similar_names(query_name, top_k=5)

    print(f"Query: {query_name}")
    print("Matches:")
    for m in matches:
        print(f"  {m['name']} (similarity: {m['similarity']}, type: {m['match_type']})")
