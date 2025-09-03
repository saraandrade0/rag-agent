import os, joblib, numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if __name__ == "__main__":
    root = project_root()
    names = joblib.load(os.path.join(root, "meta.joblib"))["names"]
    X = np.load(os.path.join(root, "embeddings.npy"))  # (n_docs, dim)

    # Vizinhos mais próximos por similaridade de cosseno
    nn = NearestNeighbors(n_neighbors=3, metric="cosine").fit(X)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    while True:
        q = input("Question (Enter to exit): ").strip()
        if not q:
            print("Bye."); break
        qv = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
        dist, idx = nn.kneighbors(qv, n_neighbors=min(3, len(names)))
        sims = 1.0 - dist[0]   # converter distância p/ similaridade
        for i, s in zip(idx[0], sims):
            print(f"> {names[i]} (sim={float(s):.3f})")
