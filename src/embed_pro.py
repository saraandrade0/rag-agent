import os, joblib, numpy as np
from sentence_transformers import SentenceTransformer

def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def docs_dir():
    return os.path.join(project_root(), "docs")

def read_docs(path):
    names, texts = [], []
    for n in os.listdir(path):
        p = os.path.join(path, n)
        if os.path.isfile(p) and n.endswith((".txt", ".md")):
            names.append(n)
            texts.append(open(p, encoding="utf-8").read())
    if not names:
        raise RuntimeError(f"No .txt/.md files in {path}")
    return names, texts

if __name__ == "__main__":
    root = project_root()
    names, texts = read_docs(docs_dir())

    print("Loading sentence-transformers model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    np.save(os.path.join(root, "embeddings.npy"), embs)
    joblib.dump({"names": names}, os.path.join(root, "meta.joblib"))
    print(f"Embeddings saved: {embs.shape} -> embeddings.npy, meta.joblib")
