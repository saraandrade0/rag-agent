import os, joblib, numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

root = project_root()
names = joblib.load(os.path.join(root, "meta.joblib"))["names"]
X = np.load(os.path.join(root, "embeddings.npy"))

nn = NearestNeighbors(n_neighbors=3, metric="cosine").fit(X)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def retrieve(question, k=3):
    qv = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    dist, idx = nn.kneighbors(qv, n_neighbors=min(k, len(names)))
    sims = 1.0 - dist[0]
    return [(names[i], float(s)) for i, s in zip(idx[0], sims)]

TOOLS = {
    "pricing": lambda _: "Pricing: Basic $19/mo, Pro $49/mo, Enterprise $199/mo.",
    "setup":   lambda _: "Setup: 1) Install deps 2) Run embed_nofaiss.py 3) Start agent."
}

def decide_tool(q):
    q = q.lower()
    if any(w in q for w in ["price", "pricing", "cost", "plan"]):
        return "pricing"
    if any(w in q for w in ["install", "setup", "run", "start"]):
        return "setup"
    return ""

def llm_answer(question, ctx, tool_out=""):
    if not OPENAI_AVAILABLE:
        return f"[LLM not installed] Based on context{' and tool output' if tool_out else ''}, answer to: {question}"

    api_key = os.getenv("OPENAI_API_KEY", "sk-FAKE_KEY_123")
    client = OpenAI(api_key=api_key)

    prompt = f"""
    You are an assistant that answers concisely using the retrieved context and tool outputs.

    Question: {question}
    Tool output: {tool_out}
    Retrieved context:
    {ctx}

    Provide a concise, helpful answer.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    while True:
        q = input("Question (Enter to exit): ").strip()
        if not q:
            print("Bye."); break

        ctx = retrieve(q)
        tool = decide_tool(q)
        tool_out = TOOLS[tool](q) if tool else ""

        print(f"[route] {tool or 'none'}")
        if tool_out:
            print("[Tool]", tool_out)

        ctx_preview = "\n".join(f"{doc} (sim={s:.3f})" for doc, s in ctx)
        answer = llm_answer(q, ctx_preview, tool_out)
        print("[Answer]", answer, "\n")
