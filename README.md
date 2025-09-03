🤖 RAG Agent  — No-FAISS + Agent + LLM

This repository demonstrates a modern RAG pipeline with:

Semantic embeddings (Sentence Transformers)

Vector search using scikit-learn (cosine similarity, no FAISS required)

Agent design with simple tool use / function calling

LLM integration (optional) with OpenAI

Runs fully offline by default. If you want LLM-generated answers, you can plug an OpenAI API key.

🚀 Quick Start (Default: No-FAISS)

Works on macOS/Linux with Python 3.12. No Docker or GPU required.

# 1) Create & activate venv
python3.12 -m venv .venv312
source .venv312/bin/activate

# 2) Install requirements
pip install -r requirements_nofaiss.txt

# 3) Generate demo docs
python src/generatedocs.py

# 4) Build embeddings
python src/embed_nofaiss.py

# 5) Run semantic queries
python src/query_nofaiss.py


Example:

Question: Do I need internet to run it?
> faq.txt   (sim=0.842)
> guide1.md (sim=0.411)

🧠 Agent Mode

Run the Agent:

python src/agent_nofaiss_llm.py

Example interaction
Question: What is the price?
[route] pricing
[Tool] Pricing: Basic $19/mo, Pro $49/mo, Enterprise $199/mo.
[Answer] Based on context and tool output, answer to: What is the price?

Question: How do I install it?
[route] setup
[Tool] Setup: 1) Install deps 2) Run embed_nofaiss.py 3) Start agent.
[Answer] Based on context and tool output, answer to: How do I install it?

🤖 Optional: LLM Integration (OpenAI)

Enable natural language answers with an LLM:

pip install openai
export OPENAI_API_KEY="sk-YOUR_KEY_HERE"   # macOS/Linux
# setx OPENAI_API_KEY sk-YOUR_KEY_HERE     # Windows (PowerShell)

python src/agent_nofaiss_llm.py


With a real key, the agent uses gpt-4o-mini to compose answers from:

retrieved context (docs)

tool outputs (pricing, setup, etc.)

📂 Project Structure
rag-agent-demo/
├─ docs/                 # Knowledge base
│  ├─ faq.txt
│  └─ guide1.md
├─ src/                  # Source code
│  ├─ generatedocs.py
│  ├─ embed_nofaiss.py
│  ├─ query_nofaiss.py
│  └─ agent_nofaiss_llm.py
├─ requirements_nofaiss.txt
├─ README.md
└─ LICENSE

🛠️ Features

RAG (Retrieval-Augmented Generation) with embeddings

Vector search via NearestNeighbors (cosine)

Agent design → routes queries to tools or docs

Tool use / Function calling (pricing, setup)

LLM orchestration (optional with OpenAI)

Matches common job requirements: “RAG, tool use, function calling, and agent design.”

🧪 Example Questions

“How does this AI agent work?”

“Do I need internet to run it?”

“Can I add new documents?”

“What is the price?” → routed to pricing tool

“How do I install it?” → routed to setup tool

📜 License

Distributed under the MIT License. See LICENSE
 for details.git add .
