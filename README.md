# 🤖 RAG Agent — Semantic Search + Agent + LLM

Portfolio project demonstrating a modern RAG pipeline with:

- **Semantic embeddings** (Sentence Transformers)
- **Vector search** using scikit-learn (cosine similarity)
- **Agent design** with simple tool use / function calling
- **LLM integration** (optional) with OpenAI

Runs fully offline by default. If you want LLM-generated answers, plug an OpenAI API key.

## 🚀 Quick Start

Works on macOS/Linux with Python 3.10+. No Docker or GPU required.
```bash
# 1) Create & activate venv
python3 -m venv .venv
source .venv/bin/activate

# 2) Install requirements
pip install -r requirements.txt

# 3) Generate demo docs
python src/generatedocs.py

# 4) Build embeddings
python src/embed_pro.py

# 5) Run semantic queries
python src/query_pro.py
```

**Example:**
Question: Do I need internet to run it?

faq.txt   (sim=0.842)
guide1.md (sim=0.411)


## 🧠 Agent Mode
```bash
python src/agent_llm.py
```

**Example interaction:**
Question: What is the price?

[route] pricing

[Tool] Pricing: Basic $19/mo, Pro $49/mo, Enterprise $199/mo.

[Answer] Based on context and tool output...

Question: How do I install it?

[route] setup

[Tool] Setup: 1) Install deps 2) Run embed_pro.py 3) Start agent.

[Answer] Based on context and tool output...

## 🤖 Optional: LLM Integration (OpenAI)

Enable natural language answers with an LLM:
```bash
pip install openai
export OPENAI_API_KEY="your-key-here"
python src/agent_llm.py
```

With a real key, the agent uses `gpt-4o-mini` to compose answers from retrieved context and tool outputs.

## 📂 Project Structure
rag-agent/

├── docs/                 # Knowledge base

│   ├── faq.txt

│   └── guide1.md

├── src/                  # Source code

│   ├── generatedocs.py   # Generate demo documents

│   ├── embed_pro.py      # Build embeddings

│   ├── query_pro.py      # Semantic search CLI

│   └── agent_llm.py      # Agent with tool use + optional LLM

├── requirements.txt

├── README.md

└── LICENSE

## 🛠️ Features

- **RAG** (Retrieval-Augmented Generation) with embeddings
- **Vector search** via `NearestNeighbors` (cosine similarity)
- **Agent design** → routes queries to tools or docs
- **Tool use / Function calling** (pricing, setup)
- **LLM orchestration** (optional with OpenAI)

## 🧪 Example Questions

- "How does this AI agent work?"
- "Do I need internet to run it?"
- "Can I add new documents?"
- "What is the price?" → routed to pricing tool
- "How do I install it?" → routed to setup tool

## 📜 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
