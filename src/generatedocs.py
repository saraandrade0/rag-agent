import os

faq_content = """Frequently Asked Questions (FAQ)

Q: How does this AI agent work?
A: It indexes documents and retrieves the most relevant passages to answer user questions.

Q: Do I need internet access to run this system?
A: No. This demo runs locally and uses only open-source libraries.

Q: Can I add new documents?
A: Yes. Simply place text files into the docs/ folder and run embed_pro.py again.

Q: Does the model understand English?
A: Yes, the vectorization pipeline can handle multiple languages, including English and Portuguese.
"""

guide_content = """# Quick Guide to the RAG Agent Demo

This project demonstrates how to build a simple Retrieval-Augmented Generation (RAG) agent.

## Main steps
1. Index documents with embeddings.
2. Query using vector similarity (FAISS).
3. Return the most relevant passages to the user question.
4. Use a simple routing logic (agent) to decide response type.

## Example questions
- "How does this AI agent work?"
- "Do I need internet to run it?"
- "Can I add new documents?"
"""

def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def generate_docs(output_dir=None):
    output_dir = output_dir or os.path.join(project_root(), "docs")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "faq.txt"), "w", encoding="utf-8") as f:
        f.write(faq_content)
    with open(os.path.join(output_dir, "guide1.md"), "w", encoding="utf-8") as f:
        f.write(guide_content)
    print(f"Demo docs created in {output_dir}/")

if __name__ == "__main__":
    generate_docs()
