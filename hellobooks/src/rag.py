"""
Hellobooks AI Assistant - RAG-Based Q&A System
Uses sentence-transformers for embeddings and FAISS for vector storage.
Default LLM: Ollama (100% free, runs locally — no API key needed!)
Also supports: OpenAI, HuggingFace
"""

import os
import glob
import pickle
import numpy as np
from pathlib import Path

# ── Vector store & embeddings ─────────────────────────────────────────────────
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

# ── LLM provider selection ────────────────────────────────────────────────────
# Options: "ollama" (free/local) | "openai" | "huggingface"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

# ── Constants ─────────────────────────────────────────────────────────────────
KNOWLEDGE_BASE_DIR = Path(__file__).parent.parent / "knowledge_base"
VECTOR_STORE_PATH  = Path(__file__).parent / "vector_store.pkl"
EMBED_MODEL_NAME   = "all-MiniLM-L6-v2"   # ~80 MB, runs on CPU, free
CHUNK_SIZE         = 500
CHUNK_OVERLAP      = 100
TOP_K              = 4
LLM_MODEL          = os.getenv("LLM_MODEL", "mistral")   # Ollama model name


# ─────────────────────────────────────────────────────────────────────────────
# Document loading & chunking
# ─────────────────────────────────────────────────────────────────────────────

def load_documents(directory: Path) -> list[dict]:
    """Load all .md and .txt files from the knowledge base directory."""
    docs = []
    for filepath in sorted(glob.glob(str(directory / "*.md")) +
                           glob.glob(str(directory / "*.txt"))):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        docs.append({"source": Path(filepath).name, "content": content})
    print(f"[+] Loaded {len(docs)} documents from {directory}")
    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def prepare_chunks(docs: list[dict]) -> list[dict]:
    """Convert documents into a flat list of text chunks with metadata."""
    all_chunks = []
    for doc in docs:
        for chunk in chunk_text(doc["content"]):
            all_chunks.append({"source": doc["source"], "text": chunk.strip()})
    print(f"[+] Created {len(all_chunks)} chunks")
    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
# Embedding & FAISS index
# ─────────────────────────────────────────────────────────────────────────────

class VectorStore:
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        if not EMBEDDINGS_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers and faiss-cpu are required.\n"
                "Run: pip install sentence-transformers faiss-cpu"
            )
        print(f"[+] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index: faiss.IndexFlatL2 | None = None
        self.chunks: list[dict] = []

    def build(self, chunks: list[dict]):
        """Embed all chunks and build a FAISS index."""
        texts = [c["text"] for c in chunks]
        print(f"[+] Embedding {len(texts)} chunks …")
        embeddings = self.model.encode(texts, show_progress_bar=True,
                                       convert_to_numpy=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings.astype("float32"))
        self.chunks = chunks
        print(f"[+] FAISS index built with {self.index.ntotal} vectors (dim={dim})")

    def save(self, path: Path = VECTOR_STORE_PATH):
        """Persist the index and metadata to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks,
                         "index_bytes": faiss.serialize_index(self.index)}, f)
        print(f"[+] Vector store saved → {path}")

    def load(self, path: Path = VECTOR_STORE_PATH):
        """Load a previously saved index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.index  = faiss.deserialize_index(data["index_bytes"])
        print(f"[+] Vector store loaded — {self.index.ntotal} vectors")

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Retrieve the top-k most relevant chunks for a query."""
        q_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        _, indices = self.index.search(q_vec, top_k)
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]


# ─────────────────────────────────────────────────────────────────────────────
# LLM answer generation
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Hellobooks AI, a knowledgeable accounting assistant for small businesses.
Answer questions using ONLY the context provided. Be concise, accurate, and helpful.
If the answer isn't in the context, say "I don't have enough information on that topic."
Format numbers clearly and use bullet points where appropriate."""


def build_prompt(question: str, context_chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )
    return f"""Context:
{context}

Question: {question}

Answer:"""


def generate_answer_ollama(question: str, context_chunks: list[dict]) -> str:
    """
    FREE: Uses Ollama running locally on your machine.
    Install Ollama from https://ollama.com then run: ollama pull mistral
    """
    import requests
    model  = os.getenv("LLM_MODEL", "mistral")
    prompt = build_prompt(question, context_chunks)
    url    = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/chat"

    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0.2},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return (
            "[Error] Ollama is not running.\n"
            "Fix: Open a terminal and run:  ollama serve\n"
            "Then in another terminal:      ollama pull mistral"
        )


def generate_answer_openai(question: str, context_chunks: list[dict]) -> str:
    """Paid: Uses OpenAI API (requires OPENAI_API_KEY)."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    except ImportError:
        return "[Error] Run: pip install openai"
    prompt = build_prompt(question, context_chunks)
    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.2,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()


def generate_answer_huggingface(question: str, context_chunks: list[dict]) -> str:
    """Free tier: HuggingFace Inference API (requires free HF account token)."""
    import requests
    api_key = os.getenv("HF_API_KEY", "")
    model   = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    prompt  = build_prompt(question, context_chunks)
    full_prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\n{prompt} [/INST]"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": full_prompt,
                "parameters": {"max_new_tokens": 500, "temperature": 0.2}}
    resp = requests.post(
        f"https://api-inference.huggingface.co/models/{model}",
        headers=headers, json=payload, timeout=60
    )
    resp.raise_for_status()
    result = resp.json()
    if isinstance(result, list):
        text = result[0].get("generated_text", "")
        if "[/INST]" in text:
            text = text.split("[/INST]", 1)[-1]
        return text.strip()
    return str(result)


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    if LLM_PROVIDER == "openai":
        return generate_answer_openai(question, context_chunks)
    if LLM_PROVIDER == "huggingface":
        return generate_answer_huggingface(question, context_chunks)
    # Default: Ollama (free, local)
    return generate_answer_ollama(question, context_chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Main RAG pipeline
# ─────────────────────────────────────────────────────────────────────────────

class HellobooksRAG:
    def __init__(self):
        self.vs = VectorStore()

    def ingest(self, force_rebuild: bool = False):
        """Build (or load) the vector store from the knowledge base."""
        if VECTOR_STORE_PATH.exists() and not force_rebuild:
            self.vs.load()
        else:
            docs   = load_documents(KNOWLEDGE_BASE_DIR)
            chunks = prepare_chunks(docs)
            self.vs.build(chunks)
            self.vs.save()

    def ask(self, question: str, verbose: bool = False) -> str:
        """Full RAG pipeline: retrieve → augment → generate."""
        retrieved = self.vs.search(question, top_k=TOP_K)
        if verbose:
            print("\n── Retrieved chunks ──────────────────────────────")
            for i, c in enumerate(retrieved, 1):
                print(f"  [{i}] {c['source']}: {c['text'][:120]} …")
            print("─────────────────────────────────────────────────\n")
        answer = generate_answer(question, retrieved)
        return answer


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hellobooks AI — Accounting Q&A")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild of the vector store")
    parser.add_argument("--question", "-q", type=str,
                        help="Single question (non-interactive mode)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show retrieved context chunks")
    args = parser.parse_args()

    print("=" * 60)
    print("  Hellobooks AI — Accounting Assistant")
    print("=" * 60)

    rag = HellobooksRAG()
    rag.ingest(force_rebuild=args.rebuild)

    if args.question:
        answer = rag.ask(args.question, verbose=args.verbose)
        print(f"\nQ: {args.question}\n")
        print(f"A: {answer}\n")
        return

    # Interactive loop
    print("\nType your accounting question (or 'quit' to exit).\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        answer = rag.ask(question, verbose=args.verbose)
        print(f"\nHellobooks AI: {answer}\n")


if __name__ == "__main__":
    main()
