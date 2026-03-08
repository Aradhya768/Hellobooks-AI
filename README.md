 Hellobooks AI — Accounting Assistant

> A **RAG-based (Retrieval-Augmented Generation)** AI assistant that answers accounting questions using a curated knowledge base. Built with Python, sentence-transformers, FAISS, and **Ollama (100% free — no API key needed!)**.

## Project Structure

```
hellobooks/
├── knowledge_base/              # Task 1: Accounting documents (Markdown)
│   ├── bookkeeping.md
│   ├── invoices.md
│   ├── profit_and_loss.md
│   ├── balance_sheet.md
│   ├── cash_flow.md
│   ├── gst_and_taxation.md
│   ├── accounts_payable_receivable.md
│   ├── financial_ratios.md
│   └── depreciation_amortization.md
│
├── src/
│   └── rag.py                   # Task 2: Full RAG pipeline
│
├── Dockerfile                   # Task 3: Container setup
├── requirements.txt
├── .env.example

```

---

## How It Works (RAG Flow)

```
User Question
     │
     ▼
Embed question → sentence-transformers (all-MiniLM-L6-v2, free, runs on CPU)
     │
     ▼
FAISS vector search → retrieve Top-4 most relevant document chunks
     │
     ▼
Build prompt: [System Prompt] + [Context Chunks] + [Question]
     │
     ▼
Send to Ollama (free local LLM — Mistral 7B running on your machine)
     │
     ▼
Return answer to user
```

---

##  Quick Start 

### Step 1 — Install Ollama (free local LLM)

Go to **https://ollama.com** → Download and install for your OS (Windows/Mac/Linux).

After installing, open a terminal and pull the Mistral model (free, ~4 GB):
```bash
ollama pull mistral
```

Keep Ollama running in the background (it starts automatically after install on most systems).

### Step 2 — Install Python dependencies

```bash
cd hellobooks
pip install -r requirements.txt
```

### Step 3 — Run the assistant

```bash
python src/rag.py
```

That's it! No API key, no credit card, no cost. 🎉

### Example questions to try:
```
You: What is a balance sheet?
You: How do I calculate gross profit margin?
You: What is GST Input Tax Credit?
You: What is the difference between cash flow and profit?
You: How is depreciation calculated?
```

##  Docker Setup (Task 3)

### Build the image
```bash
docker build -t hellobooks-ai .
```

### Run with Ollama (free)
```bash
# First, make sure Ollama is running on your host machine
docker run -it \
  -e LLM_PROVIDER=ollama \
  -e OLLAMA_URL=http://host.docker.internal:11434 \
  hellobooks-ai
```

### Run with OpenAI (if you have a key)
```bash
docker run -it \
  -e LLM_PROVIDER=openai \
  -e OPENAI_API_KEY=sk-your-key-here \
  hellobooks-ai
```

##  Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama` (free) / `openai` / `huggingface` |
| `LLM_MODEL` | `mistral` | Ollama model name |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server address |
| `OPENAI_API_KEY` | — | Only needed for OpenAI provider |
| `HF_API_KEY` | — | Only needed for HuggingFace provider |


##  Knowledge Base Topics

| File | Topics Covered |
|------|----------------|
| `bookkeeping.md` | Double-entry, chart of accounts, accounting equation |
| `invoices.md` | Invoice components, payment terms, AR vs AP |
| `profit_and_loss.md` | Revenue, COGS, gross/net profit, margins |
| `balance_sheet.md` | Assets, liabilities, equity, key ratios |
| `cash_flow.md` | Operating/investing/financing activities, FCF |
| `gst_and_taxation.md` | GST slabs, ITC, TDS, income tax rates |
| `accounts_payable_receivable.md` | DSO, DPO, aging report, cash conversion cycle |
| `financial_ratios.md` | Liquidity, profitability, leverage, efficiency ratios |
| `depreciation_amortization.md` | SLM, WDV methods, EBITDA |


## Tech Stack

| Component | Technology | Cost |
|-----------|-----------|------|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) 
| Vector Store | `FAISS` (Facebook AI Similarity Search) | 
| LLM | Ollama + Mistral 7B (runs locally) | 
| Language | Python 3.11 | 
| Container | Docker | 


##  License

MIT License — feel free to use, modify, and distribute.
