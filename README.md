# 🧠 Corvit AI — FAISS Powered RAG Knowledge Assistant

A production-style **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about **Corvit Systems** — grounded strictly in a local knowledge-base PDF. Built with **FAISS**, **Sentence Transformers**, **Groq LLM**, and **Streamlit**.

---

## 📌 What This Is

This is **not** a generic ChatGPT-style wrapper. It demonstrates a real, end-to-end RAG pipeline:

```
PDF → Text Extraction → Cleaning → Chunking → Embeddings
    → FAISS Vector Index → Semantic Search → Top-K Context
    → Groq LLM → Grounded Answer
```

The assistant answers **only** from the retrieved content of `assets/corvit_knowledge.pdf`. If the answer isn't in the knowledge base, it says so instead of guessing:

> "I couldn't find this information in the provided Corvit knowledge base."

---

## ✨ Features

- 🔍 **True semantic search** via FAISS (`IndexFlatIP` + L2-normalized embeddings = cosine similarity) — not keyword matching
- 🧩 **Smart chunking** with configurable size/overlap and page-level metadata for accurate citations
- 🧠 **Sentence Transformers embeddings** (`all-MiniLM-L6-v2`) that capture meaning, not just words
- 🤖 **Groq LLM** for fast, grounded answer generation with a strict anti-hallucination system prompt
- 💾 **Persisted FAISS index** (`faiss_index/`) — built once, reused on every run (no rebuilding per question)
- ⚡ **Streamlit caching** (`st.cache_resource`) so the embedding model and index load once per session
- 💬 **Chat history** with `st.session_state`, a Clear Chat button, and clickable sample questions
- 🔎 **Transparent retrieval** — every answer shows retrieved chunks, similarity scores, and source pages
- ⚠️ **Volatile-info awareness** — flags prices, phone numbers, and schedules as subject to change
- 🔐 **Secure API key handling** — read from a local file, never hardcoded, never logged, never shown in the UI

---

## 🏗️ Project Structure

```
Corvit-AI-Knowledge-Assistant/
│
├── assets/
│   └── corvit_knowledge.pdf     # Knowledge base (source of truth)
│
├── faiss_index/                 # Auto-generated — persisted vector index
│   ├── index.faiss
│   └── metadata.pkl
│
├── app.py                       # Full RAG pipeline + Streamlit UI
├── requirements.txt
├── api-key.txt                  # Your Groq API key (never committed)
└── .gitignore
```

---

## ⚙️ How the RAG Pipeline Works

| Stage | What Happens |
|---|---|
| **1. Text Extraction** | `pypdf` reads every page of `corvit_knowledge.pdf` and extracts raw text |
| **2. Cleaning** | Whitespace is normalized while headings/paragraphs are preserved |
| **3. Chunking** | Text is split into ~800-character chunks with 150-character overlap, tagged with `chunk_id`, `source`, and `page` |
| **4. Embeddings** | Each chunk is converted into a vector using `sentence-transformers/all-MiniLM-L6-v2` — vectors capture *meaning*, so "car repair tips" and "automobile fixing guide" land close together even without shared keywords |
| **5. FAISS Index** | Vectors are L2-normalized and stored in a FAISS `IndexFlatIP` index (inner product on normalized vectors = cosine similarity) |
| **6. Query Time** | The user's question is embedded and normalized the same way, then FAISS returns the **top-K** most similar chunks with real similarity scores |
| **7. Context Building** | Retrieved chunks are assembled into a labeled context block (source + page) |
| **8. Groq LLM** | The context + question + a strict system prompt are sent to Groq, which must answer **only** from the given context |
| **9. Answer + Sources** | The final answer is shown along with an expandable "🔎 View Retrieved Context" panel and a sources list |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/AbdulRehman-developer1/Corvit-AI-Knowledge-Assistant.git
cd Corvit-AI-Knowledge-Assistant
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key

Create/edit `api-key.txt` in the project root and paste your key (no quotes, no extra text):

```
gsk_your_actual_groq_api_key_here
```

Get a free key from [console.groq.com](https://console.groq.com).

> ⚠️ `api-key.txt` is listed in `.gitignore` and must **never** be committed to GitHub.

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. On first run it will build and persist the FAISS index (`faiss_index/`); subsequent runs load it instantly.

---

## 🔄 Rebuilding the Index

The FAISS index is only rebuilt automatically if `faiss_index/` is missing or the PDF file changes. If you edit the **chunking logic**, **embedding model**, or other pipeline code in `app.py`, delete the stale index manually so it regenerates from the new logic:

**Windows (PowerShell):**
```powershell
Remove-Item -Recurse -Force faiss_index
```

**macOS / Linux:**
```bash
rm -rf faiss_index
```

Then re-run `streamlit run app.py`.

---

## 🧠 Key Concepts Explained

**Embeddings** — Numerical vectors that place semantically similar text close together in high-dimensional space, regardless of exact word overlap.

**FAISS (Facebook AI Similarity Search)** — A library for extremely fast similarity search over large vector collections. Here it performs exact cosine-similarity search via `IndexFlatIP` on normalized vectors.

**Semantic Search** — Retrieval based on *meaning* rather than literal keyword matches, powered by comparing query and document embeddings.

**RAG (Retrieval-Augmented Generation)** — Instead of relying on an LLM's internal (and possibly outdated or wrong) knowledge, relevant real documents are retrieved first and given to the LLM as grounding context, drastically reducing hallucination.

**Groq** — An LLM inference provider known for very fast response times, used here to generate the final answer from the retrieved context.

---

## 🛠️ Configuration

Key settings live at the top of `app.py`:

```python
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "openai/gpt-oss-120b"   # Groq model — change if deprecated
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
```

> **Note:** Groq periodically deprecates models. If you get a `model_not_found` error, check [Groq's models page](https://console.groq.com/docs/models) and update `MODEL_NAME` accordingly — it's the only place the model name needs to change.

---

## ❗ Troubleshooting

| Problem | Fix |
|---|---|
| `api-key.txt was not found` | Create the file in the project root with your Groq key |
| `model_not_found` (404) from Groq | The model in `MODEL_NAME` was deprecated — update it to a current Groq model |
| Answers seem outdated after code changes | Delete `faiss_index/` and re-run so it rebuilds |
| Sample question click doesn't leave the chat input visible | Update to the latest `app.py` — this was fixed by always rendering `st.chat_input()` |
| `PDF not found` | Ensure `assets/corvit_knowledge.pdf` exists |

---

## 📦 Requirements

```
streamlit
pypdf
sentence-transformers
faiss-cpu
groq
numpy
```

---

## ⚠️ Disclaimer

This chatbot answers from a static PDF knowledge base that may not reflect Corvit Systems' current prices, contact numbers, schedules, or course availability. Always verify volatile information (fees, phone numbers, timings) directly with Corvit before making decisions.

---

## 📄 License

Add your preferred license here (e.g., MIT).
