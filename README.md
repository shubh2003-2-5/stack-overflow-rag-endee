# Stack Overflow Semantic Search + RAG (Endee)

A complete end-to-end project that builds a semantic search and retrieval-augmented generation (RAG) assistant using:

- **Vector database:** Endee (https://github.com/endee-io/endee)
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **UI:** Streamlit
- **Dataset:** Stack Overflow questions (CSV)

---

## 📌 Project Structure

```
stack_overflow_rag/
│
├── data/
│   ├── train/Questions.csv        # original dataset (already exists)
│   └── processed/sample.csv       # sampled dataset (generated)
│
├── src/
│   ├── config.py
│   ├── sample_data.py
│   ├── embed.py
│   ├── ingest.py
│   ├── retrieve.py
│   ├── rag.py
│   └── utils.py
│
├── app.py                        # Streamlit UI
├── requirements.txt
└── README.md
```

---

## 🚀 Setup

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Run Endee** (local server)

Follow https://github.com/endee-io/endee to run the Endee server locally. By default the code expects:

- `http://localhost:8080` as the API URL

If your server runs elsewhere, set:

```bash
export ENDEE_API_URL=http://<host>:<port>
export ENDEE_API_TOKEN=<token_if_needed>
```

3. **Prepare data**

Place your Stack Overflow CSV at `data/train/Questions.csv`.

Then sample it (creates `data/processed/sample.csv`):

```bash
python -m src.sample_data
```

4. **Generate embeddings (optional, for faster ingestion)**

```bash
python -m src.embed
```

5. **Ingest embeddings into Endee**

```bash
python -m src.ingest
```

6. **Run Streamlit UI**

```bash
streamlit run app.py
```

## 🛠️ Troubleshooting

- If you see errors about missing modules, run `pip install -r requirements.txt`.
- If Streamlit is not recognized, install it: `pip install streamlit`.
- If you see PyTorch errors, install the correct version: `pip install torch`.
- For sentence-transformers, ensure `torch` and `scikit-learn` are installed.
- If Endee index is missing, check server logs and index creation logic.
- For OpenAI errors, set `OPENAI_API_KEY` or check your quota.

---

## 🧠 How it works

### 1) Sampling

`src/sample_data.py` loads the large CSV in chunks, drops incomplete rows, and uses reservoir sampling to keep memory usage low.

### 2) Embedding + Storage

`src/ingest.py` uses `sentence-transformers/all-MiniLM-L6-v2` to embed each question (title + body), then stores the vectors in Endee under the configured collection.

### 3) Retrieval

`src/retrieve.py` embeds the user query and uses Endee’s `query()` to fetch the most similar docs.

### 4) RAG

`src/rag.py` builds a prompt from retrieved documents and calls OpenAI (if `OPENAI_API_KEY` is set). Otherwise, it returns a simple fallback answer built from the retrieved snippets.

### 5) UI

`app.py` provides a simple Streamlit UI to enter queries, view top results, and see generated answers.

---

## 🧩 Configuration

Configuration values are in `src/config.py` and can be overridden with environment variables:

- `ENDEE_API_URL` – Endee API URL (default: `http://localhost:8000`)
- `ENDEE_API_TOKEN` – Endee token (if required)
- `EMBEDDING_MODEL` – Embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `TOP_K` – Number of results to retrieve
- `OPENAI_API_KEY` – OpenAI API key for RAG

---

## ✅ Example query

In the Streamlit app, try simple questions like:

- "How do I merge two pandas DataFrames?"
- "What is the difference between list and tuple in Python?"
- "How do I fix a null pointer exception in Java?"

---

## 📌 Where Endee is used

Endee is used as the vector store:

1. `src/ingest.py` creates an Endee index and upserts embeddings + metadata.
2. `src/retrieve.py` queries Endee with an embedding for semantic search.
3. `src/rag.py` uses the retrieved documents to build context for generation.
