# CareerCompass – Interface Contract Document

> This document defines the exact function signatures, shared data structures, and communication contracts between all four modules.  
> **All members must follow this document exactly.** Any deviation will cause import or runtime errors in `app.py`.

---

## 1. Shared Data Structure – The `Chunk` Dictionary

Every module passes data using a standard Python dictionary called a **Chunk**.  
No custom classes are used — a plain `dict` keeps it beginner-friendly and avoids import dependencies between modules.

### Chunk Schema

```python
chunk: dict = {
    "text":        str,   # The actual text content of this chunk
    "source":      str,   # Filename of the PDF this chunk came from
    "page":        int,   # Page number (1-indexed) within the PDF
    "chunk_index": int    # Sequential index of this chunk within its document
}
```

### Example Chunk

```python
{
    "text": "A Python Developer must have strong knowledge of Python, Django, REST APIs, and SQL databases.",
    "source": "python_developer_guide.pdf",
    "page": 3,
    "chunk_index": 7
}
```

### Rules

- `text` must be a non-empty string.
- `source` must be the original filename (e.g. `"job_roles.pdf"`), not a full path.
- `page` starts at **1**, not 0.
- `chunk_index` starts at **0** and increments across all chunks from one document.
- All four modules use `List[dict]` (i.e. `List[Chunk]`) as the shared collection type.

---

## 2. Module 1 – `document_processor.py` (Member 1)

**Responsibility:** PDF extraction, text cleaning, and chunking.

---

### Function 1: `extract_text_from_pdf`

```python
def extract_text_from_pdf(pdf_file) -> list[dict]:
```

| Item | Detail |
|---|---|
| **Arguments** | `pdf_file` – a file-like object (e.g. from `open()` or Streamlit's `UploadedFile`) |
| **Returns** | `list[dict]` – list of raw page dicts before chunking |
| **Each dict contains** | `{"text": str, "source": str, "page": int}` |
| **Notes** | Use `PyPDF2.PdfReader`. Skip pages with no extractable text. `source` is obtained from `pdf_file.name`. |

**Example usage:**
```python
with open("career_guide.pdf", "rb") as f:
    pages = extract_text_from_pdf(f)
# pages[0] → {"text": "...", "source": "career_guide.pdf", "page": 1}
```

---

### Function 2: `clean_text`

```python
def clean_text(text: str) -> str:
```

| Item | Detail |
|---|---|
| **Arguments** | `text` – raw string extracted from a PDF page |
| **Returns** | `str` – cleaned string |
| **Notes** | Remove excessive newlines, leading/trailing whitespace, non-printable characters. Do not remove punctuation. |

---

### Function 3: `chunk_text`

```python
def chunk_text(text: str, source: str, page: int,
               chunk_size: int = 500, overlap: int = 50) -> list[dict]:
```

| Item | Detail |
|---|---|
| **Arguments** | `text` – cleaned page text; `source` – filename; `page` – page number; `chunk_size` – max characters per chunk (default 500); `overlap` – overlap in characters between consecutive chunks (default 50) |
| **Returns** | `list[dict]` – list of Chunk dicts with all four keys populated |
| **Notes** | Split by character count. Each chunk gets a `chunk_index` starting from 0 within this call. `chunk_index` should be globally unique per document — Member 4 will offset if needed. |

---

### Function 4: `process_pdf` *(main entry point)*

```python
def process_pdf(pdf_file, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
```

| Item | Detail |
|---|---|
| **Arguments** | `pdf_file` – file-like object; `chunk_size` – characters per chunk; `overlap` – overlap characters |
| **Returns** | `list[dict]` – complete list of Chunk dicts for the entire PDF |
| **Notes** | Calls `extract_text_from_pdf` → `clean_text` → `chunk_text` for every page. Concatenates all chunk lists into one. This is the only function `app.py` calls from this module. |

**Example usage (for independent testing):**
```python
# Run: python document_processor.py
if __name__ == "__main__":
    with open("sample.pdf", "rb") as f:
        chunks = process_pdf(f)
    print(f"Total chunks: {len(chunks)}")
    print("First chunk:", chunks[0])
```

---

## 3. Module 2 – `vector_database.py` (Member 2)

**Responsibility:** Embedding generation, FAISS indexing, and semantic retrieval.

---

### Function 1: `load_embedding_model`

```python
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
```

| Item | Detail |
|---|---|
| **Arguments** | `model_name` – HuggingFace model name (default: `"all-MiniLM-L6-v2"`) |
| **Returns** | A loaded `SentenceTransformer` model object |
| **Notes** | Call once and reuse. The default model is small (80 MB) and suitable for a college project. |

---

### Function 2: `build_faiss_index`

```python
def build_faiss_index(chunks: list[dict], model: SentenceTransformer) -> tuple[faiss.Index, list[dict]]:
```

| Item | Detail |
|---|---|
| **Arguments** | `chunks` – `List[Chunk]` from `document_processor`; `model` – loaded SentenceTransformer |
| **Returns** | `tuple` of `(faiss_index, chunks)` — the FAISS index and the original chunk list (order preserved) |
| **Notes** | Encode `chunk["text"]` for each chunk. Build a `faiss.IndexFlatL2` index. The chunk list order must match the FAISS index row order exactly. |

---

### Function 3: `retrieve_relevant_chunks`

```python
def retrieve_relevant_chunks(query: str, faiss_index: faiss.Index,
                              chunks: list[dict], model: SentenceTransformer,
                              top_k: int = 5) -> list[dict]:
```

| Item | Detail |
|---|---|
| **Arguments** | `query` – user's question string; `faiss_index` – built FAISS index; `chunks` – original chunk list (same order as when index was built); `model` – SentenceTransformer; `top_k` – number of results to return (default 5) |
| **Returns** | `list[dict]` – top-k most relevant Chunk dicts, ordered by relevance (most relevant first) |
| **Notes** | Encode the query, search the FAISS index, map result indices back to chunks. |

**Example usage (for independent testing):**
```python
# Run: python vector_database.py
if __name__ == "__main__":
    # Dummy chunks for testing without Member 1's module
    dummy_chunks = [
        {"text": "Python developers need NumPy and Pandas skills.", "source": "test.pdf", "page": 1, "chunk_index": 0},
        {"text": "Data Analysts use SQL and Power BI.", "source": "test.pdf", "page": 2, "chunk_index": 1},
        {"text": "AI Engineers require knowledge of TensorFlow and PyTorch.", "source": "test.pdf", "page": 3, "chunk_index": 2},
    ]
    model = load_embedding_model()
    index, chunks = build_faiss_index(dummy_chunks, model)
    results = retrieve_relevant_chunks("Python developer skills", index, chunks, model, top_k=2)
    for r in results:
        print(r)
```

---

## 4. Module 3 – `rag_engine.py` (Member 3)

**Responsibility:** Gemini API integration, prompt engineering, and answer generation.

---

### Function 1: `initialize_gemini`

```python
def initialize_gemini(api_key: str) -> genai.GenerativeModel:
```

| Item | Detail |
|---|---|
| **Arguments** | `api_key` – Gemini API key string |
| **Returns** | A configured `genai.GenerativeModel` instance |
| **Notes** | Call `genai.configure(api_key=api_key)`. Use model name `"gemini-1.5-flash"`. Do not hardcode the key. |

---

### Function 2: `build_prompt`

```python
def build_prompt(question: str, context_chunks: list[dict]) -> str:
```

| Item | Detail |
|---|---|
| **Arguments** | `question` – user's question string; `context_chunks` – `List[Chunk]` from retrieval |
| **Returns** | `str` – fully formatted prompt string ready to send to Gemini |
| **Notes** | Concatenate chunk texts with source/page labels. Instruct Gemini to answer only from the provided context. Keep prompt under 4000 tokens where possible. |

**Prompt Template (must follow this structure):**
```
You are CareerCompass, an AI career guidance assistant.
Use ONLY the information provided in the context below to answer the student's question.
If the answer is not found in the context, say "I could not find this information in the uploaded documents."

Context:
---
[Source: filename.pdf | Page: 3]
<chunk text here>

[Source: filename.pdf | Page: 5]
<chunk text here>
---

Student Question: <question here>

Answer:
```

---

### Function 3: `generate_answer`

```python
def generate_answer(question: str, context_chunks: list[dict],
                    model: genai.GenerativeModel) -> str:
```

| Item | Detail |
|---|---|
| **Arguments** | `question` – user's question; `context_chunks` – retrieved chunks; `model` – initialized Gemini model |
| **Returns** | `str` – the generated answer text |
| **Notes** | Calls `build_prompt` internally. Sends prompt to Gemini. Returns `response.text`. Wrap in try/except and return an error message string on failure. |

**Example usage (for independent testing):**
```python
# Run: python rag_engine.py
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    dummy_chunks = [
        {"text": "Python developers need skills in NumPy, Pandas, Django, and REST APIs.", "source": "guide.pdf", "page": 1, "chunk_index": 0},
        {"text": "Proficiency in SQL and data visualization tools like Matplotlib is also expected.", "source": "guide.pdf", "page": 2, "chunk_index": 1},
    ]
    api_key = os.getenv("GEMINI_API_KEY")
    model = initialize_gemini(api_key)
    answer = generate_answer("What skills does a Python developer need?", dummy_chunks, model)
    print("Answer:", answer)
```

---

## 5. Module 4 – `app.py` (Member 4)

**Responsibility:** Streamlit UI and pipeline orchestration.

**This module imports from the other three:**
```python
from document_processor import process_pdf
from vector_database import load_embedding_model, build_faiss_index, retrieve_relevant_chunks
from rag_engine import initialize_gemini, generate_answer
```

### Orchestration Flow (inside `app.py`)

```
User uploads PDF(s)
        │
        ▼
process_pdf(pdf_file)                      → all_chunks: list[dict]
        │
        ▼
load_embedding_model()                     → embedding_model
build_faiss_index(all_chunks, model)       → (faiss_index, all_chunks)
        │  [stored in st.session_state]
        ▼
User types a question
        │
        ▼
retrieve_relevant_chunks(question, ...)    → context_chunks: list[dict]
        │
        ▼
initialize_gemini(api_key)                 → gemini_model
generate_answer(question, context_chunks, gemini_model)  → answer: str
        │
        ▼
Display answer + source attribution in Streamlit
```

### Session State Keys (Member 4 must use exactly these names)

```python
st.session_state["faiss_index"]    # faiss.Index object
st.session_state["all_chunks"]     # list[dict] – full chunk list
st.session_state["embedding_model"] # SentenceTransformer object
st.session_state["gemini_model"]   # genai.GenerativeModel object
```

### No functions are exported from `app.py`
`app.py` is the entry point, not a library. No other module imports from it.

---

## 6. Module Communication Summary

```
document_processor.py
    └── process_pdf(pdf_file) ──────────────────────────► list[dict]  (Chunks)
                                                                │
                                                                ▼
vector_database.py
    ├── build_faiss_index(chunks, model) ───────────────► (faiss.Index, list[dict])
    └── retrieve_relevant_chunks(query, index, chunks, model, top_k)
                                                         ► list[dict]  (top-k Chunks)
                                                                │
                                                                ▼
rag_engine.py
    └── generate_answer(question, context_chunks, model) ► str  (answer)
                                                                │
                                                                ▼
app.py  ─────────────────────────────────────────────────► Displays answer to student
```

---

## 7. Error Handling Contracts

Each module must handle errors internally and return safe values:

| Module | Error Scenario | Expected Behavior |
|---|---|---|
| `document_processor` | PDF cannot be read / corrupted | Print warning, skip page, continue |
| `document_processor` | Page has no extractable text | Skip that page silently |
| `vector_database` | Empty chunk list passed | Raise `ValueError("No chunks provided to build index.")` |
| `vector_database` | `top_k` exceeds number of chunks | Return all available chunks instead |
| `rag_engine` | Gemini API error / timeout | Return `"Error: Could not generate answer. Please try again."` |
| `rag_engine` | Empty context chunks | Return `"I could not find this information in the uploaded documents."` |
| `app.py` | No PDF uploaded yet | Show a Streamlit info message, do not crash |
| `app.py` | Missing API key | Show a Streamlit error message, do not crash |

---

## 8. Dependency Map (Who Imports What)

```
app.py
  ├── imports from document_processor  → process_pdf
  ├── imports from vector_database     → load_embedding_model, build_faiss_index, retrieve_relevant_chunks
  └── imports from rag_engine          → initialize_gemini, generate_answer

document_processor.py  → imports: PyPDF2, re (standard library)
vector_database.py     → imports: sentence_transformers, faiss, numpy
rag_engine.py          → imports: google.generativeai
app.py                 → imports: streamlit, os, dotenv (+ the three modules above)
```

No module imports from another member's module except `app.py`.  
`document_processor`, `vector_database`, and `rag_engine` are fully independent of each other.

---

## 9. Integration Testing Sequence

### Step 1 – Member 1 (Independent)
```bash
# Place a sample PDF in the project folder
python document_processor.py
# Expected output: Total chunk count + first chunk printed
```

### Step 2 – Member 2 (Independent, uses dummy data)
```bash
python vector_database.py
# Expected output: Top-k retrieved chunks printed for a test query
```

### Step 3 – Member 3 (Independent, uses dummy data + real API key)
```bash
# Ensure .env file exists with GEMINI_API_KEY
python rag_engine.py
# Expected output: A Gemini-generated answer printed to console
```

### Step 4 – Member 4 (Full Integration)
```bash
streamlit run app.py
# Open browser at http://localhost:8501
# Upload a PDF → Ask a question → Verify answer appears
```

---

## 10. Quick Reference Card

| Module | File | Entry Point for app.py | Returns |
|---|---|---|---|
| Member 1 | `document_processor.py` | `process_pdf(pdf_file)` | `list[dict]` |
| Member 2 | `vector_database.py` | `build_faiss_index(chunks, model)` | `(faiss.Index, list[dict])` |
| Member 2 | `vector_database.py` | `retrieve_relevant_chunks(query, index, chunks, model)` | `list[dict]` |
| Member 3 | `rag_engine.py` | `initialize_gemini(api_key)` | `GenerativeModel` |
| Member 3 | `rag_engine.py` | `generate_answer(question, chunks, model)` | `str` |
| Member 4 | `app.py` | *(entry point, no exports)* | — |
