# CareerCompass – AI Career Guidance Assistant
## Project Specification Document

---

## 1. Project Overview

**Project Name:** CareerCompass – AI Career Guidance Assistant  
**Subject:** Generative AI  
**Type:** Mini-Project (Team of Four)  
**Purpose:** Help students explore career paths by asking natural language questions about uploaded career-related PDF documents.

---

## 2. Objectives

- Allow students to upload one or more career-related PDF documents (job descriptions, career guides, skill requirements, qualification documents).
- Extract, clean, and chunk text from uploaded PDFs for efficient retrieval.
- Convert text chunks into vector embeddings and store them in a FAISS index for semantic search.
- Retrieve the most relevant chunks for a given student question.
- Use Google Gemini LLM to generate a clear, grounded answer based on the retrieved content.
- Provide a simple, beginner-friendly Streamlit web interface.

---

## 3. Key Features

| Feature | Description |
|---|---|
| PDF Upload | Students can upload one or multiple PDFs through the UI |
| Text Extraction | Extract raw text from each PDF page |
| Text Chunking | Split text into overlapping chunks for better retrieval |
| Semantic Search | Find the most relevant chunks using cosine similarity on embeddings |
| AI Answer Generation | Use Gemini to generate answers grounded in retrieved chunks |
| Source Attribution | Show which document/page each retrieved chunk came from |
| Multi-document Support | Query across all uploaded documents simultaneously |

---

## 4. Sample Questions the System Must Answer

- What skills are required for a Python Developer?
- What technologies should I learn for Data Analyst roles?
- What is the difference between a Java Developer and a Python Developer?
- What qualifications are required for an AI Engineer?

---

## 5. Overall RAG Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        STUDENT (Browser)                        │
│                  Uploads PDFs + Asks Questions                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    app.py  (Member 4)                           │
│              Streamlit Frontend + Orchestration                 │
└──────┬──────────────────┬──────────────────┬────────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌─────────────┐  ┌─────────────────┐  ┌─────────────────┐
│document_    │  │ vector_database │  │  rag_engine.py  │
│processor.py │  │     .py         │  │   (Member 3)    │
│ (Member 1)  │  │  (Member 2)     │  │                 │
│             │  │                 │  │  Gemini LLM     │
│ PDF Extract │  │ Embeddings      │  │  Prompt Build   │
│ Text Clean  │  │ FAISS Store     │  │  Answer Gen     │
│ Chunking    │  │ Retrieval       │  │                 │
└──────┬──────┘  └────────┬────────┘  └────────┬────────┘
       │                  │                     │
       │   List[Chunk]    │   List[Chunk]        │  str (answer)
       └──────────────────┴─────────────────────┘

Data Flow:
1. PDF uploaded → document_processor extracts + chunks → List[Chunk]
2. List[Chunk] → vector_database builds FAISS index
3. User question → vector_database retrieves top-k Chunks
4. Retrieved Chunks + question → rag_engine generates answer
5. Answer displayed in app.py
```

---

## 6. Technology Stack

| Technology | Version (Recommended) | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| Streamlit | ≥ 1.32.0 | Web frontend |
| PyPDF2 | ≥ 3.0.0 | PDF text extraction |
| sentence-transformers | ≥ 2.6.0 | Text embeddings |
| faiss-cpu | ≥ 1.8.0 | Vector storage and retrieval |
| google-generativeai | ≥ 0.5.0 | Gemini LLM API |
| numpy | ≥ 1.26.0 | Array operations for FAISS |
| python-dotenv | ≥ 1.0.0 | Load API key from .env file |

---

## 7. Team Responsibilities

### Member 1 – document_processor.py
- Extract raw text from uploaded PDF files using PyPDF2.
- Clean extracted text (remove excessive whitespace, special characters).
- Split cleaned text into overlapping chunks of configurable size.
- Attach metadata (source filename, page number) to every chunk.
- **Output:** `List[Chunk]` — the shared data structure consumed by all other modules.

### Member 2 – vector_database.py
- Load a Sentence Transformer model to generate embeddings.
- Accept a `List[Chunk]` and build a FAISS index from their embeddings.
- Accept a query string and return the top-k most relevant `List[Chunk]`.
- Support saving and loading the FAISS index to/from disk (optional).

### Member 3 – rag_engine.py
- Initialize and authenticate the Google Gemini API using an API key.
- Accept a user question and a `List[Chunk]` (retrieved context).
- Build a structured prompt that combines the question with retrieved context.
- Call Gemini and return a clean answer string.
- Handle API errors gracefully.

### Member 4 – app.py
- Build the Streamlit UI: file uploader, question input, answer display.
- Orchestrate the pipeline: call document_processor → vector_database → rag_engine.
- Display source attribution (filename, page) alongside the answer.
- Manage session state for the FAISS index across multiple questions.
- Load the Gemini API key from environment variables / .env file.

---

## 8. Project Folder Structure

```
CareerCompass/
│
├── document_processor.py      # Member 1
├── vector_database.py         # Member 2
├── rag_engine.py              # Member 3
├── app.py                     # Member 4
│
├── requirements.txt           # All dependencies
├── .env                       # API key (NOT committed to Git)
├── .gitignore                 # Exclude .env and __pycache__
│
├── PROJECT_SPEC.md            # This document
└── INTERFACES.md              # Interface contract document
```

---

## 9. Required Dependencies (requirements.txt)

```
streamlit>=1.32.0
PyPDF2>=3.0.0
sentence-transformers>=2.6.0
faiss-cpu>=1.8.0
google-generativeai>=0.5.0
numpy>=1.26.0
python-dotenv>=1.0.0
```

---

## 10. Environment Setup

1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```
5. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## 11. Integration and Testing Sequence

Each member should test their module independently before integration.

| Step | Who | Action |
|---|---|---|
| 1 | Member 1 | Test `document_processor.py` with a sample PDF. Verify `List[Chunk]` output. |
| 2 | Member 2 | Test `vector_database.py` using dummy `List[Chunk]` data. Verify retrieval works. |
| 3 | Member 3 | Test `rag_engine.py` with a hardcoded question and dummy chunks. Verify Gemini returns an answer. |
| 4 | Member 4 | Integrate all three modules in `app.py`. Run full end-to-end test with a real PDF. |

---

## 12. Git Workflow

Each member works only on their assigned file.

```bash
# Each member clones the repo
git clone <repo-url>

# Create a personal branch
git checkout -b member1-document-processor   # adjust per member

# Work on your file, then commit
git add document_processor.py
git commit -m "Member 1: Add PDF extraction, cleaning and chunking"

# Push to GitHub
git push origin member1-document-processor

# Create a Pull Request to merge into main
```

**Rule:** Never edit another member's file in your branch.

---

## 13. .gitignore (Recommended)

```
.env
__pycache__/
*.pyc
venv/
*.faiss
```
