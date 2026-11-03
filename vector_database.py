# =============================================================================
# CareerCompass – AI Career Guidance Assistant
# =============================================================================
# MODULE  : vector_database.py
# MEMBER  : Member 2
# PURPOSE : Generate sentence embeddings for text chunks, store them in a
#           FAISS vector index, and retrieve the most relevant chunks for a
#           given user query using semantic similarity search.
#
# EXPORTS (used by app.py):
#   - load_embedding_model(model_name)                          -> SentenceTransformer
#   - build_faiss_index(chunks, model)                          -> (faiss.Index, list[dict])
#   - retrieve_relevant_chunks(query, faiss_index, chunks,
#                              model, top_k)                    -> list[dict]
#
# DEPENDENCIES:
#   - sentence-transformers  (pip install sentence-transformers)
#   - faiss-cpu              (pip install faiss-cpu)
#   - numpy                  (pip install numpy)
#
# INTERFACE CONTRACT:
#   Receives and returns the shared Chunk dict format:
#       {
#           "text"        : str  – chunk content
#           "source"      : str  – original PDF filename
#           "page"        : int  – 1-indexed page number
#           "chunk_index" : int  – sequential chunk number
#       }
#   This module does NOT import from any other CareerCompass module.
#   It can be tested independently with dummy chunk data (see __main__ block).
# =============================================================================

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# -----------------------------------------------------------------------------
# FUNCTION 1: load_embedding_model
# -----------------------------------------------------------------------------

def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load and return a Sentence Transformer embedding model.

    The default model "all-MiniLM-L6-v2" is a good balance between speed
    and accuracy for a college-level project:
        - Model size  : ~80 MB
        - Embedding dim: 384
        - Download    : automatic on first use (cached after that)

    Call this function ONCE and pass the returned model object to both
    build_faiss_index() and retrieve_relevant_chunks() to avoid reloading
    the model on every call.

    Args:
        model_name (str): HuggingFace model identifier.
                          Default: "all-MiniLM-L6-v2"

    Returns:
        SentenceTransformer: A loaded model ready to encode text.

    Example:
        model = load_embedding_model()
        # reuse `model` for all subsequent calls
    """
    print(f"[vector_database] Loading embedding model: '{model_name}' ...")
    model = SentenceTransformer(model_name)
    print(f"[vector_database] Model loaded successfully. "
          f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


# -----------------------------------------------------------------------------
# FUNCTION 2: build_faiss_index
# -----------------------------------------------------------------------------

def build_faiss_index(chunks: list[dict],
                      model: SentenceTransformer) -> tuple[faiss.Index, list[dict]]:
    """
    Generate embeddings for all chunks and build a FAISS index for fast
    nearest-neighbour search.

    How it works:
        1. Extract the "text" field from each chunk.
        2. Encode all texts in one batch using the Sentence Transformer.
        3. Normalise the vectors to unit length (enables cosine similarity
           via dot product, which is faster than L2 on normalised vectors).
        4. Build a faiss.IndexFlatIP (Inner Product) index and add all vectors.

    The order of rows in the FAISS index mirrors the order of `chunks` exactly.
    retrieve_relevant_chunks() relies on this alignment.

    Args:
        chunks (list[dict]): List of Chunk dicts from document_processor.
                             Must not be empty.
        model  (SentenceTransformer): Loaded model from load_embedding_model().

    Returns:
        tuple[faiss.Index, list[dict]]:
            - faiss_index : The built FAISS index (IndexFlatIP).
            - chunks      : The same list passed in, returned for convenience
                            so app.py can store both together in session state.

    Raises:
        ValueError: If `chunks` is empty (nothing to index).

    Example:
        model = load_embedding_model()
        faiss_index, chunks = build_faiss_index(all_chunks, model)
    """
    if not chunks:
        raise ValueError(
            "[vector_database] No chunks provided to build index. "
            "Make sure the PDF was processed successfully before calling this function."
        )

    print(f"[vector_database] Building FAISS index for {len(chunks)} chunk(s) ...")

    # Extract plain text strings in the same order as the chunks list
    texts = [chunk["text"] for chunk in chunks]

    # Encode all texts into embedding vectors (returns a numpy float32 array)
    # show_progress_bar=True helps when there are many chunks
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Cast to float32 explicitly – FAISS requires float32
    embeddings = embeddings.astype(np.float32)

    # Normalise vectors to unit length so that Inner Product == cosine similarity
    faiss.normalize_L2(embeddings)

    # Embedding dimension (e.g. 384 for all-MiniLM-L6-v2)
    embedding_dim = embeddings.shape[1]

    # IndexFlatIP: exact (brute-force) cosine similarity search.
    # For a college project with a few hundred/thousand chunks this is fast enough.
    faiss_index = faiss.IndexFlatIP(embedding_dim)
    faiss_index.add(embeddings)

    print(f"[vector_database] FAISS index built. "
          f"Total vectors stored: {faiss_index.ntotal}")

    return faiss_index, chunks


# -----------------------------------------------------------------------------
# FUNCTION 3: retrieve_relevant_chunks
# -----------------------------------------------------------------------------

def retrieve_relevant_chunks(query: str,
                              faiss_index: faiss.Index,
                              chunks: list[dict],
                              model: SentenceTransformer,
                              top_k: int = 5) -> list[dict]:
    """
    Find the top-k most semantically relevant chunks for a given query.

    Steps:
        1. Encode the query into an embedding vector.
        2. Normalise the query vector (matches the normalisation done at index time).
        3. Search the FAISS index for the nearest `top_k` vectors.
        4. Map the returned FAISS row indices back to the original chunk dicts.
        5. Return chunks ordered by relevance (highest cosine similarity first).

    Args:
        query       (str)              : The student's question string.
        faiss_index (faiss.Index)      : The FAISS index built by build_faiss_index().
        chunks      (list[dict])       : The same chunk list passed to build_faiss_index()
                                         (order must be identical).
        model       (SentenceTransformer): The same model used to build the index.
        top_k       (int)              : Number of chunks to return. Default: 5.
                                         If top_k > len(chunks), all chunks are returned.

    Returns:
        list[dict]: Up to top_k Chunk dicts, ordered from most to least relevant.
                    Returns an empty list if the index is empty or query is blank.

    Example:
        results = retrieve_relevant_chunks("Python developer skills",
                                           faiss_index, chunks, model, top_k=3)
        for r in results:
            print(r["source"], r["page"], r["text"][:100])
    """
    if not query or not query.strip():
        print("[vector_database] Warning: empty query received. Returning no results.")
        return []

    if faiss_index.ntotal == 0:
        print("[vector_database] Warning: FAISS index is empty. Returning no results.")
        return []

    # Guard: top_k cannot exceed the number of indexed vectors
    actual_top_k = min(top_k, faiss_index.ntotal)
    if actual_top_k < top_k:
        print(f"[vector_database] Note: top_k={top_k} exceeds available chunks "
              f"({faiss_index.ntotal}). Returning all {actual_top_k} chunk(s).")

    # Step 1 & 2: Encode and normalise the query vector
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype(np.float32)

    faiss.normalize_L2(query_embedding)

    # Step 3: Search — returns arrays of shape (1, actual_top_k)
    # scores: cosine similarity scores (higher = more relevant)
    # indices: row positions in the FAISS index
    scores, indices = faiss_index.search(query_embedding, actual_top_k)

    # Step 4 & 5: Map indices back to chunk dicts, skip any invalid (-1) indices
    relevant_chunks = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            # FAISS returns -1 when there are not enough vectors to fill top_k
            continue
        chunk = chunks[idx].copy()   # Return a copy to avoid mutating the stored chunk
        chunk["_score"] = float(score)   # Attach similarity score for optional display
        relevant_chunks.append(chunk)

    return relevant_chunks


# =============================================================================
# INDEPENDENT TEST BLOCK
# =============================================================================
# Run this file directly to test Member 2's module without needing any other
# module from the project.
#
# Usage:
#   1. Open a terminal in the project folder and run:
#        python vector_database.py
#   2. Expected output:
#        - Model load confirmation with embedding dimension.
#        - FAISS index build confirmation with vector count.
#        - Top-k retrieved chunks printed for three different test queries.
# =============================================================================

if __name__ == "__main__":

    # --- Dummy chunk data (simulates output from document_processor.py) ---
    dummy_chunks = [
        {
            "text"        : "A Python Developer must have strong knowledge of Python, "
                            "Django, Flask, REST APIs, and SQL databases. "
                            "Familiarity with NumPy and Pandas is also expected.",
            "source"      : "python_developer_guide.pdf",
            "page"        : 1,
            "chunk_index" : 0
        },
        {
            "text"        : "Data Analysts are expected to be proficient in SQL, "
                            "Excel, Power BI, and Tableau. "
                            "Python knowledge with Pandas and Matplotlib is a strong advantage.",
            "source"      : "data_analyst_roles.pdf",
            "page"        : 2,
            "chunk_index" : 1
        },
        {
            "text"        : "An AI Engineer should have expertise in Machine Learning, "
                            "Deep Learning frameworks such as TensorFlow and PyTorch, "
                            "and experience with model deployment using Docker and cloud platforms.",
            "source"      : "ai_engineer_requirements.pdf",
            "page"        : 1,
            "chunk_index" : 2
        },
        {
            "text"        : "A Java Developer needs solid understanding of Core Java, "
                            "Spring Boot, Hibernate, and RESTful web services. "
                            "Knowledge of Maven and CI/CD pipelines is highly valued.",
            "source"      : "java_developer_guide.pdf",
            "page"        : 3,
            "chunk_index" : 3
        },
        {
            "text"        : "Qualifications for an AI Engineer typically include a Bachelor's "
                            "degree in Computer Science, Data Science, or a related field. "
                            "A postgraduate degree or specialised certifications in AI/ML are preferred.",
            "source"      : "ai_engineer_requirements.pdf",
            "page"        : 2,
            "chunk_index" : 4
        },
        {
            "text"        : "The difference between a Java Developer and a Python Developer "
                            "lies mainly in their primary language and ecosystem. "
                            "Java is preferred for enterprise-grade applications, "
                            "while Python dominates in data science and automation.",
            "source"      : "career_comparison.pdf",
            "page"        : 4,
            "chunk_index" : 5
        },
    ]

    # Step 1: Load the embedding model
    embedding_model = load_embedding_model()

    # Step 2: Build the FAISS index
    index, stored_chunks = build_faiss_index(dummy_chunks, embedding_model)

    # Step 3: Run test queries
    test_queries = [
        "What skills are required for a Python Developer?",
        "What technologies should I learn for Data Analyst roles?",
        "What qualifications are required for an AI Engineer?",
    ]

    print("\n" + "=" * 60)
    print("RETRIEVAL TEST RESULTS")
    print("=" * 60)

    for query in test_queries:
        print(f"\n Query : {query}")
        print("-" * 60)
        results = retrieve_relevant_chunks(
            query      = query,
            faiss_index= index,
            chunks     = stored_chunks,
            model      = embedding_model,
            top_k      = 2
        )
        for rank, chunk in enumerate(results, start=1):
            print(f"  Rank {rank}:")
            print(f"    Source : {chunk['source']}  |  Page: {chunk['page']}")
            print(f"    Score  : {chunk.get('_score', 'N/A'):.4f}")
            print(f"    Text   : {chunk['text'][:120]}...")
        print()