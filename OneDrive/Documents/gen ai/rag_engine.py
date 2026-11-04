# =============================================================================
# CareerCompass – AI Career Guidance Assistant
# =============================================================================
# MODULE  : rag_engine.py
# MEMBER  : Member 3
# PURPOSE : Integrate with Groq LLM API to generate career guidance answers.
#           Builds a structured RAG prompt from retrieved context chunks and
#           returns a grounded answer string.
#
# EXPORTS (used by app.py):
#   - initialize_gemini(api_key)                        -> Groq client
#   - generate_answer(question, context_chunks, model)  -> str
#
# INTERNAL HELPER:
#   - build_prompt(question, context_chunks)            -> str
#
# DEPENDENCIES:
#   - groq           (pip install groq)
#   - python-dotenv  (pip install python-dotenv)  [for __main__ test only]
#
# NOTE: Uses Groq API with llama-3.3-70b-versatile model.
#       Function names kept identical to INTERFACES.md so app.py needs
#       no changes.
# =============================================================================

import os
import httpx
from groq import Groq

# ---------------------------------------------------------------------------
# SSL FIX for college/corporate networks with SSL inspection
# Patches httpx globally to disable certificate verification.
# Must stay at the top of the file before any network calls.
# ---------------------------------------------------------------------------
def _patch_httpx_ssl():
    _orig_client = httpx.Client.__init__
    def _client_no_ssl(self, **kwargs):
        kwargs["verify"] = False
        _orig_client(self, **kwargs)
    httpx.Client.__init__ = _client_no_ssl

    _orig_async = httpx.AsyncClient.__init__
    def _async_no_ssl(self, **kwargs):
        kwargs["verify"] = False
        _orig_async(self, **kwargs)
    httpx.AsyncClient.__init__ = _async_no_ssl

_patch_httpx_ssl()


# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

GROQ_MODEL_NAME = "qwen/qwen3.8-27b"

# Max characters of context to include in the prompt (~3000 tokens)
MAX_CONTEXT_CHARS = 12000


# -----------------------------------------------------------------------------
# FUNCTION 1: initialize_gemini
# (name kept as per INTERFACES.md so app.py import works without changes)
# -----------------------------------------------------------------------------

def initialize_gemini(api_key: str) -> Groq:
    """
    Create and return a Groq client.

    Args:
        api_key (str): Groq API key from https://console.groq.com
                       Never hardcode — read from .env file.

    Returns:
        Groq: Configured client ready to generate content.

    Raises:
        ValueError: If api_key is empty or None.
    """
    if not api_key or not api_key.strip():
        raise ValueError(
            "[rag_engine] GROQ_API_KEY is missing or empty. "
            "Add it to your .env file as: GROQ_API_KEY=your_key_here"
        )

    try:
        client = Groq(api_key=api_key)
        print(f"[rag_engine] Groq client initialised (model: {GROQ_MODEL_NAME}).")
        return client
    except Exception as e:
        print(f"[rag_engine] Error initialising Groq client: {e}")
        raise


# -----------------------------------------------------------------------------
# FUNCTION 2: build_prompt  (internal helper)
# -----------------------------------------------------------------------------

def build_prompt(question: str, context_chunks: list[dict]) -> str:
    """
    Build a structured RAG prompt combining retrieved context with the question.

    Args:
        question       (str)       : Student's career question.
        context_chunks (list[dict]): Retrieved Chunk dicts from vector_database.

    Returns:
        str: Fully formatted prompt string ready to send to the LLM.
    """
    if not context_chunks:
        context_block = "No relevant context was found in the uploaded documents."
    else:
        context_parts = []
        total_chars   = 0

        for chunk in context_chunks:
            label      = f"[Source: {chunk['source']} | Page: {chunk['page']}]"
            chunk_text = chunk["text"].strip()
            entry      = f"{label}\n{chunk_text}"

            if total_chars + len(entry) > MAX_CONTEXT_CHARS:
                context_parts.append(
                    "[Additional context omitted to stay within prompt length limits.]"
                )
                break

            context_parts.append(entry)
            total_chars += len(entry)

        context_block = "\n\n".join(context_parts)

    prompt = (
        "You are CareerCompass, an AI career guidance assistant.\n"
        "Use ONLY the information provided in the context below to answer the student's question.\n"
        'If the answer is not found in the context, say '
        '"I could not find this information in the uploaded documents."\n'
        "Be clear, concise, and helpful. Use bullet points where appropriate.\n"
        "\n"
        "Context:\n"
        "---\n"
        f"{context_block}\n"
        "---\n"
        "\n"
        f"Student Question: {question.strip()}\n"
        "\n"
        "Answer:"
    )

    return prompt


# -----------------------------------------------------------------------------
# FUNCTION 3: generate_answer  ← MAIN ENTRY POINT (called by app.py)
# -----------------------------------------------------------------------------

def generate_answer(question: str,
                    context_chunks: list[dict],
                    model: Groq) -> str:
    """
    Generate a grounded career guidance answer using Groq LLM.

    Args:
        question       (str)   : Student's question.
        context_chunks (list[dict]): Top-k retrieved chunks from vector_database.
        model          (Groq)  : Initialised client from initialize_gemini().

    Returns:
        str: Generated answer, or a safe error/fallback string. Never raises.
    """
    if not question or not question.strip():
        return "Please enter a question to get career guidance."

    if not context_chunks:
        return (
            "I could not find this information in the uploaded documents. "
            "Please upload relevant career documents and try rephrasing your question."
        )

    try:
        prompt = build_prompt(question, context_chunks)

        response = model.chat.completions.create(
            model    = GROQ_MODEL_NAME,
            messages = [
                {
                    "role"   : "user",
                    "content": prompt
                }
            ],
            temperature = 0.3,    # Lower = more factual, less creative
            max_tokens  = 1024,
        )

        answer = response.choices[0].message.content.strip()

        if not answer:
            return "The model returned an empty response. Please try rephrasing your question."

        return answer

    except Exception as e:
        error_message = str(e)
        print(f"[rag_engine] Groq API error: {error_message}")

        if "401" in error_message or "invalid_api_key" in error_message.lower():
            return "Error: Invalid Groq API key. Check your .env file."
        elif "429" in error_message or "rate_limit" in error_message.lower():
            return "Error: Groq rate limit reached. Please wait a moment and try again."
        elif "timeout" in error_message.lower():
            return "Error: Request timed out. Check your internet connection and try again."
        else:
            return f"Error: Could not generate answer. Details: {error_message}"


# =============================================================================
# INDEPENDENT TEST BLOCK
# =============================================================================
# Prerequisites:
#   - .env file with GROQ_API_KEY=your_key_here
# Run:
#   python rag_engine.py
# =============================================================================

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        print("[TEST] ERROR: GROQ_API_KEY not found in .env file.")
        sys.exit(1)

    dummy_chunks_python = [
        {"text": "A Python Developer must have strong knowledge of Python, Django, Flask, REST APIs, and SQL databases.",
         "source": "python_developer_guide.pdf", "page": 1, "chunk_index": 0},
        {"text": "Version control using Git is essential. AWS or GCP experience is a plus.",
         "source": "python_developer_guide.pdf", "page": 2, "chunk_index": 1},
    ]

    dummy_chunks_ai = [
        {"text": "Qualifications for an AI Engineer include a Bachelor's in CS or Data Science. Postgrad or AI/ML certs preferred.",
         "source": "ai_engineer_requirements.pdf", "page": 1, "chunk_index": 0},
        {"text": "AI Engineers need expertise in TensorFlow, PyTorch, and model deployment with Docker.",
         "source": "ai_engineer_requirements.pdf", "page": 2, "chunk_index": 1},
    ]

    print("\n[TEST] Initialising Groq client ...\n")
    groq_client = initialize_gemini(api_key)

    test_cases = [
        ("What skills are required for a Python Developer?",     dummy_chunks_python),
        ("What qualifications are required for an AI Engineer?", dummy_chunks_ai),
    ]

    print("\n" + "=" * 65)
    print("RAG ENGINE TEST RESULTS")
    print("=" * 65)

    for question, chunks in test_cases:
        print(f"\n Question: {question}")
        print("-" * 65)
        answer = generate_answer(question, chunks, groq_client)
        print(f" Answer:\n{answer}\n")
