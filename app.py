import os
import httpx
import streamlit as st
from dotenv import load_dotenv

# Network SSL patch
def _patch_httpx_ssl():
    _orig = httpx.Client.__init__
    def _no_ssl(self, **kwargs):
        kwargs["verify"] = False
        _orig(self, **kwargs)
    httpx.Client.__init__ = _no_ssl
_patch_httpx_ssl()

# Module imports
from document_processor import process_pdf
from vector_database import load_embedding_model, build_faiss_index, retrieve_relevant_chunks
from rag_engine import initialize_gemini, generate_answer

st.set_page_config(page_title="CareerCompass – AI Career Guidance", page_icon="🧭", layout="wide")

load_dotenv()
GEMINI_API_KEY = os.getenv("GROQ_API_KEY", "")

def init_session_state():
    defaults = {
        "faiss_index": None,
        "all_chunks": None,
        "embedding_model": None,
        "gemini_model": None,
        "last_answer": None,
        "last_question": None,
        "last_sources": None,
        "processed_files": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

def load_models_if_needed():
    if st.session_state["embedding_model"] is None:
        with st.spinner("Loading embedding model..."):
            st.session_state["embedding_model"] = load_embedding_model()
    if st.session_state["gemini_model"] is None:
        if not GEMINI_API_KEY:
            return False
        try:
            with st.spinner("Connecting to Gemini API..."):
                st.session_state["gemini_model"] = initialize_gemini(GEMINI_API_KEY)
        except Exception as e:
            st.error(f"Failed to initialise Gemini: {e}")
            return False
    return True

def process_uploaded_files(uploaded_files):
    new_files = [f for f in uploaded_files if f.name not in st.session_state["processed_files"]]
    if not new_files:
        return True
    all_new_chunks = []
    for pdf_file in new_files:
        try:
            chunks = process_pdf(pdf_file)
            if chunks:
                all_new_chunks.extend(chunks)
                st.session_state["processed_files"].append(pdf_file.name)
        except Exception as e:
            st.error(f"Error processing '{pdf_file.name}': {e}")
    if not all_new_chunks:
        return False
    existing_chunks = st.session_state["all_chunks"] or []
    combined_chunks = existing_chunks + all_new_chunks
    try:
        faiss_index, indexed_chunks = build_faiss_index(combined_chunks, st.session_state["embedding_model"])
        st.session_state["faiss_index"] = faiss_index
        st.session_state["all_chunks"] = indexed_chunks
    except Exception as e:
        st.error(str(e))
        return False
    return True

def format_sources(context_chunks):
    seen = set()
    sources = []
    for chunk in context_chunks:
        label = f"{chunk['source']} – Page {chunk['page']}"
        if label not in seen:
            seen.add(label)
            sources.append(label)
    return sources

def render_sidebar():
    with st.sidebar:
        st.title("CareerCompass")
        st.caption("AI Career Guidance Assistant")
        st.divider()
        st.subheader("🔑 API Key Status")
        if GEMINI_API_KEY:
            st.success("API key loaded ✓")
        else:
            st.error("GROQ_API_KEY not found in .env")
        st.divider()
        uploaded_files = st.file_uploader("Upload Career Documents", type=["pdf"], accept_multiple_files=True)
        if st.button("Clear All & Reset", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    return uploaded_files

def render_main(uploaded_files):
    st.title("🧭 CareerCompass")
    st.caption("AI-Powered Career Guidance using RAG")
    st.divider()

    if not GEMINI_API_KEY:
        st.warning("Please add your API key to .env file to continue.")
        st.stop()
    if not load_models_if_needed():
        st.error("Failed to load models.")
        st.stop()
    if not uploaded_files:
        st.info("👈 Upload PDF documents in the sidebar to get started.")
        st.stop()
    if not process_uploaded_files(uploaded_files):
        st.error("Could not extract text from uploaded files.")
        st.stop()

    question = st.text_input("Ask a Career Question:")
    if st.button("Get Guidance", type="primary"):
        if question.strip():
            with st.spinner("Finding answer..."):
                context_chunks = retrieve_relevant_chunks(
                    query=question,
                    faiss_index=st.session_state["faiss_index"],
                    chunks=st.session_state["all_chunks"],
                    model=st.session_state["embedding_model"],
                    top_k=3
                )
                answer = generate_answer(
                    question=question,
                    context_chunks=context_chunks,
                    model=st.session_state["gemini_model"]
                )
                st.session_state["last_question"] = question
                st.session_state["last_answer"] = answer
                st.session_state["last_sources"] = format_sources(context_chunks)

    if st.session_state.get("last_answer"):
        st.markdown("### Answer")
        st.write(st.session_state["last_answer"])
        if st.session_state.get("last_sources"):
            with st.expander("Sources"):
                for s in st.session_state["last_sources"]:
                    st.write(f"- {s}")

def main():
    uploaded_files = render_sidebar()
    render_main(uploaded_files)

if __name__ == "__main__":
    main()