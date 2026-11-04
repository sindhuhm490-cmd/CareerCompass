# =============================================================================
# CareerCompass – AI Career Guidance Assistant
# =============================================================================
# MODULE  : document_processor.py
# MEMBER  : Member 1
# PURPOSE : Extract text from uploaded PDF files, clean the raw text,
#           and split it into overlapping chunks with metadata.
#
# EXPORTS (used by app.py):
#   - process_pdf(pdf_file, chunk_size, overlap) -> list[dict]
#
# DEPENDENCIES:
#   - PyPDF2  (pip install PyPDF2)
#   - re      (Python standard library)
#
# INTERFACE CONTRACT:
#   Every chunk returned is a plain dict with exactly these four keys:
#       {
#           "text"        : str  – chunk content
#           "source"      : str  – original PDF filename
#           "page"        : int  – 1-indexed page number
#           "chunk_index" : int  – 0-indexed sequential chunk number
#       }
# =============================================================================

import re
import PyPDF2


# -----------------------------------------------------------------------------
# FUNCTION 1: extract_text_from_pdf
# -----------------------------------------------------------------------------

def extract_text_from_pdf(pdf_file) -> list[dict]:
    """
    Read a PDF file and extract the raw text from each page.

    Args:
        pdf_file: A file-like object opened in binary mode.
                  Works with both open("file.pdf", "rb") and
                  Streamlit's UploadedFile objects (both have a .name attribute).

    Returns:
        list[dict]: One dict per page that contains extractable text.
                    Each dict has the keys: "text", "source", "page".
                    Pages with no extractable text are silently skipped.

    Example:
        with open("career_guide.pdf", "rb") as f:
            pages = extract_text_from_pdf(f)
        # pages[0] → {"text": "...", "source": "career_guide.pdf", "page": 1}
    """
    pages = []

    # Get just the filename (not a full path) to use as the source label.
    # Both open() file objects and Streamlit UploadedFile have a .name attribute.
    try:
        source_name = pdf_file.name
        # If the name includes a path, keep only the filename part.
        source_name = source_name.replace("\\", "/").split("/")[-1]
    except AttributeError:
        # Fallback in case the file object does not expose .name
        source_name = "unknown.pdf"

    try:
        reader = PyPDF2.PdfReader(pdf_file)
        total_pages = len(reader.pages)

        for page_number in range(total_pages):
            try:
                page_obj = reader.pages[page_number]
                raw_text = page_obj.extract_text()

                # Skip pages that return None or only whitespace
                if not raw_text or not raw_text.strip():
                    print(f"[document_processor] Skipping page {page_number + 1} "
                          f"of '{source_name}' – no extractable text.")
                    continue

                pages.append({
                    "text"  : raw_text,
                    "source": source_name,
                    "page"  : page_number + 1   # Convert to 1-indexed
                })

            except Exception as page_error:
                # A single bad page should not stop the whole document
                print(f"[document_processor] Warning: could not read page "
                      f"{page_number + 1} of '{source_name}'. "
                      f"Reason: {page_error}")
                continue

    except Exception as pdf_error:
        # The PDF itself could not be opened (corrupted, encrypted, etc.)
        print(f"[document_processor] Error: could not open PDF '{source_name}'. "
              f"Reason: {pdf_error}")

    return pages


# -----------------------------------------------------------------------------
# FUNCTION 2: clean_text
# -----------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean raw text extracted from a PDF page.

    Cleaning steps applied (in order):
        1. Replace form-feed characters (\\f) with a space.
        2. Collapse runs of whitespace-only lines into a single blank line.
        3. Replace tabs with a single space.
        4. Collapse multiple consecutive spaces into one space.
        5. Strip leading and trailing whitespace from the whole string.
        6. Remove non-printable / control characters (except newlines).

    Punctuation is intentionally preserved because it helps sentence
    boundary detection during chunking and improves LLM context quality.

    Args:
        text (str): Raw string from a single PDF page.

    Returns:
        str: Cleaned string ready for chunking.
    """
    if not text:
        return ""

    # Step 1: Normalise form-feed characters used as page breaks in some PDFs
    text = text.replace("\f", " ")

    # Step 2: Collapse three or more consecutive newlines into two (one blank line)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Step 3: Replace tab characters with a space
    text = text.replace("\t", " ")

    # Step 4: Collapse multiple spaces (but not newlines) into a single space
    text = re.sub(r"[ ]{2,}", " ", text)

    # Step 5: Remove non-printable control characters (keep printable ASCII
    #         and common Unicode; keep newlines \n and carriage returns \r)
    text = re.sub(r"[^\x20-\x7E\n\r\u00A0-\uFFFF]", "", text)

    # Step 6: Strip leading/trailing whitespace from the entire result
    text = text.strip()

    return text


# -----------------------------------------------------------------------------
# FUNCTION 3: chunk_text
# -----------------------------------------------------------------------------

def chunk_text(text: str, source: str, page: int,
               chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Split a cleaned page text into overlapping character-level chunks.

    Chunking strategy:
        - Move a sliding window of `chunk_size` characters across the text.
        - Each step advances by (chunk_size - overlap) characters so that
          consecutive chunks share `overlap` characters.
        - Any remaining text shorter than chunk_size is added as a final chunk.
        - Chunks that consist only of whitespace after stripping are skipped.

    Args:
        text       (str): Cleaned text for one PDF page.
        source     (str): PDF filename (used to populate the "source" key).
        page       (int): 1-indexed page number (used to populate the "page" key).
        chunk_size (int): Maximum number of characters per chunk. Default: 500.
        overlap    (int): Number of characters shared between adjacent chunks.
                          Must be less than chunk_size. Default: 50.

    Returns:
        list[dict]: List of Chunk dicts.  chunk_index starts at 0 for this call
                    and increments by 1 for each chunk produced.

    Example output:
        [
            {"text": "...", "source": "guide.pdf", "page": 2, "chunk_index": 0},
            {"text": "...", "source": "guide.pdf", "page": 2, "chunk_index": 1},
        ]
    """
    chunks = []

    if not text or not text.strip():
        return chunks   # Nothing to chunk

    # Guard: overlap must be smaller than chunk_size to make forward progress
    if overlap >= chunk_size:
        print(f"[document_processor] Warning: overlap ({overlap}) >= chunk_size "
              f"({chunk_size}). Resetting overlap to chunk_size // 10.")
        overlap = chunk_size // 10

    step = chunk_size - overlap   # How far to advance the window each iteration
    chunk_index = 0
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk_content = text[start:end].strip()

        # Only keep the chunk if it has actual content
        if chunk_content:
            chunks.append({
                "text"        : chunk_content,
                "source"      : source,
                "page"        : page,
                "chunk_index" : chunk_index
            })
            chunk_index += 1

        start += step

    return chunks


# -----------------------------------------------------------------------------
# FUNCTION 4: process_pdf   ← MAIN ENTRY POINT (called by app.py)
# -----------------------------------------------------------------------------

def process_pdf(pdf_file, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Full pipeline: extract → clean → chunk a single PDF file.

    This is the ONLY function that app.py (Member 4) calls from this module.

    Pipeline:
        1. extract_text_from_pdf(pdf_file)      – get raw page dicts
        2. clean_text(page["text"])             – clean each page's text
        3. chunk_text(cleaned, source, page)    – split into chunks
        4. Re-number chunk_index globally       – unique across the whole document

    Args:
        pdf_file   : File-like object (binary mode). Must have a .name attribute.
        chunk_size (int): Max characters per chunk. Default: 500.
        overlap    (int): Overlap characters between consecutive chunks. Default: 50.

    Returns:
        list[dict]: All chunks from the entire PDF, each with keys:
                    "text", "source", "page", "chunk_index".
                    Returns an empty list if nothing could be extracted.

    Example:
        with open("python_careers.pdf", "rb") as f:
            chunks = process_pdf(f)
        # chunks[0] → {"text": "...", "source": "python_careers.pdf",
        #               "page": 1, "chunk_index": 0}
    """
    all_chunks = []
    global_chunk_index = 0   # Counter that increments across all pages

    # Step 1: Extract raw text page by page
    pages = extract_text_from_pdf(pdf_file)

    if not pages:
        print("[document_processor] No pages extracted. "
              "The PDF may be empty, corrupted, or image-only.")
        return all_chunks

    # Steps 2 & 3: Clean each page and chunk it
    for page_data in pages:
        cleaned = clean_text(page_data["text"])

        if not cleaned:
            # After cleaning, the page turned out to be empty – skip it
            continue

        page_chunks = chunk_text(
            text       = cleaned,
            source     = page_data["source"],
            page       = page_data["page"],
            chunk_size = chunk_size,
            overlap    = overlap
        )

        # Step 4: Re-number chunk_index globally so every chunk in the
        #         document has a unique index, not just per-page.
        for chunk in page_chunks:
            chunk["chunk_index"] = global_chunk_index
            global_chunk_index += 1

        all_chunks.extend(page_chunks)

    print(f"[document_processor] Processed '{pages[0]['source']}': "
          f"{len(pages)} page(s) → {len(all_chunks)} chunk(s).")

    return all_chunks


# =============================================================================
# INDEPENDENT TEST BLOCK
# =============================================================================
# Run this file directly to test Member 1's module without needing any other
# module from the project.
#
# Usage:
#   1. Place a PDF file named "sample.pdf" in the same folder as this script.
#   2. Open a terminal in that folder and run:
#        python document_processor.py
#   3. Expected output:
#        - A line showing how many pages and chunks were produced.
#        - The first chunk's full content printed to the console.
#        - A preview of every chunk (index, source, page, first 80 chars).
# =============================================================================

if __name__ == "__main__":
    import os
    import sys

    TEST_PDF = "sample.pdf"   # Change this to any PDF file you have

    if not os.path.exists(TEST_PDF):
        print(f"[TEST] '{TEST_PDF}' not found in the current directory.")
        print("[TEST] Please place a PDF named 'sample.pdf' here and re-run.")
        sys.exit(1)

    print(f"[TEST] Opening '{TEST_PDF}' for processing ...\n")

    with open(TEST_PDF, "rb") as f:
        chunks = process_pdf(f, chunk_size=500, overlap=50)

    print(f"\n[TEST] Total chunks produced : {len(chunks)}")

    if chunks:
        print("\n[TEST] === First Chunk (full content) ===")
        print(f"  source      : {chunks[0]['source']}")
        print(f"  page        : {chunks[0]['page']}")
        print(f"  chunk_index : {chunks[0]['chunk_index']}")
        print(f"  text        :\n{chunks[0]['text']}")

        print("\n[TEST] === Preview of All Chunks ===")
        for c in chunks:
            preview = c["text"][:80].replace("\n", " ")
            print(f"  [{c['chunk_index']:>3}] page {c['page']:>2} | "
                  f"{c['source']} | \"{preview}...\"")
    else:
        print("[TEST] No chunks were produced. "
             "Check that the PDF contains selectable text (not scanned images).")

