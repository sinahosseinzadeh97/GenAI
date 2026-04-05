import fitz  # PyMuPDF

def parse_pdf(file_content: bytes, filename: str) -> list[dict]:
    """
    Parse a PDF file from bytes.
    Returns: list of {text: str, page_number: int, filename: str}
    """
    chunks = []
    with fitz.open("pdf", file_content) as doc:
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                chunks.append({
                    "text": text,
                    "page_number": i + 1,
                    "filename": filename
                })
    return chunks
