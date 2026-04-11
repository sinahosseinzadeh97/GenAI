from pypdf import PdfReader
import io

def parse_pdf(file_content: bytes, filename: str) -> list[dict]:
    """
    Parse a PDF file from bytes.
    Returns: list of {text: str, page_number: int, filename: str}
    """
    chunks = []
    reader = PdfReader(io.BytesIO(file_content))
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            chunks.append({
                "text": text,
                "page_number": page_num + 1,
                "filename": filename
            })
    return chunks
