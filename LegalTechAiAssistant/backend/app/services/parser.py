from pathlib import Path
from typing import Tuple
from pypdf import PdfReader
import docx

SUPPORTED = {"application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain"}

def parse_bytes_to_text(filename: str, content_type: str, file_path: Path) -> Tuple[str, str]:
    """Return (text, detected_title)."""
    ext = Path(filename).suffix.lower()
    text = ""
    title = Path(filename).stem

    if content_type == "application/pdf" or ext == ".pdf":
        reader = PdfReader(str(file_path))
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
    elif content_type.endswith("wordprocessingml.document") or ext == ".docx":
        d = docx.Document(str(file_path))
        text = "\n".join(p.text for p in d.paragraphs)
    else:
        # fallback to plain text
        text = file_path.read_text(encoding="utf-8", errors="ignore")

    text = text.strip()
    return text, title