"""PDF tool: validate an upload, extract its text, split it into chunks, and pick
the chunks most relevant to a question using simple keyword overlap (no embeddings).
"""
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

MAX_BYTES = 20 * 1024 * 1024   # 20 MB
CHUNK_WORDS = 2000
OVERLAP_WORDS = 200
MAX_CHUNKS = 6

STOPWORDS = set(
    "the a an and or of to in on for with by from at as is are was were be been this that these those "
    "what which who how why when where does do did can could should would about into than then there "
    "their its it they them we our you your paper papers pdf document".split()
)


class PDFError(ValueError):
    pass


def sanitize_filename(name: str) -> str:
    """Keep only safe characters and drop any folder part, e.g. '../../x.pdf' -> 'x.pdf'."""
    base = Path(name or "upload.pdf").name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(base).stem).strip("._") or "upload"
    return f"{stem[:80]}.pdf"


def save_upload(name: str, data: bytes, uploads_dir: Path) -> Path:
    """Validate the bytes, then save them inside uploads_dir only."""
    if not (name or "").lower().endswith(".pdf"):
        raise PDFError("Only .pdf files are accepted.")
    if len(data) > MAX_BYTES:
        raise PDFError("The PDF is larger than 20 MB. Please upload a smaller file.")
    if not data.startswith(b"%PDF"):
        raise PDFError("This file is not a valid PDF.")
    uploads_dir = Path(uploads_dir).resolve()
    uploads_dir.mkdir(parents=True, exist_ok=True)
    path = (uploads_dir / sanitize_filename(name)).resolve()
    if path.parent != uploads_dir:
        raise PDFError("Invalid file name.")
    path.write_bytes(data)
    return path


def extract_text(path: Path) -> tuple[str, int]:
    """Return (text, page_count). Rejects encrypted and image-only PDFs."""
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise PDFError("The PDF is larger than 20 MB.")
    try:
        reader = PdfReader(str(path))
    except (PdfReadError, OSError, ValueError) as e:
        raise PDFError(f"Could not read this PDF ({type(e).__name__}).") from None
    if reader.is_encrypted:
        raise PDFError("This PDF is encrypted / password-protected. Please upload an unlocked copy.")
    pages = [(page.extract_text() or "") for page in reader.pages]
    text = "\n".join(pages).strip()
    if len(text.split()) < 5:
        raise PDFError("No text found in this PDF. It is probably scanned images (OCR is not supported).")
    return text, len(pages)


def chunk_words(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    words = text.split()
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def load_pdf(path: Path) -> dict:
    text, page_count = extract_text(path)
    chunks = chunk_words(text)
    return {"name": Path(path).name, "pages": page_count, "words": len(text.split()), "chunks": chunks}


def _keywords(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9][a-z0-9-]+", text.lower()) if w not in STOPWORDS]


def top_chunks(chunks: list[str], question: str, k: int = MAX_CHUNKS) -> list[tuple[int, str]]:
    """Score each chunk by how often the question's keywords appear in it."""
    terms = set(_keywords(question))
    scored = []
    for i, chunk in enumerate(chunks):
        counts = Counter(_keywords(chunk))
        score = sum(min(counts[t], 5) for t in terms)   # cap so one repeated word can't dominate
        scored.append((score, i))
    best = sorted(scored, key=lambda s: (-s[0], s[1]))[:k]
    return [(i, chunks[i]) for _, i in sorted(best, key=lambda s: s[1])]
