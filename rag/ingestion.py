from pathlib import Path

from loguru import logger

from rag.embeddings import Embedder
from storage.duckdb_store import DuckDBStore

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_document(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8")
    raise ValueError(f"Unsupported file extension: {ext}")


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if not text.strip():
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def ingest_directory(
    directory: Path,
    store: DuckDBStore,
    embedder: Embedder,
    *,
    chunk_size: int,
    chunk_overlap: int,
    clear_existing: bool = False,
) -> int:
    if not directory.exists():
        logger.warning("Ingestion directory does not exist: {}", directory)
        return 0

    if clear_existing:
        store.clear()

    files = sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        logger.warning("No supported documents found in {}", directory)
        return 0

    rows: list[tuple[str, str, list[float]]] = []
    for path in files:
        logger.info("Loading {}", path.name)
        try:
            text = load_document(path)
        except Exception as exc:
            logger.error("Failed to load {}: {}", path.name, exc)
            continue
        if not text.strip():
            logger.warning("Empty document: {}", path.name)
            continue
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        if not chunks:
            continue
        vectors = embedder.embed_texts(chunks)
        rows.extend((path.name, chunk, vec) for chunk, vec in zip(chunks, vectors))
        logger.debug("{}: {} chunks", path.name, len(chunks))

    if rows:
        store.insert_chunks(rows)
    logger.info("Ingestion complete: {} chunks from {} files", len(rows), len(files))
    return len(rows)
