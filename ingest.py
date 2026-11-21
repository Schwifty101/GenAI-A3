"""Ingest PDFs in corpus/ into a persistent Chroma collection."""
import glob
import os

DB_DIR = "chroma_db"
COLLECTION = "papers"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def chunk_text(text):
    """Character window with overlap, preferring to break at whitespace."""
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + CHUNK_SIZE // 2, end)
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
        nxt = text.find(" ", start)  # begin at a word boundary
        if 0 <= nxt < end:
            start = nxt + 1
    return chunks


def page_text(page):
    text = page.get_text().strip()
    if text:
        return text
    try:
        import io
        import pytesseract
        from PIL import Image
        pix = page.get_pixmap(dpi=200)
        return pytesseract.image_to_string(Image.open(io.BytesIO(pix.tobytes("png")))).strip()
    except Exception:
        print(f"  warning: page {page.number + 1} has no text and OCR is unavailable, skipping")
        return ""


def main():
    import chromadb
    import pymupdf
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    for path in sorted(glob.glob("corpus/*.pdf")):
        source = os.path.basename(path)
        ids, docs, metas = [], [], []
        for page in pymupdf.open(path):
            for n, chunk in enumerate(chunk_text(page_text(page))):
                ids.append(f"{source}-p{page.number + 1}-c{n}")
                docs.append(chunk)
                metas.append({"source": source, "page": page.number + 1})
        if docs:
            embs = model.encode(docs, normalize_embeddings=True, batch_size=64).tolist()
            col.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
        print(f"{source}: {len(docs)} chunks")
    print(f"Total in collection: {col.count()}")


if __name__ == "__main__":
    main()
