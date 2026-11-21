"""Retrieve chunks from Chroma and answer with a local Qwen model."""
import sys

from ingest import COLLECTION, DB_DIR, EMBED_MODEL
from prompts import SYSTEM_PROMPT, USER_TEMPLATE

K = 5
THRESHOLD = 0.35
GEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
NO_INFO = "I don't have that information."

_cache = {}


def _embedder():
    if "emb" not in _cache:
        from sentence_transformers import SentenceTransformer
        _cache["emb"] = SentenceTransformer(EMBED_MODEL)
    return _cache["emb"]


def _collection():
    if "col" not in _cache:
        import chromadb
        _cache["col"] = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)
    return _cache["col"]


def _generator():
    if "gen" not in _cache:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(GEN_MODEL)
        model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, torch_dtype=torch.float32)
        model.eval()
        _cache["gen"] = (tok, model)
    return _cache["gen"]


def retrieve(query, k=K):
    emb = _embedder().encode([query], normalize_embeddings=True).tolist()
    res = _collection().query(query_embeddings=emb, n_results=k)
    return [
        {"text": d, "source": m["source"], "page": m["page"], "score": 1 - dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def generate(question, chunks):
    import torch
    tok, model = _generator()
    context = "\n\n".join(f"[{c['source']} p.{c['page']}]\n{c['text']}" for c in chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(context=context, question=question)},
    ]
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=300, do_sample=False)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def answer(query, k=K, threshold=THRESHOLD):
    chunks = retrieve(query, k)
    good = [c for c in chunks if c["score"] >= threshold]
    if not good or chunks[0]["score"] < threshold:
        return NO_INFO, chunks
    text = generate(query, good)
    if NO_INFO.rstrip(".") in text:
        return NO_INFO, chunks
    pages = []
    for c in good:
        ref = f"{c['source']} p.{c['page']}"
        if ref not in pages:
            pages.append(ref)
    return f"{text}\n\nSources: {', '.join(pages)}", chunks


def show(query):
    text, chunks = answer(query)
    print(f"\nAnswer:\n{text}\n\n--- retrieved chunks (debug) ---")
    for i, c in enumerate(chunks, 1):
        snippet = c["text"][:200].replace("\n", " ")
        print(f"{i}. score={c['score']:.3f} {c['source']} p.{c['page']}: {snippet}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
    else:
        while True:
            try:
                q = input("\nQuestion (blank to quit): ").strip()
            except EOFError:
                break
            if not q:
                break
            show(q)
