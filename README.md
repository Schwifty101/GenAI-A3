# RAG over research papers
pip install pymupdf chromadb sentence-transformers streamlit transformers torch
python3 ingest.py            # parse, chunk, embed, persist ./chroma_db
python3 ask.py "question"    # CLI answer + retrieved chunks
streamlit run app.py         # tiny UI
python3 eval.py              # precision@k
