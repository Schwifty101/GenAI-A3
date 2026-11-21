import streamlit as st

import ask

st.set_page_config(page_title="Paper Q&A")
st.title("Ask the research papers")


@st.cache_resource
def load_models():
    ask._embedder()
    ask._collection()
    ask._generator()
    return True


load_models()

question = st.text_input("Your question")
if st.button("Ask") and question.strip():
    with st.spinner("Thinking..."):
        text, chunks = ask.answer(question.strip())
    st.subheader("Answer")
    st.write(text)
    with st.expander("Sources (page refs)"):
        for i, c in enumerate(chunks, 1):
            st.markdown(f"**{i}. {c['source']}, p.{c['page']}** (score {c['score']:.3f})")
            st.caption(c["text"][:250].replace("\n", " "))
