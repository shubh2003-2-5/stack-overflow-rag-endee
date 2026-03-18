import streamlit as st

from src.config import TOP_K
from src.rag import answer_query
from src.retrieve import retrieve


def main():
    st.set_page_config(page_title="Stack Overflow RAG", layout="wide")

    st.title("Stack Overflow Semantic Search (RAG) — Powered by Endee Vector DB")
    st.write(
        "Enter a query and the app will retrieve relevant Stack Overflow posts using Endee vector database and generate an answer with Azure OpenAI GPT-4.1."
    )

    api_key = st.text_input(
        "OpenAI API key (optional)",
        value="",
        type="password",
        help="Provide your OpenAI API key to get a generated answer; leave blank to see retrieved snippets only.",
    )

    query = st.text_input("Search query", value="", max_chars=256)
    submitted = st.button("Search")

    if submitted:
        if not query.strip():
            st.warning("Please enter a query to search.")
            return

        with st.spinner("Retrieving results..."):
            results = retrieve(query, top_k=TOP_K)

        if not results:
            st.info("No results found for that query.")
            return

        st.subheader("Top results")
        for idx, r in enumerate(results, start=1):
            meta = r.get("meta", {})
            title = meta.get("title") or "(no title)"
            snippet = (meta.get("text") or "").replace("\n", " ")
            st.markdown(f"**{idx}. {title}**")
            st.write(snippet[:400] + ("..." if len(snippet) > 400 else ""))
            st.write(f"Score: {r.get('similarity'):.4f}")
            st.write("---")

        with st.spinner("Generating answer..."):
            rag_resp = answer_query(
                query,
                top_k=TOP_K,
                openai_api_key=api_key,
            )

        st.subheader("Generated answer")
        st.write(rag_resp.get("answer"))
        st.caption(f"Source: {rag_resp.get('source')}")


if __name__ == "__main__":
    main()
