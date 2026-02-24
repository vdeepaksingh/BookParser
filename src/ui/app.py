"""Phase 5 — Streamlit UI for BookParser."""
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="BookParser", layout="centered")
st.title("📚 BookParser")

tab_ask, tab_search = st.tabs(["Ask", "Search"])

with tab_ask:
    st.subheader("Ask a question")
    with st.form(key="ask_form"):
        query = st.text_input("Question", placeholder="e.g. What is the principle of relativity?")
        submitted = st.form_submit_button("Ask")
    if submitted and query:
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(f"{API_URL}/ask", json={"query": query}, timeout=300)
                resp.raise_for_status()
                data = resp.json()
                st.markdown(data["answer"])
                if data.get("citations"):
                    st.markdown("**Citations:**")
                    for i, c in enumerate(data["citations"], 1):
                        st.markdown(f"- **[{i}]** {c}")
            except Exception as e:
                st.error(str(e))

with tab_search:
    st.subheader("Search chunks")
    with st.form(key="search_form"):
        search_query = st.text_input("Search", placeholder="e.g. Maxwell equations")
        top_k = st.slider("Top K", 1, 10, 5)
        submitted_search = st.form_submit_button("Search")
    if submitted_search and search_query:
        with st.spinner("Searching..."):
            try:
                resp = requests.get(f"{API_URL}/search", params={"query": search_query, "top_k": top_k}, timeout=60)
                resp.raise_for_status()
                for r in resp.json()["results"]:
                    m = r["metadata"]
                    st.markdown(f"**{m.get('book','?')}** — {m.get('chapter_title','?')}, p.{m.get('page_start','?')}")
                    st.caption(r["text"][:300] + "...")
                    st.divider()
            except Exception as e:
                st.error(str(e))
