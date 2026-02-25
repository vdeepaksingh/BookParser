"""Phase 5 — Streamlit UI for BookParser."""
import tempfile

import requests
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from config import API_URL, GRAPH_ENTITY_CAP, GRAPH_NER_TEXT_CAP

def _show_section(book, chapter, section):
    """Fetch and display section text from API."""
    try:
        r = requests.get(f"{API_URL}/section",
                         params={"book": book, "chapter": chapter, "section": section}, timeout=10)
        r.raise_for_status()
        d = r.json()
        st.markdown(f"*p.{d.get('page_start')}–{d.get('page_end')}*")
        st.markdown(d["text"])
    except Exception as e:
        st.error(str(e))


st.set_page_config(page_title="BookParser", layout="centered")
st.title("📚 BookParser")

tab_ask, tab_search, tab_graph, tab_flash = st.tabs(["Ask", "Search", "Graph", "Flashcards"])

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
                results = resp.json().get("results", [])
                if not results:
                    st.info("No results found.")
                for r in results:
                    m = r["metadata"]
                    st.markdown(f"**{m.get('book','?')}** — {m.get('chapter_title','?')}, p.{m.get('page_start','?')}")
                    st.caption(r["text"][:GRAPH_NER_TEXT_CAP] + "...")
                    st.divider()
            except Exception as e:
                st.error(str(e))

with tab_graph:
    st.subheader("Knowledge Graph")
    graph_mode = st.radio("View", ["Book Structure", "Entity Search"], horizontal=True)

    if graph_mode == "Book Structure":
        try:
            books = requests.get(f"{API_URL}/graph/books", timeout=10).json().get("books", [])
        except Exception as e:
            st.error(f"API unavailable: {e}")
            books = []

        if not books:
            st.warning("No books in graph. Run `python main.py graph-struct` first.")
        else:
            selected = st.selectbox("Select book", books)
            if st.button("Visualize", key="graph_book_btn"):
                with st.spinner("Building graph..."):
                    try:
                        data = requests.get(f"{API_URL}/graph/book/{requests.utils.quote(selected)}", timeout=10).json()
                        net = Network(height="500px", width="100%", bgcolor="#1e1e1e", font_color="white", cdn_resources="in_line")
                        net.add_node(selected, label=selected[:40], color="#4e9af1", size=30, title=selected)
                        for ch in data.get("chapters", []):
                            ch_id = f"{selected}::{ch['chapter']}"
                            net.add_node(ch_id, label=ch["chapter"][:30], color="#f1a94e", size=20, title=ch["chapter"])
                            net.add_edge(selected, ch_id)
                            for sec in ch.get("sections", []):
                                sec_id = f"{ch_id}::{sec}"
                                net.add_node(sec_id, label=sec[:25], color="#7ec87e", size=12, title=sec)
                                net.add_edge(ch_id, sec_id)
                        for ent in data.get("entities", [])[:GRAPH_ENTITY_CAP]:
                            ent_id = f"ent::{ent['name']}"
                            net.add_node(ent_id, label=ent["name"][:20], color="#e07eb8", size=10,
                                         title=f"{ent['label']}: {ent['name']}")
                            net.add_edge(selected, ent_id)
                        net.toggle_stabilization(True)
                        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
                            net.save_graph(f.name)
                            html = open(f.name, encoding="utf-8").read()
                        components.html(html, height=520, scrolling=False)
                        st.divider()
                        st.markdown("**Browse sections:**")
                        for ch in data.get("chapters", []):
                            with st.expander(ch["chapter"]):
                                for sec in ch.get("sections", []):
                                    with st.expander(f"    {sec}", expanded=False):
                                        _show_section(selected, ch["chapter"], sec)
                    except Exception as e:
                        st.error(str(e))

    else:  # Entity Search
        with st.form(key="entity_form"):
            entity_name = st.text_input("Entity name", placeholder="e.g. Einstein")
            submitted_entity = st.form_submit_button("Search")
        if submitted_entity and entity_name:
            with st.spinner("Searching..."):
                try:
                    data = requests.get(f"{API_URL}/graph/entity/{requests.utils.quote(entity_name)}", timeout=10).json()
                    if "error" in data:
                        st.warning(data["error"])
                    else:
                        st.markdown(f"**{entity_name}** appears in **{len(data['appears_in_books'])}** book(s):")
                        for b in data["appears_in_books"]:
                            st.markdown(f"- {b}")
                        if not data.get("sections"):
                            st.info("No sections found.")
                        else:
                            st.markdown(f"**Mentioned in {len(data['sections'])} section(s):**")
                            for s in data["sections"][:10]:
                                with st.expander(f"{s['book']} › {s['chapter']} › {s['section']}"):
                                    _show_section(s["book"], s["chapter"], s["section"])
                except Exception as e:
                    st.error(str(e))

with tab_flash:
    st.subheader("Flashcards")
    try:
        books = requests.get(f"{API_URL}/flashcards/books", timeout=10).json().get("books", [])
    except Exception as e:
        st.error(f"API unavailable: {e}")
        books = []

    if not books:
        st.warning("No flashcards found. Run `python main.py flashcards` first.")
    else:
        selected_book = st.selectbox("Select book", books, key="flash_book")
        if st.button("Load flashcards"):
            with st.spinner("Loading..."):
                try:
                    data = requests.get(f"{API_URL}/flashcards",
                                        params={"book": selected_book}, timeout=30).json()
                    total = sum(len(s["cards"]) for ch in data["chapters"] for s in ch["sections"])
                    st.caption(f"{total} cards across {len(data['chapters'])} chapters")
                    for ch in data["chapters"]:
                        with st.expander(ch["chapter"]):
                            for sec in ch["sections"]:
                                st.markdown(f"**{sec['section']}**")
                                for i, card in enumerate(sec["cards"], 1):
                                    with st.expander(f"Q{i}: {card['question']}"):
                                        st.markdown(card["answer"])
                except Exception as e:
                    st.error(str(e))
