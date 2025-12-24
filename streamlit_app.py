import json
from typing import Any, Dict, List

import requests
import streamlit as st


def call_query_api(base_url: str, query: str, timeout: int) -> Dict[str, Any]:
    response = requests.post(
        f"{base_url.rstrip('/')}/query",
        json={"query": query},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def format_source_label(entry: Dict[str, Any]) -> str:
    source = entry.get("source") or "unknown"
    page = entry.get("page")
    section = entry.get("section_path") or ""
    label = f"{source}"
    if page is not None:
        label += f" p.{page}"
    if section:
        label += f" · {section}"
    return label


def render_sources(citations: List[Dict[str, Any]], matched_sources: Dict[str, Any]) -> None:
    if citations:
        st.subheader("Citations")
        for citation in citations:
            label = format_source_label(citation)
            st.markdown(f"- **{citation.get('citation', 'C?')}**: {label}")

    if matched_sources:
        st.subheader("Matched Sources")
        primary = matched_sources.get("primary", [])
        auxiliary = matched_sources.get("auxiliary", [])
        if primary:
            st.markdown("**Primary**")
            st.dataframe(primary, use_container_width=True)
        if auxiliary:
            st.markdown("**Auxiliary (HyDE)**")
            st.dataframe(auxiliary, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="HyRAG QA", layout="wide")
    st.title("HyRAG Question Answering")
    st.caption("Ask a question, get grounded answers with citations.")

    with st.sidebar:
        st.header("Settings")
        base_url = st.text_input("API Base URL", value="http://localhost:8000")
        timeout = st.number_input("Request timeout (seconds)", min_value=5, value=120, step=5)
        show_debug = st.toggle("Show debug payload", value=False)

    query = st.text_area("Enter your question", placeholder="Ask a question about your documents...")
    col1, col2 = st.columns([1, 3])
    with col1:
        submit = st.button("Ask")
    with col2:
        st.markdown("The app will call the FastAPI `/query` endpoint.")

    if submit:
        if not query.strip():
            st.warning("Please enter a question before submitting.")
            return
        with st.spinner("Querying the QA service..."):
            try:
                payload = call_query_api(base_url, query, int(timeout))
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
                return

        answer = payload.get("answer", "")
        st.subheader("Answer")
        st.write(answer or "No answer returned.")

        citations = payload.get("citations", [])
        matched_sources = payload.get("matched_sources", {})
        render_sources(citations, matched_sources)

        hypothetical = payload.get("hypothetical_answer")
        if hypothetical:
            st.subheader("Hypothetical Answer (retrieval aid)")
            st.write(hypothetical)

        if show_debug:
            st.subheader("Debug Payload")
            st.code(json.dumps(payload.get("debug", {}), indent=2))


if __name__ == "__main__":
    main()
