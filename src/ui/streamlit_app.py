"""Streamlit UI - entrada principal do app."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

load_dotenv()

import streamlit as st  # noqa: E402

from src.observability.trace import log_event, trace  # noqa: E402
from src.pipeline.cache import ExactCache, SemanticCache  # noqa: E402
from src.pipeline.rag import build_rag_pipeline  # noqa: E402
from src.pipeline.routing import classify_complexity  # noqa: E402
from src.pipeline.tools import cite_article  # noqa: E402


st.set_page_config(page_title="Assistente LGPD RAG", page_icon="⚖️", layout="centered")

st.title("⚖️ Assistente LGPD RAG")
st.caption(
    "Q&A com RAG, citações por página, tool-use para artigos da LGPD, cache e roteamento cheap-first."
)


@st.cache_resource
def get_pipeline():
    return build_rag_pipeline(corpus_dir=str(_ROOT / "data" / "corpus"))


@st.cache_resource
def get_exact_cache():
    return ExactCache()


@st.cache_resource
def get_semantic_cache():
    return SemanticCache(threshold=0.93)


with st.spinner("Inicializando pipeline RAG e indexando corpus LGPD..."):
    pipeline = get_pipeline()
    exact_cache = get_exact_cache()
    semantic_cache = get_semantic_cache()

with st.sidebar:
    st.header("Métricas da demo")
    st.metric("Chunks indexados", pipeline.collection.count())
    st.metric("Exact cache", exact_cache.stats()["size"])
    st.metric("Semantic cache", semantic_cache.stats()["size"])
    st.caption("Corpus fixo: LGPD + orientações ANPD sintetizadas para uso educacional.")

    st.divider()
    st.subheader("Tool: citar artigo")
    art = st.number_input("Artigo da LGPD", min_value=1, max_value=65, value=18, step=1)
    if st.button("Consultar artigo"):
        st.info(cite_article(int(art)))

    if st.button("Limpar caches"):
        exact_cache.clear()
        semantic_cache.clear()
        st.success("Caches limpos. Recarregue a página para atualizar as métricas.")

examples = [
    "Quais princípios devo observar ao criar um formulário de cadastro?",
    "Posso armazenar CPF para emitir nota fiscal?",
    "O que diz o artigo 18 da LGPD?",
    "O que fazer se houver vazamento de e-mails e telefones de clientes?",
]

st.write("**Perguntas de exemplo:**")
cols = st.columns(2)
for idx, ex in enumerate(examples):
    with cols[idx % 2]:
        if st.button(ex, use_container_width=True):
            st.session_state["query"] = ex

query = st.text_input(
    "Sua pergunta:",
    value=st.session_state.get("query", ""),
    placeholder="Pergunte algo sobre LGPD, dados pessoais, incidentes, consentimento...",
)

if query:
    with trace("query_handle", query=query) as ctx:
        trace_id = ctx["trace_id"]

        cached = exact_cache.get(query)
        if cached:
            st.success("Cache hit (exact)")
            st.write(cached)
            log_event("cache_hit", trace_id=trace_id, layer="exact")
            st.stop()

        try:
            cached = semantic_cache.get(query)
        except Exception as e:  # demo robusta quando API de embedding falha
            cached = None
            st.warning(f"Semantic cache indisponível nesta chamada: {e}")

        if cached:
            st.success("Cache hit (semantic)")
            st.write(cached)
            log_event("cache_hit", trace_id=trace_id, layer="semantic")
            st.stop()

        decision = classify_complexity(query)
        st.info(f"Routing: {decision.complexity} -> {decision.model} ({decision.reason})")
        log_event("route_decision", trace_id=trace_id, **decision.__dict__)

        with st.spinner("Buscando no corpus e gerando resposta..."):
            result = pipeline.answer(query)

        st.write(result["answer"])
        if result.get("sources"):
            with st.expander("Fontes citadas"):
                for source, page in result["sources"]:
                    st.write(f"- `{source}:p{page}`")

        exact_cache.put(query, result["answer"])
        try:
            semantic_cache.put(query, result["answer"])
        except Exception as e:
            st.warning(f"Não foi possível gravar no semantic cache: {e}")
        log_event("answer_generated", trace_id=trace_id, sources=len(result.get("sources", [])))

st.divider()
st.caption(
    "Uso educacional. Para aplicar em caso real, valide com responsável jurídico/DPO e consulte o texto legal atualizado."
)
