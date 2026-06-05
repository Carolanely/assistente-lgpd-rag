"""RAG pipeline - chunk, embed, index, retrieve, generate.

Projeto: Assistente LGPD para pequenas equipes de produto e desenvolvimento.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pypdf import PdfReader

from src.pipeline.tools import TOOLS, run_tool_call


def _make_client() -> tuple[OpenAI, str | None]:
    """Inicializa cliente OpenAI-compatible conforme provider escolhido no .env."""
    if "GEMINI_API_KEY" in os.environ:
        client = OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        embed_api_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
    elif "OPENAI_API_KEY" in os.environ:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        embed_api_base = None
    else:
        raise RuntimeError("Configure GEMINI_API_KEY ou OPENAI_API_KEY no .env")
    return client, embed_api_base


class RAGPipeline:
    """Pipeline RAG end-to-end com Chroma local."""

    def __init__(
        self,
        corpus_dir: str = "data/corpus",
        persist_dir: str = "data/chroma",
        collection_name: str = "lgpd_docs",
        llm_model: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        self.client, embed_api_base = _make_client()
        self.llm_model = llm_model or os.environ.get("LLM_MODEL", "gemini-2.5-flash-lite")
        self.embed_model = embed_model or os.environ.get("EMBED_MODEL", "gemini-embedding-001")

        embed_kwargs: dict[str, Any] = {
            "api_key": os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            "model_name": self.embed_model,
        }
        if embed_api_base:
            embed_kwargs["api_base"] = embed_api_base
        self.embed_fn = OpenAIEmbeddingFunction(**embed_kwargs)

        self.corpus_dir = Path(corpus_dir)
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        chroma = chromadb.PersistentClient(path=persist_dir)
        self.collection = chroma.get_or_create_collection(
            name=collection_name, embedding_function=self.embed_fn
        )

    def ingest_and_index(self) -> int:
        """Le PDFs de `corpus_dir`, faz chunking e indexa em Chroma."""
        docs: list[dict[str, Any]] = []
        pdf_files = sorted(self.corpus_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(
                f"Nenhum PDF encontrado em {self.corpus_dir}. Adicione pelo menos 1 PDF."
            )

        for pdf_path in pdf_files:
            reader = PdfReader(str(pdf_path))
            for page_idx, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    docs.append({"text": text, "source": pdf_path.name, "page": page_idx})

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "; ", ", ", " "],
        )

        chunks: list[dict[str, Any]] = []
        for doc_idx, doc in enumerate(docs):
            split_texts = splitter.split_text(doc["text"])
            for chunk_idx, text in enumerate(split_texts):
                chunks.append(
                    {
                        "id": f"{doc['source']}::p{doc['page']}::d{doc_idx}::c{chunk_idx}",
                        "text": text,
                        "source": doc["source"],
                        "page": int(doc["page"]),
                    }
                )

        if not chunks:
            raise ValueError("PDFs encontrados, mas nenhum texto extraivel foi localizado.")

        # Reindexacao idempotente: evita duplicar documentos em reruns locais.
        existing = self.collection.get(ids=[c["id"] for c in chunks])
        existing_ids = set(existing.get("ids", []))
        new_chunks = [c for c in chunks if c["id"] not in existing_ids]

        if new_chunks:
            self.collection.add(
                ids=[c["id"] for c in new_chunks],
                documents=[c["text"] for c in new_chunks],
                metadatas=[{"source": c["source"], "page": c["page"]} for c in new_chunks],
            )

        return self.collection.count()

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Busca top-k chunks similares a query."""
        if self.collection.count() == 0:
            self.ingest_and_index()

        result = self.collection.query(query_texts=[query], n_results=k)
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        hits: list[dict[str, Any]] = []
        for text, metadata, distance in zip(docs, metadatas, distances):
            metadata = metadata or {}
            hits.append(
                {
                    "text": text,
                    "source": metadata.get("source", "corpus"),
                    "page": int(metadata.get("page", 0)),
                    "distance": float(distance),
                }
            )
        return hits

    def _context_from_hits(self, hits: list[dict[str, Any]]) -> str:
        blocks = []
        for h in hits:
            blocks.append(f"[{h['source']}:pagina {h['page']}]\n{h['text']}")
        return "\n\n---\n\n".join(blocks)

    def _maybe_tool_context(self, question: str) -> str:
        """Chama a tool quando a pergunta menciona explicitamente um artigo."""
        match = re.search(r"(?:art\.?|artigo)\s*(\d{1,2})", question, flags=re.IGNORECASE)
        if not match:
            return ""
        article_number = int(match.group(1))
        return run_tool_call("cite_article", f'{{"article_number": {article_number}}}')

    def answer(self, question: str, k: int = 5) -> dict[str, Any]:
        """Pipeline completo: retrieve + tool-use + augment + generate."""
        hits = self.retrieve(question, k=k)
        rag_context = self._context_from_hits(hits)
        deterministic_tool_context = self._maybe_tool_context(question)

        context = rag_context
        if deterministic_tool_context:
            context = f"[tool:cite_article]\n{deterministic_tool_context}\n\n---\n\n{rag_context}"

        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "Voce e um assistente de compliance LGPD para fins educacionais. "
                    "Nao substitui advogado nem DPO. Seja direto, cite fonte e destaque incertezas."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        # Function-calling real: o modelo pode pedir a tool; se pedir, executamos e fazemos a resposta final.
        first = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )
        choice = first.choices[0].message

        if getattr(choice, "tool_calls", None):
            messages.append(choice.model_dump(exclude_none=True))
            for tool_call in choice.tool_calls or []:
                tool_output = run_tool_call(tool_call.function.name, tool_call.function.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": tool_output,
                    }
                )
            final = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.2,
            )
            answer = final.choices[0].message.content or "Nao encontrado no corpus."
        else:
            answer = choice.content or "Nao encontrado no corpus."

        unique_sources = []
        seen = set()
        for h in hits:
            key = (h["source"], h["page"])
            if key not in seen:
                unique_sources.append(key)
                seen.add(key)

        return {"answer": answer, "sources": unique_sources}


PROMPT_TEMPLATE = """Responda APENAS com base no contexto abaixo.
Se a informacao nao estiver no contexto, diga "Nao encontrado no corpus".
Use linguagem clara e profissional em portugues brasileiro.
Sempre cite a fonte usando o formato [arquivo:pagina].
Quando a resposta envolver interpretacao juridica, inclua a frase: "Validar com responsavel juridico/DPO antes de aplicar.".

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA:"""


def build_rag_pipeline(corpus_dir: str = "data/corpus") -> RAGPipeline:
    """Factory: cria pipeline e indexa corpus se ainda nao indexado."""
    pipeline = RAGPipeline(corpus_dir=corpus_dir)
    if pipeline.collection.count() == 0:
        pipeline.ingest_and_index()
    return pipeline
