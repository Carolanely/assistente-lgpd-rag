"""Model routing cheap-first com fallback."""

from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class RouteDecision:
    model: str
    complexity: str  # "simple" | "complex"
    reason: str


def classify_complexity(query: str) -> RouteDecision:
    """Classifica a query para escolher modelo barato ou premium.

    A heuristica privilegia o modelo barato para perguntas factuais e usa o modelo
    premium quando ha pedido de analise, comparacao, decisao ou multiplos criterios.
    """
    cheap_model = os.environ.get("CHEAP_MODEL", os.environ.get("LLM_MODEL", "gemini-2.5-flash-lite"))
    premium_model = os.environ.get("PREMIUM_MODEL", "gemini-2.5-pro")

    q = query.strip().lower()
    complex_terms = {
        "analise",
        "analisa",
        "compare",
        "comparar",
        "explique em detalhes",
        "decida",
        "risco",
        "impacto",
        "relatorio",
        "passo a passo",
        "estrategia",
        "base legal adequada",
        "dados sensiveis",
        "crianca",
        "adolescente",
        "incidente",
    }
    multi_part = q.count("?") > 1 or any(sep in q for sep in [" e ", ";", "1)", "2)"])

    if any(term in q for term in complex_terms) or len(q) > 180 or multi_part:
        return RouteDecision(
            model=premium_model,
            complexity="complex",
            reason="pergunta exige analise, decisao ou avaliacao de risco",
        )

    return RouteDecision(
        model=cheap_model,
        complexity="simple",
        reason="pergunta factual/curta, adequada para modelo barato",
    )


def make_client() -> OpenAI:
    """Cliente OpenAI-compatible para o provider configurado."""
    if "GEMINI_API_KEY" in os.environ:
        return OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
