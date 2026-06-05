"""Tools de dominio para o Assistente LGPD.

A tool principal, `cite_article`, retorna trechos controlados de artigos da LGPD.
Ela reduz alucinacoes comuns em numeros de artigos e deixa a resposta mais auditavel.
"""

from __future__ import annotations

import json
from typing import Any, Callable


LGPD_ARTICLES: dict[int, str] = {
    1: (
        "Art. 1º: A LGPD dispoe sobre o tratamento de dados pessoais, inclusive nos meios "
        "digitais, por pessoa natural ou por pessoa juridica de direito publico ou privado, "
        "com o objetivo de proteger os direitos fundamentais de liberdade e de privacidade e "
        "o livre desenvolvimento da personalidade da pessoa natural."
    ),
    5: (
        "Art. 5º: Traz definicoes essenciais. Dado pessoal e informacao relacionada a pessoa "
        "natural identificada ou identificavel; dado pessoal sensivel inclui origem racial ou "
        "etnica, conviccao religiosa, opiniao politica, filiacao a sindicato, dado referente a "
        "saude ou vida sexual, dado genetico ou biometrico; titular e a pessoa natural a quem "
        "se referem os dados; controlador decide sobre o tratamento; operador realiza o "
        "tratamento em nome do controlador."
    ),
    6: (
        "Art. 6º: O tratamento de dados pessoais deve observar boa-fe e principios como "
        "finalidade, adequacao, necessidade, livre acesso, qualidade dos dados, transparencia, "
        "seguranca, prevencao, nao discriminacao e responsabilizacao/prestacao de contas."
    ),
    7: (
        "Art. 7º: O tratamento de dados pessoais somente pode ocorrer nas hipoteses legais, "
        "incluindo consentimento, cumprimento de obrigacao legal ou regulatoria, execucao de "
        "politicas publicas, estudos por orgao de pesquisa, execucao de contrato, exercicio "
        "regular de direitos, protecao da vida, tutela da saude, legitimo interesse e protecao "
        "do credito."
    ),
    8: (
        "Art. 8º: O consentimento deve ser fornecido por escrito ou por outro meio que demonstre "
        "a manifestacao de vontade do titular. Quando escrito, deve constar em clausula destacada."
    ),
    9: (
        "Art. 9º: O titular tem direito a acesso facilitado a informacoes sobre o tratamento, "
        "incluindo finalidade especifica, forma e duracao, identificacao do controlador, "
        "informacoes de contato, compartilhamento, responsabilidades dos agentes e direitos do titular."
    ),
    10: (
        "Art. 10: O legitimo interesse do controlador somente pode fundamentar tratamento para "
        "finalidades legitimas, consideradas a partir de situacoes concretas, e exige avaliacao "
        "de necessidade, transparencia e respeito aos direitos e liberdades fundamentais do titular."
    ),
    11: (
        "Art. 11: O tratamento de dados pessoais sensiveis tem hipoteses especificas, como "
        "consentimento especifico e destacado, cumprimento de obrigacao legal, politicas publicas, "
        "estudos por orgao de pesquisa, exercicio regular de direitos, protecao da vida, tutela "
        "da saude, garantia de prevencao a fraude e seguranca do titular."
    ),
    14: (
        "Art. 14: O tratamento de dados pessoais de criancas e adolescentes deve ser realizado "
        "em seu melhor interesse. O tratamento de dados de criancas exige consentimento especifico "
        "e em destaque por pelo menos um dos pais ou responsavel legal, salvo excecoes previstas."
    ),
    15: (
        "Art. 15: O termino do tratamento de dados pessoais ocorre em hipoteses como alcance da "
        "finalidade, fim do periodo de tratamento, comunicacao do titular para revogacao do "
        "consentimento ou determinacao da autoridade nacional."
    ),
    16: (
        "Art. 16: Apos o termino do tratamento, os dados pessoais devem ser eliminados, salvo "
        "conservacao para cumprimento de obrigacao legal ou regulatoria, estudo por orgao de "
        "pesquisa, transferencia a terceiro observados requisitos legais, ou uso exclusivo do "
        "controlador com dados anonimizados."
    ),
    18: (
        "Art. 18: O titular tem direitos perante o controlador, como confirmacao da existencia "
        "de tratamento, acesso, correcao, anonimizacao, bloqueio ou eliminacao de dados "
        "desnecessarios ou tratados em desconformidade, portabilidade, informacao sobre "
        "compartilhamento, revogacao do consentimento e revisao de decisoes automatizadas."
    ),
    37: (
        "Art. 37: O controlador e o operador devem manter registro das operacoes de tratamento "
        "de dados pessoais que realizarem, especialmente quando baseado no legitimo interesse."
    ),
    38: (
        "Art. 38: A autoridade nacional pode determinar que o controlador elabore relatorio de "
        "impacto a protecao de dados pessoais, inclusive de dados sensiveis, referente a suas "
        "operacoes de tratamento."
    ),
    41: (
        "Art. 41: O controlador deve indicar encarregado pelo tratamento de dados pessoais. As "
        "atividades do encarregado incluem aceitar reclamacoes e comunicacoes dos titulares, "
        "prestar esclarecimentos, receber comunicacoes da autoridade nacional, orientar "
        "funcionarios e contratados e executar atribuicoes determinadas pelo controlador ou em normas."
    ),
    46: (
        "Art. 46: Os agentes de tratamento devem adotar medidas de seguranca, tecnicas e "
        "administrativas aptas a proteger os dados pessoais de acessos nao autorizados e de "
        "situacoes acidentais ou ilicitas de destruicao, perda, alteracao, comunicacao ou difusao."
    ),
    48: (
        "Art. 48: O controlador deve comunicar a autoridade nacional e o titular sobre incidente "
        "de seguranca que possa acarretar risco ou dano relevante aos titulares. A comunicacao "
        "deve ocorrer em prazo razoavel e conter informacoes relevantes sobre o incidente."
    ),
    52: (
        "Art. 52: Os agentes de tratamento que infringirem a LGPD ficam sujeitos a sancoes "
        "administrativas, como advertencia, multa simples, multa diaria, publicizacao da infracao, "
        "bloqueio ou eliminacao dos dados pessoais, entre outras medidas previstas na lei."
    ),
}


def cite_article(article_number: int) -> str:
    """Retorna explicacao controlada de um artigo da LGPD.

    Args:
        article_number: numero do artigo da LGPD.

    Returns:
        Texto do artigo/sintese controlada. Se o artigo nao estiver mapeado,
        orienta a usar o RAG sobre o corpus.
    """
    article = LGPD_ARTICLES.get(int(article_number))
    if article:
        return article
    return (
        f"Art. {article_number}: artigo nao esta mapeado na tool local. "
        "Use a busca RAG no corpus para verificar o texto oficial e cite a pagina retornada."
    )


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "cite_article",
            "description": (
                "Retorna uma referencia controlada de um artigo especifico da LGPD. "
                "Use quando a pergunta mencionar um numero de artigo ou exigir confirmacao legal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "article_number": {
                        "type": "integer",
                        "description": "Numero do artigo da LGPD, por exemplo 6, 7, 18, 46 ou 48.",
                    }
                },
                "required": ["article_number"],
            },
        },
    }
]


TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    "cite_article": cite_article,
}


def run_tool_call(name: str, arguments_json: str) -> str:
    """Executa uma tool call e retorna o resultado como string."""
    if name not in TOOL_REGISTRY:
        return f"ERROR: tool '{name}' nao registrada"
    try:
        kwargs = json.loads(arguments_json)
        return TOOL_REGISTRY[name](**kwargs)
    except Exception as e:  # pragma: no cover - protecao para demo ao vivo
        return f"ERROR ao executar {name}: {e}"
