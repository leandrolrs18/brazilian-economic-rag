from __future__ import annotations

import time
import logging
from typing import List, Literal, Tuple, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from app.config.settings import settings
from app.rag.embeddings import EmbeddingModel
from app.rag.hybrid_retriever import build_retriever
from app.rag.reranker import Reranker
from app.rag.vector_store import get_client

# Configuração de Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SINGLETONS PARA PERFORMANCE ---
_RERANKER_CACHE: Optional[Reranker] = None
_EMBEDDER_CACHE: Optional[EmbeddingModel] = None

def get_reranker() -> Reranker:
    global _RERANKER_CACHE
    if _RERANKER_CACHE is None:
        logger.info("  └─ 🚀 Carregando Reranker na memória pela primeira vez...")
        _RERANKER_CACHE = Reranker()
    return _RERANKER_CACHE

def get_embedder() -> EmbeddingModel:
    global _EMBEDDER_CACHE
    if _EMBEDDER_CACHE is None:
        _EMBEDDER_CACHE = EmbeddingModel()
    return _EMBEDDER_CACHE

def _build_chat_llm(temperature: float = 0.0) -> ChatGroq:
    """Configura a Groq Cloud usando o modelo definido no Settings."""
    return ChatGroq(
        model_name=settings.groq_model, 
        groq_api_key=settings.groq_api_key,
        temperature=temperature,
    )

@tool
def search_bcb_knowledge_base(question: str) -> str:
    """Use para perguntas macroeconômicas do Banco Central (IPCA, SELIC, câmbio e séries SGS)."""
    return f"rotear_bcb:{question}"

@tool
def search_b3_knowledge_base(question: str) -> str:
    """Use para perguntas de ações da bolsa/B3 (ticker, cotação, fechamento, IBOV, dividendos)."""
    return f"rotear_b3:{question}"

def _fallback_router(question: str) -> Literal["bcb", "b3"]:
    """Classificador rápido de intenção caso o bind_tools falhe."""
    start_time = time.perf_counter()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Classifique a pergunta em apenas uma palavra: 'bcb' ou 'b3'. "
            "Use 'b3' para bolsa/ações/tickers; use 'bcb' para macroeconomia/IPCA/SELIC/câmbio."
        ),
        ("user", "{question}"),
    ])
    chain = prompt | _build_chat_llm(temperature=0.0) | StrOutputParser()
    raw = chain.invoke({"question": question}).strip().lower()
    
    selected = "b3" if "b3" in raw else "bcb"
    logger.info(f"  └─ 🧭 Fallback Router selecionou: {selected} ({time.perf_counter() - start_time:.2f}s)")
    return selected

def route_question_with_tools(question: str) -> Literal["bcb", "b3"]:
    """Roteia a pergunta entre BCB e B3 usando Tool Calling ou Fallback."""
    start_time = time.perf_counter()
    llm = _build_chat_llm(temperature=0.0)
    tools = [search_bcb_knowledge_base, search_b3_knowledge_base]
    
    if not hasattr(llm, "bind_tools"):
        return _fallback_router(question)

    router_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a router. Choose:\n"
            "- search_b3_knowledge_base for stocks, tickers, B3\n"
            "- search_bcb_knowledge_base for macroeconomics, inflation, interest rates\n"
            "Return ONLY the tool.\n\n"
            "Você também pode interpretar perguntas em português."
        ),
        ("user", "{question}"),
    ])
    
    try:
        router_chain = router_prompt | llm.bind_tools(tools)
        result = router_chain.invoke({"question": question})
        tool_calls = getattr(result, "tool_calls", []) or []
        
        if not tool_calls:
            return _fallback_router(question)

        chosen = tool_calls[0].get("name", "")
        selected = "b3" if chosen == "search_b3_knowledge_base" else "bcb"
        logger.info(f"  └─ 🧭 Roteador (Tool Call) selecionou: {selected} ({time.perf_counter() - start_time:.2f}s)")
        return selected
    except Exception as e:
        logger.warning(f"  └─ ⚠️ Erro no roteamento de tools: {e}. Usando fallback.")
        return _fallback_router(question)

def retrieve_documents(question: str, agent: str | None = None) -> List[Document]:
    """Realiza a busca híbrida com filtro por agente, garantindo integridade dos dados."""
    
    # 1. Construção do Retriever
    start_build = time.perf_counter()
    retriever = build_retriever(filter_source=agent)
    logger.info(f"  └─ 🏗️ Retriever híbrido preparado em {time.perf_counter() - start_build:.4f}s")
    
    # 2. Execução da Busca
    start_search = time.perf_counter()
    docs = []
    try:
        # Usando .invoke para evitar LangChainDeprecationWarning
        raw_docs = retriever.invoke(question)
        # Filtro de segurança: remove nulos e garante strings para o Pydantic
        docs = [
            d for d in raw_docs 
            if d and hasattr(d, "page_content") and isinstance(d.page_content, str) and d.page_content.strip()
        ]
        logger.info(f"  └─ 🔍 Busca Híbrida retornou {len(docs)} documentos válidos em {time.perf_counter() - start_search:.2f}s")
    except Exception as e:
        logger.error(f"  └─ ❌ Erro na busca híbrida: {e}. Iniciando fallback vetorial puro.")
        client = get_client()
        vector = get_embedder().embed_query(question)
        
        search_filter = None
        if agent:
            search_filter = Filter(must=[FieldCondition(key="source", match=MatchValue(value=agent))])
            
        results = client.search(
            collection_name=settings.collection_name,
            query_vector=vector,
            limit=settings.dense_top_k,
            query_filter=search_filter,
        )
        
        docs = []
        for p in results:
            content = getattr(p, "payload", {}).get("text")
            if content:
                docs.append(Document(page_content=str(content)))
        logger.info(f"  └─ 🔍 Fallback Vetorial retornou {len(docs)} documentos.")

    # 3. Reranking (Refinamento da relevância)
    if settings.rerank_enabled and docs:
        start_rerank = time.perf_counter()
        candidates = docs[:settings.rerank_candidates]
        # Usa instância cacheada do Reranker
        docs = get_reranker().rerank(question, candidates)
        logger.info(f"  └─ 🧬 Reranking concluído em {time.perf_counter() - start_rerank:.2f}s")
        
    return docs

def retrieve_documents_with_router(question: str) -> Tuple[List[Document], Literal["bcb", "b3"]]:
    """Coordena o roteamento e a recuperação de documentos."""
    selected_agent = route_question_with_tools(question)
    docs = retrieve_documents(question, agent=selected_agent)
    return docs, selected_agent

def answer_with_context(question: str, context_chunks: List[str], lang: str = "pt") -> str:
    """Gera a resposta final usando a Groq Cloud."""
    start_gen = time.perf_counter()
    
    if lang == "en":
        system_prompt = (
            "You are a Senior Financial Analyst specialized in the Brazilian market. "
            "Analyze the provided context (BCB and B3 data) to answer the user.\n\n"
            "GUIDELINES:\n"
            "1. Identify trends and correlations between macro indicators and stocks.\n"
            "2. Be technical and precise. Cite exact dates and values.\n"
            "3. If the information is not in the context, clearly say the data is insufficient."
        )
    else:
        system_prompt = (
            "Você é um Analista Financeiro Sênior especializado em mercado brasileiro. "
            "Analise o contexto fornecido (dados do BCB e B3) para responder ao usuário.\n\n"
            "DIRETRIZES:\n"
            "1. Identifique tendências e correlações entre indicadores macro e ações.\n"
            "2. Seja técnico e preciso. Cite datas e valores exatos.\n"
            "3. Se a informação não constar no contexto, diga claramente que os dados atuais não permitem a conclusão."
        )

    if lang == "en":
        user_prompt = "CONTEXT:\n{context}\n\nQUESTION: {question}"
    else:
        user_prompt = "CONTEXTO:\n{context}\n\nPERGUNTA: {question}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt),
    ])
    
    llm = _build_chat_llm(temperature=0.2)
    chain = prompt | llm | StrOutputParser()
    
    logger.info(f"  └─ 🧠 Groq processando análise de {context_chunks} chunks...")
    response = chain.invoke({
        "context": "\n\n".join(context_chunks),
        "question": question
    })
    
    logger.info(f"  └─ ✨ Resposta finalizada em {time.perf_counter() - start_gen:.2f}s")
    return response

# def _build_chat_llm(temperature: float = 0.0) -> ChatOllama:
#     """Configura a instância do Ollama com os timeouts definidos."""
#     llm = ChatOllama(
#         model=settings.ollama_model,
#         base_url=settings.ollama_base_url,
#         temperature=temperature,
#     )
#     if hasattr(llm, "request_timeout"):
#         llm.request_timeout = settings.ollama_timeout
#     elif hasattr(llm, "timeout"):
#         llm.timeout = settings.ollama_timeout
#     return llm