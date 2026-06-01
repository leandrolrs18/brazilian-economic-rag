from __future__ import annotations

try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:  # fallback for older LangChain paths
    from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    from langchain_community.retrievers import BM25Retriever
except ImportError:
    from langchain.retrievers import BM25Retriever
from langchain_community.vectorstores import Qdrant

from app.config.settings import parse_hybrid_weights, settings
from app.rag.bm25_store import load_corpus
from app.rag.vector_store import get_client
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

def build_retriever(filter_source: str | None = None):
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    client = get_client()
    
    # Criamos os filtros do Qdrant se o filter_source existir
    search_kwargs = {"k": settings.dense_top_k}
    if filter_source:
        search_kwargs["filter"] = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=filter_source))]
        )

    vectorstore = Qdrant(
        client=client,
        collection_name=settings.collection_name,
        embeddings=embeddings,
    )
    
    # O retriever denso agora respeitará o filtro do Qdrant
    dense_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    if not settings.use_hybrid_search:
        return dense_retriever

    corpus = load_corpus()
    if not corpus:
        return dense_retriever

    # Filtramos o corpus do BM25 manualmente para o agente selecionado
    if filter_source:
        # Só inclui no BM25 textos que contenham a tag do agente no início
        tag = f"[agent:{filter_source}]"
        filtered_corpus = [txt for txt in corpus if txt.startswith(tag)]
    else:
        filtered_corpus = corpus

    if not filtered_corpus: # Caso o filtro resulte em nada, volta pro denso
        return dense_retriever

    bm25 = BM25Retriever.from_texts(filtered_corpus)
    bm25.k = settings.bm25_top_k

    weights = parse_hybrid_weights(settings.hybrid_weights)
    return EnsembleRetriever(
        retrievers=[dense_retriever, bm25],
        weights=weights,
    )
