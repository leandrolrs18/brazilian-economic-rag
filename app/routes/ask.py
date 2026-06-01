from __future__ import annotations

import uuid
import re
from datetime import datetime
from typing import List, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
from qdrant_client.http.models import PointStruct
from requests import HTTPError, RequestException

from app.config.settings import settings
from app.loaders.b3_yahoo_loader import build_stocks_corpus
from app.loaders.bcb_loader import build_corpus
from app.rag.bm25_store import load_corpus, save_corpus
from app.rag.embeddings import EmbeddingModel
from app.rag.langchain_rag import answer_with_context, retrieve_documents_with_router
from app.rag.vector_store import ensure_collection, get_client

import time
import logging

router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# 📅 EXTRAÇÃO DE DATA
# =========================
MONTHS = {
    "janeiro": 1, "jan": 1,
    "fevereiro": 2, "fev": 2,
    "março": 3, "mar": 3,
    "abril": 4, "abr": 4,
    "maio": 5,
    "junho": 6, "jun": 6,
    "julho": 7, "jul": 7,
    "agosto": 8, "ago": 8,
    "setembro": 9, "set": 9,
    "outubro": 10, "out": 10,
    "novembro": 11, "nov": 11,
    "dezembro": 12, "dez": 12,
}


def extract_month_year(question: str):
    q = question.lower()

    for name, month in MONTHS.items():
        if name in q:
            match = re.search(r"(20\d{2})", q)
            if match:
                return month, int(match.group(1))

    return None, None


def filter_chunks_by_date(chunks: list[str], month: int, year: int) -> list[str]:
    filtered = []

    for chunk in chunks:
        matches = re.findall(r"(\d{2})/(\d{2})/(\d{4})", chunk)

        for d, m, y in matches:
            if int(m) == month and int(y) == year:
                filtered.append(chunk)
                break

    return filtered


# =========================
# 📦 MODELS
# =========================
class IngestResponse(BaseModel):
    ingested_chunks: int
    series_ids: List[str]


class AskRequest(BaseModel):
    question: str
    lang: str = "pt"


class AskResponse(BaseModel):
    answer: str
    context: List[str]
    agent: Literal["bcb", "b3"]
    lang: str = "pt"

def ingest_default_series() -> IngestResponse:
    series_ids = [s.strip() for s in settings.bcb_series.split(",") if s.strip()]
    symbols = [s.strip() for s in settings.b3_symbols.split(",") if s.strip()]
    return run_ingest(series_ids, symbols)


def is_ingest_ready() -> bool:
    bm25_corpus = load_corpus()
    if not bm25_corpus:
        return False

    has_bcb_chunks = any("[agent:bcb]" in chunk for chunk in bm25_corpus)
    if not has_bcb_chunks:
        return False

    client = get_client()
    collections = client.get_collections().collections
    has_collection = any(c.name == settings.collection_name for c in collections)

    if not has_collection:
        return False

    return True


@router.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    return ingest_default_series()

# =========================
# ❤️ HEALTH
# =========================
@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


# =========================
# 🔪 CHUNK BUILDER
# =========================
def _build_chunks(corpus: List[str], source: str) -> list[str]:
    prefixed = [f"[agent:{source}]\n{text}" for text in corpus]

    if source == "b3":
        # Em vez de juntar tudo e separar por tamanho, 
        # trate cada item do corpus (cada ação) como um chunk individual
        return corpus

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    ipca_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.ipca_chunk_size,
        chunk_overlap=settings.ipca_chunk_overlap,
    )

    chunks: list[str] = []

    for text in prefixed:
        if "SGS 432" in text or "SGS 433" in text:
            chunks.extend(ipca_splitter.split_text(text))
        else:
            chunks.extend(splitter.split_text(text))

    return [chunk for chunk in chunks if chunk.strip()]


# =========================
# 🚀 INGEST
# =========================
def run_ingest(series_ids: List[str], symbols: List[str]) -> IngestResponse:
    if not series_ids and not symbols:
        raise HTTPException(status_code=400, detail="Nenhuma fonte configurada para ingest")

    # BCB
    try:
        bcb_corpus = build_corpus(series_ids) if series_ids else []
        bcb_corpus = [t for t in bcb_corpus if t and isinstance(t, str) and len(t.strip()) > 0]
    except (ValueError, HTTPError) as exc:
        raise HTTPException(status_code=502, detail=f"BCB erro: {exc}")
    except RequestException:
        raise HTTPException(status_code=503, detail="Erro ao acessar BCB")

    # B3
    try:
        b3_corpus = build_stocks_corpus(symbols) if symbols else []
        b3_corpus = [t for t in b3_corpus if t and isinstance(t, str) and len(t.strip()) > 0]
    except Exception as e:
        logger.warning(f"Erro Yahoo: {e}")
        b3_corpus = []

    chunks_bcb = _build_chunks(bcb_corpus, "bcb")
    chunks_b3 = _build_chunks(b3_corpus, "b3")

    all_chunks = [c for c in (chunks_bcb + chunks_b3) if c.strip()]

    if not all_chunks:
        return IngestResponse(ingested_chunks=0, series_ids=series_ids)

    embedder = EmbeddingModel()
    vectors = embedder.embed_texts(all_chunks)

    client = get_client()
    ensure_collection(client, vector_size=embedder.dimension)

    # limpa antigo
    for source in ("bcb", "b3"):
        client.delete(
            collection_name=settings.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            ),
        )

    points = []

    for text, vector in zip(all_chunks, vectors):
        source = "b3" if "[agent:b3]" in text else "bcb"

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": text, "source": source},
            )
        )

    client.upsert(collection_name=settings.collection_name, points=points)
    save_corpus(all_chunks)

    return IngestResponse(ingested_chunks=len(all_chunks), series_ids=series_ids)


# =========================
# ❓ ASK
# =========================
@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    start_total = time.perf_counter()

    try:
        docs, selected_agent = retrieve_documents_with_router(payload.question)

        if not docs:
            return AskResponse(
                answer="No sufficient context found." if payload.lang == "en"
                    else "Não encontrei contexto suficiente.",
                context=[],
                agent=selected_agent,
                lang=payload.lang
            )

        context_chunks = [doc.page_content for doc in docs]

        # 🔥 FILTRO TEMPORAL
        month, year = extract_month_year(payload.question)

        if month and year:
            filtered = filter_chunks_by_date(context_chunks, month, year)

            if filtered:
                logger.info(f"📅 Filtro aplicado: {month}/{year} ({len(filtered)} chunks)")
                context_chunks = filtered
            else:
                logger.warning("⚠️ Nenhum dado encontrado pro período")

        answer = answer_with_context(payload.question, context_chunks, lang=payload.lang)

        return AskResponse(
            answer=answer,
            context=context_chunks,
            agent=selected_agent,
            lang=payload.lang
        )

    except Exception as exc:
        logger.error(f"Erro: {str(exc)}")
        raise HTTPException(status_code=500, detail=str(exc))
