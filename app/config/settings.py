from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    collection_name: str = "bcb_economic_series"

    # Ollama (local LLM)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_timeout: int = 180

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1500
    chunk_overlap: int = 200
    ipca_chunk_size: int = 500
    ipca_chunk_overlap: int = 100
    top_k: int = 20

    # Hybrid retrieval + reranking
    use_hybrid_search: bool = True
    dense_top_k: int = 20
    bm25_top_k: int = 10
    hybrid_weights: str = "0.7,0.3"  # dense,bm25
    bm25_store_path: str = "data/bm25_corpus.json"
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 15
    rerank_top_k: int = 3

    # Comma-separated SGS series IDs
    bcb_series: str = "432,433,1,12,10813,1208,4380"
    b3_symbols: str = "PETR4.SA,VALE3.SA,ITUB4.SA,BBDC4.SA,ABEV3.SA"
    max_rows_per_series: int = 120
    daily_window_days: int = 180
    series_window_months: int = 4

    # Hugging Face Spaces startup
    auto_ingest_on_startup: bool = True


settings = Settings()


def parse_hybrid_weights(raw: str) -> list[float]:
    try:
        parts = [float(p.strip()) for p in raw.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError
        return parts
    except ValueError:
        return [0.6, 0.4]
