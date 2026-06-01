from __future__ import annotations

from typing import List

from sentence_transformers import CrossEncoder

from app.config.settings import settings


class Reranker:
    def __init__(self) -> None:
        self._model = CrossEncoder(settings.rerank_model)

    def rerank(self, query: str, docs: List) -> List:
        if not docs:
            return []
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self._model.predict(pairs)
        scored = list(zip(scores, docs))
        scored.sort(key=lambda item: item[0], reverse=True)
        top_k = settings.rerank_top_k or len(scored)
        return [doc for _, doc in scored[:top_k]]
