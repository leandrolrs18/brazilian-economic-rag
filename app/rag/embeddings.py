from __future__ import annotations

from sentence_transformers import SentenceTransformer

from app.config.settings import settings


class EmbeddingModel:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self._model = SentenceTransformer(self.model_name)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [vec.tolist() for vec in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]
        return vector.tolist()
