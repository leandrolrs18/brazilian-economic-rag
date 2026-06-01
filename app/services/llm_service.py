from __future__ import annotations

import requests

from app.config.settings import settings


class LLMService:
    def answer(self, question: str, context: str) -> str:
        system_prompt = (
            "Você é um assistente econômico. Use apenas o contexto fornecido para responder. "
            "Se faltar informação, diga que não encontrou dados suficientes no Banco Central."
        )
        payload = {
            "model": settings.ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Contexto:\n{context}\n\nPergunta: {question}"},
            ],
            "options": {"temperature": 0.2},
            "stream": False,
        }
        response = requests.post(
            f"{settings.ollama_base_url}/api/chat",
            json=payload,
            timeout=settings.ollama_timeout,
        )
        response.raise_for_status()
        data = response.json()
        message = data.get("message", {})
        return message.get("content", "")
