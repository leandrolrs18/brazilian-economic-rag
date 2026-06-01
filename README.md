---
title: Brazil Economic RAG
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Brazil Economic RAG (Project Complet)

Este projeto mantém o app original intacto e adiciona:
- Orquestração com LangChain
- Hybrid search (dense + BM25)
- Reranking com cross-encoder
- Pipeline de fine-tuning (pasta `fine_tuning/`)

## Execução rápida
```
uvicorn app.main:app --reload
```

## Deploy no Hugging Face Spaces

1. Crie um Space em https://huggingface.co/spaces/new
2. Escolha `Docker` como SDK.
3. Configure o secret `GROQ_API_KEY` em `Settings > Repository secrets`.
4. Suba este diretório para o Space:
   ```
   cd /Users/leandro/Desktop/brazil-economic-rag/project_complet
   git init
   git branch -M main
   git add .
   git commit -m "Deploy Brazil Economic RAG"
   git remote add space https://huggingface.co/spaces/SEU_USUARIO/SEU_SPACE
   git push --force space main
   ```
5. O Space roda a ingestão automaticamente em background sempre que iniciar. Se quiser forçar manualmente:
   ```
   curl -X POST https://SEU_USUARIO-SEU_SPACE.hf.space/ingest
   ```

Para persistir o Qdrant entre reinicializações, habilite `Persistent storage` no Space. O Dockerfile usa `/data/qdrant` e `/data/bm25_corpus.json`.

## Fluxo do RAG
1. `POST /ingest` cria chunks, embeddings e grava no Qdrant.
2. `POST /ask` faz hybrid search, reranking e gera resposta com Groq.

## Configurações úteis
Arquivo: `/Users/leandro/Desktop/brazil-economic-rag/project_complet/app/config/settings.py`
- `use_hybrid_search`
- `dense_top_k` / `bm25_top_k`
- `rerank_enabled` / `rerank_model`
- `bm25_store_path`
- `groq_model`
- `auto_ingest_on_startup`

## Troubleshooting rápido
- Se o Qdrant do Docker falhar (erro de storage), use modo local:
  ```
  QDRANT_URL=":memory:" python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
  ```
- Se o Space retornar erro de autenticação da Groq, confirme se `GROQ_API_KEY` foi configurada em `Settings > Repository secrets`.
