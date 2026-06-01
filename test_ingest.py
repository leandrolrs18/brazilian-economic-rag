# test_ingest.py  (arquivo na raiz do projeto)

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

print("Diretório raiz:", root_dir)

# Imports corrigidos - ask.py está em app/routes/
from app.routes.ask import is_ingest_ready, load_corpus
from app.rag.vector_store import get_client
from app.config.settings import settings
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

print("\n=== DIAGNÓSTICO DE INGEST - Brazil Economic RAG ===\n")

print("is_ingest_ready() →", is_ingest_ready())

corpus = load_corpus()
print(f"\nTotal de chunks no BM25: {len(corpus)}")

bcb_count = sum(1 for c in corpus if "[agent:bcb]" in c)
b3_count = sum(1 for c in corpus if "[agent:b3]" in c)

print(f"Chunks BCB: {bcb_count}")
print(f"Chunks B3 (ações): {b3_count}")

if b3_count > 0:
    print("\n✅ Dados das ações B3 foram carregados!")
    # Mostra exemplo
    for chunk in corpus:
        if "[agent:b3]" in chunk:
            print("\nPrimeiro chunk B3 (primeiros 600 caracteres):")
            print(repr(chunk[:600] + "..."))
            break
else:
    print("\n❌ Nenhum chunk de ações B3 encontrado. Problema no Yahoo Finance.")

# Verificação no Qdrant
client = get_client()
try:
    total = client.count(collection_name=settings.collection_name, exact=False).count
    print(f"\nTotal de pontos no Qdrant: {total}")

    b3_filter = Filter(must=[FieldCondition(key="source", match=MatchValue(value="b3"))])
    b3_qdrant = client.count(
        collection_name=settings.collection_name,
        count_filter=b3_filter,
        exact=False
    ).count
    print(f"Pontos B3 no Qdrant: {b3_qdrant}")

except Exception as e:
    print(f"\nErro ao consultar Qdrant: {e}")

print("\n=== Diagnóstico finalizado ===")