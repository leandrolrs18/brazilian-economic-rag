# force_full_ingest.py (na raiz)
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from app.routes.ask import run_ingest
from app.config.settings import settings

print("🚀 Iniciando ingestão forçada (B3 + BCB)...")

# Pega os símbolos e séries configurados no .env
symbols = [s.strip() for s in settings.b3_symbols.split(",") if s.strip()]
series_ids = [s.strip() for s in settings.bcb_series.split(",") if s.strip()]

print(f"📈 Símbolos B3: {symbols}")
print(f"🏦 Séries BCB: {series_ids}")

try:
    response = run_ingest(series_ids=series_ids, symbols=symbols)
    print("\n✅ Ingest concluído com sucesso!")
    print(f"📦 Total de chunks armazenados no Qdrant: {response.ingested_chunks}")
except Exception as e:
    print(f"\n❌ Erro crítico durante o ingest: {e}")