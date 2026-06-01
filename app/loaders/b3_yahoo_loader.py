from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from time import sleep
from typing import Iterable

import requests
import logging

from app.config.settings import settings

# Configuração de logging para debug
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

@dataclass
class StockSeriesData:
    symbol: str
    rows: list[dict]

def _fetch_symbol(symbol: str) -> StockSeriesData:
    """Busca dados históricos do Yahoo Finance respeitando a janela do settings.py"""
    # 1. Calcula a janela de tempo baseada nas configurações
    months = max(settings.series_window_months, 1)
    today = datetime.now()
    # Subtrai 150 dias (5 meses x 30 dias) para garantir a janela
    start_date = today - timedelta(days=45) 

    start_str = start_date.strftime("%d/%m/%Y")
    end_str = today.strftime("%d/%m/%Y")

    logger.info(f"📅 Janela de 5 meses definida: {start_str} até {end_str}")

    params = {
        "period1": int(start_date.timestamp()),
        "period2": int(today.timestamp()), # Até hoje
        "interval": "1d",
    }
        
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "*/*",
        "Connection": "keep-alive"
    }

    logger.info(f"📥 Buscando {symbol} de {start_date.strftime('%d/%m/%Y')} até {today.strftime('%d/%m/%Y')}")

    for attempt in range(4):
        try:
            response = requests.get(
                YAHOO_CHART_URL.format(symbol=symbol),
                params=params,
                timeout=30,
                headers=headers,
            )
            if response.status_code == 429:
                sleep(2 * (attempt + 1))
                continue
            
            response.raise_for_status()
            payload = response.json()
            
            result = payload.get("chart", {}).get("result", [])
            if not result:
                return StockSeriesData(symbol=symbol, rows=[])

            timestamps = result[0].get("timestamp", [])
            indicators = result[0].get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])
            
            rows = []
            for ts, val in zip(timestamps, indicators):
                if val is not None:
                    dt = datetime.fromtimestamp(ts).strftime("%d/%m/%Y")
                    rows.append({"data": dt, "valor": round(val, 2)})
            
            return StockSeriesData(symbol=symbol, rows=rows)

        except Exception as e:
            logger.error(f"❌ Erro na tentativa {attempt+1} para {symbol}: {e}")
            if attempt == 3:
                return StockSeriesData(symbol=symbol, rows=[])
            sleep(1)

def stock_series_to_text(series: StockSeriesData) -> str:
    """Converte os dados em texto estruturado com injeção de contexto por linha."""
    rows = series.rows

    # 1. Filtra a quantidade de linhas
    if settings.max_rows_per_series and len(rows) > settings.max_rows_per_series:
        rows = rows[-settings.max_rows_per_series :]

    # 2. Cálculos estatísticos
    try:
        valores = [float(str(r['valor']).replace(',', '.')) for r in rows if r.get('valor')]
        if valores:
            v_min, v_max = min(valores), max(valores)
            v_avg = sum(valores) / len(valores)
            resumo = f"Resumo do Período ({series.symbol}): Mín {v_min:.2f} | Máx {v_max:.2f} | Média {v_avg:.2f}"
        else:
            resumo = f"Resumo ({series.symbol}): Dados indisponíveis."
    except (ValueError, ZeroDivisionError):
        resumo = f"Resumo ({series.symbol}): Erro ao processar valores."

    # 3. Montagem do texto com identificador de agente e metadados repetidos
    final_lines = [f"[agent:b3]"]
    final_lines.append(f"Acao B3 ({series.symbol}) - fechamento diario")
    final_lines.append(resumo)
    final_lines.append(f"--- Histórico de Preços de {series.symbol} ---")
    
    for row in rows:
        # Repetir o símbolo em cada linha é crucial para a recuperação (Retrieval)
        final_lines.append(f"Ativo: {series.symbol} | Data: {row['data']} | Fechamento: {row['valor']}")
    
    return "\n".join(final_lines)

def build_stocks_corpus(symbols: Iterable[str]) -> list[str]:
    corpus = []
    for sym in symbols:
        data = _fetch_symbol(sym)
        if data.rows:
            # Cada ação vira uma string independente na lista
            text = stock_series_to_text(data)
            corpus.append(text)
    return corpus