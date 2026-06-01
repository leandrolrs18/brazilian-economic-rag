from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import logging
from time import sleep
from typing import Iterable

import requests

from app.config.settings import settings

logger = logging.getLogger(__name__)

SGS_BASE_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados?formato=json"
SGS_CSV_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados?formato=csv"

SERIES_LABELS = {
    "432": "IPCA (Inflação) - índice",
    "433": "IPCA (Inflação) - variação %",
    "1": "Taxa de câmbio - dólar (venda)",
    "12": "Taxa Selic - meta ao mês",
    "10813": "Taxa Selic - meta ao ano",
    "1208": "PIB mensal - índice",
    "4380": "PIB acumulado em 12 meses - variação %",
}


@dataclass
class SeriesData:
    series_id: str
    title: str
    rows: list[dict]


def _parse_csv_payload(payload: str) -> list[dict]:
    lines = [line.strip() for line in payload.splitlines() if line.strip()]
    if not lines:
        return []
    # Expected header: data;valor
    rows = []
    for line in lines[1:]:
        parts = line.split(";")
        if len(parts) < 2:
            continue
        rows.append({"data": parts[0].strip(), "valor": parts[1].strip()})
    return rows


def _parse_date(raw: str) -> date | None:
    if not raw:
        return None
    cleaned = raw.strip().strip('"').strip("'")
    try:
        return datetime.strptime(cleaned, "%d/%m/%Y").date()
    except ValueError:
        return None


def _fetch_series(series_id: str) -> SeriesData:
    url = SGS_BASE_URL.format(series_id=series_id)
    csv_url = SGS_CSV_URL.format(series_id=series_id)
    if settings.series_window_months and settings.series_window_months > 0:
        today = date.today()
        # Calcula 5 meses para trás (aprox. 150 dias)
        # Isso garante que se hoje é Abril/2026, ele pegue desde Novembro/2025
        start_date = today - timedelta(days=45)
        
        start_str = start_date.strftime("%d/%m/%Y")
        end_str = today.strftime("%d/%m/%Y")
        
        url = f"{url}&dataInicial={start_str}&dataFinal={end_str}"
        csv_url = f"{csv_url}&dataInicial={start_str}&dataFinal={end_str}"
                

    headers_variants = [
        {
            "Accept": "application/json; charset=utf-8",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        },
        {
            "Accept": "application/json, text/plain, */*",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        },
        {
            "Accept": "*/*",
            "User-Agent": "brazil-economic-rag/1.0",
        },
    ]
    last_exc: Exception | None = None
    for headers in headers_variants:
        # Retry a few times for transient upstream errors
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=30, headers=headers)
                response.raise_for_status()
                try:
                    rows = response.json()
                except ValueError:
                    rows = None
                if isinstance(rows, str):
                    rows = _parse_csv_payload(rows)
                if not isinstance(rows, list) or (rows and not isinstance(rows[0], dict)):
                    # Fallback: try CSV explicitly (sem Accept para evitar 406)
                    csv_headers = {k: v for k, v in headers.items() if k.lower() != "accept"}
                    csv_headers = csv_headers or None
                    csv_response = requests.get(csv_url, timeout=30, headers=csv_headers)
                    csv_response.raise_for_status()
                    rows = _parse_csv_payload(csv_response.text)
                if not isinstance(rows, list):
                    raise ValueError("Resposta inesperada do BCB")
                title = SERIES_LABELS.get(series_id, f"Série {series_id}")
                return SeriesData(series_id=series_id, title=title, rows=rows)
            except requests.HTTPError as exc:
                last_exc = exc
                status = exc.response.status_code if exc.response is not None else None
                if status in (502, 503, 504):
                    sleep(1.5 * (attempt + 1))
                    continue
                if status == 406:
                    # Algumas séries diárias exigem janela de consulta (máx. 10 anos).
                    # Tente novamente com dataInicial/dataFinal e, se falhar, CSV.
                    today = date.today()
                    # Limita a janela para reduzir payload (ex.: séries diárias)
                    start = today - timedelta(days=settings.daily_window_days)
                    start_str = start.strftime("%d/%m/%Y")
                    end_str = today.strftime("%d/%m/%Y")
                    dated_url = f"{url}&dataInicial={start_str}&dataFinal={end_str}"
                    dated_csv_url = f"{csv_url}&dataInicial={start_str}&dataFinal={end_str}"
                    try:
                        response = requests.get(dated_url, timeout=30, headers=headers)
                        response.raise_for_status()
                        rows = response.json()
                        if isinstance(rows, str):
                            rows = _parse_csv_payload(rows)
                        if not isinstance(rows, list):
                            raise ValueError("Resposta inesperada do BCB")
                        title = SERIES_LABELS.get(series_id, f"Série {series_id}")
                        return SeriesData(series_id=series_id, title=title, rows=rows)
                    except requests.RequestException:
                        pass
                    # Fallback: CSV com janela de datas
                    try:
                        csv_headers = {k: v for k, v in headers.items() if k.lower() != "accept"}
                        csv_headers = csv_headers or None
                        csv_response = requests.get(dated_csv_url, timeout=30, headers=csv_headers)
                        csv_response.raise_for_status()
                        rows = _parse_csv_payload(csv_response.text)
                        title = SERIES_LABELS.get(series_id, f"Série {series_id}")
                        return SeriesData(series_id=series_id, title=title, rows=rows)
                    except requests.RequestException:
                        pass
                    # Try next header variant
                    break
                raise
            except requests.RequestException as exc:
                last_exc = exc
                sleep(1.0 * (attempt + 1))
                continue
    if last_exc:
        raise last_exc
    raise RuntimeError("Falha desconhecida ao consultar o BCB")


def fetch_series(series_ids: Iterable[str]) -> list[SeriesData]:
    series_list = []
    for series_id in series_ids:
        try:
            series_list.append(_fetch_series(series_id))
        except requests.HTTPError as exc:
            logger.warning("Ignorando serie SGS %s por erro HTTP: %s", series_id, exc)
        except requests.RequestException as exc:
            logger.warning("Ignorando serie SGS %s por erro de rede: %s", series_id, exc)
    return series_list


def series_to_text(series: SeriesData) -> str:
    rows = series.rows
    
    # 1. Filtra a quantidade de linhas primeiro
    if settings.max_rows_per_series and len(rows) > settings.max_rows_per_series:
        parsed = [(d, row) for row in rows if (d := _parse_date(row.get("data", "")))]
        if parsed:
            parsed.sort(key=lambda item: item[0])
            rows = [row for _, row in parsed][-settings.max_rows_per_series :]
        else:
            rows = rows[-settings.max_rows_per_series :]

    # 2. Cálculos estatísticos (Protegido contra erro de divisão por zero)
    try:
        valores = [float(r['valor'].replace(',', '.')) for r in rows if r.get('valor')]
        v_min, v_max = min(valores), max(valores)
        v_avg = sum(valores) / len(valores)
        resumo = f"Resumo do Período: Mín {v_min:.2f} | Máx {v_max:.2f} | Média {v_avg:.2f}"
    except (ValueError, ZeroDivisionError):
        resumo = "Resumo: Dados insuficientes para cálculos."

    # 3. Montagem do texto (Resumo primeiro, depois os dados)
    lines = [f"{series.title} (SGS {series.series_id})"]
    lines.append(resumo)
    lines.append("--- Histórico de Dados ---")
    
    for row in rows:
        data = row.get("data", "")
        valor = row.get("valor", "")
        lines.append(f"Data: {data} | Valor: {valor}")
    
    return "\n".join(lines)


def build_corpus(series_ids: Iterable[str]) -> list[str]:
    corpus = []
    for series in fetch_series(series_ids):
        corpus.append(series_to_text(series))
    return corpus
