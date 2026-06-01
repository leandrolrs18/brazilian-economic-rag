import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path

from app.config.settings import settings
from app.routes.ask import ingest_default_series, router as ask_router

logger = logging.getLogger(__name__)


async def run_startup_ingest() -> None:
    logger.info("Ingestao automatica iniciada em background.")
    try:
        response = await asyncio.to_thread(ingest_default_series)
        logger.info(
            "Ingestao automatica finalizada: %s chunks ingeridos.",
            response.ingested_chunks,
        )
    except Exception:
        logger.exception("Falha na ingestao automatica.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.auto_ingest_on_startup:
        app.state.startup_ingest_task = asyncio.create_task(run_startup_ingest())
    yield


app = FastAPI(title="Brazil Economic RAG", lifespan=lifespan)

app.include_router(ask_router)

BASE_DIR = Path(__file__).resolve().parents[1]

@app.get("/", include_in_schema=False)
def home_pt():
    return FileResponse(BASE_DIR / "index_pt.html")

@app.get("/en", include_in_schema=False)
def home_en():
    return FileResponse(BASE_DIR / "index_en.html")
