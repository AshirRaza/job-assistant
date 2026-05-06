from fastapi import FastAPI

from utils.config import get_settings
from utils.helpers import setup_logger

app = FastAPI(title="CV Analyzer API", version="0.1.0")
logger = setup_logger()
settings = get_settings()


@app.get("/health")
async def health() -> dict[str, str]:
    logger.debug("Health check received.")
    return {"status": "ok"}
