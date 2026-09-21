import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

load_dotenv(override=True)

from agent import LAST_REPORT, run_bot  # noqa: E402
from tutor.config import Settings  # noqa: E402

settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY is not set; the agent will fail to connect to OpenAI.")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    try:
        await run_bot(websocket, settings)
    except Exception as e:
        logger.exception(f"Exception in run_bot: {e}")


@app.post("/connect")
async def bot_connect(request: Request) -> Dict[Any, Any]:
    host = request.headers.get("host", f"localhost:{settings.port}").split(":")[0]
    return {"ws_url": f"ws://{host}:{settings.port}/ws"}


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"ok": True, "llm": settings.llm_model, "stt": settings.stt_model, "tts": settings.tts_model}


@app.get("/report/latest")
async def latest_report() -> Dict[str, Any]:
    return LAST_REPORT or {"empty": True}


async def main():
    config = uvicorn.Config(app, host=settings.host, port=settings.port)
    server = uvicorn.Server(config)
    try:
        await server.serve()
    except asyncio.CancelledError:
        logger.info("Server shutdown.")


if __name__ == "__main__":
    asyncio.run(main())
