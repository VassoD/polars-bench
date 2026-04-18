import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.model import CodeGenerator
from src.prompt import build_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_generator: CodeGenerator | None = None
_model_ready = threading.Event()
_model_error: str | None = None


def _load_model() -> None:
    global _generator, _model_error
    try:
        log.info("Loading model...")
        _generator = CodeGenerator()
        log.info("Model ready.")
        _model_ready.set()
    except Exception as exc:
        _model_error = str(exc)
        log.error("Model load failed: %s", exc)
        _model_ready.set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_load_model, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    tables: dict


class ChatResponse(BaseModel):
    response: str


def _require_model() -> CodeGenerator:
    if _generator is None:
        _model_ready.wait(timeout=600)
    if _generator is None:
        raise HTTPException(status_code=503, detail=_model_error or "Model not loaded")
    return _generator


@app.get("/")
def root() -> dict:
    if _generator is None:
        raise HTTPException(status_code=503, detail="model loading")
    return {"status": "ok"}


@app.get("/health")
def health() -> dict:
    if _generator is None:
        raise HTTPException(status_code=503, detail="model loading")
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    gen = _require_model()
    schema = ", ".join(f"{k}: {v}" for k, v in req.tables.items())
    prompt = build_prompt(schema, req.message)
    code = gen.generate(prompt)
    log.info("Q: %s | Code: %s", req.message[:80], code[:80])
    return ChatResponse(response=code)
