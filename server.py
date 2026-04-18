import base64
import io
import logging
from contextlib import asynccontextmanager
from typing import Any

import polars as pl
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.executor import execute_code
from src.model import CodeGenerator
from src.prompt import build_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_generator: CodeGenerator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _generator
    log.info("Loading model...")
    _generator = CodeGenerator()
    log.info("Model ready.")
    yield


app = FastAPI(lifespan=lifespan)


class PredictRequest(BaseModel):
    model_config = {"populate_by_name": True}
    question_id: str = ""
    question: str
    df_schema: str = Field(..., alias="schema")
    data_path: str | None = None
    data_b64: str | None = None


class PredictResponse(BaseModel):
    question_id: str
    answer: Any


def _load_df(req: PredictRequest) -> pl.DataFrame:
    if req.data_b64:
        return pl.read_parquet(io.BytesIO(base64.b64decode(req.data_b64)))
    return pl.read_parquet(req.data_path or "data/sales.parquet")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _generator is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    log.info("Q[%s]: %s", req.question_id, req.question)
    df = _load_df(req)
    prompt = build_prompt(req.df_schema, req.question)

    code = _generator.generate(prompt)
    result, error = execute_code(code, df)

    if error is not None:
        log.warning("Self-repair triggered: %s", error)
        repair_prompt = (
            f"{prompt}\n{code}\n\n"
            f"That code failed with error: {error}\n"
            f"Corrected code (expression only):"
        )
        code = _generator.generate(repair_prompt)
        result, error = execute_code(code, df)

    if isinstance(result, pl.DataFrame):
        answer = result.to_dicts()
    else:
        answer = result

    log.info("A[%s]: %s (err=%s)", req.question_id, answer, error)
    return PredictResponse(question_id=req.question_id, answer=answer)
