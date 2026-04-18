# Polars Bench Submission

Text-to-Polars code generation using Qwen2.5-Coder (MLX backend, 4-bit quantization).

## Approach

- **Model:** `mlx-community/Qwen2.5-Coder-7B-Instruct-4bit` (or 3B variant for speed)
- **Prompting:** System instruction with Polars-specific syntax rules + 5 carefully chosen few-shot examples targeting common LLM failure modes (date handling, sort direction, membership tests, scalar extraction)
- **Self-repair loop:** If generated code throws an exception, the error is fed back to the model for one retry
- **Output parsing:** Strips markdown fences, special tokens (`<|im_end|>`), and extracts executable expression

## Results on local eval set

16/16 correct in ~39s (N/T ≈ 0.41)

## Setup

**Apple Silicon (MLX backend):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-apple.txt
python data/make_data.py
```

**Linux / CUDA (transformers + bitsandbytes backend):**
```bash
python3 -m venv .venv
source .venv/bin/activate
# Install torch with the CUDA version matching your driver (example: cu121)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python data/make_data.py
```

## Run

**Benchmark server (used by the platform runner):**
```bash
bash start.sh
# or: uvicorn server:app --host 0.0.0.0 --port 9000
```

The server exposes:
- `POST /predict` — receives `{question_id, question, schema, data_path?, data_b64?}`, returns `{question_id, answer}`
- `GET /health` — readiness probe

**Local eval loop (development only):**
```bash
python run.py
```

The model backend is selected automatically: MLX on Apple Silicon, transformers (4-bit via bitsandbytes, float16 fallback) on Linux/CUDA.

## Files

- `server.py` — FastAPI inference server (benchmark entrypoint)
- `start.sh` — Starts the server on port 8000
- `src/model.py` — Code generator (MLX on Apple Silicon, transformers on Linux/CUDA)
- `src/prompt.py` — System instruction + few-shot examples
- `src/executor.py` — Safe code execution with timeout and output cleanup
- `src/evaluator.py` — Eval loop with self-repair retry
- `run.py` — Local evaluation script (development only)
- `data/eval_set.json` — Ground-truth test cases
- `data/make_data.py` — Generates synthetic sales parquet

## Key optimizations

1. **Targeted few-shots** — each example fixes a specific Polars footgun (`.dt.month()` parens, `.is_in()` vs `.isin()`, `descending=True` not `ascending=True`)
2. **Explicit syntax rules in system prompt** — cheaper than adding more few-shots
3. **Self-repair** — catches transient generation errors with one retry
4. **`max_tokens=200`** — covers complex group-by chains without over-generating

## Notes

- Developed on Apple Silicon (M-series). `src/model.py` auto-selects MLX on Apple Silicon and `transformers` (4-bit via bitsandbytes, float16 fallback) on Linux/CUDA.
- Linux target model: `Qwen/Qwen2.5-Coder-7B-Instruct` (same model family, downloaded from HuggingFace Hub on first run).
