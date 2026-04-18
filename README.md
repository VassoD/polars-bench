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

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python data/make_data.py
```

## Run

```bash
python run.py
```

## Files

- `src/model.py` — MLX-based code generator
- `src/prompt.py` — System instruction + few-shot examples
- `src/executor.py` — Safe code execution with timeout and output cleanup
- `src/evaluator.py` — Eval loop with self-repair retry
- `data/eval_set.json` — Ground-truth test cases
- `data/make_data.py` — Generates synthetic sales parquet

## Key optimizations

1. **Targeted few-shots** — each example fixes a specific Polars footgun (`.dt.month()` parens, `.is_in()` vs `.isin()`, `descending=True` not `ascending=True`)
2. **Explicit syntax rules in system prompt** — cheaper than adding more few-shots
3. **Self-repair** — catches transient generation errors with one retry
4. **Tight `max_tokens=150`** — Polars one-liners don't need more

## Notes

- Developed on Apple Silicon (M-series) using MLX. For CUDA targets, swap `src/model.py` to use `transformers` or `vllm` with the same `CodeGenerator` interface.
