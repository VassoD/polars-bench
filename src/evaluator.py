import time
import json
import polars as pl

from src.executor import execute_code
from src.prompt import build_prompt


def compare(actual, expected, type_hint: str) -> bool:
    if actual is None:
        return False
    if type_hint == "float":
        try:
            return abs(float(actual) - float(expected)) < 1e-2
        except (TypeError, ValueError):
            return False
    if type_hint == "scalar":
        return actual == expected
    if isinstance(actual, pl.DataFrame):
        return actual.to_dicts() == expected
    return actual == expected


def evaluate(eval_set_path: str, parquet_path: str, generator, max_retries: int = 1):
    df = pl.read_parquet(parquet_path)
    with open(eval_set_path) as f:
        cases = json.load(f)

    correct = 0
    total_time = 0.0
    results_log = []

    for case in cases:
        t0 = time.time()
        prompt = build_prompt(case["schema"], case["question"])
        code = generator.generate(prompt)
        result, error = execute_code(code, df)

        # Self-repair loop
        attempts = 0
        while error is not None and attempts < max_retries:
            repair_prompt = (
                f"{prompt}\n{code}\n\n"
                f"That code failed with error: {error}\n"
                f"Corrected code (expression only):"
            )
            code = generator.generate(repair_prompt)
            result, error = execute_code(code, df)
            attempts += 1

        elapsed = time.time() - t0
        total_time += elapsed

        is_correct = compare(result, case["expected"], case.get("type", "scalar"))
        correct += int(is_correct)

        mark = "✓" if is_correct else "✗"
        print(f"{mark} {case['id']} ({elapsed:.1f}s) | code: {code[:80]}")
        if not is_correct:
            print(f"    expected={case['expected']}, got={result}, err={error}")

        results_log.append({
            "id": case["id"],
            "correct": is_correct,
            "time": elapsed,
            "code": code,
            "error": error,
        })

    print(f"\n{'='*60}")
    print(f"N = {correct}/{len(cases)}   T = {total_time:.1f}s")
    print(f"Approx score proxy = N/T = {correct/total_time:.4f}")
    print(f"{'='*60}")

    return results_log