import polars as pl
import signal
import threading
import re
from contextlib import contextmanager


class TimeoutError(Exception):
    pass


@contextmanager
def timeout(seconds):
    # signal.alarm only works on the main thread; skip in worker threads
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    def handler(signum, frame):
        raise TimeoutError(f"Execution exceeded {seconds}s")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def clean_code(raw: str) -> str:
    """Strip markdown fences, special tokens, and explanation prose."""
    code = raw.strip()

    # Remove special tokens
    for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        code = code.replace(tok, "")

    # Extract from markdown fence if present
    fence_match = re.search(r"```(?:python|polars)?\s*\n?(.*?)```", code, re.DOTALL)
    if fence_match:
        code = fence_match.group(1)

    return code.strip()


def execute_code(code: str, df: pl.DataFrame, timeout_s: int = 10):
    """Run the generated code. Returns (result, error_string_or_None)."""
    code = clean_code(code)
    if not code:
        return None, "Empty code"

    local_ns = {"df": df, "pl": pl}
    safe_builtins = {
        "len": len, "sum": sum, "min": min, "max": max,
        "abs": abs, "round": round, "sorted": sorted,
        "list": list, "dict": dict, "set": set, "tuple": tuple,
        "int": int, "float": float, "str": str, "bool": bool,
        "range": range, "enumerate": enumerate, "zip": zip,
        "True": True, "False": False, "None": None,
    }

    try:
        with timeout(timeout_s):
            try:
                # Try as single expression (most common case)
                result = eval(code, {"__builtins__": safe_builtins}, local_ns)
            except SyntaxError:
                # Multi-line code: exec, then look for `result` variable or last expression
                exec(code, {"__builtins__": safe_builtins}, local_ns)
                result = local_ns.get("result")
                if result is None:
                    # Grab last DataFrame-ish value from the namespace
                    for k, v in reversed(list(local_ns.items())):
                        if k not in ("df", "pl") and not k.startswith("_"):
                            result = v
                            break
        return result, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"