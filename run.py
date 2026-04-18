from src.model import CodeGenerator
from src.evaluator import evaluate

if __name__ == "__main__":
    gen = CodeGenerator()
    evaluate(
        eval_set_path="data/eval_set.json",
        parquet_path="data/sales.parquet",
        generator=gen,
        max_retries=1,
    )