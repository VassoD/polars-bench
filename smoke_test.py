from mlx_lm import load, generate

print("Downloading + loading model (first run takes a few minutes)...")
model, tokenizer = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
print("Model loaded.")

prompt = """Generate ONLY Polars code, no explanation, no markdown fences.
Schema: df has columns: name (str), age (i64), city (str)
Question: How many people live in Paris?
Code:"""

messages = [{"role": "user", "content": prompt}]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

output = generate(model, tokenizer, prompt=formatted, max_tokens=128, verbose=True)
print("\n\n=== OUTPUT ===")
print(output)