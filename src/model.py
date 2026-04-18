from mlx_lm import load, generate as mlx_generate


class CodeGenerator:
    def __init__(self, model_name: str = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"):
        print(f"Loading {model_name}...")
        self.model, self.tokenizer = load(model_name)
        print("Model loaded.")

    def generate(self, prompt: str, max_tokens: int = 150) -> str:
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        output = mlx_generate(
            self.model,
            self.tokenizer,
            prompt=formatted,
            max_tokens=max_tokens,
            verbose=False,
        )
        return (
            output.strip()
            .replace("<|im_end|>", "")
            .replace("<|endoftext|>", "")
            .strip()
        )