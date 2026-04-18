import platform
import sys


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


class CodeGenerator:
    def __init__(self, model_name: str | None = None):
        if _is_apple_silicon():
            self._init_mlx(model_name or "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit")
        else:
            self._init_transformers(model_name or "Qwen/Qwen2.5-Coder-1.5B-Instruct")

    def _init_mlx(self, model_name: str) -> None:
        from mlx_lm import load
        self._backend = "mlx"
        print(f"Loading {model_name} (MLX)...")
        self.model, self.tokenizer = load(model_name)
        print("Model loaded.")

    def _init_transformers(self, model_name: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._backend = "transformers"
        print(f"Loading {model_name} (transformers)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        load_kwargs: dict = {"device_map": "auto"}
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            print("Using 4-bit quantization (bitsandbytes).")
        except (ImportError, Exception):
            load_kwargs["torch_dtype"] = torch.float16
            print("bitsandbytes unavailable, loading in float16.")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        print("Model loaded.")

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        if self._backend == "mlx":
            return self._generate_mlx(prompt, max_tokens)
        return self._generate_transformers(prompt, max_tokens)

    def _generate_mlx(self, prompt: str, max_tokens: int) -> str:
        from mlx_lm import generate as mlx_generate
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        output = mlx_generate(
            self.model, self.tokenizer, prompt=formatted, max_tokens=max_tokens, verbose=False
        )
        return (
            output.strip()
            .replace("<|im_end|>", "")
            .replace("<|endoftext|>", "")
            .strip()
        )

    def _generate_transformers(self, prompt: str, max_tokens: int) -> str:
        import torch
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
