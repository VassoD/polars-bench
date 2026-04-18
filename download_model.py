from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"

print(f"Downloading {MODEL_ID}...")
path = snapshot_download(
    repo_id=MODEL_ID,
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
)
print(f"Model cached at: {path}")
