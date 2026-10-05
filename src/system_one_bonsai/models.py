"""Pinned Hugging Face repositories of the base model. Both are Apache-2.0 and are downloaded at first use."""

# FP16 weights for PyTorch; also the tokenizer source for both backends.
TORCH_REPO = "prism-ml/Ternary-Bonsai-4B-unpacked"
TORCH_REVISION = "4485fae7a00129467b9329b738110d88b2942a1a"

# 2-bit packed weights for MLX (about 1.1 GB).
MLX_REPO = "prism-ml/Ternary-Bonsai-4B-mlx-2bit"
MLX_REVISION = "e1374ad6bf9b1b56afd743936b8faa33c409a75f"
