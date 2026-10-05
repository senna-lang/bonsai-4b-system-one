# Third-party components

This repository contains only project-authored source code (Apache-2.0, see `LICENSE`).
No third-party source code, model weights, or datasets are included.

| Component | Use | License | Included here |
| --- | --- | --- | --- |
| [prism-ml/Ternary-Bonsai-4B-mlx-2bit](https://huggingface.co/prism-ml/Ternary-Bonsai-4B-mlx-2bit) (PrismML) | Base model for MLX, downloaded at runtime (pinned revision) | Apache-2.0 | No |
| [prism-ml/Ternary-Bonsai-4B-unpacked](https://huggingface.co/prism-ml/Ternary-Bonsai-4B-unpacked) (PrismML) | Base model for PyTorch and tokenizer for both backends, downloaded at runtime (pinned revision) | Apache-2.0 | No |
| [PyTorch](https://github.com/pytorch/pytorch), [Transformers](https://github.com/huggingface/transformers), [huggingface_hub](https://github.com/huggingface/huggingface_hub) | Runtime dependencies (imported) | BSD-3-Clause / Apache-2.0 | No |
| [MLX](https://github.com/ml-explore/mlx), [MLX-LM](https://github.com/ml-explore/mlx-lm) | Apple Silicon runtime dependencies (imported) | MIT | No |

The YaRN frequency computation in `inference_mlx.py` reimplements the formula of Transformers' `_compute_yarn_parameters`; its output matches Transformers 4.57.6 bit for bit for this model.

This project is not affiliated with or endorsed by PrismML, Alibaba Cloud/Qwen, or TypeSafe.
