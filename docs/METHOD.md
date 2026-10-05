# Method

## Request format

A request is one `state` (the record) and one or more branches. Each branch is rendered as a separate chat prompt (Ternary-Bonsai-4B uses the Qwen3 chat template):

```
Read the record and answer the question. Respond with exactly one listed letter and no other text.

Record:
<state>

Question: <instructions>
Answers:
A. <option 1>
B. <option 2>
...

Answer:
```

`noul` branches always use `A. No` / `B. Yes`. The prompt is passed through the model's chat template with `enable_thinking=False` and a generation prompt.

## Packing

`pack_request` (`src/system_one/packing.py`) tokenizes every branch prompt, takes the longest common token prefix of all of them as the shared state, and appends each prompt's remaining suffix as a branch. At least one token per branch stays in its suffix. The result is:

- `input_ids`: shared prefix followed by every branch suffix
- `state_len`: length of the shared prefix
- `spans`: the inclusive token range of each branch

## Tree attention mask and positions

Token *i* may attend to token *j* only if *j ≤ i* and *j* is either in the shared prefix or in the same branch as *i*. Position IDs run `0 … state_len-1` over the prefix and restart at `state_len` for each branch, so each branch sees the same positions it would have as a stand-alone prompt.

Both backends build this mask densely (`T × T`, where `T` is the packed length), so memory grows quadratically with packed length.

## Readout

The model runs once over the packed sequence. At the last token of each branch, the logits of the answer-letter tokens `A`, `B`, … (one per listed option) are passed through softmax. Probabilities are conditional on the listed options. The code uses one canonical option order and does not average across option orders.

## Model

The model is [Ternary-Bonsai-4B](https://huggingface.co/prism-ml/Ternary-Bonsai-4B-unpacked) (Qwen3 architecture, ternary weights), used **frozen and unmodified**: no adapter, no fine-tuning, the native LM head. Both repositories are pinned to a commit in `src/system_one_bonsai/models.py`.

The model config uses YaRN rope scaling (factor 4, original context 8,192). Transformers applies it at every position, so the MLX backend computes the same YaRN inverse frequencies and attention factor (in `inference_mlx.py`, matching Transformers bit for bit) instead of plain RoPE.

## Backends

| Backend | File | Weights | Notes |
| --- | --- | --- | --- |
| PyTorch | `inference.py` | `Ternary-Bonsai-4B-unpacked` (FP16, ~8 GB) | fp16 on CUDA and MPS, fp32 on CPU; SDPA with a boolean mask |
| MLX | `inference_mlx.py` | `Ternary-Bonsai-4B-mlx-2bit` (2-bit packed, ~1.1 GB) | packed weights are used as is; fp16 activations; custom forward over MLX-LM's Qwen3 modules to apply the tree mask and custom positions |

The unpacked FP16 weights and the dequantized 2-bit pack are not bit-identical (on every attention and MLP projection, a few hundred elements differ by at most 2.4e-4, i.e. FP16 rounding). Differences between backends come from this, precision, and kernels; see [RESULTS.md](RESULTS.md).
