"""Apple Silicon inference on the frozen 2-bit packed Ternary-Bonsai-4B weights."""
from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
from mlx_lm import load as mlx_load
from mlx_lm.models.activations import swiglu
from transformers import AutoTokenizer

from .models import MLX_REPO, MLX_REVISION, TORCH_REPO, TORCH_REVISION
from .packing import Branch, LETTERS, PackedRequest, pack_request, position_ids


class MLXModel:
    def __init__(self):
        self.model, _ = mlx_load(MLX_REPO, revision=MLX_REVISION)
        self.tokenizer = AutoTokenizer.from_pretrained(TORCH_REPO, revision=TORCH_REVISION)
        self.model.set_dtype(mx.float16)  # float parts only; packed weights stay 2-bit
        mx.eval(self.model.parameters())
        args = self.model.args
        self._tied = args.tie_word_embeddings
        self._inv_freq, self._rope_factor = self._rope_frequencies()
        self._labels: dict[int, mx.array] = {}

    def _rope_frequencies(self) -> tuple[mx.array, float]:
        """Inverse frequencies and attention factor; YaRN follows transformers' `_compute_yarn_parameters`."""
        args = self.model.args
        dim, base = args.head_dim, args.rope_theta
        pos_freqs = base ** (np.arange(0, dim, 2, dtype=np.float32) / dim)
        scaling = args.rope_scaling
        if not scaling:
            return mx.array(1.0 / pos_freqs), 1.0
        if scaling.get("rope_type", scaling.get("type")) != "yarn":
            raise ValueError(f"unsupported rope_scaling: {scaling}")
        original = scaling.get("original_max_position_embeddings") or args.max_position_embeddings
        factor = args.max_position_embeddings / original if "original_max_position_embeddings" in scaling else scaling["factor"]
        attention_factor = scaling.get("attention_factor") or (0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0)
        if scaling.get("mscale") or scaling.get("mscale_all_dim"):
            raise ValueError("YaRN mscale is not supported")

        def correction_dim(rotations: float) -> float:
            return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low, high = correction_dim(scaling.get("beta_fast") or 32), correction_dim(scaling.get("beta_slow") or 1)
        if scaling.get("truncate", True):
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, dim - 1)
        if low == high:
            high += 0.001
        ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - low) / (high - low), 0, 1)
        extrapolation = 1 - ramp
        inv_freq = (1 / (factor * pos_freqs)) * (1 - extrapolation) + (1 / pos_freqs) * extrapolation
        return mx.array(inv_freq.astype(np.float32)), float(attention_factor)

    @staticmethod
    def _rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        half = x.shape[-1] // 2
        xf = x.astype(mx.float32)
        rotated = mx.concatenate([-xf[..., half:], xf[..., :half]], axis=-1)
        return (xf * cos + rotated * sin).astype(x.dtype)

    def _forward(self, ids: mx.array, positions: mx.array, mask: mx.array, decisions: mx.array) -> mx.array:
        inner = self.model.model
        length = ids.shape[0]
        freqs = positions.astype(mx.float32)[:, None] * self._inv_freq[None]
        angles = mx.concatenate([freqs, freqs], axis=-1)
        cos, sin = mx.cos(angles)[None, None], mx.sin(angles)[None, None]
        if self._rope_factor != 1.0:  # YaRN attention factor, applied to cos/sin as in transformers
            cos, sin = cos * self._rope_factor, sin * self._rope_factor
        mask = mask[None, None]
        h = inner.embed_tokens(ids[None])
        for layer in inner.layers:
            attn = layer.self_attn
            x = layer.input_layernorm(h)
            q = attn.q_norm(attn.q_proj(x).reshape(1, length, attn.n_heads, -1)).transpose(0, 2, 1, 3)
            k = attn.k_norm(attn.k_proj(x).reshape(1, length, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
            v = attn.v_proj(x).reshape(1, length, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
            q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=mask)
            h = h + attn.o_proj(out.transpose(0, 2, 1, 3).reshape(1, length, -1))
            x = layer.post_attention_layernorm(h)
            h = h + layer.mlp.down_proj(swiglu(layer.mlp.gate_proj(x), layer.mlp.up_proj(x)))
        h = inner.norm(h[0, decisions])
        return inner.embed_tokens.as_linear(h) if self._tied else self.model.lm_head(h)

    def _label_ids(self, count: int) -> mx.array:
        if count not in self._labels:
            ids = [self.tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS[:count]]
            if any(len(token) != 1 for token in ids):
                raise ValueError("answer labels must each be one token")
            self._labels[count] = mx.array([token[0] for token in ids])
        return self._labels[count]


def load_model() -> tuple[MLXModel, object]:
    model = MLXModel()
    return model, model.tokenizer


def predict_request(model: MLXModel, tokenizer, request: PackedRequest) -> list[list[float]]:
    """Return canonical-order distributions in one tree-masked forward pass."""
    ids, state_len, spans = pack_request(request, tokenizer)
    branches = [-1] * len(ids)
    for index, (start, end) in enumerate(spans):
        branches[start:end + 1] = [index] * (end - start + 1)
    branch_ids = mx.array(branches)
    t = len(ids)
    mask = mx.tril(mx.ones((t, t), dtype=mx.bool_)) & (
        (branch_ids == -1)[None, :] | (branch_ids[:, None] == branch_ids[None, :])
    )
    logits = model._forward(
        mx.array(ids), mx.array(position_ids(state_len, spans)), mask, mx.array([end for _, end in spans])
    )
    probs = [
        mx.softmax(logits[index, model._label_ids(len(branch.options or ["No", "Yes"]))].astype(mx.float32), axis=-1)
        for index, branch in enumerate(request.branches)
    ]
    mx.eval(probs)
    return [p.tolist() for p in probs]

