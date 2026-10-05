"""PyTorch canonical-order inference on the frozen Ternary-Bonsai-4B FP16 weights."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .models import TORCH_REPO, TORCH_REVISION
from .packing import Branch, LETTERS, PackedRequest, pack_request, position_ids as build_position_ids


def load_model(device: str | None = None):
    """Download the pinned FP16 Bonsai weights as needed and load them frozen."""
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(TORCH_REPO, revision=TORCH_REVISION)
    model = AutoModelForCausalLM.from_pretrained(TORCH_REPO, revision=TORCH_REVISION, dtype=dtype, attn_implementation="sdpa")
    return model.to(device).eval(), tokenizer


@torch.inference_mode()
def predict_request(model, tokenizer, request: PackedRequest) -> list[list[float]]:
    """Return one probability vector per branch, in the original option order.

    A noul branch uses [No, Yes]. All branches share the chat prompt's common
    prefix; each branch sees only that prefix and its own suffix.
    """
    ids, state_len, spans = pack_request(request, tokenizer)

    branch_ids = torch.full((len(ids),), -1, dtype=torch.long)
    for index, (start, end) in enumerate(spans):
        branch_ids[start:end + 1] = index
    t = len(ids)
    mask = torch.tril(torch.ones((t, t), dtype=torch.bool)) & (
        (branch_ids == -1).unsqueeze(0) | (branch_ids[:, None] == branch_ids[None, :])
    )
    device = next(model.parameters()).device
    decisions = torch.tensor([end for _, end in spans], device=device)
    logits = model(
        input_ids=torch.tensor([ids], device=device),
        attention_mask=mask.to(device)[None, None],
        position_ids=torch.tensor([build_position_ids(state_len, spans)], device=device),
        logits_to_keep=decisions,
        use_cache=False,
    ).logits[0]
    counts = [len(branch.options or ["No", "Yes"]) for branch in request.branches]
    label_ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS[:max(counts)]]
    if any(len(token) != 1 for token in label_ids):
        raise ValueError("answer labels must each be one token")
    labels = torch.tensor([token[0] for token in label_ids], device=device)
    return [torch.softmax(row[labels[:count]].float(), dim=-1).tolist() for row, count in zip(logits, counts)]

