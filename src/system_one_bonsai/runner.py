"""Helpers shared by the example and verification scripts."""
from __future__ import annotations

import json
from pathlib import Path

from .packing import Branch, PackedRequest


def load_backend(backend: str):
    """Return (model, tokenizer, predict_request) for "torch" or "mlx"."""
    if backend == "torch":
        from .inference import load_model, predict_request
    elif backend == "mlx":
        from .inference_mlx import load_model, predict_request
    else:
        raise ValueError(f"unknown backend: {backend}")
    model, tokenizer = load_model()
    return model, tokenizer, predict_request


def read_requests(path: str | Path) -> list[tuple[str, PackedRequest]]:
    """Read JSONL rows of {id, state, branches: [{kind, instructions, options?}]}."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        branches = [Branch(b["kind"], b["instructions"], b.get("options", [])) for b in row["branches"]]
        rows.append((row["id"], PackedRequest(row["state"], branches)))
    return rows
