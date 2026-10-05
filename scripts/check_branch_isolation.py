"""Check that a branch's probabilities do not depend on the other branches packed with it.

For every request in the fixture, each branch is scored three ways:
  packed   - with all branches in fixture order
  reversed - with all branches in reverse order
  alone    - as the only branch
With a correct tree mask, the three distributions differ only by floating-point noise.

    python scripts/check_branch_isolation.py --backend mlx --output outputs/isolation-mlx.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from system_one_bonsai import PackedRequest
from system_one_bonsai.runner import load_backend, read_requests

ROOT = Path(__file__).resolve().parents[1]


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=["torch", "mlx"], required=True)
    parser.add_argument("--input", default=ROOT / "fixtures/behavior.jsonl", type=Path)
    parser.add_argument("--output", type=Path, help="write per-branch rows and the summary as JSON")
    parser.add_argument("--tolerance", type=float, default=1e-2, help="max allowed absolute probability difference")
    args = parser.parse_args()

    model, tokenizer, predict = load_backend(args.backend)
    rows = []
    for request_id, request in read_requests(args.input):
        packed = predict(model, tokenizer, request)
        reversed_ = predict(model, tokenizer, PackedRequest(request.state, request.branches[::-1]))[::-1]
        for index, branch in enumerate(request.branches):
            alone = predict(model, tokenizer, PackedRequest(request.state, [branch]))[0]
            diff = max(abs(a - b) for other in (reversed_[index], alone) for a, b in zip(packed[index], other))
            same_argmax = argmax(packed[index]) == argmax(reversed_[index]) == argmax(alone)
            rows.append({"id": request_id, "branch": index, "kind": branch.kind, "packed": packed[index],
                         "reversed": reversed_[index], "alone": alone, "max_abs_diff": diff, "same_argmax": same_argmax})

    summary = {
        "backend": args.backend,
        "requests": len({row["id"] for row in rows}),
        "branches": len(rows),
        "max_abs_diff": max(row["max_abs_diff"] for row in rows),
        "argmax_agreement": sum(row["same_argmax"] for row in rows) / len(rows),
        "tolerance": args.tolerance,
    }
    summary["passed"] = summary["max_abs_diff"] <= args.tolerance and summary["argmax_agreement"] == 1.0
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
