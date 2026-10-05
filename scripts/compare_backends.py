"""Compare PyTorch and MLX predictions on the fixture.

The two backends pin different transformers versions, so predictions are written from
separate environments and compared afterwards:

    # in the torch environment
    python scripts/compare_backends.py predict --backend torch --output outputs/torch.jsonl
    # in the mlx environment
    python scripts/compare_backends.py predict --backend mlx --output outputs/mlx.jsonl
    # in either environment
    python scripts/compare_backends.py compare outputs/torch.jsonl outputs/mlx.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def predict(args: argparse.Namespace) -> None:
    from system_one_bonsai.runner import load_backend, read_requests

    model, tokenizer, predict_request = load_backend(args.backend)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for request_id, request in read_requests(args.input):
            for index, probabilities in enumerate(predict_request(model, tokenizer, request)):
                out.write(json.dumps({"id": request_id, "branch": index, "kind": request.branches[index].kind,
                                      "backend": args.backend, "probabilities": probabilities}) + "\n")


def compare(args: argparse.Namespace) -> None:
    def load(path: Path) -> dict:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return {(row["id"], row["branch"]): row for row in rows}

    a, b = load(args.a), load(args.b)
    if a.keys() != b.keys():
        raise SystemExit("prediction files cover different branches")
    diffs, agree = [], 0
    for key in sorted(a):
        p, q = a[key]["probabilities"], b[key]["probabilities"]
        diffs.append(max(abs(x - y) for x, y in zip(p, q)))
        agree += max(range(len(p)), key=p.__getitem__) == max(range(len(q)), key=q.__getitem__)
    summary = {"branches": len(diffs), "max_abs_diff": max(diffs), "mean_max_abs_diff": sum(diffs) / len(diffs),
               "argmax_agreement": agree / len(diffs), "tolerance": args.tolerance}
    summary["passed"] = summary["max_abs_diff"] <= args.tolerance and agree == len(diffs)
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["passed"] else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict", help="score the fixture with one backend")
    p.add_argument("--backend", choices=["torch", "mlx"], required=True)
    p.add_argument("--input", default=ROOT / "fixtures/behavior.jsonl", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.set_defaults(run=predict)
    c = sub.add_parser("compare", help="compare two prediction files")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    c.add_argument("--tolerance", type=float, default=2e-2, help="max allowed absolute probability difference")
    c.set_defaults(run=compare)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
