"""Measure request latency of tree-masked packing versus one forward pass per question.

For each fixture record and each question count N, the request holds N branches taken
cyclically from that record's own branches. Two paths answer the same N questions:
  tree        - one tree-masked forward pass over the shared prefix and all N branches
  independent - N separate forward passes, each re-encoding the record with one branch
Model loading is excluded; every timed call returns host-side probabilities.

    python scripts/bench_latency.py --backend mlx --output outputs/latency-mlx.json
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

from system_one_bonsai import PackedRequest, pack_request
from system_one_bonsai.runner import load_backend, read_requests

ROOT = Path(__file__).resolve().parents[1]


def timed(fn) -> float:
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=["torch", "mlx"], required=True)
    parser.add_argument("--input", default=ROOT / "fixtures/behavior.jsonl", type=Path)
    parser.add_argument("--counts", default="1,2,4,8,16", help="comma-separated question counts")
    parser.add_argument("--repeats", type=int, default=3, help="timed repeats per record and path")
    parser.add_argument("--output", type=Path, help="write raw timings and the summary as JSON")
    args = parser.parse_args()
    counts = [int(n) for n in args.counts.split(",")]

    model, tokenizer, predict = load_backend(args.backend)
    records = read_requests(args.input)
    for _, request in records[:2]:  # warm up kernels and caches
        predict(model, tokenizer, request)

    rows = []
    for n in counts:
        for request_id, base in records:
            branches = [base.branches[i % len(base.branches)] for i in range(n)]
            request = PackedRequest(base.state, branches)
            singles = [PackedRequest(base.state, [branch]) for branch in branches]
            packed_tokens = len(pack_request(request, tokenizer)[0])
            independent_tokens = sum(len(pack_request(single, tokenizer)[0]) for single in singles)
            for _ in range(args.repeats):
                tree_ms = timed(lambda: predict(model, tokenizer, request))
                independent_ms = timed(lambda: [predict(model, tokenizer, single) for single in singles])
                rows.append({"id": request_id, "questions": n, "tree_ms": tree_ms, "independent_ms": independent_ms,
                             "tree_tokens": packed_tokens, "independent_tokens": independent_tokens})
        print(f"N={n} done", flush=True)

    summary = []
    for n in counts:
        sel = [row for row in rows if row["questions"] == n]
        tree = statistics.median(row["tree_ms"] for row in sel)
        independent = statistics.median(row["independent_ms"] for row in sel)
        summary.append({
            "questions": n,
            "tree_ms_median": round(tree, 1),
            "independent_ms_median": round(independent, 1),
            "speedup": round(independent / tree, 2),
            "tree_tokens_median": statistics.median(row["tree_tokens"] for row in sel),
            "independent_tokens_median": statistics.median(row["independent_tokens"] for row in sel),
        })
    meta = {"backend": args.backend, "machine": platform.machine(), "platform": platform.platform(),
            "processor": platform.processor(), "records": len(records), "repeats": args.repeats}
    print(json.dumps({"meta": meta, "summary": summary}, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"meta": meta, "summary": summary, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
