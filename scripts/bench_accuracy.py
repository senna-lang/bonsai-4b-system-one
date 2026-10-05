"""Zero-shot accuracy of frozen Ternary-Bonsai-4B on public datasets.

The protocol (datasets, splits, sample size, prompts, options, metrics) is fixed in docs/BENCHMARKS.md.
One canonical option order, one tree-masked forward pass per request.

    python scripts/bench_accuracy.py --backend mlx --output outputs/accuracy
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from system_one_bonsai import Branch, PackedRequest

SAMPLE_SIZE = 500
SEED = 0
BOOTSTRAP = 2000
SST5_LEVELS = ["very negative", "negative", "neutral", "positive", "very positive"]


@dataclass
class Task:
    name: str
    hf_args: tuple
    split: str
    build: Callable[[dict], tuple[PackedRequest, int]]  # row -> (request, gold option index)
    kind: str


def _ag_news(row):
    options = ["World", "Sports", "Business", "Sci/Tech"]
    return PackedRequest(row["text"], [Branch("choice", "What is the topic of this news article?", options)]), row["label"]


MASSIVE_SCENARIOS = ["alarm", "audio", "calendar", "cooking", "datetime", "email", "general", "iot", "lists",
                     "music", "news", "play", "qa", "recommendation", "social", "takeaway", "transport", "weather"]


def _massive(row):
    question = "Which scenario does this request to a voice assistant belong to?"
    request = PackedRequest(row["text"], [Branch("choice", question, MASSIVE_SCENARIOS)])
    return request, MASSIVE_SCENARIOS.index(row["label_text"])


def _mnli(row):
    state = f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}"
    question = "What is the relationship between the premise and the hypothesis?"
    return PackedRequest(state, [Branch("choice", question, ["entailment", "neutral", "contradiction"])]), row["label"]


def _boolq(row):
    question = row["question"].strip()
    question = question[0].upper() + question[1:] + ("" if question.endswith("?") else "?")
    return PackedRequest(row["passage"], [Branch("noul", question)]), int(row["answer"])  # options are [No, Yes]


def _sst5(row):
    question = "How positive is the sentiment of this text?"
    return PackedRequest(row["text"], [Branch("score", question, SST5_LEVELS)]), row["label"]


TASKS = [
    Task("AG News", ("fancyzhx/ag_news",), "test", _ag_news, "choice"),
    Task("MASSIVE scenario (en-US)", ("mteb/amazon_massive_scenario", "en"), "test", _massive, "choice"),
    Task("MNLI (matched)", ("nyu-mll/glue", "mnli"), "validation_matched", _mnli, "choice"),
    Task("BoolQ", ("google/boolq",), "validation", _boolq, "noul"),
    Task("SST-5", ("SetFit/sst5",), "test", _sst5, "score"),
]


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def bootstrap_ci(values: list[float], rng: random.Random) -> list[float]:
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOTSTRAP))
    return [round(means[int(0.025 * BOOTSTRAP)], 4), round(means[int(0.975 * BOOTSTRAP) - 1], 4)]


def summarize(task: Task, rows: list[dict]) -> dict:
    rng = random.Random(SEED)
    correct = [float(argmax(row["probabilities"]) == row["gold"]) for row in rows]
    out = {"task": task.name, "kind": task.kind, "n": len(rows), "options": len(rows[0]["probabilities"]),
           "accuracy": round(sum(correct) / len(rows), 4), "accuracy_ci95": bootstrap_ci(correct, rng)}
    if task.kind == "score":
        out["mae"] = round(sum(abs(argmax(row["probabilities"]) - row["gold"]) for row in rows) / len(rows), 4)
    return out


def main() -> None:
    import platform

    from datasets import load_dataset

    from system_one_bonsai.runner import load_backend

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=["torch", "mlx"], required=True)
    parser.add_argument("--output", required=True, type=Path, help="output directory")
    parser.add_argument("--limit", type=int, default=SAMPLE_SIZE, help="rows per task (smoke tests only)")
    args = parser.parse_args()

    model, tokenizer, predict_request = load_backend(args.backend)
    args.output.mkdir(parents=True, exist_ok=True)
    meta = {"backend": args.backend, "sample_size": args.limit, "seed": SEED, "machine": platform.machine(),
            "platform": platform.platform()}
    if args.backend == "torch":
        import torch

        meta["torch"] = torch.__version__
        meta["device"] = torch.cuda.get_device_name() if torch.cuda.is_available() else str(model.device)
    summaries = []
    for task in TASKS:
        dataset = load_dataset(*task.hf_args, split=task.split).shuffle(seed=SEED).select(range(args.limit))
        rows, start = [], time.perf_counter()
        for index, row in enumerate(dataset):
            request, gold = task.build(row)
            rows.append({"index": index, "gold": gold, "probabilities": predict_request(model, tokenizer, request)[0]})
        summary = summarize(task, rows)
        summary["seconds"] = round(time.perf_counter() - start, 1)
        summaries.append(summary)
        slug = task.name.split(" ")[0].lower().replace("-", "")
        (args.output / f"{slug}.predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(json.dumps(summary), flush=True)
    (args.output / "summary.json").write_text(json.dumps({"meta": meta, "results": summaries}, indent=2) + "\n")


if __name__ == "__main__":
    main()
