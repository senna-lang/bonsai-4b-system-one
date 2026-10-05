# Results

All results on this page are reproducible from this repository. Measured on an Apple M2 (24 GB), 2026-10-05.

## Behavior checks

[`fixtures/behavior.jsonl`](../fixtures/behavior.jsonl) holds 16 fictional records with 42 branches (16 `choice`, 10 `score`, 16 `noul`). They test behavior, not accuracy: there are no labels.

### Branch isolation

```bash
python scripts/check_branch_isolation.py --backend mlx   --output outputs/isolation-mlx.json
python scripts/check_branch_isolation.py --backend torch --output outputs/isolation-torch.json
```

Each branch is scored packed with the others, packed in reverse order, and alone.

| Backend | Branches | Max abs diff | Argmax agreement |
| --- | ---: | ---: | ---: |
| MLX (2-bit packed, fp16 activations) | 42 | 0.0034 | 42/42 |
| PyTorch (MPS, fp16) | 42 | 0.0039 | 42/42 |

The default tolerance is 0.01. The remaining difference is fp16 noise from different sequence layouts. Raw rows: [`results/isolation/`](../results/isolation/).

### PyTorch vs MLX

```bash
python scripts/compare_backends.py predict --backend torch --output outputs/torch.jsonl   # torch env
python scripts/compare_backends.py predict --backend mlx   --output outputs/mlx.jsonl     # mlx env
python scripts/compare_backends.py compare outputs/torch.jsonl outputs/mlx.jsonl
```

| Branches | Max abs diff | Mean per-branch max diff | Argmax agreement |
| ---: | ---: | ---: | ---: |
| 42 | 0.0035 | 0.0004 | 42/42 |

PyTorch uses the unpacked FP16 weights and MLX the 2-bit pack; the two are not bit-identical (see [METHOD.md](METHOD.md#backends)). CUDA has not been measured. Raw predictions: [`results/backends/`](../results/backends/).

## Public accuracy

```bash
pip install -e '.[mlx,bench]'
python scripts/bench_accuracy.py --backend mlx --output outputs/accuracy
```

Protocol: [BENCHMARKS.md](BENCHMARKS.md). MLX backend.

| Task | Accuracy | 95% CI |
| --- | ---: | :---: |
| AG News | 0.868 | 0.838–0.896 |
| MASSIVE scenario (en-US) | 0.562 | 0.518–0.604 |
| MNLI (matched) | 0.824 | 0.790–0.856 |
| BoolQ | 0.836 | 0.802–0.868 |
| SST-5 | 0.402 | 0.360–0.444 |

SST-5 mean absolute error: 0.766 levels. Each task took 6–15 minutes on the M2. Row-level probabilities: [`results/accuracy/`](../results/accuracy/).

## Latency: tree-masked packing vs one forward per question

```bash
python scripts/bench_latency.py --backend mlx   --output outputs/latency-mlx.json
python scripts/bench_latency.py --backend torch --output outputs/latency-mps.json
```

Each fixture record is asked N questions (its own branches, repeated cyclically), answered either in one tree-masked forward pass or in N separate forward passes. Model loading excluded; medians over 16 records × 3 repeats.

| Questions | Packed tokens | MLX tree | MLX per-question | Speedup | MPS tree | MPS per-question | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 86 vs 86 | 0.53 s | 0.53 s | **1.00×** | 0.89 s | 0.86 s | **0.97×** |
| 2 | 114.5 vs 164.5 | 0.69 s | 1.05 s | **1.51×** | 0.82 s | 1.32 s | **1.60×** |
| 4 | 183 vs 332 | 0.98 s | 2.00 s | **2.03×** | 0.90 s | 1.90 s | **2.12×** |
| 8 | 313 vs 658.5 | 1.77 s | 4.18 s | **2.37×** | 1.53 s | 3.77 s | **2.46×** |
| 16 | 581 vs 1324 | 3.43 s | 8.66 s | **2.52×** | 3.10 s | 8.14 s | **2.63×** |

The gain comes from encoding the record once. Records in this fixture are short; longer records should benefit more, which has not been measured. Absolute times are for a laptop GPU and are not a deployment latency claim. Raw timings: [`results/latency/`](../results/latency/).
