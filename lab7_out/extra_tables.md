### Unstructured pruning, global-wise
| Target sparsity | Real sparsity | Acc before FT (%) | Acc after FT (%) | Non-zero params | Dense KB | CSR KB | gzip KB | Latency b=1 (ms) | Effective MACs (M) |
|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.000 | 92.53 | 93.05 | 997,434 | 3900 | 7799 | 3633 | 4.71 | 30.03 |
| 0.25 | 0.250 | 92.51 | 93.10 | 748,170 | 3900 | 5852 | 2956 | 4.90 | 27.90 |
| 0.50 | 0.500 | 92.44 | 93.02 | 498,906 | 3900 | 3904 | 2100 | 5.53 | 24.74 |
| 0.75 | 0.750 | 92.34 | 93.20 | 249,642 | 3900 | 1957 | 1233 | 5.59 | 16.08 |
| 0.90 | 0.900 | 90.11 | 93.12 | 100,084 | 3900 | 788 | 651 | 5.52 | 8.49 |
| 0.95 | 0.950 | 77.21 | 92.46 | 50,231 | 3900 | 399 | 436 | 4.28 | 5.15 |

### Unstructured pruning, layer-wise
| Target sparsity | Real sparsity | Acc before FT (%) | Acc after FT (%) | Non-zero params | Dense KB | CSR KB | gzip KB | Latency b=1 (ms) | Effective MACs (M) |
|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.000 | 92.53 | 92.74 | 997,434 | 3900 | 7799 | 3633 | 4.19 | 30.03 |
| 0.25 | 0.250 | 92.29 | 92.95 | 748,170 | 3900 | 5852 | 2963 | 4.01 | 22.52 |
| 0.50 | 0.500 | 91.63 | 92.76 | 498,906 | 3900 | 3904 | 2114 | 4.15 | 15.02 |
| 0.75 | 0.750 | 81.57 | 93.16 | 249,642 | 3900 | 1957 | 1237 | 4.28 | 7.51 |
| 0.90 | 0.900 | 36.18 | 91.01 | 100,085 | 3900 | 788 | 659 | 4.16 | 3.00 |
| 0.95 | 0.950 | 10.00 | 86.67 | 50,233 | 3900 | 399 | 437 | 4.17 | 1.50 |

CONTROL (0% sparsity + same fine-tuning): 93.05%   vs   dense baseline without extra fine-tuning: 92.53%
-> use the control, not the baseline, as the fair reference for the pruned models.

### Per-layer sparsity actually obtained (shows WHERE each method cuts)
- global @ 90%: conv1=23%, conv2=68%, conv3=75%, fc1=92%, fc2=70%
- global @ 95%: conv1=31%, conv2=80%, conv3=85%, fc1=96%, fc2=80%
- layer @ 90%: conv1=90%, conv2=90%, conv3=90%, fc1=90%, fc2=90%
- layer @ 95%: conv1=95%, conv2=95%, conv3=95%, fc1=95%, fc2=95%

### Structured (filter) pruning
| Filters removed | Params | Acc before FT (%) | Acc after FT (%) | Dense KB | Latency b=1 (ms) | Latency b=64 (ms) | MACs (M) |
|---|---|---|---|---|---|---|---|
| 0% (baseline) | 997,434 | - | 92.53 | 3900 | 4.69 | 210.1 | 30.03 |
| 10% | 888,665 | 92.51 | 92.87 | 3475 | 3.75 | 204.9 | 24.65 |
| 25% | 731,194 | 91.71 | 93.27 | 2860 | 2.67 | 143.3 | 17.11 |
| 50% | 476,474 | 87.35 | 93.07 | 1865 | 1.59 | 77.3 | 7.79 |

### Largest layer (fc1): dense vs CSR matmul, batch 1
| Global sparsity | Dense (ms) | CSR (ms) | CSR / dense |
|---|---|---|---|
| 0.00 | 0.147 | 2.688 | 18.3x |
| 0.25 | 0.172 | 2.271 | 13.2x |
| 0.50 | 0.178 | 1.729 | 9.7x |
| 0.75 | 0.209 | 0.908 | 4.3x |
| 0.90 | 0.153 | 0.355 | 2.3x |
| 0.95 | 0.157 | 0.219 | 1.4x |

### Distillation (T=4, alpha=0.5)
- plain student accuracies per seed: [91.32, 91.71, 92.02]  mean 91.68
- distilled student per seed:        [91.46, 91.64, 91.47]  mean 91.52
- per-seed difference (distilled - plain): [0.14, -0.07, -0.55]
- teacher (baseline): 92.53% | student params 106,954
- latency b=1: plain 0.90 ms, distilled 0.87 ms (p95: 1.31 / 1.12)
- baseline latency p95: 7.01 ms (median 4.69)