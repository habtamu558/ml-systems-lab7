"""Reads lab7_out/results.json and prints the extra tables needed for the report.
Run:  python summarize.py
No training, takes <1 second."""
import json

R = json.load(open("lab7_out/results.json"))
b = R["baseline"]
out = []


def add(s=""):
    out.append(s)


# 1. Unstructured pruning: accuracy + size + latency per sparsity
for mode in ["global", "layer"]:
    add(f"### Unstructured pruning, {mode}-wise")
    add("| Target sparsity | Real sparsity | Acc before FT (%) | Acc after FT (%) | Non-zero params "
        "| Dense KB | CSR KB | gzip KB | Latency b=1 (ms) | Effective MACs (M) |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for r in R["unstructured"][mode]:
        add(f"| {r['target']:.2f} | {r['weight_sparsity']:.3f} | {r['acc_before_ft']:.2f} | {r['acc']:.2f} "
            f"| {r['nonzero']:,} | {r['dense_kb']:.0f} | {r['sparse_kb']:.0f} | {r['gzip_kb']:.0f} "
            f"| {r['lat_b1_ms']:.2f} | {r['eff_macs'] / 1e6:.2f} |")
    add()
ctrl = R["unstructured"]["global"][0]
add(f"CONTROL (0% sparsity + same fine-tuning): {ctrl['acc']:.2f}%   vs   dense baseline without extra fine-tuning: {b['acc']:.2f}%")
add("-> use the control, not the baseline, as the fair reference for the pruned models.")
add()

# 2. Per-layer sparsity: why global differs from layer-wise
add("### Per-layer sparsity actually obtained (shows WHERE each method cuts)")
for mode in ["global", "layer"]:
    for r in R["unstructured"][mode]:
        if r["target"] in (0.9, 0.95):
            ls = ", ".join(f"{k.replace('.weight', '')}={v * 100:.0f}%" for k, v in r["layer_sparsity"].items())
            add(f"- {mode} @ {int(r['target'] * 100)}%: {ls}")
add()

# 3. Structured
add("### Structured (filter) pruning")
add("| Filters removed | Params | Acc before FT (%) | Acc after FT (%) | Dense KB | Latency b=1 (ms) | Latency b=64 (ms) | MACs (M) |")
add("|---|---|---|---|---|---|---|---|")
add(f"| 0% (baseline) | {b['params']:,} | - | {b['acc']:.2f} | {b['dense_kb']:.0f} | {b['lat_b1_ms']:.2f} | {b['lat_b64_ms']:.1f} | {b['macs'] / 1e6:.2f} |")
for r in R["structured"]:
    add(f"| {int(r['target'] * 100)}% | {r['params']:,} | {r['acc_before_ft']:.2f} | {r['acc']:.2f} | {r['dense_kb']:.0f} "
        f"| {r['lat_b1_ms']:.2f} | {r['lat_b64_ms']:.1f} | {r['macs'] / 1e6:.2f} |")
add()

# 4. fc1 dense vs CSR kernel
add("### Largest layer (fc1): dense vs CSR matmul, batch 1")
add("| Global sparsity | Dense (ms) | CSR (ms) | CSR / dense |")
add("|---|---|---|---|")
for r in R["unstructured"]["global"]:
    d, s = r["fc1_dense_ms"], r["fc1_csr_ms"]
    add(f"| {r['target']:.2f} | {d:.3f} | {s:.3f} | {s / d:.1f}x |" if s else f"| {r['target']:.2f} | {d:.3f} | n/a | n/a |")
add()

# 5. Distillation
D = R["distill"]
import statistics as st
pa, ka = D["plain_accs"], D["kd_accs"]
add("### Distillation (T=4, alpha=0.5)")
add(f"- plain student accuracies per seed: {[round(x, 2) for x in pa]}  mean {st.mean(pa):.2f}")
add(f"- distilled student per seed:        {[round(x, 2) for x in ka]}  mean {st.mean(ka):.2f}")
add(f"- per-seed difference (distilled - plain): {[round(k - p, 2) for k, p in zip(ka, pa)]}")
add(f"- teacher (baseline): {b['acc']:.2f}% | student params {R['student_params']:,}")
add(f"- latency b=1: plain {D['plain_profile']['lat_b1_ms']:.2f} ms, distilled {D['kd_profile']['lat_b1_ms']:.2f} ms "
    f"(p95: {D['plain_profile']['lat_b1_p95_ms']:.2f} / {D['kd_profile']['lat_b1_p95_ms']:.2f})")
add(f"- baseline latency p95: {b['lat_b1_p95_ms']:.2f} ms (median {b['lat_b1_ms']:.2f})")

text = "\n".join(out)
print(text)
open("lab7_out/extra_tables.md", "w").write(text)
print("\nSaved to lab7_out/extra_tables.md")