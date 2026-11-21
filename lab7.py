"""
Lab 7: Compression - pruning and distillation (Fashion-MNIST)
=============================================================
Run:   python lab7.py            (full run)
       python lab7.py --quick    (fast sanity run: fewer epochs / less data)

Outputs go to ./lab7_out : plots (.png), results.json, results_table.md

Sections (match the lab sheet):
  1. Baseline CNN (~1M params)
  2. Unstructured magnitude pruning (global + layer-wise) and structured filter pruning
  3. Latency + on-disk size (dense vs sparse) + sparse-kernel micro-benchmark
  4. Knowledge distillation (student ~100k params, with vs without teacher)
  5. Summary table by Ch. 9 pillars (data / model / compute / energy)
"""
import argparse, copy, io, json, os, time, zlib
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import sparse

OUT = "lab7_out"
SPARSITIES = [0, 0.25, 0.5, 0.75, 0.9, 0.95]
STRUCT_RATIOS = [0.10, 0.25, 0.50]
T, ALPHA = 4.0, 0.5

CFG_FULL = dict(base_ep=10, ft_ep=3, stu_ep=15, seeds=[0, 1, 2], lat_runs=200,
                n_train=60000, n_test=10000)
CFG_QUICK = dict(base_ep=3, ft_ep=1, stu_ep=4, seeds=[0], lat_runs=50,
                 n_train=15000, n_test=10000)


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------
def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)


def load_data(device, cfg, fake=False):
    """Load Fashion-MNIST fully into memory as normalised tensors (fast, simple)."""
    if fake:  # random data, only for smoke-testing the code
        g = torch.Generator().manual_seed(0)
        mk = lambda n: (torch.randn(n, 1, 28, 28, generator=g).to(device),
                        torch.randint(0, 10, (n,), generator=g).to(device))
        return mk(cfg["n_train"]), mk(cfg["n_test"])
    from torchvision import datasets
    tr = datasets.FashionMNIST("data", train=True, download=True)
    te = datasets.FashionMNIST("data", train=False, download=True)

    def prep(ds, n):
        x = ds.data[:n].float().div(255).sub(0.2860).div(0.3530).unsqueeze(1)
        return x.to(device), ds.targets[:n].to(device)
    return prep(tr, cfg["n_train"]), prep(te, cfg["n_test"])


class CNN(nn.Module):
    """3 conv layers + 2 FC layers. Channel widths are configurable so the same
    class is used for the baseline, the structurally-pruned nets and the student.
      baseline: CNN(32, 64, 128, 144) -> ~997k params
      student : CNN(8, 16, 32, 64)    -> ~107k params
    """
    def __init__(self, c1=32, c2=64, c3=128, hidden=144, drop=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1)
        self.fc1 = nn.Linear(c3 * 7 * 7, hidden)
        self.fc2 = nn.Linear(hidden, 10)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = F.relu(self.conv1(x))                      # 28x28
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)     # 14x14
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)     # 7x7
        x = x.flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


def n_params(m): return sum(p.numel() for p in m.parameters())
def n_nonzero(m): return sum((p != 0).sum().item() for p in m.parameters())
def prunable(m): return [(n, p) for n, p in m.named_parameters() if p.dim() > 1]  # weights only


def sparsity_of(m):
    P = prunable(m)
    return sum((p == 0).sum().item() for _, p in P) / sum(p.numel() for _, p in P)


def layer_sparsity(m):
    return {n: round((p == 0).float().mean().item(), 4) for n, p in prunable(m)}


@torch.no_grad()
def evaluate(model, data, bs=2000):
    model.eval(); X, y = data; c = 0
    for i in range(0, len(X), bs):
        c += (model(X[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    return 100.0 * c / len(X)


# ----------------------------------------------------------------------------
# Training (plain, masked fine-tuning, and distillation in one loop)
# ----------------------------------------------------------------------------
def kd_loss(s_logits, t_logits, y, T=T, alpha=ALPHA):
    """alpha*CE + (1-alpha)*T^2*KL.  F.kl_div(log q_student, p_teacher) computes
    KL(teacher || student), which is the standard Hinton distillation term."""
    ce = F.cross_entropy(s_logits, y)
    kl = F.kl_div(F.log_softmax(s_logits / T, 1), F.softmax(t_logits / T, 1),
                  reduction="batchmean")
    return alpha * ce + (1 - alpha) * T * T * kl


@torch.no_grad()
def apply_masks(model, masks):
    for n, p in model.named_parameters():
        if n in masks:
            p.mul_(masks[n])          # pruned weights are forced back to exactly 0


def train(model, train_data, test_data, epochs, lr, masks=None, teacher_logits=None,
          bs=128, seed=0, tag=""):
    X, y = train_data
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X), generator=g).to(X.device)
        tot = 0.0
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            out = model(X[idx])
            loss = (F.cross_entropy(out, y[idx]) if teacher_logits is None
                    else kd_loss(out, teacher_logits[idx], y[idx]))
            opt.zero_grad(); loss.backward(); opt.step()
            if masks is not None:
                apply_masks(model, masks)   # keep the mask FIXED during fine-tuning
            tot += loss.item() * len(idx)
        print(f"   [{tag}] epoch {ep + 1}/{epochs}  loss {tot / len(X):.4f}  "
              f"test acc {evaluate(model, test_data):.2f}%")
    return model


# ----------------------------------------------------------------------------
# Pruning
# ----------------------------------------------------------------------------
@torch.no_grad()
def make_masks(model, s, mode):
    """Magnitude pruning masks. mode='global': one threshold over ALL weights.
    mode='layer': every layer pruned to the same sparsity s."""
    P = prunable(model)
    if s == 0:
        return {n: torch.ones_like(p) for n, p in P}
    if mode == "global":
        allw = torch.cat([p.abs().flatten() for _, p in P])
        thr = allw.kthvalue(int(s * allw.numel())).values
        return {n: (p.abs() > thr).float() for n, p in P}
    masks = {}
    for n, p in P:
        thr = p.abs().flatten().kthvalue(int(s * p.numel())).values
        masks[n] = (p.abs() > thr).float()
    return masks


@torch.no_grad()
def structured_prune(model, ratio):
    """Remove whole conv filters with the smallest L2 norm (per layer) and build a
    PHYSICALLY SMALLER network (so latency really changes). Removing filter k of
    conv_i also removes input channel k of the next layer."""
    dev = next(model.parameters()).device

    def keep(conv):
        k = max(1, round(conv.out_channels * (1 - ratio)))
        return conv.weight.flatten(1).norm(dim=1).topk(k).indices.sort().values

    k1, k2, k3 = keep(model.conv1), keep(model.conv2), keep(model.conv3)
    h = model.fc1.out_features
    new = CNN(len(k1), len(k2), len(k3), h, model.drop.p).to(dev)
    new.conv1.weight.copy_(model.conv1.weight[k1]);          new.conv1.bias.copy_(model.conv1.bias[k1])
    new.conv2.weight.copy_(model.conv2.weight[k2][:, k1]);   new.conv2.bias.copy_(model.conv2.bias[k2])
    new.conv3.weight.copy_(model.conv3.weight[k3][:, k2]);   new.conv3.bias.copy_(model.conv3.bias[k3])
    # fc1 input is flatten(C,7,7): channel c owns columns c*49 ... c*49+48
    W = model.fc1.weight.view(h, model.conv3.out_channels, 49)[:, k3, :].reshape(h, -1)
    new.fc1.weight.copy_(W);                                 new.fc1.bias.copy_(model.fc1.bias)
    new.fc2.weight.copy_(model.fc2.weight);                  new.fc2.bias.copy_(model.fc2.bias)
    return new


# ----------------------------------------------------------------------------
# Measurements: latency, size, MACs, energy proxy
# ----------------------------------------------------------------------------
def time_fn(fn, runs, warmup=30):
    for _ in range(warmup): fn()
    ts = []
    for _ in range(runs):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts)), float(np.percentile(ts, 95))


def latency(model, batch=1, runs=200):
    """Median CPU latency in ms. Single thread -> less noise, fair comparison."""
    m = copy.deepcopy(model).cpu().eval()
    x = torch.randn(batch, 1, 28, 28)
    old = torch.get_num_threads(); torch.set_num_threads(1)
    with torch.inference_mode():
        med, p95 = time_fn(lambda: m(x), runs)
    torch.set_num_threads(old)
    return med, p95


def fc1_dense_vs_csr(model, batch=1, runs=300):
    """Micro-benchmark of the biggest layer (fc1): dense matmul vs CSR sparse matmul."""
    W = model.fc1.weight.detach().cpu()
    x = torch.randn(batch, W.shape[1]); xt = x.t().contiguous()
    old = torch.get_num_threads(); torch.set_num_threads(1)
    with torch.inference_mode():
        d, _ = time_fn(lambda: F.linear(x, W), runs)
        try:
            Wc = W.to_sparse_csr()
            s, _ = time_fn(lambda: torch.sparse.mm(Wc, xt), runs)
        except Exception as e:
            print("   (sparse CSR matmul unavailable:", e, ")"); s = None
    torch.set_num_threads(old)
    return d, s


def dense_bytes(model):
    buf = io.BytesIO()
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, buf)
    return buf.getbuffer().nbytes


def gzip_bytes(model):
    buf = io.BytesIO()
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, buf)
    return len(zlib.compress(buf.getvalue(), 9))


def sparse_bytes(model):
    """Store every weight tensor in CSR (values float32 + int32 column idx + int32 row ptr)."""
    arrs = {}
    for k, v in model.state_dict().items():
        v = v.cpu().numpy()
        if v.ndim > 1:
            c = sparse.csr_matrix(v.reshape(v.shape[0], -1))
            arrs[k + ".data"] = c.data.astype(np.float32)
            arrs[k + ".idx"] = c.indices.astype(np.int32)
            arrs[k + ".ptr"] = c.indptr.astype(np.int32)
            arrs[k + ".shape"] = np.array(v.shape, dtype=np.int32)
        else:
            arrs[k] = v
    buf = io.BytesIO(); np.savez(buf, **arrs)
    return buf.getbuffer().nbytes


@torch.no_grad()
def count_macs(model, effective=False):
    """Multiply-accumulates for one image. effective=True counts only non-zero weights
    (the THEORETICAL work if hardware could skip zeros)."""
    total = [0]; hooks = []

    def nz(w): return (w != 0).sum().item() if effective else w.numel()
    hooks.append(None)
    hs = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hs.append(m.register_forward_hook(lambda mod, i, o: total.__setitem__(0, total[0] + nz(mod.weight) * o.shape[2] * o.shape[3])))
        elif isinstance(m, nn.Linear):
            hs.append(m.register_forward_hook(lambda mod, i, o: total.__setitem__(0, total[0] + nz(mod.weight))))
    dev = next(model.parameters()).device
    model.eval(); model(torch.zeros(1, 1, 28, 28, device=dev))
    for h in hs: h.remove()
    return total[0]


def energy_uJ(eff_macs, nonzero_params):
    """ENERGY PROXY (not a measurement). Horowitz (ISSCC 2014, 45nm): fp32 MAC ~4.6 pJ,
    32-bit DRAM read ~640 pJ. Assume every non-zero weight is fetched from DRAM once."""
    return (eff_macs * 4.6e-12 + nonzero_params * 640e-12) * 1e6


def profile(model, test, lat_runs):
    model.eval()
    b1, b1p = latency(model, 1, lat_runs)
    b64, _ = latency(model, 64, max(20, lat_runs // 4))
    emacs = count_macs(model, effective=True)
    nz = n_nonzero(model)
    return dict(acc=evaluate(model, test), params=n_params(model), nonzero=nz,
                weight_sparsity=sparsity_of(model),
                dense_kb=dense_bytes(model) / 1024, sparse_kb=sparse_bytes(model) / 1024,
                gzip_kb=gzip_bytes(model) / 1024,
                lat_b1_ms=b1, lat_b1_p95_ms=b1p, lat_b64_ms=b64,
                macs=count_macs(model), eff_macs=emacs, energy_uJ=energy_uJ(emacs, nz))


# ----------------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fake", action="store_true", help="random data (smoke test only)")
    args = ap.parse_args()
    cfg = dict(CFG_QUICK if args.quick else CFG_FULL)
    if args.fake:
        cfg.update(n_train=2000, n_test=500, base_ep=1, ft_ep=1, stu_ep=1, lat_runs=20)
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training device:", device, "| latency is always measured on CPU")
    set_seed(0)
    train_d, test_d = load_data(device, cfg, args.fake)
    R = {"cfg": cfg}

    # ---------------- 1. Baseline ----------------
    print("\n=== 1. BASELINE ===")
    base = CNN(32, 64, 128, 144).to(device)
    print("Baseline parameters:", n_params(base))
    train(base, train_d, test_d, cfg["base_ep"], 1e-3, tag="baseline")
    R["baseline"] = profile(base, test_d, cfg["lat_runs"])
    R["baseline"].update(train_samples=cfg["n_train"], epochs=cfg["base_ep"])
    print(json.dumps(R["baseline"], indent=1))

    # ---------------- 2. Unstructured pruning ----------------
    print("\n=== 2a. UNSTRUCTURED MAGNITUDE PRUNING ===")
    R["unstructured"] = {"global": [], "layer": []}
    for mode in ["global", "layer"]:
        for s in SPARSITIES:
            m = copy.deepcopy(base)
            masks = make_masks(m, s, mode); apply_masks(m, masks)
            acc_before = evaluate(m, test_d)
            print(f" {mode} sparsity {s}: acc right after pruning = {acc_before:.2f}%")
            train(m, train_d, test_d, cfg["ft_ep"], 5e-4, masks=masks, tag=f"{mode}-{s}")
            rec = profile(m, test_d, cfg["lat_runs"])
            rec.update(target=s, acc_before_ft=acc_before, layer_sparsity=layer_sparsity(m))
            if mode == "global":
                d, sp = fc1_dense_vs_csr(m)
                rec.update(fc1_dense_ms=d, fc1_csr_ms=sp)
            R["unstructured"][mode].append(rec)
            print(f"   -> after FT acc {rec['acc']:.2f}% | real sparsity {rec['weight_sparsity']:.3f} "
                  f"| latency {rec['lat_b1_ms']:.3f} ms")

    print("\n=== 2b. STRUCTURED FILTER PRUNING ===")
    R["structured"] = []
    for r in STRUCT_RATIOS:
        m = structured_prune(base, r)
        acc_before = evaluate(m, test_d)
        print(f" structured {int(r * 100)}%: params {n_params(m)}, acc right after pruning = {acc_before:.2f}%")
        train(m, train_d, test_d, cfg["ft_ep"], 5e-4, tag=f"struct-{r}")
        rec = profile(m, test_d, cfg["lat_runs"])
        rec.update(target=r, acc_before_ft=acc_before)
        R["structured"].append(rec)
        print(f"   -> after FT acc {rec['acc']:.2f}% | latency {rec['lat_b1_ms']:.3f} ms")

    # ---------------- 4. Distillation ----------------
    print("\n=== 4. DISTILLATION (T=4, alpha=0.5) ===")
    base.eval()
    with torch.no_grad():
        t_logits = torch.cat([base(train_d[0][i:i + 2000]) for i in range(0, len(train_d[0]), 2000)])
    plain_accs, kd_accs, plain_m, kd_m = [], [], None, None
    for sd in cfg["seeds"]:
        set_seed(sd)
        sp = CNN(8, 16, 32, 64).to(device)
        train(sp, train_d, test_d, cfg["stu_ep"], 1e-3, seed=sd, tag=f"plain s{sd}")
        plain_accs.append(evaluate(sp, test_d))
        set_seed(sd)
        sk = CNN(8, 16, 32, 64).to(device)   # identical init (same seed) -> fair comparison
        train(sk, train_d, test_d, cfg["stu_ep"], 1e-3, teacher_logits=t_logits, seed=sd, tag=f"KD s{sd}")
        kd_accs.append(evaluate(sk, test_d))
        if plain_m is None: plain_m, kd_m = sp, sk
    R["student_params"] = n_params(plain_m)
    R["distill"] = dict(plain_accs=plain_accs, kd_accs=kd_accs,
                        plain_profile=profile(plain_m, test_d, cfg["lat_runs"]),
                        kd_profile=profile(kd_m, test_d, cfg["lat_runs"]))
    print(f" student params: {R['student_params']}")
    print(f" plain student accs {plain_accs}  mean {np.mean(plain_accs):.2f}")
    print(f" distilled student  {kd_accs}  mean {np.mean(kd_accs):.2f}")

    # ---------------- plots ----------------
    make_plots(R)
    # ---------------- 5. summary table ----------------
    table = make_table(R, cfg)
    open(f"{OUT}/results_table.md", "w").write(table)
    json.dump(R, open(f"{OUT}/results.json", "w"), indent=1)
    print("\n" + table)
    print(f"\nDone. See ./{OUT}/ for plots, results.json and results_table.md")


def make_plots(R):
    base = R["baseline"]; U = R["unstructured"]; S = R["structured"]
    # Fig 1: accuracy vs sparsity (unstructured, both flavours)
    plt.figure(figsize=(6.5, 4.2))
    for mode, c in [("global", "C0"), ("layer", "C1")]:
        xs = [r["target"] * 100 for r in U[mode]]
        plt.plot(xs, [r["acc"] for r in U[mode]], "o-", color=c, label=f"{mode}-wise, after fine-tune")
        plt.plot(xs, [r["acc_before_ft"] for r in U[mode]], "x--", color=c, alpha=.5, label=f"{mode}-wise, before fine-tune")
    plt.axhline(base["acc"], color="k", ls=":", label="dense baseline")
    plt.xlabel("Sparsity (% weights set to zero)"); plt.ylabel("Test accuracy (%)")
    plt.title("Unstructured magnitude pruning"); plt.legend(fontsize=7); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig1_unstructured_acc_vs_sparsity.png", dpi=150); plt.close()

    # Fig 2: structured
    plt.figure(figsize=(6.5, 4.2))
    xs = [0] + [r["target"] * 100 for r in S]
    plt.plot(xs, [base["acc"]] + [r["acc"] for r in S], "o-", label="after fine-tune")
    plt.plot(xs, [base["acc"]] + [r["acc_before_ft"] for r in S], "x--", alpha=.6, label="before fine-tune")
    plt.xlabel("Filters removed per conv layer (%)"); plt.ylabel("Test accuracy (%)")
    plt.title("Structured (filter) pruning"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig2_structured_acc_vs_pruning.png", dpi=150); plt.close()

    # Fig 3: both on one axis: accuracy vs non-zero parameters
    plt.figure(figsize=(6.5, 4.2))
    plt.plot([r["nonzero"] for r in U["global"]], [r["acc"] for r in U["global"]], "o-", label="unstructured global")
    plt.plot([r["nonzero"] for r in U["layer"]], [r["acc"] for r in U["layer"]], "s-", label="unstructured layer-wise")
    plt.plot([base["nonzero"]] + [r["nonzero"] for r in S], [base["acc"]] + [r["acc"] for r in S], "^-", label="structured")
    d = R["distill"]
    plt.scatter([d["plain_profile"]["nonzero"]], [np.mean(d["plain_accs"])], marker="*", s=150, label="student (plain)")
    plt.scatter([d["kd_profile"]["nonzero"]], [np.mean(d["kd_accs"])], marker="*", s=150, label="student (distilled)")
    plt.xscale("log"); plt.xlabel("Non-zero parameters (log)"); plt.ylabel("Test accuracy (%)")
    plt.title("Accuracy vs. model size"); plt.legend(fontsize=7); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig3_accuracy_vs_nonzero_params.png", dpi=150); plt.close()

    # Fig 4: honest latency analysis
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    frac_u = [r["nonzero"] / base["params"] for r in U["global"]]
    ax[0].plot(frac_u, [r["lat_b1_ms"] for r in U["global"]], "o-", label="unstructured (masked, dense kernel)")
    ax[0].plot([1] + [r["nonzero"] / base["params"] for r in S], [base["lat_b1_ms"]] + [r["lat_b1_ms"] for r in S],
               "^-", label="structured (smaller dense net)")
    ax[0].set_xlabel("Fraction of parameters remaining"); ax[0].set_ylabel("CPU latency, batch 1 (ms)")
    ax[0].invert_xaxis(); ax[0].set_title("Whole-network latency"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    xs = [r["target"] * 100 for r in U["global"]]
    ax[1].plot(xs, [r["fc1_dense_ms"] for r in U["global"]], "o-", label="fc1 dense matmul")
    if all(r["fc1_csr_ms"] is not None for r in U["global"]):
        ax[1].plot(xs, [r["fc1_csr_ms"] for r in U["global"]], "s-", label="fc1 CSR sparse matmul")
    ax[1].set_xlabel("Global sparsity (%)"); ax[1].set_ylabel("ms"); ax[1].set_title("Largest layer: dense vs sparse kernel")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig4_latency_analysis.png", dpi=150); plt.close()


def make_table(R, cfg):
    b = R["baseline"]; U = R["unstructured"]["global"]; S = R["structured"]; D = R["distill"]
    g90 = next(r for r in U if r["target"] == 0.9)
    s50 = next(r for r in S if r["target"] == 0.5)
    sp, kd = dict(D["plain_profile"]), dict(D["kd_profile"])
    sp["acc"], kd["acc"] = float(np.mean(D["plain_accs"])), float(np.mean(D["kd_accs"]))
    cols = [("Baseline", b, f"{cfg['n_train']} img / {cfg['base_ep']} ep"),
            ("Unstruct. 90% (global)", g90, f"+{cfg['ft_ep']} ep fine-tune"),
            ("Struct. 50% filters", s50, f"+{cfg['ft_ep']} ep fine-tune"),
            ("Student plain", sp, f"{cfg['n_train']} img / {cfg['stu_ep']} ep"),
            ("Student distilled", kd, f"{cfg['n_train']} img / {cfg['stu_ep']} ep + teacher soft labels")]
    rows = [
        ("DATA", "Training data / schedule", lambda p, d: d),
        ("MODEL", "Test accuracy (%)", lambda p, d: f"{p['acc']:.2f}"),
        ("MODEL", "Parameters (total)", lambda p, d: f"{p['params']:,}"),
        ("MODEL", "Non-zero parameters", lambda p, d: f"{p['nonzero']:,}"),
        ("MODEL", "Size, dense file (KB)", lambda p, d: f"{p['dense_kb']:.0f}"),
        ("MODEL", "Size, sparse CSR file (KB)", lambda p, d: f"{p['sparse_kb']:.0f}"),
        ("COMPUTE", "Dense MACs / image (M)", lambda p, d: f"{p['macs'] / 1e6:.2f}"),
        ("COMPUTE", "Effective (non-zero) MACs (M)", lambda p, d: f"{p['eff_macs'] / 1e6:.2f}"),
        ("COMPUTE", "CPU latency batch=1 (ms)", lambda p, d: f"{p['lat_b1_ms']:.3f}"),
        ("COMPUTE", "CPU latency batch=64 (ms)", lambda p, d: f"{p['lat_b64_ms']:.2f}"),
        ("ENERGY", "Energy proxy (uJ/inference)", lambda p, d: f"{p['energy_uJ']:.0f}"),
    ]
    lines = ["| Pillar | Metric | " + " | ".join(c[0] for c in cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    for pillar, metric, fn in rows:
        lines.append(f"| {pillar} | {metric} | " + " | ".join(fn(p, d) for _, p, d in cols) + " |")
    lines.append("")
    lines.append(f"Student accuracy: plain {np.mean(D['plain_accs']):.2f} +/- {np.std(D['plain_accs']):.2f}, "
                 f"distilled {np.mean(D['kd_accs']):.2f} +/- {np.std(D['kd_accs']):.2f} "
                 f"over seeds {cfg['seeds']}.")
    lines.append("Energy proxy = eff_MACs*4.6 pJ + non-zero weights*640 pJ (Horowitz 2014 numbers); an estimate, not a measurement.")
    return "\n".join(lines)


if __name__ == "__main__":
    main()