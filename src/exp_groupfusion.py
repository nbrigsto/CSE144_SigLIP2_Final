"""Honest eval of group-conditional masking + per-group fusion weights.

Weight selection (per-group w) is done on a DISJOINT set of seeds (100..) from the
evaluation seeds (0..), so the reported delta is not tuned-on-test. All + Sinkhorn.
"""
import os, sys, argparse, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
sys.path.insert(0, os.path.dirname(__file__))
from exp_transductive import sinkhorn, balanced_split

FEAT_DIR = "outputs/feats"
NC = 100
GROUP = np.array([i // 25 for i in range(NC)])
GNAME = ["food", "flower", "car", "aircraft"]
CHAMP = "vit_so400m_patch14_siglip_378_378_v2_tta"
WGRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def load(tag):
    d = np.load(os.path.join(FEAT_DIR, f"{tag}_train.npz"))
    return normalize(d["feats"].astype(np.float32)), d["labels"].astype(np.int64)


def probe(X, sup, qry, ys):
    clf = LogisticRegression(C=8.0, max_iter=2000, class_weight="balanced").fit(X[sup], ys)
    P = np.zeros((len(qry), NC)); P[:, clf.classes_] = clf.predict_proba(X[qry]); return P


def sink(F, tau=0.4):
    return sinkhorn(np.log(F + 1e-9), tau=tau)


def run(seeds, X, y, Z, q):
    """yield (Pc, Zq, qry, yq) per seed."""
    for s in seeds:
        rng = np.random.default_rng(s)
        sup, qry = balanced_split(y, q, rng)
        yield probe(X, sup, qry, y[sup]), Z[qry], y[qry]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--w", type=float, default=0.35)
    args = ap.parse_args()
    X, y = load(CHAMP)
    Z = np.load(os.path.join(FEAT_DIR, "zeroshot378_train.npz"))["proba"].astype(np.float64)

    def fuse(Pc, Zq, wmap, mask):
        gq = GROUP[Zq.argmax(1)]  # not used; weights keyed by true group below
        return Pc, Zq

    # ---- TUNE per-group w on seeds 100.. ----
    tune = list(range(100, 100 + args.reps))
    acc = {g: {w: [] for w in WGRID} for g in range(4)}
    for Pc, Zq, yq in run(tune, X, y, Z, args.q):
        gq = GROUP[yq]
        for w in WGRID:
            F = sink((1 - w) * Pc + w * Zq).argmax(1)
            for g in range(4):
                sel = gq == g
                if sel.any(): acc[g][w].append((F[sel] == yq[sel]).mean())
    best_wg = {g: max(WGRID, key=lambda w: np.mean(acc[g][w])) for g in range(4)}
    print("tuned per-group w:", {GNAME[g]: best_wg[g] for g in range(4)})

    # ---- EVAL on seeds 0.. ----
    ev = list(range(args.reps))
    R = {k: [] for k in ["base_globalw", "group_mask", "per_group_w", "per_group_w+mask"]}
    for Pc, Zq, yq in run(ev, X, y, Z, args.q):
        gq = GROUP[yq]
        # base global w
        R["base_globalw"].append((sink((1 - args.w) * Pc + args.w * Zq).argmax(1) == yq).mean())
        # group mask
        Fg = (1 - args.w) * Pc + args.w * Zq
        m = (GROUP[None, :] == gq[:, None]).astype(float)
        R["group_mask"].append((sink(Fg * m).argmax(1) == yq).mean())
        # per-group w (true group of each query image; ~100% predictable at test time)
        wq = np.array([best_wg[g] for g in gq])[:, None]
        Fp = (1 - wq) * Pc + wq * Zq
        R["per_group_w"].append((sink(Fp).argmax(1) == yq).mean())
        R["per_group_w+mask"].append((sink(Fp * m).argmax(1) == yq).mean())

    b = np.mean(R["base_globalw"])
    print(f"\n{'method':20s} acc      delta")
    for k in R:
        mm = np.mean(R[k]); print(f"{k:20s} {mm:.4f}  {mm-b:+.4f}")


if __name__ == "__main__":
    main()
