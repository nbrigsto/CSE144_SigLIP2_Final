"""Confidence-aware fusion of probe (P) and zero-shot text (Z).

Tests whether trusting the 'peakier' source per image beats a fixed additive weight:
additive, product-of-experts, and entropy/max-prob gated variants, all + Sinkhorn.
"""
import os, sys, argparse, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
sys.path.insert(0, os.path.dirname(__file__))
from exp_transductive import sinkhorn, balanced_split

FEAT_DIR = "outputs/feats"
SEED = 42
EPS = 1e-9


def sharpen(Pr, t):
    """Temperature-sharpen a probability matrix (t<1 sharper, t>1 flatter)."""
    L = np.log(Pr + EPS) / t
    L -= L.max(1, keepdims=True)
    e = np.exp(L)
    return e / e.sum(1, keepdims=True)


def norm_entropy(Pr):
    """Per-row entropy normalized to [0,1] (0=peaky, 1=flat)."""
    h = -(Pr * np.log(Pr + EPS)).sum(1)
    return h / np.log(Pr.shape[1])


def acc(F, yq):
    return (F.argmax(1) == yq).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="vit_so400m_patch14_siglip_378_378_v2_tta")
    ap.add_argument("--zs", default="zeroshot378")
    ap.add_argument("--q", type=int, default=5)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--C", type=float, default=8.0)
    ap.add_argument("--tau", type=float, default=0.4)
    ap.add_argument("--tP", type=float, default=1.0)
    ap.add_argument("--tZ", type=float, default=1.0)
    ap.add_argument("--nc", type=int, default=100)
    args = ap.parse_args()
    nc = args.nc

    d = np.load(os.path.join(FEAT_DIR, f"{args.tag}_train.npz"))
    X = normalize(d["feats"].astype(np.float32)); y = d["labels"].astype(np.int64)
    Zall = np.load(os.path.join(FEAT_DIR, f"{args.zs}_train.npz"))["proba"].astype(np.float64)

    def sink(F):
        return sinkhorn(np.log(F + EPS), tau=args.tau)

    res = {}
    def add(k, v): res.setdefault(k, []).append(v)

    for r in range(args.reps):
        rng = np.random.default_rng(SEED + r)
        sup, qry = balanced_split(y, args.q, rng)
        clf = LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced").fit(X[sup], y[sup])
        P = np.zeros((len(qry), nc)); P[:, clf.classes_] = clf.predict_proba(X[qry])
        Z = Zall[qry]; yq = y[qry]
        P = sharpen(P, args.tP); Z = sharpen(Z, args.tZ)

        # 1) additive baseline (current best)
        for w in (0.2, 0.3):
            add(f"add_w{w}", acc(sink((1 - w) * P + w * Z), yq))

        # 2) product of experts: softmax(logP + b logZ)
        for b in (0.3, 0.5, 0.7, 1.0):
            F = np.exp(np.log(P + EPS) + b * np.log(Z + EPS))
            F /= F.sum(1, keepdims=True)
            add(f"poe_b{b}", acc(sink(F), yq))

        # 3) confidence-weighted additive: weight each source by its peak prob
        cP = P.max(1, keepdims=True); cZ = Z.max(1, keepdims=True)
        for g in (1.0, 2.0, 4.0):
            wP = cP ** g; wZ = cZ ** g
            F = (wP * P + wZ * Z) / (wP + wZ)
            add(f"confadd_g{g}", acc(sink(F), yq))

        # 4) confidence-weighted PoE: Z exponent scales with Z peakiness (1-entropy)
        bZ = (1 - norm_entropy(Z))[:, None]    # in [0,1], high when Z is peaky
        for s in (1.0, 1.5, 2.0):
            F = np.exp(np.log(P + EPS) + s * bZ * np.log(Z + EPS))
            F /= F.sum(1, keepdims=True)
            add(f"confpoe_s{s}", acc(sink(F), yq))

        # 5) finer additive weight sweep
        for w in (0.25, 0.35, 0.4):
            add(f"add_w{w}", acc(sink((1 - w) * P + w * Z), yq))

        # 6) probe-uncertainty-gated: trust text more only where the PROBE is jumbled
        hP = norm_entropy(P)[:, None]          # high when probe unsure
        for wmax in (0.4, 0.6, 0.8):
            wi = wmax * hP                      # per-image text weight in [0, wmax]
            add(f"gate_wmax{wmax}", acc(sink((1 - wi) * P + wi * Z), yq))

    base = np.mean(res["add_w0.2"])
    print(f"q={args.q} reps={args.reps} tau={args.tau} tP={args.tP} tZ={args.tZ}")
    print(f"{'method':14s}  acc       delta vs add_w0.2")
    for k in res:
        m = np.mean(res[k]); print(f"{k:14s}  {m:.4f}   {m-base:+.4f}")


if __name__ == "__main__":
    main()
