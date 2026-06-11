"""Does iterated transductive pseudo-labeling help on top of fused+Sinkhorn?

Balanced holdout: self-train on the most confident (class-balanced) query predictions,
refit, re-predict. Query true labels are never used; reported vs the fused+Sinkhorn base.
"""
import os, sys, argparse, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
sys.path.insert(0, os.path.dirname(__file__))
from exp_transductive import sinkhorn, balanced_split

FEAT_DIR = "outputs/feats"
SEED = 42


def fused_probs(clf, Xq, Zq, nc, w):
    P = np.zeros((len(Xq), nc)); P[:, clf.classes_] = clf.predict_proba(Xq)
    return (1 - w) * P + w * Zq


def select_balanced(probsk, fracs, nc):
    """Top (frac*n/nc) query points per class by Sinkhorn-balanced prob (stays balanced)."""
    n = len(probsk); per = max(1, int(round(fracs * n / nc)))
    assign = probsk.argmax(1)
    idx, lab = [], []
    for c in range(nc):
        ci = np.where(assign == c)[0]
        if len(ci) == 0:
            continue
        ci = ci[np.argsort(-probsk[ci, c])][:per]
        idx.extend(ci); lab.extend([c] * len(ci))
    return np.array(idx, dtype=int), np.array(lab, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="vit_so400m_patch14_siglip_378_378_v2_tta")
    ap.add_argument("--zs", default="zeroshot378")
    ap.add_argument("--q", type=int, default=5)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--C", type=float, default=8.0)
    ap.add_argument("--w", type=float, default=0.2)
    ap.add_argument("--tau", type=float, default=0.4)
    ap.add_argument("--pl_weight", type=float, default=0.5, help="sample weight for pseudo-labels")
    ap.add_argument("--schedule", default="0.5,0.8,1.0", help="pseudo-label fraction per iter")
    ap.add_argument("--nc", type=int, default=100)
    args = ap.parse_args()
    nc = args.nc
    sched = [float(x) for x in args.schedule.split(",")]

    d = np.load(os.path.join(FEAT_DIR, f"{args.tag}_train.npz"))
    X = normalize(d["feats"].astype(np.float32)); y = d["labels"].astype(np.int64)
    Z = np.load(os.path.join(FEAT_DIR, f"{args.zs}_train.npz"))["proba"].astype(np.float64)

    base_accs, pl_accs = [], []
    for r in range(args.reps):
        rng = np.random.default_rng(SEED + r)
        sup, qry = balanced_split(y, args.q, rng)
        Xs, ys, Xq, Zq, yq = X[sup], y[sup], X[qry], Z[qry], y[qry]

        clf = LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced").fit(Xs, ys)
        F = fused_probs(clf, Xq, Zq, nc, args.w)
        S = sinkhorn(np.log(F + 1e-9), tau=args.tau)
        base_accs.append((S.argmax(1) == yq).mean())

        # iterated self-training
        for frac in sched:
            idx, plab = select_balanced(S, frac, nc)
            Xtr = np.concatenate([Xs, Xq[idx]]); ytr = np.concatenate([ys, plab])
            sw = np.concatenate([np.ones(len(ys)), np.full(len(idx), args.pl_weight)])
            clf = LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced")
            clf.fit(Xtr, ytr, sample_weight=sw)
            F = fused_probs(clf, Xq, Zq, nc, args.w)
            S = sinkhorn(np.log(F + 1e-9), tau=args.tau)
        pl_accs.append((S.argmax(1) == yq).mean())

    b, p = np.mean(base_accs), np.mean(pl_accs)
    print(f"w={args.w} tau={args.tau} pl_weight={args.pl_weight} schedule={sched}")
    print(f"  fused+sinkhorn base : {b:.4f}")
    print(f"  + pseudo-label      : {p:.4f}   delta={p-b:+.4f}")


if __name__ == "__main__":
    main()
