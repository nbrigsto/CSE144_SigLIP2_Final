"""Does zero-shot text fusion help on top of the probe (and Sinkhorn)?

Balanced held-out validation; sweeps the fusion weight w with and without Sinkhorn.
"""
import os, sys, argparse, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
sys.path.insert(0, os.path.dirname(__file__))
from exp_transductive import sinkhorn, balanced_split

FEAT_DIR = "outputs/feats"
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="vit_so400m_patch14_siglip_378_378_v2_tta")
    ap.add_argument("--zs", default="zeroshot378")
    ap.add_argument("--q", type=int, default=5)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--C", type=float, default=8.0)
    ap.add_argument("--tau", type=float, default=0.4)
    ap.add_argument("--nc", type=int, default=100)
    args = ap.parse_args()
    nc = args.nc

    d = np.load(os.path.join(FEAT_DIR, f"{args.tag}_train.npz"))
    X = normalize(d["feats"].astype(np.float32)); y = d["labels"].astype(np.int64)
    Z = np.load(os.path.join(FEAT_DIR, f"{args.zs}_train.npz"))["proba"].astype(np.float64)
    print(f"zero-shot alone (train) acc = {(Z.argmax(1)==y).mean():.4f}")

    ws = [0.05, 0.1, 0.2, 0.3, 0.5]
    res = {"probe": [], "sink_probe": []}
    for w in ws:
        res[f"fuse{w}"] = []; res[f"sink_fuse{w}"] = []
    for r in range(args.reps):
        rng = np.random.default_rng(SEED + r)
        sup, qry = balanced_split(y, args.q, rng)
        clf = LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced").fit(X[sup], y[sup])
        P = np.zeros((len(qry), nc)); P[:, clf.classes_] = clf.predict_proba(X[qry])
        Zq = Z[qry]
        yq = y[qry]
        res["probe"].append((P.argmax(1) == yq).mean())
        res["sink_probe"].append((sinkhorn(np.log(P + 1e-9), tau=args.tau).argmax(1) == yq).mean())
        for w in ws:
            F = (1 - w) * P + w * Zq
            res[f"fuse{w}"].append((F.argmax(1) == yq).mean())
            res[f"sink_fuse{w}"].append((sinkhorn(np.log(F + 1e-9), tau=args.tau).argmax(1) == yq).mean())

    base = np.mean(res["probe"])
    print(f"{'method':16s}  acc       delta")
    for k in res:
        m = np.mean(res[k]); print(f"{k:16s}  {m:.4f}   {m-base:+.4f}")


if __name__ == "__main__":
    main()
