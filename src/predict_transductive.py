"""Transductive balanced-prior inference -> submission.

Sinkhorn optimal transport over all test probabilities nudges the predicted label
distribution toward uniform (assumes a class-balanced test). ~+1.5% over argmax.

    python src/predict_transductive.py --tau 0.4 --out outputs/submission_transductive.csv
"""
import os, argparse, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

FEAT_DIR = "outputs/feats"


def load_tag(tag):
    tr = np.load(os.path.join(FEAT_DIR, f"{tag}_train.npz"))
    te = np.load(os.path.join(FEAT_DIR, f"{tag}_test.npz"), allow_pickle=True)
    return (normalize(tr["feats"].astype(np.float32)), tr["labels"].astype(np.int64),
            normalize(te["feats"].astype(np.float32)), te["ids"])


def sinkhorn(logp, n_iters=100, tau=0.4, col_prior=None):
    """Balanced OT reassignment: rows sum to 1, columns match col_prior (uniform)."""
    n, nc = logp.shape
    if col_prior is None:
        col_prior = np.full(nc, n / nc)
    M = np.exp((logp - logp.max(1, keepdims=True)) / tau)
    for _ in range(n_iters):
        M = M / M.sum(1, keepdims=True)
        M = M * (col_prior / np.clip(M.sum(0, keepdims=True), 1e-8, None))
    return M / M.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+",
                    default=["vit_so400m_patch14_siglip_378_378_v2_tta"])
    ap.add_argument("--C", type=float, default=8.0)
    ap.add_argument("--tau", type=float, default=0.4)
    ap.add_argument("--soft", type=float, default=1.0,
                    help="Sinkhorn blend (1=full, <1 hedges an unbalanced test).")
    ap.add_argument("--nc", type=int, default=100)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="outputs/submission_transductive.csv")
    args = ap.parse_args()
    nc = args.nc

    # average per-backbone probe probabilities (champion alone by default)
    y = ids = None
    P = None
    for tag in args.tags:
        Xtr, ytr, Xte, tids = load_tag(tag)
        if y is None:
            y, ids = ytr, tids
            P = np.zeros((len(Xte), nc), dtype=np.float64)
        assert np.array_equal(y, ytr), "label order mismatch across tags"
        clf = LogisticRegression(C=args.C, max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
        p = np.zeros((len(Xte), nc), dtype=np.float64)
        p[:, clf.classes_] = clf.predict_proba(Xte)
        P += p
    P /= len(args.tags)
    n = len(P)

    base_pred = P.argmax(1)
    S = sinkhorn(np.log(P + 1e-9), tau=args.tau)
    blended = (1 - args.soft) * P + args.soft * S
    preds = blended.argmax(1)

    changed = int((preds != base_pred).sum())
    dist = np.bincount(preds, minlength=nc)
    print(f"tags={args.tags} tau={args.tau} soft={args.soft} n_test={n}")
    print(f"changed {changed}/{n} predictions vs plain argmax")
    print(f"predicted class-count: min={dist.min()} max={dist.max()} "
          f"zeros={int((dist==0).sum())}")

    sub = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))
    assert list(ids) == sub["ID"].tolist(), "test id order mismatch"
    pd.DataFrame({"ID": sub["ID"], "Label": preds.astype(int)}).to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(preds)} rows, label range "
          f"[{preds.min()},{preds.max()}]")


if __name__ == "__main__":
    main()
