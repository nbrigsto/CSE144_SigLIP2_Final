"""Best pipeline: logistic probe + SigLIP2 zero-shot text fusion + Sinkhorn
balanced prior -> submission. ~+3.3% over the plain probe.

    python src/predict_fused.py --w 0.35 --tau 0.4 --out outputs/submission_fused.csv
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
    ap.add_argument("--tag", default="vit_so400m_patch14_siglip_378_378_v2_tta")
    ap.add_argument("--zs", default="zeroshot378")
    ap.add_argument("--C", type=float, default=8.0)
    ap.add_argument("--w", type=float, default=0.35, help="zero-shot fusion weight")
    ap.add_argument("--tau", type=float, default=0.4)
    ap.add_argument("--soft", type=float, default=1.0)
    ap.add_argument("--no_sinkhorn", action="store_true")
    ap.add_argument("--nc", type=int, default=100)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="outputs/submission_fused.csv")
    args = ap.parse_args()
    nc = args.nc

    Xtr, y, Xte, ids = load_tag(args.tag)
    clf = LogisticRegression(C=args.C, max_iter=3000, class_weight="balanced").fit(Xtr, y)
    P = np.zeros((len(Xte), nc), dtype=np.float64)
    P[:, clf.classes_] = clf.predict_proba(Xte)

    Z = np.load(os.path.join(FEAT_DIR, f"{args.zs}_test.npz"), allow_pickle=True)
    Zp = Z["proba"].astype(np.float64)
    assert list(Z["ids"]) == list(ids), "zero-shot test id order mismatch"

    F = (1 - args.w) * P + args.w * Zp              # probe + zero-shot
    base_pred = F.argmax(1)
    if args.no_sinkhorn:
        preds = base_pred
    else:
        S = sinkhorn(np.log(F + 1e-9), tau=args.tau)
        preds = ((1 - args.soft) * F + args.soft * S).argmax(1)

    dist = np.bincount(preds, minlength=nc)
    print(f"tag={args.tag} w={args.w} tau={args.tau} soft={args.soft} "
          f"sinkhorn={not args.no_sinkhorn}")
    print(f"changed {int((preds!=base_pred).sum())}/{len(preds)} vs fused-argmax; "
          f"class-count min={dist.min()} max={dist.max()} zeros={(dist==0).sum()}")

    sub = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))
    assert list(ids) == sub["ID"].tolist(), "test id order mismatch"
    pd.DataFrame({"ID": sub["ID"], "Label": preds.astype(int)}).to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(preds)} rows")


if __name__ == "__main__":
    main()
