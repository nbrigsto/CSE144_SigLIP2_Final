"""Does transductive balanced-prior inference beat plain argmax?

Balanced query splits; compares probe argmax, Sinkhorn reassignment, NCM cosine,
and blends. Averaged over many splits for a stable estimate.
"""
import os, sys, argparse, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

FEAT_DIR = "outputs/feats"
SEED = 42


def load(tag):
    d = np.load(os.path.join(FEAT_DIR, f"{tag}_train.npz"))
    return d["feats"].astype(np.float32), d["labels"].astype(np.int64)


def power_transform(X, beta=0.5):
    """PT-MAP preprocessing: signed power then L2-normalize (Gaussian-izes features)."""
    Xp = np.sign(X) * np.abs(X) ** beta
    return normalize(Xp)


def sinkhorn(logp, n_iters=50, tau=1.0, col_prior=None):
    """Balanced OT reassignment: rows sum to 1, columns match col_prior (uniform)."""
    n, nc = logp.shape
    if col_prior is None:
        col_prior = np.full(nc, n / nc)
    M = np.exp((logp - logp.max(1, keepdims=True)) / tau)
    for _ in range(n_iters):
        M = M / M.sum(1, keepdims=True)                    # rows sum to 1
        M = M * (col_prior / np.clip(M.sum(0, keepdims=True), 1e-8, None))  # match column prior
    M = M / M.sum(1, keepdims=True)
    return M


def ncm_logits(Xs, ys, Xq, nc, tau=10.0):
    """Centered + L2 cosine nearest-class-mean logits (SimpleShot-style)."""
    mu = Xs.mean(0)
    Xs2 = normalize(Xs - mu); Xq2 = normalize(Xq - mu)
    cmeans = np.zeros((nc, Xs.shape[1]), dtype=np.float32)
    for c in range(nc):
        m = Xs2[ys == c]
        if len(m):
            cmeans[c] = normalize(m.mean(0, keepdims=True))[0]
    return tau * (Xq2 @ cmeans.T)


def balanced_split(y, q, rng):
    """Hold out exactly q query images per class (skips classes with < q+1 imgs)."""
    sup, qry = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        if len(idx) <= q:              # too few -> keep all as support, none as query
            sup.extend(idx); continue
        qry.extend(idx[:q]); sup.extend(idx[q:])
    return np.array(sup), np.array(qry)


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+",
                    default=["vit_so400m_patch14_siglip_378_378_v2_tta"])
    ap.add_argument("--q", type=int, default=3, help="balanced query imgs per class")
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--C", type=float, default=8.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--nc", type=int, default=100)
    args = ap.parse_args()
    nc = args.nc

    # concat (L2-norm each) the requested tags into one feature matrix
    feats, y = [], None
    for t in args.tags:
        X, yy = load(t)
        feats.append(normalize(X))
        y = yy if y is None else y
    X = normalize(np.concatenate(feats, 1))
    Xpt = power_transform(np.concatenate([f for f in feats], 1))
    print(f"tags={args.tags} dim={X.shape[1]} q={args.q} reps={args.reps}")

    res = {k: [] for k in ["lr", "sink_lr", "ncm_pt", "sink_ncm", "blend", "sink_blend"]}
    for r in range(args.reps):
        rng = np.random.default_rng(SEED + r)
        sup, qry = balanced_split(y, args.q, rng)
        ys, yq = y[sup], y[qry]

        # --- logistic probe ---
        clf = LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced").fit(X[sup], ys)
        P = np.zeros((len(qry), nc), dtype=np.float64)
        P[:, clf.classes_] = clf.predict_proba(X[qry])
        res["lr"].append((P.argmax(1) == yq).mean())
        S = sinkhorn(np.log(P + 1e-9), tau=args.tau)
        res["sink_lr"].append((S.argmax(1) == yq).mean())
        # soft blend: robust if test isn't perfectly balanced
        for lam in (0.25, 0.5, 0.75):
            res.setdefault(f"soft{lam}", [])
            res[f"soft{lam}"].append((((1-lam)*P + lam*S).argmax(1) == yq).mean())

        # --- power-transform NCM cosine ---
        zl = ncm_logits(Xpt[sup], ys, Xpt[qry], nc, tau=10.0)
        Pn = softmax(zl)
        res["ncm_pt"].append((Pn.argmax(1) == yq).mean())
        res["sink_ncm"].append((sinkhorn(zl, tau=args.tau).argmax(1) == yq).mean())

        # --- blend probe + ncm, then sinkhorn ---
        Pb = (P + Pn) / 2
        res["blend"].append((Pb.argmax(1) == yq).mean())
        res["sink_blend"].append((sinkhorn(np.log(Pb + 1e-9), tau=args.tau).argmax(1) == yq).mean())

    print(f"{'method':16s}  acc       delta vs lr")
    base = np.mean(res["lr"])
    for k in res:
        m, s = np.mean(res[k]), np.std(res[k])
        print(f"{k:16s}  {m:.4f}±{s:.3f}   {m-base:+.4f}")


if __name__ == "__main__":
    main()
