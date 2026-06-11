"""Compare classifier-head strategies on cached frozen features via repeated CV:
per-tag LR, concat LR, concat PCA-whiten, NCM/SimpleShot, and blends.

  python src/eval_heads.py --tags <tag1> <tag2> ...
"""
import os, sys, argparse, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import PCA

FEAT_DIR = "outputs/feats"
SEED = 42
N_REPEATS = 5
C_GRID = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]


def load(tag):
    tr = np.load(os.path.join(FEAT_DIR, f"{tag}_train.npz"))
    return normalize(tr["feats"].astype(np.float32)), tr["labels"].astype(np.int64)


def repeated_oof(fit_predict, y, nc):
    """fit_predict(tr_idx, va_idx) -> proba (len(va), nc). Returns OOF proba averaged
    over N_REPEATS shuffled 5-fold splits."""
    acc = np.zeros((len(y), nc), dtype=np.float64)
    for r in range(N_REPEATS):
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED + r)
        for tr, va in skf.split(np.zeros(len(y)), y):
            acc[va] += fit_predict(tr, va)
    return acc / N_REPEATS


def lr_oof(X, y, nc, C):
    def fp(tr, va):
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        p = np.zeros((len(va), nc)); p[:, clf.classes_] = clf.predict_proba(X[va])
        return p
    return repeated_oof(fp, y, nc)


def pick_C(X, y, nc):
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    best, bc = -1, C_GRID[0]
    for c in C_GRID:
        oof = np.zeros((len(y), nc))
        for tr, va in skf.split(np.zeros(len(y)), y):
            clf = LogisticRegression(C=c, max_iter=1500, class_weight="balanced").fit(X[tr], y[tr])
            oof[np.ix_(va, clf.classes_)] = clf.predict_proba(X[va])
        a = (oof.argmax(1) == y).mean()
        if a > best: best, bc = a, c
    return bc


def ncm_oof(X, y, nc):
    # SimpleShot: center by global mean, L2-norm, nearest class mean (cosine)
    def fp(tr, va):
        mu = X[tr].mean(0)
        Xtr = normalize(X[tr] - mu); Xva = normalize(X[va] - mu)
        cmeans = np.stack([normalize(Xtr[y[tr] == c].mean(0, keepdims=True))[0]
                           if (y[tr] == c).any() else np.zeros(X.shape[1]) for c in range(nc)])
        logits = Xva @ cmeans.T
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    return repeated_oof(fp, y, nc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--nc", type=int, default=100)
    args = ap.parse_args()
    nc = args.nc

    feats, y = {}, None
    for t in args.tags:
        X, yy = load(t); feats[t] = X
        y = yy if y is None else y
        assert np.array_equal(y, yy)
    print(f"tags={args.tags} N={len(y)}")

    # per-tag LR + late-fusion greedy
    oofs = []
    for t in args.tags:
        C = pick_C(feats[t], y, nc)
        oof = lr_oof(feats[t], y, nc, C)
        a = (oof.argmax(1) == y).mean()
        print(f"  lr_l2 [{t}] C={C} acc={a:.4f}")
        oofs.append(oof)
    oofs = np.array(oofs)
    # greedy late fusion
    counts = np.zeros(len(oofs), int); counts[oofs.mean((1,2)).argmax()] = 0
    singles = [(oofs[i].argmax(1) == y).mean() for i in range(len(oofs))]
    counts[int(np.argmax(singles))] = 1; blend = oofs[int(np.argmax(singles))].copy(); best = max(singles)
    for _ in range(30):
        ca, ci = best, -1
        for i in range(len(oofs)):
            a = ((blend + oofs[i]).argmax(1) == y).mean()
            if a > ca + 1e-9: ca, ci = a, i
        if ci < 0: break
        counts[ci]+=1; blend = blend+oofs[ci]; best = ca
    print(f"  late-fusion greedy acc={best:.4f} weights={dict(zip(args.tags, (counts/counts.sum()).round(3)))}")

    # concat LR
    Xc = np.concatenate([feats[t] for t in args.tags], axis=1)
    Xc = normalize(Xc)
    C = pick_C(Xc, y, nc)
    concat_oof = lr_oof(Xc, y, nc, C)
    print(f"  concat_lr (dim={Xc.shape[1]}) C={C} acc={(concat_oof.argmax(1)==y).mean():.4f}")

    # concat PCA-whiten LR
    for ncomp in [256, 512]:
        if ncomp >= min(Xc.shape): continue
        def fp(tr, va, ncomp=ncomp):
            sc = StandardScaler().fit(Xc[tr]); Xtr = sc.transform(Xc[tr]); Xva = sc.transform(Xc[va])
            pca = PCA(n_components=ncomp, whiten=True, random_state=SEED).fit(Xtr)
            Ztr, Zva = pca.transform(Xtr), pca.transform(Xva)
            clf = LogisticRegression(C=8.0, max_iter=2000, class_weight="balanced").fit(Ztr, y[tr])
            p = np.zeros((len(va), nc)); p[:, clf.classes_] = clf.predict_proba(Zva); return p
        oof = repeated_oof(fp, y, nc)
        print(f"  concat_pca{ncomp}_whiten acc={(oof.argmax(1)==y).mean():.4f}")

    # NCM / SimpleShot on concat
    ncm = ncm_oof(Xc, y, nc)
    print(f"  ncm/simpleshot (concat) acc={(ncm.argmax(1)==y).mean():.4f}")

    # blend concat_lr + late-fusion greedy
    blend_norm = blend / blend.sum(1, keepdims=True)
    mix = (concat_oof + blend_norm) / 2
    print(f"  blend concat_lr + late-greedy acc={(mix.argmax(1)==y).mean():.4f}")


if __name__ == "__main__":
    main()
