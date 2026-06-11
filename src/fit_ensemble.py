"""Fit linear probes on cached features, CV-score each backbone, and greedy-ensemble
them into a submission.

    python src/fit_ensemble.py --tags ALL --out outputs/submission.csv
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize

FEAT_DIR = "outputs/feats"
SEED = 42
C_GRID = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]


def load_tag(tag):
    tr = np.load(os.path.join(FEAT_DIR, f"{tag}_train.npz"))
    te = np.load(os.path.join(FEAT_DIR, f"{tag}_test.npz"), allow_pickle=True)
    Xtr = normalize(tr["feats"].astype(np.float32))
    Xte = normalize(te["feats"].astype(np.float32))
    return Xtr, tr["labels"].astype(np.int64), Xte, te["ids"]


N_REPEATS = 5  # repeated CV -> stable OOF estimate


def cv_oof_probs(X, y, n_classes, n_folds=5, C=None):
    """Return repeated-CV out-of-fold probabilities (N, n_classes) and chosen C."""
    if C is None:  # pick C with one fast CV pass, then refine with repeats
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        best_C, best_acc = C_GRID[0], -1
        for c in C_GRID:
            acc = (_oof(X, y, n_classes, skf, c).argmax(1) == y).mean()
            if acc > best_acc:
                best_acc, best_C = acc, c
        C = best_C
    oof = _oof_repeated(X, y, n_classes, n_folds, C)
    return oof, C


def _oof_repeated(X, y, n_classes, n_folds, C):
    acc = np.zeros((len(y), n_classes), dtype=np.float32)
    for r in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED + r)
        acc += _oof(X, y, n_classes, skf, C)
    return acc / N_REPEATS


def _oof(X, y, n_classes, skf, C):
    oof = np.zeros((len(y), n_classes), dtype=np.float32)
    for tr_idx, va_idx in skf.split(X, y):
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(X[tr_idx], y[tr_idx])
        p = clf.predict_proba(X[va_idx])
        # map clf.classes_ to full label space
        oof[np.ix_(va_idx, clf.classes_)] = p
    return oof


def fit_full_predict(X, y, Xte, n_classes, C):
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    clf.fit(X, y)
    p = np.zeros((len(Xte), n_classes), dtype=np.float32)
    p[:, clf.classes_] = clf.predict_proba(Xte)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--n_classes", type=int, default=100)
    ap.add_argument("--out", default="outputs/submission.csv")
    ap.add_argument("--data_dir", default="data")
    args = ap.parse_args()

    if args.tags == ["ALL"]:
        tags = sorted(os.path.basename(f)[:-10]
                      for f in glob.glob(os.path.join(FEAT_DIR, "*_train.npz")))
    else:
        tags = args.tags
    print("ensemble tags:", tags)

    y = None
    ids = None
    oof_list, test_list, weights = [], [], []
    for tag in tags:
        Xtr, ytr, Xte, tids = load_tag(tag)
        if y is None:
            y, ids = ytr, tids
        assert np.array_equal(y, ytr), "label order mismatch across tags"
        oof, C = cv_oof_probs(Xtr, ytr, args.n_classes)
        acc = (oof.argmax(1) == ytr).mean()
        print(f"  [{tag}] dim={Xtr.shape[1]} bestC={C} CV_acc={acc:.4f}")
        test_p = fit_full_predict(Xtr, ytr, Xte, args.n_classes, C)
        oof_list.append(oof)
        test_list.append(test_p)
        weights.append(acc)  # weight stronger models more

    oof_arr = np.array(oof_list)   # (M, N, C)
    test_arr = np.array(test_list)
    M = len(tags)

    # equal-weight ensemble (reference)
    ens_acc = (oof_arr.mean(0).argmax(1) == y).mean()
    print(f"ENSEMBLE (equal, all {M})   CV_acc={ens_acc:.4f}")

    # Greedy ensemble selection (Caruana et al. 2004): add the model that most
    # improves blended OOF accuracy; integer counts -> weights.
    counts = np.zeros(M, dtype=int)
    # seed with the single best model
    singles = [(oof_arr[i].argmax(1) == y).mean() for i in range(M)]
    counts[int(np.argmax(singles))] = 1
    best_acc = max(singles)
    blend = oof_arr[int(np.argmax(singles))].copy()
    for _ in range(40):
        cand_acc, cand_i = best_acc, -1
        for i in range(M):
            b = blend + oof_arr[i]
            a = (b.argmax(1) == y).mean()
            if a > cand_acc + 1e-9:
                cand_acc, cand_i = a, i
        if cand_i < 0:
            break
        counts[cand_i] += 1
        blend = blend + oof_arr[cand_i]
        best_acc = cand_acc
    w = counts / counts.sum()
    sel = {t: round(float(wi), 3) for t, wi in zip(tags, w) if wi > 0}
    print(f"ENSEMBLE (greedy)          CV_acc={best_acc:.4f}  weights={sel}")

    # choose the better of greedy vs equal for the final test prediction
    if best_acc >= ens_acc:
        ens_test = np.tensordot(w, test_arr, axes=(0, 0))
        chosen = "greedy"
    else:
        ens_test = test_arr.mean(0)
        chosen = "equal"
    preds = ens_test.argmax(1)
    print(f"final ensemble: {chosen}")

    sub = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))
    assert list(ids) == sub["ID"].tolist(), "test id order mismatch"
    out_df = pd.DataFrame({"ID": sub["ID"], "Label": preds.astype(int)})
    out_df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(out_df)} rows, "
          f"label range [{preds.min()},{preds.max()}]")


if __name__ == "__main__":
    main()
