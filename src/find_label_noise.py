"""Surface likely-mislabeled train images for review: flag where the champion probe's
out-of-fold prediction confidently disagrees with the given label.
"""
import os, sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
sys.path.insert(0, os.path.dirname(__file__))
from dataset import list_train_samples

FEAT_DIR = "outputs/feats"
TAG = "vit_so400m_patch14_siglip_378_378_v2_tta"
SEED = 42


def main():
    d = np.load(os.path.join(FEAT_DIR, f"{TAG}_train.npz"))
    X = normalize(d["feats"].astype(np.float32)); y = d["labels"].astype(np.int64)
    nc = 100
    paths = [p for p, _ in list_train_samples("data/train")]
    assert len(paths) == len(y)

    P = np.zeros((len(y), nc), dtype=np.float64)
    for r in range(5):
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED + r)
        for tr, va in skf.split(X, y):
            clf = LogisticRegression(C=8.0, max_iter=2000, class_weight="balanced").fit(X[tr], y[tr])
            P[np.ix_(va, clf.classes_)] += clf.predict_proba(X[va])
    P /= 5

    pred = P.argmax(1)
    true_p = P[np.arange(len(y)), y]
    pred_p = P[np.arange(len(y)), pred]
    margin = pred_p - true_p                      # how confidently the model overrides the label
    counts = np.bincount(y, minlength=nc)

    susp = np.where(pred != y)[0]
    susp = susp[np.argsort(-margin[susp])]        # most confident disagreements first
    print(f"{len(susp)} OOF-misclassified / {len(y)} train images. "
          f"Top confident disagreements (review these):\n")
    print(f"{'path':40s} {'given':>5} {'pred':>5} {'p_giv':>6} {'p_prd':>6} {'#given':>6}")
    for i in susp[:40]:
        rel = os.path.relpath(paths[i]).replace("\\", "/")
        print(f"{rel:40s} {y[i]:5d} {pred[i]:5d} {true_p[i]:6.2f} {pred_p[i]:6.2f} {counts[y[i]]:6d}")


if __name__ == "__main__":
    main()
