"""Fit the champion logistic probe on all training features and serialize it
(+ a model card) to outputs/model/ for the Google Drive bundle.

    python src/save_model.py
"""
import os, json, numpy as np, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

FEAT_DIR = "outputs/feats"
OUT_DIR = "outputs/model"
TAG = "vit_so400m_patch14_siglip_378_378_v2_tta"
C = 8.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    d = np.load(os.path.join(FEAT_DIR, f"{TAG}_train.npz"))
    X = normalize(d["feats"].astype(np.float32))
    y = d["labels"].astype(np.int64)
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced",
                             multi_class="multinomial").fit(X, y)
    train_acc = float((clf.predict(X) == y).mean())

    path = os.path.join(OUT_DIR, "probe_siglip2_so400m.joblib")
    joblib.dump({
        "probe": clf,
        "backbone": "vit_so400m_patch14_siglip_378.v2_webli",
        "feature_tag": TAG,
        "img_size": 378,
        "tta": "multi (center-crop + squash-resize, each +/- hflip)",
        "C": C,
        "n_classes": 100,
        "normalize": "L2 per-view, average views, L2 again",
        "note": "Inference: see README / src/predict_fused.py",
    }, path)

    with open(os.path.join(OUT_DIR, "MODEL_CARD.json"), "w") as f:
        json.dump({"feature_tag": TAG, "C": C, "train_n": int(len(y)),
                   "train_accuracy_in_sample": round(train_acc, 4),
                   "public_lb": 0.96363}, f, indent=2)
    print(f"wrote {path}  (in-sample train acc {train_acc:.4f}, n={len(y)})")


if __name__ == "__main__":
    main()
