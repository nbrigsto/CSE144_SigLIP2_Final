"""Feature-augmentation probe: keep the backbone frozen but extract many augmented-crop
embeddings per train image to enrich the probe. Image-level CV avoids leakage; test uses
the clean TTA embedding.

  python src/feature_aug_probe.py --backbone <bb> --clean_tag <tag> --K 20
"""
import os, sys, argparse, numpy as np, pandas as pd, torch, timm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(__file__))
from dataset import list_train_samples

FEAT_DIR = "outputs/feats"
SEED = 42
C_GRID = [4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]


class AugDataset(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, lab, idx = self.items[i]
        return self.transform(Image.open(p).convert("RGB")), idx


@torch.no_grad()
def extract_aug(model, items, transform, device, K, bs, dim):
    """K augmented passes over the train set -> (K*N, dim) feats + img indices."""
    feats = np.zeros((K * len(items), dim), dtype=np.float32)
    idxs = np.zeros(K * len(items), dtype=np.int64)
    loader = DataLoader(AugDataset(items, transform), batch_size=bs, shuffle=False, num_workers=0)
    ptr = 0
    for k in range(K):
        for x, idx in loader:
            x = x.to(device)
            with torch.autocast("cuda", enabled=device == "cuda"):
                f = model(x).float()
            f = torch.nn.functional.normalize(f, dim=1).cpu().numpy()
            n = len(idx)
            feats[ptr:ptr+n] = f; idxs[ptr:ptr+n] = idx.numpy(); ptr += n
        print(f"  aug pass {k+1}/{K}", flush=True)
    return feats, idxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--clean_tag", required=True, help="existing clean TTA feature tag")
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_classes", type=int, default=100)
    ap.add_argument("--n_repeats", type=int, default=3)
    ap.add_argument("--out", default="outputs/submission_aug.csv")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nc = args.n_classes

    # clean features (already extracted, TTA-averaged)
    ctr = np.load(os.path.join(FEAT_DIR, f"{args.clean_tag}_train.npz"))
    cte = np.load(os.path.join(FEAT_DIR, f"{args.clean_tag}_test.npz"), allow_pickle=True)
    Xclean = normalize(ctr["feats"].astype(np.float32)); y = ctr["labels"].astype(np.int64)
    Xtest = normalize(cte["feats"].astype(np.float32)); test_ids = cte["ids"]
    N = len(y)

    # build / load augmented train features (cached)
    aug_path = os.path.join(FEAT_DIR, f"{args.clean_tag}_AUG{args.K}.npz")
    if os.path.exists(aug_path):
        d = np.load(aug_path); Xaug, aug_idx = normalize(d["feats"]), d["idx"]
        print(f"loaded cached aug feats {Xaug.shape}")
    else:
        ck = dict(pretrained=True, num_classes=0)
        if args.img_size: ck.update(img_size=args.img_size, dynamic_img_size=True)
        model = timm.create_model(args.backbone, **ck).to(device).eval()
        dcfg = timm.data.resolve_model_data_config(model)
        sz = args.img_size or dcfg["input_size"][-1]
        mean, std = dcfg["mean"], dcfg["std"]
        aug_tf = T.Compose([
            T.RandomResizedCrop(sz, scale=(0.35, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.ToTensor(), T.Normalize(mean, std),
        ])
        with torch.no_grad():
            dim = model(torch.zeros(1, 3, sz, sz).to(device)).shape[1]
        items = [(p, lab, i) for i, (p, lab) in enumerate(list_train_samples("data/train"))]
        Xaug, aug_idx = extract_aug(model, items, aug_tf, device, args.K, args.batch_size, dim)
        np.savez(aug_path, feats=Xaug, idx=aug_idx)
        Xaug = normalize(Xaug)
        print(f"saved aug feats {Xaug.shape}")

    aug_label = y[aug_idx]

    def pick_C(train_img_idx):
        # quick C selection on one split
        mask = np.isin(aug_idx, train_img_idx)
        Xtr = np.concatenate([Xclean[train_img_idx], Xaug[mask]])
        ytr = np.concatenate([y[train_img_idx], aug_label[mask]])
        # validate on clean of a held-out slice
        sub = StratifiedKFold(5, shuffle=True, random_state=SEED)
        best, bc = -1, C_GRID[0]
        tr2, va2 = next(iter(sub.split(np.zeros(len(train_img_idx)), y[train_img_idx])))
        ti, vi = train_img_idx[tr2], train_img_idx[va2]
        m2 = np.isin(aug_idx, ti)
        Xf = np.concatenate([Xclean[ti], Xaug[m2]]); yf = np.concatenate([y[ti], aug_label[m2]])
        for c in C_GRID:
            clf = LogisticRegression(C=c, max_iter=1500, class_weight="balanced").fit(Xf, yf)
            a = (clf.predict(Xclean[vi]) == y[vi]).mean()
            if a > best: best, bc = a, c
        return bc

    # repeated image-level CV
    accs = []
    C = None
    for r in range(args.n_repeats):
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED + r)
        oof = np.zeros(N, dtype=np.int64)
        for tr_img, va_img in skf.split(np.zeros(N), y):
            if C is None:
                C = pick_C(tr_img); print("picked C =", C)
            mask = np.isin(aug_idx, tr_img)
            Xtr = np.concatenate([Xclean[tr_img], Xaug[mask]])
            ytr = np.concatenate([y[tr_img], aug_label[mask]])
            clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
            oof[va_img] = clf.predict(Xclean[va_img])
        a = (oof == y).mean(); accs.append(a)
        print(f"repeat {r}: CV acc = {a:.4f}")
    print(f"FEATURE-AUG probe CV acc = {np.mean(accs):.4f} +/- {np.std(accs):.4f}")

    # final model on all clean+aug, predict clean test
    Xtr = np.concatenate([Xclean, Xaug]); ytr = np.concatenate([y, aug_label])
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
    preds = clf.predict(Xtest)
    sub = pd.read_csv("data/sample_submission.csv")
    assert list(test_ids) == sub["ID"].tolist()
    pd.DataFrame({"ID": sub["ID"], "Label": preds.astype(int)}).to_csv(args.out, index=False)
    # also save test probabilities for ensembling
    proba = np.zeros((len(Xtest), nc), dtype=np.float32); proba[:, clf.classes_] = clf.predict_proba(Xtest)
    np.savez(os.path.join(FEAT_DIR, f"{args.clean_tag}_AUGPROBE_testproba.npz"), proba=proba, ids=test_ids)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
