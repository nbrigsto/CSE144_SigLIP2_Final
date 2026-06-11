"""Extract frozen backbone embeddings (L2-normalized, with TTA) for train + test,
cached to outputs/feats/.

    python src/extract_features.py --backbone vit_so400m_patch14_siglip_378.v2_webli
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import timm
from torch.utils.data import DataLoader, Dataset
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from dataset import list_train_samples

FEAT_DIR = "outputs/feats"


def short_tag(backbone, img_size):
    parts = backbone.split(".")
    base = parts[0]
    # keep a weights hint so .webli vs .v2_webli don't collide
    wsuffix = ""
    if len(parts) > 1:
        w = parts[1]
        wsuffix = "_v2" if "v2" in w else ("_" + w.split("_")[0][:6] if w else "")
    return f"{base}_{img_size}{wsuffix}"


class PathDataset(Dataset):
    def __init__(self, items, transform):
        # items: list of (path, key) where key is label (train) or filename id (test)
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, key = self.items[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), idx  # return idx to keep order robust


@torch.no_grad()
def extract(model, loaders, device, n, dim, tta=True):
    """Average L2-normalized features over TTA views (transforms x hflip)."""
    acc = np.zeros((n, dim), dtype=np.float32)
    for loader in loaders:
        for x, idx in loader:
            x = x.to(device, non_blocking=True)
            views = [x, torch.flip(x, dims=[3])] if tta else [x]
            f_sum = 0.0
            for v in views:
                with torch.autocast("cuda", enabled=device == "cuda"):
                    f = model(v).float()
                f_sum = f_sum + torch.nn.functional.normalize(f, dim=1)
            acc[idx.numpy()] += (f_sum / len(views)).cpu().numpy()
    # average across base transforms, then L2-normalize
    acc /= len(loaders)
    norms = np.linalg.norm(acc, axis=1, keepdims=True)
    return acc / np.clip(norms, 1e-8, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--img_size", type=int, default=None,
                    help="Override input size (else use model default).")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--no_tta", action="store_true")
    ap.add_argument("--tta", choices=["flip", "multi", "multi3"], default="flip",
                    help="flip: crop+hflip; multi: +squash view; multi3: +zoom view.")
    ap.add_argument("--out_suffix", default="",
                    help="Appended to the saved tag (e.g. '_tta') to avoid overwrite.")
    ap.add_argument("--skip_train", action="store_true",
                    help="Only (re)extract the test split (train features already cached).")
    args = ap.parse_args()

    os.makedirs(FEAT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    create_kwargs = dict(pretrained=True, num_classes=0)
    if args.img_size:
        # ViT needs img_size + dynamic_img_size for a non-default resolution
        create_kwargs.update(img_size=args.img_size, dynamic_img_size=True)
    try:
        model = timm.create_model(args.backbone, **create_kwargs).to(device).eval()
    except TypeError:
        create_kwargs.pop("dynamic_img_size", None)
        model = timm.create_model(args.backbone, **create_kwargs).to(device).eval()
    data_cfg = timm.data.resolve_model_data_config(model)
    if args.img_size:
        data_cfg["input_size"] = (3, args.img_size, args.img_size)
    transform = timm.data.create_transform(**data_cfg, is_training=False)
    img_size = data_cfg["input_size"][-1]
    transforms_list = [transform]
    if args.tta in ("multi", "multi3"):
        # full-frame "squash" view: resize to square, no crop
        from torchvision import transforms as T
        mean, std = data_cfg["mean"], data_cfg["std"]
        interp = T.InterpolationMode.BICUBIC
        squash = T.Compose([
            T.Resize((img_size, img_size), interpolation=interp),
            T.ToTensor(), T.Normalize(mean, std),
        ])
        transforms_list.append(squash)
        if args.tta == "multi3":
            # zoomed view: resize larger then center-crop
            zoom = T.Compose([
                T.Resize(int(round(img_size * 1.4)), interpolation=interp),
                T.CenterCrop(img_size),
                T.ToTensor(), T.Normalize(mean, std),
            ])
            transforms_list.append(zoom)
    tag = short_tag(args.backbone, img_size) + args.out_suffix
    print(f"backbone={args.backbone} img_size={img_size} tag={tag} "
          f"tta={args.tta} views={len(transforms_list)}")
    print("transform:", transform)

    # probe feature dim
    with torch.no_grad():
        dummy = torch.zeros(1, *data_cfg["input_size"]).to(device)
        dim = model(dummy).shape[1]
    print("feature dim:", dim)

    # ---- train ----
    if not args.skip_train:
        train_samples = list_train_samples(os.path.join(args.data_dir, "train"))
        labels = np.array([lab for _, lab in train_samples], dtype=np.int64)
        train_items = [(p, lab) for p, lab in train_samples]
        train_loaders = [DataLoader(PathDataset(train_items, tf), batch_size=args.batch_size,
                                    shuffle=False, num_workers=0) for tf in transforms_list]
        tr_feats = extract(model, train_loaders, device, len(train_items), dim,
                           tta=not args.no_tta)
        np.savez(os.path.join(FEAT_DIR, f"{tag}_train.npz"), feats=tr_feats, labels=labels)
        print(f"saved train feats {tr_feats.shape}")
    else:
        print("skip_train: leaving cached train feats untouched")

    # ---- test (driven by sample_submission order) ----
    sub = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))
    test_ids = sub["ID"].tolist()
    test_items = [(os.path.join(args.data_dir, "test", i), i) for i in test_ids]
    test_loaders = [DataLoader(PathDataset(test_items, tf), batch_size=args.batch_size,
                               shuffle=False, num_workers=0) for tf in transforms_list]
    te_feats = extract(model, test_loaders, device, len(test_items), dim,
                       tta=not args.no_tta)
    np.savez(os.path.join(FEAT_DIR, f"{tag}_test.npz"), feats=te_feats,
             ids=np.array(test_ids))
    print(f"saved test feats {te_feats.shape}")


if __name__ == "__main__":
    main()
