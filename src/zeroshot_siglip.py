"""Zero-shot SigLIP2 text classifier -> per-class probs for train + test.

Same backbone as the champion; class names from data/class_names.csv; prompt
templates ensembled. Writes outputs/feats/zeroshot378_{train,test}.npz.

    python src/zeroshot_siglip.py --names_csv data/class_names.csv
"""
import os, sys, argparse, numpy as np, pandas as pd, torch, open_clip
from torch.utils.data import DataLoader, Dataset
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from dataset import list_train_samples

FEAT_DIR = "outputs/feats"
MODEL = "ViT-SO400M-14-SigLIP2-378"
PRET = "webli"

GROUP_WORD = {"food": "food", "flower": "flower", "car": "car", "aircraft": "aircraft"}
TEMPLATES = [
    "a photo of {n}.",
    "a photo of {n}, a type of {g}.",
    "a close-up photo of {n}.",
    "a {g}: {n}.",
]


class ImgDS(Dataset):
    def __init__(self, items, pre): self.items, self.pre = items, pre
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, _ = self.items[i]
        return self.pre(Image.open(p).convert("RGB")), i


@torch.no_grad()
def encode_images(model, items, pre, device, bs, hflip=True):
    loader = DataLoader(ImgDS(items, pre), batch_size=bs, shuffle=False, num_workers=0)
    out = None
    for x, idx in loader:
        x = x.to(device)
        views = [x, torch.flip(x, dims=[3])] if hflip else [x]
        f = 0.0
        for v in views:
            with torch.autocast("cuda", enabled=device == "cuda"):
                e = model.encode_image(v).float()
            f = f + torch.nn.functional.normalize(e, dim=-1)
        f = torch.nn.functional.normalize(f, dim=-1).cpu().numpy()
        if out is None:
            out = np.zeros((len(items), f.shape[1]), dtype=np.float32)
        out[idx.numpy()] = f
    return out


@torch.no_grad()
def build_text(model, tokenizer, names, groups, device):
    feats = []
    for name, grp in zip(names, groups):
        g = GROUP_WORD.get(grp, grp)
        prompts = [t.format(n=name, g=g) for t in TEMPLATES]
        tok = tokenizer(prompts).to(device)
        with torch.autocast("cuda", enabled=device == "cuda"):
            e = model.encode_text(tok).float()
        e = torch.nn.functional.normalize(e, dim=-1).mean(0)
        feats.append(torch.nn.functional.normalize(e, dim=-1))
    return torch.stack(feats).cpu().numpy()    # (nc, dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names_csv", default="data/class_names.csv")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--data_dir", default="data")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    nm = pd.read_csv(args.names_csv).sort_values("label")
    names = nm["name"].tolist(); groups = nm["group"].tolist()
    assert len(names) == 100

    model, _, pre = open_clip.create_model_and_transforms(MODEL, pretrained=PRET)
    tokenizer = open_clip.get_tokenizer(MODEL)
    model = model.to(device).eval()

    T = build_text(model, tokenizer, names, groups, device)   # (100, d)

    # image embeddings cached so name edits only recompute text
    train_samples = list_train_samples(os.path.join(args.data_dir, "train"))
    y = np.array([lab for _, lab in train_samples], dtype=np.int64)
    emb_path = os.path.join(FEAT_DIR, "zeroshot378_imgemb.npz")
    sub_ids = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))["ID"].tolist()
    if os.path.exists(emb_path):
        c = np.load(emb_path, allow_pickle=True)
        Itr, Ite = c["Itr"], c["Ite"]
        print("loaded cached image embeddings")
    else:
        Itr = encode_images(model, train_samples, pre, device, args.batch_size)
        test_items0 = [(os.path.join(args.data_dir, "test", i), i) for i in sub_ids]
        Ite = encode_images(model, test_items0, pre, device, args.batch_size)
        np.savez(emb_path, Itr=Itr, Ite=Ite, ids=np.array(sub_ids))
        print("cached image embeddings ->", emb_path)
    # SigLIP logit scale/bias
    ls = model.logit_scale.exp().item()
    lb = model.logit_bias.item() if hasattr(model, "logit_bias") and model.logit_bias is not None else 0.0
    def probs(I):
        z = (I @ T.T) * ls + lb
        z = z - z.max(1, keepdims=True)
        e = np.exp(z); return e / e.sum(1, keepdims=True)
    Ptr = probs(Itr)
    np.savez(os.path.join(FEAT_DIR, "zeroshot378_train.npz"), proba=Ptr, labels=y)

    # per-group zero-shot accuracy (name sanity check)
    pred = Ptr.argmax(1)
    print(f"zero-shot TRAIN acc (overall) = {(pred==y).mean():.4f}")
    for g in ["food", "flower", "car", "aircraft"]:
        mask = np.array([gr == g for gr in [groups[c] for c in y]])
        if mask.any():
            print(f"  {g:9s} acc = {(pred[mask]==y[mask]).mean():.4f}  (n={mask.sum()})")

    Pte = probs(Ite)
    np.savez(os.path.join(FEAT_DIR, "zeroshot378_test.npz"), proba=Pte, ids=np.array(sub_ids))
    print(f"saved zero-shot probs: train {Ptr.shape} test {Pte.shape}")


if __name__ == "__main__":
    main()
