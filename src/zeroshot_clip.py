"""Model-agnostic open_clip zero-shot extractor -> per-class probs (e.g. a second
DFN-5B text model). Handles CLIP- and SigLIP-style heads.

    python src/zeroshot_clip.py --model ViT-H-14-quickgelu --pretrained dfn5b --tag dfn5b
"""
import os, sys, argparse, numpy as np, pandas as pd, torch, open_clip
from torch.utils.data import DataLoader, Dataset
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from dataset import list_train_samples
from zeroshot_siglip import TEMPLATES, GROUP_WORD

FEAT_DIR = "outputs/feats"


class ImgDS(Dataset):
    def __init__(self, items, pre): self.items, self.pre = items, pre
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, _ = self.items[i]
        return self.pre(Image.open(p).convert("RGB")), i


@torch.no_grad()
def encode_images(model, items, pre, device, bs, hflip=True):
    out = None
    for x, idx in DataLoader(ImgDS(items, pre), batch_size=bs, num_workers=0):
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
def build_text(model, tok, names, groups, device):
    feats = []
    for name, grp in zip(names, groups):
        g = GROUP_WORD.get(grp, grp)
        prompts = [t.format(n=name, g=g) for t in TEMPLATES]
        with torch.autocast("cuda", enabled=device == "cuda"):
            e = model.encode_text(tok(prompts).to(device)).float()
        e = torch.nn.functional.normalize(e, dim=-1).mean(0)
        feats.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    return np.stack(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--tag", required=True, help="short id for output files")
    ap.add_argument("--names_csv", default="data/class_names.csv")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--data_dir", default="data")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    nm = pd.read_csv(args.names_csv).sort_values("label")
    names, groups = nm["name"].tolist(), nm["group"].tolist()
    assert len(names) == 100

    model, _, pre = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tok = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()
    ls = model.logit_scale.exp().item()
    lb = model.logit_bias.item() if getattr(model, "logit_bias", None) is not None else 0.0
    print(f"model={args.model}/{args.pretrained} logit_scale={ls:.2f} logit_bias={lb:.3f}")

    T = build_text(model, tok, names, groups, device)

    emb_path = os.path.join(FEAT_DIR, f"zeroshot_{args.tag}_imgemb.npz")
    sub_ids = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))["ID"].tolist()
    train_samples = list_train_samples(os.path.join(args.data_dir, "train"))
    y = np.array([lab for _, lab in train_samples], dtype=np.int64)
    if os.path.exists(emb_path):
        c = np.load(emb_path, allow_pickle=True); Itr, Ite = c["Itr"], c["Ite"]
        print("loaded cached image embeddings")
    else:
        Itr = encode_images(model, train_samples, pre, device, args.batch_size)
        test_items = [(os.path.join(args.data_dir, "test", i), i) for i in sub_ids]
        Ite = encode_images(model, test_items, pre, device, args.batch_size)
        np.savez(emb_path, Itr=Itr, Ite=Ite, ids=np.array(sub_ids))
        print("cached image embeddings ->", emb_path)

    def probs(I):
        z = (I @ T.T) * ls + lb
        z = z - z.max(1, keepdims=True)
        e = np.exp(z); return e / e.sum(1, keepdims=True)

    Ptr, Pte = probs(Itr), probs(Ite)
    np.savez(os.path.join(FEAT_DIR, f"zeroshot_{args.tag}_train.npz"), proba=Ptr, labels=y)
    np.savez(os.path.join(FEAT_DIR, f"zeroshot_{args.tag}_test.npz"), proba=Pte, ids=np.array(sub_ids))
    pred = Ptr.argmax(1)
    print(f"zero-shot[{args.tag}] TRAIN acc = {(pred==y).mean():.4f}")
    for g in ["food", "flower", "car", "aircraft"]:
        sel = np.array([groups[c] == g for c in y])
        if sel.any(): print(f"  {g:9s} {(pred[sel]==y[sel]).mean():.4f}  (n={sel.sum()})")
    print(f"saved zeroshot_{args.tag}_{{train,test}}.npz  train{Ptr.shape} test{Pte.shape}")


if __name__ == "__main__":
    main()
