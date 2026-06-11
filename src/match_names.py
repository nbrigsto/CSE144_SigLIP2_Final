"""Auto-name classes: rank each group's candidate names by cosine similarity to each
class's mean image embedding (SigLIP2, prompt-ensembled). Writes top-5 suggestions
(top-1 prefilled) to data/class_names_suggested.csv.
"""
import os, sys, numpy as np, pandas as pd, torch, open_clip
from torch.utils.data import DataLoader, Dataset
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from dataset import list_train_samples
from zeroshot_siglip import MODEL, PRET, TEMPLATES, GROUP_WORD

GROUP_OF = {**{c: "food" for c in range(0, 25)},
            **{c: "flower" for c in range(25, 50)},
            **{c: "car" for c in range(50, 75)},
            **{c: "aircraft" for c in range(75, 100)}}
CAND_FILE = {"food": "food101", "flower": "flowers102", "car": "cars", "aircraft": "aircraft"}


class ImgDS(Dataset):
    def __init__(self, items, pre): self.items, self.pre = items, pre
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, _ = self.items[i]; return self.pre(Image.open(p).convert("RGB")), i


@torch.no_grad()
def enc_imgs(model, items, pre, device, bs=8):
    out = None
    for x, idx in DataLoader(ImgDS(items, pre), batch_size=bs):
        x = x.to(device)
        with torch.autocast("cuda", enabled=device == "cuda"):
            e = model.encode_image(x).float()
        e = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
        if out is None:
            out = np.zeros((len(items), e.shape[1]), dtype=np.float32)
        out[idx.numpy()] = e
    return out


@torch.no_grad()
def enc_text(model, tok, names, group, device):
    g = GROUP_WORD[group]; feats = []
    for n in names:
        prompts = [t.format(n=n, g=g) for t in TEMPLATES]
        e = model.encode_text(tok(prompts).to(device)).float()
        e = torch.nn.functional.normalize(e, dim=-1).mean(0)
        feats.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    return np.stack(feats)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, pre = open_clip.create_model_and_transforms(MODEL, pretrained=PRET)
    tok = open_clip.get_tokenizer(MODEL); model = model.to(device).eval()

    samples = list_train_samples("data/train")
    y = np.array([lab for _, lab in samples])
    I = enc_imgs(model, samples, pre, device)

    # candidate text features per group (cached in-mem)
    cand = {g: [l.strip() for l in open(f"data/candidates/{CAND_FILE[g]}.txt") if l.strip()]
            for g in CAND_FILE}
    candT = {g: enc_text(model, tok, cand[g], g, device) for g in cand}

    rows = []
    for c in range(100):
        g = GROUP_OF[c]
        mean = I[y == c].mean(0); mean /= np.linalg.norm(mean)
        sims = candT[g] @ mean
        order = np.argsort(-sims)[:5]
        top = [(cand[g][i], float(sims[i])) for i in order]
        top5 = " | ".join(f"{n}({s:.2f})" for n, s in top)
        print(f"[{c:3d} {g:8s}] {top5}")
        rows.append({"label": c, "group": g, "name": top[0][0], "conf": "auto",
                     "top5": top5})
    pd.DataFrame(rows).to_csv("data/class_names_suggested.csv", index=False)
    print("\nwrote data/class_names_suggested.csv (top-1 prefilled; verify against top5 col)")


if __name__ == "__main__":
    main()
