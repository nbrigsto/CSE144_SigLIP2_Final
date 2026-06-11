"""Does richer prompt ensembling improve the zero-shot signal? (new info, not re-weighting)

Image embeddings are cached (zeroshot378_imgemb.npz), so we only recompute text -> fast.
Compares template sets by zero-shot TRAIN accuracy (overall + per group).
"""
import os, sys, numpy as np, pandas as pd, torch, open_clip
sys.path.insert(0, os.path.dirname(__file__))
from zeroshot_siglip import MODEL, PRET, GROUP_WORD

FEAT_DIR = "outputs/feats"
GROUP = np.array([i // 25 for i in range(100)])
GNAME = ["food", "flower", "car", "aircraft"]

TEMPLATE_SETS = {
    "current4": [
        "a photo of {n}.", "a photo of {n}, a type of {g}.",
        "a close-up photo of {n}.", "a {g}: {n}.",
    ],
    "single": ["This is a photo of a {n}."],
    "rich": [
        "a photo of {n}.", "a photo of {n}, a type of {g}.",
        "a close-up photo of {n}.", "a {g}: {n}.",
        "a cropped photo of {n}.", "a bright photo of {n}.",
        "a photo of the {n}.", "a good photo of a {n}.",
        "a photo of one {n}.", "{n}.", "a {n}.",
        "a photo of a nice {n}.", "a photo of a {g} called {n}.",
    ],
    "domain": [
        "a photo of {n}.", "a photo of {n}, a type of {g}.",
        "{dom}", "a close-up photo of {n}.", "a {g}: {n}.",
        "a photo of the {g} {n}.",
    ],
}
DOM = {  # one domain-descriptive template per group
    "food": "a photo of {n}, a plated dish of food.",
    "flower": "a photo of {n}, a flower in bloom.",
    "car": "a photo of {n}, a car parked outside.",
    "aircraft": "a photo of {n}, an airplane on the tarmac.",
}


@torch.no_grad()
def text_feats(model, tok, names, groups, templates, device):
    out = []
    for n, grp in zip(names, groups):
        g = GROUP_WORD[grp]
        prompts = [t.format(n=n, g=g, dom=DOM[grp].format(n=n)) for t in templates]
        e = model.encode_text(tok(prompts).to(device)).float()
        e = torch.nn.functional.normalize(e, dim=-1).mean(0)
        out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    return np.stack(out)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nm = pd.read_csv("data/class_names.csv").sort_values("label")
    names, groups = nm["name"].tolist(), nm["group"].tolist()
    c = np.load(os.path.join(FEAT_DIR, "zeroshot378_imgemb.npz"), allow_pickle=True)
    Itr = c["Itr"]
    d = np.load(os.path.join(FEAT_DIR, "zeroshot378_train.npz"))
    y = d["labels"].astype(np.int64)

    model, _, _ = open_clip.create_model_and_transforms(MODEL, pretrained=PRET)
    tok = open_clip.get_tokenizer(MODEL); model = model.to(device).eval()
    ls = model.logit_scale.exp().item()
    lb = model.logit_bias.item() if getattr(model, "logit_bias", None) is not None else 0.0

    print(f"{'template set':12s} overall   food   flower  car   aircraft")
    for name, tmpl in TEMPLATE_SETS.items():
        T = text_feats(model, tok, names, groups, tmpl, device)
        z = (Itr @ T.T) * ls + lb
        pred = z.argmax(1)
        line = f"{name:12s} {(pred==y).mean():.4f}"
        for g in range(4):
            sel = GROUP[y] == g
            line += f"  {(pred[sel]==y[sel]).mean():.4f}"
        print(line)


if __name__ == "__main__":
    main()
