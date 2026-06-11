"""Fine-tune a pretrained ViT on the few-shot task (8 GB friendly): gradient
checkpointing, layer-wise LR decay, frozen early blocks, heavy aug, MixUp/CutMix, EMA.
Reference experiment — it overfits and does not beat the frozen probe.

  python src/finetune.py --backbone vit_large_patch16_siglip_384.webli --img_size 384 --folds 0
"""
import os, sys, argparse, numpy as np, pandas as pd, torch, timm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
from dataset import list_train_samples
from utils import set_seed, seed_worker, ModelEMA, mixup_cutmix, SoftTargetCrossEntropy

FEAT_DIR = "outputs/feats"


class ImgDataset(Dataset):
    def __init__(self, items, tf): self.items, self.tf = items, tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, lab = self.items[i]; return self.tf(Image.open(p).convert("RGB")), lab


def build_transforms(sz, mean, std):
    train = T.Compose([
        T.RandomResizedCrop(sz, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.TrivialAugmentWide(interpolation=T.InterpolationMode.BICUBIC),
        T.ColorJitter(0.3, 0.3, 0.3, 0.1),
        T.ToTensor(), T.Normalize(mean, std), T.RandomErasing(0.25),
    ])
    eval_ = T.Compose([
        T.Resize(int(sz * 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(sz), T.ToTensor(), T.Normalize(mean, std),
    ])
    return train, eval_


def freeze_early(model, n_freeze):
    for p in model.parameters():
        p.requires_grad_(True)
    # freeze stem
    for name in ["patch_embed", "pos_embed", "cls_token", "reg_token"]:
        mod = getattr(model, name, None)
        if mod is not None:
            if isinstance(mod, torch.nn.Parameter): mod.requires_grad_(False)
            else:
                for p in mod.parameters(): p.requires_grad_(False)
    blocks = model.blocks
    for i, blk in enumerate(blocks):
        if i < n_freeze:
            for p in blk.parameters(): p.requires_grad_(False)


def param_groups_llrd(model, base_lr, head_lr, wd, decay=0.75):
    """Layer-wise LR decay: deeper blocks get higher LR, head highest."""
    n = len(model.blocks)
    groups = []
    head = model.get_classifier()
    head_ids = {id(p) for p in head.parameters()}
    for i, blk in enumerate(model.blocks):
        lr = base_lr * (decay ** (n - 1 - i))
        ps = [p for p in blk.parameters() if p.requires_grad]
        if ps: groups.append({"params": ps, "lr": lr})
    other = [p for n_, p in model.named_parameters()
             if p.requires_grad and id(p) not in head_ids and not n_.startswith("blocks")]
    if other: groups.append({"params": other, "lr": base_lr})
    hp = [p for p in head.parameters() if p.requires_grad]
    groups.append({"params": hp, "lr": head_lr})
    return torch.optim.AdamW(groups, weight_decay=wd)


@torch.no_grad()
def predict_proba(model, loader, device, nc, tta=True):
    model.eval(); out = []
    for x, _ in loader:
        x = x.to(device)
        views = [x, torch.flip(x, [3])] if tta else [x]
        p = 0.0
        for v in views:
            with torch.autocast("cuda", enabled=device == "cuda"):
                p = p + torch.softmax(model(v).float(), 1)
        out.append((p / len(views)).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="vit_large_patch16_siglip_384.webli")
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--freeze_blocks", type=int, default=14)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--base_lr", type=float, default=1e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--nc", type=int, default=100)
    ap.add_argument("--tag", default="ft_siglipL384")
    args = ap.parse_args()

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = list_train_samples("data/train")
    y = np.array([l for _, l in samples])

    # test items (sample order)
    sub = pd.read_csv("data/sample_submission.csv")
    test_items = [(os.path.join("data/test", i), -1) for i in sub["ID"]]

    skf = StratifiedKFold(args.n_folds, shuffle=True, random_state=42)
    splits = list(skf.split(np.zeros(len(y)), y))

    for fold in args.folds:
        tr_idx, va_idx = splits[fold]
        model = timm.create_model(args.backbone, pretrained=True, num_classes=args.nc,
                                  img_size=args.img_size, drop_rate=0.1, drop_path_rate=0.1).to(device)
        if hasattr(model, "set_grad_checkpointing"): model.set_grad_checkpointing(True)
        freeze_early(model, args.freeze_blocks)
        dcfg = timm.data.resolve_model_data_config(model)
        train_tf, eval_tf = build_transforms(args.img_size, dcfg["mean"], dcfg["std"])

        tr_items = [samples[i] for i in tr_idx]; va_items = [samples[i] for i in va_idx]
        g = torch.Generator(); g.manual_seed(42)
        tl = DataLoader(ImgDataset(tr_items, train_tf), batch_size=args.batch_size, shuffle=True,
                        num_workers=0, worker_init_fn=seed_worker, generator=g, drop_last=False)
        vl = DataLoader(ImgDataset(va_items, eval_tf), batch_size=args.batch_size, shuffle=False, num_workers=0)
        tel = DataLoader(ImgDataset(test_items, eval_tf), batch_size=args.batch_size, shuffle=False, num_workers=0)

        opt = param_groups_llrd(model, args.base_lr, args.head_lr, args.wd)
        steps = max(1, len(tl) // args.grad_accum)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (
            (e + 1) / max(1, args.warmup) if e < args.warmup
            else 0.5 * (1 + np.cos(np.pi * (e - args.warmup) / max(1, args.epochs - args.warmup)))))
        scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
        soft_ce = SoftTargetCrossEntropy(); ema = ModelEMA(model, args.ema)

        best, best_proba = -1, None
        for ep in range(args.epochs):
            model.train(); opt.zero_grad(set_to_none=True); tot = 0
            for i, (x, yb) in enumerate(tl):
                x, yb = x.to(device), yb.to(device)
                x, tgt = mixup_cutmix(x, yb, args.nc, 0.2, 1.0, 0.5, args.ls)
                with torch.autocast("cuda", enabled=device == "cuda"):
                    loss = soft_ce(model(x), tgt) / args.grad_accum
                scaler.scale(loss).backward(); tot += loss.item() * args.grad_accum
                if (i + 1) % args.grad_accum == 0:
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); ema.update(model)
            sched.step()
            # eval EMA + raw
            va_y = np.array([l for _, l in va_items])
            pr_raw = predict_proba(model, vl, device, args.nc)
            pr_ema = predict_proba(ema.ema, vl, device, args.nc)
            a_raw = (pr_raw.argmax(1) == va_y).mean(); a_ema = (pr_ema.argmax(1) == va_y).mean()
            best_model = ema.ema if a_ema >= a_raw else model
            a = max(a_raw, a_ema)
            print(f"[fold{fold}] ep{ep+1}/{args.epochs} loss={tot/len(tl):.3f} "
                  f"val raw={a_raw:.4f} ema={a_ema:.4f}", flush=True)
            if a > best:
                best = a
                best_proba = predict_proba(best_model, tel, device, args.nc)
        print(f"[fold{fold}] BEST val acc = {best:.4f}", flush=True)
        np.savez(os.path.join(FEAT_DIR, f"{args.tag}_fold{fold}_testproba.npz"),
                 proba=best_proba, ids=sub["ID"].values, val_acc=best)


if __name__ == "__main__":
    main()
